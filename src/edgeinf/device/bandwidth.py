"""The device's estimate of its link to the edge: round-trip time, uplink bandwidth, edge queue.

Samples come from two sources.

    Transfers   every Infer the edge answers. Its time on the network is the round trip without
                the edge's own time, and the bytes sent took that minus the RTT.
    Probes      Health calls, made only when an estimate is older than PROBE_INTERVAL_S: an
                empty one for the RTT, then one with padding for the bandwidth. The device makes
                them in the idle time between two requests (device/client.py), so they are never
                part of a request's latency.

Both estimates are exponentially weighted moving averages. A transfer that took less than
MIN_TRANSFER_MS on top of the RTT cannot be timed, because the RTT itself varies by about a
millisecond; it only shows that the link carries those bytes within MIN_TRANSFER_MS, so it can
raise the estimate to that rate and never lowers it. A probe is sized to take about
PROBE_TARGET_MS at the current estimate: long enough to time, and no longer.

This module does no I/O. Times in ms, bandwidth in Mbps (10^6 bits per second), sizes in bytes.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

ALPHA = 0.3  # weight of a new sample in the moving averages
PROBE_INTERVAL_S = 2.0  # an estimate older than this is probed for
PROBE_TARGET_MS = 20.0  # how long the padding of a bandwidth probe should take to send
MIN_TRANSFER_MS = 4.0  # a shorter transfer is inside the jitter of the RTT
PROBE_START_BYTES = 8_000  # padding of a bandwidth probe while there is no estimate
PROBE_MIN_BYTES = 1_000
PROBE_MAX_BYTES = 48_000  # below gRPC's initial 64 KB flow-control window
RTT_GUESS_MS = 50.0  # expected cost of an RTT probe while there is no estimate
BW_GUESS_MBPS = 1.0  # expected rate of the first bandwidth probe

BITS_PER_BYTE = 8
MS_PER_S = 1e3
BITS_PER_MBIT = 1e6


def transfer_ms(n_bytes: float, bw_mbps: float) -> float:
    """Time in ms to send `n_bytes` at `bw_mbps`."""
    return n_bytes * BITS_PER_BYTE / (bw_mbps * BITS_PER_MBIT) * MS_PER_S


def _rate_mbps(n_bytes: float, ms: float) -> float:
    return n_bytes * BITS_PER_BYTE / BITS_PER_MBIT / (ms / MS_PER_S)


class LinkEstimator:
    """What the device believes about its link. Every value is None until it has a sample.

    `clock` counts seconds and is only used for the age of the estimates.
    """

    def __init__(self, alpha: float = ALPHA, clock: Callable[[], float] = time.monotonic) -> None:
        self.alpha = alpha
        self.rtt_ms: float | None = None
        self.bw_mbps: float | None = None
        self.queue_len: int | None = None  # the edge's latest report, not smoothed
        self._clock = clock
        self._rtt_at = -math.inf
        self._bw_at = -math.inf

    def _smooth(self, old: float | None, new: float) -> float:
        return new if old is None else old + self.alpha * (new - old)

    def add_rtt(self, round_trip_ms: float) -> None:
        """Take the round trip of an empty Health call as an RTT sample."""
        self.rtt_ms = self._smooth(self.rtt_ms, round_trip_ms)
        self._rtt_at = self._clock()

    def add_transfer(self, n_bytes: int, network_ms: float) -> bool:
        """Take a transfer of `n_bytes` whose round trip spent `network_ms` on the network.

        Returns True if it was a full bandwidth sample. Without an RTT nothing can be said; a
        transfer too short to time can only raise the estimate (see the module docstring).
        """
        if self.rtt_ms is None or n_bytes <= 0:
            return False
        sending_ms = network_ms - self.rtt_ms
        if sending_ms < MIN_TRANSFER_MS:
            at_least = _rate_mbps(n_bytes, MIN_TRANSFER_MS)
            if self.bw_mbps is None or at_least > self.bw_mbps:
                self.bw_mbps = at_least
            return False
        self.bw_mbps = self._smooth(self.bw_mbps, _rate_mbps(n_bytes, sending_ms))
        self._bw_at = self._clock()
        return True

    def rtt_due(self) -> bool:
        """True if the RTT estimate is missing or older than PROBE_INTERVAL_S."""
        return self._clock() - self._rtt_at > PROBE_INTERVAL_S

    def bw_due(self) -> bool:
        """True if there has been no full bandwidth sample for PROBE_INTERVAL_S."""
        return self._clock() - self._bw_at > PROBE_INTERVAL_S

    def probe_bytes(self) -> int:
        """Padding for the next bandwidth probe: PROBE_TARGET_MS worth at the current estimate."""
        if self.bw_mbps is None:
            return PROBE_START_BYTES
        size = self.bw_mbps * BITS_PER_MBIT * PROBE_TARGET_MS / MS_PER_S / BITS_PER_BYTE
        return int(min(max(size, PROBE_MIN_BYTES), PROBE_MAX_BYTES))

    def rtt_probe_ms(self) -> float:
        """How long an RTT probe is expected to take."""
        return RTT_GUESS_MS if self.rtt_ms is None else self.rtt_ms

    def bw_probe_ms(self) -> float:
        """How long the next bandwidth probe is expected to take."""
        bw_mbps = BW_GUESS_MBPS if self.bw_mbps is None else self.bw_mbps
        return self.rtt_probe_ms() + transfer_ms(self.probe_bytes(), bw_mbps)

    def as_dict(self) -> dict[str, float | int | None]:
        """The estimates, for the record of a run."""
        return {"bw_mbps": self.bw_mbps, "rtt_ms": self.rtt_ms, "queue_len": self.queue_len}
