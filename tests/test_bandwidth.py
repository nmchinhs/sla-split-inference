"""The device's link estimate (device/bandwidth.py): smoothing, what counts as a sample, probes."""

import pytest

from edgeinf.device.bandwidth import (
    ALPHA,
    MIN_TRANSFER_MS,
    PROBE_INTERVAL_S,
    PROBE_MAX_BYTES,
    PROBE_MIN_BYTES,
    PROBE_START_BYTES,
    PROBE_TARGET_MS,
    LinkEstimator,
    transfer_ms,
)

RTT_MS = 20.0
BITS_PER_BYTE = 8


class _Clock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def link(clock) -> LinkEstimator:
    link = LinkEstimator(clock=clock)
    link.add_rtt(RTT_MS)
    return link


def _round_trip_ms(n_bytes: int, mbps: float) -> float:
    return RTT_MS + transfer_ms(n_bytes, mbps)


def test_transfer_time_is_bits_over_rate():
    assert transfer_ms(125_000, 1.0) == pytest.approx(1000.0)  # 1 Mbit at 1 Mbit/s
    assert transfer_ms(125_000, 50.0) == pytest.approx(20.0)
    assert transfer_ms(0, 5.0) == 0.0


def test_nothing_is_known_before_the_first_sample(clock):
    link = LinkEstimator(clock=clock)
    assert (link.rtt_ms, link.bw_mbps, link.queue_len) == (None, None, None)
    assert link.rtt_due() and link.bw_due()


def test_the_first_sample_is_taken_as_it_is_and_later_ones_are_smoothed(link):
    assert link.rtt_ms == RTT_MS
    link.add_rtt(30.0)
    assert link.rtt_ms == pytest.approx(RTT_MS + ALPHA * (30.0 - RTT_MS))

    assert link.add_transfer(50_000, link.rtt_ms + transfer_ms(50_000, 5.0))
    assert link.bw_mbps == pytest.approx(5.0)
    assert link.add_transfer(50_000, link.rtt_ms + transfer_ms(50_000, 10.0))
    assert link.bw_mbps == pytest.approx(5.0 + ALPHA * (10.0 - 5.0))


def test_bandwidth_is_the_payload_over_the_round_trip_without_the_rtt(link):
    # 6 800 bytes in 74.4 ms with 20 ms of round trip: 54.4 ms for 54 400 bits.
    assert link.add_transfer(6_800, 74.4)
    assert link.bw_mbps == pytest.approx(1.0)


def test_a_transfer_cannot_be_timed_before_the_rtt_is_known(clock):
    link = LinkEstimator(clock=clock)
    assert not link.add_transfer(50_000, 100.0)
    assert link.bw_mbps is None


def test_a_transfer_too_short_to_time_only_says_the_link_is_at_least_that_fast(link):
    # 1 ms on top of the round trip is inside its jitter. All it shows is that the bytes took
    # no longer than MIN_TRANSFER_MS.
    floor = 8_000 * BITS_PER_BYTE / MIN_TRANSFER_MS / 1e3
    assert not link.add_transfer(8_000, RTT_MS + 1.0)
    assert link.bw_mbps == pytest.approx(floor)  # nothing was known: now at least this

    link.add_transfer(48_000, _round_trip_ms(48_000, 50.0))
    known = link.bw_mbps
    assert known > floor
    assert not link.add_transfer(8_000, RTT_MS + 1.0)  # says "at least 16", which is no news
    assert link.bw_mbps == known
    assert not link.add_transfer(0, RTT_MS)  # nothing was sent
    assert link.bw_mbps == known


def test_a_short_transfer_does_not_count_as_a_fresh_sample(link, clock):
    link.add_transfer(48_000, _round_trip_ms(48_000, 50.0))
    clock.now += PROBE_INTERVAL_S + 1
    link.add_transfer(6_800, RTT_MS + 1.0)
    assert link.bw_due()  # a probe is still needed


def test_an_estimate_is_due_for_a_probe_once_it_is_older_than_the_interval(link, clock):
    link.add_transfer(50_000, _round_trip_ms(50_000, 5.0))
    assert not link.rtt_due() and not link.bw_due()
    clock.now += PROBE_INTERVAL_S - 0.1
    assert not link.rtt_due() and not link.bw_due()
    clock.now += 0.2
    assert link.rtt_due() and link.bw_due()
    link.add_rtt(RTT_MS)
    assert not link.rtt_due() and link.bw_due()


def test_a_probe_is_sized_to_take_the_target_time_at_the_current_estimate(link):
    assert link.probe_bytes() == PROBE_START_BYTES  # no estimate yet
    link.add_transfer(50_000, _round_trip_ms(50_000, 5.0))
    size = link.probe_bytes()
    assert transfer_ms(size, 5.0) == pytest.approx(PROBE_TARGET_MS, rel=0.01)
    assert PROBE_MIN_BYTES < size < PROBE_MAX_BYTES


def test_probe_sizes_stay_within_their_limits(clock):
    slow, fast = LinkEstimator(clock=clock), LinkEstimator(clock=clock)
    for link, mbps in ((slow, 0.1), (fast, 500.0)):
        link.add_rtt(RTT_MS)
        link.add_transfer(100_000, _round_trip_ms(100_000, mbps))
    assert slow.probe_bytes() == PROBE_MIN_BYTES
    assert fast.probe_bytes() == PROBE_MAX_BYTES


def test_expected_probe_times_come_from_the_estimates(link):
    link.add_transfer(50_000, _round_trip_ms(50_000, 5.0))
    assert link.rtt_probe_ms() == pytest.approx(RTT_MS)
    assert link.bw_probe_ms() == pytest.approx(RTT_MS + PROBE_TARGET_MS, rel=0.01)


def test_the_estimate_follows_a_change_of_the_link(link):
    link.add_transfer(50_000, _round_trip_ms(50_000, 50.0))
    for _ in range(25):  # 0.7 ** 25 of the old value is left
        link.add_transfer(50_000, _round_trip_ms(50_000, 2.0))
    assert link.bw_mbps == pytest.approx(2.0, rel=0.05)
