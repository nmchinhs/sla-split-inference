"""Measure achieved uplink throughput and RTT from a device to the edge.

Run inside a device container, after shaping its egress with scripts/netem.sh:

    docker exec device-1 python scripts/smoke_test.py --expect-mbps 5 --expect-rtt-ms 20

RTT is the median round trip of an empty Health() call. Throughput is payload bits divided
by the Infer() round trip minus that RTT, so it is the rate seen by the tensor itself.
With --expect-* the script exits non-zero if a measurement is out of tolerance.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

from edgeinf.common.grpc_options import DEFAULT_EDGE_PORT
from edgeinf.device.client import EdgeClient

DEFAULT_PAYLOAD_BYTES = 1_000_000
DEFAULT_REPEATS = 5
RTT_PROBES = 20
WARMUP_CALLS = 3
TIMEOUT_S = 60.0
THROUGHPUT_TOLERANCE = 0.10  # PLAN.md Phase 1: within +-10% of the set rate
RTT_SLACK_MS = 10.0  # allowed RTT above the set one-way delay
RTT_UNDERSHOOT_MS = 1.0
BITS_PER_BYTE = 8
MS_PER_S = 1000.0
BITS_PER_MBIT = 1e6


def measure_rtt_ms(client: EdgeClient, probes: int = RTT_PROBES) -> float:
    """Median round-trip time of Health(), in ms."""
    samples = []
    for _ in range(probes):
        start = time.perf_counter()
        if client.health() is None:
            raise RuntimeError("edge is unreachable or unhealthy")
        samples.append((time.perf_counter() - start) * MS_PER_S)
    return statistics.median(samples)


def measure_throughput_mbps(
    client: EdgeClient, payload_bytes: int, repeats: int, rtt_ms: float
) -> list[float]:
    """Achieved uplink throughput in Mbps for each of `repeats` transfers of `payload_bytes`."""
    payload = os.urandom(payload_bytes)
    samples = []
    for i in range(repeats):
        start = time.perf_counter()
        reply = client.infer(payload, k=0, request_id=i)
        elapsed_s = time.perf_counter() - start
        if reply.tensor_bytes != payload_bytes:
            raise RuntimeError(f"edge saw {reply.tensor_bytes} bytes, sent {payload_bytes}")
        transfer_s = max(elapsed_s - rtt_ms / MS_PER_S, 1e-9)
        samples.append(payload_bytes * BITS_PER_BYTE / transfer_s / BITS_PER_MBIT)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--edge", default=os.environ.get("EDGE_ADDR", f"localhost:{DEFAULT_EDGE_PORT}")
    )
    parser.add_argument("--payload-bytes", type=int, default=DEFAULT_PAYLOAD_BYTES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--expect-mbps", type=float, help="netem rate to check against")
    parser.add_argument("--expect-rtt-ms", type=float, help="netem delay to check against")
    args = parser.parse_args()

    with EdgeClient(args.edge, os.environ.get("DEVICE_ID", "smoke"), TIMEOUT_S) as client:
        for _ in range(WARMUP_CALLS):  # connection setup must not count towards RTT
            if client.health() is None:
                print(f"FAIL: cannot reach edge at {args.edge}")
                return 1
        rtt_ms = measure_rtt_ms(client)
        mbps = measure_throughput_mbps(client, args.payload_bytes, args.repeats, rtt_ms)

    mean_mbps = statistics.mean(mbps)
    print(f"edge:        {args.edge}")
    print(f"payload:     {args.payload_bytes} bytes x {args.repeats}")
    print(f"rtt:         {rtt_ms:.2f} ms (median of {RTT_PROBES})")
    print(f"throughput:  {mean_mbps:.2f} Mbps mean (min {min(mbps):.2f}, max {max(mbps):.2f})")

    ok = True
    if args.expect_mbps is not None:
        passed = abs(mean_mbps - args.expect_mbps) <= THROUGHPUT_TOLERANCE * args.expect_mbps
        ok &= passed
        print(
            f"{'PASS' if passed else 'FAIL'}: throughput within "
            f"{THROUGHPUT_TOLERANCE:.0%} of {args.expect_mbps} Mbps"
        )
    if args.expect_rtt_ms is not None:
        low = args.expect_rtt_ms - RTT_UNDERSHOOT_MS
        high = args.expect_rtt_ms + RTT_SLACK_MS
        passed = low <= rtt_ms <= high
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}: rtt in [{low:.0f}, {high:.0f}] ms")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
