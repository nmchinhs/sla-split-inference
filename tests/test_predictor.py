"""The predictor (scheduler/predictor.py): latency and accuracy of a plan from the profiles."""

import pytest

from edgeinf.scheduler.predictor import Link, Profiles, load_profiles, predict


def _row(tau: float, **columns: list[float]) -> dict:
    return {"tau": tau, **columns}


# A model of four blocks with exits after blocks 0 and 2. Times in ms.
PROFILES = Profiles(
    model="toy",
    n_blocks=4,
    device={
        "device_ms": [1, 10, 20, 30, 40],
        "device_p95_ms": [2, 13, 25, 36, 48],
        "bytes": [1000, 5000, 2500, 1250, 0],
        "exit_blocks": [0, 2],
        "exit_ms": [12, 34],
        "exit_p95_ms": [15, 40],
        "device_exits_ms": [1, 12, 22, 34, 44],
        "device_exits_p95_ms": [2, 15, 27, 40, 52],
    },
    edge={"edge_ms": [4, 3, 2, 1], "edge_p95_ms": [5, 4, 3, 2]},
    exits={
        "final_accuracy": 0.9,
        "by_tau": [
            _row(0.5, exit_rate=[0.6, 0.36], p_continue=[0.4, 0.04], accuracy=[0.85, 0.8]),
            _row(0.8, exit_rate=[0.8, 0.14], p_continue=[0.2, 0.06], accuracy=[0.86, 0.84]),
            _row(0.9, exit_rate=[0.3, 0.2], p_continue=[0.7, 0.5], accuracy=[0.89, 0.88]),
            _row(1.0, exit_rate=[0.0, 0.0], p_continue=[1.0, 1.0], accuracy=[0.9, 0.9]),
        ],
    },
    jpeg={
        50: {
            "device_ms": 0.7,
            "device_p95_ms": 0.9,
            "bytes_mean": 500,
            "bytes_p95": 750,
            "edge_ms": 5,
            "edge_p95_ms": 6,
            "accuracy": 0.88,
        }
    },
)
LINK = Link(bw_mbps=1.0, rtt_ms=20.0)  # 1000 bytes take 8 ms


def test_all_local_without_exits_is_the_device_time_of_the_whole_model():
    prediction = predict(PROFILES, k=4, tau=1.0, link=LINK)
    assert (prediction.mean_ms, prediction.p95_ms) == (40, 48)
    assert (prediction.accuracy, prediction.p_continue) == (0.9, 1.0)
    assert [(path.name, path.share) for path in prediction.paths] == [("local", 1.0)]


def test_a_split_without_exits_adds_transfer_round_trip_and_edge():
    prediction = predict(PROFILES, k=2, tau=1.0, link=LINK)
    assert prediction.mean_ms == pytest.approx(20 + 20 + 20 + 2)  # 2500 bytes take 20 ms
    assert prediction.p95_ms == pytest.approx(25 + 20 + 20 + 3)
    assert prediction.accuracy == 0.9


def test_everything_offloaded_sends_the_raw_frame():
    assert predict(PROFILES, k=0, tau=0.5, link=LINK).mean_ms == pytest.approx(1 + 8 + 20 + 4)


def test_requests_ahead_at_the_edge_multiply_its_time():
    busy = Link(bw_mbps=1.0, rtt_ms=20.0, queue_len=2)
    assert predict(PROFILES, k=2, tau=1.0, link=busy).mean_ms == pytest.approx(60 + 3 * 2)
    assert predict(PROFILES, k=4, tau=1.0, link=busy).mean_ms == 40  # nothing is sent


def test_only_the_exits_before_the_split_point_are_in_use():
    # k = 2: the exit after block 0 is checked, the one after block 2 is behind the split point.
    prediction = predict(PROFILES, k=2, tau=0.9, link=LINK)
    shares = [(path.name, path.share) for path in prediction.paths]
    assert shares == [("exit 0", 0.3), ("edge", 0.7)]
    slow = 22 + 20 + 20 + 2  # the device time with the head run
    assert prediction.paths[1].ms == pytest.approx(slow)
    assert prediction.mean_ms == pytest.approx(0.3 * 12 + 0.7 * slow)
    assert (prediction.accuracy, prediction.p_continue) == (0.89, 0.7)


def test_the_mean_is_the_paths_weighted_by_their_shares():
    prediction = predict(PROFILES, k=4, tau=0.5, link=LINK)
    assert prediction.mean_ms == pytest.approx(0.6 * 12 + 0.36 * 34 + 0.04 * 44)
    assert (prediction.accuracy, prediction.p_continue) == (0.8, 0.04)
    assert sum(path.share for path in prediction.paths) == pytest.approx(1.0)


def test_the_p95_is_read_inside_the_path_the_95th_percentile_falls_in():
    # tau = 0.9: half the requests take the slow path, so the 95th percentile of all requests
    # is the 90th of that path. A path's time is linear in its quantile through its median
    # (44 ms) and its 95th percentile (52 ms).
    assert predict(PROFILES, k=4, tau=0.9, link=LINK).p95_ms == pytest.approx(
        44 + (0.9 - 0.5) / 0.45 * (52 - 44)
    )
    # With one exit in use 40% go on to the edge: quantile 1 - 0.05 / 0.4 of that path.
    assert predict(PROFILES, k=2, tau=0.5, link=LINK).p95_ms == pytest.approx(
        64 + (0.875 - 0.5) / 0.45 * (70 - 64)
    )


def test_the_p95_of_a_plan_with_one_path_is_that_paths_p95():
    assert predict(PROFILES, k=4, tau=1.0, link=LINK).p95_ms == 48
    assert predict(PROFILES, k=2, tau=1.0, link=LINK).p95_ms == pytest.approx(68)


def test_the_p95_is_in_an_exit_path_when_under_five_percent_go_on():
    # tau = 0.5: 60% leave at the first exit, 36% at the second and 4% run to the end. The
    # 95th percentile is in the second exit's path, at its quantile 0.35 / 0.36.
    quantile = (0.95 - 0.6) / 0.36
    assert predict(PROFILES, k=4, tau=0.5, link=LINK).p95_ms == pytest.approx(
        34 + (quantile - 0.5) / 0.45 * (40 - 34)
    )


def test_the_p95_can_fall_in_the_fast_half_of_a_path():
    # tau = 0.8: the exits answer 94%, so the 95th percentile is the slow path's quantile
    # 0.01 / 0.06, below its median of 44 ms.
    quantile = (0.95 - 0.94) / 0.06
    prediction = predict(PROFILES, k=4, tau=0.8, link=LINK)
    assert prediction.p95_ms == pytest.approx(44 + (quantile - 0.5) / 0.45 * (52 - 44))
    assert 34 < prediction.p95_ms < 44


def test_with_tau_one_no_head_runs_so_the_plain_device_time_counts():
    with_heads = predict(PROFILES, k=4, tau=0.9, link=LINK).paths[-1]
    without = predict(PROFILES, k=4, tau=1.0, link=LINK).paths[-1]
    assert (with_heads.ms, without.ms) == (44, 40)


def test_a_split_point_before_the_first_exit_has_no_exit_whatever_tau():
    assert predict(PROFILES, k=0, tau=0.5, link=LINK) == predict(PROFILES, k=0, tau=1.0, link=LINK)


def test_the_frame_as_a_jpeg_has_its_own_sizes_times_and_accuracy():
    prediction = predict(PROFILES, k=0, tau=1.0, link=LINK, jpeg_quality=50)
    assert prediction.mean_ms == pytest.approx(0.7 + 4 + 20 + 5)  # 500 bytes on average
    assert prediction.p95_ms == pytest.approx(0.9 + 6 + 20 + 6)  # a large one: 750 bytes
    assert prediction.accuracy == 0.88


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"k": 5, "tau": 1.0}, "split point"),
        ({"k": -1, "tau": 1.0}, "split point"),
        ({"k": 4, "tau": 0.7}, "tau"),  # not a threshold the exit profile has
        ({"k": 2, "tau": 1.0, "jpeg_quality": 50}, "JPEG"),
        ({"k": 0, "tau": 1.0, "jpeg_quality": 75}, "quality"),
    ],
)
def test_plans_the_profiles_do_not_cover_are_refused(arguments, message):
    with pytest.raises(ValueError, match=message):
        predict(PROFILES, link=LINK, **arguments)


def test_a_link_must_have_a_bandwidth():
    with pytest.raises(ValueError, match="bandwidth"):
        predict(PROFILES, k=2, tau=1.0, link=Link(bw_mbps=0.0, rtt_ms=20.0))
    predict(PROFILES, k=4, tau=1.0, link=Link(bw_mbps=0.0, rtt_ms=20.0))  # all local: no link


# ------------------------------------------------------------------------------ the saved profiles


@pytest.mark.parametrize(("model", "n_blocks"), [("mobilenetv2", 20), ("resnet18", 10)])
def test_the_saved_profiles_load_and_cover_every_plan(model, n_blocks):
    profiles = load_profiles(model)
    assert profiles.n_blocks == n_blocks
    link = Link(bw_mbps=5.0, rtt_ms=20.0)
    for row in profiles.exits["by_tau"]:
        for k in range(n_blocks + 1):
            prediction = predict(profiles, k, row["tau"], link)
            assert 0 < prediction.mean_ms
            assert prediction.paths[0].ms < prediction.p95_ms <= prediction.paths[-1].p95_ms
            assert 0.5 < prediction.accuracy < 1.0
    for quality in profiles.jpeg:
        assert predict(profiles, 0, 1.0, link, jpeg_quality=quality).mean_ms > 20


def test_the_saved_profiles_give_the_numbers_of_the_report_log():
    profiles = load_profiles("mobilenetv2")
    nowhere = Link(bw_mbps=1.0, rtt_ms=20.0)
    plain = predict(profiles, 20, 1.0, nowhere)
    assert plain.mean_ms == pytest.approx(94.5, abs=0.1)
    assert plain.accuracy == pytest.approx(0.8923, abs=1e-4)
    # Finding of 2026-11-12: all local with exits at tau = 0.9 averages about 61 ms.
    exits = predict(profiles, 20, 0.9, nowhere)
    assert exits.mean_ms == pytest.approx(61, abs=1.0)
    assert exits.p_continue == pytest.approx(0.189, abs=1e-3)
    assert exits.accuracy == pytest.approx(0.887, abs=1e-3)
    assert exits.p95_ms > 100  # in the slow path: 19% take it
    # Only tau = 0.5 leaves under 5% for it.
    assert predict(profiles, 20, 0.5, nowhere).p95_ms < 100
