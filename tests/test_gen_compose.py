import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gen_compose.py"
spec = importlib.util.spec_from_file_location("gen_compose", SCRIPT)
gen_compose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_compose)

N_CPUS = 16
PAIRED = [cpu // 2 for cpu in range(N_CPUS)]  # CPUs 0-1 are core 0, 2-3 core 1, ...
UNPAIRED = list(range(N_CPUS))  # no hardware threads: every CPU is its own core
SPLIT_PAIRS = [cpu % 8 for cpu in range(N_CPUS)]  # CPU k and CPU k + 8 are one core
# CPUs 2 and 3 are the second threads of the reserved CPUs 0 and 1; cores 2 and 3 follow.
RESERVED_PAIRED_FIRST = [0, 1, 0, 1, 2, 2, 3, 3]
WHOLE_PERCENT = 100


def test_devices_are_explicit_and_limited():
    services = gen_compose.build_compose(4, PAIRED)["services"]
    devices = {name: svc for name, svc in services.items() if name != "edge"}
    assert list(devices) == ["device-1", "device-2", "device-3", "device-4"]
    for svc in devices.values():
        assert svc["cpu_quota"] == gen_compose.DEVICE_CPU_QUOTA_US
        assert svc["cpu_period"] == gen_compose.DEVICE_CPU_PERIOD_US
        assert "cpus" not in svc  # Docker refuses cpus together with cpu_period
        assert svc["mem_limit"] == gen_compose.DEVICE_MEMORY
        assert svc["cap_add"] == ["NET_ADMIN"]
        assert "gpus" not in svc


def test_cpu_share_is_an_exact_whole_percent_with_a_quota_docker_accepts():
    quota, period = gen_compose.DEVICE_CPU_QUOTA_US, gen_compose.DEVICE_CPU_PERIOD_US
    assert quota * WHOLE_PERCENT % period == 0
    assert quota >= 1_000


def test_first_devices_get_a_physical_core_each():
    services = gen_compose.build_compose(7, PAIRED)["services"]
    cpus = [int(services[f"device-{i}"]["cpuset"]) for i in range(1, 8)]
    assert cpus == [2, 4, 6, 8, 10, 12, 14]


def test_further_devices_share_cores_and_no_cpu_is_used_twice():
    cpus = gen_compose.device_cpus(PAIRED)
    assert cpus[7:10] == [3, 5, 7]
    assert len(set(cpus)) == len(cpus) == N_CPUS - gen_compose.RESERVED_CORES
    assert min(cpus) >= gen_compose.RESERVED_CORES


def test_pinning_follows_the_cores_not_the_cpu_numbers():
    # CPUs 0 and 1 are reserved, so their second threads (8 and 9) are not cores of their own.
    cpus = gen_compose.device_cpus(SPLIT_PAIRS)
    assert cpus[:6] == [2, 3, 4, 5, 6, 7]
    assert sorted(cpus[6:]) == [8, 9, 10, 11, 12, 13, 14, 15]


def test_a_thread_of_a_reserved_core_is_not_a_core_of_its_own():
    # Devices first get the cores nobody else uses (CPUs 4 and 6), although CPUs 2 and 3 come
    # first by number: those share a core with the OS and the edge.
    assert gen_compose.device_cpus(RESERVED_PAIRED_FIRST) == [4, 6, 2, 3, 5, 7]


def test_without_hardware_threads_devices_take_consecutive_cpus():
    assert gen_compose.device_cpus(UNPAIRED)[:3] == [2, 3, 4]


def test_edge_has_gpu_and_keeps_to_the_reserved_cpus():
    services = gen_compose.build_compose(N_CPUS - gen_compose.RESERVED_CORES, PAIRED)["services"]
    edge = services.pop("edge")
    assert edge["gpus"] == "all"
    assert edge["cpuset"] == "0-1"
    # No device is ever placed on a CPU the edge may use.
    assert all(int(device["cpuset"]) >= gen_compose.RESERVED_CORES for device in services.values())


@pytest.mark.parametrize("n_devices", [0, N_CPUS - 1])
def test_rejects_device_counts_that_do_not_fit(n_devices):
    with pytest.raises(ValueError):
        gen_compose.build_compose(n_devices, PAIRED)


def test_edge_is_given_the_checkpoints_and_told_which_to_serve():
    edge = gen_compose.build_compose(1, PAIRED)["services"]["edge"]
    assert edge["volumes"] == ["./checkpoints:/app/checkpoints:ro"]
    served = edge["environment"]["EDGE_CHECKPOINTS"].split(",")
    assert served == ["checkpoints/train_mobilenetv2.pt", "checkpoints/train_resnet18.pt"]
    # Every file it is told to serve is inside the folder it is given.
    assert all(path.startswith("checkpoints/") for path in served)


def test_devices_are_given_checkpoints_and_frames_read_only():
    services = gen_compose.build_compose(3, PAIRED)["services"]
    for i in range(1, 4):
        volumes = services[f"device-{i}"]["volumes"]
        assert volumes == ["./checkpoints:/app/checkpoints:ro", "./data/frames:/app/data/frames:ro"]
        # A device has no dataset and writes nothing to the host.
        assert all(volume.endswith(":ro") for volume in volumes)
        assert not any("imagenette" in volume or "results" in volume for volume in volumes)
