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


def test_without_hardware_threads_devices_take_consecutive_cpus():
    assert gen_compose.device_cpus(UNPAIRED)[:3] == [2, 3, 4]


def test_edge_has_gpu_and_is_not_pinned():
    edge = gen_compose.build_compose(1, PAIRED)["services"]["edge"]
    assert edge["gpus"] == "all"
    assert "cpuset" not in edge


@pytest.mark.parametrize("n_devices", [0, N_CPUS - 1])
def test_rejects_device_counts_that_do_not_fit(n_devices):
    with pytest.raises(ValueError):
        gen_compose.build_compose(n_devices, PAIRED)
