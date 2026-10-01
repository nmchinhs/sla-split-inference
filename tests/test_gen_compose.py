import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gen_compose.py"
spec = importlib.util.spec_from_file_location("gen_compose", SCRIPT)
gen_compose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_compose)

N_CORES = 16


def test_devices_are_explicit_and_pinned_to_distinct_cores():
    services = gen_compose.build_compose(4, N_CORES)["services"]
    devices = {name: svc for name, svc in services.items() if name != "edge"}
    assert list(devices) == ["device-1", "device-2", "device-3", "device-4"]
    cores = [int(svc["cpuset"]) for svc in devices.values()]
    assert len(set(cores)) == len(cores)
    assert min(cores) >= gen_compose.RESERVED_CORES
    for svc in devices.values():
        assert svc["cpus"] == gen_compose.DEVICE_CPUS
        assert svc["mem_limit"] == gen_compose.DEVICE_MEMORY
        assert svc["cap_add"] == ["NET_ADMIN"]
        assert "gpus" not in svc


def test_edge_has_gpu_and_is_not_pinned():
    edge = gen_compose.build_compose(1, N_CORES)["services"]["edge"]
    assert edge["gpus"] == "all"
    assert "cpuset" not in edge


@pytest.mark.parametrize("n_devices", [0, N_CORES - 1])
def test_rejects_device_counts_that_do_not_fit(n_devices):
    with pytest.raises(ValueError):
        gen_compose.build_compose(n_devices, N_CORES)
