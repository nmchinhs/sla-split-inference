# Layer profiles

One JSON file per model and tier, written by `scripts/profile_layers.py` (PLAN.md Phase 2 step 6).
Meaning of the fields: `src/edgeinf/profiler/layer_profile.py`.

Device tier, in a device container with the calibrated CPU limit (`experiments/calibration.md`):

    docker compose run --rm --no-deps -T -v $PWD/src:/app/src:ro -v $PWD/scripts:/app/scripts:ro \
        -v $PWD/checkpoints:/app/checkpoints:ro device-1 \
        python scripts/profile_layers.py device mobilenetv2 --checkpoint checkpoints/exits_mobilenetv2.pt \
        > experiments/profiles/mobilenetv2_device.json

Edge tier, in an edge container started from the compose file, so with the GPU and pinned to
CPUs 0-1 as the edge is in the experiments:

    docker compose run --rm --no-deps -T -v $PWD/src:/app/src:ro -v $PWD/scripts:/app/scripts:ro \
        -v $PWD/checkpoints:/app/checkpoints:ro edge \
        python scripts/profile_layers.py edge mobilenetv2 --checkpoint checkpoints/exits_mobilenetv2.pt \
        > experiments/profiles/mobilenetv2_edge.json

The same for `resnet18`. Each value is the median of 100 runs after 10 warm-up runs. Keep the
machine idle while they run, and take all four in one session.

Every timed run is preceded by idle time, as a request in the system is: 50 ms on the device,
200 ms on the edge (`gap_ms` in each file). A GPU that has been idle is slower on the first work
it gets, so an edge profile taken back to back is too fast: by 0.6 to 1.3 ms for MobileNetV2 and
0.4 to 0.9 ms for ResNet-18 (the edge files of 2026-11-12, without a gap, against those of
2026-11-13).

With a checkpoint that has exit heads (`checkpoints/exits_<model>.pt`, written by
`scripts/train_exits.py`) the device profile also holds `exit_blocks`, `exit_ms` and
`device_exits_ms`: the times with the heads running. The device files here were taken that way
on 2026-11-12 and the edge files on 2026-11-13, on the pinned images (torch 2.15.0).
