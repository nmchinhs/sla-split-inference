# Layer profiles

One JSON file per model and tier, written by `scripts/profile_layers.py` (PLAN.md Phase 2 step 6).
Meaning of the fields: `src/edgeinf/profiler/layer_profile.py`.

Device tier, in a device container with the calibrated CPU limit (`experiments/calibration.md`):

    docker compose run --rm --no-deps -T -v $PWD/src:/app/src:ro -v $PWD/scripts:/app/scripts:ro \
        -v $PWD/checkpoints:/app/checkpoints:ro device-1 \
        python scripts/profile_layers.py device mobilenetv2 --checkpoint checkpoints/train_mobilenetv2.pt \
        > experiments/profiles/mobilenetv2_device.json

Edge tier, in an edge container started from the compose file, so with the GPU and pinned to
CPUs 0-1 as the edge is in the experiments:

    docker compose run --rm --no-deps -T -v $PWD/src:/app/src:ro -v $PWD/scripts:/app/scripts:ro \
        -v $PWD/checkpoints:/app/checkpoints:ro edge \
        python scripts/profile_layers.py edge mobilenetv2 --checkpoint checkpoints/train_mobilenetv2.pt \
        > experiments/profiles/mobilenetv2_edge.json

The same for `resnet18`. Each value is the median of 100 runs after 10 warm-up runs. Keep the
machine idle while they run, and take all four in one session.
