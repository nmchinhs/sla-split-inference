# JPEG for the `offload_jpeg` variant

Written by `scripts/profile_jpeg.py` (PLAN.md Phase 2 step 7). Meaning of the fields:
`src/edgeinf/profiler/jpeg_profile.py`. Everything here is measured on the **validation** split
(947 frames, `data/frames/val_frames.npy` from `scripts/export_frames.py`), because the quality
is a tuned value.

Device tier, in a device container with the calibrated CPU limit: the time to turn a frame into
the bytes to send, raw and as a JPEG of quality 50, 75, 90 and 95.

    docker compose run --rm --no-deps -T device-1 \
        python scripts/profile_jpeg.py device > experiments/jpeg/device.json

Edge tier, in an edge container started from the compose file (GPU, pinned to CPUs 0-1): accuracy
and bytes over all 947 frames, the time from the received bytes to the answer, and the quality
that the rule picks.

    docker compose run --rm --no-deps -T -v $PWD/scripts:/app/scripts:ro \
        -v $PWD/data/frames:/app/data/frames:ro edge \
        python scripts/profile_jpeg.py edge mobilenetv2 --checkpoint checkpoints/train_mobilenetv2.pt \
        > experiments/jpeg/mobilenetv2_edge.json

The same for `resnet18`. Times are the median of 100 runs after 10 warm-up runs, each run on
another frame, 100 frames spread evenly over the split.

The rule (report log, D19): the lowest quality that loses less than half a percentage point of
accuracy against the raw 8-bit frame. The result is `chosen_quality` in the edge files and
`jpeg_quality` in the experiment configs.
