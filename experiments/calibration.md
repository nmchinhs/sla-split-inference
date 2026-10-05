# CPU calibration of the device containers

PLAN.md Phase 2 step 5. Measured on 2026-10-05 with `scripts/calibrate_cpu.py`. The configs are
in `experiments/configs/calibration_*.yaml`, one summary per config is in
`experiments/calibration/`, and the per-request rows are in `results/calibration/` (not in git).

## Result

| What | Recommended | Instead of |
|---|---|---|
| CPU limit of a device | `cpu_quota: 1050`, `cpu_period: 5000`: 21% of one CPU, about 1 ms every 5 ms | `cpus: 0.25` (25 ms every 100 ms) |
| Memory allocator of a device | `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=268435456` | glibc defaults |
| Pinning | one device per physical core while there are enough cores (CPUs 2, 4, 6, ...) | consecutive CPUs (2, 3, 4, 5), which are pairs of threads on one core |

**Status: measured and recommended, not yet applied.** `scripts/gen_compose.py` and
`docker/Dockerfile.device` are unchanged.

With these settings one MobileNetV2 inference at 224x224 takes about **101 ms** on the device when
requests arrive back to back and about 98 ms when they are separated by idle time. ResNet-18 takes
about 288 ms and 280 ms. The target was 103-111 ms for MobileNetV2 (section 1).

All numbers below were taken on the project laptop on mains power with the `performance`
governor (driver `amd-pstate-epp`), kernel 7.0 with a 1 ms scheduler tick, Docker 29.8 with the
systemd cgroup driver, one container pinned to CPU 8 with its sibling thread idle, random weights
(the time does not depend on the values of the weights; checked in section 2).

## 1. The target: a Raspberry Pi 4

| Source | Model and runtime | Raspberry Pi 4 |
|---|---|---|
| DeepEdgeBench [1] | MobileNetV2 **1.4x wide**, 224x224, 32-bit floats, TensorFlow | 1037.7 s for 5,000 images = 208 ms each |
| DeepEdgeBench [1] | the same model, TensorFlow Lite | 965.1 s = 193 ms each |
| DeepEdgeBench [1] | the same model quantised to 8 bits, TensorFlow Lite | 640.0 s = 128 ms each |
| PyTorch tutorial [3] | MobileNetV2 1.0x, quantised to 8 bits, TorchScript | 26.4 ms model time (33.7 frames/s) |

DeepEdgeBench is the peer-reviewed source. Its time "only includes the inference performed by the
framework, excluding image pre-processing and evaluation of the results", on 5,000 ImageNet
validation images, on a Pi 4 with 4 GB (quad-core Cortex-A72 at 1.5 GHz, 64-bit Raspbian).

Its MobileNetV2 is not the one this project uses. The paper names the graph `float_v2_1.4_224`,
which is the 1.4x-wide variant; torchvision's `mobilenet_v2` is the standard 1.0x. The official
model table [2] gives both on the same phone (a Pixel 1): 138.0 ms for 1.4x (582 M multiply-adds)
and 73.8 ms for 1.0x (300 M), a ratio of 0.535. Applying that ratio to the Pi 4 figures gives

**193-208 ms x 0.535 = 103-111 ms for the model used here.**

What this target is and is not:
- It is derived, not measured: no published Pi 4 figure was found for the 1.0x model with 32-bit
  floats. The paper does not say how many threads were used.
- The runtime matters as much as the board. The quantised PyTorch figure is four times faster
  than this estimate.
  The devices here run unquantised 32-bit floats, so the float figures are the ones to match.
- Using the paper's 193-208 ms directly would calibrate to a model twice as expensive. A setting
  for that is given in section 4 in case a slower device class is wanted.
- Only the total time is matched. How the time is divided among the blocks is the laptop's.

## 2. First problem: the memory allocator

Before any quota, the unthrottled time of MobileNetV2 itself was not stable: 18-20 ms in one
process and 13-16 ms in another, with the same weights and input.

Cause. Block 2 of MobileNetV2 works on 96x112x112 float tensors of 4.8 MB each. With its default
settings glibc's `malloc` gives such blocks back to the kernel when they are freed and gets fresh
pages when they are needed again, so **every forward pass takes about 2,600 page faults** (10 MB),
all in block 2. glibc raises its thresholds only when a larger block is freed, so a single large
allocation earlier in the process (for example one forward pass on a batch of 8) switches the
process to the fast behaviour for the rest of its life. ResNet-18 is not affected (0 faults).

It was not the values: restoring the original BatchNorm statistics in the same process kept it
fast, a new process with the changed statistics was slow again, scaling block 2's input by
1e-6 to 1e4 changed nothing, and no layer output contains subnormal numbers.

Fix. Two glibc tunables pin the thresholds, so freed tensors are kept and reused:

    GLIBC_TUNABLES=glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=268435456

Effect in the device image (`calibration_default_malloc.csv` against `calibration_check.csv`;
median over 3 repeats, MobileNetV2):

| | Page faults per request | Unthrottled | 21% quota, back to back | 21%, spaced | 21%, blocks [0, 5) back to back |
|---|---|---|---|---|---|
| glibc defaults | 2,614 | 18.5 ms | 141.1 ms | 130.7 ms | 71.9 ms |
| thresholds pinned | 0 | 16.2 ms | 103.6 ms | 98.6 ms | 46.5 ms |

So the allocator changes the calibrated latency by a third, and it changes the shape of the
profile: back to back, the device has spent 51% of the full-model time when it reaches split
point 5 with the defaults, and 45% with pinned thresholds. Peak memory is the same (362 MB against
355 MB resident for MobileNetV2 in a 512 MB container). Setting only the trim threshold makes
things much worse (46 ms and 16,440 faults per unthrottled pass, measured on the host), because
then every tensor is mapped and unmapped.

**Everything below uses the pinned thresholds, and the device runtime has to use them too.**

## 3. How a quota slows a request

`calibration_slice.yaml`: the share of CPU time is held at 15% while the length of one slice
varies. MobileNetV2, median of 3 repeats, in ms. "Blocks [0, 5)" is the device part of a request
split at k = 5.

| Slice / period (ms) | Whole model, back to back | Whole model, spaced | Blocks [0, 5), back to back | Blocks [0, 5), spaced | CPU time per request |
|---|---|---|---|---|---|
| no quota | 13.8 | 14.2 | 6.2 | 6.6 | 13.9 |
| 25 / 100 (Phase 1 setting, 25%) | 88.6 | **14.8** | 6.6 | 6.9 | 14.4 |
| 1.05 / 7 | 145.5 | 140.2 | 52.5 | 45.5 | 23.2 |
| 1.6 / 10.5 | 136.9 | 119.5 | 52.8 | 38.5 | 20.8 |
| 2.1 / 14 | 121.6 | 101.5 | 43.7 | 29.6 | 18.6 |
| 3.2 / 21 | 111.9 | 85.9 | 42.8 | 20.6 | 16.9 |
| 4.2 / 28 | 102.8 | 77.8 | 45.6 | 19.0 | 15.5 |
| 6.3 / 42 | 86.7 | 61.3 | 42.0 | **6.5** | 15.0 |
| 8.4 / 56 | 109.8 | 43.1 | 53.5 | 7.3 | 15.2 |
| 14.7 / 98 | 97.7 | 17.2 | 33.6 | 7.0 | 14.6 |

What the table shows:
1. **A request's latency is its CPU time plus a number of pauses.** After each slice the container
   waits for the rest of the period. With long slices a request meets only 0-3 pauses, so its
   latency jumps between a few values, and work that fits into one slice is not slowed at all:
   blocks [0, 5) take 6.5 ms when requests are spaced and 42 ms when they are back to back.
2. **The Phase 1 setting does not emulate a slow device.** A whole inference fits into its 25 ms
   slice, so a spaced request runs at full laptop speed (14.8 ms), and so does the device part of
   every split request.
3. **Short slices cost extra CPU work.** During a pause the core goes idle (its deepest idle
   state takes 350 us to leave) and has to warm up again; each restart costs about 0.5 ms of
   work. CPU time per request rises from 13.9 ms to 23.2 ms with 1 ms slices. The container does
   get its share (0.150 measured), the work itself is slower. This is harmless once the share is
   calibrated on the real workload, which is what section 4 does.
4. **A spaced request meets one to two pauses fewer** than one in a back-to-back stream, at every
   slice length, because it starts with a full slice. The difference therefore only becomes small
   when a pause is short.

So the slice has to be as short as possible. Docker's minimum quota is 1 ms.

`calibration_period.yaml` compares slices of 1, 1.5 and 2 ms at about the same latency
(96-114 ms back to back, 5 repeats each):

| Slice | Spaced faster than back to back, whole model | the same, blocks [0, 5) | CPU time per request |
|---|---|---|---|
| 1 ms | 6-9% | 17-21% | 21.9-23.2 ms |
| 1.5 ms | 11-12% | 25-28% | 20.8-21.3 ms |
| 2 ms | 14-16% | 31-39% | 18.9-19.3 ms |

The spread of the whole-model latency is the same for all three (standard deviation 6-8% of the
mean). **A slice of about 1 ms is the best available**: it halves the dependence on how requests
are spaced, at no cost in stability.

## 4. The share

Docker does not enforce exactly what is asked for. With the systemd cgroup driver the share
quota / period is **rounded up to a whole percent of a CPU**, and the kernel gets that percentage
of the period (read from `cpu.max` in the container):

| Asked for (us) | Kernel got (us) | Share |
|---|---|---|
| 1000 / 4700 | 1034 / 4700 | 22% |
| 1000 / 4900 | 1029 / 4900 | 21% |
| 1000 / 5000 | 1000 / 5000 | 20% |
| 1000 / 5100 | 1020 / 5100 | 20% |
| 1000 / 9700 | 1067 / 9700 | 11% |
| 1050 / 5000 | 1050 / 5000 | 21% |

A setting should therefore be an exact whole percent, so that it means the same on every host.
The script warns when a setting is changed by the rounding. (In sections 3 and 5 the settings were
asked for as 1000 us and rounded; the tables give what the kernel enforced.)

`calibration_final.yaml`: exact shares with a 5 ms period. Median over 5 repeats, with the lowest
and highest repeat in brackets, in ms.

| Share (quota per 5 ms) | MobileNetV2, back to back | MobileNetV2, spaced | ResNet-18, back to back | ResNet-18, spaced |
|---|---|---|---|---|
| no quota | 14.1 (12.8-15.4) | 14.7 (13.3-16.6) | 41.5 (38.4-45.8) | 41.9 (39.0-45.4) |
| 20% (1000 us) | 110.2 (105.4-121.0) | 101.4 (97.1-107.7) | 307.2 (298.1-329.5) | 294.9 (289.6-308.7) |
| **21% (1050 us)** | **101.4 (97.2-106.0)** | **97.6 (90.0-112.7)** | **288.3 (283.2-290.2)** | **280.2 (275.7-286.2)** |
| 22% (1100 us) | 99.2 (95.7-105.9) | 90.1 (85.9-93.6) | 270.5 (265.7-275.7) | 265.3 (256.8-273.2) |

The published figure is an average over a back-to-back loop, so that column is the one to
compare. 21% gave 101.4 ms here and 103.6 ms in the earlier run of section 5, against a target of
103-111 ms; 20% gave 110.2 ms here and 114.3 ms in `calibration_period.csv`. Both are in the range
that the target can claim. **21% is recommended**; 20% is equally defensible.

For a device class twice as slow (the paper's 193-208 ms taken as it is): 11% in 1 ms slices
(1067 us per 9700 us) gave 228 ms back to back and 211 ms spaced for MobileNetV2, and 566 ms and
550 ms for ResNet-18 (`calibration_check.csv`). About 12% should give 200 ms; that was not
measured.

## 5. Check on several split points

`calibration_check.yaml`, at 21% (asked for as 1000 / 4900 us). Median of 3 repeats in ms; in
brackets the share of the whole-model time.

| | Blocks run | No quota, back to back | 21%, back to back | 21%, spaced |
|---|---|---|---|---|
| MobileNetV2 | [0, 1) | 0.4 | 0.7 | 0.9 |
| | [0, 5) | 6.7 (41%) | 46.5 (45%) | 35.2 (36%) |
| | [0, 12) | 10.4 (64%) | 72.1 (70%) | 63.7 (65%) |
| | whole model | 16.2 | 103.6 | 98.6 |
| ResNet-18 | [0, 6) | 26.8 (62%) | 177.4 (60%) | 171.3 (61%) |
| | whole model | 43.4 | 294.9 | 283.1 |

- Both models are slowed by about the same factor (6.4 and 6.8 back to back).
- The division of time among the blocks stays close to the unthrottled one, within about 6
  percentage points, except for short prefixes of spaced requests: the one to two pauses that a
  spaced request saves are a visible part of the time of blocks [0, 5).
- Work shorter than one slice is not slowed (blocks [0, 1)).

Consequence for the layer profile (PLAN.md Phase 2 step 6): measure the device time of each
prefix [0, k) directly, in the device container, with requests arriving as they will in the
experiments. Sums of per-block times measured one block at a time would be too small.

## 6. Several devices at once

`calibration_pair.yaml`, MobileNetV2 at 21%, the same workload in every container at the same
time. Each scenario was run twice; the table gives the lowest and highest median over all devices
and both runs, in ms (`calibration_pair.csv`).

| Devices running | Back to back | Spaced |
|---|---|---|
| one, alone (CPU 8) | 102-108 | 100-103 |
| two on different cores (CPUs 8, 10) | 99-104 | 94-98 |
| four on different cores (CPUs 8, 10, 12, 14) | 104-113 | 95-102 |
| two on the two threads of one core (CPUs 8, 9) | 86-103 | 83-92 |
| four on two cores (CPUs 8-11), as `gen_compose.py` pins them now | 97-126 | 94-119 |

- Devices on different cores hardly disturb each other (a few percent with four of them).
- Two devices on the two threads of one core are **faster** than one alone, by up to 20%: the
  sibling keeps the core awake, so there is less of the restart cost of section 3. The speed of a
  device then depends on what its neighbour is doing, and devices differ from each other by up to
  25%.
- `gen_compose.py` pins devices to CPUs 2, 3, 4, 5, ..., and on this laptop CPUs 2 and 3 are one
  core, as are 4 and 5. Pinning to 2, 4, 6, ... gives up to seven devices a core of their own.
  With 8 or 10 devices some have to share a core, and the report should say so.

## 7. Limits

- The calibration holds for this laptop on mains power with the `performance` governor. On
  battery or with another power profile the numbers will differ. They should be re-measured
  before the experiment sweep and whenever the host setup changes.
- The laptop's own speed is not constant: the same unthrottled model was up to 20% slower in one
  repeat than in another a few minutes later (12.8-15.4 ms for MobileNetV2 in section 4, with
  ResNet-18 moving the same way). Differences of a few percent between settings are below this.
- The device runtime allocates memory in a different order than this probe, which can also move
  the time by several percent. Repeat the back-to-back check with the real runtime (Phase 2
  step 7) and adjust the quota by one percent if needed.
- The target is derived from a wider model (section 1), and only total inference time is matched.
- The 1 ms slice is the minimum Docker accepts. Requests are therefore never slowed perfectly
  evenly: 4% for the whole model and about 19% for blocks [0, 5) between spaced and back-to-back
  requests at the recommended setting.
- The rounding to whole percents belongs to the systemd cgroup driver. The recommended setting is
  an exact percent, so it does not depend on it.

## 8. To reproduce

    conda run -n edgeinf python scripts/calibrate_cpu.py experiments/configs/calibration_slice.yaml
    conda run -n edgeinf python scripts/calibrate_cpu.py experiments/configs/calibration_period.yaml
    conda run -n edgeinf python scripts/calibrate_cpu.py experiments/configs/calibration_check.yaml
    conda run -n edgeinf python scripts/calibrate_cpu.py experiments/configs/calibration_default_malloc.yaml
    conda run -n edgeinf python scripts/calibrate_cpu.py experiments/configs/calibration_final.yaml
    # two devices at once: start both commands together
    conda run -n edgeinf python scripts/calibrate_cpu.py experiments/configs/calibration_pair.yaml --cpu 8
    conda run -n edgeinf python scripts/calibrate_cpu.py experiments/configs/calibration_pair.yaml --cpu 9

Each run needs the `edgeinf-device` image and starts its own throwaway container; the testbed
containers are not touched. The sweeps take 2 to 12 minutes each. Keep the machine idle meanwhile.

## Sources

1. S. P. Baller, A. Jindal, M. Chadha, M. Gerndt, "DeepEdgeBench: Benchmarking Deep Neural
   Networks on Edge Devices," IEEE IC2E 2021. Preprint: https://arxiv.org/abs/2108.09457
   (read from the preprint; page numbers and DOI of the published version not checked).
2. TensorFlow Model Garden, MobileNet model table,
   https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md
3. T. Rice, "Real Time Inference on Raspberry Pi 4 and 5", PyTorch tutorials,
   https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html
