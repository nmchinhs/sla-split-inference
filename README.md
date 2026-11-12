# SLA-Aware Split Inference with Early Exit for IoT

This is a hands-on term project for the **Distributed Systems** course (CSIE52400, National Dong
Hwa University). It explores Edge AI and IoT integration, building mainly on SPINN [1].

## What it does

A multi-exit CNN (MobileNetV2 or ResNet-18, trained on Imagenette at 224×224) is split between an
IoT device and an edge server. At runtime a scheduler picks two values for each request:

- **`k`**, the split point: the device runs the first `k` blocks and the edge runs the rest.
- **`tau`**, the early-exit threshold: if a device-side exit is confident enough, the answer is
  returned at once and nothing is sent.

The goal is to meet a user-defined SLA, such as a p95 latency bound with the best possible
accuracy, while bandwidth and edge load change. Many devices can share one edge, and a device
falls back to local inference when the edge fails.

## Testbed

Everything runs on one laptop with Docker Compose:

- **`device-*` containers** are CPU-only and CPU-capped to behave like a Raspberry Pi 4. Each one
  is pinned to its own core.
- **The `edge` container** has the GPU.
- **The network** between them is emulated with `tc netem`. Bandwidth traces are in
  `experiments/traces/`.

Devices talk to the edge over gRPC (`proto/inference.proto`).

## Quick start

```bash
make setup                                              # conda env "edgeinf" (Python 3.11)
make proto                                              # generate the gRPC stubs
make test                                               # pytest
make lint                                               # ruff check + format check

conda run -n edgeinf python scripts/gen_compose.py 4    # docker-compose.yml with 4 devices
docker compose up --build

scripts/netem.sh <container> <rate_mbit> <delay_ms> [loss_pct]
conda run --no-capture-output -n edgeinf python scripts/run_experiment.py experiments/configs/<name>.yaml
```

Every experiment is driven by a YAML file in `experiments/configs/`. Raw results go to `results/`,
and `scripts/plots.py` generates the figures from them.

## Layout

```
proto/          gRPC service definition
src/edgeinf/    models, profilers, scheduler, device and edge runtimes, shared utilities
docker/         device (CPU) and edge (CUDA) images
scripts/        training, profiling, calibration, experiment runner, plots, netem
experiments/    configs, bandwidth traces, profiles, calibration notes
tests/          pytest suite
```

## References

Main basis:

1. S. Laskaridis, S. I. Venieris, M. Almeida, I. Leontiadis and N. D. Lane, "SPINN: Synergistic
   Progressive Inference of Neural Networks over Device and Cloud," in *Proc. ACM MobiCom*, 2020.
   doi:[10.1145/3372224.3419194](https://doi.org/10.1145/3372224.3419194)

Split computing, early exiting and edge inference:

2. E. Li, L. Zeng, Z. Zhou and X. Chen, "Edge AI: On-Demand Accelerating Deep Neural Network
   Inference via Edge Computing," *IEEE Trans. Wireless Commun.*, vol. 19, no. 1, pp. 447–457, 2020.
   doi:[10.1109/TWC.2019.2946140](https://doi.org/10.1109/TWC.2019.2946140)
3. L. Zeng, X. Chen, Z. Zhou, L. Yang and J. Zhang, "CoEdge: Cooperative DNN Inference With
   Adaptive Workload Partitioning Over Heterogeneous Edge Devices," *IEEE/ACM Trans. Netw.*,
   vol. 29, no. 2, pp. 595–608, 2021.
   doi:[10.1109/TNET.2020.3042320](https://doi.org/10.1109/TNET.2020.3042320)
4. H. Li, X. Li, Q. Fan, Q. He, X. Wang and V. C. M. Leung, "Distributed DNN Inference With
   Fine-Grained Model Partitioning in Mobile Edge Computing Networks," *IEEE Trans. Mobile Comput.*,
   vol. 23, no. 10, pp. 9060–9074, 2024.
   doi:[10.1109/TMC.2024.3357874](https://doi.org/10.1109/TMC.2024.3357874)
5. Y. Matsubara, M. Levorato and F. Restuccia, "Split Computing and Early Exiting for Deep Learning
   Applications: Survey and Research Challenges," *ACM Comput. Surv.*, vol. 55, no. 5, art. 90, 2022.
   doi:[10.1145/3527155](https://doi.org/10.1145/3527155)
6. Y. Kang, J. Hauswald, C. Gao, A. Rovinski, T. Mudge, J. Mars and L. Tang, "Neurosurgeon:
   Collaborative Intelligence Between the Cloud and Mobile Edge," in *Proc. ACM ASPLOS*, 2017.
   doi:[10.1145/3037697.3037698](https://doi.org/10.1145/3037697.3037698)
7. S. Teerapittayanon, B. McDanel and H. T. Kung, "BranchyNet: Fast Inference via Early Exiting
   from Deep Neural Networks," in *Proc. ICPR*, 2016.
   doi:[10.1109/ICPR.2016.7900006](https://doi.org/10.1109/ICPR.2016.7900006)

Models and dataset:

8. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen, "MobileNetV2: Inverted Residuals
   and Linear Bottlenecks," in *Proc. IEEE CVPR*, 2018.
   doi:[10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474)
9. K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," in
   *Proc. IEEE CVPR*, 2016. doi:[10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
10. J. Howard, *Imagenette*, fast.ai, 2019. <https://github.com/fastai/imagenette>
