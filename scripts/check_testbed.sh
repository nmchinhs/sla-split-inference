#!/usr/bin/env bash
# PLAN.md Phase 1 acceptance checks. Run from the host with the testbed up:
#
#   docker compose up -d --build
#   scripts/check_testbed.sh [device_container]
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${1:-device-1}"
RATE_MBIT=5
DELAY_MS=20

echo "== GPU visible in edge"
docker exec edge nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
docker exec edge python -c \
    'import torch; assert torch.cuda.is_available(); print("edge   torch", torch.__version__, "cuda=True")'

echo "== no CUDA in $DEVICE"
docker exec "$DEVICE" python -c \
    'import torch; assert not torch.cuda.is_available(); print("device torch", torch.__version__, "cuda=False")'

echo "== unshaped link"
scripts/netem.sh "$DEVICE" clear
docker exec "$DEVICE" python scripts/smoke_test.py

echo "== netem ${RATE_MBIT} Mbit/s, ${DELAY_MS} ms"
trap 'scripts/netem.sh "$DEVICE" clear' EXIT
scripts/netem.sh "$DEVICE" "$RATE_MBIT" "$DELAY_MS"
docker exec "$DEVICE" python scripts/smoke_test.py \
    --expect-mbps "$RATE_MBIT" --expect-rtt-ms "$DELAY_MS"

echo "== all Phase 1 testbed checks passed"
