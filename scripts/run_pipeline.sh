#!/usr/bin/env bash
# Data, target check, tests, one model, replay. The ablation is separate
# because it trains two neural networks and takes a while on a CPU.
set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-data}"
ARTIFACTS="${ARTIFACTS:-artifacts}"
MODEL="${MODEL:-lgbm}"

cd "$(dirname "$0")/.."

step() { printf '\n%s\n  %s\n%s\n' "==============================" "$1" "=============================="; }

step "1/5  Synthetic data"
[ -f "$DATA_DIR/train.csv" ] && echo "already present, skipping" \
    || "$PYTHON" scripts/generate_synthetic_data.py --output-dir "$DATA_DIR"

step "2/5  Target reimplementation check"
"$PYTHON" scripts/verify_target.py --data-dir "$DATA_DIR"

step "3/5  Unit tests"
"$PYTHON" -m pytest tests/ -q

step "4/5  Cross-validate and train ($MODEL)"
"$PYTHON" -m cryptoforecast.train --model "$MODEL" --data-dir "$DATA_DIR" \
    --output-dir "$ARTIFACTS"

step "5/5  Minute-by-minute replay"
"$PYTHON" scripts/run_replay.py --artifacts "$ARTIFACTS" --name "$MODEL" \
    --data-dir "$DATA_DIR" --minutes 600

step "Done - for the full model comparison, run: make ablation"
