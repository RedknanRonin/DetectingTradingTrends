#!/usr/bin/env bash
# Full experiment pipeline (run from the repository root once the dataset is downloaded).
#   DATA=/c/data/debs2022 bash scripts/run_experiments.sh [step ...]
# Steps: profile validate week bench plots   (default: all, in that order)
set -euo pipefail
DATA=${DATA:-/c/data/debs2022}
PY=${PY:-.venv/Scripts/python}
D="$DATA/debs2022-gc-trading-day"
WEEK=("$D-08-11-21.csv" "$D-09-11-21.csv" "$D-10-11-21.csv" "$D-11-11-21.csv" "$D-12-11-21.csv" "$D-13-11-21.csv" "$D-14-11-21.csv")
MON="$D-08-11-21.csv"
TUE="$D-09-11-21.csv"
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(profile validate week bench plots)
log() { echo "[$(date +%H:%M:%S)] $*"; }

for step in "${STEPS[@]}"; do case $step in
profile)
  log "profiling full week"
  $PY scripts/profile_data.py "${WEEK[@]}" --out results/profile
  log "profiling Monday (weights for balanced partitioning of later days)"
  $PY scripts/profile_data.py "$MON" --out results/profile_mon
  $PY scripts/make_subscriptions.py --counts results/profile/price_events_per_symbol.csv
  ;;
validate)
  # correctness on the full week, all symbols; K=0 (literal spec) and K=38 (default)
  for K in 0 38; do
    log "validation K=$K: streaming (emit all)"
    $PY -m tickstream "${WEEK[@]}" --cores 8 --warmup $K --emit-all --out results/runs/validate_k$K
    log "validation K=$K: reference"
    $PY scripts/reference.py "${WEEK[@]}" --warmup $K --out results/validation/reference_k$K.parquet
    $PY scripts/validate.py results/runs/validate_k$K/windows.parquet results/validation/reference_k$K.parquet \
      | tee results/validation/validate_k$K.json
  done
  ;;
week)
  log "full week, subscribed symbols, default configuration (8 cores)"
  $PY -m tickstream "${WEEK[@]}" --cores 8 --out results/runs/week
  ;;
bench)
  # scaling experiments on one weekday (Tuesday, fits in the OS page cache)
  W="--weights results/profile_mon/price_events_per_symbol.csv"
  $PY scripts/bench.py "$TUE" --exp cores preload --cores-list 1 2 4 6 8 14 --reps 2 --warmup-run --tag tue_scaling
  $PY scripts/bench.py "$TUE" --exp partition --cores-list 2 4 8 14 --reps 2 --tag tue_partition --extra $W
  $PY scripts/bench.py "$TUE" --exp lateness block --fixed-cores 4 --reps 2 --tag tue_knobs
  ;;
plots)
  $PY scripts/plots.py --bench results/bench/tue_*/summary.csv --profile results/profile \
    --run results/runs/week --symbol $(grep -v '^#' config/subscriptions.txt | head -3)
  ;;
*) echo "unknown step $step"; exit 1 ;;
esac; done
log "done"
