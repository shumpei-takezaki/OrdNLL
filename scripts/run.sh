#!/bin/bash
# Train a method on 5 folds and report the mean/std of the last-10-epoch averages.
# Usage: bash scripts/run.sh METHOD NOISE_TYPE NOISE_RATE [extra train args...]
#   e.g. bash scripts/run.sh ordnll quasi 0.4
#        WORKDIR=./expr/ordnll_kappa0.5 bash scripts/run.sh ordnll quasi 0.4 --kappa=0.5
set -e
METHOD=$1; NOISE_TYPE=$2; NOISE_RATE=$3; shift 3
WORKDIR=${WORKDIR:-./expr/$METHOD}
ARGS="--workdir=$WORKDIR --config=./src/$METHOD/config.yaml --noise_type=$NOISE_TYPE --noise_rate=$NOISE_RATE"

for FOLD in 1 2 3 4 5; do
    uv run python -m src.$METHOD.train $ARGS --fold=$FOLD "$@"
done

case $METHOD in
    co_teaching|codis|self_relaxed) TEST=test_two_nets ;;
    *) TEST=test ;;
esac
uv run python -m src.$TEST $ARGS
