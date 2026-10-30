#!/bin/bash
# Tables 2 and 4: all methods on LIMUC with Quasi-Gaussian / truncated noise (eps = 0.2, 0.4).
# Ord-NLL must run before Ord-NLL+ (Ord-NLL+ loads ./expr/ordnll checkpoints).
set -e
for SETTING in "quasi 0.2" "quasi 0.4" "truncated 0.2" "truncated 0.4"; do
    for METHOD in ce soft_label f_correction reweight mixup cdr co_teaching codis garg self_relaxed nll ordnll ordnll_plus; do
        bash scripts/run.sh $METHOD $SETTING
    done
done
