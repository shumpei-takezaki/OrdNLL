#!/bin/bash
# LIMUC, Quasi-Gaussian, eps = 0.4. Requires ./expr/ordnll from scripts/reproduce_main.sh.
set -e

# Table 6: decay kappa of the ordinal soft target (kappa = 1.0 is the default Ord-NLL)
for KAPPA in 0.5 2.0; do
    WORKDIR=./expr/ordnll_kappa$KAPPA bash scripts/run.sh ordnll quasi 0.4 --kappa=$KAPPA
done

# Table 7: selection threshold tau of Ord-NLL+ (tau = 0.25 = 1/C is the default)
for TAU in 0.20 0.30; do
    WORKDIR=./expr/ordnll_plus_tau$TAU bash scripts/run.sh ordnll_plus quasi 0.4 --tau=$TAU
done
uv run python -m src.ordnll_plus.analyze_selection --noise_type=quasi --noise_rate=0.4
