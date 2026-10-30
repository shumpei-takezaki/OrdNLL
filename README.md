# Ord-NLL: Negative Label Learning for Ordinal Noisy Labels

[[Paper]](https://ieeexplore.ieee.org/document/11641564) — S. Takezaki, K. Shiku, S. Harada, K. Tanaka, and S. Uchida, *IEEE Access*, 2026.

Ord-NLL samples negative labels from `π_ord(ỹ) = (1 − l_ord(ỹ)) / (C − 1)`, so classes far from the observed
label are more likely to be negatives. Ord-NLL+ retrains on samples the Ord-NLL model finds reliable.
This repository reproduces the LIMUC experiments (Tables 2, 4, 6 and 7).

## Quick start

```bash
uv sync                                 # PyTorch for CUDA 12.6; edit the index URL in pyproject.toml for others
uv run python dataset/download_limuc.py # LIMUC from Zenodo (1.9 GB, CC BY 4.0)
bash scripts/run.sh ordnll quasi 0.4    # Ord-NLL, 5 folds -> expr/ordnll/limuc/quasi-0.4/results.txt
bash scripts/run.sh ordnll_plus quasi 0.4
```

`scripts/run.sh METHOD NOISE_TYPE NOISE_RATE` trains five folds and reports the mean ± std of the last-10-epoch test
metrics. To reproduce everything:

```bash
bash scripts/reproduce_main.sh      # Tables 2 and 4
bash scripts/reproduce_ablation.sh  # Tables 6 and 7
```

## Methods

| Paper | `METHOD` | | Paper | `METHOD` |
|---|---|---|---|---|
| CE loss | `ce` | | Co-teaching | `co_teaching` |
| Soft label | `soft_label` | | CoDis | `codis` |
| F-correction | `f_correction` | | Garg | `garg` |
| Reweight | `reweight` | | Self-relaxed | `self_relaxed` |
| Mixup | `mixup` | | NLL | `nll` |
| CDR | `cdr` | | **Ord-NLL** | `ordnll` (`--kappa`, Table 6) |
| | | | **Ord-NLL+** | `ordnll_plus` (`--tau`, Table 7) |

Hyperparameters are in `src/METHOD/config.yaml`.

## Data

The splits and noisy labels used in the paper are in `dataset/limuc/5fold/` (`label`: reference, `noisy_label`:
used for training and validation). Use these files to reproduce the paper.

| Directory | Noise | ρ |
|---|---|---|
| `quasi-0.2` / `quasi-0.4` | Quasi-Gaussian | 0.1 / 0.2 |
| `truncated-0.2` / `truncated-0.4` | Truncated | 0.15 / 0.3 |

## Notes

- Ord-NLL+ needs the Ord-NLL checkpoints in `expr/ordnll`. Its retraining target is the posterior of the Ord-NLL
  model, and samples are selected on each mini-batch.
- For Co-teaching, CoDis and Self-relaxed, the paper reports the `net1&net2` row of `results.txt`.
- `cudnn.benchmark` is enabled, so results vary slightly between runs.

## Citation

```bibtex
@article{takezaki2026ordnll,
  author  = {Takezaki, Shumpei and Shiku, Kaito and Harada, Shota and Tanaka, Kiyohito and Uchida, Seiichi},
  title   = {Ord-{NLL}: Negative Label Learning for Ordinal Noisy Labels},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {121763--121775},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3720528}
}
```

Code: [MIT License](LICENSE). LIMUC: CC BY 4.0.
