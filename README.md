# Ord-NLL: Negative Label Learning for Ordinal Noisy Labels

[[Paper]](https://ieeexplore.ieee.org/document/11641564) — S. Takezaki, K. Shiku, S. Harada, K. Tanaka, and S. Uchida, *IEEE Access*, 2026.

Ord-NLL samples negative labels from `π_ord(ỹ) = (1 − l_ord(ỹ)) / (C − 1)`, so classes far from the observed
label are more likely to be negatives. Ord-NLL+ retrains on samples the Ord-NLL model finds reliable.
This repository reproduces the LIMUC experiments (Tables 2, 4, 6 and 7).

## Quick start

```bash
uv sync
uv run python dataset/download_limuc.py
bash scripts/run.sh ordnll quasi 0.4       # Ord-NLL
bash scripts/run.sh ordnll_plus quasi 0.4  # Ord-NLL+ (uses the Ord-NLL checkpoints above)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU.

```bash
git clone https://github.com/shumpei-takezaki/OrdNLL.git
cd OrdNLL
uv sync
```

PyTorch is installed for CUDA 12.6 (NVIDIA driver >= 560). For another CUDA version, change the `pytorch-cu126`
index URL in `pyproject.toml`.

## Data

```bash
uv run python dataset/download_limuc.py
```

This downloads [LIMUC](https://zenodo.org/records/5827695) (1.9 GB, CC BY 4.0) to
`dataset/limuc/patient_based_classified_images/`.

The patient-disjoint 5-fold splits and the noisy labels used in the paper are included in `dataset/limuc/5fold/`.
Each CSV has the reference label (`label`) and the noisy label (`noisy_label`, used for training and validation).

| Directory | Noise | ρ |
|---|---|---|
| `quasi-0.2` / `quasi-0.4` | Quasi-Gaussian | 0.1 / 0.2 |
| `truncated-0.2` / `truncated-0.4` | Truncated | 0.15 / 0.3 |

They can be regenerated with `dataset/csv_5fold.py` and `dataset/noisy_csv_5fold.py`.

## Training and evaluation

```bash
bash scripts/run.sh METHOD NOISE_TYPE NOISE_RATE   # NOISE_TYPE: quasi | truncated, NOISE_RATE: 0.2 | 0.4
```

This trains five folds and writes the mean ± std of the last-10-epoch test metrics (accuracy, MAE, macro F1, …) to
`expr/METHOD/limuc/NOISE_TYPE-NOISE_RATE/results.txt`. Hyperparameters are in `src/METHOD/config.yaml`.

| Paper | `METHOD` | | Paper | `METHOD` |
|---|---|---|---|---|
| CE loss | `ce` | | Co-teaching | `co_teaching` |
| Soft label | `soft_label` | | CoDis | `codis` |
| F-correction | `f_correction` | | Garg | `garg` |
| Reweight | `reweight` | | Self-relaxed | `self_relaxed` |
| Mixup | `mixup` | | NLL | `nll` |
| CDR | `cdr` | | **Ord-NLL** | `ordnll` |
| | | | **Ord-NLL+** | `ordnll_plus` |

To reproduce all the tables:

```bash
bash scripts/reproduce_main.sh      # Tables 2 and 4: all methods and noise settings
bash scripts/reproduce_ablation.sh  # Table 6 (--kappa of ordnll) and Table 7 (--tau of ordnll_plus)
```

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
