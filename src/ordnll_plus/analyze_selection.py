"""
Standalone analysis: sample selection behavior of Ord-NLL+ (src/ordnll_plus/train.py)

Computes, for each threshold tau in {0.20, 0.25, 0.30}:
  - selected_ratio
  - class-wise selected_ratio (per observed noisy class)
  - clean_precision
  - clean_recall
  - noisy_remaining_ratio

Uses pre-trained Ord-NLL (src/ordnll) checkpoints. No retraining.
Run from workspace root:
  uv run python -m src.ordnll_plus.analyze_selection
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.cls_utils import NoisyDataset
from utils.model_utils import build_model, load_weight


# NOTE: existing Ord-NLL+ uses '>' (not '>=')
# ref: src/ordnll_plus/train.py
#   ind_select = pseudo_probs_positive > threshold
COMPARE_OP = 'gt'


def run_inference(model, csv_file, batch_size, device):
    """Forward pass on training set. Returns (all_probs, all_noisy, all_clean)."""
    ds = NoisyDataset(csv_file=csv_file, noisy=True, train=False)
    dl = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    all_probs = []
    all_noisy = []
    all_clean = []

    model.eval()
    with torch.no_grad():
        for images, noisy_labels, clean_labels, _ in dl:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu()
            all_probs.append(probs)
            all_noisy.append(noisy_labels)
            all_clean.append(clean_labels)

    all_probs = torch.cat(all_probs, dim=0).numpy()   # (N, C)
    all_noisy = torch.cat(all_noisy, dim=0).numpy()   # (N,)
    all_clean = torch.cat(all_clean, dim=0).numpy()   # (N,)

    return all_probs, all_noisy, all_clean


def compute_metrics(all_probs, all_noisy, all_clean, thresholds, num_classes):
    """Compute selection metrics for each threshold."""
    N = len(all_noisy)
    # confidence for the observed noisy label
    conf_noisy = all_probs[np.arange(N), all_noisy]   # (N,)
    is_clean = (all_noisy == all_clean)                # (N,) bool
    total_clean = is_clean.sum()
    total_noisy = (~is_clean).sum()

    rows = []
    for tau in thresholds:
        # S_tau: use '>' to match existing Ord-NLL+ implementation
        selected = conf_noisy > tau
        n_selected = selected.sum()

        selected_ratio = n_selected / N

        class_selected_ratios = []
        for c in range(num_classes):
            mask_c = (all_noisy == c)
            n_c = mask_c.sum()
            if n_c == 0:
                class_selected_ratios.append(float('nan'))
            else:
                class_selected_ratios.append((selected & mask_c).sum() / n_c)

        if n_selected == 0:
            clean_precision = float('nan')
        else:
            clean_precision = is_clean[selected].sum() / n_selected

        clean_recall = is_clean[selected].sum() / total_clean if total_clean > 0 else float('nan')

        noisy_remaining = (~is_clean)[selected].sum() / total_noisy if total_noisy > 0 else float('nan')

        row = {
            'threshold': tau,
            'selected_ratio': float(selected_ratio),
            'selected_ratio_class0': float(class_selected_ratios[0]),
            'selected_ratio_class1': float(class_selected_ratios[1]),
            'selected_ratio_class2': float(class_selected_ratios[2]),
            'selected_ratio_class3': float(class_selected_ratios[3]),
            'clean_precision': float(clean_precision),
            'clean_recall': float(clean_recall),
            'noisy_remaining_ratio': float(noisy_remaining),
        }
        rows.append(row)

    return rows


COLUMNS = [
    'threshold',
    'selected_ratio',
    'selected_ratio_class0',
    'selected_ratio_class1',
    'selected_ratio_class2',
    'selected_ratio_class3',
    'clean_precision',
    'clean_recall',
    'noisy_remaining_ratio',
]


def aggregate_fold_results(all_fold_rows, thresholds):
    """Compute mean +- std across folds for each threshold."""
    agg = []
    for tau in thresholds:
        tau_rows = [r for fold_rows in all_fold_rows for r in fold_rows if r['threshold'] == tau]
        row_agg = {'threshold': tau}
        for col in COLUMNS[1:]:
            vals = [r[col] for r in tau_rows if not np.isnan(r[col])]
            mean = np.mean(vals)
            std = np.std(vals)
            row_agg[col] = (mean, std)
        agg.append(row_agg)
    return agg


def save_per_fold_csv(all_fold_rows, folds, outpath):
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold'] + COLUMNS)
        for fold, fold_rows in zip(folds, all_fold_rows):
            for row in fold_rows:
                writer.writerow([fold] + [f'{row[c]:.6f}' for c in COLUMNS])


def save_aggregated_csv(agg_rows, outpath):
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['threshold'] + [f'{c}_mean' for c in COLUMNS[1:]] + [f'{c}_std' for c in COLUMNS[1:]]
        writer.writerow(header)
        for row in agg_rows:
            means = [f'{row[c][0]:.6f}' for c in COLUMNS[1:]]
            stds  = [f'{row[c][1]:.6f}' for c in COLUMNS[1:]]
            writer.writerow([row['threshold']] + means + stds)


def save_markdown_table(agg_rows, outpath):
    col_labels = {
        'selected_ratio':        'sel. ratio',
        'selected_ratio_class0': 'sel. cls0',
        'selected_ratio_class1': 'sel. cls1',
        'selected_ratio_class2': 'sel. cls2',
        'selected_ratio_class3': 'sel. cls3',
        'clean_precision':       'clean prec.',
        'clean_recall':          'clean rec.',
        'noisy_remaining_ratio': 'noisy rem.',
    }
    header = ['tau'] + list(col_labels.values())
    sep    = ['---'] * len(header)

    lines = []
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join(sep) + ' |')
    for row in agg_rows:
        cells = [str(row['threshold'])]
        for col in COLUMNS[1:]:
            mean, std = row[col]
            cells.append(f'{mean:.3f}±{std:.3f}')
        lines.append('| ' + ' | '.join(cells) + ' |')

    with open(outpath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # also print to stdout
    print('\n'.join(lines))


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    thresholds = args.thresholds
    folds = list(range(1, args.num_folds + 1))
    num_classes = args.num_classes

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_fold_rows = []

    for fold in folds:
        csv_path = (Path('./dataset/limuc/5fold')
                    / f'{args.noise_type}-{args.noise_rate}'
                    / f'fold{fold}'
                    / f'train_fold{fold}.csv')
        ckpt_path = (Path(args.weightdir)
                     / 'limuc'
                     / f'{args.noise_type}-{args.noise_rate}'
                     / f'fold{fold}'
                     / 'ckpt'
                     / 'net_last.ckpt')

        print(f'\n[Fold {fold}] CSV : {csv_path}')
        print(f'[Fold {fold}] Ckpt: {ckpt_path}')

        if not csv_path.exists():
            raise FileNotFoundError(f'CSV not found: {csv_path}')
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

        model = build_model(name='resnet18', pretrained=False, num_classes=num_classes)
        model = load_weight(model, ckpt_path, device)
        model = model.to(device)

        all_probs, all_noisy, all_clean = run_inference(
            model, csv_path, args.batch_size, device
        )

        N = len(all_noisy)
        is_clean = (all_noisy == all_clean)
        print(f'[Fold {fold}] N={N}, clean={is_clean.sum()} ({is_clean.mean():.3f}), '
              f'noisy={N - is_clean.sum()} ({1-is_clean.mean():.3f})')

        fold_rows = compute_metrics(all_probs, all_noisy, all_clean, thresholds, num_classes)
        all_fold_rows.append(fold_rows)

        for row in fold_rows:
            print(f'  tau={row["threshold"]:.2f} | '
                  f'sel={row["selected_ratio"]:.3f} | '
                  f'clean_prec={row["clean_precision"]:.3f} | '
                  f'clean_rec={row["clean_recall"]:.3f} | '
                  f'noisy_rem={row["noisy_remaining_ratio"]:.3f}')

    # Save per-fold CSV
    per_fold_csv = outdir / 'selection_behavior_per_fold.csv'
    save_per_fold_csv(all_fold_rows, folds, per_fold_csv)
    print(f'\nPer-fold results saved: {per_fold_csv}')

    # Aggregate
    agg_rows = aggregate_fold_results(all_fold_rows, thresholds)

    agg_csv = outdir / 'selection_behavior_aggregated.csv'
    save_aggregated_csv(agg_rows, agg_csv)
    print(f'Aggregated CSV saved: {agg_csv}')

    md_path = outdir / 'selection_behavior_table.md'
    save_markdown_table(agg_rows, md_path)
    print(f'Markdown table saved: {md_path}')

    # Print mean±std summary
    print('\n=== mean ± std across folds ===')
    for row in agg_rows:
        print(f'tau={row["threshold"]:.2f}:')
        for col in COLUMNS[1:]:
            mean, std = row[col]
            print(f'  {col}: {mean:.3f}±{std:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze Ord-NLL+ sample selection behavior (no retraining)')
    parser.add_argument('--weightdir', type=str, default='./expr/ordnll',
                        help='root dir of Ord-NLL checkpoints')
    parser.add_argument('--outdir', type=str, default='./expr/ordnll_plus_selection',
                        help='output directory for CSV and Markdown table')
    parser.add_argument('--noise_type', type=str, default='quasi',
                        choices=['quasi', 'truncated'])
    parser.add_argument('--noise_rate', type=float, default=0.4,
                        choices=[0.2, 0.4])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.20, 0.25, 0.30])
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args)
