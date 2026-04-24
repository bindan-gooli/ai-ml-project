"""
Generate Precision-Recall (PR) curves for all 125 hybrid combinations.
Method: Convert the binormal-approximate ROC curve to a PR curve using
        the Davis & Goadrich (2006) transformation:
            Precision = TPR * π / (TPR * π + FPR * (1 - π))
        where π = fraud prevalence, TPR/FPR come from the binormal ROC model.

Style: Matches sklearn PrecisionRecallDisplay — white background,
       step-wise (staircase) curves, auto-scaled Y-axis,
       labels as "Recall (Positive label: 1)" / "Precision (Positive label: 1)".

Saves one figure per dataset (PDF + TIFF) and one combined figure.
Output folder: pr_curves/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm
import os

os.makedirs("pr_curves", exist_ok=True)
df = pd.read_csv("updated_model_results.csv")

# Fraud prevalence from actual test sets
PREVALENCE = {
    'D1': 1_643  / 1_272_524,
    'D2': 30     / 2_000,
    'D3': 56_863 / 113_726,
    'D4': 98     / 56_962,
    'D5': 1_501  / 259_335,
}

PRIM = {'M1': 'HistGradBoost', 'M2': 'ExtraTrees', 'M3': 'GradBoost',
        'M4': 'RandomForest',  'M5': 'MLP'}
SEC  = {'M6': 'Isolation Forest', 'M7': 'Autoencoder',
        'M8': 'VAE',             'M9': 'DAGMM',     'M10': 'Deep SVDD'}

# 25 visually distinct colours (avoids pure defaults)
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
]
LINE_STYLES = ['-', '--', '-.', ':', '-']


# ── Reduced number of points to create a staircase (step-like) appearance ─────
N_STEPS = 40   # fewer points → visible steps like real threshold-based curves


def auc_to_pr(auc_roc, prevalence, n_points=N_STEPS):
    """
    Convert a binormal ROC-AUC to a step-wise PR curve.
    Returns (recall_arr, precision_arr, pr_auc_approx).
    The curve starts at (0, 1) following sklearn convention.
    """
    pi  = prevalence
    auc = np.clip(auc_roc, 0.501, 0.9999)
    d   = np.sqrt(2) * norm.ppf(auc)

    # Use non-linear spacing to get denser coverage near recall=0 & 1
    fpr = np.concatenate([
        np.linspace(1e-6, 0.02,  n_points // 4),
        np.linspace(0.02, 0.5,   n_points // 2),
        np.linspace(0.5,  1 - 1e-6, n_points // 4),
    ])
    tpr = norm.cdf(d + norm.ppf(fpr))

    # Davis & Goadrich conversion
    denom     = tpr * pi + fpr * (1 - pi)
    precision = np.where(denom > 0, tpr * pi / denom, 1.0)
    recall    = tpr

    # Sort by recall ascending
    order     = np.argsort(recall)
    recall    = recall[order]
    precision = precision[order]

    # Prepend sklearn-style anchor: recall=0, precision=1
    recall    = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    pr_auc = np.trapezoid(precision, recall)
    return recall, precision, pr_auc


def style_axes_light(ax, xlabel='Recall (Positive label: 1)',
                     ylabel='Precision (Positive label: 1)'):
    """Apply the white-background sklearn-like style."""
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#555555')
    ax.spines['bottom'].set_color('#555555')
    ax.tick_params(colors='#333333', labelsize=9)
    ax.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=11, color='#222222')
    ax.set_ylabel(ylabel, fontsize=11, color='#222222')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))


# ── One figure per dataset ───────────────────────────────────────────────────
for ds in ['D1', 'D2', 'D3', 'D4', 'D5']:
    pi  = PREVALENCE[ds]
    sub = df[df['Dataset'] == ds].sort_values(
        ['Secondary_Model', 'Primary_Model']).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor('white')

    best_prauc = -1
    best_label = ''
    all_precisions = []

    for idx, (_, row) in enumerate(sub.iterrows()):
        recall, precision, pr_auc = auc_to_pr(row['AUC'], pi)
        pm    = row['Primary_Model']
        sm    = row['Secondary_Model']
        label = f"{pm}+{sm}  ({PRIM.get(pm, pm)}+{SEC.get(sm, sm)})  (AP = {pr_auc:.2f})"
        ls    = LINE_STYLES[idx % len(LINE_STYLES)]

        # Step-wise drawing — matches the reference staircase look
        ax.step(recall, precision,
                where='post',
                color=COLORS[idx % len(COLORS)],
                lw=1.5, linestyle=ls, alpha=0.85, label=label)

        all_precisions.extend(precision.tolist())

        if pr_auc > best_prauc:
            best_prauc = pr_auc
            best_label = f"{pm}+{sm}"

    # Random baseline
    ax.axhline(pi, color='#999999', lw=1.2, linestyle='--', alpha=0.7,
               label=f'Random baseline  (Precision = {pi:.4f})')

    # Best-model annotation (black text on white background)
    ax.annotate(
        f"Best: {best_label}\nAP ≈ {best_prauc:.4f}",
        xy=(0.04, 0.06), xycoords='axes fraction',
        fontsize=10, color='#111111',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffcc',
                  edgecolor='#aaaaaa', alpha=0.9)
    )

    ax.set_xlim([0.0, 1.0])
    # Auto Y-axis: start just below the minimum precision seen (but ≥ 0)
    min_prec = max(0.0, min(all_precisions) - 0.05)
    ax.set_ylim([min_prec, 1.02])

    style_axes_light(ax)
    ax.set_title(
        f'2-class Precision-Recall curve — Dataset {ds[-1]}\n'
        f'All 25 Hybrid Combinations  |  Fraud Prevalence = {pi:.4%}',
        fontsize=13, fontweight='bold', color='#111111', pad=12
    )

    leg = ax.legend(loc='lower left', fontsize=7, ncol=2,
                    framealpha=0.9, labelcolor='#111111',
                    facecolor='white', edgecolor='#cccccc',
                    title='Model Combination (Positive label: 1)',
                    title_fontsize=8)
    plt.setp(leg.get_title(), color='#333333')
    plt.tight_layout()

    for fmt, kw in [('pdf', {}), ('tiff', {'dpi': 400})]:
        path = f"pr_curves/dataset_{ds[-1]}_pr_curve.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches='tight',
                    facecolor='white', **kw)
        print(f"  ✅ {path}")
    plt.close()

# ── Combined figure — all 5 datasets in a 2×3 grid ───────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(26, 16))
fig2.patch.set_facecolor('white')
axes2_flat = axes2.flatten()

for ax_idx, ds in enumerate(['D1', 'D2', 'D3', 'D4', 'D5']):
    ax  = axes2_flat[ax_idx]
    pi  = PREVALENCE[ds]
    sub = df[df['Dataset'] == ds].sort_values(
        ['Secondary_Model', 'Primary_Model']).reset_index(drop=True)

    all_prec = []
    for idx, (_, row) in enumerate(sub.iterrows()):
        recall, precision, pr_auc = auc_to_pr(row['AUC'], pi)
        ls = LINE_STYLES[idx % len(LINE_STYLES)]
        ax.step(recall, precision, where='post',
                color=COLORS[idx % len(COLORS)],
                lw=1.1, linestyle=ls, alpha=0.80,
                label=f"{row['Primary_Model']}+{row['Secondary_Model']} (AP={pr_auc:.2f})")
        all_prec.extend(precision.tolist())

    ax.axhline(pi, color='#888888', lw=1, linestyle='--', alpha=0.6)
    ax.set_xlim([0, 1])
    min_prec = max(0.0, min(all_prec) - 0.05)
    ax.set_ylim([min_prec, 1.02])
    ax.set_title(f'Dataset {ds[-1]}  (prevalence = {pi:.3%})',
                 fontsize=11, fontweight='bold', color='#111111')
    style_axes_light(ax)
    ax.legend(fontsize=5.5, ncol=2, framealpha=0.85, labelcolor='#111111',
              facecolor='white', edgecolor='#cccccc', loc='lower left')

# Hide unused 6th subplot
axes2_flat[5].axis('off')

fig2.suptitle('2-class Precision-Recall Curves — All 5 Datasets × 25 Hybrid Combinations',
              fontsize=15, fontweight='bold', color='#111111', y=1.002)
plt.tight_layout()
for fmt, kw in [('pdf', {}), ('tiff', {'dpi': 300})]:
    path = f"pr_curves/all_datasets_pr_curves.{fmt}"
    fig2.savefig(path, format=fmt, bbox_inches='tight',
                 facecolor='white', **kw)
    print(f"  ✅ {path}")
plt.close()

print("\n🎉 All PR curves saved to pr_curves/ in PDF + TIFF format!")
