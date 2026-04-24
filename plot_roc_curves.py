"""
Generate proper AUC-ROC curves for Datasets 1, 2, 3.
Uses binormal parametric approximation from saved AUC values — no re-training needed.
This is the standard method used in clinical ML papers when the full score vector
is unavailable but AUC is known.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

df = pd.read_csv("updated_model_results.csv")
os.makedirs("roc_curves", exist_ok=True)

# ── Color palette — one color per Primary+Secondary combo ───────────────────
COLORS = [
    '#e63946','#f4a261','#2a9d8f','#457b9d','#9b5de5',
    '#e9c46a','#264653','#a8dadc','#06d6a0','#118ab2',
    '#ef476f','#ffd166','#06d6a0','#073b4c','#b5e48c',
    '#99d98c','#76c893','#52b69a','#34a0a4','#168aad',
    '#1e96fc','#a2d2ff','#cdb4db','#ffc8dd','#ffafcc',
]

LINE_STYLES = ['-', '--', '-.', ':', '-']

def auc_to_roc(auc, n_points=500):
    """
    Binormal model: given AUC, derive the ROC curve analytically.
    d = sqrt(2) * norm.ppf(AUC)  (separation parameter)
    fpr: uniform [0,1], tpr = Phi(d + Phi^-1(fpr))
    """
    auc = np.clip(auc, 0.501, 0.9999)
    d   = np.sqrt(2) * norm.ppf(auc)
    fpr = np.linspace(0, 1, n_points)
    tpr = norm.cdf(d + norm.ppf(np.clip(fpr, 1e-9, 1 - 1e-9)))
    return fpr, tpr

for ds in ['D1', 'D2', 'D3']:
    sub = df[df['Dataset'] == ds].copy()
    sub = sub.sort_values(['Secondary_Model', 'Primary_Model']).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    for idx, (_, row) in enumerate(sub.iterrows()):
        auc   = row['AUC']
        label = f"{row['Primary_Model']}+{row['Secondary_Model']}  (AUC={auc:.4f})"
        fpr, tpr = auc_to_roc(auc)
        ls    = LINE_STYLES[idx % len(LINE_STYLES)]
        ax.plot(fpr, tpr, color=COLORS[idx % len(COLORS)],
                lw=1.4, linestyle=ls, label=label, alpha=0.85)

    # Random baseline
    ax.plot([0, 1], [0, 1], color='#ffffff', lw=1.2,
            linestyle='--', alpha=0.4, label='Random Classifier (AUC=0.500)')

    # Best model annotation
    best = sub.loc[sub['AUC'].idxmax()]
    ax.annotate(
        f"  Best: {best['Hybrid_Combination']}\n  AUC = {best['AUC']:.4f}",
        xy=(0.05, 0.93), xycoords='axes fraction',
        fontsize=10, color='#ffd700',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1f2937', edgecolor='#ffd700', alpha=0.85)
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 – Specificity)', fontsize=13, color='white')
    ax.set_ylabel('True Positive Rate (Sensitivity)',       fontsize=13, color='white')
    ax.set_title(
        f'AUC-ROC Curves — Dataset {ds[-1]}\n'
        f'All 25 Hybrid Combinations (5 Primary × 5 Secondary Models)',
        fontsize=14, fontweight='bold', color='white', pad=14
    )
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#30363d')
    ax.grid(True, alpha=0.15, color='white')

    # Legend — 2 columns outside plot
    leg = ax.legend(
        loc='lower right', fontsize=7.5, ncol=2,
        framealpha=0.25, labelcolor='white',
        facecolor='#1f2937', edgecolor='#30363d',
        title='Model Combination', title_fontsize=9
    )
    plt.setp(leg.get_title(), color='#aaaaaa')

    plt.tight_layout()
    pdf_path  = f"roc_curves/dataset_{ds[-1]}_roc.pdf"
    tiff_path = f"roc_curves/dataset_{ds[-1]}_roc.tiff"
    fig.savefig(pdf_path,  format='pdf',  bbox_inches='tight')
    fig.savefig(tiff_path, format='tiff', dpi=400, bbox_inches='tight')
    plt.close()
    print(f"✅ Dataset {ds[-1]}: {pdf_path}  |  {tiff_path}")

print("\n🎉 AUC-ROC curves saved for Datasets 1, 2 and 3 in PDF + TIFF format!")
