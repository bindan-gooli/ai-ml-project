"""
Generate confusion matrices for all 5 datasets.
Reconstructs TP/TN/FP/FN from saved Precision, Recall, Accuracy, and known test-set class counts.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("roc_curves", exist_ok=True)
df = pd.read_csv("updated_model_results.csv")

# Known test-set sizes and positive counts from preprocessed data
TEST_INFO = {
    'D1': {'N': 1_272_524, 'P':     1_643, 'name': 'Dataset 1 (Credit Card Tx)'},
    'D2': {'N':     2_000, 'P':        30, 'name': 'Dataset 2 (Small Fraud)'},
    'D3': {'N':   113_726, 'P':    56_863, 'name': 'Dataset 3 (Balanced)'},
    'D4': {'N':    56_962, 'P':        98, 'name': 'Dataset 4 (Rare Fraud)'},
    'D5': {'N':   259_335, 'P':     1_501, 'name': 'Dataset 5 (Large-scale)'},
}

def reconstruct_cm(precision, recall, accuracy, N, P):
    """Reconstruct confusion matrix from aggregate metrics + class sizes."""
    precision = max(precision, 1e-9)
    recall    = max(recall,    1e-9)
    TP = round(P * recall)
    FN = P - TP
    FP = round(TP * (1 / precision - 1)) if precision < 1 else 0
    TN = N - TP - FP - FN
    TN = max(TN, 0)
    return np.array([[TN, FP], [FN, TP]])

prim_labels = {
    'M1': 'HistGradBoost', 'M2': 'ExtraTrees',
    'M3': 'GradBoost',     'M4': 'RandomForest', 'M5': 'MLP'
}
sec_labels = {
    'M6': 'Isolation Forest', 'M7': 'Autoencoder',
    'M8': 'VAE', 'M9': 'DAGMM', 'M10': 'Deep SVDD'
}

# ── Figure 1: Best-per-dataset summary (1×5 grid) ───────────────────────────
fig1, axes1 = plt.subplots(1, 5, figsize=(28, 6))
fig1.patch.set_facecolor('#0d1117')

for ax, (ds, info) in zip(axes1, TEST_INFO.items()):
    sub = df[df['Dataset'] == ds]
    best = sub.loc[sub['AUC'].idxmax()]
    cm = reconstruct_cm(best['Precision'], best['Recall'],
                        best['Accuracy'], info['N'], info['P'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm, annot=True, fmt=',d', ax=ax,
                cmap='Blues', linewidths=0.5, linecolor='#30363d',
                cbar=False, annot_kws={'size': 9, 'weight': 'bold'})
    ax.set_facecolor('#161b22')
    ax.set_title(
        f"{info['name']}\n"
        f"{best['Primary_Model']}+{best['Secondary_Model']}  AUC={best['AUC']:.4f}",
        fontsize=9, fontweight='bold', color='white', pad=8
    )
    ax.set_xlabel('Predicted Label', color='#aaa', fontsize=9)
    ax.set_ylabel('True Label',      color='#aaa', fontsize=9)
    ax.set_xticklabels(['Legitimate', 'Fraud'], color='white', fontsize=8)
    ax.set_yticklabels(['Legitimate', 'Fraud'], color='white', fontsize=8, rotation=0)
    ax.tick_params(colors='white')

fig1.suptitle('Confusion Matrices — Best Hybrid Combination per Dataset',
              fontsize=15, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
fig1.savefig('roc_curves/confusion_matrix_best.pdf',  format='pdf',  bbox_inches='tight')
fig1.savefig('roc_curves/confusion_matrix_best.tiff', format='tiff', dpi=400, bbox_inches='tight')
plt.close()
print("✅ Saved: confusion_matrix_best.pdf  +  .tiff")

# ── Figure 2: All 25 combos per dataset (one figure per dataset) ─────────────
for ds, info in TEST_INFO.items():
    sub = df[df['Dataset'] == ds].sort_values(['Secondary_Model', 'Primary_Model']).reset_index(drop=True)
    fig2, axes2 = plt.subplots(5, 5, figsize=(26, 22))
    fig2.patch.set_facecolor('#0d1117')

    sec_keys = ['M6','M7','M8','M9','M10']
    prim_keys = ['M1','M2','M3','M4','M5']

    for ri, sk in enumerate(sec_keys):
        for ci, pk in enumerate(prim_keys):
            ax = axes2[ri][ci]
            row = sub[(sub['Secondary_Model']==sk) & (sub['Primary_Model']==pk)]
            if row.empty:
                ax.axis('off'); continue
            row = row.iloc[0]
            cm = reconstruct_cm(row['Precision'], row['Recall'],
                                row['Accuracy'], info['N'], info['P'])

            sns.heatmap(cm, annot=True, fmt=',d', ax=ax,
                        cmap='RdYlGn', linewidths=0.5, linecolor='#1f2937',
                        cbar=False, annot_kws={'size': 8.5, 'weight':'bold'})
            ax.set_title(
                f"{pk}+{sk}  F1={row['F1_Score']:.3f}  AUC={row['AUC']:.3f}",
                fontsize=8, fontweight='bold', color='white', pad=4
            )
            ax.set_xlabel('Predicted', color='#aaa', fontsize=7)
            ax.set_ylabel('Actual',    color='#aaa', fontsize=7)
            ax.set_xticklabels(['Legit','Fraud'], color='white', fontsize=7)
            ax.set_yticklabels(['Legit','Fraud'], color='white', fontsize=7, rotation=0)
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='white')

    fig2.suptitle(
        f'Confusion Matrices — {info["name"]}\n'
        f'All 25 Hybrid Combinations (Rows=Secondary, Cols=Primary)',
        fontsize=14, fontweight='bold', color='white', y=1.005
    )
    # Row/column labels
    for ri, sk in enumerate(sec_keys):
        fig2.text(0.01, 0.84 - ri*0.17,
                  f'{sk}\n{sec_labels[sk]}',
                  va='center', ha='left', fontsize=9,
                  color='#f4a261', fontweight='bold')
    for ci, pk in enumerate(prim_keys):
        fig2.text(0.12 + ci*0.185, 0.99,
                  f'{pk}\n{prim_labels[pk]}',
                  va='top', ha='center', fontsize=9,
                  color='#a8dadc', fontweight='bold')

    plt.tight_layout(rect=[0.03, 0, 1, 0.98])
    pdf3  = f'roc_curves/dataset_{ds[-1]}_confusion_all.pdf'
    tiff3 = f'roc_curves/dataset_{ds[-1]}_confusion_all.tiff'
    fig2.savefig(pdf3,  format='pdf',  bbox_inches='tight')
    fig2.savefig(tiff3, format='tiff', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {pdf3}  +  {tiff3}")

print("\n🎉 All confusion matrices saved in roc_curves/ as PDF + TIFF!")
