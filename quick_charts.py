"""
Generate AUC performance charts for Datasets 1, 2, 3 — saved as PDF and TIFF.
Reads directly from updated_model_results.csv (no re-training needed).
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np, os

df = pd.read_csv("updated_model_results.csv")
os.makedirs("roc_curves", exist_ok=True)

sec_colors = {
    'M6': '#e63946',  # Isolation Forest — red
    'M7': '#2a9d8f',  # DAE — teal
    'M8': '#457b9d',  # VAE — blue
    'M9': '#e9c46a',  # DAGMM — yellow
    'M10':'#9b5de5',  # Deep SVDD — purple
}
sec_labels = {
    'M6':'Isolation Forest','M7':'Deep Autoencoder',
    'M8':'VAE','M9':'DAGMM','M10':'Deep SVDD'
}
prim_labels = {
    'M1':'HistGradBoost','M2':'ExtraTrees',
    'M3':'GradBoost','M4':'RandomForest','M5':'MLP'
}

# ── Figure 1: AUC bar chart (3 datasets side-by-side) ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=False)
fig.patch.set_facecolor('#1a1a2e')

for ax, ds in zip(axes, ['D1', 'D2', 'D3']):
    sub = df[df['Dataset'] == ds].copy()
    sub['combo'] = sub['Primary_Model'] + '+' + sub['Secondary_Model']
    sub = sub.sort_values(['Secondary_Model', 'Primary_Model']).reset_index(drop=True)

    x      = np.arange(len(sub))
    colors = [sec_colors.get(r, '#aaa') for r in sub['Secondary_Model']]

    bars = ax.bar(x, sub['AUC'], color=colors, edgecolor='#ffffff22',
                  linewidth=0.4, alpha=0.9, zorder=3, width=0.7)

    for bar, val in zip(bars, sub['AUC']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=5.5, rotation=90, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{prim_labels.get(r['Primary_Model'], r['Primary_Model'])}\n+{sec_labels.get(r['Secondary_Model'], r['Secondary_Model'])[:4]}"
         for _, r in sub.iterrows()],
        rotation=90, fontsize=5.5, color='#cccccc'
    )
    top_auc = sub['AUC'].max()
    top_combo = sub.loc[sub['AUC'].idxmax(), 'combo']
    ax.set_ylim([0.3, 1.10])
    ax.set_title(
        f'Dataset {ds[-1]}\nBest: {top_combo}  AUC={top_auc:.4f}',
        fontsize=11, fontweight='bold', color='white', pad=10
    )
    ax.set_ylabel('AUC Score', color='white', fontsize=10)
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.grid(axis='y', alpha=0.2, zorder=0, color='white')
    ax.axhline(0.90, color='#00ff99', ls='--', lw=1, alpha=0.7, label='AUC=0.90')
    ax.axhline(0.95, color='#00cfff', ls='--', lw=1, alpha=0.7, label='AUC=0.95')
    ax.axhline(1.00, color='#ffdd00', ls=':', lw=1, alpha=0.5, label='AUC=1.00')

# Shared legend for secondary models
handles = [mpatches.Patch(color=c, label=f"{k} – {sec_labels[k]}")
           for k, c in sec_colors.items()]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=9, framealpha=0.15, labelcolor='white',
           facecolor='#1a1a2e', edgecolor='#555',
           title='Secondary Model (Feature Generator)', title_fontsize=10)

fig.suptitle('Hybrid Model AUC Performance — Datasets 1, 2 & 3',
             fontsize=16, fontweight='bold', color='white', y=1.01)
plt.tight_layout(rect=[0, 0.08, 1, 1])

out_pdf  = "roc_curves/datasets_1_2_3_auc.pdf"
out_tiff = "roc_curves/datasets_1_2_3_auc.tiff"
fig.savefig(out_pdf,  format='pdf',  bbox_inches='tight')
fig.savefig(out_tiff, format='tiff', dpi=400, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {out_pdf}")
print(f"✅ Saved: {out_tiff}")

# ── Figure 2: Per-dataset detailed breakdown (one figure per dataset) ─────────
for ds in ['D1', 'D2', 'D3']:
    sub = df[df['Dataset'] == ds].copy()
    sub = sub.sort_values(['Secondary_Model', 'Primary_Model']).reset_index(drop=True)

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
    fig2.patch.set_facecolor('#1a1a2e')

    # Left: AUC bar
    ax1 = axes2[0]
    x = np.arange(len(sub))
    colors = [sec_colors.get(r, '#aaa') for r in sub['Secondary_Model']]
    ax1.bar(x, sub['AUC'], color=colors, edgecolor='#ffffff22', alpha=0.9, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{r['Primary_Model']}+{r['Secondary_Model']}" for _, r in sub.iterrows()],
        rotation=90, fontsize=8, color='#cccccc'
    )
    ax1.set_ylim([0.3, 1.08])
    ax1.set_title('AUC Score', color='white', fontsize=12, fontweight='bold')
    ax1.set_facecolor('#16213e'); ax1.tick_params(colors='white')
    ax1.spines[:].set_color('#444')
    ax1.grid(axis='y', alpha=0.2, color='white')
    ax1.axhline(0.90, color='#00ff99', ls='--', lw=1, alpha=0.7)
    ax1.axhline(0.95, color='#00cfff', ls='--', lw=1, alpha=0.7)

    # Right: F1 Score bar
    ax2 = axes2[1]
    ax2.bar(x, sub['F1_Score'], color=colors, edgecolor='#ffffff22', alpha=0.9, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{r['Primary_Model']}+{r['Secondary_Model']}" for _, r in sub.iterrows()],
        rotation=90, fontsize=8, color='#cccccc'
    )
    ax2.set_ylim([0.0, 1.08])
    ax2.set_title('F1 Score', color='white', fontsize=12, fontweight='bold')
    ax2.set_facecolor('#16213e'); ax2.tick_params(colors='white')
    ax2.spines[:].set_color('#444')
    ax2.grid(axis='y', alpha=0.2, color='white')

    handles = [mpatches.Patch(color=c, label=f"{k} – {sec_labels[k]}")
               for k, c in sec_colors.items()]
    fig2.legend(handles=handles, loc='lower center', ncol=5,
                fontsize=9, framealpha=0.15, labelcolor='white',
                facecolor='#1a1a2e', edgecolor='#555')
    fig2.suptitle(f'Dataset {ds[-1]} — AUC & F1 Breakdown (All 25 Combinations)',
                  fontsize=14, fontweight='bold', color='white', y=1.01)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    pdf2  = f"roc_curves/dataset_{ds[-1]}_detail.pdf"
    tiff2 = f"roc_curves/dataset_{ds[-1]}_detail.tiff"
    fig2.savefig(pdf2,  format='pdf',  bbox_inches='tight')
    fig2.savefig(tiff2, format='tiff', dpi=400, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {pdf2}  +  {tiff2}")

print("\n🎉 All charts saved to roc_curves/ in PDF and TIFF format!")
