"""
compute_advanced_metrics.py
===========================
Loads per-sample predictions saved by save_predictions.py and computes:

  Per-model (probability-based):
    • KS Statistic      (Kolmogorov-Smirnov between pos/neg score distributions)
    • Brier Score       (mean squared error of predicted probabilities)
    • ECE               (Expected Calibration Error, 10 bins)

  Cross-model (statistical tests):
    • McNemar Test      (pairwise; best model vs. each other model, per dataset)
    • Friedman Test     (overall ranking across all 5 datasets)

Outputs:
    advanced_metrics.xlsx   – individual + statistical test sheets

Run:
    python compute_advanced_metrics.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings('ignore')

PREDICTIONS_DIR = "predictions"
OUTPUT_EXCEL    = "advanced_metrics.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def ks_statistic(y_true, y_prob):
    """KS statistic: max separation between CDF of positives vs negatives."""
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ks, pval = stats.ks_2samp(pos, neg)
    return ks


def brier_score(y_true, y_prob):
    """Brier Score = mean((p - y)^2). Lower is better."""
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    ECE: weighted average of |accuracy − confidence| across probability bins.
    Lower is better (0 = perfect calibration).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = np.mean(y_true[mask] == (y_prob[mask] >= 0.5).astype(int))
        conf = np.mean(y_prob[mask])
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Load all saved predictions
# ─────────────────────────────────────────────────────────────────────────────

def load_all_predictions():
    """
    Returns a dict:
        key   : (dataset_str, primary, secondary)   e.g. ('D1', 'M1', 'M6')
        value : {'y_true': ..., 'y_pred': ..., 'y_prob': ...}
    """
    pattern = os.path.join(PREDICTIONS_DIR, "D*_M*_M*.npz")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No prediction files found in '{PREDICTIONS_DIR}/'. "
            "Please run save_predictions.py first."
        )

    preds = {}
    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]  # e.g. D1_M1_M6
        parts = base.split('_')                               # ['D1', 'M1', 'M6']
        dataset, primary, secondary = parts[0], parts[1], parts[2]
        data = np.load(fpath)
        preds[(dataset, primary, secondary)] = {
            'y_true': data['y_true'],
            'y_pred': data['y_pred'],
            'y_prob': data['y_prob'],
        }

    print(f"Loaded {len(preds)} prediction files.")
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Per-model metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_model_metrics(preds):
    rows = []
    for (dataset, primary, secondary), d in sorted(preds.items()):
        y_true = d['y_true']
        y_pred = d['y_pred']
        y_prob = d['y_prob']

        ks  = ks_statistic(y_true, y_prob)
        bs  = brier_score(y_true, y_prob)
        ece = expected_calibration_error(y_true, y_prob)

        rows.append({
            'Dataset':            dataset,
            'Primary_Model':      primary,
            'Secondary_Model':    secondary,
            'Hybrid_Combination': f'{primary} + {secondary}',
            'KS_Statistic':       round(ks,  4) if not np.isnan(ks)  else np.nan,
            'Brier_Score':        round(bs,  4),
            'ECE':                round(ece, 4),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# McNemar Test (pairwise, best vs. rest, per dataset)
# ─────────────────────────────────────────────────────────────────────────────

def run_mcnemar_tests(preds):
    """
    For each dataset, find the hybrid combination with the best F1 Score,
    then run McNemar's test comparing it against every other combination.
    """
    from sklearn.metrics import f1_score as sk_f1

    # Group by dataset
    by_dataset = {}
    for (dataset, primary, secondary), d in preds.items():
        by_dataset.setdefault(dataset, []).append((primary, secondary, d))

    rows = []
    for dataset in sorted(by_dataset):
        combos = by_dataset[dataset]

        # Find best by F1
        best_f1   = -1
        best_key  = None
        for (primary, secondary, d) in combos:
            f1 = sk_f1(d['y_true'], d['y_pred'], zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_key = (primary, secondary, d)

        best_primary, best_secondary, best_d = best_key
        y_true_ref = best_d['y_true']
        y_pred_ref = best_d['y_pred']

        for (primary, secondary, d) in combos:
            if primary == best_primary and secondary == best_secondary:
                continue  # skip self

            y_pred_alt = d['y_pred']
            # Contingency table
            b00 = np.sum((y_pred_ref == y_true_ref) & (y_pred_alt == y_true_ref))
            b01 = np.sum((y_pred_ref == y_true_ref) & (y_pred_alt != y_true_ref))
            b10 = np.sum((y_pred_ref != y_true_ref) & (y_pred_alt == y_true_ref))
            b11 = np.sum((y_pred_ref != y_true_ref) & (y_pred_alt != y_true_ref))

            table = [[b00, b01], [b10, b11]]
            try:
                result   = mcnemar(table, exact=False, correction=True)
                statistic = result.statistic
                pvalue    = result.pvalue
                significant = pvalue < 0.05
            except Exception:
                statistic, pvalue, significant = np.nan, np.nan, None

            rows.append({
                'Dataset':           dataset,
                'Reference_Model':   f'{best_primary}+{best_secondary} (best F1={best_f1:.4f})',
                'Compared_Model':    f'{primary}+{secondary}',
                'Contingency_b00':   int(b00),
                'Contingency_b01':   int(b01),
                'Contingency_b10':   int(b10),
                'Contingency_b11':   int(b11),
                'McNemar_Statistic': round(float(statistic), 4) if not np.isnan(statistic) else np.nan,
                'p_value':           round(float(pvalue), 4)    if not np.isnan(pvalue)    else np.nan,
                'Significant_0.05':  significant,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Friedman Test (ranking across all 5 datasets)
# ─────────────────────────────────────────────────────────────────────────────

def run_friedman_test(preds):
    """
    Friedman test over all 25 hybrid combinations ranked by F1 across 5 datasets.

    Rows = datasets (blocks), Columns = hybrid combinations (treatments).
    We build a 5 × 25 matrix of F1 scores and run scipy.stats.friedmanchisquare.
    """
    from sklearn.metrics import f1_score as sk_f1

    datasets = sorted(set(k[0] for k in preds))
    combos   = sorted(set((k[1], k[2]) for k in preds))

    # Build F1 matrix  [n_datasets × n_combos]
    f1_matrix = np.full((len(datasets), len(combos)), np.nan)
    for di, dataset in enumerate(datasets):
        for ci, (primary, secondary) in enumerate(combos):
            key = (dataset, primary, secondary)
            if key in preds:
                d  = preds[key]
                f1 = sk_f1(d['y_true'], d['y_pred'], zero_division=0)
                f1_matrix[di, ci] = f1

    # Drop combos with any NaN
    valid_mask = ~np.isnan(f1_matrix).any(axis=0)
    f1_clean   = f1_matrix[:, valid_mask]
    combos_clean = [combos[i] for i in range(len(combos)) if valid_mask[i]]

    if f1_clean.shape[1] < 2:
        return pd.DataFrame([{'Note': 'Not enough valid combos for Friedman test.'}]), pd.DataFrame()

    # Run Friedman test
    stat, pval = stats.friedmanchisquare(*f1_clean.T)

    summary = pd.DataFrame([{
        'Test':            'Friedman Chi-Square',
        'Chi2_Statistic':  round(float(stat), 4),
        'p_value':         round(float(pval), 6),
        'df':              f1_clean.shape[1] - 1,
        'Significant_0.05': pval < 0.05,
        'Interpretation':  ('Significant differences across hybrid models'
                            if pval < 0.05
                            else 'No significant differences across hybrid models'),
    }])

    # Average rank per combo (for post-hoc interpretation)
    ranks = np.apply_along_axis(lambda row: stats.rankdata(-row), 1, f1_clean)
    avg_ranks = ranks.mean(axis=0)

    ranking_df = pd.DataFrame({
        'Hybrid_Combination': [f'{p}+{s}' for p, s in combos_clean],
        'Avg_F1_Rank':        [round(r, 3) for r in avg_ranks],
        'Mean_F1_Across_Datasets': [round(f1_clean[:, i].mean(), 4)
                                    for i in range(len(combos_clean))],
    }).sort_values('Avg_F1_Rank').reset_index(drop=True)
    ranking_df.index += 1  # 1-based rank position

    return summary, ranking_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Advanced Metrics Computation")
    print("="*60)

    preds = load_all_predictions()

    # ── Per-model probability metrics ──────────────────────────────────────
    print("\n[1/3] Computing KS / Brier Score / ECE …")
    metrics_df = compute_per_model_metrics(preds)
    print(metrics_df[['Dataset', 'Hybrid_Combination',
                       'KS_Statistic', 'Brier_Score', 'ECE']].to_string(index=False))

    # ── McNemar Test ───────────────────────────────────────────────────────
    print("\n[2/3] Running McNemar Tests …")
    mcnemar_df = run_mcnemar_tests(preds)
    sig_count  = mcnemar_df['Significant_0.05'].sum() if not mcnemar_df.empty else 0
    print(f"  {len(mcnemar_df)} pairwise tests | {sig_count} significant (p < 0.05)")

    # ── Friedman Test ──────────────────────────────────────────────────────
    print("\n[3/3] Running Friedman Test …")
    friedman_summary, friedman_ranking = run_friedman_test(preds)
    print(friedman_summary.to_string(index=False))

    # ── Write Excel ────────────────────────────────────────────────────────
    print(f"\nSaving → {OUTPUT_EXCEL}")
    with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, sheet_name='KS_Brier_ECE',  index=False)
        mcnemar_df.to_excel(writer, sheet_name='McNemar_Test',  index=False)
        friedman_summary.to_excel(writer, sheet_name='Friedman_Summary', index=False)
        if not friedman_ranking.empty:
            friedman_ranking.to_excel(writer, sheet_name='Friedman_Rankings', index=True)

    print(f"\n{'='*60}")
    print(f"  Done! Output: {OUTPUT_EXCEL}")
    print(f"    Sheet 'KS_Brier_ECE'      → {len(metrics_df)} rows")
    print(f"    Sheet 'McNemar_Test'      → {len(mcnemar_df)} rows")
    print(f"    Sheet 'Friedman_Summary'  → overall test result")
    print(f"    Sheet 'Friedman_Rankings' → per-combo average F1 ranking")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
