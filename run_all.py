#!/usr/bin/env python3
"""
============================================================
  AI ML Hybrid Fraud Detection — Full Pipeline Runner
============================================================
  Runs the entire project end-to-end in the correct order.
  Usage:  python run_all.py
============================================================
"""

import subprocess
import sys
import time
import os

# All scripts in execution order
PIPELINE_STEPS = [
    {
        "script": "preprocess_datasets.py",
        "name": "Step 1/8 — Preprocess Datasets",
        "desc": "Loading raw CSVs, encoding, scaling, and splitting into train/test sets",
    },
    {
        "script": "hybrid_model_pipeline.py",
        "name": "Step 2/8 — Hybrid Model Pipeline",
        "desc": "Training 125 hybrid combinations (5 Primary × 5 Secondary × 5 Datasets)",
    },
    {
        "script": "save_predictions.py",
        "name": "Step 3/8 — Save Predictions",
        "desc": "Saving per-sample predictions (y_true, y_pred, y_prob) for all 125 combos",
    },
    {
        "script": "compute_advanced_metrics.py",
        "name": "Step 4/8 — Compute Advanced Metrics",
        "desc": "Calculating KS Statistic, Brier Score, ECE, McNemar Test, Friedman Test",
    },
    {
        "script": "generate_roc_curves.py",
        "name": "Step 5/8 — Generate ROC Curves",
        "desc": "Generating ROC curve plots (PDF + TIFF) for each dataset",
    },
    {
        "script": "generate_pr_curves.py",
        "name": "Step 6/8 — Generate PR Curves",
        "desc": "Generating Precision-Recall curve plots (PDF + TIFF) for each dataset",
    },
    {
        "script": "generate_confusion_matrices.py",
        "name": "Step 7/8 — Generate Confusion Matrices",
        "desc": "Generating confusion matrix plots for all model combinations",
    },
    {
        "script": "export_cm_excel.py",
        "name": "Step 8/8 — Export to Excel",
        "desc": "Exporting confusion matrices and final results to Excel",
    },
]


def run_step(step, project_dir):
    """Run a single pipeline step and return success status."""
    script_path = os.path.join(project_dir, step["script"])

    if not os.path.exists(script_path):
        print(f"  ⚠️  Script not found: {step['script']} — SKIPPING")
        return True  # Don't fail the whole pipeline for a missing optional script

    print(f"\n{'='*60}")
    print(f"  {step['name']}")
    print(f"  {step['desc']}")
    print(f"{'='*60}\n")

    start = time.time()

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=project_dir,
    )

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if result.returncode == 0:
        print(f"\n  ✅ {step['name']} — completed in {minutes}m {seconds}s")
        return True
    else:
        print(f"\n  ❌ {step['name']} — FAILED (exit code {result.returncode})")
        return False


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    total_start = time.time()

    print("\n" + "█" * 60)
    print("█  AI ML Hybrid Fraud Detection — Full Pipeline")
    print("█  Running all 8 steps...")
    print("█" * 60)

    passed = 0
    failed = 0
    skipped_scripts = []

    for step in PIPELINE_STEPS:
        success = run_step(step, project_dir)
        if success:
            passed += 1
        else:
            failed += 1
            skipped_scripts.append(step["script"])
            # Ask whether to continue or abort
            print(f"\n  Step failed: {step['script']}")
            print(f"  Continuing with remaining steps...\n")

    total_elapsed = time.time() - total_start
    total_min = int(total_elapsed // 60)
    total_sec = int(total_elapsed % 60)

    print("\n" + "█" * 60)
    print("█  PIPELINE COMPLETE")
    print(f"█  Total time: {total_min}m {total_sec}s")
    print(f"█  Passed: {passed}/{len(PIPELINE_STEPS)}")
    if failed:
        print(f"█  Failed: {failed} — {', '.join(skipped_scripts)}")
    print("█" * 60)

    print("\n📊 Output files:")
    print("   • updated_model_results.csv / .xlsx    — All 125 combo results")
    print("   • hybrid_evaluation_results.csv / .xlsx — Extended evaluation")
    print("   • top_10_models_analysis.xlsx           — Top 10 models")
    print("   • roc_curves/                           — ROC + confusion matrix plots")
    print("   • pr_curves/                            — Precision-Recall plots")
    print("   • predictions/                          — Per-sample prediction files")
    print()


if __name__ == "__main__":
    main()
