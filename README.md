# AI ML Hybrid Fraud Detection Project

This repository contains the code and evaluation scripts for a hybrid machine learning pipeline designed for fraud detection.

## Datasets

The raw datasets (Data set 1 to Data set 5) are not included in this repository due to size limitations. Download them from the Kaggle links below and place them in their corresponding folders.

| Folder | Dataset | Kaggle Link |
|--------|---------|-------------|
| `Data set 1/` | Credit Card Fraud Dataset | [Download](https://www.kaggle.com/datasets/dylanmoraes/credit-card-fraud-dataset) |
| `data set 2/` | Credit Card Fraud Detection Dataset | [Download](https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset) |
| `data set 3/` | Credit Card Fraud Detection Dataset 2023 | [Download](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) |
| `data set 4/` | Credit Card Fraud (MLG-ULB) | [Download](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `data set 5/` | Fraud Detection | [Download](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |

## Project Structure
- `hybrid_model_pipeline.py`: Main execution script for the hybrid models.
- `save_predictions.py`: Saves per-sample predictions (y_true, y_pred, y_prob).
- `compute_advanced_metrics.py`: Script for calculating performance metrics (KS Stat, Brier Score, ECE, McNemar, Friedman).
- `generate_roc_curves.py`: ROC curve generation.
- `generate_pr_curves.py`: Precision-Recall curve generation.
- `generate_confusion_matrices.py`: Confusion matrix plots.
- `export_cm_excel.py`: Exports confusion matrices to Excel.
- `preprocess_datasets.py`: Dataset preprocessing.
- `requirements.txt`: Project dependencies.
- `*.xlsx` / `*.csv`: Evaluation results and performance summaries.

## Getting Started
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download the datasets from the Kaggle links above and place them in the corresponding folders.
4. Run the pipeline: `python hybrid_model_pipeline.py`.
