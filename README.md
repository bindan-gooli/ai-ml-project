# 🔍 Hybrid Machine Learning Framework for Credit Card Fraud Detection

A comprehensive evaluation framework that combines **5 supervised (primary) models** with **5 unsupervised anomaly-detection (secondary) models** to create **25 hybrid combinations**, tested across **5 real-world fraud detection datasets** — totaling **125 experiments**.

---

## 📌 Overview

Credit card fraud detection is a critical challenge in financial security. This project implements a **hybrid approach** where unsupervised anomaly-detection models generate anomaly scores that are appended as additional features to the original dataset, which are then used by supervised classifiers for final fraud prediction.

### Key Highlights
- **125 hybrid model combinations** (5 Primary × 5 Secondary × 5 Datasets)
- **Deep learning** anomaly detectors: Autoencoder, VAE, DAGMM, Deep SVDD
- **Apple Silicon (MPS) acceleration** with CPU fallback
- **Automated pipeline** with resume capability — picks up where it left off
- **Publication-ready outputs**: ROC curves, PR curves, confusion matrices (PDF + TIFF)
- **Advanced statistical metrics**: KS Statistic, Brier Score, ECE, McNemar Test, Friedman Test

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Raw Dataset (CSV)                     │
│                         │                                │
│                    Preprocessing                         │
│            (Scaling, Encoding, Split)                    │
│                    │          │                           │
│             ┌──────┘          └──────┐                   │
│             ▼                        ▼                   │
│     Secondary Model            Original Features         │
│    (Anomaly Scores)                  │                   │
│             │                        │                   │
│             └────── Concatenate ─────┘                   │
│                         │                                │
│                    Primary Model                         │
│               (Supervised Classifier)                    │
│                         │                                │
│                   Fraud Prediction                       │
│              (Accuracy, F1, AUC, etc.)                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🤖 Models

### Primary Models (Supervised Classifiers)
| Code | Model | Implementation |
|------|-------|----------------|
| M1 | Hist Gradient Boosting | `sklearn.ensemble.HistGradientBoostingClassifier` |
| M2 | Extra Trees | `sklearn.ensemble.ExtraTreesClassifier` |
| M3 | Gradient Boosting | `sklearn.ensemble.GradientBoostingClassifier` |
| M4 | Random Forest | `sklearn.ensemble.RandomForestClassifier` |
| M5 | MLP Neural Network | `sklearn.neural_network.MLPClassifier` |

### Secondary Models (Unsupervised Anomaly Detectors)
| Code | Model | Implementation |
|------|-------|----------------|
| M6 | Isolation Forest | `sklearn.ensemble.IsolationForest` |
| M7 | Deep Autoencoder (DAE) | PyTorch custom |
| M8 | Variational Autoencoder (VAE) | PyTorch custom |
| M9 | DAGMM | PyTorch custom |
| M10 | Deep SVDD | PyTorch custom |

---

## 📊 Datasets

The raw datasets are **not included** in this repository due to size (total ~1.4 GB). Download them from Kaggle and place in the corresponding folders:

| Folder | Dataset | Size | Kaggle Link |
|--------|---------|------|-------------|
| `Data set 1/` | Credit Card Fraud Dataset | 471 MB | [Download](https://www.kaggle.com/datasets/dylanmoraes/credit-card-fraud-dataset) |
| `data set 2/` | Credit Card Fraud Detection (10K) | 352 KB | [Download](https://www.kaggle.com/datasets/miadul/credit-card-fraud-detection-dataset) |
| `data set 3/` | Credit Card Fraud Detection 2023 | 310 MB | [Download](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) |
| `data set 4/` | Credit Card Fraud (MLG-ULB) | 144 MB | [Download](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| `data set 5/` | Fraud Detection (Kartik2112) | 478 MB | [Download](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- macOS (Apple Silicon recommended) or Linux/Windows with CUDA

### Step 1: Clone the Repository
```bash
git clone https://github.com/bindan-gooli/ai-ml-project.git
cd ai-ml-project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or using a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### Step 3: Download Datasets
Download all 5 datasets from the Kaggle links above and place them in their respective folders:
```
ai-ml-project/
├── Data set 1/
│   └── Fraud.csv
├── data set 2/
│   └── credit_card_fraud_10k.csv
├── data set 3/
│   └── creditcard_2023.csv
├── data set 4/
│   └── creditcard.csv
└── data set 5/
    ├── fraudTrain.csv
    └── fraudTest.csv
```

### Step 4: Preprocess the Datasets
```bash
python preprocess_datasets.py
```
This will:
- Load each dataset
- Drop irrelevant columns (IDs, names, timestamps)
- Label-encode categorical features
- Standardize (z-score) all features
- Split into 80/20 train/test
- Save as `.npz` files in `preprocessed_data/`

### Step 5: Run the Hybrid Model Pipeline
```bash
python hybrid_model_pipeline.py
```
This trains all **125 hybrid combinations** and saves:
- `updated_model_results.csv` — evaluation metrics
- `updated_model_results.xlsx` — Excel version
- `roc_curves/` — ROC curve plots (PDF + TIFF)

> **Note:** The pipeline supports **auto-resume**. If interrupted, just re-run the same command and it will skip already-completed combinations.

---

## 📈 Additional Analysis Scripts

After the main pipeline completes, run these scripts for deeper analysis:

| Script | Purpose | Command |
|--------|---------|---------|
| `save_predictions.py` | Save per-sample predictions (y_true, y_pred, y_prob) for all 125 combos | `python save_predictions.py` |
| `compute_advanced_metrics.py` | Calculate KS Statistic, Brier Score, ECE, McNemar Test, Friedman Test | `python compute_advanced_metrics.py` |
| `generate_roc_curves.py` | Generate detailed ROC curves per dataset | `python generate_roc_curves.py` |
| `generate_pr_curves.py` | Generate Precision-Recall curves per dataset | `python generate_pr_curves.py` |
| `generate_confusion_matrices.py` | Generate confusion matrix plots | `python generate_confusion_matrices.py` |
| `export_cm_excel.py` | Export confusion matrices to Excel | `python export_cm_excel.py` |
| `generate_final_analysis.py` | Generate final summary analysis | `python generate_final_analysis.py` |
| `quick_charts.py` | Quick comparison charts | `python quick_charts.py` |

---

## 📂 Project Structure

```
ai-ml-project/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore rules
│
├── 🐍 SCRIPTS
│   ├── preprocess_datasets.py            # Step 1: Preprocess raw CSVs
│   ├── hybrid_model_pipeline.py          # Step 2: Main pipeline (125 combos)
│   ├── save_predictions.py               # Step 3: Save per-sample predictions
│   ├── compute_advanced_metrics.py       # Step 4: Advanced statistical metrics
│   ├── generate_roc_curves.py            # ROC curve generation
│   ├── generate_pr_curves.py             # Precision-Recall curve generation
│   ├── generate_confusion_matrices.py    # Confusion matrix generation
│   ├── export_cm_excel.py                # Export confusion matrices to Excel
│   ├── generate_final_analysis.py        # Final analysis report
│   ├── plot_roc_curves.py                # ROC plotting utilities
│   └── quick_charts.py                   # Quick comparison charts
│
├── 📊 RESULTS
│   ├── updated_model_results.csv         # All 125 combo results
│   ├── updated_model_results.xlsx        # Excel version
│   ├── hybrid_evaluation_results.csv     # Extended evaluation results
│   ├── hybrid_evaluation_results.xlsx    # Excel version
│   └── top_10_models_analysis.xlsx       # Top 10 models comparison
│
├── 📁 preprocessed_data/                 # Preprocessed .npz files (5 datasets)
├── 📁 predictions/                       # Per-sample predictions (125 .npz files)
├── 📁 roc_curves/                        # ROC, PR, confusion matrix PDFs & TIFFs
├── 📁 pr_curves/                         # Precision-Recall curve outputs
│
├── 📁 Data set 1/ → 5/                  # Raw datasets (not in repo — download from Kaggle)
└── 📄 AI_Traffic_Congestion_Project_Report.pdf
```

---

## 📋 Evaluation Metrics

Each hybrid combination is evaluated on:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **AUC** | Area Under the ROC Curve |
| **KS Statistic** | Max separation between TPR and FPR distributions |
| **Brier Score** | Mean squared error of predicted probabilities |
| **ECE** | Expected Calibration Error |
| **McNemar Test** | Pairwise statistical significance test |
| **Friedman Test** | Multi-model ranking comparison |

---

## ⚙️ Configuration

Key parameters in `hybrid_model_pipeline.py`:

```python
BATCH_SIZE          = 256         # Batch size for deep learning models
EPOCHS              = 1           # Training epochs for secondary models
PRIMARY_SAMPLE_SIZE = 200_000     # Max training rows (prevents OOM on large datasets)
```

---

## 🖥️ Hardware

- Developed and tested on **Apple Silicon (M-series)** with MPS acceleration
- Falls back to CPU automatically if MPS is unavailable
- Compatible with CUDA GPUs (PyTorch)

---

## 📜 License

This project is for academic and research purposes.
