import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt

# Secondary Models
from sklearn.ensemble import IsolationForest

# Primary Models
from sklearn.ensemble import HistGradientBoostingClassifier  # M1: replaces XGBoost
from sklearn.ensemble import ExtraTreesClassifier            # M2: replaces LightGBM – pure sklearn, no hang
from sklearn.ensemble import GradientBoostingClassifier      # M3: replaces CatBoost – pure sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier             # M5: replaces TabNet

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
PREPROCESSED_DIR    = "preprocessed_data"
RESULTS_FILE        = "updated_model_results.csv"
BATCH_SIZE          = 256
EPOCHS              = 1
PRIMARY_SAMPLE_SIZE = 200_000   # cap primary-model training rows to avoid segfault on Apple Silicon

# Hardware acceleration (Apple Silicon MPS, or fall back to CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# Deep-learning Secondary Models
# ─────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(),
                                     nn.Linear(32, 16),         nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(),
                                     nn.Linear(32, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

class VAE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1        = nn.Linear(input_dim, 32)
        self.fc2_mu     = nn.Linear(32, 16)
        self.fc2_logvar = nn.Linear(32, 16)
        self.fc3        = nn.Linear(16, 32)
        self.fc4        = nn.Linear(32, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.fc4(torch.relu(self.fc3(z)))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

class DAGMM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.Tanh(),
                                     nn.Linear(32, 16), nn.Tanh(),
                                     nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.Tanh(),
                                     nn.Linear(16, 32), nn.Tanh(),
                                     nn.Linear(32, input_dim))
        
        self.estimation = nn.Sequential(nn.Linear(10, 16), nn.Tanh(),
                                        nn.Dropout(0.5),
                                        nn.Linear(16, 4),
                                        nn.Softmax(dim=1))
        
    def forward(self, x):
        z_c = self.encoder(x)
        x_prime = self.decoder(z_c)
        
        rec_euclidean = torch.nn.functional.pairwise_distance(x, x_prime, p=2).unsqueeze(1)
        rec_cosine = torch.nn.functional.cosine_similarity(x, x_prime, dim=1).unsqueeze(1)
        
        z = torch.cat([z_c, rec_euclidean, rec_cosine], dim=1)
        gamma = self.estimation(z)
        
        return z_c, x_prime, gamma, z

class DeepSVDD(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(),
                                     nn.Linear(32, 16))
    def forward(self, x):
        return self.encoder(x)

def train_torch_model(model, X_train, model_type='ae'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tensor_x  = torch.Tensor(X_train).to(device)
    
    loader = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    
    if model_type == 'deep_svdd':
        # Initialize center c using subset
        model.eval()
        with torch.no_grad():
            sub_x = tensor_x[:50000]
            outputs = model(sub_x)
            c = torch.mean(outputs, dim=0).detach()
        model.train()
        
    for _ in range(EPOCHS):
        for (batch_x,) in loader:
            optimizer.zero_grad()
            if model_type == 'vae':
                recon_x, mu, logvar = model(batch_x)
                loss = (nn.functional.mse_loss(recon_x, batch_x, reduction='sum')
                        - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
            elif model_type == 'dagmm':
                z_c, x_prime, gamma, z = model(batch_x)
                loss_rec = torch.mean(torch.sum((batch_x - x_prime)**2, dim=1))
                
                gamma_sum = torch.sum(gamma, dim=0)
                phi = gamma_sum / gamma.size(0)
                mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
                z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
                z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
                cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)
                cov = cov + torch.eye(z.size(1)).to(device) * 1e-6
                
                try:
                    cov_inv = torch.inverse(cov)
                    det_cov = torch.det(cov)
                except RuntimeError:
                    cov_inv = torch.pinverse(cov)
                    det_cov = torch.ones(cov.shape[0]).to(device)
                
                exp_term = torch.exp(-0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inv.unsqueeze(0), dim=-2) * z_mu, dim=-1))
                det_cov = torch.clamp(det_cov, min=1e-12)
                p_z = torch.sum(phi.unsqueeze(0) * exp_term / torch.sqrt((2 * np.pi)**z.size(1) * det_cov.unsqueeze(0)), dim=1)
                p_z = torch.clamp(p_z, min=1e-12)
                
                loss_energy = torch.mean(-torch.log(p_z))
                cov_diag = torch.diagonal(cov, dim1=1, dim2=2)
                loss_cov = torch.sum(1.0 / cov_diag)
                
                loss = loss_rec + 0.1 * loss_energy + 0.005 * loss_cov
            elif model_type == 'deep_svdd':
                outputs = model(batch_x)
                dist = torch.sum((outputs - c) ** 2, dim=1)
                loss = torch.mean(dist)
            else:
                recon_x = model(batch_x)
                loss    = nn.MSELoss()(recon_x, batch_x)
            loss.backward()
            optimizer.step()
            
    if model_type == 'dagmm':
        model.eval()
        with torch.no_grad():
            sub_x = tensor_x[:50000]
            z_c, x_prime, gamma, z = model(sub_x)
            gamma_sum = torch.sum(gamma, dim=0)
            phi = gamma_sum / gamma.size(0)
            mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
            z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
            z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
            cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / gamma_sum.unsqueeze(-1).unsqueeze(-1)
            cov = cov + torch.eye(z.size(1)).to(device) * 1e-6
        return model, (phi, mu, cov)
    elif model_type == 'deep_svdd':
        return model, c
    return model

def get_torch_scores(model, X, model_type='ae', extra=None):
    model.eval()
    tensor_x = torch.Tensor(X).to(device)
    loader = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE*4, shuffle=False)
    
    all_errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            if model_type == 'vae':
                recon_x, mu, logvar = model(batch_x)
                errors = torch.mean((batch_x - recon_x) ** 2, dim=1)
            elif model_type == 'dagmm':
                phi, mu, cov = extra
                z_c, x_prime, gamma, z = model(batch_x)
                try:
                    cov_inv = torch.inverse(cov)
                    det_cov = torch.det(cov)
                except RuntimeError:
                    cov_inv = torch.pinverse(cov)
                    det_cov = torch.ones(cov.shape[0]).to(device)
                z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
                exp_term = torch.exp(-0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inv.unsqueeze(0), dim=-2) * z_mu, dim=-1))
                det_cov = torch.clamp(det_cov, min=1e-12)
                p_z = torch.sum(phi.unsqueeze(0) * exp_term / torch.sqrt((2 * np.pi)**z.size(1) * det_cov.unsqueeze(0)), dim=1)
                p_z = torch.clamp(p_z, min=1e-12)
                errors = -torch.log(p_z)
            elif model_type == 'deep_svdd':
                c = extra
                outputs = model(batch_x)
                errors = torch.sum((outputs - c) ** 2, dim=1)
            else:
                recon_x = model(batch_x)
                errors  = torch.mean((batch_x - recon_x) ** 2, dim=1)
            all_errors.append(errors.cpu().numpy())
            
    return np.concatenate(all_errors)


# ─────────────────────────────────────────────
# Load already-completed combinations (for resuming)
# ─────────────────────────────────────────────
def load_completed_keys():
    """Return a set of (Dataset, Primary_Model, Secondary_Model) tuples already in the CSV."""
    if not os.path.exists(RESULTS_FILE):
        return set()
    try:
        df = pd.read_csv(RESULTS_FILE)
        return set(zip(df['Dataset'], df['Primary_Model'], df['Secondary_Model']))
    except Exception:
        return set()

def append_result(row_dict, file_exists):
    """Append a single result row to the CSV and refresh the Excel file."""
    pd.DataFrame([row_dict]).to_csv(
        RESULTS_FILE, mode='a', header=not file_exists, index=False
    )
    # Always keep Excel up to date
    try:
        pd.read_csv(RESULTS_FILE).to_excel(
            RESULTS_FILE.replace('.csv', '.xlsx'), index=False
        )
    except Exception:
        pass

# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def main():
    all_results   = []
    completed_keys = load_completed_keys()
    csv_exists     = os.path.exists(RESULTS_FILE)

    if csv_exists and completed_keys:
        all_results = pd.read_csv(RESULTS_FILE).to_dict('records')

    os.makedirs("roc_curves", exist_ok=True)

    for dataset_idx in range(1, 6):
        print(f"\n{'='*60}")
        print(f"  Processing Dataset {dataset_idx}")
        print(f"{'='*60}")

        npz_path = os.path.join(PREPROCESSED_DIR, f"dataset_{dataset_idx}.npz")
        if not os.path.exists(npz_path):
            print(f"  [ERROR] Preprocessed file not found: {npz_path}")
            continue

        try:
            data    = np.load(npz_path)
            X_train = data['X_train']
            X_test  = data['X_test']
            y_train = data['y_train']
            y_test  = data['y_test']
        except Exception as e:
            print(f"  [ERROR] Failed to load {npz_path}: {e}")
            continue

        input_dim = X_train.shape[1]
        print(f"  Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")
        
        # ── Secondary Models ─────────────────────────────────────────
        print("\n  Training secondary (anomaly-detection) models...")
        secondary_scores = {}

        # M6 – Isolation Forest
        print("    [M6] Isolation Forest...")
        m6 = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
        m6.fit(X_train)
        secondary_scores['M6'] = {
            'train': -m6.score_samples(X_train),
            'test':  -m6.score_samples(X_test)
        }

        # M7 – Autoencoder
        print("    [M7] Deep Autoencoder (DAE)...")
        m7 = train_torch_model(Autoencoder(input_dim), X_train, 'ae')
        secondary_scores['M7'] = {
            'train': get_torch_scores(m7, X_train, 'ae'),
            'test':  get_torch_scores(m7, X_test,  'ae')
        }

        # M8 – VAE
        print("    [M8] Variational Autoencoder (VAE)...")
        m8 = train_torch_model(VAE(input_dim), X_train, 'vae')
        secondary_scores['M8'] = {
            'train': get_torch_scores(m8, X_train, 'vae'),
            'test':  get_torch_scores(m8, X_test,  'vae')
        }

        # M9 – DAGMM
        print("    [M9] DAGMM...")
        m9, dagmm_params = train_torch_model(DAGMM(input_dim), X_train, 'dagmm')
        secondary_scores['M9'] = {
            'train': get_torch_scores(m9, X_train, 'dagmm', extra=dagmm_params),
            'test':  get_torch_scores(m9, X_test,  'dagmm', extra=dagmm_params)
        }

        # M10 – Deep SVDD
        print("    [M10] Deep SVDD...")
        m10, svdd_c = train_torch_model(DeepSVDD(input_dim), X_train, 'deep_svdd')
        secondary_scores['M10'] = {
            'train': get_torch_scores(m10, X_train, 'deep_svdd', extra=svdd_c),
            'test':  get_torch_scores(m10, X_test,  'deep_svdd', extra=svdd_c)
        }

        # ── Primary × Secondary combinations ─────────────────────────
        primary_models = {
            'M1': lambda: HistGradientBoostingClassifier(max_iter=100, random_state=42),        # replaces XGBoost
            'M2': lambda: ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=4),   # replaces LightGBM
            'M3': lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),        # replaces CatBoost
            'M4': lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4),
            'M5': lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42),
        }

        plt.figure(figsize=(12, 10))
        roc_plotted = False

        for sec_name, scores in secondary_scores.items():
            print(f"\n  ── Secondary: {sec_name} ──────────────────────────────")
            X_train_full = np.column_stack((X_train, scores['train'])).astype(np.float32)
            X_test_h     = np.column_stack((X_test,  scores['test'])).astype(np.float32)

            # Stratified subsample for primary models to avoid C-level OOM/segfault
            if len(X_train_full) > PRIMARY_SAMPLE_SIZE:
                rng = np.random.default_rng(42)
                # Stratified: sample proportionally from each class
                idx_pos = np.where(y_train == 1)[0]
                idx_neg = np.where(y_train == 0)[0]
                n_pos = min(len(idx_pos), int(PRIMARY_SAMPLE_SIZE * len(idx_pos) / len(y_train)))
                n_neg = PRIMARY_SAMPLE_SIZE - n_pos
                n_neg = min(n_neg, len(idx_neg))
                chosen = np.concatenate([
                    rng.choice(idx_pos, n_pos, replace=False),
                    rng.choice(idx_neg, n_neg, replace=False)
                ])
                rng.shuffle(chosen)
                X_train_h = X_train_full[chosen]
                y_train_h = y_train[chosen]
                print(f"    (Subsampled train: {len(X_train_h):,} rows for primary models)")
            else:
                X_train_h = X_train_full
                y_train_h = y_train

            for prim_name, model_fn in primary_models.items():
                key = (f'D{dataset_idx}', prim_name, sec_name)
                if key in completed_keys:
                    print(f"    [{prim_name}+{sec_name}] ✓ already done – skipping")
                    continue

                print(f"    [{prim_name}+{sec_name}] Training...", end='', flush=True)
                t0 = time.time()
                try:
                    model = model_fn()
                    model.fit(X_train_h, y_train_h)
                    y_pred = model.predict(X_test_h)
                    y_prob = (model.predict_proba(X_test_h)[:, 1]
                              if hasattr(model, 'predict_proba') else y_pred.astype(float))

                    acc  = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, zero_division=0)
                    rec  = recall_score(y_test, y_pred, zero_division=0)
                    f1   = f1_score(y_test, y_pred, zero_division=0)
                    try:
                        auc = roc_auc_score(y_test, y_prob)
                    except ValueError:
                        auc = np.nan

                    elapsed = time.time() - t0
                    print(f"  F1={f1:.4f}  AUC={auc:.4f}  ({elapsed:.1f}s)")

                    if not np.isnan(auc):
                        try:
                            fpr, tpr, _ = roc_curve(y_test, y_prob)
                            plt.plot(fpr, tpr, lw=1,
                                     label=f'{prim_name}+{sec_name} (AUC={auc:.3f})')
                            roc_plotted = True
                        except Exception:
                            pass

                    row = {
                        'Dataset':          f'D{dataset_idx}',
                        'Primary_Model':    prim_name,
                        'Secondary_Model':  sec_name,
                        'Hybrid_Combination': f'{prim_name} + {sec_name}',
                        'Accuracy':         acc,
                        'Precision':        prec,
                        'Recall':           rec,
                        'F1_Score':         f1,
                        'AUC':              auc,
                        'Time_Seconds':     elapsed,
                    }
                    all_results.append(row)
                    completed_keys.add(key)
                    append_result(row, csv_exists)
                    csv_exists = True

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  ERROR: {e}")
                    row = {
                        'Dataset':          f'D{dataset_idx}',
                        'Primary_Model':    prim_name,
                        'Secondary_Model':  sec_name,
                        'Hybrid_Combination': f'{prim_name} + {sec_name}',
                        'Accuracy':         np.nan,
                        'Precision':        np.nan,
                        'Recall':           np.nan,
                        'F1_Score':         np.nan,
                        'AUC':              np.nan,
                        'Time_Seconds':     elapsed,
                    }
                    all_results.append(row)
                    completed_keys.add(key)
                    append_result(row, csv_exists)
                    csv_exists = True

        if roc_plotted:
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves – Dataset {dataset_idx}')
            plt.legend(loc='lower right', fontsize='small', ncol=2)
            plt.tight_layout()
            plt.savefig(f"roc_curves/dataset_{dataset_idx}_roc.pdf",  format='pdf')
            plt.savefig(f"roc_curves/dataset_{dataset_idx}_roc.tiff", format='tiff', dpi=400)
            print(f"\n  ROC curves saved for Dataset {dataset_idx}.")
        plt.close()

    if all_results:
        final_df = pd.DataFrame(all_results)
        excel_out = "updated_model_results.xlsx"
        final_df.to_excel(excel_out, index=False)
        print(f"\n{'='*60}")
        print(f"  Pipeline complete!")
        print(f"  Results  → {RESULTS_FILE}  ({len(final_df)} rows)")
        print(f"  Excel    → {excel_out}")
        print(f"  ROC PDFs → roc_curves/")
        print(f"{'='*60}")
    else:
        print("\nNo new results were generated.")


if __name__ == "__main__":
    main()
