"""
save_predictions.py
===================
Re-runs all 125 hybrid model combinations (same config as hybrid_model_pipeline.py)
and saves per-sample predictions to predictions/<dataset>_<primary>_<secondary>.npz

Each .npz contains:
    y_true  – ground-truth labels
    y_pred  – binary predictions
    y_prob  – predicted probabilities (class=1)

These files are required by compute_advanced_metrics.py to compute:
    KS Statistic, Brier Score, ECE, McNemar Test, Friedman Test

Run:
    python save_predictions.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# ── Secondary Models ──────────────────────────────────────────────────────────
from sklearn.ensemble import IsolationForest

# ── Primary Models ────────────────────────────────────────────────────────────
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Config (must match hybrid_model_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
PREPROCESSED_DIR    = "preprocessed_data"
PREDICTIONS_DIR     = "predictions"
BATCH_SIZE          = 256
EPOCHS              = 1
PRIMARY_SAMPLE_SIZE = 200_000

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Deep-learning secondary models (identical to pipeline)
# ─────────────────────────────────────────────────────────────────────────────
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
        rec_cosine    = torch.nn.functional.cosine_similarity(x, x_prime, dim=1).unsqueeze(1)
        z     = torch.cat([z_c, rec_euclidean, rec_cosine], dim=1)
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
    loader    = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE, shuffle=True)
    model.train()

    if model_type == 'deep_svdd':
        model.eval()
        with torch.no_grad():
            c = torch.mean(model(tensor_x[:50000]), dim=0).detach()
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
                mu  = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
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
                exp_term = torch.exp(-0.5 * torch.sum(
                    torch.sum(z_mu.unsqueeze(-1) * cov_inv.unsqueeze(0), dim=-2) * z_mu, dim=-1))
                det_cov = torch.clamp(det_cov, min=1e-12)
                p_z = torch.sum(phi.unsqueeze(0) * exp_term /
                                torch.sqrt((2 * np.pi)**z.size(1) * det_cov.unsqueeze(0)), dim=1)
                p_z = torch.clamp(p_z, min=1e-12)
                loss_energy = torch.mean(-torch.log(p_z))
                cov_diag    = torch.diagonal(cov, dim1=1, dim2=2)
                loss_cov    = torch.sum(1.0 / cov_diag)
                loss = loss_rec + 0.1 * loss_energy + 0.005 * loss_cov
            elif model_type == 'deep_svdd':
                outputs = model(batch_x)
                dist    = torch.sum((outputs - c) ** 2, dim=1)
                loss    = torch.mean(dist)
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
            mu  = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(-1)
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
    loader   = DataLoader(TensorDataset(tensor_x), batch_size=BATCH_SIZE * 4, shuffle=False)
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
                exp_term = torch.exp(-0.5 * torch.sum(
                    torch.sum(z_mu.unsqueeze(-1) * cov_inv.unsqueeze(0), dim=-2) * z_mu, dim=-1))
                det_cov = torch.clamp(det_cov, min=1e-12)
                p_z = torch.sum(phi.unsqueeze(0) * exp_term /
                                torch.sqrt((2 * np.pi)**z.size(1) * det_cov.unsqueeze(0)), dim=1)
                p_z = torch.clamp(p_z, min=1e-12)
                errors = -torch.log(p_z)
            elif model_type == 'deep_svdd':
                c = extra
                outputs = model(batch_x)
                errors  = torch.sum((outputs - c) ** 2, dim=1)
            else:
                recon_x = model(batch_x)
                errors  = torch.mean((batch_x - recon_x) ** 2, dim=1)
            all_errors.append(errors.cpu().numpy())
    return np.concatenate(all_errors)


def pred_filename(dataset_idx, prim_name, sec_name):
    return os.path.join(PREDICTIONS_DIR,
                        f"D{dataset_idx}_{prim_name}_{sec_name}.npz")


def main():
    total_saved  = 0
    total_skip   = 0

    for dataset_idx in range(1, 6):
        print(f"\n{'='*60}")
        print(f"  Dataset {dataset_idx}")
        print(f"{'='*60}")

        npz_path = os.path.join(PREPROCESSED_DIR, f"dataset_{dataset_idx}.npz")
        if not os.path.exists(npz_path):
            print(f"  [SKIP] {npz_path} not found"); continue

        data    = np.load(npz_path)
        X_train = data['X_train']
        X_test  = data['X_test']
        y_train = data['y_train']
        y_test  = data['y_test']
        input_dim = X_train.shape[1]
        print(f"  Train {X_train.shape}  |  Test {X_test.shape}")

        # ── Secondary models ───────────────────────────────────────────────
        print("  Training secondary models …")
        secondary_scores = {}

        print("    [M6] Isolation Forest …")
        m6 = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
        m6.fit(X_train)
        secondary_scores['M6'] = {
            'train': -m6.score_samples(X_train),
            'test':  -m6.score_samples(X_test)
        }

        print("    [M7] Autoencoder …")
        m7 = train_torch_model(Autoencoder(input_dim), X_train, 'ae')
        secondary_scores['M7'] = {
            'train': get_torch_scores(m7, X_train, 'ae'),
            'test':  get_torch_scores(m7, X_test,  'ae')
        }

        print("    [M8] VAE …")
        m8 = train_torch_model(VAE(input_dim), X_train, 'vae')
        secondary_scores['M8'] = {
            'train': get_torch_scores(m8, X_train, 'vae'),
            'test':  get_torch_scores(m8, X_test,  'vae')
        }

        print("    [M9] DAGMM …")
        m9, dagmm_params = train_torch_model(DAGMM(input_dim), X_train, 'dagmm')
        secondary_scores['M9'] = {
            'train': get_torch_scores(m9, X_train, 'dagmm', extra=dagmm_params),
            'test':  get_torch_scores(m9, X_test,  'dagmm', extra=dagmm_params)
        }

        print("    [M10] Deep SVDD …")
        m10, svdd_c = train_torch_model(DeepSVDD(input_dim), X_train, 'deep_svdd')
        secondary_scores['M10'] = {
            'train': get_torch_scores(m10, X_train, 'deep_svdd', extra=svdd_c),
            'test':  get_torch_scores(m10, X_test,  'deep_svdd', extra=svdd_c)
        }

        # ── Primary models ─────────────────────────────────────────────────
        primary_models = {
            'M1': lambda: HistGradientBoostingClassifier(max_iter=100, random_state=42),
            'M2': lambda: ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=4),
            'M3': lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
            'M4': lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4),
            'M5': lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42),
        }

        for sec_name, scores in secondary_scores.items():
            X_train_full = np.column_stack((X_train, scores['train'])).astype(np.float32)
            X_test_h     = np.column_stack((X_test,  scores['test'])).astype(np.float32)

            # Stratified subsample (identical logic to pipeline)
            if len(X_train_full) > PRIMARY_SAMPLE_SIZE:
                rng    = np.random.default_rng(42)
                idx_pos = np.where(y_train == 1)[0]
                idx_neg = np.where(y_train == 0)[0]
                n_pos = min(len(idx_pos), int(PRIMARY_SAMPLE_SIZE * len(idx_pos) / len(y_train)))
                n_neg = min(PRIMARY_SAMPLE_SIZE - n_pos, len(idx_neg))
                chosen = np.concatenate([rng.choice(idx_pos, n_pos, replace=False),
                                         rng.choice(idx_neg, n_neg, replace=False)])
                rng.shuffle(chosen)
                X_train_h = X_train_full[chosen]
                y_train_h = y_train[chosen]
            else:
                X_train_h = X_train_full
                y_train_h = y_train

            for prim_name, model_fn in primary_models.items():
                out_path = pred_filename(dataset_idx, prim_name, sec_name)
                if os.path.exists(out_path):
                    print(f"    [{prim_name}+{sec_name}] already saved – skip")
                    total_skip += 1
                    continue

                print(f"    [{prim_name}+{sec_name}] training …", end='', flush=True)
                t0 = time.time()
                try:
                    model  = model_fn()
                    model.fit(X_train_h, y_train_h)
                    y_pred = model.predict(X_test_h)
                    y_prob = (model.predict_proba(X_test_h)[:, 1]
                              if hasattr(model, 'predict_proba')
                              else y_pred.astype(float))

                    np.savez_compressed(out_path,
                                        y_true=y_test,
                                        y_pred=y_pred,
                                        y_prob=y_prob)
                    total_saved += 1
                    print(f"  saved ({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"  ERROR: {e}")

    print(f"\nDone. Saved: {total_saved}  Skipped: {total_skip}")
    print(f"Files in → {PREDICTIONS_DIR}/")


if __name__ == "__main__":
    main()
