"""
SideShield — Autoencoder 1D : Anomaly Detection
=================================================
Principe :
  - Entraînement UNIQUEMENT sur traces normales (profiling set)
  - Le modèle apprend la structure d'une trace AES normale
  - Détection : erreur de reconstruction MSE > seuil → ATTAQUE

Architecture :
  Encoder : Conv1D progressif → Bottleneck
  Decoder : ConvTranspose1D progressif → Reconstruction

Métriques :
  - MSE de reconstruction (loss)
  - AUC-ROC (détection normale vs attaque)
  - Seuil optimal (95e percentile)
  - Visualisation scores d'anomalie
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import os
import math
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DESKTOP  = os.path.join(os.path.expanduser("~"), "Desktop")
DB_PATH  = os.path.join(DESKTOP, "ASCAD", "ASCAD_databases")
OUT_PATH = os.path.join(DESKTOP, "SideShield")
os.makedirs(OUT_PATH, exist_ok=True)

TRACE_LEN     = 700
BATCH_SIZE    = 256
EPOCHS        = 60
LR            = 1e-3
LATENT_DIM    = 64      # dimension du bottleneck
ANOMALY_PCTL  = 95      # percentile pour le seuil de détection
SEED          = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class TraceDataset(Dataset):
    """Dataset de traces — labels optionnels pour évaluation"""
    def __init__(self, traces, labels=None):
        self.X = torch.tensor(traces, dtype=torch.float32).unsqueeze(1)
        self.y = labels  # None pendant entraînement

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ─────────────────────────────────────────────
# CHARGEMENT + SPLIT NORMAL / ATTAQUE
# ─────────────────────────────────────────────
def load_and_split(filepath):
    """
    Stratégie de split pour anomaly detection :

    NORMAL  = Profiling traces (entraînement du modèle)
              → ce sont les traces "propres" d'exécution AES standard

    ATTAQUE = Attack traces (évaluation)
              → ces traces sont collectées dans un contexte d'attaque
              → le modèle doit les détecter comme anormales

    On simule aussi des traces "normales de test" en gardant
    20% du profiling set pour la validation.
    """
    with h5py.File(filepath, "r") as f:
        X_profiling = f["Profiling_traces/traces"][:].astype(np.float32)
        y_profiling = f["Profiling_traces/labels"][:]
        X_attack    = f["Attack_traces/traces"][:].astype(np.float32)
        y_attack    = f["Attack_traces/labels"][:]
        meta_attack = f["Attack_traces/metadata"][:]

    # Normalisation z-score (params du profiling uniquement)
    mean = X_profiling.mean(axis=0)
    std  = np.where(X_profiling.std(axis=0) == 0,
                    1e-8, X_profiling.std(axis=0))
    X_profiling = (X_profiling - mean) / std
    X_attack    = (X_attack    - mean) / std

    # Split profiling → train (80%) / val_normal (20%)
    n        = len(X_profiling)
    n_train  = int(n * 0.8)
    idx      = np.random.permutation(n)
    X_train  = X_profiling[idx[:n_train]]
    X_val    = X_profiling[idx[n_train:]]

    print(f"  Train (normal)     : {X_train.shape}")
    print(f"  Val   (normal)     : {X_val.shape}")
    print(f"  Test  (attaque)    : {X_attack.shape}")
    print(f"  Normalisation      : mean={X_profiling.mean():.4f}  "
          f"std={X_profiling.std():.4f}")

    # Sauvegarde params normalisation
    np.save(os.path.join(OUT_PATH, "ae_mean.npy"), mean)
    np.save(os.path.join(OUT_PATH, "ae_std.npy"),  std)

    return X_train, X_val, X_attack, y_attack, meta_attack, mean, std


# ─────────────────────────────────────────────
# ARCHITECTURE : AUTOENCODER 1D
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    """
    Encoder progressif :
    [B, 1, 700] → [B, 128, 87] → [B, latent_dim]
    """
    def __init__(self, latent_dim=64):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Block 1 : 700 → 350
            nn.Conv1d(1,   32,  kernel_size=11, padding=5,  bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.AvgPool1d(2),

            # Block 2 : 350 → 175
            nn.Conv1d(32,  64,  kernel_size=7,  padding=3,  bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.AvgPool1d(2),

            # Block 3 : 175 → 87
            nn.Conv1d(64,  128, kernel_size=5,  padding=2,  bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.AvgPool1d(2),
        )

        # Bottleneck FC
        self.flatten    = nn.Flatten()
        self.bottleneck = nn.Sequential(
            nn.Linear(128 * 87, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),   # borne le latent space dans [-1, 1]
        )

    def forward(self, x):
        x = self.conv_layers(x)    # [B, 128, 87]
        x = self.flatten(x)        # [B, 128*87]
        z = self.bottleneck(x)     # [B, latent_dim]
        return z


class Decoder(nn.Module):
    """
    Decoder miroir de l'encoder :
    [B, latent_dim] → [B, 128, 87] → [B, 1, 700]
    """
    def __init__(self, latent_dim=64):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 87),
            nn.SiLU(),
        )

        self.deconv_layers = nn.Sequential(
            # Block 1 : 87 → 175
            nn.ConvTranspose1d(128, 64,  kernel_size=5,
                               stride=2, padding=2, output_padding=1,
                               bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Block 2 : 175 → 350
            nn.ConvTranspose1d(64,  32,  kernel_size=7,
                               stride=2, padding=3, output_padding=1,
                               bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            # Block 3 : 350 → 700
            nn.ConvTranspose1d(32,  1,   kernel_size=11,
                               stride=2, padding=5, output_padding=1),
        )

    def forward(self, z):
        import torch.nn.functional as F
        x = self.fc(z)                    # [B, 128*87]
        x = x.view(-1, 128, 87)           # [B, 128, 87]
        x = self.deconv_layers(x)         # [B, 1, ~700]
        if x.shape[-1] != 700:
            x = F.interpolate(x, size=700, mode='linear', align_corners=False)
        return x


class SideShieldAutoencoder(nn.Module):
    """
    Autoencoder 1D complet pour détection d'anomalies SCA
    -------------------------------------------------------
    Score d'anomalie = MSE(x, x_reconstructed)
    Seuil appris sur le percentile 95 des scores normaux
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder  = Encoder(latent_dim)
        self.decoder  = Decoder(latent_dim)
        self.threshold = None   # appris après entraînement
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z    = self.encoder(x)
        x_r  = self.decoder(z)
        return x_r, z

    def anomaly_score(self, x):
        """MSE par trace — score d'anomalie"""
        self.eval()
        with torch.no_grad():
            x_r, _ = self(x)
            scores  = ((x - x_r) ** 2).mean(dim=(1, 2))  # [B]
        return scores.cpu().numpy()

    def count_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ─────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────
def get_scheduler(optimizer, warmup=5, total=60):
    def fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        t = (ep - warmup) / max(1, total - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler):
    model.train()
    loss_sum = n = 0
    criterion = nn.MSELoss()
    for X in loader:
        if isinstance(X, (list, tuple)):
            X = X[0]
        X = X.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type):
            x_r, _ = model(X)
            loss    = criterion(x_r, X)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(X)
        n        += len(X)
    return loss_sum / n


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    loss_sum = n = 0
    criterion = nn.MSELoss()
    for X in loader:
        if isinstance(X, (list, tuple)):
            X = X[0]
        X = X.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type):
            x_r, _ = model(X)
            loss    = criterion(x_r, X)
        loss_sum += loss.item() * len(X)
        n        += len(X)
    return loss_sum / n


def train(model, tr_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, warmup=5, total=EPOCHS)
    scaler    = torch.amp.GradScaler()
    best_loss = float("inf")
    best_path = os.path.join(OUT_PATH, "sideshield_ae_best.pt")
    history   = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\n  {'─'*55}")
    print(f"  {'Ep':>4} {'Train MSE':>11} {'Val MSE':>10} "
          f"{'LR':>10} {'s':>5}")
    print(f"  {'─'*55}")

    for ep in range(1, EPOCHS + 1):
        t0      = time.time()
        tr_loss = train_epoch(model, tr_loader, optimizer, scaler)
        val_loss= eval_epoch(model, val_loader)
        scheduler.step()
        cur_lr  = optimizer.param_groups[0]['lr']

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)

        mark = ""
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"epoch": ep,
                        "model_state": model.state_dict(),
                        "best_loss": best_loss}, best_path)
            mark = " ✅"

        print(f"  {ep:>4} {tr_loss:>11.6f} {val_loss:>10.6f} "
              f"{cur_lr:>10.6f} {time.time()-t0:>4.1f}s{mark}")

    print(f"\n  Best val MSE : {best_loss:.6f}  →  {best_path}")
    return history, best_loss, best_path


# ─────────────────────────────────────────────
# SEUIL DE DÉTECTION
# ─────────────────────────────────────────────
@torch.no_grad()
def compute_threshold(model, val_loader, percentile=95):
    """
    Calcule le seuil de détection sur les traces normales de validation.
    Toute trace avec score > seuil est classée comme ATTAQUE.
    """
    model.eval()
    all_scores = []
    for X in val_loader:
        if isinstance(X, (list, tuple)):
            X = X[0]
        scores = model.anomaly_score(X.to(DEVICE))
        all_scores.extend(scores.tolist())

    all_scores = np.array(all_scores)
    threshold  = np.percentile(all_scores, percentile)

    print(f"\n  Scores normaux (val) :")
    print(f"    Mean   : {all_scores.mean():.6f}")
    print(f"    Std    : {all_scores.std():.6f}")
    print(f"    P95    : {threshold:.6f}  ← seuil de détection")
    print(f"    Max    : {all_scores.max():.6f}")

    return threshold, all_scores


# ─────────────────────────────────────────────
# ÉVALUATION COMPLÈTE
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, attack_loader, threshold):
    """
    Évalue les performances de détection :
    - Normal (val)  → label 0
    - Attaque (test)→ label 1
    """
    model.eval()

    # Scores normaux
    scores_normal = []
    for X in val_loader:
        if isinstance(X, (list, tuple)):
            X = X[0]
        scores_normal.extend(model.anomaly_score(X.to(DEVICE)).tolist())

    # Scores attaque
    scores_attack = []
    for X in attack_loader:
        if isinstance(X, (list, tuple)):
            X = X[0]
        scores_attack.extend(model.anomaly_score(X.to(DEVICE)).tolist())

    scores_normal = np.array(scores_normal)
    scores_attack = np.array(scores_attack)

    # Labels binaires
    y_true  = np.concatenate([
        np.zeros(len(scores_normal)),
        np.ones(len(scores_attack))
    ])
    y_score = np.concatenate([scores_normal, scores_attack])
    y_pred  = (y_score > threshold).astype(int)

    # Métriques
    auc  = roc_auc_score(y_true, y_score)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0   # Recall / Sensitivity
    tnr  = tn / (tn + fp) if (tn + fp) > 0 else 0   # Specificity
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0   # Precision
    f1   = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0
    acc  = (tp + tn) / len(y_true)

    print(f"\n  {'─'*50}")
    print(f"  RÉSULTATS DÉTECTION")
    print(f"  {'─'*50}")
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  Accuracy     : {acc*100:.2f}%")
    print(f"  Precision    : {prec*100:.2f}%")
    print(f"  Recall (TPR) : {tpr*100:.2f}%")
    print(f"  Specificity  : {tnr*100:.2f}%")
    print(f"  F1-Score     : {f1*100:.2f}%")
    print(f"  {'─'*50}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    fpr, tpr_curve, _ = roc_curve(y_true, y_score)

    return {
        "auc": auc, "acc": acc, "f1": f1,
        "precision": prec, "recall": tpr,
        "scores_normal": scores_normal,
        "scores_attack": scores_attack,
        "fpr": fpr, "tpr": tpr_curve,
        "threshold": threshold,
        "cm": cm
    }


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    eps = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax = axes[0]
    ax.set_facecolor("#161b22")
    ax.plot(eps, history["train_loss"], color="#ff6b6b",
            lw=1.5, label="Train MSE")
    ax.plot(eps, history["val_loss"],   color="#00d4ff",
            lw=1.5, ls="--", label="Val MSE")
    ax.set_title("Reconstruction Loss (MSE)",
                 color="white", fontsize=10)
    ax.set_xlabel("Epoch", color="#aaaaaa")
    ax.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    # LR
    ax = axes[1]
    ax.set_facecolor("#161b22")
    ax.plot(eps, history["lr"], color="#ffd700", lw=1.5)
    ax.set_title("Learning Rate", color="white", fontsize=10)
    ax.set_xlabel("Epoch", color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    fig.suptitle("SideShield Autoencoder — Training",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(DESKTOP, "sideshield_ae_training.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Training → {out}")
    plt.show()


def plot_results(results):
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.4, wspace=0.35)

    BG2 = "#161b22"
    C_N = "#00d4ff"    # normal
    C_A = "#ff6b6b"    # attaque
    C_T = "#ffd700"    # seuil

    def style(ax, title):
        ax.set_facecolor(BG2)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    sn = results["scores_normal"]
    sa = results["scores_attack"]
    th = results["threshold"]

    # ── Distribution des scores
    ax1 = fig.add_subplot(gs[0, :2])
    bins = np.linspace(min(sn.min(), sa.min()),
                       max(sn.max(), sa.max()), 80)
    ax1.hist(sn, bins=bins, color=C_N, alpha=0.7,
             label="Normal (profiling)", density=True)
    ax1.hist(sa, bins=bins, color=C_A, alpha=0.7,
             label="Attaque (test)",     density=True)
    ax1.axvline(th, color=C_T, lw=2, ls="--",
                label=f"Seuil P{ANOMALY_PCTL} = {th:.4f}")
    ax1.set_xlabel("Score d'anomalie (MSE)", color="#aaaaaa")
    ax1.set_ylabel("Densité", color="#aaaaaa")
    ax1.legend(fontsize=8, facecolor=BG2, labelcolor="white")
    style(ax1, "Distribution des scores d'anomalie")

    # ── ROC Curve
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(results["fpr"], results["tpr"],
             color=C_N, lw=2,
             label=f"AUC = {results['auc']:.4f}")
    ax2.plot([0,1],[0,1], color="#444", ls="--", lw=1)
    ax2.set_xlabel("FPR", color="#aaaaaa")
    ax2.set_ylabel("TPR", color="#aaaaaa")
    ax2.legend(fontsize=8, facecolor=BG2, labelcolor="white")
    style(ax2, "ROC Curve")

    # ── Scores normaux dans le temps
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sn[:500], color=C_N, lw=0.6, alpha=0.8)
    ax3.axhline(th, color=C_T, lw=1.5, ls="--", label="Seuil")
    ax3.set_xlabel("Trace index", color="#aaaaaa")
    ax3.set_ylabel("Score MSE", color="#aaaaaa")
    ax3.legend(fontsize=7, facecolor=BG2, labelcolor="white")
    style(ax3, "Scores normaux (500 premiers)")

    # ── Scores attaque dans le temps
    ax4 = fig.add_subplot(gs[1, 1])
    above = sa > th
    ax4.plot(sa[:500], color=C_A, lw=0.6, alpha=0.8)
    ax4.axhline(th, color=C_T, lw=1.5, ls="--", label="Seuil")
    ax4.fill_between(range(min(500, len(sa))),
                     th, np.where(sa[:500] > th, sa[:500], th),
                     alpha=0.3, color=C_A)
    ax4.set_xlabel("Trace index", color="#aaaaaa")
    ax4.set_ylabel("Score MSE", color="#aaaaaa")
    ax4.legend(fontsize=7, facecolor=BG2, labelcolor="white")
    style(ax4, "Scores attaque (500 premiers)")

    # ── Métriques
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(BG2)
    ax5.axis('off')
    metrics = [
        ("AUC-ROC",      f"{results['auc']:.4f}"),
        ("Accuracy",     f"{results['acc']*100:.2f}%"),
        ("Precision",    f"{results['precision']*100:.2f}%"),
        ("Recall",       f"{results['recall']*100:.2f}%"),
        ("F1-Score",     f"{results['f1']*100:.2f}%"),
        ("Seuil",        f"{th:.6f}"),
        ("Normal moy.",  f"{sn.mean():.6f}"),
        ("Attaque moy.", f"{sa.mean():.6f}"),
    ]
    for i, (label, val) in enumerate(metrics):
        y_pos = 0.9 - i * 0.11
        ax5.text(0.05, y_pos, label, color="#aaaaaa",
                 fontsize=9, transform=ax5.transAxes)
        color = "#39ff14" if "AUC" in label or "F1" in label \
                else "white"
        ax5.text(0.65, y_pos, val, color=color,
                 fontsize=9, fontweight="bold",
                 transform=ax5.transAxes)
    ax5.set_title("Métriques de détection",
                  color="white", fontsize=9, pad=6)
    for sp in ax5.spines.values():
        sp.set_edgecolor("#30363d")

    fig.suptitle("SideShield — Résultats Anomaly Detection",
                 color="white", fontsize=13, fontweight="bold")
    out = os.path.join(DESKTOP, "sideshield_ae_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Résultats → {out}")
    plt.show()


def plot_reconstruction(model, X_normal, X_attack, n=3):
    """Visualise reconstruction normale vs attaque"""
    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    fig.patch.set_facecolor("#0d1117")

    for i in range(n):
        for col, (X, label, color) in enumerate([
            (X_normal, "NORMAL",  "#00d4ff"),
            (X_attack, "ATTAQUE", "#ff6b6b"),
        ]):
            ax = axes[i, col]
            ax.set_facecolor("#161b22")
            x_t = torch.tensor(X[i:i+1], dtype=torch.float32
                                ).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                x_r, _ = model(x_t)
            orig  = x_t.cpu().squeeze().numpy()
            recon = x_r.cpu().squeeze().numpy()
            mse   = float(((orig - recon) ** 2).mean())

            ax.plot(orig,  color=color,    lw=0.8, alpha=0.8,
                    label="Original")
            ax.plot(recon, color="#ffd700", lw=0.8, alpha=0.8,
                    ls="--", label="Reconstruit")
            ax.fill_between(range(len(orig)),
                            orig, recon,
                            alpha=0.2, color="#ff9f43")
            ax.set_title(f"{label} — MSE={mse:.6f}",
                         color="white", fontsize=9)
            ax.legend(fontsize=7, facecolor="#161b22",
                      labelcolor="white")
            ax.tick_params(colors="#aaaaaa", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")

    fig.suptitle("SideShield — Reconstruction : Normal vs Attaque",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(DESKTOP, "sideshield_ae_reconstruction.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Reconstruction → {out}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  SideShield — Autoencoder 1D : Anomaly Detection")
    print("="*60)

    # ── Chargement
    print("\n  [1/5] Chargement et split des données...")
    (X_train, X_val, X_attack,
     y_attack, meta_attack, mean, std) = load_and_split(
        os.path.join(DB_PATH, "ASCAD.h5")
    )

    # ── DataLoaders
    tr_loader  = DataLoader(TraceDataset(X_train),
                            BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader = DataLoader(TraceDataset(X_val),
                            BATCH_SIZE, shuffle=False, num_workers=0)
    atk_loader = DataLoader(TraceDataset(X_attack),
                            BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Modèle
    print("\n  [2/5] Initialisation du modèle...")
    model = SideShieldAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    print(f"  Paramètres : {model.count_params():,}")
    print(f"  Latent dim : {LATENT_DIM}")

    # Vérif VRAM
    if DEVICE.type == "cuda":
        dummy = torch.randn(BATCH_SIZE, 1, TRACE_LEN).to(DEVICE)
        with torch.amp.autocast(device_type="cuda"):
            _ = model(dummy)
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM utilisée : {vram:.3f} GB")
        del dummy
        torch.cuda.empty_cache()

    # ── Entraînement
    print("\n  [3/5] Entraînement...")
    history, best_loss, best_path = train(model, tr_loader, val_loader)
    plot_training(history)

    # ── Chargement meilleur modèle
    print("\n  [4/5] Évaluation...")
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    # Seuil de détection
    threshold, scores_val = compute_threshold(
        model, val_loader, ANOMALY_PCTL
    )
    model.threshold = threshold

    # Évaluation complète
    results = evaluate(model, val_loader, atk_loader, threshold)

    # ── Visualisations
    print("\n  [5/5] Visualisations...")
    plot_results(results)
    plot_reconstruction(model, X_val, X_attack, n=3)

    # ── Sauvegarde finale
    torch.save({
        "model_state" : model.state_dict(),
        "threshold"   : threshold,
        "latent_dim"  : LATENT_DIM,
        "norm_mean"   : mean,
        "norm_std"    : std,
        "results"     : {k: v for k, v in results.items()
                         if not isinstance(v, np.ndarray)},
    }, os.path.join(OUT_PATH, "sideshield_ae_final.pt"))
    print(f"\n  Modèle final → {OUT_PATH}/sideshield_ae_final.pt")

    # ── Résumé
    print("\n" + "="*60)
    print("  RÉSUMÉ FINAL")
    print("="*60)
    print(f"  AUC-ROC   : {results['auc']:.4f}")
    print(f"  Accuracy  : {results['acc']*100:.2f}%")
    print(f"  F1-Score  : {results['f1']*100:.2f}%")
    print(f"  Recall    : {results['recall']*100:.2f}%")
    print(f"  Seuil     : {threshold:.6f}")
    print(f"\n  Interprétation AUC :")
    if results['auc'] >= 0.90:
        print("  ✅ Excellent — détection très fiable")
    elif results['auc'] >= 0.75:
        print("  ✅ Bon — détection fiable")
    elif results['auc'] >= 0.60:
        print("  ⚠️  Modéré — amélioration possible")
    else:
        print("  ❌ Faible — revoir architecture")
    print("="*60 + "\n")