"""
SideShield — Anomaly Detection avec Simulation d'Attaques
===========================================================
Principe :
  - Entraînement sur traces normales (profiling ASCAD)
  - Simulation de 4 types d'anomalies physiques réalistes :
      1. Gaussian Noise    → brouillage EM actif
      2. Desynchronization → timing attack / jitter
      3. Amplitude Scaling → fault injection (voltage glitch)
      4. Spike Injection   → electromagnetic fault injection (EMFI)
  - Autoencoder 1D détecte les anomalies par erreur de reconstruction

Justification physique :
  Un attaquant qui tente une SCA perturbe l'environnement EM
  du chip. Ces perturbations créent des anomalies détectables
  dans le profil de consommation — c'est exactement ce que
  SideShield apprend à reconnaître.
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

TRACE_LEN    = 700
BATCH_SIZE   = 256
EPOCHS       = 60
LR           = 1e-3
LATENT_DIM   = 64
ANOMALY_PCTL = 95
SEED         = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ─────────────────────────────────────────────
# SIMULATION D'ATTAQUES PHYSIQUES
# ─────────────────────────────────────────────
def simulate_attacks(X_normal, attack_ratio=0.25):
    """
    Génère 4 types d'anomalies physiques réalistes.

    Chaque type simule un vecteur d'attaque réel :

    1. GAUSSIAN NOISE — brouillage EM actif
       Un attaquant injecte du bruit EM pour perturber
       les mesures de protection → SNR dégradé

    2. DESYNCHRONIZATION — timing attack / clock jitter
       L'attaquant manipule l'horloge du chip pour
       désynchroniser les traces → shift temporel

    3. AMPLITUDE SCALING — voltage glitch (fault injection)
       Injection d'une perturbation de tension → les pics
       de consommation sont amplifiés ou réduits anormalement

    4. SPIKE INJECTION — EMFI (EM Fault Injection)
       Un pulse EM localisé crée un spike dans la trace
       à un moment précis → anomalie locale intense

    Retourne X_attack et y (0=normal, 1=anomalie)
    """
    n        = len(X_normal)
    n_each   = int(n * attack_ratio / 4)  # ~25% par type
    std_norm = X_normal.std()             # std globale pour calibrer

    attacks = []
    labels  = []

    # ── Type 1 : Gaussian Noise (brouillage EM)
    idx = np.random.choice(n, n_each, replace=False)
    X_noisy = X_normal[idx].copy()
    noise_level = np.random.uniform(0.5, 1.5, n_each)
    for i, nl in enumerate(noise_level):
        X_noisy[i] += np.random.normal(0, nl * std_norm, TRACE_LEN)
    attacks.append(X_noisy)
    labels.extend([1] * n_each)

    # ── Type 2 : Desynchronization (timing attack)
    idx = np.random.choice(n, n_each, replace=False)
    X_desync = X_normal[idx].copy()
    for i in range(n_each):
        shift = np.random.randint(10, 80)   # shift entre 10 et 80 points
        X_desync[i] = np.roll(X_desync[i], shift)
        # Brouille les bords introduits par le roll
        X_desync[i, :shift] = np.random.normal(
            0, 0.3 * std_norm, shift)
    attacks.append(X_desync)
    labels.extend([2] * n_each)

    # ── Type 3 : Amplitude Scaling (voltage glitch)
    idx = np.random.choice(n, n_each, replace=False)
    X_scaled = X_normal[idx].copy()
    for i in range(n_each):
        # Glitch sur une fenêtre aléatoire
        start  = np.random.randint(0, TRACE_LEN - 100)
        length = np.random.randint(20, 100)
        scale  = np.random.uniform(1.8, 3.5)   # amplification anormale
        X_scaled[i, start:start+length] *= scale
    attacks.append(X_scaled)
    labels.extend([3] * n_each)

    # ── Type 4 : Spike Injection (EMFI)
    idx = np.random.choice(n, n_each, replace=False)
    X_spike = X_normal[idx].copy()
    for i in range(n_each):
        n_spikes = np.random.randint(1, 5)
        for _ in range(n_spikes):
            pos       = np.random.randint(5, TRACE_LEN - 5)
            amplitude = np.random.uniform(3.0, 8.0) * std_norm
            width     = np.random.randint(2, 8)
            spike     = amplitude * np.exp(
                -0.5 * ((np.arange(TRACE_LEN) - pos) / width) ** 2)
            X_spike[i] += spike
    attacks.append(X_spike)
    labels.extend([4] * n_each)

    X_attack = np.vstack(attacks).astype(np.float32)
    y_attack = np.array(labels, dtype=np.int64)

    # Mélange
    perm     = np.random.permutation(len(X_attack))
    X_attack = X_attack[perm]
    y_attack = y_attack[perm]

    print(f"\n  Attaques simulées :")
    print(f"    Type 1 — Gaussian Noise    : {n_each} traces")
    print(f"    Type 2 — Desynchronization : {n_each} traces")
    print(f"    Type 3 — Amplitude Scaling : {n_each} traces")
    print(f"    Type 4 — Spike Injection   : {n_each} traces")
    print(f"    Total attaques             : {len(X_attack)} traces")

    return X_attack, y_attack


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class TraceDataset(Dataset):
    def __init__(self, traces, labels=None):
        self.X = torch.tensor(traces, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.long) \
                 if labels is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
def load_data(filepath):
    with h5py.File(filepath, "r") as f:
        X_all = f["Profiling_traces/traces"][:].astype(np.float32)

    # Z-score
    mean = X_all.mean(axis=0)
    std  = np.where(X_all.std(axis=0) == 0, 1e-8, X_all.std(axis=0))
    X_all = (X_all - mean) / std

    # Split : 70% train / 15% val / 15% test_normal
    n      = len(X_all)
    idx    = np.random.permutation(n)
    n_tr   = int(n * 0.70)
    n_val  = int(n * 0.15)

    X_train      = X_all[idx[:n_tr]]
    X_val        = X_all[idx[n_tr:n_tr+n_val]]
    X_test_norm  = X_all[idx[n_tr+n_val:]]

    print(f"  Train (normal)      : {X_train.shape}")
    print(f"  Val   (normal)      : {X_val.shape}")
    print(f"  Test  (normal ref)  : {X_test_norm.shape}")

    np.save(os.path.join(OUT_PATH, "ae_mean.npy"), mean)
    np.save(os.path.join(OUT_PATH, "ae_std.npy"),  std)

    return X_train, X_val, X_test_norm, mean, std


# ─────────────────────────────────────────────
# ARCHITECTURE AUTOENCODER 1D
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,   32,  11, padding=5,  bias=False),
            nn.BatchNorm1d(32), nn.SiLU(), nn.AvgPool1d(2),   # 350

            nn.Conv1d(32,  64,  7,  padding=3,  bias=False),
            nn.BatchNorm1d(64), nn.SiLU(), nn.AvgPool1d(2),   # 175

            nn.Conv1d(64,  128, 5,  padding=2,  bias=False),
            nn.BatchNorm1d(128), nn.SiLU(), nn.AvgPool1d(2),  # 87
        )
        self.flat = nn.Flatten()
        self.fc   = nn.Sequential(
            nn.Linear(128 * 87, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(self.flat(self.conv(x)))


class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 87), nn.SiLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.BatchNorm1d(64), nn.SiLU(),

            nn.ConvTranspose1d(64, 32, 7, stride=2,
                               padding=3, output_padding=1, bias=False),
            nn.BatchNorm1d(32), nn.SiLU(),

            nn.ConvTranspose1d(32, 1, 11, stride=2,
                               padding=5, output_padding=1),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 87)
        x = self.deconv(x)
        if x.shape[-1] != TRACE_LEN:
            x = F.interpolate(x, size=TRACE_LEN,
                              mode='linear', align_corners=False)
        return x


class SideShieldAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder   = Encoder(latent_dim)
        self.decoder   = Decoder(latent_dim)
        self.threshold = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        z   = self.encoder(x)
        x_r = self.decoder(z)
        return x_r, z

    @torch.no_grad()
    def score(self, x):
        self.eval()
        x_r, _ = self(x)
        return ((x - x_r) ** 2).mean(dim=(1, 2)).cpu().numpy()

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
def run_epoch(model, loader, optimizer, scaler, train=True):
    model.train() if train else model.eval()
    total_loss = n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            X = batch[0] if isinstance(batch, (list, tuple)) else batch
            X = X.to(DEVICE)
            if train:
                optimizer.zero_grad()
            with torch.amp.autocast(device_type=DEVICE.type):
                x_r, _ = model(X)
                loss    = F.mse_loss(x_r, X)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(X)
            n          += len(X)
    return total_loss / n


def train_model(model, tr_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=1e-5)
    scheduler = get_scheduler(optimizer, 5, EPOCHS)
    scaler    = torch.amp.GradScaler()
    best_loss = float("inf")
    best_path = os.path.join(OUT_PATH, "sideshield_ae_best.pt")
    history   = {"train": [], "val": [], "lr": []}

    print(f"\n  {'─'*55}")
    print(f"  {'Ep':>4} {'Train MSE':>11} {'Val MSE':>10} "
          f"{'LR':>10} {'s':>5}")
    print(f"  {'─'*55}")

    for ep in range(1, EPOCHS + 1):
        t0  = time.time()
        trl = run_epoch(model, tr_loader, optimizer, scaler, train=True)
        vl  = run_epoch(model, val_loader, optimizer, scaler, train=False)
        scheduler.step()
        lr  = optimizer.param_groups[0]['lr']
        history["train"].append(trl)
        history["val"].append(vl)
        history["lr"].append(lr)

        mark = ""
        if vl < best_loss:
            best_loss = vl
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "best_loss": best_loss}, best_path)
            mark = " ✅"
        print(f"  {ep:>4} {trl:>11.6f} {vl:>10.6f} "
              f"{lr:>10.6f} {time.time()-t0:>4.1f}s{mark}")

    print(f"\n  Best val MSE : {best_loss:.6f}")
    return history, best_path


# ─────────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────────
@torch.no_grad()
def get_scores(model, loader):
    model.eval()
    all_scores = []
    for batch in loader:
        X = batch[0] if isinstance(batch, (list, tuple)) else batch
        all_scores.extend(model.score(X.to(DEVICE)).tolist())
    return np.array(all_scores)


def evaluate(model, val_loader, atk_loader, atk_types):
    # Scores
    s_normal = get_scores(model, val_loader)
    s_attack = get_scores(model, atk_loader)

    # Seuil
    threshold = np.percentile(s_normal, ANOMALY_PCTL)

    print(f"\n  Scores normaux  : mean={s_normal.mean():.6f}  "
          f"std={s_normal.std():.6f}  P95={threshold:.6f}")
    print(f"  Scores attaque  : mean={s_attack.mean():.6f}  "
          f"std={s_attack.std():.6f}")

    # Métriques binaires
    y_true  = np.concatenate([np.zeros(len(s_normal)),
                               np.ones(len(s_attack))])
    y_score = np.concatenate([s_normal, s_attack])
    y_pred  = (y_score > threshold).astype(int)

    auc       = roc_auc_score(y_true, y_score)
    cm        = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp = cm.ravel()
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    acc       = (tp + tn) / len(y_true)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    print(f"\n  {'─'*48}")
    print(f"  RÉSULTATS DÉTECTION")
    print(f"  {'─'*48}")
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  Accuracy     : {acc*100:.2f}%")
    print(f"  Precision    : {precision*100:.2f}%")
    print(f"  Recall (TPR) : {recall*100:.2f}%")
    print(f"  F1-Score     : {f1*100:.2f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # Par type d'attaque
    print(f"\n  Détection par type d'attaque :")
    type_names = {1:"Gaussian Noise", 2:"Desync",
                  3:"Amplitude Scale", 4:"Spike EMFI"}
    for t in [1, 2, 3, 4]:
        mask   = atk_types == t
        s_t    = s_attack[mask]
        det    = (s_t > threshold).mean() * 100
        print(f"    Type {t} ({type_names[t]:18s}) : "
              f"{det:.1f}% détectées  "
              f"(mean MSE={s_t.mean():.6f})")

    return {
        "auc": auc, "acc": acc, "f1": f1,
        "precision": precision, "recall": recall,
        "s_normal": s_normal, "s_attack": s_attack,
        "fpr": fpr, "tpr": tpr,
        "threshold": threshold, "cm": cm,
        "atk_types": atk_types,
    }


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_attacks(X_normal, X_attack, atk_types):
    """Visualise les 4 types d'attaques vs trace normale"""
    type_names = {1:"Gaussian Noise (Brouillage EM)",
                  2:"Desynchronization (Timing Attack)",
                  3:"Amplitude Scaling (Voltage Glitch)",
                  4:"Spike Injection (EMFI)"}
    colors = ["#ff6b6b","#ffd700","#ff9f43","#ff4ecb"]

    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")

    normal_ex = X_normal[0]
    for row, t in enumerate([1, 2, 3, 4]):
        ax    = axes[row]
        ax.set_facecolor("#161b22")
        idx_t = np.where(atk_types == t)[0][0]
        atk_ex = X_attack[idx_t]

        ax.plot(normal_ex, color="#00d4ff", lw=0.8,
                alpha=0.7, label="Normale")
        ax.plot(atk_ex, color=colors[row], lw=0.8,
                alpha=0.8, label=type_names[t])
        ax.fill_between(range(TRACE_LEN),
                        normal_ex, atk_ex,
                        alpha=0.15, color=colors[row])
        ax.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    fig.suptitle("SideShield — Types d'Attaques Simulées",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(DESKTOP, "sideshield_attack_types.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Types d'attaques → {out}")
    plt.show()


def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    eps = range(1, len(history["train"]) + 1)

    for ax, (k1, k2, title) in zip(axes, [
        ("train", "val", "Reconstruction Loss (MSE)"),
        ("lr",    None,  "Learning Rate"),
    ]):
        ax.set_facecolor("#161b22")
        ax.plot(eps, history[k1], color="#ff6b6b", lw=1.5, label="Train")
        if k2:
            ax.plot(eps, history[k2], color="#00d4ff",
                    lw=1.5, ls="--", label="Val")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("Epoch", color="#aaaaaa")
        ax.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    fig.suptitle("SideShield — Training",
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
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    BG2 = "#161b22"
    sn  = results["s_normal"]
    sa  = results["s_attack"]
    th  = results["threshold"]
    at  = results["atk_types"]

    def style(ax, title):
        ax.set_facecolor(BG2)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    # ── Distribution globale
    ax1 = fig.add_subplot(gs[0, :2])
    bins = np.linspace(min(sn.min(), sa.min()),
                       max(sn.max(), sa.max()), 100)
    ax1.hist(sn, bins=bins, color="#00d4ff", alpha=0.6,
             label="Normal", density=True)
    ax1.hist(sa, bins=bins, color="#ff6b6b", alpha=0.6,
             label="Attaque", density=True)
    ax1.axvline(th, color="#ffd700", lw=2, ls="--",
                label=f"Seuil P{ANOMALY_PCTL}={th:.4f}")
    ax1.set_xlabel("Score MSE", color="#aaaaaa")
    ax1.set_ylabel("Densité", color="#aaaaaa")
    ax1.legend(fontsize=8, facecolor=BG2, labelcolor="white")
    style(ax1, "Distribution globale des scores d'anomalie")

    # ── ROC
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(results["fpr"], results["tpr"], color="#00d4ff",
             lw=2, label=f"AUC={results['auc']:.4f}")
    ax2.plot([0,1],[0,1], "#444", ls="--", lw=1)
    ax2.set_xlabel("FPR", color="#aaaaaa")
    ax2.set_ylabel("TPR", color="#aaaaaa")
    ax2.legend(fontsize=9, facecolor=BG2, labelcolor="white")
    style(ax2, "ROC Curve")

    # ── Par type d'attaque
    ax3 = fig.add_subplot(gs[1, :2])
    type_names  = {1:"Gaussian\nNoise", 2:"Desync",
                   3:"Amplitude\nScale", 4:"Spike\nEMFI"}
    type_colors = ["#ff6b6b","#ffd700","#ff9f43","#ff4ecb"]
    x_pos = 0
    for t, (name, color) in enumerate(
            zip(type_names.values(), type_colors), 1):
        mask = at == t
        s_t  = sa[mask]
        ax3.boxplot(s_t, positions=[x_pos], widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.7),
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(color="#aaaaaa"),
                    capprops=dict(color="#aaaaaa"),
                    flierprops=dict(marker='.', color=color,
                                    alpha=0.3, ms=2))
        x_pos += 1

    ax3.boxplot(sn, positions=[x_pos], widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor="#00d4ff", alpha=0.7),
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(color="#aaaaaa"),
                capprops=dict(color="#aaaaaa"),
                flierprops=dict(marker='.', color="#00d4ff",
                                alpha=0.3, ms=2))
    ax3.axhline(th, color="#ffd700", lw=1.5, ls="--",
                label="Seuil")
    ax3.set_xticks(range(5))
    ax3.set_xticklabels([*type_names.values(), "Normal"],
                        color="white", fontsize=8)
    ax3.set_ylabel("Score MSE", color="#aaaaaa")
    ax3.legend(fontsize=8, facecolor=BG2, labelcolor="white")
    style(ax3, "Distribution MSE par type d'attaque")

    # ── Métriques
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    ax4.set_facecolor(BG2)
    metrics = [
        ("AUC-ROC",    f"{results['auc']:.4f}"),
        ("Accuracy",   f"{results['acc']*100:.2f}%"),
        ("Precision",  f"{results['precision']*100:.2f}%"),
        ("Recall",     f"{results['recall']*100:.2f}%"),
        ("F1-Score",   f"{results['f1']*100:.2f}%"),
        ("Seuil P95",  f"{th:.6f}"),
        ("Normal moy.",f"{sn.mean():.6f}"),
        ("Atk moy.",   f"{sa.mean():.6f}"),
        ("Séparation", f"{(sa.mean()-sn.mean())/sn.std():.2f}σ"),
    ]
    for i, (label, val) in enumerate(metrics):
        y = 0.92 - i * 0.10
        ax4.text(0.05, y, label, color="#aaaaaa", fontsize=9,
                 transform=ax4.transAxes)
        clr = "#39ff14" if i < 2 else "white"
        ax4.text(0.62, y, val, color=clr, fontsize=9,
                 fontweight="bold", transform=ax4.transAxes)
    style(ax4, "Métriques")
    for sp in ax4.spines.values():
        sp.set_edgecolor("#30363d")

    fig.suptitle("SideShield — Résultats Anomaly Detection",
                 color="white", fontsize=13, fontweight="bold")
    out = os.path.join(DESKTOP, "sideshield_ae_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Résultats → {out}")
    plt.show()


def plot_reconstruction(model, X_normal, X_attack, atk_types):
    type_names = {1:"Gaussian Noise", 2:"Desync",
                  3:"Amplitude Scale", 4:"Spike EMFI"}
    colors     = {1:"#ff6b6b", 2:"#ffd700",
                  3:"#ff9f43", 4:"#ff4ecb"}

    fig, axes = plt.subplots(5, 1, figsize=(16, 14))
    fig.patch.set_facecolor("#0d1117")

    samples = [(X_normal[0], "NORMALE", "#00d4ff", 0)]
    for t in [1, 2, 3, 4]:
        idx = np.where(atk_types == t)[0][0]
        samples.append((X_attack[idx],
                        type_names[t], colors[t], t))

    threshold = model.threshold

    for ax, (x, label, color, t) in zip(axes, samples):
        ax.set_facecolor("#161b22")
        x_t = torch.tensor(x[None, None, :],
                           dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            x_r, _ = model(x_t)
        orig  = x
        recon = x_r.cpu().squeeze().numpy()
        mse   = float(((orig - recon) ** 2).mean())

        ax.plot(orig,  color=color,    lw=0.8, alpha=0.8,
                label="Original")
        ax.plot(recon, color="#ffffff", lw=0.8, alpha=0.7,
                ls="--", label="Reconstruit")
        ax.fill_between(range(TRACE_LEN), orig, recon,
                        alpha=0.2, color="#ff9f43")

        status = ""
        if threshold is not None:
            status = ("🚨 ATTAQUE DÉTECTÉE" if mse > threshold
                      else "✅ NORMAL")
        ax.set_title(
            f"{label}  |  MSE={mse:.6f}  {status}",
            color="white", fontsize=9)
        ax.legend(fontsize=7, facecolor="#161b22",
                  labelcolor="white")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    fig.suptitle("SideShield — Reconstruction : Normal vs Attaques",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(DESKTOP, "sideshield_reconstruction.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Reconstruction → {out}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  SideShield — Anomaly Detection (Attaques Simulées)")
    print("="*60)

    # ── Chargement
    print("\n  [1/6] Chargement des données...")
    X_train, X_val, X_test_norm, mean, std = load_data(
        os.path.join(DB_PATH, "ASCAD.h5")
    )

    # ── Simulation des attaques
    print("\n  [2/6] Simulation des attaques physiques...")
    X_attack, y_attack = simulate_attacks(
        X_test_norm, attack_ratio=1.0   # autant d'attaques que de normales
    )

    # Visualiser les types d'attaques
    plot_attacks(X_test_norm, X_attack, y_attack)

    # ── DataLoaders
    tr_loader  = DataLoader(TraceDataset(X_train),
                            BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader = DataLoader(TraceDataset(X_val),
                            BATCH_SIZE, shuffle=False, num_workers=0)
    atk_loader = DataLoader(TraceDataset(X_attack, y_attack),
                            BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Modèle
    print("\n  [3/6] Initialisation autoencoder...")
    model = SideShieldAE(latent_dim=LATENT_DIM).to(DEVICE)
    print(f"  Paramètres : {model.count_params():,}")

    if DEVICE.type == "cuda":
        dummy = torch.randn(BATCH_SIZE, 1, TRACE_LEN).to(DEVICE)
        with torch.amp.autocast(device_type="cuda"):
            _ = model(dummy)
        print(f"  VRAM utilisée : "
              f"{torch.cuda.memory_allocated()/1e9:.3f} GB")
        del dummy; torch.cuda.empty_cache()

    # ── Entraînement
    print("\n  [4/6] Entraînement sur traces normales...")
    history, best_path = train_model(model, tr_loader, val_loader)
    plot_training(history)

    # ── Évaluation
    print("\n  [5/6] Évaluation...")
    ckpt = torch.load(best_path, map_location=DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Seuil sur validation normale
    s_val     = get_scores(model, val_loader)
    threshold = np.percentile(s_val, ANOMALY_PCTL)
    model.threshold = threshold
    print(f"  Seuil P{ANOMALY_PCTL} : {threshold:.6f}")

    results = evaluate(model, val_loader, atk_loader, y_attack)

    # ── Visualisations
    print("\n  [6/6] Visualisations...")
    plot_results(results)
    plot_reconstruction(model, X_test_norm, X_attack, y_attack)

    # ── Sauvegarde
    torch.save({
        "model_state": model.state_dict(),
        "threshold"  : threshold,
        "latent_dim" : LATENT_DIM,
        "norm_mean"  : mean,
        "norm_std"   : std,
        "auc"        : results["auc"],
        "f1"         : results["f1"],
    }, os.path.join(OUT_PATH, "sideshield_ae_final.pt"))

    # ── Résumé
    print("\n" + "="*60)
    print("  RÉSUMÉ FINAL")
    print("="*60)
    print(f"  AUC-ROC   : {results['auc']:.4f}")
    print(f"  Accuracy  : {results['acc']*100:.2f}%")
    print(f"  F1-Score  : {results['f1']*100:.2f}%")
    print(f"  Recall    : {results['recall']*100:.2f}%")
    sep = (results['s_attack'].mean() - results['s_normal'].mean()) \
          / results['s_normal'].std()
    print(f"  Séparation: {sep:.2f}σ")
    print()
    if results["auc"] >= 0.90:
        print("  ✅ Excellent — modèle très performant")
    elif results["auc"] >= 0.75:
        print("  ✅ Bon — détection fiable")
    elif results["auc"] >= 0.60:
        print("  ⚠️  Modéré — amélioration possible")
    else:
        print("  ❌ Faible — revoir les paramètres d'attaque")
    print("="*60 + "\n")
