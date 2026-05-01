"""
SideShield — Phase 2 : Preprocessing Pipeline
===============================================
Normalisation, sélection POI, DataLoader PyTorch
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
DB_PATH = os.path.join(DESKTOP, "ASCAD", "ASCAD_databases")
OUT_PATH = os.path.join(DESKTOP, "SideShield")
os.makedirs(OUT_PATH, exist_ok=True)

TARGET_BYTE = 2          # byte de clé ciblé (index 2 — standard ASCAD)
N_POI       = 100        # nombre de points d'intérêt à sélectionner
BATCH_SIZE  = 256
SEED        = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────
def load_ascad(filepath):
    with h5py.File(filepath, "r") as f:
        X_train   = f["Profiling_traces/traces"][:].astype(np.float32)
        y_train   = f["Profiling_traces/labels"][:]
        meta_tr   = f["Profiling_traces/metadata"][:]

        X_test    = f["Attack_traces/traces"][:].astype(np.float32)
        y_test    = f["Attack_traces/labels"][:]
        meta_te   = f["Attack_traces/metadata"][:]

    return X_train, y_train, meta_tr, X_test, y_test, meta_te


# ─────────────────────────────────────────────
# 2. NORMALISATION
# ─────────────────────────────────────────────
def normalize(X_train, X_test, method="zscore"):
    """
    Trois méthodes disponibles :
    - zscore   : (x - mean) / std         → standard pour CNN
    - minmax   : (x - min) / (max - min)  → valeurs dans [0, 1]
    - robust   : (x - median) / IQR       → résistant aux outliers
    """
    if method == "zscore":
        mean = X_train.mean(axis=0)   # moyenne par point temporel
        std  = X_train.std(axis=0)
        std  = np.where(std == 0, 1e-8, std)   # éviter division par 0
        X_train_n = (X_train - mean) / std
        X_test_n  = (X_test  - mean) / std     # même params que train !

    elif method == "minmax":
        mn = X_train.min(axis=0)
        mx = X_train.max(axis=0)
        rng = np.where((mx - mn) == 0, 1e-8, mx - mn)
        X_train_n = (X_train - mn) / rng
        X_test_n  = (X_test  - mn) / rng

    elif method == "robust":
        median = np.median(X_train, axis=0)
        q75    = np.percentile(X_train, 75, axis=0)
        q25    = np.percentile(X_train, 25, axis=0)
        iqr    = np.where((q75 - q25) == 0, 1e-8, q75 - q25)
        X_train_n = (X_train - median) / iqr
        X_test_n  = (X_test  - median) / iqr

    print(f"  Normalisation [{method}] :")
    print(f"    Train — mean={X_train_n.mean():.4f}  std={X_train_n.std():.4f}"
          f"  min={X_train_n.min():.2f}  max={X_train_n.max():.2f}")
    print(f"    Test  — mean={X_test_n.mean():.4f}  std={X_test_n.std():.4f}"
          f"  min={X_test_n.min():.2f}  max={X_test_n.max():.2f}")

    return X_train_n, X_test_n


# ─────────────────────────────────────────────
# 3. SÉLECTION DES POI (Points of Interest)
# ─────────────────────────────────────────────
def select_poi(X_train, y_train, n_poi=100, method="variance"):
    """
    Deux méthodes :
    - variance : points avec la plus haute variance globale
    - snr      : Signal-to-Noise Ratio par classe (plus précis)
    """
    if method == "variance":
        scores = X_train.var(axis=0)

    elif method == "snr":
        # SNR = variance(signal) / variance(bruit)
        # signal = moyenne par classe, bruit = variance intra-classe
        classes = np.unique(y_train)
        class_means = np.array([X_train[y_train == c].mean(axis=0)
                                for c in classes])
        class_vars  = np.array([X_train[y_train == c].var(axis=0)
                                for c in classes])
        signal_var  = class_means.var(axis=0)
        noise_var   = class_vars.mean(axis=0)
        noise_var   = np.where(noise_var == 0, 1e-8, noise_var)
        scores      = signal_var / noise_var

    poi_indices = np.argsort(scores)[-n_poi:]
    poi_indices = np.sort(poi_indices)   # ordre chronologique

    print(f"\n  Sélection POI [{method}] : {n_poi} points")
    print(f"    Indices POI (top 5 scores) : "
          f"{np.argsort(scores)[-5:][::-1].tolist()}")

    return poi_indices, scores


# ─────────────────────────────────────────────
# 4. HAMMING WEIGHT (pour analyse)
# ─────────────────────────────────────────────
# S-Box AES standard
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

def hamming_weight(x):
    return bin(int(x)).count('1')

HW = np.array([hamming_weight(SBOX[i]) for i in range(256)])


# ─────────────────────────────────────────────
# 5. DATASET PYTORCH
# ─────────────────────────────────────────────
class ASCADDataset(Dataset):
    def __init__(self, traces, labels, poi_indices=None):
        if poi_indices is not None:
            traces = traces[:, poi_indices]
        # Shape pour CNN 1D : (N, 1, L)
        self.X = torch.tensor(traces, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(X_train, y_train, X_test, y_test,
                      poi_indices, batch_size=256):
    train_ds = ASCADDataset(X_train, y_train, poi_indices)
    test_ds  = ASCADDataset(X_test,  y_test,  poi_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"\n  DataLoaders construits :")
    print(f"    Train : {len(train_ds)} samples  → {len(train_loader)} batches")
    print(f"    Test  : {len(test_ds)} samples   → {len(test_loader)} batches")
    print(f"    Shape batch : {next(iter(train_loader))[0].shape}")

    return train_loader, test_loader


# ─────────────────────────────────────────────
# 6. VISUALISATION PREPROCESSING
# ─────────────────────────────────────────────
def plot_preprocessing(X_raw, X_norm, scores_var, scores_snr,
                       poi_var, poi_snr, y_train, name="desync0"):

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.3)

    BG2   = "#161b22"
    C1    = "#00d4ff"
    C2    = "#ffd700"
    C3    = "#39ff14"
    C4    = "#ff6b6b"
    C5    = "#ff9f43"

    def style_ax(ax, title):
        ax.set_facecolor(BG2)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # ── 6.1 Avant / Après normalisation (trace unique)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X_raw[0],  color=C4, linewidth=0.8, label="Brut (int8)", alpha=0.8)
    style_ax(ax1, "Trace brute — avant normalisation")
    ax1.set_ylabel("Amplitude", color="#aaaaaa", fontsize=8)
    ax1.legend(fontsize=7, facecolor=BG2, labelcolor="white")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(X_norm[0], color=C1, linewidth=0.8, label="Normalisé (z-score)", alpha=0.8)
    style_ax(ax2, "Trace normalisée — z-score")
    ax2.set_ylabel("Amplitude", color="#aaaaaa", fontsize=8)
    ax2.legend(fontsize=7, facecolor=BG2, labelcolor="white")

    # ── 6.2 Score Variance avec POI
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(scores_var, color=C5, linewidth=0.7)
    ax3.scatter(poi_var, scores_var[poi_var], color=C4,
                s=8, zorder=5, label=f"POI sélectionnés (n={len(poi_var)})")
    ax3.set_xlabel("Points temporels", color="#aaaaaa", fontsize=8)
    ax3.set_ylabel("Variance", color="#aaaaaa", fontsize=8)
    ax3.legend(fontsize=7, facecolor=BG2, labelcolor="white")
    style_ax(ax3, "Méthode Variance — sélection POI")

    # ── 6.3 Score SNR avec POI
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(scores_snr, color=C3, linewidth=0.7)
    ax4.scatter(poi_snr, scores_snr[poi_snr], color=C4,
                s=8, zorder=5, label=f"POI sélectionnés (n={len(poi_snr)})")
    ax4.set_xlabel("Points temporels", color="#aaaaaa", fontsize=8)
    ax4.set_ylabel("SNR", color="#aaaaaa", fontsize=8)
    ax4.legend(fontsize=7, facecolor=BG2, labelcolor="white")
    style_ax(ax4, "Méthode SNR — sélection POI")

    # ── 6.4 Trace avec POI surlignés
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(X_norm[0], color=C1, linewidth=0.7, alpha=0.7, label="Trace normalisée")
    for p in poi_snr:
        ax5.axvline(x=p, color=C4, linewidth=0.4, alpha=0.4)
    ax5.scatter(poi_snr, X_norm[0][poi_snr], color=C4,
                s=10, zorder=5, label="POI (SNR)")
    ax5.set_xlabel("Points temporels", color="#aaaaaa", fontsize=8)
    ax5.set_ylabel("Amplitude normalisée", color="#aaaaaa", fontsize=8)
    ax5.legend(fontsize=7, facecolor=BG2, labelcolor="white")
    style_ax(ax5, "Trace normalisée avec Points d'Intérêt surlignés (SNR)")

    # ── 6.5 Distribution HW des labels
    ax6 = fig.add_subplot(gs[3, 0])
    hw_labels = HW[y_train]
    hw_counts = np.bincount(hw_labels, minlength=9)
    bars = ax6.bar(range(9), hw_counts, color=C2, alpha=0.85, width=0.7)
    ax6.set_xlabel("Hamming Weight", color="#aaaaaa", fontsize=8)
    ax6.set_ylabel("Fréquence", color="#aaaaaa", fontsize=8)
    ax6.set_xticks(range(9))
    for bar, count in zip(bars, hw_counts):
        if count > 0:
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     str(count), ha='center', va='bottom',
                     color='white', fontsize=6)
    style_ax(ax6, "Distribution Hamming Weight des labels")

    # ── 6.6 Traces groupées par HW
    ax7 = fig.add_subplot(gs[3, 1])
    colors_hw = plt.cm.plasma(np.linspace(0.1, 0.9, 9))
    for hw_val in range(9):
        idx = np.where(hw_labels == hw_val)[0]
        if len(idx) > 0:
            mean_trace = X_norm[idx[:50]].mean(axis=0)
            ax7.plot(mean_trace[poi_snr], color=colors_hw[hw_val],
                     linewidth=0.8, alpha=0.8, label=f"HW={hw_val}")
    ax7.set_xlabel("POI index", color="#aaaaaa", fontsize=8)
    ax7.set_ylabel("Amplitude moy.", color="#aaaaaa", fontsize=8)
    ax7.legend(fontsize=6, facecolor=BG2, labelcolor="white",
               ncol=3, loc="upper right")
    style_ax(ax7, "Traces moyennes par Hamming Weight (sur POI SNR)")

    fig.suptitle(f"SideShield — Phase 2 : Preprocessing [{name}]",
                 color="white", fontsize=13, fontweight="bold", y=0.99)

    out = os.path.join(DESKTOP, f"sideshield_preprocessing_{name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Figure sauvegardée → {out}")
    plt.show()


# ─────────────────────────────────────────────
# 7. COMPARAISON DESYNC
# ─────────────────────────────────────────────
def plot_desync_poi(db_path):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0d1117")

    configs = [
        ("desync0",   "ASCAD.h5",           "#00d4ff"),
        ("desync50",  "ASCAD_desync50.h5",  "#ffd700"),
        ("desync100", "ASCAD_desync100.h5", "#ff6b6b"),
    ]

    for row, (name, fname, color) in enumerate(configs):
        path = os.path.join(db_path, fname)
        with h5py.File(path, "r") as f:
            X  = f["Profiling_traces/traces"][:5000].astype(np.float32)
            y  = f["Profiling_traces/labels"][:5000]

        # Normalise
        mean = X.mean(axis=0); std = X.std(axis=0)
        std  = np.where(std == 0, 1e-8, std)
        X_n  = (X - mean) / std

        # SNR
        poi, scores = select_poi(X_n, y, n_poi=50, method="snr")

        # Superposition traces
        ax_left = axes[row, 0]
        ax_left.set_facecolor("#161b22")
        for i in range(30):
            ax_left.plot(X_n[i], color=color, linewidth=0.3, alpha=0.2)
        ax_left.plot(X_n.mean(axis=0), color=color, linewidth=1.0)
        ax_left.set_title(f"{name} — Traces superposées",
                          color="white", fontsize=9)
        ax_left.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax_left.spines.values():
            sp.set_edgecolor("#30363d")

        # SNR scores
        ax_right = axes[row, 1]
        ax_right.set_facecolor("#161b22")
        ax_right.plot(scores, color=color, linewidth=0.7)
        ax_right.scatter(poi, scores[poi], color="white", s=8, zorder=5)
        ax_right.set_title(f"{name} — SNR (POI en blanc)",
                           color="white", fontsize=9)
        ax_right.tick_params(colors="#aaaaaa", labelsize=7)
        for sp in ax_right.spines.values():
            sp.set_edgecolor("#30363d")

    fig.suptitle("SideShield — Effet Desync sur les POI",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(DESKTOP, "sideshield_desync_poi.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figure desync sauvegardée → {out}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  SideShield — Phase 2 : Preprocessing")
    print("="*60)

    path_d0 = os.path.join(DB_PATH, "ASCAD.h5")

    # ── Chargement
    print("\n  [1/5] Chargement des données...")
    X_train, y_train, meta_tr, X_test, y_test, meta_te = load_ascad(path_d0)
    print(f"  Train : {X_train.shape}  Test : {X_test.shape}")

    # ── Normalisation
    print("\n  [2/5] Normalisation z-score...")
    X_train_n, X_test_n = normalize(X_train, X_test, method="zscore")

    # ── Sélection POI
    print("\n  [3/5] Sélection des Points d'Intérêt...")
    poi_var, scores_var = select_poi(X_train_n, y_train, N_POI, "variance")
    poi_snr, scores_snr = select_poi(X_train_n, y_train, N_POI, "snr")

    print(f"\n  Overlap POI variance vs SNR : "
          f"{len(set(poi_var) & set(poi_snr))}/{N_POI} points communs")

    # ── DataLoaders
    print("\n  [4/5] Construction DataLoaders PyTorch...")
    train_loader, test_loader = build_dataloaders(
        X_train_n, y_train, X_test_n, y_test,
        poi_indices=poi_snr, batch_size=BATCH_SIZE
    )

    # Vérification GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device détecté : {device.upper()}")
    if device == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")

    # ── Sauvegarde du pipeline
    print("\n  [5/5] Sauvegarde du pipeline preprocessing...")
    pipeline = {
        "poi_snr"     : poi_snr,
        "poi_var"     : poi_var,
        "scores_snr"  : scores_snr,
        "scores_var"  : scores_var,
        "mean_train"  : X_train.mean(axis=0),
        "std_train"   : X_train.std(axis=0),
        "n_classes"   : 256,
        "n_poi"       : N_POI,
        "target_byte" : TARGET_BYTE,
    }
    pipe_path = os.path.join(OUT_PATH, "preprocessing_pipeline.pkl")
    with open(pipe_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  Pipeline sauvegardé → {pipe_path}")

    # ── Visualisations
    print("\n  Génération des visualisations...")
    plot_preprocessing(X_train, X_train_n, scores_var, scores_snr,
                       poi_var, poi_snr, y_train, name="desync0")

    print("\n  Comparaison effet desync sur POI...")
    plot_desync_poi(DB_PATH)

    print("\n" + "="*60)
    print("  Phase 2 terminée.")
    print("  Prochaine étape : Phase 3 — Architecture du modèle CNN")
    print("="*60 + "\n")
