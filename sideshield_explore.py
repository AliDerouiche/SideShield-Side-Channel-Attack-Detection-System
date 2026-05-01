"""
SideShield — Phase 1 : Exploration des traces ASCAD
====================================================
Script d'exploration visuelle et analytique du dataset ASCAD
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─────────────────────────────────────────────
# CONFIG — adapte ce chemin à ton bureau
# ─────────────────────────────────────────────
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
DB_PATH = os.path.join(DESKTOP, "ASCAD", "ASCAD_databases")

FILES = {
    "desync0"   : "ASCAD.h5",
    "desync50"  : "ASCAD_desync50.h5",
    "desync100" : "ASCAD_desync100.h5",
    "raw"       : "ATMega8515_raw_traces.h5",
}

# ─────────────────────────────────────────────
# 1. INSPECTION STRUCTURE H5
# ─────────────────────────────────────────────
def inspect_h5(filepath):
    print(f"\n{'='*60}")
    print(f"  Fichier : {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    def print_tree(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}📊 {name.split('/')[-1]}"
                  f"  shape={obj.shape}  dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}📁 {name.split('/')[-1]}/")

    with h5py.File(filepath, "r") as f:
        f.visititems(print_tree)


# ─────────────────────────────────────────────
# 2. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
def load_ascad(filepath):
    with h5py.File(filepath, "r") as f:
        # Profiling (train)
        X_train    = f["Profiling_traces/traces"][:]
        y_train    = f["Profiling_traces/labels"][:]
        meta_raw   = f["Profiling_traces/metadata"][:]   # structured array
        meta_train = {
            "plaintext" : meta_raw["plaintext"],
            "key"       : meta_raw["key"],
            "masks"     : meta_raw["masks"],
        }

        # Attack (test)
        X_test    = f["Attack_traces/traces"][:]
        y_test    = f["Attack_traces/labels"][:]
        meta_raw2 = f["Attack_traces/metadata"][:]
        meta_test = {
            "plaintext" : meta_raw2["plaintext"],
            "key"       : meta_raw2["key"],
            "masks"     : meta_raw2["masks"],
        }

    return X_train, y_train, meta_train, X_test, y_test, meta_test


# ─────────────────────────────────────────────
# 3. STATISTIQUES GÉNÉRALES
# ─────────────────────────────────────────────
def print_stats(name, X_train, y_train, X_test, y_test):
    X_tr = X_train.astype(np.float32)
    X_te = X_test.astype(np.float32)
    print(f"\n{'─'*60}")
    print(f"  STATISTIQUES — {name}")
    print(f"{'─'*60}")
    print(f"  Train  : {X_train.shape[0]:>6} traces x {X_train.shape[1]} points  dtype={X_train.dtype}")
    print(f"  Test   : {X_test.shape[0]:>6} traces x {X_test.shape[1]} points  dtype={X_test.dtype}")
    print(f"  Classes (key bytes) : {len(np.unique(y_train))}")
    print(f"  Train — min={X_tr.min():.1f}  max={X_tr.max():.1f}  mean={X_tr.mean():.4f}  std={X_tr.std():.4f}")
    print(f"  Test  — min={X_te.min():.1f}  max={X_te.max():.1f}  mean={X_te.mean():.4f}  std={X_te.std():.4f}")


# ─────────────────────────────────────────────
# 4. VISUALISATION PRINCIPALE
# ─────────────────────────────────────────────
def plot_exploration(name, X_train, y_train, X_test, y_test):
    # int8 → float32 pour les calculs et plots
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    COLOR_MAIN  = "#00d4ff"
    COLOR_SEC   = "#ff6b6b"
    COLOR_ACC   = "#ffd700"
    COLOR_GREEN = "#39ff14"
    BG         = "#0d1117"
    BG2        = "#161b22"

    def style_ax(ax, title):
        ax.set_facecolor(BG2)
        ax.set_title(title, color="white", fontsize=9, pad=6)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # ── 4.1 Trace individuelle
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(X_train[0], color=COLOR_MAIN, linewidth=0.7, alpha=0.9)
    ax1.set_xlabel("Points temporels", color="#aaaaaa", fontsize=8)
    ax1.set_ylabel("Consommation", color="#aaaaaa", fontsize=8)
    style_ax(ax1, f"Trace individuelle — {name} (train[0])")

    # ── 4.2 Moyenne ± std
    ax2 = fig.add_subplot(gs[0, 2])
    mean_t = X_train.mean(axis=0)
    std_t  = X_train.std(axis=0)
    x_pts  = np.arange(len(mean_t))
    ax2.plot(mean_t, color=COLOR_ACC, linewidth=0.8)
    ax2.fill_between(x_pts, mean_t - std_t, mean_t + std_t,
                     alpha=0.25, color=COLOR_ACC)
    ax2.set_xlabel("Points temporels", color="#aaaaaa", fontsize=8)
    style_ax(ax2, "Moyenne ± std (60k traces)")

    # ── 4.3 Superposition de 50 traces
    ax3 = fig.add_subplot(gs[1, :2])
    for i in range(50):
        ax3.plot(X_train[i], color=COLOR_MAIN, linewidth=0.3, alpha=0.15)
    ax3.plot(mean_t, color=COLOR_ACC, linewidth=1.0, label="Moyenne")
    ax3.legend(fontsize=7, facecolor=BG2, labelcolor="white")
    ax3.set_xlabel("Points temporels", color="#aaaaaa", fontsize=8)
    style_ax(ax3, "Superposition de 50 traces")

    # ── 4.4 Distribution des labels (key bytes)
    ax4 = fig.add_subplot(gs[1, 2])
    counts = np.bincount(y_train, minlength=256)
    ax4.bar(range(256), counts, color=COLOR_SEC, width=1.0, alpha=0.8)
    ax4.set_xlabel("Key byte value (0–255)", color="#aaaaaa", fontsize=8)
    ax4.set_ylabel("Fréquence", color="#aaaaaa", fontsize=8)
    style_ax(ax4, "Distribution des labels")

    # ── 4.5 FFT d'une trace (spectre fréquentiel)
    ax5 = fig.add_subplot(gs[2, 0])
    fft_vals = np.abs(np.fft.rfft(X_train[0]))
    freqs    = np.fft.rfftfreq(X_train.shape[1])
    ax5.plot(freqs, fft_vals, color=COLOR_GREEN, linewidth=0.7)
    ax5.set_xlabel("Fréquence normalisée", color="#aaaaaa", fontsize=8)
    ax5.set_ylabel("|FFT|", color="#aaaaaa", fontsize=8)
    style_ax(ax5, "Spectre FFT — trace[0]")

    # ── 4.6 Variance par point temporel (POI detection)
    ax6 = fig.add_subplot(gs[2, 1])
    var_t = X_train.var(axis=0)
    ax6.plot(var_t, color="#ff9f43", linewidth=0.7)
    top5 = np.argsort(var_t)[-5:]
    ax6.scatter(top5, var_t[top5], color=COLOR_SEC, zorder=5, s=20,
                label="Top 5 POI")
    ax6.legend(fontsize=7, facecolor=BG2, labelcolor="white")
    ax6.set_xlabel("Points temporels", color="#aaaaaa", fontsize=8)
    ax6.set_ylabel("Variance", color="#aaaaaa", fontsize=8)
    style_ax(ax6, "Variance → Points d'Intérêt (POI)")

    # ── 4.7 Comparaison train vs test (moyenne)
    ax7 = fig.add_subplot(gs[2, 2])
    mean_test = X_test.mean(axis=0)
    ax7.plot(mean_t,    color=COLOR_MAIN, linewidth=0.8, label="Train", alpha=0.8)
    ax7.plot(mean_test, color=COLOR_SEC,  linewidth=0.8, label="Test",  alpha=0.8)
    ax7.legend(fontsize=7, facecolor=BG2, labelcolor="white")
    ax7.set_xlabel("Points temporels", color="#aaaaaa", fontsize=8)
    style_ax(ax7, "Moyenne Train vs Test")

    fig.suptitle(f"SideShield — Exploration ASCAD [{name}]",
                 color="white", fontsize=13, fontweight="bold", y=0.98)

    out_path = os.path.join(DESKTOP, f"sideshield_exploration_{name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  ✅ Figure sauvegardée → {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# 5. COMPARAISON DESYNC0 vs DESYNC50 vs DESYNC100
# ─────────────────────────────────────────────
def plot_desync_comparison(data_dict):
    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    fig.patch.set_facecolor("#0d1117")
    colors = ["#00d4ff", "#ffd700", "#ff6b6b"]

    for ax, (name, color) in zip(axes, zip(data_dict.keys(), colors)):
        X = data_dict[name]
        ax.set_facecolor("#161b22")
        for i in range(20):
            ax.plot(X[i], color=color, linewidth=0.4, alpha=0.3)
        ax.plot(X.mean(axis=0), color=color, linewidth=1.2,
                label=f"Moyenne — {name}")
        ax.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig.suptitle("SideShield — Effet du Desynchronisation sur les Traces",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(DESKTOP, "sideshield_desync_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  ✅ Comparaison desync sauvegardée → {out_path}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  SideShield — Phase 1 : Exploration ASCAD")
    print("="*60)

    # ── Inspection structure
    for key, fname in FILES.items():
        path = os.path.join(DB_PATH, fname)
        if os.path.exists(path):
            inspect_h5(path)
        else:
            print(f"\n  ⚠️  Fichier non trouvé : {fname}")

    # ── Chargement et exploration principale (desync0)
    path_d0 = os.path.join(DB_PATH, FILES["desync0"])
    if os.path.exists(path_d0):
        print("\n\n  Chargement ASCAD.h5 (desync0)...")
        X_train, y_train, meta_train, X_test, y_test, meta_test = load_ascad(path_d0)
        print_stats("desync0", X_train, y_train, X_test, y_test)

        # Infos clé (toutes les traces ont la même clé fixe dans ASCAD)
        print(f"\n  Clé AES (première trace) : "
              f"{' '.join(f'{b:02X}' for b in meta_train['key'][0])}")
        print(f"  Byte cible (index 2)     : {meta_train['key'][0][2]:02X}")

        plot_exploration("desync0", X_train, y_train, X_test, y_test)

    # ── Comparaison desync
    print("\n\n  Chargement des variantes desync pour comparaison...")
    desync_data = {}
    for key in ["desync0", "desync50", "desync100"]:
        path = os.path.join(DB_PATH, FILES[key])
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                desync_data[key] = f["Profiling_traces/traces"][:200].astype(np.float32)
            print(f"  ✅ {key} chargé")

    if len(desync_data) == 3:
        plot_desync_comparison(desync_data)

    print("\n\n  Phase 1 terminée. Tu peux maintenant passer au preprocessing.\n")