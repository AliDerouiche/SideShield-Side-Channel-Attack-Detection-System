"""
SideShield — Reproduction exacte CNN ASCAD paper
==================================================
Référence : "Study of Deep Learning Techniques for Side-Channel
Analysis and Introduction to ASCAD Database"
Benadjila et al. — Thales + ANSSI, 2019

Architecture EXACTE du paper :
  5 × ConvBlock(ReLU + AvgPool) — SANS BatchNorm, SANS Dropout
  3 × FC(ReLU)
  Adam lr=1e-3, batch=200, epochs=75
  Input = 700 points complets
  Output = 256 classes (KB) ou 9 classes (HW)
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DESKTOP  = os.path.join(os.path.expanduser("~"), "Desktop")
DB_PATH  = os.path.join(DESKTOP, "ASCAD", "ASCAD_databases")
OUT_PATH = os.path.join(DESKTOP,
           "Side-Channel Attack Detection with Deep Learning",
           "SideShield")
os.makedirs(OUT_PATH, exist_ok=True)

# Hyperparamètres EXACTS du paper ASCAD
BATCH_SIZE = 200
EPOCHS     = 75
LR         = 1e-3
SEED       = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# S-BOX + HW
# ─────────────────────────────────────────────
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

HW = np.array([bin(SBOX[i]).count('1') for i in range(256)],
               dtype=np.int64)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class ASCADDataset(Dataset):
    def __init__(self, traces, labels, mode="kb"):
        # Normalisation MinMax [0,1] — exacte du paper
        self.X = torch.tensor(traces, dtype=torch.float32).unsqueeze(1)
        if mode == "hw":
            self.y = torch.tensor(HW[labels], dtype=torch.long)
        else:
            self.y = torch.tensor(labels.astype(np.int64),
                                  dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# CHARGEMENT + NORMALISATION MINMAX
# ─────────────────────────────────────────────
def load_data(filepath):
    with h5py.File(filepath, "r") as f:
        X_train = f["Profiling_traces/traces"][:].astype(np.float32)
        y_train = f["Profiling_traces/labels"][:]
        meta_tr = f["Profiling_traces/metadata"][:]
        X_test  = f["Attack_traces/traces"][:].astype(np.float32)
        y_test  = f["Attack_traces/labels"][:]
        meta_te = f["Attack_traces/metadata"][:]

    # Normalisation MinMax [0,1] — exactement comme le paper
    mn  = X_train.min()
    mx  = X_train.max()
    rng = mx - mn if mx != mn else 1e-8
    X_train = (X_train - mn) / rng
    X_test  = (X_test  - mn) / rng

    print(f"  Train : {X_train.shape}  dtype={X_train.dtype}")
    print(f"  Test  : {X_test.shape}")
    print(f"  Normalisation MinMax : [{X_train.min():.3f}, "
          f"{X_train.max():.3f}]")
    print(f"  Classes KB : {len(np.unique(y_train))}")
    print(f"  Cle cible  : {meta_tr['key'][0][2]:02X}h")

    return X_train, y_train, meta_tr, X_test, y_test, meta_te


# ─────────────────────────────────────────────
# ARCHITECTURE EXACTE DU PAPER ASCAD
# ─────────────────────────────────────────────
class ASCAD_CNN(nn.Module):
    """
    CNN exact du paper Thales/ANSSI 2019
    ─────────────────────────────────────
    Input : [B, 1, 700]

    Conv(1→64,   k=11) → ReLU → AvgPool(2) → [B, 64,  350]
    Conv(64→128, k=11) → ReLU → AvgPool(2) → [B, 128, 175]
    Conv(128→256,k=11) → ReLU → AvgPool(2) → [B, 256,  87]
    Conv(256→512,k=11) → ReLU → AvgPool(2) → [B, 512,  43]
    Conv(512→512,k=11) → ReLU → AvgPool(2) → [B, 512,  21]
    Flatten                                 → [B, 10752]
    FC(10752→4096) → ReLU
    FC(4096→4096)  → ReLU
    FC(4096→n_cls)

    IMPORTANT : pas de BatchNorm, pas de Dropout
    C'est volontaire — le paper n'en utilise pas.
    """
    def __init__(self, n_classes=9):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1,   64,  kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),
            # Block 2
            nn.Conv1d(64,  128, kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),
            # Block 4
            nn.Conv1d(256, 512, kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),
            # Block 5
            nn.Conv1d(512, 512, kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )

        # Calcul automatique de la taille après convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 700)
            feat_size = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, n_classes),
        )

        self._init_weights()
        print(f"  Feature size après conv : {feat_size}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.glorot_uniform_(m.weight
                    if hasattr(nn.init, 'glorot_uniform_')
                    else m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.glorot_uniform_(m.weight
                    if hasattr(nn.init, 'glorot_uniform_')
                    else m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ─────────────────────────────────────────────
# TRAINING — exact paper setup
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum = correct = n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        correct  += (logits.detach().argmax(1) == y).sum().item()
        loss_sum += loss.item() * len(y)
        n        += len(y)
    return loss_sum / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    loss_sum = correct = n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss   = criterion(logits, y)
        correct  += (logits.argmax(1) == y).sum().item()
        loss_sum += loss.item() * len(y)
        n        += len(y)
    return loss_sum / n, correct / n


def train(model, tr_loader, te_loader, tag="hw"):
    # Adam exact du paper — pas de weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0
    best_path = os.path.join(OUT_PATH,
                f"sideshield_ascad_{tag}_best.pt")
    history   = {k: [] for k in
                 ["tr_loss","tr_acc","te_loss","te_acc"]}

    n_cls = 9 if tag == "hw" else 256
    print(f"\n  {'─'*62}")
    print(f"  {'Ep':>4} {'Tr.Loss':>9} {'Tr.Acc':>8} "
          f"{'Te.Loss':>9} {'Te.Acc':>8} {'s':>5}  "
          f"[{n_cls} classes]")
    print(f"  {'─'*62}")

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        trl, tra = train_epoch(model, tr_loader,
                               optimizer, criterion)
        tel, tea = eval_epoch(model, te_loader, criterion)

        for k, v in zip(history.keys(),
                        [trl, tra, tel, tea]):
            history[k].append(v)

        mark = ""
        if tea > best_acc:
            best_acc = tea
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "best_acc": best_acc,
                "tag": tag,
            }, best_path)
            mark = " ✅"

        print(f"  {ep:>4} {trl:>9.4f} {tra*100:>7.2f}% "
              f"{tel:>9.4f} {tea*100:>7.2f}% "
              f"{time.time()-t0:>4.1f}s{mark}")

    print(f"\n  Best acc : {best_acc*100:.2f}%")
    return history, best_acc, best_path


# ─────────────────────────────────────────────
# RANK ANALYSIS
# ─────────────────────────────────────────────
@torch.no_grad()
def rank_analysis(model, X_test, y_test, meta_te,
                  max_traces=2000, tag="kb"):
    model.eval()
    true_key = int(meta_te["key"][0][2])
    pt_test  = meta_te["plaintext"][:, 2].astype(int)
    print(f"\n  Rank Analysis — cle : {true_key:02X}h")

    ds  = ASCADDataset(X_test, y_test, mode=tag)
    ldr = DataLoader(ds, 512, shuffle=False, num_workers=0)

    all_logits = []
    for X, _ in ldr:
        all_logits.append(model(X.to(DEVICE)).cpu().float())
    all_logits = torch.cat(all_logits).numpy()

    # Log-softmax
    mx        = all_logits.max(axis=1, keepdims=True)
    log_probs = all_logits - mx - np.log(
        np.exp(all_logits - mx).sum(axis=1, keepdims=True))

    n     = min(max_traces, len(X_test))
    ranks = []

    for t in range(1, n + 1):
        scores = np.zeros(256)
        for k_hyp in range(256):
            tgts = SBOX[pt_test[:t] ^ k_hyp].astype(int)
            if tag == "hw":
                tgts = HW[tgts]
            scores[k_hyp] = log_probs[:t][
                np.arange(t), tgts].sum()

        rank = int((scores > scores[true_key]).sum())
        ranks.append(rank)

        if rank == 0 and t >= 5:
            print(f"  Rang 0 a {t} traces !")
            break

    print(f"  Rang final : {ranks[-1]}")
    return ranks


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_training(history, best_acc, tag):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    eps = range(1, len(history["tr_loss"]) + 1)

    panels = [
        ("tr_loss","te_loss","Loss",     "#ff6b6b","#00d4ff"),
        ("tr_acc", "te_acc", "Accuracy", "#ff6b6b","#00d4ff"),
    ]
    for ax, (k1, k2, title, c1, c2) in zip(axes, panels):
        ax.set_facecolor("#161b22")
        ax.plot(eps, history[k1], color=c1,
                lw=1.5, label="Train")
        ax.plot(eps, history[k2], color=c2,
                lw=1.5, ls="--", label="Test")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("Epoch", color="#aaaaaa", fontsize=9)
        ax.legend(fontsize=8, facecolor="#161b22",
                  labelcolor="white")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        if k1 == "tr_acc":
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda y, _: f"{y*100:.1f}%"))

    n_cls = 9 if tag == "hw" else 256
    fig.suptitle(
        f"SideShield — ASCAD CNN Paper [{tag.upper()}]  |  "
        f"Best : {best_acc*100:.2f}%  |  {n_cls} classes",
        color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(DESKTOP,
          f"sideshield_ascad_{tag}_training.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figure → {out}")
    plt.show()


def plot_rank(ranks, tag):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(range(1, len(ranks)+1), ranks,
            color="#00d4ff", lw=1.5)
    ax.axhline(0, color="#39ff14", lw=1.0,
               ls="--", label="Rang 0")

    try:
        ntd = next(i for i,r in enumerate(ranks) if r==0) + 1
        ax.axvline(ntd, color="#ffd700", lw=1.0,
                   ls="--", label=f"NtD={ntd} traces")
        title = f"Rank Analysis [{tag.upper()}]  |  NtD={ntd}"
    except StopIteration:
        title = (f"Rank Analysis [{tag.upper()}]  |  "
                 f"Rang final={ranks[-1]}")

    ax.set_title(title, color="white",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Traces d'attaque",
                  color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Rang cle", color="#aaaaaa", fontsize=10)
    ax.legend(fontsize=9, facecolor="#161b22",
              labelcolor="white")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    plt.tight_layout()
    out = os.path.join(DESKTOP,
          f"sideshield_ascad_{tag}_rank.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Rank → {out}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  SideShield — CNN Exact ASCAD Paper (Thales 2019)")
    print("="*60)

    # Chargement
    print("\n  Chargement ASCAD.h5...")
    (X_train, y_train, meta_tr,
     X_test,  y_test,  meta_te) = load_data(
        os.path.join(DB_PATH, "ASCAD.h5"))

    # ─────────────────────────────────────────
    # PHASE A — HW (9 classes) — plus facile
    # ─────────────────────────────────────────
    print("\n" + "─"*60)
    print("  PHASE A — Hamming Weight (9 classes)")
    print("─"*60)

    tr_hw = DataLoader(
        ASCADDataset(X_train, y_train, "hw"),
        BATCH_SIZE, shuffle=True,  num_workers=0,
        pin_memory=True)
    te_hw = DataLoader(
        ASCADDataset(X_test,  y_test,  "hw"),
        BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True)

    model_hw = ASCAD_CNN(n_classes=9).to(DEVICE)
    print(f"  Parametres : {model_hw.count_params():,}")

    hist_hw, best_hw, path_hw = train(
        model_hw, tr_hw, te_hw, tag="hw")
    plot_training(hist_hw, best_hw, "hw")

    # Rank Analysis HW
    ckpt = torch.load(path_hw, map_location=DEVICE,
                      weights_only=False)
    model_hw.load_state_dict(ckpt["model_state"])
    ranks_hw = rank_analysis(
        model_hw, X_test, y_test, meta_te,
        max_traces=2000, tag="hw")
    plot_rank(ranks_hw, "hw")

    # ─────────────────────────────────────────
    # PHASE B — KB (256 classes)
    # ─────────────────────────────────────────
    print("\n" + "─"*60)
    print("  PHASE B — Key Byte (256 classes)")
    print("─"*60)

    tr_kb = DataLoader(
        ASCADDataset(X_train, y_train, "kb"),
        BATCH_SIZE, shuffle=True,  num_workers=0,
        pin_memory=True)
    te_kb = DataLoader(
        ASCADDataset(X_test,  y_test,  "kb"),
        BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True)

    model_kb = ASCAD_CNN(n_classes=256).to(DEVICE)
    print(f"  Parametres : {model_kb.count_params():,}")

    hist_kb, best_kb, path_kb = train(
        model_kb, tr_kb, te_kb, tag="kb")
    plot_training(hist_kb, best_kb, "kb")

    # Rank Analysis KB
    ckpt = torch.load(path_kb, map_location=DEVICE,
                      weights_only=False)
    model_kb.load_state_dict(ckpt["model_state"])
    ranks_kb = rank_analysis(
        model_kb, X_test, y_test, meta_te,
        max_traces=2000, tag="kb")
    plot_rank(ranks_kb, "kb")

    # ─────────────────────────────────────────
    # RÉSUMÉ
    # ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  RÉSUMÉ")
    print("="*60)
    print(f"  Phase A HW  (9 cls) : {best_hw*100:.2f}%")
    print(f"  Phase B KB (256 cls): {best_kb*100:.2f}%")
    print(f"  Rang final HW       : {ranks_hw[-1]}")
    print(f"  Rang final KB       : {ranks_kb[-1]}")
    print()
    print(f"  Ref paper ASCAD CNN :")
    print(f"    HW desync0 : ~80-85%")
    print(f"    KB desync0 : ~30-40%")
    print()

    for tag, best, ranks in [("HW", best_hw, ranks_hw),
                              ("KB", best_kb, ranks_kb)]:
        if ranks[-1] == 0:
            ntd = next(i for i,r in enumerate(ranks)
                       if r == 0) + 1
            print(f"  {tag} NtD : {ntd} traces")
        else:
            print(f"  {tag} rang final : {ranks[-1]}")

    print("="*60 + "\n")