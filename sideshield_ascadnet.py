"""
SideShield — Reproduction exacte CNN ASCAD paper
==================================================
Reference : "Study of Deep Learning Techniques for Side-Channel
             Analysis and Introduction to ASCAD Database"
             Benadjila et al. — Thales / ANSSI 2019

Architecture EXACTE du paper :
  5x Conv(ReLU + AvgPool) — SANS BatchNorm, SANS Dropout
  3x FC(ReLU)
  Adam lr=1e-3, batch=200, epochs=75
  Input = 700 points complets
  Labels = S_Box[plaintext XOR key] — 256 classes
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

# Hyperparamètres EXACTES du paper ASCAD
TRACE_LEN  = 700
BATCH_SIZE = 200    # exact paper
EPOCHS     = 75     # exact paper
LR         = 1e-3   # exact paper
N_CLASSES  = 256
TARGET_BYTE= 2
SEED       = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# S-BOX
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

HW = np.array([bin(x).count('1') for x in range(256)], dtype=np.int64)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class ASCADDataset(Dataset):
    def __init__(self, traces, labels):
        # Normalisation interne [0,1] — exact paper ASCAD
        X = traces.astype(np.float32)
        X = (X - X.min()) / (X.max() - X.min() + 1e-8)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(labels.astype(np.int64),
                              dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
def load_ascad(filepath):
    with h5py.File(filepath, "r") as f:
        X_train   = f["Profiling_traces/traces"][:]
        y_train   = f["Profiling_traces/labels"][:]
        meta_train= f["Profiling_traces/metadata"][:]
        X_test    = f["Attack_traces/traces"][:]
        y_test    = f["Attack_traces/labels"][:]
        meta_test = f["Attack_traces/metadata"][:]

    pt_test  = meta_test["plaintext"][:, TARGET_BYTE].astype(int)
    key_test = meta_test["key"][:, TARGET_BYTE].astype(int)

    print(f"  Train : {X_train.shape}  labels={len(np.unique(y_train))} classes")
    print(f"  Test  : {X_test.shape}")
    print(f"  Cle   : {meta_train['key'][0][TARGET_BYTE]:02X}h "
          f"(byte {TARGET_BYTE})")
    print(f"  Labels range : [{y_train.min()} — {y_train.max()}]")

    return X_train, y_train, X_test, y_test, pt_test, key_test


# ─────────────────────────────────────────────
# ARCHITECTURE EXACTE PAPER ASCAD
# ─────────────────────────────────────────────
class ASCADNet(nn.Module):
    """
    CNN exact du paper ASCAD (Benadjila et al. 2019)
    ──────────────────────────────────────────────────
    PAS de BatchNorm
    PAS de Dropout
    PAS de régularisation
    ReLU partout
    AvgPool(2) après chaque conv

    Input  : [B, 1, 700]
    Output : [B, 256]

    Table 2 du paper :
      Conv(64,  k=11) → ReLU → AvgPool(2) → 350
      Conv(128, k=11) → ReLU → AvgPool(2) → 175
      Conv(256, k=11) → ReLU → AvgPool(2) → 87
      Conv(512, k=11) → ReLU → AvgPool(2) → 43
      Conv(512, k=11) → ReLU → AvgPool(2) → 21
      Flatten → 512*21 = 10752
      FC(4096) → ReLU
      FC(4096) → ReLU
      FC(256)  → Softmax
    """
    def __init__(self, n_classes=256):
        super().__init__()

        self.features = nn.Sequential(
            # Conv 1 : 700 → 350
            nn.Conv1d(1,   64,  kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),

            # Conv 2 : 350 → 175
            nn.Conv1d(64,  128, kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),

            # Conv 3 : 175 → 87
            nn.Conv1d(128, 256, kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),

            # Conv 4 : 87 → 43
            nn.Conv1d(256, 512, kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),

            # Conv 5 : 43 → 21
            nn.Conv1d(512, 512, kernel_size=11,
                      padding=5, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(2),
        )

        # 512 * 21 = 10752
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 21, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ─────────────────────────────────────────────
# TRAINING — exact paper (Adam, pas de scheduler)
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    loss_sum = correct = n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE.type):
            logits = model(X)
            loss   = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
        with torch.amp.autocast(device_type=DEVICE.type):
            logits = model(X)
            loss   = criterion(logits, y)
        correct  += (logits.argmax(1) == y).sum().item()
        loss_sum += loss.item() * len(y)
        n        += len(y)
    return loss_sum / n, correct / n


def train(model, tr_loader, te_loader):
    # Exact paper : Adam, lr=1e-3, pas de weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler    = torch.amp.GradScaler()

    best_acc  = 0.0
    best_path = os.path.join(OUT_PATH, "ascadnet_best.pt")
    history   = {k: [] for k in
                 ["tr_loss","tr_acc","te_loss","te_acc"]}

    print(f"\n  {'─'*60}")
    print(f"  {'Ep':>4} {'Tr.Loss':>9} {'Tr.Acc':>8} "
          f"{'Te.Loss':>9} {'Te.Acc':>8} {'s':>5}")
    print(f"  {'─'*60}")

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        trl, tra = train_epoch(model, tr_loader,
                               optimizer, criterion, scaler)
        tel, tea = eval_epoch(model, te_loader, criterion)

        for k, v in zip(history.keys(),
                        [trl, tra, tel, tea]):
            history[k].append(v)

        mark = ""
        if tea > best_acc:
            best_acc = tea
            torch.save({
                "epoch"      : ep,
                "model_state": model.state_dict(),
                "best_acc"   : best_acc,
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
def rank_analysis(model, te_loader, pt_test, key_test,
                  max_traces=3000):
    model.eval()
    true_key = int(key_test[0])
    print(f"\n  Rank Analysis — cle cible : {true_key:02X}h")

    all_logits = []
    for X, _ in te_loader:
        with torch.amp.autocast(device_type=DEVICE.type):
            all_logits.append(
                model(X.to(DEVICE)).cpu().float())
    all_logits = torch.cat(all_logits).numpy()

    # Log-softmax
    lp = all_logits - np.log(
        np.exp(all_logits).sum(axis=1, keepdims=True))

    n = min(max_traces, len(pt_test))
    ranks = []

    for t in range(1, n + 1):
        scores = np.zeros(256)
        for k in range(256):
            tgts = SBOX[pt_test[:t] ^ k]
            scores[k] = lp[:t][np.arange(t),
                               tgts.astype(int)].sum()
        rank = int((scores > scores[true_key]).sum())
        ranks.append(rank)
        if rank == 0 and t >= 5:
            print(f"  Rang 0 atteint a {t} traces !")
            break

    print(f"  Rang final ({len(ranks)} traces) : {ranks[-1]}")
    return ranks


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
def plot_training(history, best_acc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    eps = range(1, len(history["tr_loss"]) + 1)

    for ax, (k1, k2, title) in zip(axes, [
        ("tr_loss","te_loss","Loss"),
        ("tr_acc", "te_acc", "Accuracy"),
    ]):
        ax.set_facecolor("#161b22")
        ax.plot(eps, history[k1], color="#ff6b6b",
                lw=1.5, label="Train")
        ax.plot(eps, history[k2], color="#00d4ff",
                lw=1.5, ls="--", label="Test")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("Epoch", color="#aaaaaa")
        ax.legend(fontsize=8, facecolor="#161b22",
                  labelcolor="white")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        if k1 == "tr_acc":
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda y, _: f"{y*100:.1f}%"))

    fig.suptitle(
        f"ASCADNet (paper exact) — Best Test : "
        f"{best_acc*100:.2f}%  |  KB 256 classes",
        color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(DESKTOP, "sideshield_ascadnet_training.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Training → {out}")
    plt.show()


def plot_rank(ranks):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(range(1, len(ranks)+1), ranks,
            color="#00d4ff", lw=1.5)
    ax.axhline(0, color="#39ff14", lw=1.0,
               ls="--", label="Rang 0")
    try:
        fz = next(i for i, r in enumerate(ranks) if r == 0)
        ax.axvline(fz+1, color="#ffd700", lw=1.0, ls="--",
                   label=f"NtD = {fz+1} traces")
        title = f"Rank Analysis | NtD = {fz+1} traces"
    except StopIteration:
        title = f"Rank Analysis | Rang final = {ranks[-1]}"
    ax.set_title(title, color="white",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Traces d'attaque", color="#aaaaaa")
    ax.set_ylabel("Rang cle", color="#aaaaaa")
    ax.legend(fontsize=9, facecolor="#161b22",
              labelcolor="white")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    plt.tight_layout()
    out = os.path.join(DESKTOP, "sideshield_ascadnet_rank.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Rank → {out}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  ASCADNet — Reproduction exacte paper Thales 2019")
    print("="*60)

    # ── Chargement
    print("\n  [1/3] Chargement ASCAD.h5...")
    (X_train, y_train, X_test, y_test,
     pt_test, key_test) = load_ascad(
        os.path.join(DB_PATH, "ASCAD.h5"))

    tr_loader = DataLoader(
        ASCADDataset(X_train, y_train),
        BATCH_SIZE, shuffle=True, num_workers=0)
    te_loader = DataLoader(
        ASCADDataset(X_test, y_test),
        BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Modèle
    print("\n  [2/3] Architecture ASCADNet (paper exact)...")
    model = ASCADNet(n_classes=N_CLASSES).to(DEVICE)
    print(f"  Parametres : {model.count_params():,}")
    print(f"  Conv layers : 5 x Conv(ReLU+AvgPool)")
    print(f"  FC layers   : FC(4096) → FC(4096) → FC(256)")
    print(f"  Batch size  : {BATCH_SIZE} (paper exact)")
    print(f"  LR          : {LR} (paper exact)")
    print(f"  Epochs      : {EPOCHS} (paper exact)")
    print(f"  Normalisation : MinMax [0,1] (paper exact)")
    print(f"  BatchNorm   : NON (paper exact)")
    print(f"  Dropout     : NON (paper exact)")

    # Vérif VRAM
    if DEVICE.type == "cuda":
        dummy = torch.randn(BATCH_SIZE, 1, TRACE_LEN).to(DEVICE)
        with torch.amp.autocast(device_type="cuda"):
            _ = model(dummy)
        print(f"  VRAM utilisee : "
              f"{torch.cuda.memory_allocated()/1e9:.3f} GB")
        del dummy
        torch.cuda.empty_cache()

    # ── Entraînement
    print("\n  [3/3] Entraînement...")
    history, best_acc, best_path = train(
        model, tr_loader, te_loader)
    plot_training(history, best_acc)

    # ── Rank Analysis
    ckpt = torch.load(best_path, map_location=DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    ranks = rank_analysis(model, te_loader,
                          pt_test, key_test)
    plot_rank(ranks)

    # ── Résumé
    print("\n" + "="*60)
    print("  RESUME FINAL")
    print("="*60)
    print(f"  Dataset     : ASCAD.h5 (Thales/ANSSI)")
    print(f"  Architecture: ASCADNet (paper exact)")
    print(f"  Best acc    : {best_acc*100:.2f}%")
    print(f"  Rang final  : {ranks[-1]}")
    if ranks[-1] == 0:
        ntd = next(i for i,r in enumerate(ranks) if r==0)+1
        print(f"  NtD         : {ntd} traces pour rank=0")
    print(f"\n  Reference paper ASCAD :")
    print(f"    KB accuracy : ~30-40% sur desync0")
    print(f"    NtD         : < 500 traces")
    print(f"\n  Random chance : {100/256:.2f}%")
    print("="*60 + "\n")
