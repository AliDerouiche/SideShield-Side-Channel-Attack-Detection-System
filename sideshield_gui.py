"""
SideShield — GUI Interface
===========================
Interface professionnelle de détection d'anomalies side-channel
Style : Dark cybersecurity dashboard — inspiré des terminaux de sécurité
"""

import sys
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QFrame, QSplitter,
    QTabWidget, QGridLayout, QSlider, QComboBox, QScrollArea,
    QGroupBox, QStatusBar, QSizePolicy
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QRect, QPoint
)
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QLinearGradient,
    QPainterPath, QPolygonF, QFontDatabase
)
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# CONFIG PATHS
# ─────────────────────────────────────────────
DESKTOP  = os.path.join(os.path.expanduser("~"), "Desktop")
DB_PATH  = os.path.join(DESKTOP, "ASCAD", "ASCAD_databases")
PROJ_DIR  = os.path.join(DESKTOP, "Side-Channel Attack Detection with Deep Learning")
OUT_PATH  = os.path.join(PROJ_DIR, "SideShield")
MODEL_PATH = os.path.join(OUT_PATH, "sideshield_ae_final.pt")

TRACE_LEN  = 700
LATENT_DIM = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PyVISA optionnel — instrument réel
try:
    import pyvisa
    PYVISA_AVAILABLE = True
except ImportError:
    PYVISA_AVAILABLE = False

# ─────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────
C = {
    "bg":        "#090e14",
    "bg2":       "#0d1520",
    "bg3":       "#111d2b",
    "panel":     "#0a1628",
    "border":    "#1a2f45",
    "border2":   "#0f2035",
    "cyan":      "#00d4ff",
    "cyan_dim":  "#005566",
    "green":     "#00ff88",
    "green_dim": "#003322",
    "red":       "#ff3355",
    "red_dim":   "#330011",
    "yellow":    "#ffcc00",
    "orange":    "#ff6b35",
    "white":     "#e8f4fd",
    "gray":      "#4a6278",
    "gray2":     "#2a3f52",
}

# ─────────────────────────────────────────────
# MODÈLE (identique à sideshield_autoencoder_v2.py)
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1,   32,  11, padding=5,  bias=False),
            nn.BatchNorm1d(32), nn.SiLU(), nn.AvgPool1d(2),
            nn.Conv1d(32,  64,  7,  padding=3,  bias=False),
            nn.BatchNorm1d(64), nn.SiLU(), nn.AvgPool1d(2),
            nn.Conv1d(64,  128, 5,  padding=2,  bias=False),
            nn.BatchNorm1d(128), nn.SiLU(), nn.AvgPool1d(2),
        )
        self.flat = nn.Flatten()
        self.fc   = nn.Sequential(
            nn.Linear(128 * 87, latent_dim),
            nn.LayerNorm(latent_dim), nn.Tanh(),
        )
    def forward(self, x):
        return self.fc(self.flat(self.conv(x)))

class Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128 * 87), nn.SiLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, stride=2, padding=2,
                               output_padding=1, bias=False),
            nn.BatchNorm1d(64), nn.SiLU(),
            nn.ConvTranspose1d(64, 32, 7, stride=2, padding=3,
                               output_padding=1, bias=False),
            nn.BatchNorm1d(32), nn.SiLU(),
            nn.ConvTranspose1d(32, 1, 11, stride=2, padding=5,
                               output_padding=1),
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
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def forward(self, x):
        return self.decoder(self.encoder(x)), self.encoder(x)
    @torch.no_grad()
    def score(self, x):
        self.eval()
        x_r, _ = self(x)
        return float(((x - x_r) ** 2).mean())

# ─────────────────────────────────────────────
# GÉNÉRATEUR D'ATTAQUES
# ─────────────────────────────────────────────
ATTACK_TYPES = {
    0: "Normal",
    1: "Gaussian Noise",
    2: "Desynchronization",
    3: "Amplitude Scale",
    4: "Spike EMFI",
}

def inject_attack(trace, attack_type, intensity=1.0):
    """Injecte une anomalie simulée dans une trace."""
    t = trace.copy()
    std = trace.std()

    if attack_type == 1:   # Gaussian Noise
        t += np.random.normal(0, intensity * std, len(t))

    elif attack_type == 2:  # Desync
        shift = int(intensity * 50)
        t = np.roll(t, shift)
        t[:shift] = np.random.normal(0, 0.3 * std, shift)

    elif attack_type == 3:  # Amplitude Scale
        start  = np.random.randint(0, len(t) - 100)
        length = np.random.randint(20, 100)
        scale  = 1.0 + intensity * 2.5
        t[start:start+length] *= scale

    elif attack_type == 4:  # Spike EMFI
        pos   = np.random.randint(10, len(t) - 10)
        amp   = intensity * 6.0 * std
        width = 5
        spike = amp * np.exp(
            -0.5 * ((np.arange(len(t)) - pos) / width) ** 2)
        t += spike

    return t.astype(np.float32)

# ─────────────────────────────────────────────
# THREAD D'ANALYSE
# ─────────────────────────────────────────────
class AnalysisThread(QThread):
    result_ready   = pyqtSignal(dict)
    progress       = pyqtSignal(int)

    def __init__(self, model, traces, threshold, attack_type,
                 intensity, norm_mean, norm_std):
        super().__init__()
        self.model       = model
        self.traces      = traces
        self.threshold   = threshold
        self.attack_type = attack_type
        self.intensity   = intensity
        self.norm_mean   = norm_mean
        self.norm_std    = norm_std

    def run(self):
        scores = []
        n = min(200, len(self.traces))
        for i in range(n):
            trace = self.traces[i].copy()
            # Dénormalise → injecte → renormalise
            trace_raw = trace * self.norm_std + self.norm_mean
            if self.attack_type > 0:
                trace_raw = inject_attack(
                    trace_raw, self.attack_type, self.intensity)
            trace_n = (trace_raw - self.norm_mean) / self.norm_std
            x = torch.tensor(trace_n[None, None, :],
                             dtype=torch.float32).to(DEVICE)
            score = self.model.score(x)
            scores.append((trace_n, score,
                           score > self.threshold))
            self.progress.emit(int((i + 1) / n * 100))

        scores_arr  = np.array([s[1] for s in scores])
        detections  = sum(1 for s in scores if s[2])
        self.result_ready.emit({
            "traces":     [s[0] for s in scores],
            "scores":     scores_arr,
            "detections": detections,
            "total":      n,
            "mean_score": scores_arr.mean(),
            "max_score":  scores_arr.max(),
            "det_rate":   detections / n * 100,
        })

# ─────────────────────────────────────────────
# WIDGETS CUSTOM
# ─────────────────────────────────────────────
class GlowLabel(QLabel):
    """Label avec effet néon."""
    def __init__(self, text, color=C["cyan"], size=12, bold=False):
        super().__init__(text)
        weight = "bold" if bold else "normal"
        self.setStyleSheet(f"""
            color: {color};
            font-family: 'Courier New', monospace;
            font-size: {size}px;
            font-weight: {weight};
        """)

class MetricCard(QFrame):
    """Carte de métrique style terminal."""
    def __init__(self, title, value="--", unit="", color=C["cyan"]):
        super().__init__()
        self.color = color
        self.setFixedHeight(90)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C['panel']};
                border: 1px solid {color}44;
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet(f"""
            color: {C['gray']};
            font-size: 9px;
            font-family: 'Courier New';
            letter-spacing: 2px;
        """)

        self.value_lbl = QLabel(value)
        self.value_lbl.setStyleSheet(f"""
            color: {color};
            font-size: 22px;
            font-family: 'Courier New';
            font-weight: bold;
        """)

        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setStyleSheet(f"""
            color: {color}88;
            font-size: 10px;
            font-family: 'Courier New';
        """)

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.unit_lbl)

    def update_value(self, value, color=None):
        self.value_lbl.setText(str(value))
        if color:
            self.value_lbl.setStyleSheet(f"""
                color: {color};
                font-size: 22px;
                font-family: 'Courier New';
                font-weight: bold;
            """)

class StatusIndicator(QWidget):
    """Indicateur de statut animé."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 200)
        self.status   = "idle"    # idle / normal / attack
        self.pulse    = 0.0
        self.timer    = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.start(50)

    def set_status(self, status):
        self.status = status

    def _animate(self):
        self.pulse = (self.pulse + 0.1) % (2 * math.pi)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, r = 100, 100, 60

        if self.status == "idle":
            color = QColor(C["gray"])
            label = "IDLE"
        elif self.status == "normal":
            color = QColor(C["green"])
            label = "NORMAL"
        else:
            color = QColor(C["red"])
            label = "ATTACK"

        # Cercles de pulse
        pulse_val = (math.sin(self.pulse) + 1) / 2
        for i in range(3):
            alpha = int(80 * (1 - i * 0.3) * pulse_val)
            pr    = r + i * 20 + pulse_val * 10
            c2    = QColor(color)
            c2.setAlpha(alpha)
            p.setPen(QPen(c2, 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(int(cx - pr), int(cy - pr),
                          int(2 * pr), int(2 * pr))

        # Cercle principal
        grad = QLinearGradient(cx - r, cy - r, cx + r, cy + r)
        c_light = QColor(color); c_light.setAlpha(40)
        c_dark  = QColor(color); c_dark.setAlpha(15)
        grad.setColorAt(0, c_light)
        grad.setColorAt(1, c_dark)
        p.setBrush(QBrush(grad))
        p.setPen(QPen(color, 2))
        p.drawEllipse(cx - r, cy - r, 2*r, 2*r)

        # Texte
        p.setPen(color)
        font = QFont("Courier New", 11, QFont.Bold)
        p.setFont(font)
        p.drawText(QRect(cx-r, cy-15, 2*r, 30),
                   Qt.AlignCenter, label)

        p.end()

class TraceCanvas(FigureCanvasQTAgg):
    """Canvas matplotlib pour les traces."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 2.5), dpi=90)
        self.fig.patch.set_facecolor(C["bg2"])
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self._style_ax()

    def _style_ax(self):
        self.ax.set_facecolor(C["bg3"])
        self.ax.tick_params(colors=C["gray"], labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(C["border"])
        self.ax.set_xlabel("Points temporels",
                           color=C["gray"], fontsize=8)
        self.ax.set_ylabel("Amplitude", color=C["gray"], fontsize=8)
        self.fig.tight_layout(pad=0.8)

    def plot_trace(self, trace, recon=None, score=None,
                   threshold=None, is_attack=False):
        self.ax.cla()
        self._style_ax()
        color = C["red"] if is_attack else C["cyan"]

        self.ax.plot(trace, color=color, lw=0.8,
                     alpha=0.85, label="Trace")
        if recon is not None:
            self.ax.plot(recon, color=C["yellow"], lw=0.8,
                         alpha=0.7, ls="--", label="Reconstruction")
            self.ax.fill_between(range(len(trace)), trace, recon,
                                 alpha=0.15, color=C["orange"])

        status = "[!] ATTACK DETECTED" if is_attack else "[OK] NORMAL"
        title  = f"MSE={score:.6f}  {status}" if score else "Trace"
        self.ax.set_title(title, color=color, fontsize=9, pad=4)
        self.ax.legend(fontsize=7, facecolor=C["bg3"],
                       labelcolor=C["white"], loc="upper right")
        self.draw()

class ScoreHistCanvas(FigureCanvasQTAgg):
    """Histogramme des scores en temps réel."""
    def __init__(self):
        self.fig = Figure(figsize=(5, 2.5), dpi=90)
        self.fig.patch.set_facecolor(C["bg2"])
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self._style_ax()

    def _style_ax(self):
        self.ax.set_facecolor(C["bg3"])
        self.ax.tick_params(colors=C["gray"], labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(C["border"])

    def update_hist(self, scores_normal, scores_attack,
                    threshold):
        self.ax.cla()
        self._style_ax()
        if len(scores_normal) > 0:
            bins = np.linspace(
                min(scores_normal.min(),
                    scores_attack.min() if len(scores_attack) > 0
                    else scores_normal.min()),
                max(scores_normal.max(),
                    scores_attack.max() if len(scores_attack) > 0
                    else scores_normal.max()), 40)
            self.ax.hist(scores_normal, bins=bins,
                         color=C["cyan"], alpha=0.6,
                         label="Normal", density=True)
            if len(scores_attack) > 0:
                self.ax.hist(scores_attack, bins=bins,
                             color=C["red"], alpha=0.6,
                             label="Attack", density=True)
            self.ax.axvline(threshold, color=C["yellow"],
                            lw=1.5, ls="--", label="Threshold")
            self.ax.legend(fontsize=7, facecolor=C["bg3"],
                           labelcolor=C["white"])
        self.ax.set_title("Score Distribution",
                          color=C["gray"], fontsize=8, pad=3)
        self.fig.tight_layout(pad=0.8)
        self.draw()

# ─────────────────────────────────────────────
# FENÊTRE PRINCIPALE
# ─────────────────────────────────────────────
class SideShieldGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model      = None
        self.threshold  = None
        self.norm_mean  = None
        self.norm_std   = None
        self.traces     = None
        self.thread     = None
        self.scores_normal = []
        self.scores_attack = []
        self.scan_idx   = 0
        self.input_mode = "sim"

        self.setWindowTitle("SideShield — Side-Channel Attack Detection System")
        self.setMinimumSize(1280, 820)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {C['bg']};
                color: {C['white']};
                font-family: 'Courier New', monospace;
            }}
            QTabWidget::pane {{
                border: 1px solid {C['border']};
                background: {C['bg2']};
            }}
            QTabBar::tab {{
                background: {C['bg3']};
                color: {C['gray']};
                padding: 8px 20px;
                border: 1px solid {C['border2']};
                font-size: 10px;
                letter-spacing: 1px;
            }}
            QTabBar::tab:selected {{
                background: {C['panel']};
                color: {C['cyan']};
                border-bottom: 2px solid {C['cyan']};
            }}
            QPushButton {{
                background: {C['bg3']};
                color: {C['cyan']};
                border: 1px solid {C['cyan']}66;
                padding: 8px 20px;
                font-family: 'Courier New';
                font-size: 11px;
                letter-spacing: 1px;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background: {C['cyan']}22;
                border: 1px solid {C['cyan']};
            }}
            QPushButton:pressed {{
                background: {C['cyan']}44;
            }}
            QPushButton:disabled {{
                color: {C['gray']};
                border-color: {C['gray2']};
            }}
            QComboBox {{
                background: {C['bg3']};
                color: {C['white']};
                border: 1px solid {C['border']};
                padding: 5px 10px;
                font-family: 'Courier New';
                font-size: 11px;
                border-radius: 3px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {C['bg3']};
                color: {C['white']};
                selection-background-color: {C['cyan']}33;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {C['bg3']};
                border: 1px solid {C['border']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {C['cyan']};
                width: 14px; height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {C['cyan']}66;
                border-radius: 2px;
            }}
            QProgressBar {{
                background: {C['bg3']};
                border: 1px solid {C['border']};
                border-radius: 3px;
                text-align: center;
                color: {C['cyan']};
                font-size: 10px;
                font-family: 'Courier New';
            }}
            QProgressBar::chunk {{
                background: {C['cyan']}88;
                border-radius: 2px;
            }}
            QScrollArea {{ border: none; }}
            QGroupBox {{
                border: 1px solid {C['border']};
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
                color: {C['gray']};
                font-size: 9px;
                letter-spacing: 2px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                color: {C['gray']};
            }}
        """)

        self._build_ui()
        self._load_model()

    # ── BUILD UI ──────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        root.addWidget(self._build_header())

        # Main content
        tabs = QTabWidget()
        tabs.addTab(self._build_dashboard(), "  DASHBOARD  ")
        tabs.addTab(self._build_analyzer(),  "  ANALYZER   ")
        tabs.addTab(self._build_about(),     "  ABOUT      ")
        root.addWidget(tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"""
            QStatusBar {{
                background: {C['bg3']};
                color: {C['gray']};
                border-top: 1px solid {C['border']};
                font-size: 10px;
                font-family: 'Courier New';
            }}
        """)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "  SideShield v1.0 | Loading model...")

    def _build_header(self):
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet(f"""
            QFrame {{
                background: {C['bg3']};
                border-bottom: 1px solid {C['border']};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 0, 20, 0)

        # Logo
        logo = QLabel("[ SIDESHIELD ]")
        logo.setStyleSheet(f"""
            color: {C['cyan']};
            font-size: 18px;
            font-weight: bold;
            font-family: 'Courier New';
            letter-spacing: 4px;
        """)

        subtitle = QLabel(
            "SIDE-CHANNEL ATTACK DETECTION SYSTEM  //  "
            "Deep Learning Anomaly Detection  //  ASCAD Dataset")
        subtitle.setStyleSheet(f"""
            color: {C['gray']};
            font-size: 9px;
            font-family: 'Courier New';
            letter-spacing: 1px;
        """)

        self.model_status = QLabel("[ MODEL: LOADING ]")
        self.model_status.setStyleSheet(f"""
            color: {C['yellow']};
            font-size: 10px;
            font-family: 'Courier New';
        """)

        layout.addWidget(logo)
        layout.addSpacing(20)
        layout.addWidget(subtitle)
        layout.addStretch()
        layout.addWidget(self.model_status)

        return header

    def _build_dashboard(self):
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # ── Left panel : Controls + Status
        left = QWidget()
        left.setFixedWidth(270)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)

        # Status indicator
        status_group = QGroupBox("SYSTEM STATUS")
        sg_layout = QVBoxLayout(status_group)
        self.indicator = StatusIndicator()
        sg_layout.addWidget(self.indicator, alignment=Qt.AlignCenter)
        left_layout.addWidget(status_group)

        # ── Input Source
        src_group = QGroupBox("INPUT SOURCE")
        src_layout = QVBoxLayout(src_group)

        # Mode buttons
        mode_layout = QHBoxLayout()
        self.btn_mode_sim  = QPushButton("SIM")
        self.btn_mode_file = QPushButton("FILE")
        self.btn_mode_live = QPushButton("LIVE")
        for btn in [self.btn_mode_sim,
                    self.btn_mode_file, self.btn_mode_live]:
            btn.setFixedHeight(28)
            btn.setCheckable(True)
            mode_layout.addWidget(btn)
        self.btn_mode_sim.setChecked(True)
        self.btn_mode_sim.clicked.connect(
            lambda: self._set_mode("sim"))
        self.btn_mode_file.clicked.connect(
            lambda: self._set_mode("file"))
        self.btn_mode_live.clicked.connect(
            lambda: self._set_mode("live"))
        src_layout.addLayout(mode_layout)

        # File import panel
        self.file_panel = QWidget()
        fp_layout = QVBoxLayout(self.file_panel)
        fp_layout.setContentsMargins(0, 4, 0, 0)
        fp_layout.setSpacing(4)
        self.file_info = GlowLabel("No file loaded", C["gray"], 9)
        self.file_info.setWordWrap(True)
        self.btn_load_file = QPushButton("[ BROWSE FILE ]")
        self.btn_load_file.clicked.connect(self._load_file)
        fp_layout.addWidget(self.btn_load_file)
        fp_layout.addWidget(self.file_info)
        self.file_panel.hide()
        src_layout.addWidget(self.file_panel)

        # Live device panel
        self.live_panel = QWidget()
        lp_layout = QVBoxLayout(self.live_panel)
        lp_layout.setContentsMargins(0, 4, 0, 0)
        lp_layout.setSpacing(4)
        self.btn_scan_devices = QPushButton("[ SCAN DEVICES ]")
        self.btn_scan_devices.clicked.connect(self._scan_devices)
        self.device_combo = QComboBox()
        self.device_combo.addItem("-- No device --")
        self.device_status = GlowLabel(
            "PyVISA: " + ("OK" if PYVISA_AVAILABLE else "NOT INSTALLED"),
            C["green"] if PYVISA_AVAILABLE else C["red"], 9)
        lp_layout.addWidget(self.device_status)
        lp_layout.addWidget(self.btn_scan_devices)
        lp_layout.addWidget(self.device_combo)
        self.live_panel.hide()
        src_layout.addWidget(self.live_panel)

        left_layout.addWidget(src_group)

        # Attack simulation (sim mode only)
        self.atk_group = QGroupBox("ATTACK SIMULATION")
        atk_layout = QVBoxLayout(self.atk_group)
        atk_layout.addWidget(GlowLabel("Attack Type:", C["gray"], 9))
        self.atk_combo = QComboBox()
        for k, v in ATTACK_TYPES.items():
            self.atk_combo.addItem(v, k)
        self.atk_combo.currentIndexChanged.connect(
            self._on_attack_changed)
        atk_layout.addWidget(self.atk_combo)
        atk_layout.addSpacing(4)
        atk_layout.addWidget(GlowLabel("Intensity:", C["gray"], 9))
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(1, 10)
        self.intensity_slider.setValue(5)
        self.intensity_slider.valueChanged.connect(
            self._on_intensity_changed)
        atk_layout.addWidget(self.intensity_slider)
        self.intensity_lbl = GlowLabel("0.5", C["cyan"], 10)
        atk_layout.addWidget(self.intensity_lbl)
        left_layout.addWidget(self.atk_group)

        # Controls
        btn_group = QGroupBox("CONTROLS")
        btn_layout = QVBoxLayout(btn_group)
        self.btn_scan = QPushButton("[ START SCAN ]")
        self.btn_scan.clicked.connect(self._start_scan)
        self.btn_stop = QPushButton("[ STOP ]")
        self.btn_stop.clicked.connect(self._stop_scan)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_scan)
        btn_layout.addWidget(self.btn_stop)
        self.progress = QProgressBar()
        self.progress.setFixedHeight(18)
        btn_layout.addWidget(self.progress)
        left_layout.addWidget(btn_group)
        left_layout.addStretch()

        # ── Center panel : Trace + Metrics
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setSpacing(12)

        # Metrics row
        metrics_row = QHBoxLayout()
        self.card_auc   = MetricCard("AUC-ROC",    "--",   "",    C["cyan"])
        self.card_det   = MetricCard("DETECTED",   "--",   "/200",C["red"])
        self.card_mse   = MetricCard("AVG MSE",    "--",   "",    C["yellow"])
        self.card_thr   = MetricCard("THRESHOLD",  "--",   "",    C["green"])
        for card in [self.card_auc, self.card_det,
                     self.card_mse, self.card_thr]:
            metrics_row.addWidget(card)
        center_layout.addLayout(metrics_row)

        # Trace canvas
        trace_group = QGroupBox("TRACE ANALYSIS")
        tg_layout   = QVBoxLayout(trace_group)
        self.trace_canvas = TraceCanvas()
        tg_layout.addWidget(self.trace_canvas)
        center_layout.addWidget(trace_group)

        # Score histogram
        hist_group = QGroupBox("SCORE DISTRIBUTION")
        hg_layout  = QVBoxLayout(hist_group)
        self.hist_canvas = ScoreHistCanvas()
        hg_layout.addWidget(self.hist_canvas)
        center_layout.addWidget(hist_group)

        layout.addWidget(left)
        layout.addWidget(center)

        return w

    def _build_analyzer(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        label = GlowLabel(
            "TRACE ANALYZER — Analyse individuelle de traces ASCAD",
            C["cyan"], 11)
        layout.addWidget(label)

        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("[ < PREV ]")
        self.btn_next = QPushButton("[ NEXT > ]")
        self.btn_prev.clicked.connect(self._prev_trace)
        self.btn_next.clicked.connect(self._next_trace)
        self.trace_idx_lbl = GlowLabel("Trace: --/--", C["gray"], 10)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        nav_layout.addWidget(self.trace_idx_lbl)
        nav_layout.addStretch()

        # Attack injection
        nav_layout.addWidget(GlowLabel("Inject:", C["gray"], 9))
        self.analyzer_combo = QComboBox()
        for k, v in ATTACK_TYPES.items():
            self.analyzer_combo.addItem(v, k)
        self.analyzer_combo.currentIndexChanged.connect(
            self._update_analyzer)
        nav_layout.addWidget(self.analyzer_combo)
        layout.addLayout(nav_layout)

        # Two canvases side by side
        canvases = QHBoxLayout()
        orig_group = QGroupBox("TRACE ORIGINALE")
        og_layout  = QVBoxLayout(orig_group)
        self.analyzer_orig = TraceCanvas()
        og_layout.addWidget(self.analyzer_orig)
        canvases.addWidget(orig_group)

        atk_group = QGroupBox("TRACE MODIFIEE (ATTAQUE SIMULEE)")
        ag_layout = QVBoxLayout(atk_group)
        self.analyzer_atk = TraceCanvas()
        ag_layout.addWidget(self.analyzer_atk)
        canvases.addWidget(atk_group)
        layout.addLayout(canvases)

        # Info
        self.analyzer_info = GlowLabel("", C["gray"], 9)
        layout.addWidget(self.analyzer_info)
        layout.addStretch()

        return w

    def _build_about(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(30, 20, 30, 20)

        about_text = """
[ SIDESHIELD v1.0 ]
Side-Channel Attack Detection System
─────────────────────────────────────────────────────────────

ARCHITECTURE
  Model     : Convolutional Autoencoder 1D
  Encoder   : Conv1D (1→32→64→128) + FC Bottleneck (64-dim)
  Decoder   : ConvTranspose1D (128→64→32→1)
  Parameters: ~1.5M
  Device    : """ + str(DEVICE).upper() + """

DETECTION METHOD
  Training  : Unsupervised — normal traces only (ASCAD profiling)
  Anomaly   : Reconstruction error (MSE) > threshold
  Threshold : 95th percentile of normal reconstruction errors

SIMULATED ATTACK TYPES
  Type 1    : Gaussian Noise    — Active EM jamming
  Type 2    : Desynchronization — Clock jitter / timing attack
  Type 3    : Amplitude Scaling — Voltage fault injection
  Type 4    : Spike Injection   — EM Fault Injection (EMFI)

PERFORMANCE
  AUC-ROC   : 0.9997
  Accuracy  : 97.44%
  F1-Score  : 97.50%
  Recall    : 99.88%

DATASET
  Source    : ASCAD (ANSSI + Thales Group, 2019)
  Target    : AES-128 on ATMega8515 microcontroller
  Traces    : 700 time samples per trace

AUTHOR
  Ali — Computer Engineering, Cybersecurity
  EPI Digital School — Sousse, Tunisia
  ResearchGate | GitHub | LinkedIn
─────────────────────────────────────────────────────────────
        """

        text_lbl = QLabel(about_text)
        text_lbl.setStyleSheet(f"""
            color: {C['gray']};
            font-family: 'Courier New';
            font-size: 11px;
            line-height: 1.6;
        """)
        text_lbl.setWordWrap(True)

        scroll = QScrollArea()
        scroll.setWidget(text_lbl)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(scroll)

        return w

    # ── LOGIC ─────────────────────────────────
    def _load_model(self):
        try:
            ae = SideShieldAE(LATENT_DIM).to(DEVICE)
            ckpt = torch.load(MODEL_PATH, map_location=DEVICE,
                              weights_only=False)
            ae.load_state_dict(ckpt["model_state"])
            ae.eval()
            self.model     = ae
            self.threshold = float(ckpt["threshold"])
            self.norm_mean = ckpt["norm_mean"]
            self.norm_std  = ckpt["norm_std"]

            # Charger les traces ASCAD
            db = os.path.join(DB_PATH, "ASCAD.h5")
            with h5py.File(db, "r") as f:
                X = f["Profiling_traces/traces"][:].astype(np.float32)
            X = (X - self.norm_mean) / np.where(
                self.norm_std == 0, 1e-8, self.norm_std)
            self.traces  = X
            self.scan_idx = 0

            # Mettre à jour UI
            self.model_status.setText("[ MODEL: READY ]")
            self.model_status.setStyleSheet(
                f"color: {C['green']}; font-size: 10px; font-family: 'Courier New';")
            self.card_thr.update_value(f"{self.threshold:.5f}")
            self.indicator.set_status("normal")
            self.status_bar.showMessage(
                f"  Model loaded | Threshold: {self.threshold:.6f} | "
                f"Device: {DEVICE} | Traces: {len(self.traces)}")

            # Afficher première trace
            self._update_analyzer()

        except Exception as e:
            self.model_status.setText("[ MODEL: ERROR ]")
            self.model_status.setStyleSheet(
                f"color: {C['red']}; font-size: 10px; font-family: 'Courier New';")
            self.status_bar.showMessage(f"  Error loading model: {e}")

    def _on_attack_changed(self, idx):
        pass

    def _on_intensity_changed(self, val):
        self.intensity_lbl.setText(f"{val / 10:.1f}")

    # ── MODE SWITCHING ─────────────────────────
    def _set_mode(self, mode):
        self.input_mode = mode
        self.btn_mode_sim.setChecked(mode == "sim")
        self.btn_mode_file.setChecked(mode == "file")
        self.btn_mode_live.setChecked(mode == "live")
        self.file_panel.setVisible(mode == "file")
        self.live_panel.setVisible(mode == "live")
        self.atk_group.setVisible(mode == "sim")

        colors = {"sim":"cyan","file":"yellow","live":"green"}
        c = C[colors[mode]]
        for btn, m in [(self.btn_mode_sim,"sim"),
                       (self.btn_mode_file,"file"),
                       (self.btn_mode_live,"live")]:
            if m == mode:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {c}22;
                        color: {c};
                        border: 1px solid {c};
                        padding: 4px 8px;
                        font-family: 'Courier New';
                        font-size: 10px;
                        border-radius: 3px;
                    }}
                """)
            else:
                btn.setStyleSheet("")

        self.status_bar.showMessage(
            f"  Mode: {mode.upper()} | "
            f"{'Select attack type' if mode=='sim' else 'Load a file' if mode=='file' else 'Connect a device'}")

    # ── FILE IMPORT ────────────────────────────
    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Trace File",
            os.path.join(DESKTOP, "ASCAD", "ASCAD_databases"),
            "Trace Files (*.h5 *.npy *.csv *.npz);;All Files (*)"
        )
        if not path:
            return

        try:
            ext = os.path.splitext(path)[1].lower()
            fname = os.path.basename(path)

            if ext == ".h5":
                # Essaie plusieurs clés communes
                with h5py.File(path, "r") as f:
                    keys = list(f.keys())
                    # ASCAD format
                    if "Profiling_traces" in keys:
                        X = f["Profiling_traces/traces"][:].astype(np.float32)
                    elif "Attack_traces" in keys:
                        X = f["Attack_traces/traces"][:].astype(np.float32)
                    else:
                        # Cherche le premier dataset numérique
                        def find_traces(name, obj):
                            if isinstance(obj, h5py.Dataset) and \
                               len(obj.shape) == 2:
                                raise StopIteration(name)
                        try:
                            f.visititems(find_traces)
                            raise ValueError("No 2D dataset found in HDF5")
                        except StopIteration as e:
                            X = f[str(e)][:].astype(np.float32)

            elif ext == ".npy":
                X = np.load(path).astype(np.float32)
                if X.ndim == 1:
                    X = X[np.newaxis, :]

            elif ext == ".npz":
                data = np.load(path)
                key  = list(data.keys())[0]
                X    = data[key].astype(np.float32)
                if X.ndim == 1:
                    X = X[np.newaxis, :]

            elif ext == ".csv":
                X = np.loadtxt(path, delimiter=",",
                               dtype=np.float32)
                if X.ndim == 1:
                    X = X[np.newaxis, :]

            else:
                raise ValueError(f"Format non supporté : {ext}")

            # Adapter la longueur des traces
            if X.shape[1] != TRACE_LEN:
                # Interpolation ou truncation
                if X.shape[1] > TRACE_LEN:
                    X = X[:, :TRACE_LEN]
                else:
                    pad = np.zeros((X.shape[0],
                                    TRACE_LEN - X.shape[1]),
                                   dtype=np.float32)
                    X = np.concatenate([X, pad], axis=1)

            # Normalisation z-score
            if self.norm_mean is not None:
                mean = self.norm_mean
                std  = np.where(self.norm_std == 0, 1e-8, self.norm_std)
            else:
                mean = X.mean(axis=0)
                std  = np.where(X.std(axis=0) == 0, 1e-8,
                                X.std(axis=0))

            X_norm = (X - mean) / std

            # Remplacer les traces courantes
            self.traces   = X_norm
            self.scan_idx = 0

            self.file_info.setText(
                f"{fname}\n{X.shape[0]} traces x {X.shape[1]} pts")
            self.file_info.setStyleSheet(
                f"color: {C['green']}; font-size: 9px; "
                f"font-family: 'Courier New';")
            self.status_bar.showMessage(
                f"  File loaded: {fname} | "
                f"{X.shape[0]} traces x {X.shape[1]} points")

            # Affiche la première trace
            self._update_analyzer()

        except Exception as e:
            self.file_info.setText(f"Error: {str(e)[:60]}")
            self.file_info.setStyleSheet(
                f"color: {C['red']}; font-size: 9px; "
                f"font-family: 'Courier New';")
            self.status_bar.showMessage(f"  Load error: {e}")

    # ── LIVE DEVICE ────────────────────────────
    def _scan_devices(self):
        self.device_combo.clear()

        if not PYVISA_AVAILABLE:
            self.device_combo.addItem("PyVISA not installed")
            QMessageBox.information(
                self, "PyVISA",
                "Install PyVISA to connect real instruments:\n\n"
                "pip install pyvisa pyvisa-py\n\n"
                "Compatible: Oscilloscopes Picoscope, Rigol, "
                "Keysight, Tektronix via USB/GPIB/LAN"
            )
            return

        try:
            rm      = pyvisa.ResourceManager()
            devices = rm.list_resources()
            if devices:
                for d in devices:
                    self.device_combo.addItem(d)
                self.status_bar.showMessage(
                    f"  Found {len(devices)} VISA device(s)")
            else:
                self.device_combo.addItem("-- No VISA device found --")
                self.status_bar.showMessage(
                    "  No VISA devices detected. "
                    "Check USB connection and drivers.")
        except Exception as e:
            self.device_combo.addItem(f"Error: {str(e)[:40]}")
            self.status_bar.showMessage(f"  VISA scan error: {e}")

    def _acquire_from_device(self, n_traces=10):
        """
        Acquisition de traces depuis un oscilloscope VISA.
        Protocole SCPI standard — compatible Rigol, Keysight, Tektronix.
        """
        if not PYVISA_AVAILABLE:
            return None
        device_str = self.device_combo.currentText()
        if not device_str or "No" in device_str or "Error" in device_str:
            return None
        try:
            rm   = pyvisa.ResourceManager()
            inst = rm.open_resource(device_str)
            inst.timeout = 5000

            # Identification
            idn = inst.query("*IDN?").strip()
            self.status_bar.showMessage(f"  Connected: {idn}")

            traces = []
            for i in range(n_traces):
                # Déclenche acquisition
                inst.write(":RUN")
                time.sleep(0.1)
                inst.write(":STOP")

                # Lit les données waveform (SCPI standard)
                inst.write(":WAV:SOUR CHAN1")
                inst.write(":WAV:MODE RAW")
                inst.write(":WAV:FORM BYTE")
                raw = inst.query_binary_values(
                    ":WAV:DATA?", datatype='B',
                    container=np.array)

                # Normalise à TRACE_LEN points
                if len(raw) != TRACE_LEN:
                    raw = np.interp(
                        np.linspace(0, len(raw)-1, TRACE_LEN),
                        np.arange(len(raw)), raw)

                traces.append(raw.astype(np.float32))
                self.progress.setValue(int((i+1)/n_traces*100))

            inst.close()
            X = np.array(traces)

            # Normalisation
            mean = X.mean(axis=0)
            std  = np.where(X.std(axis=0)==0, 1e-8, X.std(axis=0))
            return (X - mean) / std

        except Exception as e:
            self.status_bar.showMessage(f"  Device error: {e}")
            return None

    def _start_scan(self):
        if self.model is None:
            return

        mode = getattr(self, 'input_mode', 'sim')

        # ── MODE LIVE : acquisition instrument réel
        if mode == "live":
            self.status_bar.showMessage("  Acquiring from device...")
            self.btn_scan.setEnabled(False)
            self.progress.setValue(0)
            X_live = self._acquire_from_device(n_traces=50)
            if X_live is None:
                QMessageBox.warning(
                    self, "Live Acquisition",
                    "No data acquired.\n\n"
                    "Make sure:\n"
                    "- Device is connected via USB\n"
                    "- PyVISA is installed\n"
                    "- Correct device is selected\n\n"
                    "Using simulation mode instead.")
                self.btn_scan.setEnabled(True)
                return
            self.traces = X_live
            self.status_bar.showMessage(
                f"  Acquired {len(X_live)} traces from device")

        # ── MODE FILE ou SIM : traces déjà chargées
        if self.traces is None:
            self.status_bar.showMessage(
                "  No traces loaded. Load a file first.")
            return

        self.scores_normal = []
        self.scores_attack = []
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)

        # En mode file/live → pas d'injection d'attaque
        atk_type  = self.atk_combo.currentData() \
                    if mode == "sim" else 0
        intensity = self.intensity_slider.value() / 10.0

        self.thread = AnalysisThread(
            self.model, self.traces, self.threshold,
            atk_type, intensity, self.norm_mean, self.norm_std
        )
        self.thread.result_ready.connect(self._on_scan_complete)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.start()
        mode_str = {"sim":"SIMULATION","file":"FILE","live":"LIVE"}[mode]
        self.status_bar.showMessage(
            f"  [{mode_str}] Scanning {min(200,len(self.traces))} traces...")

    def _stop_scan(self):
        if self.thread and self.thread.isRunning():
            self.thread.terminate()
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("  Scan stopped.")

    def _on_scan_complete(self, result):
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)

        det  = result["detections"]
        tot  = result["total"]
        rate = result["det_rate"]
        atk  = self.atk_combo.currentData()

        # Mise à jour métriques
        self.card_det.update_value(
            f"{det}",
            C["red"] if det > 0 else C["green"])
        self.card_mse.update_value(f"{result['mean_score']:.5f}")

        # Status indicator
        if atk == 0:
            self.indicator.set_status("normal")
        elif det > tot * 0.5:
            self.indicator.set_status("attack")
        else:
            self.indicator.set_status("normal")

        # Dernière trace analysée
        last_trace = result["traces"][-1]
        last_score = result["scores"][-1]
        is_atk     = last_score > self.threshold

        # Reconstruction
        x_t = torch.tensor(
            last_trace[None, None, :], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            x_r, _ = self.model(x_t)
        recon = x_r.cpu().squeeze().numpy()

        self.trace_canvas.plot_trace(
            last_trace, recon, last_score, self.threshold, is_atk)

        # Histogramme
        scores = result["scores"]
        if atk == 0:
            self.scores_normal = scores.tolist()
            sn = np.array(self.scores_normal)
            sa = np.array([])
        else:
            self.scores_attack = scores.tolist()
            sn = np.array(self.scores_normal) if self.scores_normal \
                 else np.array([self.threshold * 0.8] * 10)
            sa = np.array(self.scores_attack)

        self.hist_canvas.update_hist(sn, sa, self.threshold)

        self.status_bar.showMessage(
            f"  Scan complete | Detected: {det}/{tot} ({rate:.1f}%) | "
            f"Avg MSE: {result['mean_score']:.6f} | "
            f"Max MSE: {result['max_score']:.6f}")

    def _prev_trace(self):
        if self.traces is None:
            return
        self.scan_idx = max(0, self.scan_idx - 1)
        self._update_analyzer()

    def _next_trace(self):
        if self.traces is None:
            return
        self.scan_idx = min(len(self.traces) - 1, self.scan_idx + 1)
        self._update_analyzer()

    def _update_analyzer(self):
        if self.model is None or self.traces is None:
            return

        idx   = self.scan_idx
        trace = self.traces[idx]
        atk   = self.analyzer_combo.currentData() \
                if hasattr(self, 'analyzer_combo') else 0

        # Score original
        x_t = torch.tensor(
            trace[None, None, :], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            x_r, _ = self.model(x_t)
        recon_orig = x_r.cpu().squeeze().numpy()
        score_orig = float(((trace - recon_orig) ** 2).mean())

        self.analyzer_orig.plot_trace(
            trace, recon_orig, score_orig, self.threshold,
            score_orig > self.threshold)

        if atk > 0 and self.norm_mean is not None:
            # Dénormalise → injecte → renormalise
            trace_raw = trace * self.norm_std + self.norm_mean
            atk_raw   = inject_attack(
                trace_raw, atk,
                self.intensity_slider.value() / 10.0
                if hasattr(self, 'intensity_slider') else 0.5)
            atk_n = (atk_raw - self.norm_mean) / np.where(
                self.norm_std == 0, 1e-8, self.norm_std)

            x_a = torch.tensor(
                atk_n[None, None, :], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                x_ra, _ = self.model(x_a)
            recon_atk = x_ra.cpu().squeeze().numpy()
            score_atk = float(((atk_n - recon_atk) ** 2).mean())

            self.analyzer_atk.plot_trace(
                atk_n, recon_atk, score_atk, self.threshold,
                score_atk > self.threshold)
        else:
            self.analyzer_atk.plot_trace(trace, recon_orig,
                                          score_orig, self.threshold,
                                          False)

        if hasattr(self, 'trace_idx_lbl'):
            self.trace_idx_lbl.setText(
                f"Trace: {idx+1}/{len(self.traces)}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("SideShield")

    window = SideShieldGUI()
    window.show()

    sys.exit(app.exec_())
