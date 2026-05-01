# SideShield — Side-Channel Attack Detection System

**Status:** Completed  
**Tech Stack:** PyTorch · Python · PyQt5 · ASCAD Dataset (Thales / ANSSI)

---

## Overview

**SideShield** is an advanced side-channel anomaly detection system based on a **1D Convolutional Autoencoder** trained in an **unsupervised** manner on electromagnetic / power traces captured from an AVR microcontroller running **AES-128**.

The system is designed to detect, in real time, multiple realistic physical perturbations commonly used in **Side-Channel Attacks (SCA)** and **Fault Injection Attacks (FIA)** against embedded cryptographic devices.

Unlike supervised approaches, SideShield does **not require labeled attack samples**, making it highly practical for industrial deployment.

---

## Problem Statement

Embedded cryptographic implementations such as:

- Smart Cards  
- Secure Elements  
- Microcontrollers  
- Hardware Security Modules (HSMs)  
- IoT Devices  

often leak physical information during execution through:

- Power Consumption  
- Electromagnetic Radiation  
- Timing Variations  

Attackers exploit these leakages or actively disturb the environment using techniques such as:

- Electromagnetic Jamming  
- Voltage Glitching  
- Clock Manipulation  
- EM Fault Injection  

These perturbations alter side-channel traces in characteristic ways.

**SideShield detects such anomalies instantly using AI.**

---

## Core Idea

The model learns the statistical distribution of **normal cryptographic traces**.

When an abnormal trace is observed:

- Reconstruction quality drops  
- Mean Squared Error (MSE) rises  
- If MSE exceeds the learned threshold → anomaly flagged



Technical Architecture
Neural Network
1D Convolutional Autoencoder
~1.5 Million Parameters

Encoder
Input: [1 × 700]

Conv1D(1 → 32, kernel=11)
→ BatchNorm
→ SiLU
→ AvgPool(2)        [700 → 350]

Conv1D(32 → 64, kernel=7)
→ BatchNorm
→ SiLU
→ AvgPool(2)        [350 → 175]

Conv1D(64 → 128, kernel=5)
→ BatchNorm
→ SiLU
→ AvgPool(2)        [175 → 87]

Flatten

FC(128×87 → 64)
→ LayerNorm
→ Tanh

Bottleneck
Latent Space = 64 Dimensions

Compact representation of normal side-channel behavior.

Decoder (Symmetric)
FC(64 → 128×87)

ConvTranspose1D × 3

Output: [1 × 700]
Dataset
ASCAD Dataset (ANSSI + Thales Group, 2019)

Widely recognized benchmark for side-channel research.

Hardware Target
ATMega8515 AVR Microcontroller
Crypto Algorithm
AES-128 with masking countermeasures
Data Size
50,000 Profiling Traces
10,000 Attack Traces
700 Time Samples / Trace
int8 format
Simulated Attack Scenarios
Attack Type	Simulation	Realistic Physical Meaning
Gaussian Noise	σ × [0.5 – 1.5]	Active EM Jamming
Desynchronization	Shift 10–80 samples	Clock Jitter / Timing Drift
Amplitude Scaling	× [1.8 – 3.5]	Voltage Glitch / Power Distortion
Spike EMFI	Local Gaussian spike × [3 – 8]σ	Electromagnetic Fault Injection
Performance Results
Metric	Score
AUC-ROC	0.9997
Accuracy	97.44%
F1-Score	97.50%
Recall	99.88%
Score Separation	189σ
Detection Rate by Attack Type
Attack	Detection Rate
Gaussian Noise	100%
Desynchronization	100%
Amplitude Scaling	99.5%
Spike EMFI	100%
Graphical User Interface

Professional desktop dashboard built with PyQt5.

Mode 1 — SIMULATION MODE

Generate synthetic attacks with real-time control.

Features:

Adjustable attack intensity
Original vs reconstructed trace comparison
Live MSE anomaly score
Score histogram visualization
Mode 2 — FILE MODE

Import external traces from:

.h5
.npy
.npz
.csv

Features:

Automatic length adaptation via interpolation
Compatible with most SCA datasets
Mode 3 — LIVE MODE

Real hardware acquisition using PyVISA + SCPI

Supported instruments:

Rigol
Keysight
Tektronix
PicoScope

Interfaces:

USB
GPIB
LAN
Skills Demonstrated
Applied Cryptography
AES S-Box Leakage Concepts
Hamming Weight Model
First-Order Masking
Points of Interest (POI)
Side-Channel Security
Artificial Intelligence
Autoencoders
Unsupervised Learning
Threshold Detection
ROC / F1 Metrics
GPU Training
Mixed Precision (AMP)
Hardware Security
Electromagnetic Leakage Modeling
Voltage Fault Injection
Clock Jitter Effects
EMFI Disturbances
Engineering

Complete end-to-end pipeline:

Signal Acquisition
→ Preprocessing
→ Deep Learning Model
→ Detection Engine
→ GUI Dashboard
→ Deployment
Industrial Relevance

This project directly targets the needs of Hardware Security Teams in organizations such as:

Thales
Leonardo
Secure IC Vendors
Defense Electronics Companies
Payment Card Manufacturers
Why It Matters

In real industrial environments:

Attack datasets are rarely labeled
New attack methods constantly appear
Hardware behavior changes over time

Therefore, unsupervised anomaly detection is a highly valuable real-world approach.

Why This Project Is Strong

✅ Combines Cybersecurity + AI + Hardware Security
✅ Uses real-world side-channel benchmark data
✅ Practical industrial use case
✅ Professional GUI included
✅ Excellent metrics
✅ Advanced research-level topic

Author

Ali Derouiche
Cybersecurity Engineer
AI & Quantum Security Researcher
