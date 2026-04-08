# MagDL

**Lightweight Deep Learning for Identification and Suppression of Subway Interference in Geomagnetic Observations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)

> Huo, Q., Wang, X., et al.  
> *Computers & Geosciences*, 2026  
> DOI: [to be assigned]

---

## Overview

Subway operations generate electromagnetic interference (EMI) that contaminates nearby geomagnetic observatory recordings, particularly in the Z-component. This repository provides inference code and pre-trained models for a two-stage processing pipeline:

1. **Identification (TinyCNN):** A lightweight binary classifier (102K parameters) that detects whether a 1-hour geomagnetic observation window contains subway interference. Achieves F1-score of 0.9997 on validation data.

2. **Suppression (ResidualCNN):** An encoder–decoder network (413K parameters) that removes subway interference from contaminated windows while preserving the natural geomagnetic signal. Uses residual learning to estimate the noise component directly.

### Pipeline

```
Raw geomagnetic data (1 Hz, Z-component)
        │
        ▼
┌─────────────────────┐
│  TinyCNN (102K)     │  Input: 1-hour window [1, 1, 3600]
│  Identification     │  Output: probability of interference
│  threshold τ = 0.80 │
└────────┬────────────┘
         │
    detected?
    ├── No  → Keep original signal
    └── Yes ↓
┌─────────────────────┐
│  ResidualCNN (413K) │  Input: contaminated window [1, 1, 3600]
│  Suppression        │  Output: denoised signal
│  y = x − n̂         │
└─────────────────────┘
         │
         ▼
   Cleaned geomagnetic data
```

---

## Repository Structure

```
MagDL/
├── tinycnn_model.py               # TinyCNN model definition (identification)
├── dl_denoise_only.py             # ResidualCNN model + denoising pipeline (suppression)
├── hzt_data_reader.py             # Geomagnetic station data reader utility
├── quick_test.py                  # Quick test script (CAGEO requirement)
├── sample_data.npz                # Real geomagnetic sample (2 windows, 12 KB)
├── checkpoints/
│   ├── best_tinycnn_v3.pth        # Pre-trained TinyCNN (1.2 MB)
│   └── best_residual_model.pth    # Pre-trained ResidualCNN (4.8 MB)
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/6argrajar/MagDL.git
cd MagDL

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10.0
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.5.0

---

## Quick Test

Run the quick test to verify that all models load and produce correct outputs:

```bash
python quick_test.py
```

**Expected output:**

```
MagDL — Quick Test

=================================================================
TEST 1: TinyCNN — Subway Interference Identification
=================================================================
  Architecture: 48->96->192->192 channels, SE(r=8)
  Parameters:   102,301
  [PASS] Parameter count = 102,301
  Checkpoint:   best_tinycnn_v3.pth (epoch 41)
  Threshold:    0.80
  ...
  [PASS] TinyCNN inference completed successfully

=================================================================
TEST 2: ResidualCNN — Subway Interference Suppression
=================================================================
  ...
  [PASS] ResidualCNN inference completed successfully

=================================================================
TEST 3: Full Pipeline — Identify then Suppress
=================================================================
  ...
  [PASS] Full pipeline processed 24 windows successfully

=================================================================
TEST 4: Real Data — BJT Station (2024-02-01)
=================================================================
  Clean window:        prob=0.000000  -> CLEAN
  Contaminated window: prob=1.000000  -> INTERFERENCE
  [PASS] Identification correct on both real windows
  Smoothness improvement:    57.0x
  [PASS] Suppression completed on real contaminated window

=================================================================
SUMMARY: 4/4 tests passed
=================================================================
```

Tests 1–3 use **synthetic signals** (no external data needed). Test 4 uses two real 1-hour windows from Beijing Seismic Station included in `sample_data.npz` (12 KB).

---

## Usage

### 1. Subway Interference Identification

```python
import torch
from tinycnn_model import load_model, mad_normalize

# Load pre-trained model
model, threshold, metadata = load_model("checkpoints/best_tinycnn_v3.pth")
# model is in eval mode with sigmoid output (probabilities)

# Prepare input: 1-hour geomagnetic Z-component at 1 Hz (3600 samples)
signal = your_data  # shape: (3600,)
x = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)  # [1, 1, 3600]
x = mad_normalize(x)  # MAD normalization

# Inference
with torch.no_grad():
    prob = model(x).item()

if prob >= threshold:  # threshold = 0.80
    print("Subway interference DETECTED")
else:
    print("CLEAN signal")
```

### 2. Subway Interference Suppression

```python
import torch
import numpy as np
from dl_denoise_only import ResidualDenoisingModel

# Load pre-trained model
model = ResidualDenoisingModel(channels=64)
ckpt = torch.load("checkpoints/best_residual_model.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Prepare input (normalize to zero-mean, unit-variance)
signal = your_data  # shape: (3600,)
mean_val, std_val = signal.mean(), signal.std() + 1e-8
signal_norm = (signal - mean_val) / std_val
x = torch.FloatTensor(signal_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, 3600]

# Inference: model estimates noise, denoised = input - noise
with torch.no_grad():
    estimated_noise, denoised = model(x)

# Convert back to original scale
denoised_signal = denoised.squeeze().numpy() * std_val + mean_val
```

### 3. Processing Daily Data

For full-day continuous observations (86,400 samples at 1 Hz), the framework uses overlapping 1-hour windows with Hann-window weighted fusion:

```bash
python dl_denoise_only.py --station BJT --date 2024-07-13 \
    --data_path /path/to/your/BJT_20240713.txt \
    --model checkpoints/best_residual_model.pth
```

---

## Model Details

### TinyCNN (Identification)

| Property | Value |
|----------|-------|
| Task | Binary classification (interference vs. clean) |
| Input | [B, 1, 3600] — 1-hour Z-component at 1 Hz |
| Output | [B, 1] — probability score |
| Channels | 48 → 96 → 192 → 192 |
| Convolutions | Standard Conv + 3× Depthwise Separable Conv |
| Attention | SE blocks (reduction ratio r = 8) |
| Classifier | FC(192→96→1) |
| Parameters | 102,301 |
| Threshold | τ = 0.80 (optimized on validation set) |
| F1-score | 0.9997 |

### ResidualCNN (Suppression)

| Property | Value |
|----------|-------|
| Task | Signal denoising (residual learning) |
| Input | [B, 1, 3600] — contaminated window |
| Output | [B, 1, 3600] — denoised signal |
| Architecture | Encoder (3 layers) – Decoder (3 layers) |
| Channels | 64 → 128 → 128 → 128 → 64 → 1 |
| Kernel sizes | 15, 11, 7, 7, 11, 15 |
| Parameters | 413,057 |
| Training labels | VMD-generated pseudo-labels |

---

## Pre-trained Checkpoints

Both checkpoints are included in the `checkpoints/` directory:

| File | Size | Description |
|------|------|-------------|
| `best_tinycnn_v3.pth` | 1.2 MB | TinyCNN V3 (epoch 41, F1=0.9997) |
| `best_residual_model.pth` | 4.8 MB | ResidualCNN (encoder-decoder, 413K params) |

---

## Training Data

The models were trained on geomagnetic Z-component data from:
- **Beijing Seismic Station (BJT):** 612 days
- **Hangzhou Botanical Garden Station (HZT):** 186 days

Cross-station generalization was validated on **Taiyuan Station (TAY)** (589 days), an unseen station not used during training.

---

## Citation

```bibtex
@article{wang2026subway,
  title={Lightweight Deep Learning for Identification and Suppression 
         of Subway Interference in Geomagnetic Observations},
  author={Huo, Qiaoling and Wang, Xizhen and Ma, Xinxin and Zhang, Suqin},
  journal={Computers \& Geosciences},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

For questions about the methodology or for academic collaboration:
- Email: huoqiaoling22@outlook.com
