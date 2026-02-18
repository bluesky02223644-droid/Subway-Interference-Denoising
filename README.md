# Pre-trained Model Weights

## 📥 How to Obtain the Model

The pre-trained ResidualCNN model (`best_residual_model.pth`) is **not included in this repository** due to file size constraints.

### Option 1: Download from Paper Supplementary Materials (Recommended)

1. Go to the journal's article page
2. Navigate to "Supplementary Materials" section
3. Download `best_residual_model.pth` (approximately 1.6 MB)
4. Place it in this directory (`checkpoints/`)

### Option 2: Download from GitHub Releases

If available, download from:

- [GitHub Releases](https://github.com/your-username/repo/releases)
- Look for the latest release (v1.0.0)
- Download `best_residual_model.pth`

### Option 3: Contact Author

If you cannot access the model through the above methods:

- Email: <huoqiaoling22@outlook.com>
- Subject: "Request for Pre-trained Model - Subway Interference Suppression"
- Include: Your affiliation and intended use

---

## ✅ Verification

After downloading, verify the file:

```bash
ls -lh checkpoints/best_residual_model.pth
```

**Expected output:**

- File size: ~1.6 MB
- SHA256 checksum: (will be provided upon model release)

---

## 📋 Model Details

- **Architecture:** ResidualCNN (Encoder-Decoder with residual learning)
- **Parameters:** 413,368
- **Input Shape:** (batch_size, 1, 3600)
- **Output Shape:** (batch_size, 1, 3600)
- **Training Data:** 798 days (Beijing 612 + Hangzhou 186)
- **Framework:** PyTorch 1.10+

---

## 🔒 Model License

The pre-trained model is provided for **academic research purposes only**.

Project Description: > This repository provides the implementation of a lightweight identification-driven framework for geomagnetic subway interference mitigation, as described in our paper.

Key Features:

TinyCNN for real-time interference identification.

ResidualCNN for frequency-targeted selective denoising.

Adaptive boundary handling for continuous multi-day observations.

For commercial use or redistribution, please contact:

- Email: <huoqiaoling22@outlook.com>

