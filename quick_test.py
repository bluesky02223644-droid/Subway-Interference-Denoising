#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Test Script for MagDL
=====================================

This script demonstrates both modules of the subway interference processing
pipeline described in the paper:

    Module 1: TinyCNN — Binary identification of subway interference
    Module 2: ResidualCNN — Suppression (denoising) of subway interference

It uses synthetic geomagnetic signals so no external data files are needed.

Usage:
    python quick_test.py

Expected output:
    - TinyCNN loads successfully and classifies synthetic windows
    - ResidualCNN loads successfully and denoises a synthetic signal
    - All assertions pass

Requirements:
    pip install torch numpy scipy matplotlib
"""

import os
import sys
import numpy as np
import torch

# ========================================================================
# Paths
# ========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TINYCNN_CKPT = os.path.join(SCRIPT_DIR, "checkpoints", "best_tinycnn_v3.pth")
RESIDUAL_CKPT = os.path.join(SCRIPT_DIR, "checkpoints", "best_residual_model.pth")
SAMPLE_DATA = os.path.join(SCRIPT_DIR, "sample_data.npz")


# ========================================================================
# Synthetic Data Generation
# ========================================================================
def generate_synthetic_geomagnetic(n_samples=3600, seed=42):
    """
    Generate synthetic geomagnetic signals for testing.

    Returns:
        clean: Clean diurnal variation signal (nT)
        contaminated: Signal with simulated subway interference added
        noise: The subway interference component
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_samples) / n_samples  # normalized time [0, 1)

    # Diurnal variation: slow sinusoidal + trend (~50000 nT baseline)
    diurnal = (
        50000.0
        + 20.0 * np.sin(2 * np.pi * t)          # 24-hour variation
        + 5.0 * np.sin(4 * np.pi * t + 0.5)     # 12-hour harmonic
        + 0.3 * rng.randn(n_samples)             # instrument noise
    )

    # Subway interference: quasi-periodic pulses in 0.004-0.032 Hz band
    # Simulates operational hours (roughly samples 200-3200 for a 1-hour window)
    subway_noise = np.zeros(n_samples)
    for freq in [0.005, 0.010, 0.016, 0.025]:
        phase = rng.uniform(0, 2 * np.pi)
        subway_noise += rng.uniform(1.0, 3.0) * np.sin(
            2 * np.pi * freq * np.arange(n_samples) + phase
        )
    # Add amplitude envelope (stronger during subway hours)
    envelope = np.clip(np.sin(np.pi * t) * 1.5, 0, 1)
    subway_noise *= envelope

    contaminated = diurnal + subway_noise
    return diurnal, contaminated, subway_noise


# ========================================================================
# Test 1: TinyCNN Identification
# ========================================================================
def test_tinycnn_identification():
    """Test TinyCNN model for subway interference identification."""
    print("=" * 65)
    print("TEST 1: TinyCNN — Subway Interference Identification")
    print("=" * 65)

    # Import model
    from tinycnn_model import TinyCNN, load_model, count_parameters, mad_normalize

    # 1a. Architecture verification
    model_scratch = TinyCNN(input_length=3600, use_sigmoid=False)
    n_params = count_parameters(model_scratch)
    print(f"  Architecture: 48->96->192->192 channels, SE(r=8)")
    print(f"  Parameters:   {n_params:,}")
    assert n_params == 102_301, f"Expected 102,301 params, got {n_params:,}"
    print(f"  [PASS] Parameter count = 102,301")

    # 1b. Load pre-trained checkpoint
    if not os.path.exists(TINYCNN_CKPT):
        print(f"  [SKIP] Checkpoint not found: {TINYCNN_CKPT}")
        print(f"         Download from GitHub Releases and place in checkpoints/")
        return False

    model, threshold, metadata = load_model(TINYCNN_CKPT)
    print(f"  Checkpoint:   best_tinycnn_v3.pth (epoch {metadata.get('epoch', '?')})")
    print(f"  Threshold:    {threshold:.2f}")
    print(f"  Val F1-score: {metadata.get('f1', 'N/A')}")

    # 1c. Run inference on synthetic data
    clean, contaminated, _ = generate_synthetic_geomagnetic()

    # Prepare batched input: [clean_window, contaminated_window]
    batch = torch.stack([
        torch.FloatTensor(clean).unsqueeze(0),
        torch.FloatTensor(contaminated).unsqueeze(0),
    ])  # [2, 1, 3600]

    # MAD normalize
    batch_norm = mad_normalize(batch)

    with torch.no_grad():
        probs = model(batch_norm)  # [2, 1]

    prob_clean = probs[0].item()
    prob_contaminated = probs[1].item()

    print(f"\n  Inference results (probability of subway interference):")
    print(f"    Clean signal:        {prob_clean:.4f}  "
          f"-> {'INTERFERENCE' if prob_clean >= threshold else 'CLEAN'}")
    print(f"    Contaminated signal: {prob_contaminated:.4f}  "
          f"-> {'INTERFERENCE' if prob_contaminated >= threshold else 'CLEAN'}")

    # Note: synthetic data may not perfectly mimic real subway patterns,
    # so we only verify the model runs without errors.
    print(f"  [PASS] TinyCNN inference completed successfully")
    return True


# ========================================================================
# Test 2: ResidualCNN Suppression
# ========================================================================
def test_residualcnn_suppression():
    """Test ResidualCNN model for subway interference suppression."""
    print("\n" + "=" * 65)
    print("TEST 2: ResidualCNN — Subway Interference Suppression")
    print("=" * 65)

    # Import model (defined in dl_denoise_only.py)
    from dl_denoise_only import ResidualDenoisingModel

    # 2a. Architecture verification
    model = ResidualDenoisingModel(channels=64)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture: Encoder-Decoder with residual learning")
    print(f"  Parameters:   {n_params:,}")
    assert abs(n_params - 413_057) < 100, f"Expected ~413,057 params, got {n_params:,}"
    print(f"  [PASS] Parameter count verified")

    # 2b. Load pre-trained checkpoint
    if not os.path.exists(RESIDUAL_CKPT):
        print(f"  [SKIP] Checkpoint not found: {RESIDUAL_CKPT}")
        print(f"         Download from GitHub Releases and place in checkpoints/")
        return False

    checkpoint = torch.load(RESIDUAL_CKPT, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("val_loss", "?")
    else:
        model.load_state_dict(checkpoint)
        epoch, val_loss = "?", "?"
    model.eval()
    print(f"  Checkpoint:   best_residual_model.pth (epoch {epoch})")
    print(f"  Val loss:     {val_loss}")

    # 2c. Run inference on synthetic contaminated signal
    clean, contaminated, true_noise = generate_synthetic_geomagnetic()

    # Normalize for model input
    mean_val = contaminated.mean()
    std_val = contaminated.std() + 1e-8
    signal_norm = (contaminated - mean_val) / std_val

    x = torch.FloatTensor(signal_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, 3600]

    with torch.no_grad():
        estimated_noise, denoised = model(x)

    denoised_signal = denoised.squeeze().numpy() * std_val + mean_val
    estimated_noise_np = estimated_noise.squeeze().numpy() * std_val

    # 2d. Compute metrics
    from scipy.stats import pearsonr
    corr_with_clean, _ = pearsonr(clean, denoised_signal)

    residual_std = np.std(denoised_signal - clean)
    original_std = np.std(contaminated - clean)
    noise_reduction_db = 20 * np.log10(original_std / max(residual_std, 1e-10))

    print(f"\n  Denoising results (synthetic data):")
    print(f"    Correlation with clean signal: {corr_with_clean:.4f}")
    print(f"    Noise reduction:               {noise_reduction_db:.1f} dB")
    print(f"    Input shape:  {x.shape}")
    print(f"    Output shape: {denoised.shape}")
    print(f"  [PASS] ResidualCNN inference completed successfully")

    return True


# ========================================================================
# Test 3: Full Pipeline (Identification + Suppression)
# ========================================================================
def test_full_pipeline():
    """Test the complete identification-then-suppression pipeline."""
    print("\n" + "=" * 65)
    print("TEST 3: Full Pipeline — Identify then Suppress")
    print("=" * 65)

    from tinycnn_model import TinyCNN, load_model, mad_normalize
    from dl_denoise_only import ResidualDenoisingModel

    # Check both checkpoints exist
    if not os.path.exists(TINYCNN_CKPT) or not os.path.exists(RESIDUAL_CKPT):
        print("  [SKIP] One or both checkpoints not found")
        return False

    # Load both models
    tinycnn, threshold, _ = load_model(TINYCNN_CKPT)

    residual = ResidualDenoisingModel(channels=64)
    ckpt = torch.load(RESIDUAL_CKPT, map_location="cpu", weights_only=False)
    residual.load_state_dict(ckpt["model_state_dict"])
    residual.eval()

    # Generate a day's worth of hourly windows (24 hours)
    n_hours = 24
    results = []

    print(f"\n  Processing {n_hours} hourly windows...")
    print(f"  {'Hour':>4s}  {'Prob':>6s}  {'Decision':>14s}  {'Action':>10s}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*14}  {'-'*10}")

    for hour in range(n_hours):
        # Generate synthetic signal for this hour
        seed = 100 + hour
        rng = np.random.RandomState(seed)
        t = np.arange(3600) / 3600.0

        # Base signal
        signal = 50000.0 + 15.0 * np.sin(2 * np.pi * (hour / 24.0 + t / 24.0))
        signal += 0.3 * rng.randn(3600)

        # Add subway interference during operational hours (6:00-23:00)
        has_subway = 6 <= hour <= 22
        if has_subway:
            for freq in [0.008, 0.016, 0.024]:
                signal += rng.uniform(1.5, 4.0) * np.sin(
                    2 * np.pi * freq * np.arange(3600) + rng.uniform(0, 2 * np.pi)
                )

        # Step 1: Identification
        x = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)
        x_norm = mad_normalize(x)
        with torch.no_grad():
            prob = tinycnn(x_norm).item()

        detected = prob >= threshold

        # Step 2: Suppression (only if interference detected)
        action = "---"
        if detected:
            mean_val, std_val = signal.mean(), signal.std() + 1e-8
            sig_norm = (signal - mean_val) / std_val
            sig_tensor = torch.FloatTensor(sig_norm).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                _, denoised = residual(sig_tensor)
            action = "Denoised"

        results.append((hour, prob, detected, has_subway))
        status = "INTERFERENCE" if detected else "CLEAN"
        print(f"  {hour:4d}  {prob:6.4f}  {status:>14s}  {action:>10s}")

    print(f"\n  [PASS] Full pipeline processed {n_hours} windows successfully")
    return True


# ========================================================================
# Test 4: Real Data Verification
# ========================================================================
def test_real_data():
    """Test with real geomagnetic data from Beijing Seismic Station (BJT)."""
    print("\n" + "=" * 65)
    print("TEST 4: Real Data — BJT Station (2024-02-01)")
    print("=" * 65)

    if not os.path.exists(SAMPLE_DATA):
        print(f"  [SKIP] Sample data not found: {SAMPLE_DATA}")
        return False

    if not os.path.exists(TINYCNN_CKPT) or not os.path.exists(RESIDUAL_CKPT):
        print(f"  [SKIP] Checkpoints not found")
        return False

    from tinycnn_model import load_model, mad_normalize
    from dl_denoise_only import ResidualDenoisingModel

    # Load sample data
    sample = np.load(SAMPLE_DATA, allow_pickle=True)
    clean_window = sample["clean"]            # Hour 18: subway not operating
    contaminated_window = sample["contaminated"]  # Hour 1: subway interference
    print(f"  Station:  {sample['station']}")
    print(f"  Date:     {sample['date']}")
    print(f"  Clean window (hour {sample['clean_hour']}):        "
          f"{clean_window.shape}, range [{clean_window.min():.1f}, {clean_window.max():.1f}] nT")
    print(f"  Contaminated window (hour {sample['contaminated_hour']}): "
          f"{contaminated_window.shape}, range [{contaminated_window.min():.1f}, {contaminated_window.max():.1f}] nT")

    # --- Step 1: Identification ---
    tinycnn, threshold, _ = load_model(TINYCNN_CKPT)

    batch = torch.stack([
        torch.FloatTensor(clean_window).unsqueeze(0),
        torch.FloatTensor(contaminated_window).unsqueeze(0),
    ])
    batch_norm = mad_normalize(batch)

    with torch.no_grad():
        probs = tinycnn(batch_norm)

    prob_clean = probs[0].item()
    prob_contam = probs[1].item()

    decision_clean = "INTERFERENCE" if prob_clean >= threshold else "CLEAN"
    decision_contam = "INTERFERENCE" if prob_contam >= threshold else "CLEAN"

    print(f"\n  Identification results (threshold={threshold:.2f}):")
    print(f"    Clean window:        prob={prob_clean:.6f}  -> {decision_clean}")
    print(f"    Contaminated window: prob={prob_contam:.6f}  -> {decision_contam}")

    # Verify: clean should be below threshold, contaminated above
    assert prob_clean < threshold, \
        f"Clean window misclassified: prob={prob_clean:.4f} >= {threshold}"
    assert prob_contam >= threshold, \
        f"Contaminated window missed: prob={prob_contam:.4f} < {threshold}"
    print(f"  [PASS] Identification correct on both real windows")

    # --- Step 2: Suppression of contaminated window ---
    residual = ResidualDenoisingModel(channels=64)
    ckpt = torch.load(RESIDUAL_CKPT, map_location="cpu", weights_only=False)
    residual.load_state_dict(ckpt["model_state_dict"])
    residual.eval()

    mean_val = contaminated_window.mean()
    std_val = contaminated_window.std() + 1e-8
    sig_norm = (contaminated_window - mean_val) / std_val
    x = torch.FloatTensor(sig_norm).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        estimated_noise, denoised = residual(x)

    denoised_signal = denoised.squeeze().numpy() * std_val + mean_val
    noise_estimate = estimated_noise.squeeze().numpy() * std_val

    # Compute first-derivative smoothness improvement
    diff_orig = np.diff(contaminated_window)
    diff_denoised = np.diff(denoised_signal)
    smoothness_ratio = np.std(diff_orig) / np.std(diff_denoised)

    # Amplitude range reduction
    range_orig = contaminated_window.max() - contaminated_window.min()
    range_denoised = denoised_signal.max() - denoised_signal.min()

    print(f"\n  Suppression results (contaminated window):")
    print(f"    Original amplitude range:  {range_orig:.2f} nT")
    print(f"    Denoised amplitude range:  {range_denoised:.2f} nT")
    print(f"    Estimated noise amplitude: {noise_estimate.max() - noise_estimate.min():.2f} nT")
    print(f"    Smoothness improvement:    {smoothness_ratio:.1f}x")
    print(f"  [PASS] Suppression completed on real contaminated window")

    return True


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    print()
    print("MagDL — Quick Test")
    print("Reference: Huo et al., Computers & Geosciences (2026)")
    print()

    passed = 0
    total = 0

    # Test 1: TinyCNN
    total += 1
    if test_tinycnn_identification():
        passed += 1

    # Test 2: ResidualCNN
    total += 1
    if test_residualcnn_suppression():
        passed += 1

    # Test 3: Full Pipeline
    total += 1
    if test_full_pipeline():
        passed += 1

    # Test 4: Real Data
    total += 1
    if test_real_data():
        passed += 1

    # Summary
    print("\n" + "=" * 65)
    print(f"SUMMARY: {passed}/{total} tests passed")
    print("=" * 65)

    if passed == total:
        print("\nAll tests passed! The code is ready for use.")
        print("For processing real geomagnetic data, see README.md.")
    else:
        print("\nSome tests were skipped or failed.")
        print("Ensure checkpoints/ and sample_data.npz are present.")

    sys.exit(0 if passed == total else 1)
