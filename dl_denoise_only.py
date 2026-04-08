#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResidualCNN Denoising Pipeline
===============================

Deep learning-based suppression of subway electromagnetic interference
in geomagnetic observations.

Pipeline:
    1. Adaptive boundary padding (adjacent-day extension or mirror padding)
    2. Sliding-window inference with Hann-window weighted overlap-add
    3. ResidualCNN estimates noise component; denoised = input - noise

Usage:
    # As a module (imported by quick_test.py):
    from dl_denoise_only import ResidualDenoisingModel

    # As a standalone script:
    python dl_denoise_only.py --station BJT --date 2024-07-13 \
        --model checkpoints/best_residual_model.pth

Reference:
    Huo et al., "Lightweight Deep Learning for Identification and Suppression
    of Subway Interference in Geomagnetic Observations",
    Computers & Geosciences (2026).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from scipy.signal import welch
from scipy.signal.windows import hann
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# ResidualCNN Model
# ================================================================

class ResidualDenoisingModel(nn.Module):
    """
    ResidualCNN: Encoder-Decoder network for geomagnetic signal denoising.

    Architecture:
        Encoder: Conv1d(1->64, k=15) -> Conv1d(64->128, k=11) -> Conv1d(128->128, k=7)
        Decoder: Conv1d(128->128, k=7) -> Conv1d(128->64, k=11) -> Conv1d(64->1, k=15)

    The model estimates the noise component n_hat, and the denoised signal
    is obtained via residual subtraction: y = x - n_hat.

    Parameters: 413,057
    Input:  [B, 1, T] — contaminated geomagnetic signal
    Output: (noise, denoised) — estimated noise and cleaned signal
    """

    def __init__(self, channels=64):
        super().__init__()

        # Encoder — extract noise features
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(channels, channels * 2, kernel_size=11, padding=5),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(channels * 2, channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=True),
        )

        # Decoder — reconstruct noise estimate
        self.decoder = nn.Sequential(
            nn.Conv1d(channels * 2, channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(channels * 2, channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(channels, 1, kernel_size=15, padding=7),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input contaminated signal [B, 1, T]

        Returns:
            noise:    Estimated noise component [B, 1, T]
            denoised: Cleaned signal [B, 1, T]  (x - noise)
        """
        features = self.encoder(x)
        noise = self.decoder(features)
        denoised = x - noise
        return noise, denoised


# ================================================================
# Sliding-Window Denoiser
# ================================================================

class DLDenoiser:
    """
    Deep learning denoiser with sliding-window overlap-add processing.

    Processes daily geomagnetic records (86,400 samples at 1 Hz) by:
    1. Adaptive boundary padding using adjacent-day data or mirroring
    2. Sliding 1-hour windows with 50% overlap
    3. Hann-window weighted fusion for seamless reconstruction
    """

    def __init__(self, model_path, window_size=3600, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.stride = int(window_size * (1 - overlap))

        # Device selection (prefer GPU if available)
        if torch.cuda.is_available():
            try:
                test_tensor = torch.randn(1, 1, 100).cuda()
                _ = test_tensor * 2
                self.device = torch.device('cuda')
                print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
            except RuntimeError as e:
                if "no kernel image" in str(e) or "not compatible" in str(e):
                    print("  GPU incompatible, falling back to CPU")
                    self.device = torch.device('cpu')
                else:
                    raise e
        else:
            self.device = torch.device('cpu')
            print("  Using CPU mode")

        self.model = ResidualDenoisingModel(channels=64).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device,
                                    weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"  Model loaded: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.hann_window = hann(window_size)

    def denoise(self, signal, prev_signal=None, next_signal=None):
        """
        Denoise a daily geomagnetic signal.

        Uses adaptive boundary padding: if adjacent-day data is available,
        it extends the signal with real observations; otherwise uses mirror
        padding to reduce edge artifacts.

        Args:
            signal:      Current day signal (numpy array, typically 86400 samples)
            prev_signal: Previous day signal (optional, for left boundary)
            next_signal: Next day signal (optional, for right boundary)

        Returns:
            denoised: Denoised signal (same length as input)
        """
        pad_size = self.window_size // 2

        # Adaptive boundary padding
        if prev_signal is not None and len(prev_signal) >= pad_size:
            left_pad = prev_signal[-pad_size:]
            left_method = "adjacent-day"
        else:
            left_pad = np.flip(signal[:pad_size])
            left_method = "mirror"

        if next_signal is not None and len(next_signal) >= pad_size:
            right_pad = next_signal[:pad_size]
            right_method = "adjacent-day"
        else:
            right_pad = np.flip(signal[-pad_size:])
            right_method = "mirror"

        print(f"  Boundary padding: left={left_method}, right={right_method}")

        signal_padded = np.concatenate([left_pad, signal, right_pad])

        n_windows = (len(signal_padded) - self.window_size) // self.stride + 1

        output = np.zeros(len(signal_padded))
        weights = np.zeros(len(signal_padded))

        print(f"  Processing {n_windows} windows...")

        with torch.no_grad():
            for i in range(n_windows):
                start = i * self.stride
                end = start + self.window_size

                if end > len(signal_padded):
                    break

                window = signal_padded[start:end]

                # Z-score normalization
                mean = window.mean()
                std = window.std() + 1e-8
                window_norm = (window - mean) / std

                # Inference
                window_tensor = torch.from_numpy(window_norm).float()
                window_tensor = window_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                _, denoised_norm = self.model(window_tensor)
                denoised_window = denoised_norm.cpu().numpy().squeeze()

                # De-normalize
                denoised_window = denoised_window * std + mean

                # Hann-weighted overlap-add
                output[start:end] += denoised_window * self.hann_window
                weights[start:end] += self.hann_window

        weights = np.maximum(weights, 1e-10)
        output = output / weights

        # Remove padding
        output = output[pad_size:-pad_size]

        return output

    def plot_comparison(self, original, denoised, station_name, date, save_dir):
        """Plot time-domain comparison of original vs denoised signals."""
        fig = plt.figure(figsize=(16, 10))
        time_hours = np.arange(len(original)) / 3600

        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(time_hours, original, 'b-', linewidth=0.5, alpha=0.7,
                 label='Original')
        ax1.set_ylabel('Magnetic Field (nT)', fontsize=12)
        ax1.set_title(f'{station_name} - {date} Original Signal',
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(time_hours, denoised, 'r-', linewidth=0.5, alpha=0.7,
                 label='Denoised')
        ax2.set_ylabel('Magnetic Field (nT)', fontsize=12)
        ax2.set_title(f'{station_name} - {date} Denoised (ResidualCNN)',
                      fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(time_hours, original, 'b-', linewidth=0.5, alpha=0.5,
                 label='Original')
        ax3.plot(time_hours, denoised, 'r-', linewidth=0.8, alpha=0.8,
                 label='Denoised')
        ax3.set_xlabel('Time (hours)', fontsize=12)
        ax3.set_ylabel('Magnetic Field (nT)', fontsize=12)
        ax3.set_title('Original vs Denoised', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = Path(save_dir) / f'{station_name}_{date}_denoising_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()

    def plot_frequency_analysis(self, original, denoised, station_name, date,
                                save_dir):
        """Plot frequency-domain analysis comparing original vs denoised."""
        fig = plt.figure(figsize=(16, 12))
        fs = 1.0  # Sampling rate: 1 Hz

        freqs_orig, psd_orig = welch(original, fs=fs, nperseg=3600,
                                     noverlap=1800)
        freqs_den, psd_den = welch(denoised, fs=fs, nperseg=3600,
                                   noverlap=1800)

        # (A) Full-band PSD comparison
        ax1 = plt.subplot(3, 2, 1)
        ax1.semilogy(freqs_orig, psd_orig, 'b-', linewidth=1, alpha=0.7,
                     label='Original')
        ax1.semilogy(freqs_den, psd_den, 'r-', linewidth=1, alpha=0.7,
                     label='Denoised')
        ax1.axvspan(0.004, 0.032, alpha=0.2, color='orange',
                    label='Subway band')
        ax1.set_xlabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylabel('PSD (nT$^2$/Hz)', fontsize=11)
        ax1.set_title('(A) Full-band PSD', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 0.5)

        # (B) Subway interference band zoom
        ax2 = plt.subplot(3, 2, 2)
        mask = (freqs_orig >= 0.004) & (freqs_orig <= 0.032)
        ax2.semilogy(freqs_orig[mask], psd_orig[mask], 'b-', linewidth=1.5,
                     alpha=0.7, label='Original')
        ax2.semilogy(freqs_den[mask], psd_den[mask], 'r-', linewidth=1.5,
                     alpha=0.7, label='Denoised')
        ax2.set_xlabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylabel('PSD (nT$^2$/Hz)', fontsize=11)
        ax2.set_title('(B) Subway Band (0.004-0.032 Hz)',
                      fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        energy_orig = np.trapz(psd_orig[mask], freqs_orig[mask])
        energy_den = np.trapz(psd_den[mask], freqs_den[mask])
        suppression_rate = (1 - energy_den / energy_orig) * 100
        ax2.text(0.98, 0.98, f'Suppression: {suppression_rate:.2f}%',
                 transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # (C) Low-frequency fidelity
        ax3 = plt.subplot(3, 2, 3)
        mask_low = freqs_orig <= 0.004
        ax3.semilogy(freqs_orig[mask_low], psd_orig[mask_low], 'b-',
                     linewidth=1.5, alpha=0.7, label='Original')
        ax3.semilogy(freqs_den[mask_low], psd_den[mask_low], 'r-',
                     linewidth=1.5, alpha=0.7, label='Denoised')
        ax3.set_xlabel('Frequency (Hz)', fontsize=11)
        ax3.set_ylabel('PSD (nT$^2$/Hz)', fontsize=11)
        ax3.set_title('(C) Low-Frequency Fidelity (<0.004 Hz)',
                      fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)

        from scipy.stats import pearsonr
        corr, _ = pearsonr(psd_orig[mask_low], psd_den[mask_low])
        ax3.text(0.98, 0.98, f'Correlation: {corr:.4f}',
                 transform=ax3.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # (D) Band energy comparison
        ax4 = plt.subplot(3, 2, 4)
        bands = {
            'Low\n(<0.004Hz)': (0, 0.004),
            'Subway\n(0.004-0.032Hz)': (0.004, 0.032),
            'High\n(>0.032Hz)': (0.032, 0.5)
        }
        energies_orig, energies_den, band_names = [], [], []
        for name, (f_low, f_high) in bands.items():
            m = (freqs_orig >= f_low) & (freqs_orig <= f_high)
            energies_orig.append(np.trapz(psd_orig[m], freqs_orig[m]))
            energies_den.append(np.trapz(psd_den[m], freqs_den[m]))
            band_names.append(name)

        x = np.arange(len(band_names))
        width = 0.35
        ax4.bar(x - width / 2, energies_orig, width, label='Original',
                alpha=0.7, color='blue')
        ax4.bar(x + width / 2, energies_den, width, label='Denoised',
                alpha=0.7, color='red')
        ax4.set_ylabel('Energy (nT$^2$)', fontsize=11)
        ax4.set_title('(D) Band Energy Comparison',
                      fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(band_names, fontsize=10)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        # (E) First-order difference
        ax5 = plt.subplot(3, 2, 5)
        diff_orig = np.diff(original)
        diff_den = np.diff(denoised)
        time_hours = np.arange(len(diff_orig)) / 3600
        ax5.plot(time_hours, diff_orig, 'b-', linewidth=0.3, alpha=0.5,
                 label='Original')
        ax5.plot(time_hours, diff_den, 'r-', linewidth=0.3, alpha=0.8,
                 label='Denoised')
        ax5.set_xlabel('Time (hours)', fontsize=11)
        ax5.set_ylabel('1st Derivative (nT/s)', fontsize=11)
        ax5.set_title('(E) First-Order Difference',
                      fontsize=13, fontweight='bold')
        ax5.legend(loc='upper right', fontsize=10)
        ax5.grid(True, alpha=0.3)

        smooth_improvement = np.std(diff_orig) / np.std(diff_den)
        ax5.text(0.02, 0.98, f'Smoothness: {smooth_improvement:.1f}x',
                 transform=ax5.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # (F) Summary statistics
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        stats_text = (
            f"  {station_name} - {date}\n"
            f"  Denoising Performance\n"
            f"  {'=' * 30}\n\n"
            f"  [Time Domain]\n"
            f"  1st-diff std (original):  {np.std(diff_orig):.4f} nT/s\n"
            f"  1st-diff std (denoised):  {np.std(diff_den):.4f} nT/s\n"
            f"  Smoothness improvement:   {smooth_improvement:.1f}x\n\n"
            f"  [Frequency Domain]\n"
            f"  Subway band suppression:  {suppression_rate:.2f}%\n"
            f"  Low-freq correlation:     {corr:.4f}\n\n"
            f"  [Parameters]\n"
            f"  Window size: {self.window_size}s\n"
            f"  Overlap:     {self.overlap * 100:.0f}%\n"
            f"  Model params: 413K"
        )
        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()
        save_path = (Path(save_dir)
                     / f'{station_name}_{date}_frequency_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()


# ================================================================
# CLI Entry Point
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ResidualCNN denoising for geomagnetic observations')
    parser.add_argument('--station', type=str, default='BJT',
                        choices=['BJT', 'HZT'], help='Station code')
    parser.add_argument('--date', type=str, default='2024-07-13',
                        help='Date (YYYY-MM-DD)')
    parser.add_argument('--model', type=str,
                        default='checkpoints/best_residual_model.pth',
                        help='Model checkpoint path')
    parser.add_argument('--prev_date', type=str, default=None,
                        help='Previous day date (optional, for boundary)')
    parser.add_argument('--next_date', type=str, default=None,
                        help='Next day date (optional, for boundary)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to input data file')
    args = parser.parse_args()

    # Station display names
    station_names = {
        'BJT': 'Beijing Seismic Station',
        'HZT': 'Hangzhou Botanical Garden Station'
    }

    print(f"\n{'=' * 70}")
    print(f"ResidualCNN Denoising: {station_names.get(args.station, args.station)}"
          f" - {args.date}")
    print(f"{'=' * 70}\n")

    if args.data_path is None:
        print("[Error] Please specify --data_path pointing to your data file.")
        print("Example: python dl_denoise_only.py --data_path data/BJT_20240713.txt")
        return

    if not os.path.exists(args.data_path):
        print(f"[Error] File not found: {args.data_path}")
        return

    # Import data reader
    from hzt_data_reader import read_hzt_file

    data = read_hzt_file(args.data_path)
    if data is None or len(data) == 0:
        print("[Error] Failed to read data file")
        return

    if isinstance(data, dict):
        signal = data.get('Z', data.get('H', None))
        if signal is None:
            print("[Error] No Z or H component found in data")
            return
    elif isinstance(data, np.ndarray):
        signal = data[:, 1] if data.ndim >= 2 and data.shape[1] >= 2 \
            else data.ravel()
    else:
        print(f"[Error] Unknown data type: {type(data)}")
        return

    print(f"  Data loaded: {len(signal)} samples")

    # Optional: load adjacent-day data for boundary handling
    prev_signal = None
    next_signal = None
    # (Users can extend this section to load prev/next day files)

    # Create denoiser and process
    denoiser = DLDenoiser(model_path=args.model, window_size=3600, overlap=0.5)

    print(f"\n  Denoising...")
    denoised = denoiser.denoise(signal, prev_signal, next_signal)
    print(f"  Done!")

    # Save results
    save_dir = 'results'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    station_name = station_names.get(args.station, args.station)

    print(f"\n  Generating plots...")
    denoiser.plot_comparison(signal, denoised, station_name, args.date, save_dir)
    denoiser.plot_frequency_analysis(signal, denoised, station_name,
                                     args.date, save_dir)

    output_file = Path(save_dir) / f'{args.station}_{args.date}_denoised.npz'
    np.savez_compressed(output_file, original=signal, denoised=denoised,
                        date=args.date, station=args.station)
    print(f"  Data saved: {output_file}")

    print(f"\n{'=' * 70}")
    print(f"Processing complete! Results saved in: {save_dir}")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    main()
