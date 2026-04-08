#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TinyCNN: Lightweight CNN for Subway Interference Identification

A compact binary classifier that determines whether subway electromagnetic
interference is present in hourly geomagnetic observation windows.

Architecture:
    Input: [B, 1, 3600] (1-hour window at 1 Hz sampling rate)
      |
    Conv1: Conv1d(1->48, k=7, s=2) + BN + ReLU + MaxPool(2)
      |
    Conv2: DWSepConv(48->96, k=5, s=2) + BN + ReLU + SE(r=8) + MaxPool(2)
      |
    Conv3: DWSepConv(96->192, k=3, s=2) + BN + ReLU + SE(r=8)
      |
    Conv4: DWSepConv(192->192, k=3, s=1) + BN + ReLU + SE(r=8) + GAP
      |
    Classifier: FC(192->96) -> ReLU -> FC(96->1)
      |
    Output: [B, 1] (classification probability)
    
    Trainable parameters: 102,301

Performance (validation set, 28,479 samples):
    - F1-score: 0.9997
    - Precision: 0.9997
    - Recall: 0.9997
    - AUC: 0.99999
    - Threshold: 0.80

Reference:
    "Lightweight Deep Learning for Identification and Suppression 
     of Subway Interference in Geomagnetic Observations"
    Computers & Geosciences (2026)
"""

import torch
import torch.nn as nn
import numpy as np


# ================================================================
# Preprocessing
# ================================================================
def mad_normalize(x):
    """
    MAD (Median Absolute Deviation) normalization.
    
    Robust amplitude standardization that handles inter-station variations
    in baseline levels (10^3 - 10^4 nT) and fluctuation amplitudes.
    
    Formula: x_norm = (x - median(x)) / (1.4826 * MAD(x))
    
    The coefficient 1.4826 ensures MAD equivalence to standard deviation
    under Gaussian assumptions (Oppenheim and Schafer, 2010).
    
    Args:
        x (torch.Tensor): Input signal [B, C, L]
    
    Returns:
        torch.Tensor: Normalized signal [B, C, L]
    """
    x = torch.clamp(x, min=-50000, max=50000)
    
    median = x.median(dim=2, keepdim=True).values
    mad = (x - median).abs().median(dim=2, keepdim=True).values
    scale = 1.4826 * mad
    scale = torch.where(
        scale < 1e-6,
        x.std(dim=2, keepdim=True).clamp_min(1e-6),
        scale
    )
    
    normalized = (x - median) / scale
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = torch.clamp(normalized, -10.0, 10.0)
    
    return normalized


# ================================================================
# Building Blocks
# ================================================================
class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise Separable Convolution (Howard et al., 2017).
    
    Factorizes standard convolution into:
    1. Depthwise: independent per-channel convolution (groups=in_channels)
    2. Pointwise: 1x1 convolution for channel mixing
    
    Computational cost: ~1/k of standard convolution (k = kernel size).
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation attention (Hu et al., CVPR 2018).
    
    Adaptively recalibrates channel-wise feature responses.
    
    Architecture:
        Squeeze:    Global Average Pooling -> [B, C, 1]
        Excitation: Conv1d(C->C/r) -> ReLU -> Conv1d(C/r->C) -> Sigmoid
        Scale:      Input * channel weights
    
    Args:
        channels: Number of input channels
        reduction: Compression ratio (default=8)
    """
    
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weight = self.se(x)     # [B, C, 1]
        return x * weight       # Channel-wise scaling


# ================================================================
# TinyCNN Model
# ================================================================
class TinyCNN(nn.Module):
    """
    TinyCNN: Lightweight binary classifier for subway interference detection.
    
    Processes 1-hour geomagnetic windows (3600 samples at 1 Hz) and outputs
    a probability score indicating subway interference presence.
    
    Args:
        input_length: Sequence length (default 3600)
        in_channels: Input channels (default 1, Z-component)
        num_classes: Output classes (default 1, binary)
        dropout: Dropout probability (default 0.2)
        use_sigmoid: Apply sigmoid in forward pass
                    - False: output logits (for training with BCEWithLogitsLoss)
                    - True: output probabilities (for inference)
    """
    
    def __init__(self, input_length=3600, in_channels=1, num_classes=1,
                 dropout=0.2, use_sigmoid=False):
        super().__init__()
        
        self.input_length = input_length
        self.in_channels = in_channels
        self.use_sigmoid = use_sigmoid
        
        # Layer 1: Standard Conv + MaxPool
        # 1 -> 48 channels, captures broad temporal patterns
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 2: DWSep + SE + MaxPool (48 -> 96 channels)
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv1D(48, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            SEBlock1D(96, reduction=8),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Layer 3: DWSep + SE, no MaxPool (96 -> 192 channels)
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv1D(96, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            SEBlock1D(192, reduction=8)
        )
        
        # Layer 4: DWSep + SE + GAP (192 -> 192, stride=1 for refinement)
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv1D(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            SEBlock1D(192, reduction=8),
            nn.Dropout(0.15),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Two-layer FC classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )
        
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, 3600] or [B, 3600]
        Returns:
            [B, 1] logits or probabilities
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        logits = self.classifier(x)
        
        if self.use_sigmoid:
            return self.sigmoid(logits)
        return logits


# ================================================================
# Utility
# ================================================================
def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint_path, device='cpu'):
    """
    Load TinyCNN from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth file
        device: 'cpu' or 'cuda'
    
    Returns:
        model: TinyCNN in eval mode
        threshold: Classification threshold
        metadata: Training metrics dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    
    model = TinyCNN(input_length=3600, use_sigmoid=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)
    
    threshold = checkpoint.get('threshold', checkpoint.get('best_threshold', 0.80))
    metrics = checkpoint.get('metrics', {})
    metadata = {
        'epoch': checkpoint.get('epoch'),
        'f1': metrics.get('f1', checkpoint.get('best_f1')),
        'precision': metrics.get('precision', checkpoint.get('best_precision')),
        'recall': metrics.get('recall', checkpoint.get('best_recall')),
        'auc': metrics.get('auc', checkpoint.get('best_auc')),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}
    
    return model, threshold, metadata


if __name__ == "__main__":
    model = TinyCNN(input_length=3600, use_sigmoid=False)
    params = count_parameters(model)
    
    print(f"TinyCNN Architecture")
    print(f"  Conv1: 1 -> 48   (Conv, k=7, s=2) + MaxPool")
    print(f"  Conv2: 48 -> 96  (DWSep + SE(r=8), k=5, s=2) + MaxPool")
    print(f"  Conv3: 96 -> 192 (DWSep + SE(r=8), k=3, s=2)")
    print(f"  Conv4: 192-> 192 (DWSep + SE(r=8), k=3, s=1) + GAP")
    print(f"  FC:    192 -> 96 -> 1")
    print(f"  Parameters: {params:,} ({'OK' if params == 102301 else 'MISMATCH'})")
    
    x = torch.randn(2, 1, 3600)
    with torch.no_grad():
        out = model(x)
    print(f"  Forward: {x.shape} -> {out.shape}")
