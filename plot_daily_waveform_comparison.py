"""
日图对比脚本
拼接窗口(50%重叠+Hann加权) + 轻度SG平滑
输出: 原始/VMD/DL/分层的对比图 + 频谱分析
"""

from hzt_data_reader import read_hzt_file
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
from scipy.signal.windows import hann
from vmdpy import VMD
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei',
                                   'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# ================== 残差式去噪网络 ==================

class ResidualDenoisingModel(nn.Module):
    """
    残差式去噪网络（仅推理）
    输出: 估计的噪声 n̂
    最终去噪: y = x - n̂
    """

    def __init__(self, channels=64):
        super().__init__()

        # 编码器 - 提取噪声特征
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(channels, channels*2, kernel_size=11, padding=5),
            nn.BatchNorm1d(channels*2),
            nn.ReLU(inplace=True),

            nn.Conv1d(channels*2, channels*2, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels*2),
            nn.ReLU(inplace=True),
        )

        # 解码器 - 重建噪声
        self.decoder = nn.Sequential(
            nn.Conv1d(channels*2, channels*2, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels*2),
            nn.ReLU(inplace=True),

            nn.Conv1d(channels*2, channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(channels, 1, kernel_size=15, padding=7),
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入含噪信号 [B, 1, T]

        Returns:
            noise: 估计的噪声 [B, 1, T]
            denoised: 去噪后的信号 [B, 1, T] (x - noise)
        """
        # 估计噪声
        features = self.encoder(x)
        noise = self.decoder(features)

        # 残差连接: 去噪信号 = 输入 - 噪声
        denoised = x - noise

        return noise, denoised


# ================== 日波形处理器 ==================

class DailyWaveformProcessor:
    """日波形处理器"""

    def __init__(self,
                 model_path,
                 window_size=3600,
                 overlap=0.5,
                 vmd_K=8,
                 vmd_alpha=2000,
                 keep_imfs=2,
                 sg_window=600,  # SG平滑窗口(10分钟)
                 sg_poly=2):

        self.window_size = window_size
        self.overlap = overlap
        self.stride = int(window_size * (1 - overlap))
        self.vmd_K = vmd_K
        self.vmd_alpha = vmd_alpha
        self.keep_imfs = keep_imfs
        self.sg_window = sg_window if sg_window % 2 == 1 else sg_window + 1
        self.sg_poly = sg_poly

        # 加载深度学习模型（强制使用CPU避免CUDA兼容性问题）
        self.device = torch.device('cpu')
        self.model = ResidualDenoisingModel(channels=64).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            print(f"✅ 模型加载成功: {model_path}")
        else:
            print(f"⚠️ 模型文件不存在: {model_path}, 将跳过DL去噪")
            self.model = None

        # Hann窗
        self.hann_window = hann(window_size)

    def vmd_denoise_full(self, signal):
        """
        VMD去噪 - 滑窗处理整条信号（改进版：镜像边界）
        """
        n_samples = len(signal)

        # 镜像填充边界，避免边界效应
        pad_size = self.window_size // 2
        signal_padded = np.pad(signal, (pad_size, pad_size), mode='reflect')

        # 计算窗口数
        n_windows = (len(signal_padded) - self.window_size) // self.stride + 1

        # 输出信号 + 权重累计
        output = np.zeros(len(signal_padded))
        weights = np.zeros(len(signal_padded))

        print(f"  VMD处理: {n_windows} 个窗口（镜像边界）...")

        for i in range(n_windows):
            start = i * self.stride
            end = start + self.window_size

            if end > len(signal_padded):
                break

            # 提取窗口
            window = signal_padded[start:end]

            try:
                # VMD分解
                u, _, _ = VMD(window, self.vmd_alpha, 0.0, self.vmd_K,
                              DC=0, init=1, tol=1e-7)

                # 保留低频IMF
                denoised_window = np.sum(u[:self.keep_imfs, :], axis=0)

                # 加权累加 (Hann窗)
                output[start:end] += denoised_window * self.hann_window
                weights[start:end] += self.hann_window

            except Exception as e:
                # VMD失败,使用原窗口
                output[start:end] += window * self.hann_window
                weights[start:end] += self.hann_window

        # 归一化
        weights = np.maximum(weights, 1e-10)  # 避免除零
        output = output / weights

        # 去除填充，返回原始长度
        output = output[pad_size:-pad_size]

        return output

    def dl_denoise_full(self, signal):
        """
        深度学习去噪 - 滑窗处理整条信号（改进版：镜像边界）
        """
        if self.model is None:
            return signal.copy()

        # 镜像填充边界
        pad_size = self.window_size // 2
        signal_padded = np.pad(signal, (pad_size, pad_size), mode='reflect')

        # 计算窗口数
        n_windows = (len(signal_padded) - self.window_size) // self.stride + 1

        # 输出信号 + 权重累计
        output = np.zeros(len(signal_padded))
        weights = np.zeros(len(signal_padded))

        print(f"  DL处理: {n_windows} 个窗口（镜像边界）...")

        with torch.no_grad():
            for i in range(n_windows):
                start = i * self.stride
                end = start + self.window_size

                if end > len(signal_padded):
                    break

                # 提取窗口
                window = signal_padded[start:end]

                # 标准化
                mean = window.mean()
                std = window.std() + 1e-8
                window_norm = (window - mean) / std

                # 推理
                window_tensor = torch.from_numpy(window_norm).float(
                ).unsqueeze(0).unsqueeze(0).to(self.device)
                _, denoised_norm = self.model(window_tensor)
                denoised_window = denoised_norm.cpu().numpy().squeeze()

                # 反标准化
                denoised_window = denoised_window * std + mean

                # 加权累加 (Hann窗)
                output[start:end] += denoised_window * self.hann_window
                weights[start:end] += self.hann_window

        # 归一化
        weights = np.maximum(weights, 1e-10)  # 避免除零
        output = output / weights

        # 去除填充，返回原始长度
        output = output[pad_size:-pad_size]

        return output

    def hierarchical_denoise(self, signal):
        """
        分层去噪: VMD -> DL
        """
        # 第一层: VMD
        vmd_output = self.vmd_denoise_full(signal)

        # 第二层: DL (在VMD输出基础上继续去噪)
        if self.model is not None:
            dl_output = self.dl_denoise_full(vmd_output)
        else:
            dl_output = vmd_output.copy()

        return vmd_output, dl_output

    def apply_sg_smooth(self, signal):
        """应用轻度SG平滑"""
        if len(signal) < self.sg_window:
            return signal
        return savgol_filter(signal, self.sg_window, self.sg_poly)

    def compute_spectrum(self, signal, fs=1.0):
        """计算频谱"""
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
        magnitude = np.abs(fft)
        return freqs, magnitude

    def plot_comparison(self, signal, vmd_output, dl_output, hierarchical_output,
                        station_name, date_str, save_dir):
        """
        绘制对比图
        5个子图: 原始/VMD/DL/分层/频谱对比
        """
        # SG平滑
        vmd_smooth = self.apply_sg_smooth(vmd_output)
        dl_smooth = self.apply_sg_smooth(dl_output)
        hierarchical_smooth = self.apply_sg_smooth(hierarchical_output)

        # 时间轴(小时)
        time_hours = np.arange(len(signal)) / 3600

        # 创建图形
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

        # 1. 原始信号
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_hours, signal, 'gray', linewidth=0.8,
                 alpha=0.6, label='Original')
        ax1.set_xlabel('Time (hours)', fontsize=11)
        ax1.set_ylabel('Amplitude (nT)', fontsize=11)
        ax1.set_title(f'{station_name} - {date_str}\nOriginal Signal',
                      fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. VMD去噪
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_hours, vmd_smooth, 'green',
                 linewidth=1.2, label='VMD Denoised')
        ax2.set_xlabel('Time (hours)', fontsize=11)
        ax2.set_ylabel('Amplitude (nT)', fontsize=11)
        ax2.set_title(
            f'VMD Denoising (Keep {self.keep_imfs} IMFs)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. DL去噪
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_hours, dl_smooth, 'blue',
                 linewidth=1.2, label='DL Denoised')
        ax3.set_xlabel('Time (hours)', fontsize=11)
        ax3.set_ylabel('Amplitude (nT)', fontsize=11)
        ax3.set_title('Deep Learning Denoising',
                      fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. 分层去噪
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_hours, hierarchical_smooth, 'red',
                 linewidth=1.2, label='Hierarchical')
        ax4.set_xlabel('Time (hours)', fontsize=11)
        ax4.set_ylabel('Amplitude (nT)', fontsize=11)
        ax4.set_title('Hierarchical Denoising (VMD + DL)',
                      fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. 四者对比
        ax5 = fig.add_subplot(gs[2, :])
        ax5.plot(time_hours, signal, 'gray', linewidth=0.5,
                 alpha=0.4, label='Original')
        ax5.plot(time_hours, vmd_smooth, 'green',
                 linewidth=1.0, alpha=0.8, label='VMD')
        ax5.plot(time_hours, dl_smooth, 'blue',
                 linewidth=1.0, alpha=0.8, label='DL')
        ax5.plot(time_hours, hierarchical_smooth, 'red',
                 linewidth=1.2, alpha=0.9, label='Hierarchical')
        ax5.set_xlabel('Time (hours)', fontsize=11)
        ax5.set_ylabel('Amplitude (nT)', fontsize=11)
        ax5.set_title('Comparison of All Methods',
                      fontsize=12, fontweight='bold')
        ax5.legend(fontsize=10, loc='best')
        ax5.grid(True, alpha=0.3)

        # 保存
        save_path = Path(save_dir) / \
            f'{station_name}_{date_str}_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 对比图已保存: {save_path}")
        plt.close()

        # 绘制频谱对比
        self.plot_spectrum_comparison(signal, vmd_output, dl_output, hierarchical_output,
                                      station_name, date_str, save_dir)

    def plot_spectrum_comparison(self, signal, vmd_output, dl_output, hierarchical_output,
                                 station_name, date_str, save_dir):
        """绘制频谱对比"""
        fs = 1.0  # 1 Hz采样率

        # 计算频谱
        freqs_orig, mag_orig = self.compute_spectrum(signal, fs)
        freqs_vmd, mag_vmd = self.compute_spectrum(vmd_output, fs)
        freqs_dl, mag_dl = self.compute_spectrum(dl_output, fs)
        freqs_hier, mag_hier = self.compute_spectrum(hierarchical_output, fs)

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 1. 全频谱
        ax1.semilogy(freqs_orig, mag_orig, 'gray',
                     linewidth=0.8, alpha=0.5, label='Original')
        ax1.semilogy(freqs_vmd, mag_vmd, 'green',
                     linewidth=1.2, alpha=0.8, label='VMD')
        ax1.semilogy(freqs_dl, mag_dl, 'blue',
                     linewidth=1.2, alpha=0.8, label='DL')
        ax1.semilogy(freqs_hier, mag_hier, 'red', linewidth=1.5,
                     alpha=0.9, label='Hierarchical')
        ax1.axvline(1e-4, color='orange', linestyle='--',
                    linewidth=1.5, label='Keep Freq (1e-4 Hz)')
        ax1.axvline(5e-4, color='purple', linestyle='--',
                    linewidth=1.5, label='Block Freq (5e-4 Hz)')
        ax1.set_xlabel('Frequency (Hz)', fontsize=11)
        ax1.set_ylabel('Magnitude', fontsize=11)
        ax1.set_title(f'{station_name} - {date_str}\nFull Spectrum Comparison',
                      fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, which='both')

        # 2. 低频放大
        low_freq_mask = freqs_orig < 0.01  # 0-0.01 Hz
        ax2.semilogy(freqs_orig[low_freq_mask], mag_orig[low_freq_mask],
                     'gray', linewidth=0.8, alpha=0.5, label='Original')
        ax2.semilogy(freqs_vmd[low_freq_mask], mag_vmd[low_freq_mask],
                     'green', linewidth=1.2, alpha=0.8, label='VMD')
        ax2.semilogy(freqs_dl[low_freq_mask], mag_dl[low_freq_mask],
                     'blue', linewidth=1.2, alpha=0.8, label='DL')
        ax2.semilogy(freqs_hier[low_freq_mask], mag_hier[low_freq_mask],
                     'red', linewidth=1.5, alpha=0.9, label='Hierarchical')
        ax2.axvline(1e-4, color='orange', linestyle='--',
                    linewidth=1.5, label='Keep Freq')
        ax2.axvline(5e-4, color='purple', linestyle='--',
                    linewidth=1.5, label='Block Freq')
        ax2.set_xlabel('Frequency (Hz)', fontsize=11)
        ax2.set_ylabel('Magnitude', fontsize=11)
        ax2.set_title('Low Frequency Detail (0-0.01 Hz)',
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which='both')

        plt.tight_layout()

        # 保存
        save_path = Path(save_dir) / f'{station_name}_{date_str}_spectrum.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 频谱图已保存: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='日波形对比脚本')
    parser.add_argument('--station', type=str, default='BJT',
                        choices=['BJT', 'HZT'], help='台站代码')
    parser.add_argument('--date', type=str, default='2025-03-01',
                        help='日期 (YYYY-MM-DD)')
    parser.add_argument('--model', type=str,
                        default='checkpoints/residual_denoising/best_residual_model.pth',
                        help='模型路径')
    args = parser.parse_args()

    # TODO: 修改为您的数据路径
    # 数据文件路径模板
    data_paths = {
        'BJT': os.path.join('data', 'raw', 'BJT', '原始秒数据_[11074]北京地震台_FGM01[G]_{date}.Txt'),
        'HZT': os.path.join('data', 'raw', 'HZT', '2025', 'OHZXH_FGMM01_PGCV_L1_DAY_{date}000000_V01.00.txt.org')
    }

    station_names = {
        'BJT': '北京地震台',
        'HZT': '杭州植物园站'
    }

    # 构建文件路径
    date_str = args.date.replace('-', '')
    file_path = data_paths[args.station].format(date=date_str)
    station_name = station_names[args.station]

    print(f"\n{'='*80}")
    print(f"日波形对比: {station_name} - {args.date}")
    print(f"{'='*80}")

    # 读取数据
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return

    data = read_hzt_file(file_path)
    if data is None or len(data) == 0:
        print(f"[错误] 数据读取失败")
        return

    # 处理dict返回（与generate_vmd_pseudo_labels.py一致）
    if isinstance(data, dict):
        # 优先Z分量，否则H分量
        signal = data.get('Z', data.get('H', None))
        if signal is None:
            print(f"[错误] 数据字典中没有Z或H分量")
            return
    elif isinstance(data, np.ndarray):
        # 提取Z分量
        if data.ndim >= 2 and data.shape[1] >= 2:
            signal = data[:, 1]
        else:
            signal = data.ravel()
    else:
        print(f"[错误] 未知数据类型: {type(data)}")
        return

    print(f"✅ 数据读取成功: {len(signal)} 点")

    # 创建处理器
    processor = DailyWaveformProcessor(
        model_path=args.model,
        window_size=3600,
        overlap=0.5,
        vmd_K=8,
        vmd_alpha=2000,
        keep_imfs=2,
        sg_window=601,  # 10分钟平滑
        sg_poly=2
    )

    # 分层去噪
    print(f"\n开始分层去噪...")
    vmd_output, hierarchical_output = processor.hierarchical_denoise(signal)

    # DL单独去噪(对比用)
    dl_output = processor.dl_denoise_full(signal)

    # 绘制对比
    save_dir = 'results/daily_comparison'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    processor.plot_comparison(
        signal, vmd_output, dl_output, hierarchical_output,
        station_name, args.date, save_dir
    )

    # 保存数据
    output_file = Path(save_dir) / f'{station_name}_{args.date}_data.npz'
    np.savez_compressed(
        output_file,
        original=signal,
        vmd=vmd_output,
        dl=dl_output,
        hierarchical=hierarchical_output
    )
    print(f"✅ 数据已保存: {output_file}")

    print(f"\n{'='*80}")
    print(f"处理完成!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
