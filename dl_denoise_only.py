"""
纯深度学习去噪脚本（不使用VMD）
功能：
1. 自适应边界填充（邻日顺延+镜像填充）
2. ResidualCNN去噪
3. 频域分析对比
4. 生成论文图表
"""

from hzt_data_reader import read_hzt_file
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei',
                                   'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


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


# ================== 深度学习去噪器 ==================

class DLDenoiser:
    """纯深度学习去噪器"""

    def __init__(self, model_path, window_size=3600, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.stride = int(window_size * (1 - overlap))

        # 加载深度学习模型（优先GPU）
        if torch.cuda.is_available():
            try:
                test_tensor = torch.randn(1, 1, 100).cuda()
                _ = test_tensor * 2
                self.device = torch.device('cuda')
                print(f"🚀 使用GPU加速: {torch.cuda.get_device_name(0)}")
            except RuntimeError as e:
                if "no kernel image" in str(e) or "not compatible" in str(e):
                    print(f"⚠️  GPU不兼容，切换到CPU模式")
                    self.device = torch.device('cpu')
                else:
                    raise e
        else:
            self.device = torch.device('cpu')
            print(f"⚠️  使用CPU模式")

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
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # Hann窗
        self.hann_window = hann(window_size)

    def denoise(self, signal, prev_signal=None, next_signal=None):
        """
        深度学习去噪 - 自适应边界填充

        Args:
            signal: 当天信号
            prev_signal: 前一天信号（可选，用于左边界）
            next_signal: 后一天信号（可选，用于右边界）

        Returns:
            denoised: 去噪后的信号
        """
        pad_size = self.window_size // 2

        # 自适应边界填充
        if prev_signal is not None and len(prev_signal) >= pad_size:
            left_pad = prev_signal[-pad_size:]
            left_method = "邻日真实数据"
        else:
            left_pad = np.flip(signal[:pad_size])
            left_method = "镜像填充"

        if next_signal is not None and len(next_signal) >= pad_size:
            right_pad = next_signal[:pad_size]
            right_method = "邻日真实数据"
        else:
            right_pad = np.flip(signal[-pad_size:])
            right_method = "镜像填充"

        print(f"  边界策略: 左={left_method}, 右={right_method}")

        signal_padded = np.concatenate([left_pad, signal, right_pad])

        # 计算窗口数
        n_windows = (len(signal_padded) - self.window_size) // self.stride + 1

        # 输出信号 + 权重累计
        output = np.zeros(len(signal_padded))
        weights = np.zeros(len(signal_padded))

        print(f"  DL处理: {n_windows} 个窗口...")

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
                window_tensor = torch.from_numpy(window_norm).float()
                window_tensor = window_tensor.unsqueeze(
                    0).unsqueeze(0).to(self.device)
                _, denoised_norm = self.model(window_tensor)
                denoised_window = denoised_norm.cpu().numpy().squeeze()

                # 反标准化
                denoised_window = denoised_window * std + mean

                # 加权累加 (Hann窗)
                output[start:end] += denoised_window * self.hann_window
                weights[start:end] += self.hann_window

        # 归一化
        weights = np.maximum(weights, 1e-10)
        output = output / weights

        # 去除填充，返回原始长度
        output = output[pad_size:-pad_size]

        return output

    def plot_comparison(self, original, denoised, station_name, date, save_dir):
        """绘制对比图"""
        fig = plt.figure(figsize=(16, 10))

        time_hours = np.arange(len(original)) / 3600

        # 子图1: 原始信号
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(time_hours, original, 'b-',
                 linewidth=0.5, alpha=0.7, label='原始信号')
        ax1.set_ylabel('磁场强度 (nT)', fontsize=12)
        ax1.set_title(f'{station_name} - {date} 原始信号',
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 子图2: 去噪信号
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(time_hours, denoised, 'r-',
                 linewidth=0.5, alpha=0.7, label='去噪信号')
        ax2.set_ylabel('磁场强度 (nT)', fontsize=12)
        ax2.set_title(f'{station_name} - {date} 去噪信号 (ResidualCNN)',
                      fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 子图3: 叠加对比
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(time_hours, original, 'b-',
                 linewidth=0.5, alpha=0.5, label='原始信号')
        ax3.plot(time_hours, denoised, 'r-',
                 linewidth=0.8, alpha=0.8, label='去噪信号')
        ax3.set_xlabel('时间 (小时)', fontsize=12)
        ax3.set_ylabel('磁场强度 (nT)', fontsize=12)
        ax3.set_title('原始 vs 去噪对比', fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = Path(save_dir) / f'{station_name}_{date}_DL去噪对比.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 时域对比图已保存: {save_path}")
        plt.close()

    def plot_frequency_analysis(self, original, denoised, station_name, date, save_dir):
        """绘制频域分析图"""
        fig = plt.figure(figsize=(16, 12))

        fs = 1.0  # 采样率1Hz

        # 计算功率谱
        freqs_orig, psd_orig = welch(
            original, fs=fs, nperseg=3600, noverlap=1800)
        freqs_den, psd_den = welch(
            denoised, fs=fs, nperseg=3600, noverlap=1800)

        # 子图1: 全频段功率谱对比（对数坐标）
        ax1 = plt.subplot(3, 2, 1)
        ax1.semilogy(freqs_orig, psd_orig, 'b-',
                     linewidth=1, alpha=0.7, label='原始信号')
        ax1.semilogy(freqs_den, psd_den, 'r-',
                     linewidth=1, alpha=0.7, label='去噪信号')

        # 标记地铁干扰频段
        ax1.axvspan(0.004, 0.032, alpha=0.2, color='orange', label='地铁干扰频段')

        ax1.set_xlabel('频率 (Hz)', fontsize=11)
        ax1.set_ylabel('功率谱密度 (nT²/Hz)', fontsize=11)
        ax1.set_title('(A) 全频段功率谱对比', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 0.5)

        # 子图2: 地铁干扰频段放大
        ax2 = plt.subplot(3, 2, 2)
        mask = (freqs_orig >= 0.004) & (freqs_orig <= 0.032)
        ax2.semilogy(freqs_orig[mask], psd_orig[mask],
                     'b-', linewidth=1.5, alpha=0.7, label='原始信号')
        ax2.semilogy(freqs_den[mask], psd_den[mask], 'r-',
                     linewidth=1.5, alpha=0.7, label='去噪信号')

        ax2.set_xlabel('频率 (Hz)', fontsize=11)
        ax2.set_ylabel('功率谱密度 (nT²/Hz)', fontsize=11)
        ax2.set_title('(B) 地铁干扰频段放大 (0.004-0.032 Hz)',
                      fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 计算能量抑制率
        energy_orig = np.trapz(psd_orig[mask], freqs_orig[mask])
        energy_den = np.trapz(psd_den[mask], freqs_den[mask])
        suppression_rate = (1 - energy_den / energy_orig) * 100

        ax2.text(0.98, 0.98, f'抑制率: {suppression_rate:.2f}%',
                 transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 子图3: 低频段对比（验证保真度）
        ax3 = plt.subplot(3, 2, 3)
        mask_low = freqs_orig <= 0.004
        ax3.semilogy(freqs_orig[mask_low], psd_orig[mask_low],
                     'b-', linewidth=1.5, alpha=0.7, label='原始信号')
        ax3.semilogy(freqs_den[mask_low], psd_den[mask_low],
                     'r-', linewidth=1.5, alpha=0.7, label='去噪信号')

        ax3.set_xlabel('频率 (Hz)', fontsize=11)
        ax3.set_ylabel('功率谱密度 (nT²/Hz)', fontsize=11)
        ax3.set_title('(C) 低频段对比 (<0.004 Hz)', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 计算低频相关系数
        from scipy.stats import pearsonr
        corr, _ = pearsonr(psd_orig[mask_low], psd_den[mask_low])
        ax3.text(0.98, 0.98, f'相关系数: {corr:.4f}',
                 transform=ax3.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        # 子图4: 频段能量对比柱状图
        ax4 = plt.subplot(3, 2, 4)

        # 计算各频段能量
        bands = {
            '低频\n(<0.004Hz)': (0, 0.004),
            '地铁干扰\n(0.004-0.032Hz)': (0.004, 0.032),
            '高频\n(>0.032Hz)': (0.032, 0.5)
        }

        energies_orig = []
        energies_den = []
        band_names = []

        for name, (f_low, f_high) in bands.items():
            mask = (freqs_orig >= f_low) & (freqs_orig <= f_high)
            energies_orig.append(np.trapz(psd_orig[mask], freqs_orig[mask]))
            energies_den.append(np.trapz(psd_den[mask], freqs_den[mask]))
            band_names.append(name)

        x = np.arange(len(band_names))
        width = 0.35

        bars1 = ax4.bar(x - width/2, energies_orig, width,
                        label='原始信号', alpha=0.7, color='blue')
        bars2 = ax4.bar(x + width/2, energies_den, width,
                        label='去噪信号', alpha=0.7, color='red')

        ax4.set_ylabel('能量 (nT²)', fontsize=11)
        ax4.set_title('(D) 频段能量对比', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(band_names, fontsize=10)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')

        # 在柱状图上标注数值
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.1f}',
                         ha='center', va='bottom', fontsize=9)

        # 子图5: 一阶差分对比
        ax5 = plt.subplot(3, 2, 5)

        diff_orig = np.diff(original)
        diff_den = np.diff(denoised)

        time_hours = np.arange(len(diff_orig)) / 3600

        ax5.plot(time_hours, diff_orig, 'b-',
                 linewidth=0.3, alpha=0.5, label='原始信号')
        ax5.plot(time_hours, diff_den, 'r-',
                 linewidth=0.3, alpha=0.8, label='去噪信号')

        ax5.set_xlabel('时间 (小时)', fontsize=11)
        ax5.set_ylabel('一阶差分 (nT/s)', fontsize=11)
        ax5.set_title('(E) 一阶差分对比', fontsize=13, fontweight='bold')
        ax5.legend(loc='upper right', fontsize=10)
        ax5.grid(True, alpha=0.3)

        # 计算平滑度提升
        smooth_improvement = np.std(diff_orig) / np.std(diff_den)
        ax5.text(0.02, 0.98, f'平滑度提升: {smooth_improvement:.1f}×',
                 transform=ax5.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        # 子图6: 统计信息
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')

        # 计算统计指标
        stats_text = f"""
        {station_name} - {date}
        去噪性能统计
        ━━━━━━━━━━━━━━━━━━━━━━

        【时域指标】
        • 一阶差分标准差
          原始: {np.std(diff_orig):.4f} nT/s
          去噪: {np.std(diff_den):.4f} nT/s
          平滑度提升: {smooth_improvement:.1f}×

        【频域指标】
        • 地铁干扰频段能量
          原始: {energies_orig[1]:.2f} nT²
          去噪: {energies_den[1]:.2f} nT²
          抑制率: {suppression_rate:.2f}%

        • 低频保真度
          相关系数: {corr:.4f}

        【算法参数】
        • 窗口大小: {self.window_size}秒
        • 重叠率: {self.overlap*100:.0f}%
        • 模型参数: 413K
        """

        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
                 fontsize=11, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()

        save_path = Path(save_dir) / f'{station_name}_{date}_频域分析.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 频域分析图已保存: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='纯深度学习去噪脚本')
    parser.add_argument('--station', type=str, default='BJT',
                        choices=['BJT', 'HZT'], help='台站代码')
    parser.add_argument('--date', type=str, default='2024-07-13',
                        help='日期 (YYYY-MM-DD)')
    parser.add_argument('--model', type=str,
                        default='checkpoints/residual_denoising/best_residual_model.pth',
                        help='模型路径')
    parser.add_argument('--prev_date', type=str, default=None,
                        help='前一天日期（可选，用于邻日顺延）')
    parser.add_argument('--next_date', type=str, default=None,
                        help='后一天日期（可选，用于邻日顺延）')
    args = parser.parse_args()

    # TODO: 修改为您的数据路径
    # 数据文件路径模板
    data_paths = {
        'BJT': os.path.join('data', 'raw', 'BJT', '原始秒数据_[11074]北京地震台_FGM01[G]_{date}.Txt'),
        'HZT': os.path.join('data', 'raw', 'HZT', '2025', 'OHZXH_FGMM01_PGCV_L1_DAY_{date}000000_V01.00.txt.org')
    station_names = {
        'BJT': '北京地震台',
        'HZT': '杭州植物园站'
    }

    print(f"\n{'='*80}")
    print(f"纯深度学习去噪: {station_names[args.station]} - {args.date}")
    print(f"{'='*80}\n")

    # 读取当天数据
    date_str = args.date.replace('-', '')
    file_path = data_paths[args.station].format(date=date_str)

    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        return

    data = read_hzt_file(file_path)
    if data is None or len(data) == 0:
        print(f"[错误] 数据读取失败")
        return

    # 处理dict返回
    if isinstance(data, dict):
        signal = data.get('Z', data.get('H', None))
        if signal is None:
            print(f"[错误] 数据字典中没有Z或H分量")
            return
    elif isinstance(data, np.ndarray):
        if data.ndim >= 2 and data.shape[1] >= 2:
            signal = data[:, 1]
        else:
            signal = data.ravel()
    else:
        print(f"[错误] 未知数据类型: {type(data)}")
        return

    print(f"✅ 当天数据读取成功: {len(signal)} 点")

    # 读取邻日数据（可选）
    prev_signal = None
    next_signal = None

    if args.prev_date:
        prev_date_str = args.prev_date.replace('-', '')
        prev_file = data_paths[args.station].format(date=prev_date_str)
        if os.path.exists(prev_file):
            prev_data = read_hzt_file(prev_file)
            if isinstance(prev_data, dict):
                prev_signal = prev_data.get('Z', prev_data.get('H', None))
            elif isinstance(prev_data, np.ndarray):
                prev_signal = prev_data[:,
                                        1] if prev_data.ndim >= 2 else prev_data.ravel()
            print(f"✅ 前一天数据读取成功: {len(prev_signal)} 点")

    if args.next_date:
        next_date_str = args.next_date.replace('-', '')
        next_file = data_paths[args.station].format(date=next_date_str)
        if os.path.exists(next_file):
            next_data = read_hzt_file(next_file)
            if isinstance(next_data, dict):
                next_signal = next_data.get('Z', next_data.get('H', None))
            elif isinstance(next_data, np.ndarray):
                next_signal = next_data[:,
                                        1] if next_data.ndim >= 2 else next_data.ravel()
            print(f"✅ 后一天数据读取成功: {len(next_signal)} 点")

    # 创建去噪器
    denoiser = DLDenoiser(
        model_path=args.model,
        window_size=3600,
        overlap=0.5
    )

    # 去噪
    print(f"\n开始去噪...")
    denoised = denoiser.denoise(signal, prev_signal, next_signal)
    print(f"✅ 去噪完成!")

    # 保存结果
    save_dir = 'results/dl_denoising'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    station_name = station_names[args.station]

    # 绘制时域对比图
    print(f"\n生成图表...")
    denoiser.plot_comparison(
        signal, denoised, station_name, args.date, save_dir)

    # 绘制频域分析图
    denoiser.plot_frequency_analysis(
        signal, denoised, station_name, args.date, save_dir)

    # 保存数据
    output_file = Path(save_dir) / f'{station_name}_{args.date}_去噪数据.npz'
    np.savez_compressed(
        output_file,
        original=signal,
        denoised=denoised,
        date=args.date,
        station=args.station
    )
    print(f"✅ 数据已保存: {output_file}")

    print(f"\n{'='*80}")
    print(f"处理完成! 结果保存在: {save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
