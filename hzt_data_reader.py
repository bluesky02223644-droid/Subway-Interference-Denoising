"""
HZT地磁台站数据读取模块
支持三种格式：
1. DAT (2018年): ASCII格式，相对观测（变化量）
2. TXT (2024年): 表格格式，绝对观测
3. ORG (2025年): 二进制格式，绝对观测

使用示例：
    from hzt_data_reader import read_hzt_file
    
    # 自动识别格式并读取
    data = read_hzt_file('path/to/file.dat')
    z_component = data['Z']  # Z分量 (numpy array)
    h_component = data['H']  # H分量
    d_component = data['D']  # D分量
"""

import numpy as np
from pathlib import Path


def read_dat_ascii(file_path):
    """
    读取2018年DAT格式（ASCII，相对观测）
    格式：头文件和数据在同一行，空格分隔
    头文件：日期 台站 设备ID 采样率代码 通道数 通道代码...
    数据：紧跟在头文件后面，每n_channels个值为一组
    
    返回：
        dict: {
            'Z': np.array,  # Z分量数据
            'H': np.array,  # H分量数据
            'D': np.array,  # D分量数据
            'date': str,    # 日期
            'station': str, # 台站代码
            'sampling_rate': str  # 采样率描述
        }
        如果读取失败返回 None
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.read().strip()
    
    parts = line.split()
    
    if len(parts) < 10:
        return None
    
    # 解析头文件
    date = parts[0]
    station = parts[1]
    device = parts[2]
    sample_code = parts[3]
    n_channels = int(parts[4])
    channel_codes = parts[5:5+n_channels]
    
    # 解析采样率
    if sample_code == '01':
        sampling_rate = '1秒采样'
    elif sample_code == '02':
        sampling_rate = '2秒采样'
    else:
        sampling_rate = f'{sample_code}采样'
    
    # 找到各分量的索引
    z_idx = channel_codes.index('3123') if '3123' in channel_codes else None
    h_idx = channel_codes.index('3124') if '3124' in channel_codes else None
    d_idx = channel_codes.index('3125') if '3125' in channel_codes else None
    
    if z_idx is None or h_idx is None or d_idx is None:
        return None
    
    # 数据从第5+n_channels个字段开始
    data_start_idx = 5 + n_channels
    data_values = parts[data_start_idx:]
    
    # 每n_channels个值为一组
    n_samples = len(data_values) // n_channels
    
    if n_samples == 0:
        return None
    
    # 解析数据
    z_data = []
    h_data = []
    d_data = []
    
    for i in range(n_samples):
        idx = i * n_channels
        try:
            z_data.append(float(data_values[idx + z_idx]))
            h_data.append(float(data_values[idx + h_idx]))
            d_data.append(float(data_values[idx + d_idx]))
        except (ValueError, IndexError):
            continue
    
    return {
        'Z': np.array(z_data),
        'H': np.array(h_data),
        'D': np.array(d_data),
        'date': date,
        'station': station,
        'sampling_rate': sampling_rate
    }


def read_txt_table(file_path):
    """
    读取2024年TXT格式（表格，绝对观测）
    格式：第一行为表头，后续每行为一个时间点的数据
    列：日期 时间 Z H D F P TC
    
    返回：
        dict: {
            'Z': np.array,  # Z分量数据
            'H': np.array,  # H分量数据
            'D': np.array,  # D分量数据
            'date': str,    # 日期
            'station': str, # 台站代码
            'sampling_rate': str  # 采样率描述
        }
        如果读取失败返回 None
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return None
    
    # 跳过表头，从第2行开始读取数据
    z_data = []
    h_data = []
    d_data = []
    
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 5:  # 至少需要日期+时间 + Z + H + D
            try:
                # 列顺序：日期(0) 时间(1) Z(2) H(3) D(4) F(5) P(6) TC(7)
                z_data.append(float(parts[2]))
                h_data.append(float(parts[3]))
                d_data.append(float(parts[4]))
            except (ValueError, IndexError):
                continue
    
    if len(z_data) == 0:
        return None
    
    # 从文件名提取日期
    filename = Path(file_path).name
    date = filename.split('_')[-1].split('.')[0] if '_' in filename else 'unknown'
    
    return {
        'Z': np.array(z_data),
        'H': np.array(h_data),
        'D': np.array(d_data),
        'date': date,
        'station': '33002',
        'sampling_rate': '1秒采样'
    }


def read_org_binary(file_path):
    """
    读取2025年ORG格式（二进制，绝对观测）
    格式：纯二进制，int32小端序，4通道循环 (H Z D T)
    物理单位：int32值除以100
    
    返回：
        dict: {
            'Z': np.array,  # Z分量数据
            'H': np.array,  # H分量数据
            'D': np.array,  # D分量数据
            'date': str,    # 日期
            'station': str, # 台站代码
            'sampling_rate': str  # 采样率描述
        }
        如果读取失败返回 None
    """
    with open(file_path, 'rb') as f:
        data_bytes = f.read()
    
    # 解析为int32数组（小端序）
    data_array = np.frombuffer(data_bytes, dtype='<i4')  # int32 little-endian
    
    # 4通道循环：H(0) Z(1) D(2) T(3)
    n_channels = 4
    n_samples = len(data_array) // n_channels
    
    if n_samples == 0:
        return None
    
    # 重塑为 (n_samples, n_channels)
    data_matrix = data_array[:n_samples * n_channels].reshape(n_samples, n_channels)
    
    # 提取各分量并转换为物理单位（除以100）
    h_data = data_matrix[:, 0] / 100.0
    z_data = data_matrix[:, 1] / 100.0
    d_data = data_matrix[:, 2] / 100.0
    
    # 从文件名提取台站名和日期
    # 文件名格式: OHZXH_FGMM01_PGCV_L1_DAY_20250101000000_V01.00.txt.org
    filename = Path(file_path).name
    parts = filename.split('_')
    station = parts[0] if len(parts) > 0 else 'unknown'
    date_part = parts[5] if len(parts) > 5 else 'unknown'
    date = date_part[:8] if len(date_part) >= 8 else 'unknown'
    
    return {
        'Z': z_data,
        'H': h_data,
        'D': d_data,
        'date': date,
        'station': station,
        'sampling_rate': '1秒采样'
    }


def read_hzt_file(file_path):
    """
    自动识别文件格式并读取HZT数据
    
    参数：
        file_path: 文件路径（str或Path对象）
    
    返回：
        dict: {
            'Z': np.array,  # Z分量数据
            'H': np.array,  # H分量数据
            'D': np.array,  # D分量数据
            'date': str,    # 日期
            'station': str, # 台站代码
            'sampling_rate': str  # 采样率描述
        }
        如果读取失败返回 None
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    # 根据文件扩展名选择读取方法
    if suffix == '.dat':
        return read_dat_ascii(file_path)
    elif suffix == '.txt':
        return read_txt_table(file_path)
    elif suffix == '.org':
        return read_org_binary(file_path)
    else:
        # 尝试按顺序识别
        for reader in [read_org_binary, read_txt_table, read_dat_ascii]:
            try:
                result = reader(file_path)
                if result is not None:
                    return result
            except Exception:
                continue
        return None


# BJT格式读取（与TXT格式类似，但列顺序不同）
def read_bjt_txt(file_path):
    """
    读取BJT台站TXT格式（表格，绝对观测）
    格式：第一行为表头，后续每行为一个时间点的数据
    列：时间戳 Z H D TC
    
    返回：
        dict: {
            'Z': np.array,  # Z分量数据
            'H': np.array,  # H分量数据
            'D': np.array,  # D分量数据
            'date': str,    # 日期
            'station': str, # 台站代码
            'sampling_rate': str  # 采样率描述
        }
        如果读取失败返回 None
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return None
    
    # 跳过表头，从第2行开始读取数据
    z_data = []
    h_data = []
    d_data = []
    
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) >= 5:  # 至少需要时间戳 + Z + H + D + TC
            try:
                # 列顺序：时间戳(0) Z(1) H(2) D(3) TC(4)
                z_data.append(float(parts[1]))
                h_data.append(float(parts[2]))
                d_data.append(float(parts[3]))
            except (ValueError, IndexError):
                continue
    
    if len(z_data) == 0:
        return None
    
    # 从文件名提取日期
    filename = Path(file_path).name
    date = filename.split('_')[-1].split('.')[0] if '_' in filename else 'unknown'
    
    return {
        'Z': np.array(z_data),
        'H': np.array(h_data),
        'D': np.array(d_data),
        'date': date,
        'station': 'BJT',
        'sampling_rate': '1秒采样'
    }


if __name__ == "__main__":
    # 简单测试
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        data = read_hzt_file(file_path)
        
        if data:
            print(f"✅ 成功读取文件: {Path(file_path).name}")
            print(f"  台站: {data['station']}")
            print(f"  日期: {data['date']}")
            print(f"  采样率: {data['sampling_rate']}")
            print(f"  数据点数: {len(data['Z'])}")
            print(f"  Z分量范围: {data['Z'].min():.2f} ~ {data['Z'].max():.2f}")
            print(f"  H分量范围: {data['H'].min():.2f} ~ {data['H'].max():.2f}")
            print(f"  D分量范围: {data['D'].min():.2f} ~ {data['D'].max():.2f}")
        else:
            print(f"❌ 读取失败: {file_path}")
    else:
        print("用法: python hzt_data_reader.py <文件路径>")
