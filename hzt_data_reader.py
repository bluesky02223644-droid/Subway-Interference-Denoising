#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geomagnetic Station Data Reader
================================

Supports three data formats commonly used by Chinese seismic/geomagnetic
observation stations:

1. **DAT format** (ASCII, pre-2024)
   - Header + 1440 rows (1-minute sampling)
   - Single column per component

2. **TXT format** (Tabular, 2024+)
   - Tab-separated with timestamp and multiple components
   - 1-second or 1-minute sampling

3. **ORG format** (Binary, 2025+)
   - 4-channel int32 little-endian (H, Z, D, T)
   - 1-second sampling, 86400 samples/day

Usage:
    from hzt_data_reader import read_hzt_file

    data = read_hzt_file('path/to/data.txt')
"""

import numpy as np
import struct
import os
from pathlib import Path


def read_dat_ascii(filepath):
    """
    Read pre-2024 ASCII DAT format.

    Format:
        - Text header (variable length)
        - 1440 rows of numeric data (1-minute sampling)

    Args:
        filepath: Path to .dat file

    Returns:
        dict: {'data': numpy array, 'sampling': '1min', 'format': 'DAT'}
        None if file cannot be read
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Find where numeric data starts (skip header)
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and stripped[0].isdigit():
                try:
                    float(stripped.split()[0])
                    data_start = i
                    break
                except ValueError:
                    continue

        # Parse numeric rows
        data_rows = []
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                values = [float(v) for v in stripped.split()]
                data_rows.append(values)
            except ValueError:
                continue

        if len(data_rows) == 0:
            print(f"  [Warning] No numeric data found in: {filepath}")
            return None

        data = np.array(data_rows)
        print(f"  DAT format: {data.shape[0]} rows x {data.shape[1]} columns")
        return {'data': data, 'sampling': '1min', 'format': 'DAT'}

    except Exception as e:
        print(f"  [Error] read_dat_ascii: {e}")
        return None


def read_txt_table(filepath):
    """
    Read 2024+ tabular TXT format.

    Format:
        - Tab-separated values
        - Columns: timestamp, H, D, Z, T (or similar)
        - 1-second sampling (86400 rows) or 1-minute (1440 rows)

    Args:
        filepath: Path to .txt file

    Returns:
        dict: {'data': numpy array, 'sampling': str, 'format': 'TXT',
               'columns': list}
        None if file cannot be read
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Detect header and delimiter
        data_start = 0
        header_cols = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                data_start = i + 1
                continue
            # Attempt to parse as numeric data
            parts = stripped.split(
                '\t') if '\t' in stripped else stripped.split()
            try:
                float(parts[-1])
                data_start = i
                break
            except (ValueError, IndexError):
                header_cols = parts
                data_start = i + 1

        # Parse data rows
        data_rows = []
        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(
                '\t') if '\t' in stripped else stripped.split()
            try:
                values = [float(v) for v in parts if _is_numeric(v)]
                if len(values) > 0:
                    data_rows.append(values)
            except ValueError:
                continue

        if len(data_rows) == 0:
            print(f"  [Warning] No numeric data found in: {filepath}")
            return None

        data = np.array(data_rows)

        # Determine sampling rate
        if data.shape[0] >= 80000:
            sampling = '1s'
        elif data.shape[0] >= 1400:
            sampling = '1min'
        else:
            sampling = 'unknown'

        print(f"  TXT format: {data.shape[0]} rows x {data.shape[1]} cols, "
              f"sampling={sampling}")
        return {'data': data, 'sampling': sampling, 'format': 'TXT',
                'columns': header_cols}

    except Exception as e:
        print(f"  [Error] read_txt_table: {e}")
        return None


def read_org_binary(filepath):
    """
    Read 2025+ binary ORG format.

    Format:
        - 4-channel int32 little-endian
        - Channels: H, Z, D, T (geomagnetic components)
        - Fixed 86400 samples per channel (1-second, 24 hours)
        - Values stored as integer counts (typically nT * 10)

    Args:
        filepath: Path to .org binary file

    Returns:
        dict: {'H': array, 'Z': array, 'D': array, 'T': array,
               'sampling': '1s', 'format': 'ORG'}
        None if file cannot be read
    """
    try:
        file_size = os.path.getsize(filepath)
        expected_size = 86400 * 4 * 4  # 86400 samples * 4 channels * 4 bytes
        n_channels = 4

        if file_size < expected_size:
            # Try smaller sample count
            n_samples = file_size // (n_channels * 4)
            print(f"  [Warning] File smaller than expected. "
                  f"Reading {n_samples} samples per channel.")
        else:
            n_samples = 86400

        with open(filepath, 'rb') as f:
            raw = f.read(n_samples * n_channels * 4)

        values = struct.unpack(f'<{n_samples * n_channels}i', raw)
        data = np.array(values, dtype=np.float64).reshape(
            n_samples, n_channels)

        # Scale from integer counts to nT (int32 / 100 = physical value in nT)
        data = data / 100.0

        result = {
            'H': data[:, 0],
            'Z': data[:, 1],
            'D': data[:, 2],
            'T': data[:, 3],
            'sampling': '1s',
            'format': 'ORG'
        }

        print(f"  ORG format: {n_samples} samples, 4 channels (H/Z/D/T)")
        return result

    except Exception as e:
        print(f"  [Error] read_org_binary: {e}")
        return None


def read_hzt_file(filepath):
    """
    Auto-detect and read a geomagnetic data file.

    Supports DAT (ASCII), TXT (tabular), and ORG (binary) formats.
    Format is detected by file extension and content inspection.

    Args:
        filepath: Path to geomagnetic data file

    Returns:
        dict with data arrays, or None if unrecognized
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"  [Error] File not found: {filepath}")
        return None

    ext = filepath.suffix.lower()
    name = filepath.name.lower()

    print(f"  Reading: {filepath.name}")

    # Detect format by extension
    if ext == '.org':
        return read_org_binary(str(filepath))

    elif ext == '.dat':
        return read_dat_ascii(str(filepath))

    elif ext in ('.txt', '.csv'):
        return read_txt_table(str(filepath))

    else:
        # Try text first, then binary
        print(f"  Unknown extension '{ext}', attempting auto-detection...")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(1024)
            # If mostly printable characters, treat as text
            printable_ratio = sum(c.isprintable() or c.isspace()
                                  for c in sample) / len(sample)
            if printable_ratio > 0.8:
                return read_txt_table(str(filepath))
            else:
                return read_org_binary(str(filepath))
        except Exception:
            return read_org_binary(str(filepath))


def read_bjt_txt(filepath):
    """
    Read Beijing Seismic Station (BJT) specific TXT format.

    This is a convenience wrapper for BJT station data files,
    which use an extended header format with station metadata.

    Args:
        filepath: Path to BJT data file

    Returns:
        dict with data arrays, or None
    """
    result = read_txt_table(filepath)
    if result is not None:
        result['station'] = 'BJT'
    return result


def _is_numeric(s):
    """Check if a string can be converted to float."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ================================================================
# CLI Entry Point — for testing data readers
# ================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Read and inspect geomagnetic data files')
    parser.add_argument('filepath', type=str,
                        help='Path to geomagnetic data file')
    parser.add_argument('--show', action='store_true',
                        help='Display first 10 rows of data')
    args = parser.parse_args()

    result = read_hzt_file(args.filepath)

    if result is None:
        print("Failed to read file.")
    else:
        fmt = result.get('format', 'unknown')
        sampling = result.get('sampling', 'unknown')
        print(f"\n  Format:   {fmt}")
        print(f"  Sampling: {sampling}")

        if 'data' in result:
            data = result['data']
            print(f"  Shape:    {data.shape}")
            print(f"  Range:    [{data.min():.4f}, {data.max():.4f}]")
            if args.show:
                print(f"\n  First 10 rows:")
                print(data[:10])
        elif 'Z' in result:
            for comp in ['H', 'Z', 'D', 'T']:
                if comp in result:
                    arr = result[comp]
                    print(f"  {comp}: samples={len(arr)}, "
                          f"range=[{arr.min():.4f}, {arr.max():.4f}]")
