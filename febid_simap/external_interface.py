#!/usr/bin/env python3
"""
External Interface Module - 极简优化版
仅保留运行 main_debug.py 必需的核心接口
"""

import numpy as np
import os


def load_scan_pattern_from_file(filepath: str, delimiter=None):
    """从外部 TXT 读取扫描路径 (格式: X Y DwellTime)"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scan pattern file not found: {filepath}")

    data = np.loadtxt(filepath, delimiter=delimiter)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(
            f"Invalid format. Expected 3 columns (x,y,time), got {data.shape}"
        )

    positions = data[:, 0:2]
    dwell_times = data[:, 2]

    print(f"\n{'=' * 50}\nLoaded Scan Pattern: {filepath}")
    print(f"  Total points: {len(positions)}")
    print(f"  Avg Dwell: {np.mean(dwell_times):.2e} s\n{'=' * 50}")

    return {
        "positions": positions,
        "dwell_times": dwell_times,
        "n_points": len(positions),
    }


def export_results_to_vtk(
    filepath: str,
    X: np.ndarray,
    Y: np.ndarray,
    height_field: np.ndarray,
    scalar_fields: dict = None,
):
    """将刻蚀高度和浓度场导出为 VTK (支持 Paraview)"""
    try:
        import pyvista as pv
    except ImportError:
        print("[Warning] pyvista not installed. Skipping VTK export.")
        return

    points = np.c_[X.ravel(), Y.ravel(), height_field.ravel()]
    cloud = pv.PolyData(points)
    cloud["height"] = height_field.ravel()

    if scalar_fields:
        for name, field in scalar_fields.items():
            if field.shape == height_field.shape:
                cloud[name] = field.ravel()

    cloud.save(filepath)
