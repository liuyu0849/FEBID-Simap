#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Debug Script - 全新架构完整版 (带可视化与VTK导出)

运行方式（作为包模块运行）：
    cd /home/user/FEBID-Simap
    python -m febid_simap.main
"""

import os
import numpy as np

from .config import Scanning2DConfig, PlotConfig
from .Scan_Pattern import ScanPattern2D, Scanning2DBeam
from .Simulator import Scanning2DFEBIPSimulator
from .visualization import visualize_profile
from .external_interface import load_scan_pattern_from_file


def create_dummy_scan_file(filepath):
    """如果文件不存在，自动生成一个简单的测试扫描路径"""
    print(f"[INFO] 未找到 {filepath}，正在自动生成测试文件...")
    # 生成一条简单的直线扫描，包含5个点，每个点停留 0.1us
    data = np.array(
        [
            [40.0, 50.0, 1e-7],
            [45.0, 50.0, 1e-7],
            [50.0, 50.0, 1e-7],
            [55.0, 50.0, 1e-7],
            [60.0, 50.0, 1e-7],
        ]
    )
    np.savetxt(filepath, data, fmt="%.2f %.2f %.2e", header="X Y DwellTime")


def main():
    # ========== 配置区 ==========
    SCAN_PATTERN_FILE = "scan_pattern.txt"
    system_type = "PSM_ETCH"  # 'COG_ETCH/DEPO' 或 'PSM_ETCH/DEPO'
    OUTPUT_FOLDER = None
    VTK_SAVE_INTERVAL = 5000
    # ============================

    print("\n" + "=" * 70)
    print("FEBIP Simulation - 完整框架测试 (带可视化)")
    print("=" * 70)

    # 1. 自动处理测试文件
    if not os.path.exists(SCAN_PATTERN_FILE):
        create_dummy_scan_file(SCAN_PATTERN_FILE)

    # 2. 初始化配置
    config = Scanning2DConfig(output_folder_name=OUTPUT_FOLDER, system_type=system_type)
    params = config.params
    output_path = config.output_path
    plot_config = PlotConfig()

    print(f"\n[当前系统]: {config.system_type}")
    print("=" * 70)

    # 3. 加载扫描路径
    scan_data = load_scan_pattern_from_file(SCAN_PATTERN_FILE)
    scan_pattern = ScanPattern2D.from_external_data(
        positions=scan_data["positions"], grid_size=config.GRID_SIZE, config=config
    )
    scanning_beam = Scanning2DBeam(
        gaussians=config.BEAM_GAUSSIANS,
        scan_pattern=scan_pattern,
        dwell_time_array=scan_data["dwell_times"],
    )

    # 4. 创建模拟器
    simulator = Scanning2DFEBIPSimulator(
        params=params,
        scanning_beam=scanning_beam,
        grid_size=config.GRID_SIZE,
        nx=config.nx,
        ny=config.ny,
    )

    ## 5. 运行模拟 (同时启用 VTK 保存)
    print("\n" + "=" * 70)
    print("正在运行通用模型仿真...")
    print("=" * 70)

    snapshots = simulator.run_scanning(
        dt=params.dt,
        save_interval=config.save_interval,  # 这是内存快照(画图用)的间隔
        vtk_save_every_n_points=VTK_SAVE_INTERVAL,  # vtk输出自定义间隔
        vtk_output_dir=os.path.join(output_path, "vtk_realtime"),
    )

    # 6. 生成 3D 可视化图像
    print("\n" + "=" * 70)
    print("正在生成可视化图像...")
    print("=" * 70)
    visualize_profile(
        simulator, snapshots, scan_pattern, plot_config, output_path=output_path
    )

    print("\n" + "=" * 70)
    print(f"✓ 完美运行！所有文件已输出到: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
