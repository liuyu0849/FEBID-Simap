#!/usr/bin/env python3
"""
FEBIP Configuration File - 扫描与可视化配置中心
物理参数已移至 physics_params.py

v2.1: 拆分物理参数到独立文件
"""

import os
from datetime import datetime
from typing import Any

# 从物理参数文件导入所有物理参数类
from .physics_params import COGEtchConfig, PSMEtchConfig, PSMDepoConfig

os.environ["NUMBA_NUM_THREADS"] = "4"


# ================================================================================
# 扫描与网格配置
# ================================================================================
class Scanning2DConfig:
    def __init__(self, output_folder_name: str = None, system_type: str = "PSM_ETCH"):
        self.system_type = system_type

        if system_type == "COG_ETCH":
            self.P_H2O = 13.3 / 1.7  # 加入配置
            self.P_XeF2 = 13.3  # 加入配置
            self.params = COGEtchConfig()
            self.params.dt = 0.4e-7  # 加入配置
            self.BEAM_GAUSSIANS = [(1.98e7, 4), (0.00e7, 10), (0e7, 24)]  # 600 V
            self.SIGMA_MIN = 4.0
            self.SIGMA_MAX = 8.0
            self.SIGMA_T_MIN = 0.01e-6
            self.SIGMA_T_MAX = 0.5e-6

        elif system_type == "PSM_ETCH":
            self.P_XeF2 = 10.0  # Pa 加入配置及
            self.params = PSMEtchConfig()
            self.params.dt = 0.04e-6  # 加入配置
            self.BEAM_GAUSSIANS = [(2.98e7, 10), (0.00e7, 20), (0e7, 24)]  # 600 V
            self.SIGMA_MIN = 2.5
            self.SIGMA_MAX = 4.0
            self.SIGMA_T_MIN = 0.01e-6
            self.SIGMA_T_MAX = 0.4e-6

        elif system_type == "PSM_DEPO":
            self.P_CrCO6 = 1.0  # 加入配置
            self.params = PSMDepoConfig()
            self.params.dt = 0.4e-7  # 加入配置
            self.BEAM_GAUSSIANS = [(2.98e7, 10), (0.00e7, 20), (0e7, 24)]  # 600 V
            self.SIGMA_MIN = 4.0
            self.SIGMA_MAX = 4.0
            self.SIGMA_T_MIN = 0.01e-6
            self.SIGMA_T_MAX = 0.4e-6

        else:
            raise ValueError(f"Unknown system_type: {system_type}")

        self.BASE_OUTPUT_DIR = ""
        if output_folder_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder_name = f"run_{system_type}_{timestamp}"
        self.output_path = os.path.join(self.BASE_OUTPUT_DIR, output_folder_name)
        os.makedirs(self.output_path, exist_ok=True)

        self.save_interval = 50
        self.VTK_SAVE_EVERY_N_POINTS = None
        self.VTK_OUTPUT_DIR = None

        self.GRID_SIZE = 100  # nm
        self.GRID_STEP = 1  # nm
        self.nx = int(self.GRID_SIZE / self.GRID_STEP) + 1
        self.ny = int(self.GRID_SIZE / self.GRID_STEP) + 1


# ================================================================================
# 可视化配置
# ================================================================================
class PlotConfig:
    def __init__(self):
        self.CROSS_SECTION_Y_MIN = None
        self.CROSS_SECTION_Y_MAX = 0.00001
        self.DPI = 300
        self.FIGSIZE_LARGE = (18, 16)
        self.FIGSIZE_MEDIUM = (12, 6)
