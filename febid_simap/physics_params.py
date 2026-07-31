#!/usr/bin/env python3
"""
FEBIP Physics Parameters - 物理参数配置文件
包含所有刻蚀和沉积模型的物理参数

此文件可使用 PyArmor 加密以保护核心物理参数
Date: 2025
"""

import numpy as np
from dataclasses import dataclass


# ================================================================================
# 基础物理参数（刻蚀和沉积通用）
# ================================================================================
@dataclass
class BaseSurfaceProcessConfig:
    """表面反应过程的基础配置（刻蚀和沉积通用）

    适用于：
    - 刻蚀模型（Etching）：材料移除过程
    - 沉积模型（Deposition）：材料添加过程
    """

    system_type: str = "BASE"
    process_type: str = "etching"  # 'etching' 或 'deposition'

    # 通用物理参数
    T: float = 300.0  # 温度 [K]，用于计算分子通量
    dt: float = 5e-8  # 默认时间步长 [s]

    # 材料特定参数（必须在子类中定义）
    # n_sites: float  # 表面吸附位点密度 [sites/nm²] - 材料特定，在子类中定义
    h_initial: float = 1.0  # 初始高度 [nm]

    # 刻蚀模型：>0（有初始厚度）
    # 沉积模型：=0（从零开始）

    @staticmethod
    def calculate_flux(P_Pa: float, m_amu: float, T_K: float) -> float:
        """计算分子通量 (Hertz-Knudsen formula)

        适用于气体分子和前驱体分子的通量计算
        """
        k_B = 1.380649e-23
        amu = 1.66054e-27
        m_kg = m_amu * amu
        F = P_Pa / np.sqrt(2 * np.pi * m_kg * k_B * T_K)
        return F * 1e-18  # 返回 [molecules/(nm^2·s)]


# ================================================================================
# 向后兼容：保留旧名称作为别名
# ================================================================================
BaseEtchConfig = BaseSurfaceProcessConfig  # 向后兼容 v1.0 代码


# ================================================================================
# COG 系统参数（Chromium on Glass - 刻蚀模型）
# ================================================================================
@dataclass
class COGEtchConfig(BaseSurfaceProcessConfig):
    """Chromium on Glass 刻蚀模型配置

    刻蚀体系：Cr + H2O/XeF2 → CrF3(g), CrO2F2(g)
    物种：H2O*, XeF2*, OH*, F*, H*, CrO2*, CrO2F2*
    """

    system_type: str = "COG_ETCH"
    process_type: str = "etching"  # 刻蚀模型

    # Chromium 表面参数
    n_sites: float = 4.0  # Cr 表面吸附位点密度 [sites/nm²]

    # H2O parameters
    s_H2O: float = 1.0
    P_H2O: float = 13.3 / 1.7
    m_H2O: float = 18.0
    sigma_H2O: float = 0.4
    tau_H2O: float = 1e-5
    tau_OH: float = 1e-4
    tau_H: float = 1e-4
    D_H2O: float = 4e7
    D_OH: float = 4e7
    D_H: float = 4e7

    # XeF2 parameters
    s_XeF2: float = 1.0
    P_XeF2: float = 13.3
    m_XeF2: float = 169.0
    sigma_XeF2: float = 0.4
    tau_XeF2: float = 1e-4
    tau_F: float = 1e-2
    D_XeF2: float = 4e6
    D_F: float = 3e7

    # CrO2 / CrO2F2 parameters
    tau_CrO2: float = 1e10
    D_CrO2: float = 1e4
    sigma_CrO2F2: float = 0.1
    tau_CrO2F2: float = 1e-8
    D_CrO2F2: float = 1e6

    # Reaction rate constants
    k_CrO2: float = 0.5e8
    k_CrF3: float = 1e3
    k_syn: float = 1e7
    k_H2: float = 1e7
    k_HF: float = 5e7

    # Material parameters
    M_Cr: float = 52.0
    rho_Cr: float = 7.19
    h_initial: float = 1.0
    V_material: float = 0.012

    def __post_init__(self):
        N_A = 6.02214076e23
        self.V_material = (self.M_Cr / self.rho_Cr) / N_A * 1e21
        self.Phi_XeF2 = self.calculate_flux(self.P_XeF2, self.m_XeF2, self.T)
        self.Phi_H2O = self.calculate_flux(self.P_H2O, self.m_H2O, self.T)

        self.tau_H2O_inv = 1.0 / self.tau_H2O
        self.tau_OH_inv = 1.0 / self.tau_OH
        self.tau_H_inv = 1.0 / self.tau_H
        self.tau_XeF2_inv = 1.0 / self.tau_XeF2
        self.tau_F_inv = 1.0 / self.tau_F
        self.tau_CrO2_inv = 1.0 / self.tau_CrO2
        self.tau_CrO2F2_inv = 1.0 / self.tau_CrO2F2

    def update_pressures(self, P_H2O=None, P_XeF2=None):
        """动态更新气体压力并重新计算通量"""
        if P_H2O is not None:
            self.P_H2O = P_H2O
            self.Phi_H2O = self.calculate_flux(self.P_H2O, self.m_H2O, self.T)
            print(
                f"  [COG] Updated P_H2O = {P_H2O:.2f} Pa → Phi_H2O = {self.Phi_H2O:.2e} molecules/(nm²·s)"
            )

        if P_XeF2 is not None:
            self.P_XeF2 = P_XeF2
            self.Phi_XeF2 = self.calculate_flux(self.P_XeF2, self.m_XeF2, self.T)
            print(
                f"  [COG] Updated P_XeF2 = {P_XeF2:.2f} Pa → Phi_XeF2 = {self.Phi_XeF2:.2e} molecules/(nm²·s)"
            )


# ================================================================================
# PSM 系统参数（Phase Shift Mask - MoSi - 刻蚀模型）
# ================================================================================
@dataclass
class PSMEtchConfig(BaseSurfaceProcessConfig):
    """Phase Shift Mask (MoSi) 刻蚀模型配置

    刻蚀体系：MoSi + 10F* → MoF6(g) + SiF4(g)
    物种：XeF2*, F*
    """

    system_type: str = "PSM_ETCH"
    process_type: str = "etching"  # 刻蚀模型

    # MoSi 表面参数
    n_sites: float = 4.0  # MoSi 表面吸附位点密度 [sites/nm²]

    # XeF2 parameters
    s_XeF2: float = 1  # 1
    P_XeF2: float = 10.0
    m_XeF2: float = 169.0
    sigma_XeF2: float = 0.15
    tau_XeF2: float = 1e-6
    tau_F: float = 1
    D_XeF2: float = 2e6
    D_F: float = 1e4

    # Reaction rate constants
    k_MoSiF10: float = 1e5

    # Material parameters
    M_Mo: float = 95.94
    M_Si: float = 28.09
    rho_MoSi: float = 6.0
    h_initial: float = 5.0
    V_material: float = 0.015

    def __post_init__(self):
        N_A = 6.02214076e23
        M_MoSi = self.M_Mo + 2 * self.M_Si
        self.V_material = (M_MoSi / self.rho_MoSi) / N_A * 1e21
        self.Phi_XeF2 = self.calculate_flux(self.P_XeF2, self.m_XeF2, self.T)

        self.tau_XeF2_inv = 1.0 / self.tau_XeF2
        self.tau_F_inv = 1.0 / self.tau_F

    def update_pressures(self, P_XeF2=None):
        """动态更新气体压力并重新计算通量"""
        if P_XeF2 is not None:
            self.P_XeF2 = P_XeF2
            self.Phi_XeF2 = self.calculate_flux(self.P_XeF2, self.m_XeF2, self.T)
            print(
                f"  [PSM] Updated P_XeF2 = {P_XeF2:.2f} Pa → Phi_XeF2 = {self.Phi_XeF2:.2e} molecules/(nm²·s)"
            )


# ================================================================================
# PSM 沉积系统参数（Phase Shift Mask - Cr(CO)6 - 沉积模型）
# ================================================================================
@dataclass
class PSMDepoConfig(BaseSurfaceProcessConfig):
    """Phase Shift Mask Cr(CO)6 沉积模型配置

    沉积体系：Cr(CO)6 → Cr↓ + 6CO*, CO* → C↓ + 1/2 O2(g)
    物种：Cr(CO)6*, CO*
    """

    system_type: str = "PSM_DEPO"
    process_type: str = "deposition"  # 沉积模型

    # 表面参数
    n_sites: float = 2.8  # 表面位点密度 [sites/nm²] (调优值)

    # 化学计量参数
    stoichiometry: int = 6  # α: 每个Cr(CO)6产生6个CO

    # Cr(CO)6 parameters (调优值)
    s_CrCO6: float = 0.01  # 吸附系数 (调优值)
    P_CrCO6: float = 10.0  # Pa
    m_CrCO6: float = 220.0  # g/mol
    sigma_CrCO6: float = 0.42  # nm² (调优值)
    tau_CrCO6: float = 1.2779e-6  # s (调优值)
    D_CrCO6: float = 1e7  # nm²/s (调优值)

    # CO parameters (调优值)
    tau_CO: float = 1e-5  # s
    sigma_CO: float = 0.3  # nm² (调优值)
    D_CO: float = 1e7  # nm²/s

    # Material parameters
    M_Cr: float = 52.0  # g/mol
    rho_Cr: float = 7.19  # g/cm³
    M_C: float = 12.0  # g/mol
    rho_C: float = 2.0  # g/cm³ (无定形碳)
    h_initial: float = 0.0  # 沉积从零开始
    V_Cr: float = 0.012  # nm³ (will be calculated)
    V_C: float = 0.010  # nm³ (will be calculated)

    def __post_init__(self):
        N_A = 6.02214076e23

        # 自动设置 β = α（位点守恒）
        self.size_ratio = self.stoichiometry

        # 计算原子体积
        self.V_Cr = (self.M_Cr / self.rho_Cr) / N_A * 1e21  # nm³/atom
        self.V_C = (self.M_C / self.rho_C) / N_A * 1e21  # nm³/atom

        # 计算分子通量
        self.Phi_CrCO6 = self.calculate_flux(self.P_CrCO6, self.m_CrCO6, self.T)

        # 预计算倒数
        self.tau_CrCO6_inv = 1.0 / self.tau_CrCO6
        self.tau_CO_inv = 1.0 / self.tau_CO

    def update_pressures(self, P_CrCO6=None):
        """动态更新气体压力并重新计算通量"""
        if P_CrCO6 is not None:
            self.P_CrCO6 = P_CrCO6
            self.Phi_CrCO6 = self.calculate_flux(self.P_CrCO6, self.m_CrCO6, self.T)
            print(
                f"  [PSM_DEPO] Updated P_CrCO6 = {P_CrCO6:.2f} Pa → Phi_CrCO6 = {self.Phi_CrCO6:.2e} molecules/(nm²·s)"
            )
