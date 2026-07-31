#!/usr/bin/env python3
"""
FEBIP PSM Model - Phase Shift Mask (MoSi刻蚀) 核心物理模型

包含：
- 平衡浓度计算（3物种）
- RHS函数（右手边微分方程）
- RK4时间积分
- Numba加速版本

刻蚀体系：MoSi + 10F* → MoF6(g) + SiF4(g)
物种：XeF2*, F*, h_MoSi（3物种体系）

Date: 2025
"""

import numpy as np
from numba import jit
from typing import Tuple
from ..config import PSMEtchConfig


# ========== 平衡浓度计算 ==========
def calculate_equilibrium_concentrations(params: PSMEtchConfig) -> Tuple:
    """计算无电子束下的平衡浓度（3物种位点守恒版本）

    在无电子束时（f=0），只有吸附/解吸平衡：
    - dn_XeF2/dt = s_XeF2 * Phi_XeF2 * n_vacant - n_XeF2 / tau_XeF2 = 0
    - 其他活性物种（F*）浓度为0

    求解：n_XeF2_eq = n_occupied_eq
    """
    n_sites = params.n_sites

    # 构建平衡方程（Langmuir吸附等温线）
    alpha_XeF2 = params.s_XeF2 * params.Phi_XeF2 * params.tau_XeF2

    # 竞争吸附平衡
    n_vacant_eq = n_sites / (1.0 + alpha_XeF2)
    n_XeF2_eq = alpha_XeF2 * n_vacant_eq

    # 活性物种初始为0
    n_F_eq = 0.0

    print(f"\n平衡浓度计算（3物种MoSi刻蚀体系）：")
    print(f"  n_sites = {n_sites:.2f} sites/nm²")
    print(f"  n_XeF2_eq = {n_XeF2_eq:.4f} molecules/nm²")
    print(f"  n_F_eq = {n_F_eq:.4f} molecules/nm²")
    print(f"  n_vacant_eq = {n_vacant_eq:.4f} sites/nm²")
    print(f"  初始总覆盖度 Θ = {n_XeF2_eq / n_sites:.3f}")

    return n_XeF2_eq, n_F_eq


# ========== Numba加速辅助函数 ==========
@jit(nopython=True)
def compute_vacant_sites_numba(n_XeF2, n_F, n_sites):
    """计算剩余空位数 [sites/nm²]（3物种版本）

    位点守恒：所有吸附物种都占据表面位点
    - XeF2*, F* 各占 1 个位点
    """
    n_occupied = n_XeF2 + n_F
    n_vacant = n_sites - n_occupied
    return max(n_vacant, 0.0)  # 确保非负


@jit(nopython=True)
def compute_MoSi_availability_numba(h_MoSi):
    """计算MoSi表面可用度 [无量纲]"""
    if h_MoSi <= 0:
        return 0.0
    return h_MoSi / (h_MoSi + 0.1)


# ========== Numba加速RHS函数（3物种版本）==========
@jit(nopython=True)
def rhs_XeF2_numba(
    n_XeF2, n_F, h_MoSi, f, s_XeF2, Phi_XeF2, tau_XeF2_inv, sigma_XeF2, n_sites
):
    """XeF2分子演化

    反应：
    - 吸附：XeF2(g) + □ → XeF2*      速率 = s * Φ * n_vacant
    - 解吸：XeF2* → XeF2(g) + □      速率 = n_XeF2 / τ
    - 解离：XeF2* + e⁻ → 2F* + Xe(g) 速率 = σ * f * n_XeF2
    """
    n_vacant = compute_vacant_sites_numba(n_XeF2, n_F, n_sites)

    adsorption = s_XeF2 * Phi_XeF2 * n_vacant / n_sites
    desorption = n_XeF2 * tau_XeF2_inv
    dissociation = sigma_XeF2 * f * n_XeF2

    return adsorption - desorption - dissociation


@jit(nopython=True)
def rhs_F_numba(n_XeF2, n_F, h_MoSi, f, sigma_XeF2, tau_F_inv, k_MoSiF10, n_sites):
    """F*碎片演化

    反应：
    - 产生：XeF2* + e⁻ → 2F* + Xe(g)        速率 = σ * f * n_XeF2
    - 解吸：F* → F(g) + □                   速率 = n_F / τ
    - 消耗：10F* + MoSi → MoF6(g) + SiF4(g) 速率 = k * n_F^10 * Θ_MoSi
    """
    production_from_XeF2 = 2.0 * sigma_XeF2 * f * n_XeF2
    desorption = n_F * tau_F_inv

    Theta_MoSi = compute_MoSi_availability_numba(h_MoSi)
    R_MoSiF10 = k_MoSiF10 * (n_F**10) * Theta_MoSi
    consumption_etch = 10.0 * R_MoSiF10

    return production_from_XeF2 - desorption - consumption_etch


@jit(nopython=True)
def rhs_h_MoSi_numba(n_XeF2, n_F, h_MoSi, f, k_MoSiF10, V_MoSi, n_sites):
    """MoSi高度变化

    刻蚀反应：10F* + MoSi → MoF6(g)↑ + SiF4(g)↑
    """
    if h_MoSi <= 0:
        return 0.0

    Theta_MoSi = compute_MoSi_availability_numba(h_MoSi)
    R_MoSiF10 = k_MoSiF10 * (n_F**10) * Theta_MoSi

    dh_dt = -V_MoSi * R_MoSiF10
    return dh_dt


# ========== Numba加速RK4积分（3物种版本）==========
@jit(nopython=True)
def rk4_step_single_point_numba(
    n_XeF2,
    n_F,
    h_MoSi,
    f,
    dt,
    s_XeF2,
    Phi_XeF2,
    tau_XeF2_inv,
    sigma_XeF2,
    tau_F_inv,
    n_sites,
    k_MoSiF10,
    V_MoSi,
):
    """3物种MoSi刻蚀模型的RK4时间步进（Numba加速+位点守恒）"""

    # k1
    k1_XeF2 = dt * rhs_XeF2_numba(
        n_XeF2, n_F, h_MoSi, f, s_XeF2, Phi_XeF2, tau_XeF2_inv, sigma_XeF2, n_sites
    )
    k1_F = dt * rhs_F_numba(
        n_XeF2, n_F, h_MoSi, f, sigma_XeF2, tau_F_inv, k_MoSiF10, n_sites
    )
    k1_h = dt * rhs_h_MoSi_numba(n_XeF2, n_F, h_MoSi, f, k_MoSiF10, V_MoSi, n_sites)

    # k2
    k2_XeF2 = dt * rhs_XeF2_numba(
        n_XeF2 + 0.5 * k1_XeF2,
        n_F + 0.5 * k1_F,
        h_MoSi + 0.5 * k1_h,
        f,
        s_XeF2,
        Phi_XeF2,
        tau_XeF2_inv,
        sigma_XeF2,
        n_sites,
    )
    k2_F = dt * rhs_F_numba(
        n_XeF2 + 0.5 * k1_XeF2,
        n_F + 0.5 * k1_F,
        h_MoSi + 0.5 * k1_h,
        f,
        sigma_XeF2,
        tau_F_inv,
        k_MoSiF10,
        n_sites,
    )
    k2_h = dt * rhs_h_MoSi_numba(
        n_XeF2 + 0.5 * k1_XeF2,
        n_F + 0.5 * k1_F,
        h_MoSi + 0.5 * k1_h,
        f,
        k_MoSiF10,
        V_MoSi,
        n_sites,
    )

    # k3
    k3_XeF2 = dt * rhs_XeF2_numba(
        n_XeF2 + 0.5 * k2_XeF2,
        n_F + 0.5 * k2_F,
        h_MoSi + 0.5 * k2_h,
        f,
        s_XeF2,
        Phi_XeF2,
        tau_XeF2_inv,
        sigma_XeF2,
        n_sites,
    )
    k3_F = dt * rhs_F_numba(
        n_XeF2 + 0.5 * k2_XeF2,
        n_F + 0.5 * k2_F,
        h_MoSi + 0.5 * k2_h,
        f,
        sigma_XeF2,
        tau_F_inv,
        k_MoSiF10,
        n_sites,
    )
    k3_h = dt * rhs_h_MoSi_numba(
        n_XeF2 + 0.5 * k2_XeF2,
        n_F + 0.5 * k2_F,
        h_MoSi + 0.5 * k2_h,
        f,
        k_MoSiF10,
        V_MoSi,
        n_sites,
    )

    # k4
    k4_XeF2 = dt * rhs_XeF2_numba(
        n_XeF2 + k3_XeF2,
        n_F + k3_F,
        h_MoSi + k3_h,
        f,
        s_XeF2,
        Phi_XeF2,
        tau_XeF2_inv,
        sigma_XeF2,
        n_sites,
    )
    k4_F = dt * rhs_F_numba(
        n_XeF2 + k3_XeF2,
        n_F + k3_F,
        h_MoSi + k3_h,
        f,
        sigma_XeF2,
        tau_F_inv,
        k_MoSiF10,
        n_sites,
    )
    k4_h = dt * rhs_h_MoSi_numba(
        n_XeF2 + k3_XeF2, n_F + k3_F, h_MoSi + k3_h, f, k_MoSiF10, V_MoSi, n_sites
    )

    # 更新状态
    n_XeF2_new = n_XeF2 + (k1_XeF2 + 2 * k2_XeF2 + 2 * k3_XeF2 + k4_XeF2) / 6.0
    n_F_new = n_F + (k1_F + 2 * k2_F + 2 * k3_F + k4_F) / 6.0
    h_MoSi_new = h_MoSi + (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6.0

    # 物理约束：非负
    n_XeF2_new = max(n_XeF2_new, 0.0)
    n_F_new = max(n_F_new, 0.0)
    h_MoSi_new = max(h_MoSi_new, 0.0)

    return n_XeF2_new, n_F_new, h_MoSi_new
