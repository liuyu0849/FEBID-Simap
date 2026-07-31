#!/usr/bin/env python3
"""
FEBIP COG Model - Chromium on Glass 刻蚀核心物理模型

包含：
- 平衡浓度计算（8物种）
- RHS函数（右手边微分方程）
- 优化的RK4时间积分
- Numba加速版本

刻蚀体系：Cr + H2O/XeF2 → CrF3(g), CrO2F2(g)
物种：H2O*, XeF2*, OH*, F*, H*, CrO2*, CrO2F2*, h_Cr（8物种体系）

Date: 2025
"""

import numpy as np
from numba import jit
from typing import Tuple
from ..config import COGEtchConfig


# ========== 平衡浓度计算 ==========
def calculate_equilibrium_concentrations(params: COGEtchConfig) -> Tuple:
    """计算无电子束下的平衡浓度（8物种位点守恒版本）"""
    n_sites = params.n_sites

    alpha_H2O = params.s_H2O * params.Phi_H2O * params.tau_H2O
    alpha_XeF2 = params.s_XeF2 * params.Phi_XeF2 * params.tau_XeF2

    n_vacant_eq = n_sites / (1.0 + alpha_H2O + alpha_XeF2)
    n_H2O_eq = alpha_H2O * n_vacant_eq
    n_XeF2_eq = alpha_XeF2 * n_vacant_eq

    n_OH_eq = 0.0
    n_F_eq = 0.0
    n_H_eq = 0.0
    n_CrO2_eq = 0.0
    n_CrO2F2_eq = 0.0

    print(f"\n平衡浓度计算（COG - 8物种位点守恒）：")
    print(f"  n_sites = {n_sites:.2f} sites/nm²")
    print(f"  n_H2O_eq = {n_H2O_eq:.4f} molecules/nm²")
    print(f"  n_XeF2_eq = {n_XeF2_eq:.4f} molecules/nm²")
    print(f"  n_vacant_eq = {n_vacant_eq:.4f} sites/nm²")
    print(f"  初始总覆盖度 Θ = {(n_H2O_eq + n_XeF2_eq) / n_sites:.3f}")

    return n_H2O_eq, n_XeF2_eq, n_OH_eq, n_F_eq, n_H_eq, n_CrO2_eq, n_CrO2F2_eq


# ========== Numba加速辅助函数 ==========
@jit(nopython=True)
def compute_vacant_sites_numba(
    n_H2O, n_XeF2, n_OH, n_F, n_H, n_CrO2, n_CrO2F2, n_sites
):
    """计算剩余空位数 [sites/nm²]（8物种版本）"""
    n_occupied = n_H2O + n_XeF2 + n_OH + n_F + n_H + n_CrO2 + n_CrO2F2
    n_vacant = n_sites - n_occupied
    return max(n_vacant, 0.0)


@jit(nopython=True)
def compute_Cr_availability_numba(h_Cr):
    """计算Cr表面可用度 [无量纲]"""
    if h_Cr <= 0:
        return 0.0
    return h_Cr / (h_Cr + 0.1)


# ========== Numba加速RHS函数（优化版：接收预计算的n_vacant和Theta_Cr）==========
@jit(nopython=True)
def rhs_H2O_numba(n_H2O, f, n_vacant, n_sites, s_H2O, Phi_H2O, tau_H2O_inv, sigma_H2O):
    """H2O分子演化"""
    adsorption = s_H2O * Phi_H2O * n_vacant / n_sites
    desorption = n_H2O * tau_H2O_inv
    dissociation = sigma_H2O * f * n_H2O
    return adsorption - desorption - dissociation


@jit(nopython=True)
def rhs_XeF2_numba(
    n_XeF2, f, n_vacant, n_sites, s_XeF2, Phi_XeF2, tau_XeF2_inv, sigma_XeF2
):
    """XeF2分子演化"""
    adsorption = s_XeF2 * Phi_XeF2 * n_vacant / n_sites
    desorption = n_XeF2 * tau_XeF2_inv
    dissociation = sigma_XeF2 * f * n_XeF2
    return adsorption - desorption - dissociation


@jit(nopython=True)
def rhs_OH_numba(n_H2O, n_OH, f, Theta_Cr, sigma_H2O, tau_OH_inv, k_CrO2):
    """OH*碎片演化"""
    production = sigma_H2O * f * n_H2O
    desorption = n_OH * tau_OH_inv
    R_CrO2 = k_CrO2 * (n_OH**2) * Theta_Cr
    consumption_CrO2 = 2.0 * R_CrO2
    return production - desorption - consumption_CrO2


@jit(nopython=True)
def rhs_F_numba(
    n_XeF2,
    n_F,
    n_H,
    n_CrO2,
    n_CrO2F2,
    f,
    Theta_Cr,
    sigma_XeF2,
    tau_F_inv,
    k_CrF3,
    k_syn,
    k_HF,
    sigma_CrO2F2,
):
    """F*碎片演化"""
    production_from_XeF2 = 2.0 * sigma_XeF2 * f * n_XeF2
    production_from_redep = 2.0 * sigma_CrO2F2 * f * n_CrO2F2
    production = production_from_XeF2 + production_from_redep

    desorption = n_F * tau_F_inv

    R_CrF3 = k_CrF3 * (n_F**3) * Theta_Cr
    consumption_CrF3 = 3.0 * R_CrF3

    R_syn = k_syn * (n_F**2) * n_CrO2 * Theta_Cr
    consumption_syn = 2.0 * R_syn

    R_HF = k_HF * n_H * n_F

    return production - desorption - consumption_CrF3 - consumption_syn - R_HF


@jit(nopython=True)
def rhs_H_numba(
    n_H2O, n_OH, n_H, n_F, f, Theta_Cr, sigma_H2O, tau_H_inv, k_H2, k_HF, k_CrO2
):
    """H*碎片演化"""
    production_from_H2O = sigma_H2O * f * n_H2O
    R_CrO2 = k_CrO2 * (n_OH**2) * Theta_Cr
    production_from_CrO2 = 2.0 * R_CrO2
    production = production_from_H2O + production_from_CrO2

    desorption = n_H * tau_H_inv
    R_H2 = k_H2 * (n_H**2)
    consumption_H2 = 2.0 * R_H2
    R_HF = k_HF * n_H * n_F

    return production - desorption - consumption_H2 - R_HF


@jit(nopython=True)
def rhs_CrO2_numba(
    n_OH, n_F, n_CrO2, n_CrO2F2, f, Theta_Cr, k_CrO2, k_syn, tau_CrO2_inv, sigma_CrO2F2
):
    """CrO2演化"""
    R_CrO2 = k_CrO2 * (n_OH**2) * Theta_Cr
    production_from_OH = R_CrO2
    production_from_redep = sigma_CrO2F2 * f * n_CrO2F2
    production = production_from_OH + production_from_redep

    R_syn = k_syn * (n_F**2) * n_CrO2 * Theta_Cr
    consumption_syn = R_syn
    desorption = n_CrO2 * tau_CrO2_inv

    return production - consumption_syn - desorption


@jit(nopython=True)
def rhs_CrO2F2_numba(
    n_F, n_CrO2, n_CrO2F2, f, Theta_Cr, k_syn, tau_CrO2F2_inv, sigma_CrO2F2
):
    """CrO2F2*吸附态演化"""
    R_syn = k_syn * (n_F**2) * n_CrO2 * Theta_Cr
    production = R_syn
    desorption = n_CrO2F2 * tau_CrO2F2_inv
    redissociation = sigma_CrO2F2 * f * n_CrO2F2

    return production - desorption - redissociation


@jit(nopython=True)
def rhs_h_Cr_numba(n_F, n_CrO2F2, Theta_Cr, k_CrF3, tau_CrO2F2_inv, V_Cr):
    """Cr高度变化"""
    R_CrF3 = k_CrF3 * (n_F**3) * Theta_Cr
    R_des = n_CrO2F2 * tau_CrO2F2_inv
    dh_dt = -V_Cr * (R_CrF3 + R_des)
    return dh_dt


# ========== 优化的RK4积分器：预先计算公共量 ==========
@jit(nopython=True)
def rk4_step_single_point_numba(
    n_H2O,
    n_XeF2,
    n_OH,
    n_F,
    n_H,
    n_CrO2,
    n_CrO2F2,
    h_Cr,
    f,
    dt,
    s_H2O,
    Phi_H2O,
    tau_H2O_inv,
    sigma_H2O,
    s_XeF2,
    Phi_XeF2,
    tau_XeF2_inv,
    sigma_XeF2,
    tau_OH_inv,
    tau_F_inv,
    tau_H_inv,
    tau_CrO2_inv,
    tau_CrO2F2_inv,
    sigma_CrO2F2,
    n_sites,
    k_CrO2,
    k_CrF3,
    k_syn,
    k_H2,
    k_HF,
    V_Cr,
):
    """优化的RK4时间步进：预先计算n_vacant和Theta_Cr"""

    # ========== k1 阶段 ==========
    n_vacant_k1 = compute_vacant_sites_numba(
        n_H2O, n_XeF2, n_OH, n_F, n_H, n_CrO2, n_CrO2F2, n_sites
    )
    Theta_Cr_k1 = compute_Cr_availability_numba(h_Cr)

    k1_H2O = dt * rhs_H2O_numba(
        n_H2O, f, n_vacant_k1, n_sites, s_H2O, Phi_H2O, tau_H2O_inv, sigma_H2O
    )
    k1_XeF2 = dt * rhs_XeF2_numba(
        n_XeF2, f, n_vacant_k1, n_sites, s_XeF2, Phi_XeF2, tau_XeF2_inv, sigma_XeF2
    )
    k1_OH = dt * rhs_OH_numba(
        n_H2O, n_OH, f, Theta_Cr_k1, sigma_H2O, tau_OH_inv, k_CrO2
    )
    k1_F = dt * rhs_F_numba(
        n_XeF2,
        n_F,
        n_H,
        n_CrO2,
        n_CrO2F2,
        f,
        Theta_Cr_k1,
        sigma_XeF2,
        tau_F_inv,
        k_CrF3,
        k_syn,
        k_HF,
        sigma_CrO2F2,
    )
    k1_H = dt * rhs_H_numba(
        n_H2O, n_OH, n_H, n_F, f, Theta_Cr_k1, sigma_H2O, tau_H_inv, k_H2, k_HF, k_CrO2
    )
    k1_CrO2 = dt * rhs_CrO2_numba(
        n_OH,
        n_F,
        n_CrO2,
        n_CrO2F2,
        f,
        Theta_Cr_k1,
        k_CrO2,
        k_syn,
        tau_CrO2_inv,
        sigma_CrO2F2,
    )
    k1_CrO2F2 = dt * rhs_CrO2F2_numba(
        n_F, n_CrO2, n_CrO2F2, f, Theta_Cr_k1, k_syn, tau_CrO2F2_inv, sigma_CrO2F2
    )
    k1_h = dt * rhs_h_Cr_numba(n_F, n_CrO2F2, Theta_Cr_k1, k_CrF3, tau_CrO2F2_inv, V_Cr)

    # ========== k2 阶段 ==========
    n_vacant_k2 = compute_vacant_sites_numba(
        n_H2O + 0.5 * k1_H2O,
        n_XeF2 + 0.5 * k1_XeF2,
        n_OH + 0.5 * k1_OH,
        n_F + 0.5 * k1_F,
        n_H + 0.5 * k1_H,
        n_CrO2 + 0.5 * k1_CrO2,
        n_CrO2F2 + 0.5 * k1_CrO2F2,
        n_sites,
    )
    Theta_Cr_k2 = compute_Cr_availability_numba(h_Cr + 0.5 * k1_h)

    k2_H2O = dt * rhs_H2O_numba(
        n_H2O + 0.5 * k1_H2O,
        f,
        n_vacant_k2,
        n_sites,
        s_H2O,
        Phi_H2O,
        tau_H2O_inv,
        sigma_H2O,
    )
    k2_XeF2 = dt * rhs_XeF2_numba(
        n_XeF2 + 0.5 * k1_XeF2,
        f,
        n_vacant_k2,
        n_sites,
        s_XeF2,
        Phi_XeF2,
        tau_XeF2_inv,
        sigma_XeF2,
    )
    k2_OH = dt * rhs_OH_numba(
        n_H2O + 0.5 * k1_H2O,
        n_OH + 0.5 * k1_OH,
        f,
        Theta_Cr_k2,
        sigma_H2O,
        tau_OH_inv,
        k_CrO2,
    )
    k2_F = dt * rhs_F_numba(
        n_XeF2 + 0.5 * k1_XeF2,
        n_F + 0.5 * k1_F,
        n_H + 0.5 * k1_H,
        n_CrO2 + 0.5 * k1_CrO2,
        n_CrO2F2 + 0.5 * k1_CrO2F2,
        f,
        Theta_Cr_k2,
        sigma_XeF2,
        tau_F_inv,
        k_CrF3,
        k_syn,
        k_HF,
        sigma_CrO2F2,
    )
    k2_H = dt * rhs_H_numba(
        n_H2O + 0.5 * k1_H2O,
        n_OH + 0.5 * k1_OH,
        n_H + 0.5 * k1_H,
        n_F + 0.5 * k1_F,
        f,
        Theta_Cr_k2,
        sigma_H2O,
        tau_H_inv,
        k_H2,
        k_HF,
        k_CrO2,
    )
    k2_CrO2 = dt * rhs_CrO2_numba(
        n_OH + 0.5 * k1_OH,
        n_F + 0.5 * k1_F,
        n_CrO2 + 0.5 * k1_CrO2,
        n_CrO2F2 + 0.5 * k1_CrO2F2,
        f,
        Theta_Cr_k2,
        k_CrO2,
        k_syn,
        tau_CrO2_inv,
        sigma_CrO2F2,
    )
    k2_CrO2F2 = dt * rhs_CrO2F2_numba(
        n_F + 0.5 * k1_F,
        n_CrO2 + 0.5 * k1_CrO2,
        n_CrO2F2 + 0.5 * k1_CrO2F2,
        f,
        Theta_Cr_k2,
        k_syn,
        tau_CrO2F2_inv,
        sigma_CrO2F2,
    )
    k2_h = dt * rhs_h_Cr_numba(
        n_F + 0.5 * k1_F,
        n_CrO2F2 + 0.5 * k1_CrO2F2,
        Theta_Cr_k2,
        k_CrF3,
        tau_CrO2F2_inv,
        V_Cr,
    )

    # ========== k3 阶段 ==========
    n_vacant_k3 = compute_vacant_sites_numba(
        n_H2O + 0.5 * k2_H2O,
        n_XeF2 + 0.5 * k2_XeF2,
        n_OH + 0.5 * k2_OH,
        n_F + 0.5 * k2_F,
        n_H + 0.5 * k2_H,
        n_CrO2 + 0.5 * k2_CrO2,
        n_CrO2F2 + 0.5 * k2_CrO2F2,
        n_sites,
    )
    Theta_Cr_k3 = compute_Cr_availability_numba(h_Cr + 0.5 * k2_h)

    k3_H2O = dt * rhs_H2O_numba(
        n_H2O + 0.5 * k2_H2O,
        f,
        n_vacant_k3,
        n_sites,
        s_H2O,
        Phi_H2O,
        tau_H2O_inv,
        sigma_H2O,
    )
    k3_XeF2 = dt * rhs_XeF2_numba(
        n_XeF2 + 0.5 * k2_XeF2,
        f,
        n_vacant_k3,
        n_sites,
        s_XeF2,
        Phi_XeF2,
        tau_XeF2_inv,
        sigma_XeF2,
    )
    k3_OH = dt * rhs_OH_numba(
        n_H2O + 0.5 * k2_H2O,
        n_OH + 0.5 * k2_OH,
        f,
        Theta_Cr_k3,
        sigma_H2O,
        tau_OH_inv,
        k_CrO2,
    )
    k3_F = dt * rhs_F_numba(
        n_XeF2 + 0.5 * k2_XeF2,
        n_F + 0.5 * k2_F,
        n_H + 0.5 * k2_H,
        n_CrO2 + 0.5 * k2_CrO2,
        n_CrO2F2 + 0.5 * k2_CrO2F2,
        f,
        Theta_Cr_k3,
        sigma_XeF2,
        tau_F_inv,
        k_CrF3,
        k_syn,
        k_HF,
        sigma_CrO2F2,
    )
    k3_H = dt * rhs_H_numba(
        n_H2O + 0.5 * k2_H2O,
        n_OH + 0.5 * k2_OH,
        n_H + 0.5 * k2_H,
        n_F + 0.5 * k2_F,
        f,
        Theta_Cr_k3,
        sigma_H2O,
        tau_H_inv,
        k_H2,
        k_HF,
        k_CrO2,
    )
    k3_CrO2 = dt * rhs_CrO2_numba(
        n_OH + 0.5 * k2_OH,
        n_F + 0.5 * k2_F,
        n_CrO2 + 0.5 * k2_CrO2,
        n_CrO2F2 + 0.5 * k2_CrO2F2,
        f,
        Theta_Cr_k3,
        k_CrO2,
        k_syn,
        tau_CrO2_inv,
        sigma_CrO2F2,
    )
    k3_CrO2F2 = dt * rhs_CrO2F2_numba(
        n_F + 0.5 * k2_F,
        n_CrO2 + 0.5 * k2_CrO2,
        n_CrO2F2 + 0.5 * k2_CrO2F2,
        f,
        Theta_Cr_k3,
        k_syn,
        tau_CrO2F2_inv,
        sigma_CrO2F2,
    )
    k3_h = dt * rhs_h_Cr_numba(
        n_F + 0.5 * k2_F,
        n_CrO2F2 + 0.5 * k2_CrO2F2,
        Theta_Cr_k3,
        k_CrF3,
        tau_CrO2F2_inv,
        V_Cr,
    )

    # ========== k4 阶段 ==========
    n_vacant_k4 = compute_vacant_sites_numba(
        n_H2O + k3_H2O,
        n_XeF2 + k3_XeF2,
        n_OH + k3_OH,
        n_F + k3_F,
        n_H + k3_H,
        n_CrO2 + k3_CrO2,
        n_CrO2F2 + k3_CrO2F2,
        n_sites,
    )
    Theta_Cr_k4 = compute_Cr_availability_numba(h_Cr + k3_h)

    k4_H2O = dt * rhs_H2O_numba(
        n_H2O + k3_H2O, f, n_vacant_k4, n_sites, s_H2O, Phi_H2O, tau_H2O_inv, sigma_H2O
    )
    k4_XeF2 = dt * rhs_XeF2_numba(
        n_XeF2 + k3_XeF2,
        f,
        n_vacant_k4,
        n_sites,
        s_XeF2,
        Phi_XeF2,
        tau_XeF2_inv,
        sigma_XeF2,
    )
    k4_OH = dt * rhs_OH_numba(
        n_H2O + k3_H2O, n_OH + k3_OH, f, Theta_Cr_k4, sigma_H2O, tau_OH_inv, k_CrO2
    )
    k4_F = dt * rhs_F_numba(
        n_XeF2 + k3_XeF2,
        n_F + k3_F,
        n_H + k3_H,
        n_CrO2 + k3_CrO2,
        n_CrO2F2 + k3_CrO2F2,
        f,
        Theta_Cr_k4,
        sigma_XeF2,
        tau_F_inv,
        k_CrF3,
        k_syn,
        k_HF,
        sigma_CrO2F2,
    )
    k4_H = dt * rhs_H_numba(
        n_H2O + k3_H2O,
        n_OH + k3_OH,
        n_H + k3_H,
        n_F + k3_F,
        f,
        Theta_Cr_k4,
        sigma_H2O,
        tau_H_inv,
        k_H2,
        k_HF,
        k_CrO2,
    )
    k4_CrO2 = dt * rhs_CrO2_numba(
        n_OH + k3_OH,
        n_F + k3_F,
        n_CrO2 + k3_CrO2,
        n_CrO2F2 + k3_CrO2F2,
        f,
        Theta_Cr_k4,
        k_CrO2,
        k_syn,
        tau_CrO2_inv,
        sigma_CrO2F2,
    )
    k4_CrO2F2 = dt * rhs_CrO2F2_numba(
        n_F + k3_F,
        n_CrO2 + k3_CrO2,
        n_CrO2F2 + k3_CrO2F2,
        f,
        Theta_Cr_k4,
        k_syn,
        tau_CrO2F2_inv,
        sigma_CrO2F2,
    )
    k4_h = dt * rhs_h_Cr_numba(
        n_F + k3_F, n_CrO2F2 + k3_CrO2F2, Theta_Cr_k4, k_CrF3, tau_CrO2F2_inv, V_Cr
    )

    # ========== 更新状态（RK4组合）==========
    n_H2O_new = n_H2O + (k1_H2O + 2 * k2_H2O + 2 * k3_H2O + k4_H2O) / 6.0
    n_XeF2_new = n_XeF2 + (k1_XeF2 + 2 * k2_XeF2 + 2 * k3_XeF2 + k4_XeF2) / 6.0
    n_OH_new = n_OH + (k1_OH + 2 * k2_OH + 2 * k3_OH + k4_OH) / 6.0
    n_F_new = n_F + (k1_F + 2 * k2_F + 2 * k3_F + k4_F) / 6.0
    n_H_new = n_H + (k1_H + 2 * k2_H + 2 * k3_H + k4_H) / 6.0
    n_CrO2_new = n_CrO2 + (k1_CrO2 + 2 * k2_CrO2 + 2 * k3_CrO2 + k4_CrO2) / 6.0
    n_CrO2F2_new = (
        n_CrO2F2 + (k1_CrO2F2 + 2 * k2_CrO2F2 + 2 * k3_CrO2F2 + k4_CrO2F2) / 6.0
    )
    h_Cr_new = h_Cr + (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6.0

    # 物理约束：非负
    n_H2O_new = max(n_H2O_new, 0.0)
    n_XeF2_new = max(n_XeF2_new, 0.0)
    n_OH_new = max(n_OH_new, 0.0)
    n_F_new = max(n_F_new, 0.0)
    n_H_new = max(n_H_new, 0.0)
    n_CrO2_new = max(n_CrO2_new, 0.0)
    n_CrO2F2_new = max(n_CrO2F2_new, 0.0)
    h_Cr_new = max(h_Cr_new, 0.0)

    return (
        n_H2O_new,
        n_XeF2_new,
        n_OH_new,
        n_F_new,
        n_H_new,
        n_CrO2_new,
        n_CrO2F2_new,
        h_Cr_new,
    )
