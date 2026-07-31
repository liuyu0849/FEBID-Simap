#!/usr/bin/env python3
"""
FEBIP PSM Deposition Model - Cr(CO)6 沉积核心物理模型

包含：
- 平衡浓度计算（2物种 + 高度）
- RHS函数（右手边微分方程）
- RK4时间积分
- Numba加速版本

沉积体系：Cr(CO)6 → Cr↓ + 6CO*, CO* → C↓ + 1/2 O2(g)
物种：Cr(CO)6*, CO*, h_material

Date: 2025
"""

import numpy as np
from numba import jit
from typing import Tuple
from ..config import PSMDepoConfig


# ========== 平衡浓度计算 ==========
def calculate_equilibrium_concentrations(params: PSMDepoConfig) -> Tuple:
    """计算无电子束下的平衡浓度（2物种沉积体系）"""
    n_sites = params.n_sites

    # Langmuir吸附等温线（无电子束时只有吸附/解吸平衡）
    alpha_CrCO6 = params.s_CrCO6 * params.Phi_CrCO6 * params.tau_CrCO6

    # 初始平衡（CO浓度为0，因为没有电子束分解）
    n_vacant_eq = n_sites / (1.0 + alpha_CrCO6)
    n_CrCO6_eq = alpha_CrCO6 * n_vacant_eq
    n_CO_eq = 0.0

    print(f"\n平衡浓度计算（PSM沉积 - Cr(CO)6体系）：")
    print(f"  n_sites = {n_sites:.2f} sites/nm²")
    print(f"  n_CrCO6_eq = {n_CrCO6_eq:.4f} molecules/nm²")
    print(f"  n_CO_eq = {n_CO_eq:.4f} molecules/nm²")
    print(f"  n_vacant_eq = {n_vacant_eq:.4f} sites/nm²")
    print(f"  初始覆盖度 Θ = {n_CrCO6_eq / n_sites:.3f}")
    print(f"  化学计量 α = {params.stoichiometry}")
    print(f"  尺寸比 β = {params.size_ratio}")

    return n_CrCO6_eq, n_CO_eq


# ========== Numba加速辅助函数 ==========
@jit(nopython=True)
def compute_vacant_sites_numba(n_CrCO6, n_CO, size_ratio, n_sites):
    """计算剩余空位数 [sites/nm²]（考虑尺寸比）"""
    n_occupied = n_CrCO6 + size_ratio * n_CO
    n_vacant = n_sites - n_occupied
    return max(n_vacant, 0.0)


# ========== Numba加速RHS函数 ==========
@jit(nopython=True)
def rhs_CrCO6_numba(
    n_CrCO6,
    n_CO,
    f,
    s_CrCO6,
    Phi_CrCO6,
    tau_CrCO6_inv,
    sigma_CrCO6,
    size_ratio,
    n_sites,
):
    """Cr(CO)6前驱体演化

    反应：
    - 吸附：Cr(CO)6(g) + □ → Cr(CO)6*
    - 解吸：Cr(CO)6* → Cr(CO)6(g) + □
    - 分解：Cr(CO)6* + e⁻ → Cr↓ + 6CO*
    """
    n_vacant = compute_vacant_sites_numba(n_CrCO6, n_CO, size_ratio, n_sites)

    adsorption = s_CrCO6 * Phi_CrCO6 * n_vacant / n_sites
    desorption = n_CrCO6 * tau_CrCO6_inv
    dissociation = sigma_CrCO6 * f * n_CrCO6

    return adsorption - desorption - dissociation


@jit(nopython=True)
def rhs_CO_numba(n_CrCO6, n_CO, f, sigma_CrCO6, stoichiometry, tau_CO_inv, sigma_CO):
    """CO中间体演化

    反应：
    - 产生：Cr(CO)6* + e⁻ → Cr↓ + 6CO*
    - 解吸：CO* → CO(g) + □
    - 分解：CO* + e⁻ → C↓ + 1/2 O2(g)
    """
    # 来自前驱体分解（化学计量）
    production = stoichiometry * sigma_CrCO6 * f * n_CrCO6

    # 解吸和分解
    desorption = n_CO * tau_CO_inv
    dissociation = sigma_CO * f * n_CO

    return production - desorption - dissociation


@jit(nopython=True)
def rhs_h_material_numba(n_CrCO6, n_CO, f, sigma_CrCO6, sigma_CO, V_Cr, V_C):
    """沉积高度变化（双沉积源）

    Cr沉积：Cr(CO)6* + e⁻ → Cr↓
    C沉积：CO* + e⁻ → C↓
    """
    # Cr沉积贡献
    rate_Cr = V_Cr * sigma_CrCO6 * f * n_CrCO6

    # C沉积贡献
    rate_C = V_C * sigma_CO * f * n_CO

    dh_dt = rate_Cr + rate_C
    return dh_dt


# ========== Numba加速RK4积分（2物种+高度）==========
@jit(nopython=True)
def rk4_step_single_point_numba(
    n_CrCO6,
    n_CO,
    h_material,
    f,
    dt,
    s_CrCO6,
    Phi_CrCO6,
    tau_CrCO6_inv,
    sigma_CrCO6,
    tau_CO_inv,
    sigma_CO,
    stoichiometry,
    size_ratio,
    n_sites,
    V_Cr,
    V_C,
):
    """2物种沉积模型的RK4时间步进（Numba加速）"""

    # ========== k1 阶段 ==========
    k1_CrCO6 = dt * rhs_CrCO6_numba(
        n_CrCO6,
        n_CO,
        f,
        s_CrCO6,
        Phi_CrCO6,
        tau_CrCO6_inv,
        sigma_CrCO6,
        size_ratio,
        n_sites,
    )
    k1_CO = dt * rhs_CO_numba(
        n_CrCO6, n_CO, f, sigma_CrCO6, stoichiometry, tau_CO_inv, sigma_CO
    )
    k1_h = dt * rhs_h_material_numba(n_CrCO6, n_CO, f, sigma_CrCO6, sigma_CO, V_Cr, V_C)

    # ========== k2 阶段 ==========
    k2_CrCO6 = dt * rhs_CrCO6_numba(
        n_CrCO6 + 0.5 * k1_CrCO6,
        n_CO + 0.5 * k1_CO,
        f,
        s_CrCO6,
        Phi_CrCO6,
        tau_CrCO6_inv,
        sigma_CrCO6,
        size_ratio,
        n_sites,
    )
    k2_CO = dt * rhs_CO_numba(
        n_CrCO6 + 0.5 * k1_CrCO6,
        n_CO + 0.5 * k1_CO,
        f,
        sigma_CrCO6,
        stoichiometry,
        tau_CO_inv,
        sigma_CO,
    )
    k2_h = dt * rhs_h_material_numba(
        n_CrCO6 + 0.5 * k1_CrCO6,
        n_CO + 0.5 * k1_CO,
        f,
        sigma_CrCO6,
        sigma_CO,
        V_Cr,
        V_C,
    )

    # ========== k3 阶段 ==========
    k3_CrCO6 = dt * rhs_CrCO6_numba(
        n_CrCO6 + 0.5 * k2_CrCO6,
        n_CO + 0.5 * k2_CO,
        f,
        s_CrCO6,
        Phi_CrCO6,
        tau_CrCO6_inv,
        sigma_CrCO6,
        size_ratio,
        n_sites,
    )
    k3_CO = dt * rhs_CO_numba(
        n_CrCO6 + 0.5 * k2_CrCO6,
        n_CO + 0.5 * k2_CO,
        f,
        sigma_CrCO6,
        stoichiometry,
        tau_CO_inv,
        sigma_CO,
    )
    k3_h = dt * rhs_h_material_numba(
        n_CrCO6 + 0.5 * k2_CrCO6,
        n_CO + 0.5 * k2_CO,
        f,
        sigma_CrCO6,
        sigma_CO,
        V_Cr,
        V_C,
    )

    # ========== k4 阶段 ==========
    k4_CrCO6 = dt * rhs_CrCO6_numba(
        n_CrCO6 + k3_CrCO6,
        n_CO + k3_CO,
        f,
        s_CrCO6,
        Phi_CrCO6,
        tau_CrCO6_inv,
        sigma_CrCO6,
        size_ratio,
        n_sites,
    )
    k4_CO = dt * rhs_CO_numba(
        n_CrCO6 + k3_CrCO6,
        n_CO + k3_CO,
        f,
        sigma_CrCO6,
        stoichiometry,
        tau_CO_inv,
        sigma_CO,
    )
    k4_h = dt * rhs_h_material_numba(
        n_CrCO6 + k3_CrCO6, n_CO + k3_CO, f, sigma_CrCO6, sigma_CO, V_Cr, V_C
    )

    # ========== 更新状态（RK4组合）==========
    n_CrCO6_new = n_CrCO6 + (k1_CrCO6 + 2 * k2_CrCO6 + 2 * k3_CrCO6 + k4_CrCO6) / 6.0
    n_CO_new = n_CO + (k1_CO + 2 * k2_CO + 2 * k3_CO + k4_CO) / 6.0
    h_material_new = h_material + (k1_h + 2 * k2_h + 2 * k3_h + k4_h) / 6.0

    # 物理约束：非负
    n_CrCO6_new = max(n_CrCO6_new, 0.0)
    n_CO_new = max(n_CO_new, 0.0)
    h_material_new = max(h_material_new, 0.0)

    return n_CrCO6_new, n_CO_new, h_material_new
