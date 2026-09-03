#!/usr/bin/env python3
"""
FEBIP 模型注册中心 - 插件式架构
所有刻蚀和沉积模型都在此注册，提供统一的数据结构接口

使用方法：
    @ModelRegistry.register
    class NewModel(BaseSurfaceModel):
        system_type = 'NEW_MODEL'
        ...
"""

import numpy as np
from numba import jit, prange
from typing import Dict, Type, Tuple

# NOTE: GPU 加速版模型 (PsmEtchGpuModel) 暂不使用；PSM_ETCH 回落到 CPU 版 PSMModel。
#       如需启用，补齐 models/PSM_Etch_Model_gpu.py 后再注册。

from .models import COG_Etch_Model
from .models import PSM_Etch_Model
from .models import PSM_Depo_Model
from .models.base_model import BaseSurfaceModel


# ================================================================================
# 模型注册表（工厂模式）
# ================================================================================
class ModelRegistry:
    """模型注册中心 - 自动管理所有模型"""

    _models: Dict[str, Type[BaseSurfaceModel]] = {}

    @classmethod
    def register(cls, model_class: Type[BaseSurfaceModel]):
        """注册新模型（装饰器用法）

        Example:
            @ModelRegistry.register
            class NewModel(BaseSurfaceModel):
                system_type = 'NEW_MODEL'
        """
        system_type = model_class.system_type
        if system_type in cls._models:
            print(f"[Warning] Model '{system_type}' already registered, overwriting...")
        cls._models[system_type] = model_class
        print(
            f"[Registry] Registered model: {system_type} ({model_class.process_type})"
        )
        return model_class

    @classmethod
    def create(cls, system_type: str, params) -> BaseSurfaceModel:
        """创建模型实例（工厂方法）

        Args:
            system_type: 系统类型标识符
            params: 参数配置对象

        Returns:
            model: 模型实例
        """
        if system_type not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Unknown system type: '{system_type}'. Available: {available}"
            )
        return cls._models[system_type](params)

    @classmethod
    def list_models(cls) -> list:
        """列出所有已注册模型"""
        return list(cls._models.keys())

    @classmethod
    def get_model_info(cls, system_type: str) -> dict:
        """获取模型信息"""
        if system_type not in cls._models:
            return None
        model_class = cls._models[system_type]
        return {
            "system_type": model_class.system_type,
            "process_type": model_class.process_type,
            "species": model_class.species_names,
            "num_species": len(model_class.species_names),
        }


# ================================================================================
# Numba 并行加速包裹函数
# ================================================================================
@jit(nopython=True, parallel=True)
def reaction_step_parallel_COG(
    n_H2O,
    n_XeF2,
    n_OH,
    n_F,
    n_H,
    n_CrO2,
    n_CrO2F2,
    h_Cr,
    flux_map,
    dt,
    ny,
    nx,
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
    """COG 系统并行反应步骤（Numba 加速）"""
    for i in prange(ny):
        for j in range(nx):
            f_local = flux_map[i, j]
            (
                n_H2O[i, j],
                n_XeF2[i, j],
                n_OH[i, j],
                n_F[i, j],
                n_H[i, j],
                n_CrO2[i, j],
                n_CrO2F2[i, j],
                h_Cr[i, j],
            ) = COG_Etch_Model.rk4_step_single_point_numba(
                n_H2O[i, j],
                n_XeF2[i, j],
                n_OH[i, j],
                n_F[i, j],
                n_H[i, j],
                n_CrO2[i, j],
                n_CrO2F2[i, j],
                h_Cr[i, j],
                f_local,
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
            )


@jit(nopython=True, parallel=True)
def reaction_step_parallel_PSM(
    n_XeF2,
    n_F,
    h_MoSi,
    flux_map,
    dt,
    ny,
    nx,
    s_XeF2,
    Phi_XeF2,
    tau_XeF2_inv,
    sigma_XeF2,
    tau_F_inv,
    n_sites,
    k_MoSiF10,
    V_MoSi,
):
    """PSM 系统并行反应步骤（Numba 加速）"""
    for i in prange(ny):
        for j in range(nx):
            f_local = flux_map[i, j]
            n_XeF2[i, j], n_F[i, j], h_MoSi[i, j] = (
                PSM_Etch_Model.rk4_step_single_point_numba(
                    n_XeF2[i, j],
                    n_F[i, j],
                    h_MoSi[i, j],
                    flux_map[i, j],
                    dt,
                    s_XeF2,
                    Phi_XeF2,
                    tau_XeF2_inv,
                    sigma_XeF2,
                    tau_F_inv,
                    n_sites,
                    k_MoSiF10,
                    V_MoSi,
                )
            )


@jit(nopython=True, parallel=True)
def reaction_step_parallel_PSM_DEPO(
    n_CrCO6,
    n_CO,
    h_material,
    flux_map,
    dt,
    ny,
    nx,
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
    """PSM 沉积系统并行反应步骤（Numba 加速）"""
    for i in prange(ny):
        for j in range(nx):
            f_local = flux_map[i, j]
            n_CrCO6[i, j], n_CO[i, j], h_material[i, j] = (
                PSM_Depo_Model.rk4_step_single_point_numba(
                    n_CrCO6[i, j],
                    n_CO[i, j],
                    h_material[i, j],
                    f_local,
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
                )
            )


# ================================================================================
# COG 系统插件（Chromium on Glass 刻蚀）
# ================================================================================
@ModelRegistry.register
class COGModel(BaseSurfaceModel):
    """Chromium on Glass 刻蚀模型

    刻蚀体系：Cr + H2O/XeF2 → CrF3(g), CrO2F2(g)
    物种：H2O*, XeF2*, OH*, F*, H*, CrO2*, CrO2F2*
    """

    process_type = "etching"
    system_type = "COG_ETCH"
    species_names = ["H2O", "XeF2", "OH", "F", "H", "CrO2", "CrO2F2"]

    def init_state(self, ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
        """初始化 COG 系统状态"""
        # 计算平衡浓度
        eq_vals = COG_Etch_Model.calculate_equilibrium_concentrations(self.params)

        # 初始化浓度场
        state_field = np.zeros((self.num_species, ny, nx))
        for i in range(self.num_species):
            state_field[i, :, :] = eq_vals[i]

        # 初始化材料高度
        h_material = np.full((ny, nx), self.params.h_initial)

        return state_field, h_material

    def get_diffusion_coefficients(self) -> np.ndarray:
        """返回 COG 扩散系数"""
        return np.array(
            [
                self.params.D_H2O,
                self.params.D_XeF2,
                self.params.D_OH,
                self.params.D_F,
                self.params.D_H,
                self.params.D_CrO2,
                self.params.D_CrO2F2,
            ]
        )

    def apply_reaction_step(
        self,
        state_field: np.ndarray,
        h_material: np.ndarray,
        flux_map: np.ndarray,
        dt: float,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用 COG 反应步骤"""
        # 应用掩模（掩模在仿真过程中不变：浮点掩模与乘积缓冲只建一次，逐步复用）
        mask_f = getattr(self, "_mask_f64", None)
        if mask_f is None or mask_f.shape != mask.shape:
            self._mask_f64 = mask_f = mask.astype(np.float64)
            self._masked_flux = np.empty_like(mask_f)
        np.multiply(flux_map, mask_f, out=self._masked_flux)
        flux_map_masked = self._masked_flux

        # 解包浓度场
        n_H2O, n_XeF2, n_OH, n_F, n_H, n_CrO2, n_CrO2F2 = state_field

        # 调用 Numba 加速函数
        reaction_step_parallel_COG(
            n_H2O,
            n_XeF2,
            n_OH,
            n_F,
            n_H,
            n_CrO2,
            n_CrO2F2,
            h_material,
            flux_map_masked,
            dt,
            h_material.shape[0],
            h_material.shape[1],
            self.params.s_H2O,
            self.params.Phi_H2O,
            self.params.tau_H2O_inv,
            self.params.sigma_H2O,
            self.params.s_XeF2,
            self.params.Phi_XeF2,
            self.params.tau_XeF2_inv,
            self.params.sigma_XeF2,
            self.params.tau_OH_inv,
            self.params.tau_F_inv,
            self.params.tau_H_inv,
            self.params.tau_CrO2_inv,
            self.params.tau_CrO2F2_inv,
            self.params.sigma_CrO2F2,
            self.params.n_sites,
            self.params.k_CrO2,
            self.params.k_CrF3,
            self.params.k_syn,
            self.params.k_H2,
            self.params.k_HF,
            self.params.V_material,
        )

        # 内核经由解包视图原地写回 state_field，无需重新打包复制
        return state_field, h_material


# ================================================================================
# PSM 系统插件（MoSi 刻蚀）
# ================================================================================
@ModelRegistry.register
class PSMModel(BaseSurfaceModel):
    """Phase Shift Mask (MoSi) 刻蚀模型

    刻蚀体系：MoSi + 10F* → MoF6(g) + SiF4(g)
    物种：XeF2*, F*
    """

    process_type = "etching"
    system_type = "PSM_ETCH"
    species_names = ["XeF2", "F"]

    def init_state(self, ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
        """初始化 PSM 系统状态"""
        # 计算平衡浓度
        eq_vals = PSM_Etch_Model.calculate_equilibrium_concentrations(self.params)

        # 初始化浓度场
        state_field = np.zeros((self.num_species, ny, nx))
        for i in range(self.num_species):
            state_field[i, :, :] = eq_vals[i]

        # 初始化材料高度
        h_material = np.full((ny, nx), self.params.h_initial)

        return state_field, h_material

    def get_diffusion_coefficients(self) -> np.ndarray:
        """返回 PSM 扩散系数"""
        return np.array([self.params.D_XeF2, self.params.D_F])

    def get_reaction_rate_bound(self, f_max: float, state_field: np.ndarray) -> float:
        """PSM 刻蚀反应速率上界：XeF2 为线性项；F 的消耗项 10·k·n_F^10 线性化后
        为 100·k·n_F^9·Θ（Θ ≤ 1），用全场 n_F 最大值取上界。"""
        p = self.params
        B_XeF2 = p.s_XeF2 * p.Phi_XeF2 / p.n_sites + p.tau_XeF2_inv + p.sigma_XeF2 * f_max
        nF_max = float(state_field[1].max())
        B_F = p.tau_F_inv + 100.0 * p.k_MoSiF10 * nF_max**9
        return max(B_XeF2, B_F)

    def apply_reaction_step(
        self,
        state_field: np.ndarray,
        h_material: np.ndarray,
        flux_map: np.ndarray,
        dt: float,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用 PSM 反应步骤"""
        # 应用掩模（掩模在仿真过程中不变：浮点掩模与乘积缓冲只建一次，逐步复用）
        mask_f = getattr(self, "_mask_f64", None)
        if mask_f is None or mask_f.shape != mask.shape:
            self._mask_f64 = mask_f = mask.astype(np.float64)
            self._masked_flux = np.empty_like(mask_f)
        np.multiply(flux_map, mask_f, out=self._masked_flux)
        flux_map_masked = self._masked_flux

        # 解包浓度场
        n_XeF2, n_F = state_field

        # 调用 Numba 加速函数
        reaction_step_parallel_PSM(
            n_XeF2,
            n_F,
            h_material,
            flux_map_masked,
            dt,
            h_material.shape[0],
            h_material.shape[1],
            self.params.s_XeF2,
            self.params.Phi_XeF2,
            self.params.tau_XeF2_inv,
            self.params.sigma_XeF2,
            self.params.tau_F_inv,
            self.params.n_sites,
            self.params.k_MoSiF10,
            self.params.V_material,
        )

        # 内核经由解包视图原地写回 state_field，无需重新打包复制
        return state_field, h_material


# ================================================================================
# PSM 沉积系统插件（Cr(CO)6 沉积）
# ================================================================================
@ModelRegistry.register
class PSMDepoModel(BaseSurfaceModel):
    """Cr(CO)6 沉积模型

    沉积体系：Cr(CO)6 → Cr↓ + 6CO*, CO* → C↓ + 1/2 O2(g)
    物种：Cr(CO)6*, CO*
    """

    process_type = "deposition"
    system_type = "PSM_DEPO"
    species_names = ["CrCO6", "CO"]

    def init_state(self, ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
        """初始化 PSM 沉积系统状态"""
        # 计算平衡浓度
        eq_vals = PSM_Depo_Model.calculate_equilibrium_concentrations(self.params)

        # 初始化浓度场
        state_field = np.zeros((self.num_species, ny, nx))
        for i in range(self.num_species):
            state_field[i, :, :] = eq_vals[i]

        # 初始化材料高度（沉积从零开始）
        h_material = np.full((ny, nx), self.params.h_initial)

        return state_field, h_material

    def get_diffusion_coefficients(self) -> np.ndarray:
        """返回 PSM 沉积扩散系数"""
        return np.array([self.params.D_CrCO6, self.params.D_CO])

    def get_reaction_rate_bound(self, f_max: float, state_field: np.ndarray) -> float:
        """PSM 沉积反应速率上界：束流冻结的单步内本体系为线性系统，
        特征值即对角项 B = sΦ/N + 1/τ + σ·f，逐物种取最大。"""
        p = self.params
        B_CrCO6 = (
            p.s_CrCO6 * p.Phi_CrCO6 / p.n_sites + p.tau_CrCO6_inv + p.sigma_CrCO6 * f_max
        )
        B_CO = p.tau_CO_inv + p.sigma_CO * f_max
        return max(B_CrCO6, B_CO)

    def apply_reaction_step(
        self,
        state_field: np.ndarray,
        h_material: np.ndarray,
        flux_map: np.ndarray,
        dt: float,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用 PSM 沉积反应步骤"""
        # 应用掩模（掩模在仿真过程中不变：浮点掩模与乘积缓冲只建一次，逐步复用）
        mask_f = getattr(self, "_mask_f64", None)
        if mask_f is None or mask_f.shape != mask.shape:
            self._mask_f64 = mask_f = mask.astype(np.float64)
            self._masked_flux = np.empty_like(mask_f)
        np.multiply(flux_map, mask_f, out=self._masked_flux)
        flux_map_masked = self._masked_flux

        # 解包浓度场
        n_CrCO6, n_CO = state_field

        # 调用 Numba 加速函数
        reaction_step_parallel_PSM_DEPO(
            n_CrCO6,
            n_CO,
            h_material,
            flux_map_masked,
            dt,
            h_material.shape[0],
            h_material.shape[1],
            self.params.s_CrCO6,
            self.params.Phi_CrCO6,
            self.params.tau_CrCO6_inv,
            self.params.sigma_CrCO6,
            self.params.tau_CO_inv,
            self.params.sigma_CO,
            self.params.stoichiometry,
            self.params.size_ratio,
            self.params.n_sites,
            self.params.V_Cr,
            self.params.V_C,
        )

        # 内核经由解包视图原地写回 state_field，无需重新打包复制
        return state_field, h_material


# ================================================================================
# 自动初始化：注册所有模型
# ================================================================================
print("\n" + "=" * 70)
print("FEBIP Model Registry - Plugin System Initialized")
print("=" * 70)
print(f"Available models: {', '.join(ModelRegistry.list_models())}")
for model_type in ModelRegistry.list_models():
    info = ModelRegistry.get_model_info(model_type)
    print(f"  [{model_type}] {info['process_type']}, {info['num_species']} species")
print("=" * 70 + "\n")
