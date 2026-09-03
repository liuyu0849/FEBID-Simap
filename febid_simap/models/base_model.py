from typing import Tuple

import numpy as np


class BaseSurfaceModel:
    """通用表面反应模型基类（刻蚀/沉积统一接口）"""

    process_type: str = "etching"  # 'etching' 或 'deposition'
    system_type: str = "BASE"  # 系统标识符（必须唯一）
    species_names: list = []  # 物种名称列表 ['XeF2', 'F', ...]
    dt_rxn_factor_default: float = 0.5  # 自动步长安全系数 C 的模型默认值（dt = C / B_max）

    def __init__(self, params):
        """初始化模型

        Args:
            params: 物理参数配置对象（COGConfig/PSMConfig等）
        """
        self.params = params
        self.num_species = len(self.species_names)

    def init_state(self, ny: int, nx: int) -> Tuple[np.ndarray, np.ndarray]:
        """初始化浓度场和材料高度场

        Args:
            ny, nx: 网格尺寸

        Returns:
            state_field: (num_species, ny, nx) 浓度数组
            h_material: (ny, nx) 材料高度数组
        """
        raise NotImplementedError(f"{self.system_type} must implement init_state()")

    def get_diffusion_coefficients(self) -> np.ndarray:
        """返回扩散系数数组

        Returns:
            D_array: (num_species,) 扩散系数 [D1, D2, ...]
        """
        raise NotImplementedError(
            f"{self.system_type} must implement get_diffusion_coefficients()"
        )

    def get_reaction_rate_bound(self, f_max: float, state_field: np.ndarray):
        """返回反应步线性化速率常数的全场上界 B_max [1/s]（用于自动步长）。

        自动步长按 dt = C / B_max 取步，C 为安全系数。返回 None 表示本模型
        未提供上界，自动步长对该模型退回固定 dt。

        Args:
            f_max: 当前步全场最大电子通量（束关闭时为 0）
            state_field: (num_species, ny, nx) 当前浓度场（供状态相关项估计上界）
        """
        return None

    def apply_reaction_step(
        self,
        state_field: np.ndarray,
        h_material: np.ndarray,
        flux_map: np.ndarray,
        dt: float,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用单步反应（包含 Numba 加速）

        Args:
            state_field: (num_species, ny, nx) 当前浓度场
            h_material: (ny, nx) 当前材料高度
            flux_map: (ny, nx) 电子束通量分布
            dt: 时间步长
            mask: (ny, nx) 反应区域掩模

        Returns:
            state_field_new: 更新后的浓度场
            h_material_new: 更新后的材料高度
        """
        raise NotImplementedError(
            f"{self.system_type} must implement apply_reaction_step()"
        )
