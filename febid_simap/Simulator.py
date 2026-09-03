#!/usr/bin/env python3
"""
FEBIP Simulator Module - 插件式架构（通用版本）
优化：应用 Numba 并行化融合扩散内核与零拷贝内存池
"""

import numpy as np
import time
from numba import jit, prange
from tqdm import tqdm
from .models_registry import ModelRegistry


# ========== Numba 融合加速：Laplacian + In-place Diffusion ==========
@jit(nopython=True, parallel=True)
def apply_diffusion_kernel(field, lap_buffer, D_dt, dx, dy, mask):
    """
    计算 Laplacian 并原地更新浓度场 (Kernel Fusion)。
    这完全替代了过去的数组切片，彻底消除了显式欧拉法中的中间临时数组。
    """
    ny, nx = field.shape
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)

    # 1. 计算 Laplacian 存入 lap_buffer (避免读写冲突)
    # 内部点
    for i in prange(1, ny - 1):
        for j in range(1, nx - 1):
            lap_buffer[i, j] = (
                field[i + 1, j] + field[i - 1, j] - 2.0 * field[i, j]
            ) * idy2 + (field[i, j + 1] + field[i, j - 1] - 2.0 * field[i, j]) * idx2

    # 边界点 (Zero-flux)
    for j in range(1, nx - 1):
        lap_buffer[0, j] = (field[1, j] - field[0, j]) * idy2 + (
            field[0, j + 1] + field[0, j - 1] - 2.0 * field[0, j]
        ) * idx2
        lap_buffer[ny - 1, j] = (field[ny - 2, j] - field[ny - 1, j]) * idy2 + (
            field[ny - 1, j + 1] + field[ny - 1, j - 1] - 2.0 * field[ny - 1, j]
        ) * idx2

    for i in range(1, ny - 1):
        lap_buffer[i, 0] = (
            field[i + 1, 0] + field[i - 1, 0] - 2.0 * field[i, 0]
        ) * idy2 + (field[i, 1] - field[i, 0]) * idx2
        lap_buffer[i, nx - 1] = (
            field[i + 1, nx - 1] + field[i - 1, nx - 1] - 2.0 * field[i, nx - 1]
        ) * idy2 + (field[i, nx - 2] - field[i, nx - 1]) * idx2

    # 角点
    lap_buffer[0, 0] = (field[1, 0] - field[0, 0]) * idy2 + (
        field[0, 1] - field[0, 0]
    ) * idx2
    lap_buffer[0, nx - 1] = (field[1, nx - 1] - field[0, nx - 1]) * idy2 + (
        field[0, nx - 2] - field[0, nx - 1]
    ) * idx2
    lap_buffer[ny - 1, 0] = (field[ny - 2, 0] - field[ny - 1, 0]) * idy2 + (
        field[ny - 1, 1] - field[ny - 1, 0]
    ) * idx2
    lap_buffer[ny - 1, nx - 1] = (
        field[ny - 2, nx - 1] - field[ny - 1, nx - 1]
    ) * idy2 + (field[ny - 1, nx - 2] - field[ny - 1, nx - 1]) * idx2

    # 2. 原地加和与非负截断 (In-place update)
    for i in prange(ny):
        for j in range(nx):
            if mask[i, j]:
                val = field[i, j] + D_dt * lap_buffer[i, j]
                field[i, j] = max(val, 0.0)


# ========== Numba 隐式 ADI 扩散内核（Peaceman-Rachford，无条件稳定）==========
@jit(nopython=True, parallel=True)
def apply_diffusion_adi(field, ustar, D_dt, dx, dy, mask):
    """隐式 ADI 扩散步：显式稳定域之外（D_dt > 0.25·dx²）仍保持稳定。

    半步1: (I - rx·δxx) u* = (I + ry·δyy) u   （x 方向隐式，逐行三对角求解）
    半步2: (I - ry·δyy) u  = (I + rx·δxx) u*  （y 方向隐式，逐列三对角求解）
    边界为零通量（与显式内核一致）；更新语义同显式内核：全场求解，
    仅写回掩模内的点，并作非负截断。
    """
    ny, nx = field.shape
    rx = 0.5 * D_dt / (dx * dx)
    ry = 0.5 * D_dt / (dy * dy)

    # 三对角消元系数各行/各列相同，预先算好（Thomas 算法前向系数）
    wx = np.empty(nx)
    cpx = np.empty(nx)
    wx[0] = 1.0 / (1.0 + rx)
    cpx[0] = -rx * wx[0]
    for j in range(1, nx):
        bj = 1.0 + 2.0 * rx if j < nx - 1 else 1.0 + rx
        wx[j] = 1.0 / (bj + rx * cpx[j - 1])
        cpx[j] = -rx * wx[j]

    wy = np.empty(ny)
    cpy = np.empty(ny)
    wy[0] = 1.0 / (1.0 + ry)
    cpy[0] = -ry * wy[0]
    for i in range(1, ny):
        bi = 1.0 + 2.0 * ry if i < ny - 1 else 1.0 + ry
        wy[i] = 1.0 / (bi + ry * cpy[i - 1])
        cpy[i] = -ry * wy[i]

    # 半步1：x 隐式（并行遍历行）
    for i in prange(ny):
        dp = np.empty(nx)
        for j in range(nx):
            if i == 0:
                lap_y = field[1, j] - field[0, j]
            elif i == ny - 1:
                lap_y = field[ny - 2, j] - field[ny - 1, j]
            else:
                lap_y = field[i + 1, j] - 2.0 * field[i, j] + field[i - 1, j]
            rhs = field[i, j] + ry * lap_y
            if j == 0:
                dp[0] = rhs * wx[0]
            else:
                dp[j] = (rhs + rx * dp[j - 1]) * wx[j]
        ustar[i, nx - 1] = dp[nx - 1]
        for j in range(nx - 2, -1, -1):
            ustar[i, j] = dp[j] - cpx[j] * ustar[i, j + 1]

    # 半步2：y 隐式（并行遍历列），结果写回掩模内并截断非负
    for j in prange(nx):
        dp = np.empty(ny)
        for i in range(ny):
            if j == 0:
                lap_x = ustar[i, 1] - ustar[i, 0]
            elif j == nx - 1:
                lap_x = ustar[i, nx - 2] - ustar[i, nx - 1]
            else:
                lap_x = ustar[i, j + 1] - 2.0 * ustar[i, j] + ustar[i, j - 1]
            rhs = ustar[i, j] + rx * lap_x
            if i == 0:
                dp[0] = rhs * wy[0]
            else:
                dp[i] = (rhs + ry * dp[i - 1]) * wy[i]
        prev = dp[ny - 1]
        if mask[ny - 1, j]:
            field[ny - 1, j] = max(prev, 0.0)
        for i in range(ny - 2, -1, -1):
            xi = dp[i] - cpy[i] * prev
            if mask[i, j]:
                field[i, j] = max(xi, 0.0)
            prev = xi


class Scanning2DFEBIPSimulator:
    """通用 FEBIP 模拟器（支持任意刻蚀/沉积模型）"""

    def __init__(
        self,
        params,
        scanning_beam,
        grid_size: float,
        nx: int,
        ny: int,
        initial_height_field: np.ndarray = None,
        etch_region: tuple = None,
    ):

        self.params = params
        self.beam = scanning_beam
        self.grid_size = grid_size
        self.nx, self.ny = nx, ny

        self.system_type = params.system_type
        self.model = ModelRegistry.create(self.system_type, params)

        # 初始化网格
        self.x = np.linspace(-grid_size / 2, grid_size / 2, nx)
        self.y = np.linspace(-grid_size / 2, grid_size / 2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx, self.dy = self.x[1] - self.x[0], self.y[1] - self.y[0]

        # 反应区域掩模
        if etch_region:
            (x_min, x_max), (y_min, y_max) = etch_region
            self.etch_region_mask = (
                (self.X >= x_min)
                & (self.X <= x_max)
                & (self.Y >= y_min)
                & (self.Y <= y_max)
            )
        else:
            self.etch_region_mask = np.ones((ny, nx), dtype=bool)

        # 自定义表面处理
        if initial_height_field is not None:
            self.initial_height_field = np.asarray(initial_height_field, dtype=float)
            self.use_custom_surface = True
        else:
            self.initial_height_field = None
            self.use_custom_surface = False

        # 初始化状态
        self.state_field, self.h_material = self.model.init_state(ny, nx)

        if self.use_custom_surface:
            self.h_material = self.initial_height_field.copy()

        self.h_initial_scalar = self.params.h_initial
        self.D_array = self.model.get_diffusion_coefficients()

        # ========== 核心内存池预分配 (Zero Allocation) ==========
        self._flux_buffer = np.zeros((ny, nx), dtype=np.float64)
        self._lap_buffer = np.zeros((ny, nx), dtype=np.float64)

        print(f"\n[Simulator] Initialized with model: {self.system_type}")
        print(f"  Process type: {self.model.process_type}")
        print(
            f"  Species ({self.model.num_species}): {', '.join(self.model.species_names)}"
        )
        print(f"  Grid: {nx} x {ny} = {nx * ny} points")
        print(f"  Custom surface: {self.use_custom_surface}")

    def step(self, dt: float, flux_map: np.ndarray, force_adi: bool = False):
        """执行单步时间演化（通用版本）

        force_adi=True 时扩散步一律走隐式 ADI（二阶精度），不再按稳定域分派到
        显式欧拉（一阶精度）；自动步长模式默认启用，固定步长模式保持原分派。
        """

        # 1. 表面反应 (通过注册表模型接口)
        self.state_field, self.h_material = self.model.apply_reaction_step(
            self.state_field, self.h_material, flux_map, dt, self.etch_region_mask
        )

        # 2. 扩散演化：显式稳定域内走原显式内核（数值不变）；
        #    超出稳定域（D_dt·(1/dx²+1/dy²) > 1/2）自动切换隐式 ADI，保持无条件稳定
        explicit_limit = 0.5 / (1.0 / (self.dx * self.dx) + 1.0 / (self.dy * self.dy))
        for i in range(self.model.num_species):
            if self.D_array[i] > 0:
                D_dt = self.D_array[i] * dt
                if D_dt <= explicit_limit and not force_adi:
                    apply_diffusion_kernel(
                        self.state_field[i],
                        self._lap_buffer,
                        D_dt,
                        self.dx,
                        self.dy,
                        self.etch_region_mask,
                    )
                else:
                    apply_diffusion_adi(
                        self.state_field[i],
                        self._lap_buffer,
                        D_dt,
                        self.dx,
                        self.dy,
                        self.etch_region_mask,
                    )

    def run_scanning(
        self,
        dt: float,
        save_interval: int = 10,
        frt_dt_multiplier: float = 100,
        stop_flag_callback=None,
        vtk_save_every_n_points: int = None,
        vtk_output_dir: str = None,
        concentration_analyzer=None,
        dt_mode: str = "fixed",
        dt_rxn_factor: float = None,
        dt_max: float = None,
        dt_diff_r_max: float = 1.0,
        force_adi: bool = None,
    ):
        """运行扫描仿真。

        步长模式（dt_mode）：
        - "fixed"：固定步长，束开时用 dt，刷新等待段用 dt·frt_dt_multiplier（原有行为）；
        - "auto"：自动步长。每步取
              dt = min( dt_rxn_factor / B_max,  dt_diff_r_max · min(dx,dy)² / D_max,  dt_max )
          其中 B_max 为模型给出的反应速率上界（束开时用全场峰值通量，束关时通量为 0），
          dt_rxn_factor 为 None 时取模型默认值 dt_rxn_factor_default（基类 0.5，PSM_ETCH 0.15）；
          D_max 为最大扩散系数，第二项是扩散步的精度上限（r = D·dt/dx² ≤ dt_diff_r_max，
          ADI 无条件稳定，但一步扩散过远会抹平束斑尺度的特征），传 None 关闭。
          随后按驻留点边界切齐：边界前的剩余时长被等分为若干子步，末子步精确落在
          边界上。模型未提供速率上界时退回固定步长。
        扩散格式（force_adi）：None 表示自动模式一律走二阶 ADI、固定模式按稳定域分派
        （原有行为）；显式指定 True/False 可覆盖。
        """

        total_time = self.beam.total_scan_time
        dt_frt = (
            dt * frt_dt_multiplier
            if getattr(self.beam, "frt", None) is not None
            else dt
        )
        if dt_mode not in ("fixed", "auto"):
            raise ValueError(f"Unknown dt_mode: {dt_mode!r} (expected 'fixed' or 'auto')")
        auto_dt = dt_mode == "auto"
        if force_adi is None:
            force_adi = auto_dt
        if dt_rxn_factor is None:
            dt_rxn_factor = getattr(self.model, "dt_rxn_factor_default", 0.5)
        D_max = float(np.max(self.D_array)) if len(self.D_array) else 0.0
        dt_diff_cap = (
            dt_diff_r_max * min(self.dx, self.dy) ** 2 / D_max
            if (auto_dt and dt_diff_r_max is not None and D_max > 0)
            else None
        )

        vtk_saver = None
        if vtk_save_every_n_points and vtk_save_every_n_points > 0:
            try:
                from .vtk_realtime_saver import VTKRealtimeSaver

                vtk_saver = VTKRealtimeSaver(
                    output_dir=vtk_output_dir or "vtk_realtime",
                    save_interval=vtk_save_every_n_points,
                )
            except ImportError:
                pass

        if concentration_analyzer is not None:
            concentration_analyzer.initialize(self)
            print(f"[Simulator] Concentration analyzer enabled")

        snapshots = {
            "time": [],
            "h_material": [],
            "beam_positions": [],
            "point_indices": [],
        }

        for species_name in self.model.species_names:
            snapshots[f"n_{species_name}"] = []

        if self.use_custom_surface:
            snapshots["initial_height_field"] = self.initial_height_field.copy()

        t, step, save_counter = 0.0, 0, 0
        start_time = time.time()
        current_point_idx = -1

        print(
            f"\n[{self.system_type}] Simulation Started. Total time: {total_time * 1e3:.2f} ms"
        )

        pbar = tqdm(total=total_time)
        while t < total_time:
            if stop_flag_callback and stop_flag_callback():
                break

            beam_pos, is_active = self.beam.get_beam_position(t)
            point_idx, is_waiting = getattr(
                self.beam, "get_current_point_index", lambda x: (0, not is_active)
            )(t)

            # 使用预分配 Buffer 获取光束通量
            flux_map = self.beam.get_flux_map(t, self.X, self.Y, out=self._flux_buffer)

            # 步长选取：auto 模式按反应速率上界取步并切齐驻留边界；fixed 模式保持原逻辑
            snap_to_boundary = None
            B_max = None
            if auto_dt:
                B_max = self.model.get_reaction_rate_bound(
                    float(flux_map.max()), self.state_field
                )
            if B_max is not None:
                dt_bound = dt_rxn_factor / B_max
                if dt_diff_cap is not None:
                    dt_bound = min(dt_bound, dt_diff_cap)
                if dt_max is not None:
                    dt_bound = min(dt_bound, dt_max)
                boundary = min(self.beam.next_boundary_time(t), total_time)
                if boundary <= t:
                    boundary = total_time
                seg = boundary - t
                n_sub = max(1, int(np.ceil(seg / dt_bound - 1e-9)))
                current_dt = seg / n_sub
                if n_sub == 1:
                    snap_to_boundary = boundary
            else:
                current_dt = (
                    dt_frt
                    if is_waiting and getattr(self.beam, "frt", None) is not None
                    else dt
                )
                current_dt = min(current_dt, total_time - t)

            if (
                point_idx is not None
                and point_idx != current_point_idx
                and not is_waiting
            ):
                current_point_idx = point_idx
                if vtk_saver and vtk_saver.should_save(current_point_idx):
                    vtk_saver.save(current_point_idx, t, self)

            self.step(current_dt, flux_map, force_adi=force_adi)

            if concentration_analyzer is not None:
                concentration_analyzer.record(t, self, point_idx)

            if save_counter % save_interval == 0:
                snapshots["time"].append(t)
                snapshots["h_material"].append(self.h_material.copy())
                snapshots["beam_positions"].append(beam_pos)
                snapshots["point_indices"].append(point_idx)

                for i, species_name in enumerate(self.model.species_names):
                    snapshots[f"n_{species_name}"].append(self.state_field[i].copy())

            t = snap_to_boundary if snap_to_boundary is not None else t + current_dt
            step += 1
            save_counter += 1
            pbar.update(current_dt)
        pbar.close()

        if vtk_saver and vtk_saver.enabled:
            vtk_saver.save_final(t, self)
            vtk_saver.finalize(
                total_points=getattr(self.beam, "n_points", 0), total_time=total_time
            )

        for key in snapshots:
            if key not in [
                "time",
                "beam_positions",
                "point_indices",
                "stopped",
                "initial_height_field",
            ]:
                snapshots[key] = np.array(snapshots[key])
        snapshots["time"] = np.array(snapshots["time"])

        print(
            f"\n[Simulator] Completed. Total steps: {step}, Time: {time.time() - start_time:.2f} s"
        )

        return snapshots
