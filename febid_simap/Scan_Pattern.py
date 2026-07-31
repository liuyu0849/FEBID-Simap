#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan_Pattern.py - External Scan Pattern Interface (Highly Optimized)
精简版：保留了所有公共接口和输出结果，重构了内部的循环与数组操作。
优化：引入 Numba 局部化高斯通量计算 (4-Sigma Cut-off)
"""

import numpy as np
from numba import jit


# ========== Numba 加速的局部高斯计算 ==========
@jit(nopython=True)
def _compute_flux_map_local_numba(
    X_row, Y_col, pos_x, pos_y, gauss_f0, gauss_sig, out_flux
):
    """
    极速局部高斯分布计算：只在每个高斯分量的 4-sigma 包围盒内计算指数衰减。
    """
    nx = len(X_row)
    ny = len(Y_col)
    dx = X_row[1] - X_row[0]
    dy = Y_col[1] - Y_col[0]

    for g in range(len(gauss_sig)):
        f0 = gauss_f0[g]
        sig = gauss_sig[g]

        # 强行跳过强度为 0 的高斯分量，节省算力
        if f0 <= 0.0:
            continue

        # 4-Sigma 截断半径
        cutoff = 4.0 * sig

        # 换算为网格索引并防越界处理
        i_min = int((pos_x - cutoff - X_row[0]) / dx)
        i_max = int((pos_x + cutoff - X_row[0]) / dx) + 1
        j_min = int((pos_y - cutoff - Y_col[0]) / dy)
        j_max = int((pos_y + cutoff - Y_col[0]) / dy) + 1

        i_min = max(0, i_min)
        i_max = min(nx - 1, i_max)
        j_min = max(0, j_min)
        j_max = min(ny - 1, j_max)

        two_sig2 = 2.0 * sig * sig

        # 仅在局部包围盒内计算
        for j in range(j_min, j_max + 1):
            dy_sq = (Y_col[j] - pos_y) ** 2
            for i in range(i_min, i_max + 1):
                dx_sq = (X_row[i] - pos_x) ** 2
                out_flux[j, i] += f0 * np.exp(-(dx_sq + dy_sq) / two_sig2)


class ScanPattern2D:
    """2D scan pattern from external data only"""

    def __init__(self, positions: np.ndarray, grid_size: float, config=None):
        self.scan_path = np.asarray(positions, dtype=float)
        if self.scan_path.ndim != 2 or self.scan_path.shape[1] != 2:
            raise ValueError(
                f"positions must be shape (N, 2), got {self.scan_path.shape}"
            )

        self.grid_size = grid_size
        self.config = config
        self.is_external = True

        self.n_points = len(self.scan_path)
        self.is_single_point = self.n_points == 1
        self.subloop_indices = np.zeros(self.n_points, dtype=int)
        self.n_subloops = 1

        if self.is_single_point:
            print(
                f"\n[INFO] External single point scan: ({self.scan_path[0, 0]:.1f}, {self.scan_path[0, 1]:.1f}) nm"
            )

        print(
            f"\n{'=' * 70}\nExternal Scan Pattern Loaded\n{'=' * 70}\n"
            f"  Total points: {self.n_points}\n"
            f"  X range: [{np.min(self.scan_path[:, 0]):.2f}, {np.max(self.scan_path[:, 0]):.2f}] nm\n"
            f"  Y range: [{np.min(self.scan_path[:, 1]):.2f}, {np.max(self.scan_path[:, 1]):.2f}] nm\n"
            f"  Grid size: {self.grid_size:.1f} nm\n"
            f"  Subloops: {self.n_subloops}\n{'=' * 70}"
        )

    @classmethod
    def from_external_data(
        cls, positions: np.ndarray, grid_size: float = None, config=None
    ):
        positions = np.asarray(positions, dtype=float)
        if grid_size is None:
            grid_size = max(np.ptp(positions[:, 0]), np.ptp(positions[:, 1])) * 1.5
            print(f"  [Auto] grid_size: {grid_size:.1f} nm")
        return cls(positions, grid_size, config)


class Scanning2DBeam:
    """2D scanning electron beam (triple Gaussian model + multi-loop support)"""

    def __init__(
        self,
        gaussians: list,
        scan_pattern: ScanPattern2D,
        dwell_time: float = None,
        dwell_time_array: np.ndarray = None,
        frt: float = None,
        n_loops: int = 1,
    ):

        self.gaussians = list(gaussians)
        self.scan_pattern = scan_pattern
        self.frt = frt
        self.n_loops = max(1, n_loops)

        self.base_scan_path = scan_pattern.scan_path
        self.base_subloop_indices = scan_pattern.subloop_indices
        self.base_n_subloops = scan_pattern.n_subloops
        self.base_n_points = len(self.base_scan_path)

        # 提取高斯参数为连续数组，专供 Numba 加速使用
        self._gauss_f0 = np.array([g[0] for g in self.gaussians], dtype=np.float64)
        self._gauss_sig = np.array([g[1] for g in self.gaussians], dtype=np.float64)

        # 驻留时间处理
        if dwell_time_array is not None:
            self.base_dwell_times = np.asarray(dwell_time_array, dtype=float)
            self.dwell_time = None
            self.variable_dwell_time = True
        else:
            self.dwell_time = dwell_time
            self.base_dwell_times = np.full(self.base_n_points, dwell_time)
            self.variable_dwell_time = False

        # 高斯光束 Sigma 自适应调整
        if scan_pattern.config is not None and self.gaussians:
            current_dwell = (
                np.mean(self.base_dwell_times)
                if self.variable_dwell_time
                else self.dwell_time
            )
            t_us = current_dwell * 1e6

            norm = 0.5 * (1.0 + np.tanh(24.21 * (t_us - 0.105)))
            target_sigma = (
                scan_pattern.config.SIGMA_MIN
                + (scan_pattern.config.SIGMA_MAX - scan_pattern.config.SIGMA_MIN) * norm
            )

            self.gaussians = [(self.gaussians[0][0], target_sigma)] + self.gaussians[1:]
            # 同步更新给 Numba 的参数数组
            self._gauss_sig[0] = target_sigma

        self._expand_loops()
        self.total_scan_time = self._calculate_total_scan_time()
        self._precompute_time_boundaries()
        self._cache_subloop_id = 0

        # 驻留时刻累计表预计算与束斑图缓存（数值结果与逐步计算完全一致）
        self._cum_dwell = np.cumsum(self.dwell_times)
        self._flux_cache = None
        self._flux_cache_idx = -1

    def _expand_loops(self):
        if self.n_loops == 1:
            self.scan_path, self.dwell_times = (
                self.base_scan_path,
                self.base_dwell_times,
            )
            self.subloop_indices, self.loop_markers = self.base_subloop_indices, [0]
            self.n_subloops, self.n_points = self.base_n_subloops, self.base_n_points
        else:
            self.scan_path = np.tile(self.base_scan_path, (self.n_loops, 1))
            self.dwell_times = np.tile(self.base_dwell_times, self.n_loops)
            self.subloop_indices = np.concatenate(
                [
                    self.base_subloop_indices + i * self.base_n_subloops
                    for i in range(self.n_loops)
                ]
            )
            self.loop_markers = [i * self.base_n_subloops for i in range(self.n_loops)]
            self.n_subloops = self.base_n_subloops * self.n_loops
            self.n_points = self.base_n_points * self.n_loops

    def _calculate_total_scan_time(self):
        if self.frt is None:
            return np.sum(self.dwell_times)
        return sum(
            max(np.sum(self.dwell_times[self.subloop_indices == i]), self.frt)
            for i in range(self.n_subloops)
        )

    def _precompute_time_boundaries(self):
        self.time_boundaries = []
        if self.frt is None:
            return

        current_time = 0.0
        for i in range(self.n_subloops):
            mask = self.subloop_indices == i
            dwells = self.dwell_times[mask]
            scan_time = np.sum(dwells)
            total_time = max(scan_time, self.frt)

            self.time_boundaries.append(
                {
                    "subloop_id": i,
                    "start_time": current_time,
                    "scan_end_time": current_time + scan_time,
                    "end_time": current_time + total_time,
                    "point_indices": np.where(mask)[0],
                    "dwell_times": dwells,
                    "cum_dwell": np.cumsum(dwells),
                    "n_points": len(dwells),
                }
            )
            current_time += total_time

    def _time_to_point_index(self, t: float):
        if t >= self.total_scan_time:
            return None, False

        if self.frt is None:
            idx = np.searchsorted(self._cum_dwell, t, side="right")
            return min(idx, self.n_points - 1), False

        if (
            self._cache_subloop_id < len(self.time_boundaries)
            and self.time_boundaries[self._cache_subloop_id]["start_time"]
            <= t
            < self.time_boundaries[self._cache_subloop_id]["end_time"]
        ):
            subloop_id = self._cache_subloop_id
        else:
            left, right = 0, len(self.time_boundaries) - 1
            while left < right:
                mid = (left + right) // 2
                if t < self.time_boundaries[mid]["end_time"]:
                    right = mid
                else:
                    left = mid + 1
            subloop_id = self._cache_subloop_id = left

        b = self.time_boundaries[subloop_id]
        t_sub = t - b["start_time"]
        if t_sub >= (b["scan_end_time"] - b["start_time"]):
            return b["point_indices"][-1], True

        idx = min(
            np.searchsorted(b["cum_dwell"], t_sub, side="right"),
            b["n_points"] - 1,
        )
        return b["point_indices"][idx], False

    def get_beam_position(self, t: float):
        idx, waiting = self._time_to_point_index(t)
        return (None, False) if idx is None else (self.scan_path[idx], not waiting)

    def get_current_point_index(self, t: float):
        return self._time_to_point_index(t)

    def get_current_subloop_id(self, t: float):
        idx, _ = self._time_to_point_index(t)
        return self.subloop_indices[idx] if idx is not None else None

    def get_current_loop_id(self, t: float):
        sub_id = self.get_current_subloop_id(t)
        return sub_id // self.base_n_subloops if sub_id is not None else None

    def get_flux_map(
        self, t: float, X: np.ndarray, Y: np.ndarray, out: np.ndarray = None
    ) -> np.ndarray:
        """
        极速生成高斯通量图：支持预分配数组传入 (out) 消除内存拷贝。
        """
        if out is None:
            out = np.zeros_like(X)

        idx, waiting = self._time_to_point_index(t)
        if idx is None or waiting:
            out.fill(0.0)
            return out

        # 束在同一扫描点停留期间光斑不变：命中缓存则整幅复制，免去重画
        if (
            self._flux_cache is not None
            and self._flux_cache_idx == idx
            and self._flux_cache.shape == out.shape
        ):
            np.copyto(out, self._flux_cache)
            return out

        out.fill(0.0)
        pos = self.scan_path[idx]

        # 提取 1D 坐标供给 Numba 寻址
        X_row = X[0, :]
        Y_col = Y[:, 0]

        # 调用局部化加速核函数
        _compute_flux_map_local_numba(
            X_row, Y_col, pos[0], pos[1], self._gauss_f0, self._gauss_sig, out
        )

        if self._flux_cache is None or self._flux_cache.shape != out.shape:
            self._flux_cache = np.empty_like(out)
        np.copyto(self._flux_cache, out)
        self._flux_cache_idx = idx

        return out
