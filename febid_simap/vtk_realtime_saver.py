#!/usr/bin/env python3
"""
VTK Real-time Saver - 适配极简 Simulator
"""

import os
import numpy as np
from datetime import datetime


class VTKRealtimeSaver:

    def __init__(self, output_dir: str, save_interval: int = None):
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.enabled = save_interval is not None and save_interval > 0
        self.saved_count = 0
        self.last_saved_point = -999

        if self.enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(output_dir, f"vtk_{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"\n{'=' * 70}")
            print(f"VTK Real-time Saver: ENABLED")
            print(f"{'=' * 70}")
            print(f"  Save interval: Every {save_interval} scan points")
            print(f"  Output directory: {os.path.abspath(self.output_dir)}")
            print(f"{'=' * 70}\n")

    def should_save(self, point_idx: int) -> bool:
        if not self.enabled:
            return False
        if point_idx == 0:
            return True
        if point_idx - self.last_saved_point >= self.save_interval:
            return True
        return False

    def _get_species_data(self, simulator, name):
        """从模拟器中安全提取物种数据（通用版本）

        Args:
            simulator: 模拟器对象
            name: 物种名称（如 'F', 'XeF2', 'OH'）

        Returns:
            浓度场数组或零数组（如果物种不存在）
        """
        # ✅ 通过模型接口查找物种索引
        if name in simulator.model.species_names:
            species_idx = simulator.model.species_names.index(name)
            return simulator.state_field[species_idx].copy()

        # 如果当前系统没有这个物种，返回全 0 矩阵
        return np.zeros((simulator.ny, simulator.nx))

    def save(self, point_idx: int, time_s: float, simulator):
        """保存当前状态到 VTK 文件（自动包含所有物种）"""
        if not self.enabled:
            return

        try:
            from .external_interface import export_results_to_vtk

            filename = f"scan_point_{point_idx:06d}.vtk"
            filepath = os.path.join(self.output_dir, filename)

            h_current = simulator.h_material.copy()

            # 计算刻蚀深度
            if getattr(simulator, "use_custom_surface", False):
                etch_depth = simulator.initial_height_field - h_current
            else:
                base_h = getattr(
                    simulator, "h_initial_scalar", simulator.params.h_initial
                )
                etch_depth = base_h - h_current

            # ✅ 自动保存所有物种浓度（根据模型动态生成）
            scalar_fields = {"etch_depth": etch_depth}

            for species_name in simulator.model.species_names:
                scalar_fields[f"{species_name}_concentration"] = self._get_species_data(
                    simulator, species_name
                )

            export_results_to_vtk(
                filepath, simulator.X, simulator.Y, h_current, scalar_fields
            )

            self.last_saved_point = point_idx
            self.saved_count += 1
            print(
                f"  [VTK] Saved #{self.saved_count}: {filename} (point {point_idx}, t={time_s * 1e6:.2f} us)"
            )

        except Exception as e:
            print(f"  [VTK ERROR] Failed to save point {point_idx}: {e}")

    def save_final(self, time_s: float, simulator):
        """保存最终状态（自动包含所有物种）"""
        if not self.enabled:
            return
        try:
            from .external_interface import export_results_to_vtk

            filename = "scan_FINAL.vtk"
            filepath = os.path.join(self.output_dir, filename)

            h_current = simulator.h_material.copy()
            if getattr(simulator, "use_custom_surface", False):
                etch_depth = simulator.initial_height_field - h_current
            else:
                base_h = getattr(
                    simulator, "h_initial_scalar", simulator.params.h_initial
                )
                etch_depth = base_h - h_current

            # ✅ 自动保存所有物种浓度
            scalar_fields = {"etch_depth": etch_depth}

            for species_name in simulator.model.species_names:
                scalar_fields[f"{species_name}_concentration"] = self._get_species_data(
                    simulator, species_name
                )

            export_results_to_vtk(
                filepath, simulator.X, simulator.Y, h_current, scalar_fields
            )
            print(f"  [VTK] Saved FINAL: {filename} (t={time_s * 1e6:.2f} us)")
        except Exception as e:
            print(f"  [VTK ERROR] Failed to save final: {e}")

    def finalize(self, total_points: int = None, total_time: float = None):
        if not self.enabled:
            return
        print(
            f"\n  [VTK] 序列已保存完成！共生成 {self.saved_count} 个文件于目录: {self.output_dir}"
        )
