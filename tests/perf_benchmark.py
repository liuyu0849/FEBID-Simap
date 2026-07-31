#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准与优化分析（刻蚀 + 沉积）
=================================

依照项目开发规范第 5 条编排：以最原始的可运行基线为参照，逐项列出各计算步骤
耗时，对比当前性能，列出前三项性能卡点与优化目标。

- 参照基线：项目首个可运行版本（commit 9a55382）。自该版本起，febid_simap 的
  计算代码未再改动；本次唯一改写到计算路径的是“扩散步”（由 numpy 路线改为
  numba 融合内核），故对该步做 head-to-head 对比以量化提升。
- 逐步骤耗时：在 150×150 网格上分别测量“通量图 / 反应步 / 扩散步”的每步耗时。
- 同时覆盖刻蚀（PSM_ETCH）与沉积（PSM_DEPO）。

运行方式：
    cd /home/user/FEBID-Simap
    python tests/perf_benchmark.py
"""

import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 便于 import consistency_test

from febid_simap.Scan_Pattern import ScanPattern2D, Scanning2DBeam
from febid_simap.Simulator import Scanning2DFEBIPSimulator, apply_diffusion_kernel
from consistency_test import (
    SYSTEMS, _build_raster_scan, _BeamConfig, GRID_N, GRID_SIZE, ACT_REGION,
)

REPORT_PATH = os.path.join(_REPO_ROOT, "tests", "report_performance.md")
BASELINE_COMMIT = "9a55382"

perf = time.perf_counter


def build_sim(sys_key: str):
    cfg = SYSTEMS[sys_key]
    params = cfg["params_cls"]()
    params.dt = cfg["dt"]
    pos, dw = _build_raster_scan()
    pat = ScanPattern2D.from_external_data(pos, GRID_SIZE, _BeamConfig(*cfg["sigma"]))
    beam = Scanning2DBeam(cfg["beam"], pat, dwell_time_array=dw)
    sim = Scanning2DFEBIPSimulator(
        params, beam, GRID_SIZE, GRID_N, GRID_N, etch_region=ACT_REGION
    )
    return sim, beam, params


def bench_pipeline(sys_key: str, n_warm: int = 5, n_timed: int = 300):
    """在固定（激活）束流位置下，分阶段测量每步耗时。"""
    sim, beam, params = build_sim(sys_key)
    dt = params.dt
    model = sim.model
    mask = sim.etch_region_mask
    X, Y, buf, lap = sim.X, sim.Y, sim._flux_buffer, sim._lap_buffer
    dx, dy, Darr = sim.dx, sim.dy, sim.D_array
    t_active = beam.total_scan_time * 0.5  # 束流处于激活状态的固定时刻

    def one_step():
        flux = beam.get_flux_map(t_active, X, Y, out=buf)
        sim.state_field, sim.h_material = model.apply_reaction_step(
            sim.state_field, sim.h_material, flux, dt, mask
        )
        for i in range(model.num_species):
            if Darr[i] > 0:
                apply_diffusion_kernel(sim.state_field[i], lap, Darr[i] * dt, dx, dy, mask)

    # 预热：触发全部 numba 编译
    for _ in range(n_warm):
        one_step()

    acc = {"flux": 0.0, "react": 0.0, "diff": 0.0}
    for _ in range(n_timed):
        a = perf(); flux = beam.get_flux_map(t_active, X, Y, out=buf); acc["flux"] += perf() - a
        a = perf()
        sim.state_field, sim.h_material = model.apply_reaction_step(
            sim.state_field, sim.h_material, flux, dt, mask
        )
        acc["react"] += perf() - a
        a = perf()
        for i in range(model.num_species):
            if Darr[i] > 0:
                apply_diffusion_kernel(sim.state_field[i], lap, Darr[i] * dt, dx, dy, mask)
        acc["diff"] += perf() - a

    for k in acc:
        acc[k] = acc[k] / n_timed * 1e6  # 每步微秒
    acc["total"] = acc["flux"] + acc["react"] + acc["diff"]
    return acc


def numpy_diffusion(field, D_dt, dx, dy):
    """原始代码采用的 numpy 路线（此处为其数值正确版本，用于公平对比）。"""
    p = np.pad(field, 1, mode="edge")
    lap = (p[2:, 1:-1] + p[:-2, 1:-1] - 2.0 * field) / (dy * dy) + (
        p[1:-1, 2:] + p[1:-1, :-2] - 2.0 * field
    ) / (dx * dx)
    out = field + D_dt * lap
    np.maximum(out, 0.0, out=out)
    return out


def bench_diffusion_step(n: int = 800):
    """扩散步 head-to-head：numba 融合内核 vs numpy 向量化路线。"""
    rng = np.random.default_rng(1)
    fld = rng.random((GRID_N, GRID_N))
    lap = np.zeros((GRID_N, GRID_N))
    mask = np.ones((GRID_N, GRID_N), dtype=bool)

    apply_diffusion_kernel(fld.copy(), lap, 0.1, 1.0, 1.0, mask)  # warmup
    a = perf()
    for _ in range(n):
        f = fld.copy()
        apply_diffusion_kernel(f, lap, 0.1, 1.0, 1.0, mask)
    t_numba = (perf() - a) / n * 1e6

    a = perf()
    for _ in range(n):
        numpy_diffusion(fld, 0.1, 1.0, 1.0)
    t_numpy = (perf() - a) / n * 1e6
    return t_numba, t_numpy


def main():
    print("=" * 70)
    print("性能基准 — 150x150 网格 / 刻蚀 + 沉积")
    print("=" * 70)

    results = {k: bench_pipeline(k) for k in SYSTEMS}
    t_numba, t_numpy = bench_diffusion_step()

    for k, acc in results.items():
        print(f"\n[{k} · {SYSTEMS[k]['kind']}] 每步 {acc['total']:.1f} us"
              f"  (通量 {acc['flux']:.1f} / 反应 {acc['react']:.1f} / 扩散 {acc['diff']:.1f})")
    print(f"\n扩散步单核对比：numba {t_numba:.2f} us vs numpy {t_numpy:.2f} us"
          f"  → 提升 {t_numpy / t_numba:.2f}×")

    write_report(results, t_numba, t_numpy)
    print(f"\n报告已写入：{REPORT_PATH}")
    return 0


def write_report(results, t_numba, t_numpy):
    # 跨体系平均的阶段占比，用于排名
    stages = ["react", "diff", "flux"]
    label = {"react": "反应步（逐点 RK4）", "diff": "扩散步（显式欧拉 + 拉普拉斯）",
             "flux": "通量图（高斯光斑）"}
    avg = {s: np.mean([results[k][s] / results[k]["total"] for k in results]) for s in stages}
    ranking = sorted(stages, key=lambda s: avg[s], reverse=True)

    L = []
    L.append("# 性能基准与优化分析（刻蚀 + 沉积）")
    L.append("")
    L.append(f"参照基线：项目首个可运行版本（commit `{BASELINE_COMMIT}`）。"
             "自该版本起 `febid_simap` 的计算代码未再改动，因此“反应步 / 通量图 / 快照”"
             "与原版**逐字一致（提升 0%）**；本次唯一改写到计算路径的是**扩散步**，"
             "下文对其单独做 head-to-head 对比。")
    L.append("")
    L.append("## 测量设置")
    L.append("")
    L.append("| 项目 | 设置 |")
    L.append("| --- | --- |")
    L.append(f"| 网格 | {GRID_N} × {GRID_N} |")
    L.append("| 束流 | 固定于激活位置，逐阶段计时 |")
    L.append("| 预热 | 先触发全部 JIT 编译，再计时 |")
    L.append("| 体系 | PSM_ETCH（刻蚀）、PSM_DEPO（沉积） |")
    L.append("")
    L.append("## 各计算步骤每步耗时")
    L.append("")
    L.append("| 步骤 | PSM_ETCH (us) | 占比 | PSM_DEPO (us) | 占比 |")
    L.append("| --- | --- | --- | --- | --- |")
    for s in stages:
        e, d = results["PSM_ETCH"], results["PSM_DEPO"]
        L.append(f"| {label[s]} | {e[s]:.2f} | {e[s]/e['total']:.0%} | "
                 f"{d[s]:.2f} | {d[s]/d['total']:.0%} |")
    e, d = results["PSM_ETCH"], results["PSM_DEPO"]
    L.append(f"| **合计（每步）** | **{e['total']:.2f}** | 100% | **{d['total']:.2f}** | 100% |")
    L.append("")
    L.append("## 扩散步：对比原版提升")
    L.append("")
    L.append(f"- 原版 numpy 向量化路线：**{t_numpy:.2f} us / 次**")
    L.append(f"- 当前 numba 融合内核：**{t_numba:.2f} us / 次**")
    L.append(f"- **提升：约 {t_numpy / t_numba:.2f}×**（且原版扩散实为空操作、数值错误，"
             "当前内核在更快的同时才是正确的）")
    L.append("")
    L.append("## 当前性能卡点前三项")
    L.append("")
    names = {"react": "① 反应步", "diff": "② 扩散步", "flux": "③ 通量图"}
    for rank, s in enumerate(ranking, 1):
        L.append(f"{rank}. **{label[s]}** — 跨体系平均占每步耗时约 {avg[s]:.0%}。")
    L.append("")
    L.append("> 此外还有一个**隐藏在‘步数’里的头号卡点**：真实工艺时长需要极多步。"
             "反应的强刚性与显式扩散的稳定性上限都逼迫时间步 dt 取得很小，"
             "于是一次完整仿真要走成百上千万步——**单步再快，也架不住步数太多**。")
    L.append("")
    L.append("## 优化目标（讲逻辑，不讲代码）")
    L.append("")
    L.append("**1. 反应步：别在‘风平浪静’的地方反复劳作。**  ")
    L.append("RK4 每前进一步要做四次试探性求值，好比走一步路先派四名侦察兵四处探路"
             "再决定落脚——精度高，但开销也大。而网格上绝大多数格点要么没有束流照射、"
             "要么早已到达稳态，风平浪静却仍被反复精算。目标：识别这些“安静”的格点，"
             "对它们跳过或降阶求解，把算力集中到束斑附近真正在剧烈变化的少数格点上——"
             "如同只在着火的房间派消防队，而不是全楼挨个泼水。")
    L.append("")
    L.append("**2. 扩散步：让它敢迈大步。**  ")
    L.append("当前显式扩散像每一步只允许热量传给紧挨着的邻居，稍微迈大一点就会“摔跤”"
             "（数值发散），所以时间步被死死限制。改用隐式 / ADI 格式相当于允许一步"
             "把影响稳稳传到更远处而不失稳——单步略贵，却能用大得多的步长，"
             "**用更少的步数走完同样的路**。这尤其能解开 COG 体系当前迈不开步的死结。")
    L.append("")
    L.append("**3. 通量图：同一枚印章，何必每帧重刻。**  ")
    L.append("束斑的形状始终不变，变的只是它落在哪里；现在却每一步都把这枚“高斯印章”"
             "从头重画一遍。目标：把印章预先刻好（预计算模板），之后只做平移贴图，"
             "省去重复的指数运算。")
    L.append("")
    L.append("**总纲：真正的大头是‘步数’而非‘单步’。**  ")
    L.append("最有价值的方向，是让每一步都能迈得更大（隐式扩散、对刚性反应做更稳的积分），"
             "从而把总步数显著压下来——这比把单步再抠快几个百分点更能决定整体墙钟时间。")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
