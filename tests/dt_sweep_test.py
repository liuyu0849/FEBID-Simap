#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动步长参数扫描与收敛性测试
==============================

目的：验证自动步长（dt_mode="auto"）不只在标准参数下成立，而是在参数空间内成立。

- 阶段一（生效性与数值健康）：在 D、τ、σ 三个轴上按 logspace 各取 3 点，共 27 组，
  施加在前驱体物种上（PSM_DEPO 的 Cr(CO)6、PSM_ETCH 的 XeF2），中间产物保持默认参数。
  每组以自动步长运行一次，记录束开/束关步长、步数、扩散路径与健康指标，并核验实际
  步长不超过速率上界给出的允许值（自动步长确实生效）。
- 阶段二（收敛性）：对每组参数把步长细化 2 倍与 3 倍（安全系数 C 取 0.5、0.25、0.167），
  比较 1×、2×、3× 三个解的高度变化场与覆盖度场，要求 e12 与 e23 均 ≤ 1%。
  非负截断会掩盖不稳定，只有细化对比才能揭示，故以此作为决定性判据。
- 附：标准参数下自动模式的 D > 0 与 D = 0 两种情形，按开发规范第 3 条覆盖。

测试网格：150 × 150（dx = dy = 1 nm），中央 100 × 100 作用区，10 × 10 栅格扫描 100 点，
每点驻留 0.1 us；另加一段刷新等待期（frt = 20 us，即扫描 10 us + 等待 10 us），
用于覆盖束关闭时自动步长放大最多的区间。

运行方式：
    cd /home/user/FEBID-Simap
    python tests/dt_sweep_test.py
"""

import contextlib
import io
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from febid_simap.physics_params import PSMDepoConfig, PSMEtchConfig
from febid_simap.Scan_Pattern import ScanPattern2D, Scanning2DBeam
from febid_simap.Simulator import Scanning2DFEBIPSimulator

# ================================================================================
# 测试设置
# ================================================================================
GRID_N = 150
GRID_SIZE = float(GRID_N - 1)          # dx = dy = 1.0 nm
REGION_HALF = 50.0
ACT_REGION = ((-REGION_HALF, REGION_HALF), (-REGION_HALF, REGION_HALF))
DWELL = 1.0e-7                          # 每点驻留 0.1 us
FRT = 2.0e-5                            # 刷新周期：扫描 10 us + 等待 10 us
DT_FIXED = 0.4e-7                       # 固定步长参考值（用于对照列）

DT_FACTOR = 0.5                         # 自动步长安全系数 C
R_MAX = 2.5                             # 扩散步精度上限 r_max = D·dt/dx²
REFINE = (1, 2, 3)                      # 细化倍数：C / k 与 r_max / k 同步缩小
TOL = 0.01                              # 收敛阈值 1%
TIGHT_FACTOR = 0.15                     # 未收敛组的复验安全系数

SWEEP_D = np.logspace(6, 7, 3)                        # 1e6, 3.16e6, 1e7 nm²/s
SWEEP_TAU = np.logspace(-7, -5, 3)                    # 1e-7, 1e-6, 1e-5 s
SWEEP_SIGMA = np.logspace(np.log10(0.1), np.log10(0.5), 3)  # 0.1, 0.224, 0.5 nm²

MODELS = {
    "PSM_DEPO": {
        "kind": "沉积",
        "params_cls": PSMDepoConfig,
        "precursor": "Cr(CO)6",
        "D_attr": "D_CrCO6", "tau_attr": "tau_CrCO6", "sigma_attr": "sigma_CrCO6",
        "D_attrs_all": ["D_CrCO6", "D_CO"],
        "beam": [(2.98e7, 10), (0.00e7, 20), (0e7, 24)],
        "sigma_beam": (4.0, 4.0),
        "species": ["CrCO6", "CO"],
        "h_sign": +1,                     # 沉积：高度只增
    },
    "PSM_ETCH": {
        "kind": "刻蚀",
        "params_cls": PSMEtchConfig,
        "precursor": "XeF2",
        "D_attr": "D_XeF2", "tau_attr": "tau_XeF2", "sigma_attr": "sigma_XeF2",
        "D_attrs_all": ["D_XeF2", "D_F"],
        "beam": [(2.98e7, 10), (0.00e7, 20), (0e7, 24)],
        "sigma_beam": (2.5, 4.0),
        "species": ["XeF2", "F"],
        "h_sign": -1,                     # 刻蚀：高度只减
    },
}

REPORT_PATH = os.path.join(_REPO_ROOT, "tests", "report_dt_sweep.md")
RESULT_PATH = os.path.join(_REPO_ROOT, "tests", "reference", "dt_sweep_results.json")


class _BeamConfig:
    def __init__(self, sigma_min, sigma_max):
        self.SIGMA_MIN = sigma_min
        self.SIGMA_MAX = sigma_max


def _build_raster_scan():
    coords = np.arange(-45.0, 46.0, 10.0)
    positions = np.array([[x, y] for y in coords for x in coords], dtype=float)
    dwell_times = np.full(len(positions), DWELL)
    return positions, dwell_times


class _StepRecorder:
    """逐步记录：时刻、是否等待段、各物种极值与总量、高度极值与总量、步后速率上界。"""

    def __init__(self, factor):
        self.factor = factor
        self.t, self.wait, self.bound_after = [], [], []
        self.n_min, self.n_max, self.n_sum = [], [], []
        self.h_min, self.h_max, self.h_sum = [], [], []

    def initialize(self, sim):
        self.model = sim.model

    def record(self, t, sim, idx):
        self.t.append(t)
        self.wait.append(bool(sim.beam._time_to_point_index(t)[1]))
        st = sim.state_field
        self.n_min.append(st.min(axis=(1, 2)).tolist())
        self.n_max.append(st.max(axis=(1, 2)).tolist())
        self.n_sum.append(st.sum(axis=(1, 2)).tolist())
        h = sim.h_material
        self.h_min.append(float(h.min()))
        self.h_max.append(float(h.max()))
        self.h_sum.append(float(h.sum()))
        # 步后状态下的速率上界：下一步（若仍在同一段内）所取 dt 不得超过 C / B
        B = self.model.get_reaction_rate_bound(float(sim._flux_buffer.max()), st)
        self.bound_after.append(self.factor / B if B else np.inf)


def make_params(model_key, D=None, tau=None, sigma=None, d_scale=1.0):
    cfg = MODELS[model_key]
    p = cfg["params_cls"]()
    p.dt = DT_FIXED
    if D is not None:
        setattr(p, cfg["D_attr"], float(D))
    if tau is not None:
        setattr(p, cfg["tau_attr"], float(tau))
    if sigma is not None:
        setattr(p, cfg["sigma_attr"], float(sigma))
    for attr in cfg["D_attrs_all"]:
        setattr(p, attr, getattr(p, attr) * d_scale)
    p.__post_init__()   # 重算 Φ、1/τ 等派生量
    return p


def run_case(model_key, params, factor, r_max=R_MAX):
    """以自动步长运行一次完整仿真（含刷新等待段），返回场与逐步记录。"""
    cfg = MODELS[model_key]
    pos, dw = _build_raster_scan()
    pattern = ScanPattern2D.from_external_data(pos, GRID_SIZE, _BeamConfig(*cfg["sigma_beam"]))
    beam = Scanning2DBeam(cfg["beam"], pattern, dwell_time_array=dw, frt=FRT)
    sim = Scanning2DFEBIPSimulator(params, beam, GRID_SIZE, GRID_N, GRID_N, etch_region=ACT_REGION)
    rec = _StepRecorder(factor)
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sim.run_scanning(dt=params.dt, save_interval=10 ** 12, concentration_analyzer=rec,
                         dt_mode="auto", dt_rxn_factor=factor, dt_diff_r_max=r_max)
    wall = time.time() - t0

    t = np.array(rec.t)
    dts = np.diff(np.append(t, beam.total_scan_time))
    wait = np.array(rec.wait)
    bound_after = np.array(rec.bound_after)
    # 生效性核验：同一段内相邻两步，后一步 dt ≤ 前一步步后状态给出的上界（允许 1e-9 相对余量）
    same_seg = wait[1:] == wait[:-1]
    viol = int(np.sum(same_seg & (dts[1:] > bound_after[:-1] * (1 + 1e-9))))

    n_min = np.array(rec.n_min); n_max = np.array(rec.n_max); n_sum = np.array(rec.n_sum)
    h_sum = np.array(rec.h_sum)
    dh_sum = np.diff(h_sum) * cfg["h_sign"]
    h_mono_viol = int(np.sum(dh_sum < -1e-12 * max(abs(h_sum).max(), 1.0)))
    # 前驱体总量序列的“增幅翻转”计数（信息项）：相邻差分变号且幅度增大
    d = np.diff(n_sum[:, 0])
    flips = int(np.sum((d[1:] * d[:-1] < 0) & (np.abs(d[1:]) > np.abs(d[:-1]))))

    f_on = float(sim.beam._gauss_f0.sum())
    return {
        "h": sim.h_material.copy(),
        "dh": sim.h_material - params.h_initial,
        "state": sim.state_field.copy(),
        "steps": int(len(t)),
        "wall": wall,
        "dt_on_med": float(np.median(dts[~wait])) if np.any(~wait) else float("nan"),
        "dt_on_min": float(dts[~wait].min()) if np.any(~wait) else float("nan"),
        "dt_off_med": float(np.median(dts[wait])) if np.any(wait) else float("nan"),
        "n_off": int(wait.sum()),
        "bound_viol": viol,
        "nan": int(np.isnan(sim.h_material).sum() + np.isnan(sim.state_field).sum()
                   + np.isnan(n_sum).sum()),
        "n_min": float(n_min.min()),
        "n_max": float(n_max.max()),
        "h_min": float(min(rec.h_min)),
        "h_mono_viol": h_mono_viol,
        "flips": flips,
        "B_on": float(sim.model.get_reaction_rate_bound(f_on, sim.state_field)),
        "B_off": float(sim.model.get_reaction_rate_bound(0.0, sim.state_field)),
        "n_sites": float(params.n_sites),
    }


def rel_err(a, b):
    denom = max(float(np.abs(b).max()), 1e-30)
    return float(np.abs(a - b).max() / denom)


def evaluate(model_key, params, label, factor=DT_FACTOR):
    """跑 1×/2×/3× 三个细化级别（C 与 r_max 同步缩小 k 倍），返回一行汇总结果。"""
    cfg = MODELS[model_key]
    runs = {k: run_case(model_key, params, factor / k, R_MAX / k) for k in REFINE}
    r1, r2, r3 = runs[1], runs[2], runs[3]
    e12_h = rel_err(r1["dh"], r2["dh"]); e23_h = rel_err(r2["dh"], r3["dh"])
    e13_h = rel_err(r1["dh"], r3["dh"])
    e12_n = rel_err(r1["state"], r2["state"]); e23_n = rel_err(r2["state"], r3["state"])
    e13_n = rel_err(r1["state"], r3["state"])
    dh_measurable = float(np.abs(r1["dh"]).max()) > 1e-12   # Δh 低于双精度分辨率则 h 判据退化

    D = getattr(params, cfg["D_attr"])
    dx2 = 1.0
    health = {
        "nan": r1["nan"] == 0,
        "range": (r1["n_min"] >= 0.0) and (r1["n_max"] <= r1["n_sites"] * (1 + 1e-9)),
        "h_mono": r1["h_mono_viol"] == 0 and (cfg["h_sign"] > 0 or r1["h_min"] >= -1e-12),
        "bound": r1["bound_viol"] == 0,
    }
    conv = (e12_h <= TOL) and (e23_h <= TOL) and (e12_n <= TOL) and (e23_n <= TOL)
    mono = (e23_h <= e12_h or e12_h < 1e-4) and (e23_n <= e12_n or e12_n < 1e-4)
    row = {
        "label": label, "factor": factor,
        "D": float(D), "tau": float(getattr(params, cfg["tau_attr"])),
        "sigma": float(getattr(params, cfg["sigma_attr"])),
        "B_on": r1["B_on"], "B_off": r1["B_off"],
        "dt_on": r1["dt_on_med"], "dt_on_min": r1["dt_on_min"], "dt_off": r1["dt_off_med"],
        "steps": [r1["steps"], r2["steps"], r3["steps"]],
        "wall": [r1["wall"], r2["wall"], r3["wall"]],
        "z_fixed": r1["B_on"] * DT_FIXED,
        "r_on": D * r1["dt_on_med"] / dx2, "r_off": D * r1["dt_off_med"] / dx2,
        "nan": r1["nan"], "n_min": r1["n_min"], "n_max": r1["n_max"], "n_sites": r1["n_sites"],
        "h_mono_viol": r1["h_mono_viol"], "flips": r1["flips"], "bound_viol": r1["bound_viol"],
        "dh_max": float(np.abs(r1["dh"]).max()),
        "health": health, "healthy": all(health.values()),
        "e12_h": e12_h, "e23_h": e23_h, "e13_h": e13_h,
        "e12_n": e12_n, "e23_n": e23_n, "e13_n": e13_n,
        "dh_measurable": bool(dh_measurable),
        "converged": bool(conv), "monotone": bool(mono),
    }
    return row


def main():
    print("=" * 70)
    print("自动步长参数扫描与收敛性测试 — 150x150 / 中央 100x100 / 沉积 + 刻蚀")
    print("=" * 70)
    t_start = time.time()
    results = {}
    for mk in MODELS:
        rows = []
        n_total = len(SWEEP_D) * len(SWEEP_TAU) * len(SWEEP_SIGMA)
        i = 0
        for D in SWEEP_D:
            for tau in SWEEP_TAU:
                for sig in SWEEP_SIGMA:
                    i += 1
                    row = evaluate(mk, make_params(mk, D, tau, sig), f"{mk}#{i}")
                    rows.append(row)
                    print(f"[{mk} {i:2d}/{n_total}] D={D:.2e} tau={tau:.1e} sigma={sig:.3f} "
                          f"dt_on={row['dt_on']:.2e} dt_off={row['dt_off']:.2e} "
                          f"steps={row['steps']} e12_h={row['e12_h']:.2e} e23_h={row['e23_h']:.2e} "
                          f"{'健康' if row['healthy'] else '异常'}/{'收敛' if row['converged'] else '未收敛'}")
        tightened = []
        for row in rows:
            if row["converged"]:
                continue
            t_row = evaluate(mk, make_params(mk, row["D"], row["tau"], row["sigma"]),
                             row["label"] + "-tight", factor=TIGHT_FACTOR)
            tightened.append(t_row)
            print(f"[{mk} 复验 C={TIGHT_FACTOR}] D={row['D']:.2e} tau={row['tau']:.1e} sigma={row['sigma']:.3f} "
                  f"dt_on={t_row['dt_on']:.2e} steps={t_row['steps']} e12_h={t_row['e12_h']:.2e} "
                  f"e23_h={t_row['e23_h']:.2e} {'收敛' if t_row['converged'] else '未收敛'}")
        std = {
            "D>0": evaluate(mk, make_params(mk), f"{mk}-std-Don"),
            "D=0": evaluate(mk, make_params(mk, d_scale=0.0), f"{mk}-std-Doff"),
        }
        for k, row in std.items():
            print(f"[{mk} 标准 {k}] dt_on={row['dt_on']:.2e} dt_off={row['dt_off']:.2e} steps={row['steps']} "
                  f"e12_h={row['e12_h']:.2e} e23_h={row['e23_h']:.2e} "
                  f"{'健康' if row['healthy'] else '异常'}/{'收敛' if row['converged'] else '未收敛'}")
        results[mk] = {"sweep": rows, "standard": std, "tightened": tightened}

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    write_report(results, time.time() - t_start)

    all_healthy = all(r["healthy"] for mk in results
                      for r in results[mk]["sweep"] + list(results[mk]["standard"].values()))
    all_conv = all(r["converged"] for mk in results
                   for r in results[mk]["sweep"] + list(results[mk]["standard"].values()))
    tight_ok = all(r["converged"] for mk in results for r in results[mk]["tightened"])
    all_ok = all_healthy and (all_conv or tight_ok)
    print(f"\n健康：{'全部通过' if all_healthy else '存在异常'}；"
          f"收敛（C={DT_FACTOR}）：{'全部通过' if all_conv else '存在未收敛组'}"
          f"{'' if all_conv else f'；未收敛组以 C={TIGHT_FACTOR} 复验：' + ('全部收敛' if tight_ok else '仍有未收敛')}")
    print(f"\n总判定：{'全部通过' if all_ok else '存在未通过项'}")
    print(f"报告已写入：{REPORT_PATH}")
    return 0 if all_ok else 1


def _fmt_path(r):
    return "ADI" if r > 0.25 else "显式"


def write_report(results, elapsed):
    L = []
    L.append("# 自动步长参数扫描与收敛性测试报告")
    L.append("")
    L.append("## 测试设置")
    L.append("")
    L.append("| 项目 | 设置 |")
    L.append("| --- | --- |")
    L.append(f"| 网格 | {GRID_N} × {GRID_N}（dx = dy = 1.0 nm） |")
    L.append("| 作用区域 | 网格中央 100 × 100 nm |")
    L.append(f"| 扫描 | 10 × 10 栅格 100 点，每点驻留 {DWELL*1e6:.1f} us；刷新周期 frt = {FRT*1e6:.0f} us（含 {FRT*1e6-10:.0f} us 等待段） |")
    L.append("| 体系 | PSM_DEPO（沉积）、PSM_ETCH（刻蚀） |")
    L.append(f"| 扫描轴 | D = {', '.join(f'{v:.2e}' for v in SWEEP_D)} nm²/s；τ = {', '.join(f'{v:.0e}' for v in SWEEP_TAU)} s；σ = {', '.join(f'{v:.3f}' for v in SWEEP_SIGMA)} nm²（各轴 logspace 3 点，共 27 组） |")
    L.append("| 参数施加对象 | 前驱体物种（Cr(CO)6 / XeF2）；中间产物（CO / F）保持默认 |")
    L.append(f"| 自动步长 | dt = min(C / B_max, r_max·dx²/D_max) 并按驻留边界切齐；C = {DT_FACTOR}，r_max = {R_MAX}；扩散步一律 ADI；细化级别 k = 1、2、3 时 C 与 r_max 同步取 1/k |")
    L.append(f"| 收敛判据 | e12 ≤ {TOL:.0%} 且 e23 ≤ {TOL:.0%}（高度变化场与覆盖度场同时满足） |")
    L.append(f"| 固定步长对照 | dt = {DT_FIXED:.1e} s（仅用于计算 B·dt 对照列，不参与判定） |")
    L.append("")
    L.append("记号：B_on / B_off 为束开 / 束关时的反应速率上界；dt_on / dt_off 为对应段内实际步长中位数；"
             "r = D·dt/dx²（> 0.25 走 ADI 隐式路径，否则走显式路径）；e12 = 1× 对 2× 细化解的相对差，"
             "e23 = 2× 对 3× 的相对差，定义为 max|Δ| / max|参考|，h 用高度变化场 Δh = h − h₀，n 用覆盖度场。")
    L.append("")

    for mk, res in results.items():
        cfg = MODELS[mk]
        rows = res["sweep"]
        L.append(f"## {mk}（{cfg['kind']}，扫描参数施加于 {cfg['precursor']}）")
        L.append("")
        L.append("### 阶段一：自动步长生效性与数值健康")
        L.append("")
        L.append("| # | D | τ | σ | B_on (1/s) | dt_on (s) | dt_off (s) | 步数 | 扩散路径 开/关 | 固定dt的 B·dt | 覆盖度最大 | NaN | 高度单调 | 上界核验 | 健康 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for i, r in enumerate(rows, 1):
            L.append(f"| {i} | {r['D']:.2e} | {r['tau']:.0e} | {r['sigma']:.3f} | {r['B_on']:.2e} | "
                     f"{r['dt_on']:.2e} | {r['dt_off']:.2e} | {r['steps'][0]} | "
                     f"{_fmt_path(r['r_on'])}/{_fmt_path(r['r_off'])} | {r['z_fixed']:.2f} | "
                     f"{r['n_max']:.3f} | {r['nan']} | {'✅' if r['h_mono_viol']==0 else '❌'} | "
                     f"{'✅' if r['bound_viol']==0 else '❌'} | {'✅' if r['healthy'] else '❌'} |")
        L.append("")
        L.append("### 阶段二：步长细化收敛性")
        L.append("")
        L.append("| # | D | τ | σ | 步数 1×/2×/3× | e12 (Δh) | e23 (Δh) | e13 (Δh) | e12 (n) | e23 (n) | e13 (n) | 单调 | 收敛 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for i, r in enumerate(rows, 1):
            note = "" if r["dh_measurable"] else "（Δh 不可测）"
            L.append(f"| {i} | {r['D']:.2e} | {r['tau']:.0e} | {r['sigma']:.3f} | "
                     f"{r['steps'][0]}/{r['steps'][1]}/{r['steps'][2]} | "
                     f"{r['e12_h']:.2e} | {r['e23_h']:.2e} | {r['e13_h']:.2e}{note} | "
                     f"{r['e12_n']:.2e} | {r['e23_n']:.2e} | {r['e13_n']:.2e} | "
                     f"{'✅' if r['monotone'] else '⚠️'} | {'✅' if r['converged'] else '❌'} |")
        L.append("")
        n_ok_h = sum(r["healthy"] for r in rows); n_ok_c = sum(r["converged"] for r in rows)
        e12_max = max(r["e12_h"] for r in rows); e23_max = max(r["e23_h"] for r in rows)
        e12n_max = max(r["e12_n"] for r in rows)
        n_unmeas = sum(not r["dh_measurable"] for r in rows)
        L.append(f"小结：健康 {n_ok_h}/{len(rows)}，收敛 {n_ok_c}/{len(rows)}"
                 f"{f'（其中 {n_unmeas} 组 Δh 低于双精度分辨率，h 判据退化，仅覆盖度场有效）' if n_unmeas else ''}；"
                 f"e12(Δh) 最大 {e12_max:.2e}，e23(Δh) 最大 {e23_max:.2e}，e12(n) 最大 {e12n_max:.2e}；"
                 f"dt_on 范围 {min(r['dt_on'] for r in rows):.2e} ~ {max(r['dt_on'] for r in rows):.2e} s，"
                 f"dt_off 范围 {min(r['dt_off'] for r in rows):.2e} ~ {max(r['dt_off'] for r in rows):.2e} s。")
        L.append("")
        if res["tightened"]:
            L.append(f"### 未收敛组复验：安全系数收紧至 C = {TIGHT_FACTOR}")
            L.append("")
            L.append("| D | τ | σ | dt_on (s) | 步数 1×/2×/3× | e12 (Δh) | e23 (Δh) | e13 (Δh) | e12 (n) | e23 (n) | 收敛 |")
            L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in res["tightened"]:
                L.append(f"| {r['D']:.2e} | {r['tau']:.0e} | {r['sigma']:.3f} | {r['dt_on']:.2e} | "
                         f"{r['steps'][0]}/{r['steps'][1]}/{r['steps'][2]} | "
                         f"{r['e12_h']:.2e} | {r['e23_h']:.2e} | {r['e13_h']:.2e} | "
                         f"{r['e12_n']:.2e} | {r['e23_n']:.2e} | {'✅' if r['converged'] else '❌'} |")
            L.append("")
        L.append("### 附：标准参数下自动模式（D > 0 与 D = 0）")
        L.append("")
        L.append("| 情形 | dt_on (s) | dt_off (s) | 步数 1×/2×/3× | Δh 最大 (nm) | 覆盖度最大 | NaN | e12 (Δh) | e23 (Δh) | e13 (Δh) | e12 (n) | e23 (n) | e13 (n) | 健康 | 收敛 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for k, r in res["standard"].items():
            L.append(f"| {k} | {r['dt_on']:.2e} | {r['dt_off']:.2e} | {r['steps'][0]}/{r['steps'][1]}/{r['steps'][2]} | "
                     f"{r['dh_max']:.3e} | {r['n_max']:.3f} | {r['nan']} | {r['e12_h']:.2e} | {r['e23_h']:.2e} | {r['e13_h']:.2e} | "
                     f"{r['e12_n']:.2e} | {r['e23_n']:.2e} | {r['e13_n']:.2e} | {'✅' if r['healthy'] else '❌'} | {'✅' if r['converged'] else '❌'} |")
        L.append("")

    L.append("## 判据说明")
    L.append("")
    L.append("- **上界核验**：同一段（束开或等待）内相邻两步，后一步的 dt 不得超过前一步步后状态给出的 C / B_max。"
             "该项为 ✅ 说明步长确实由速率上界决定，且随参数（τ、σ）和状态（刻蚀体系的 n_F）自动收缩。")
    L.append("- **数值健康**：无 NaN/Inf；覆盖度落在 [0, n_sites]；沉积高度只增、刻蚀高度只减且不小于零。")
    L.append("- **收敛**：e12 与 e23 均 ≤ 1%（高度变化场与覆盖度场同时满足）；e13 为 1× 解相对最细解的直接偏差估计；"
             "“单调”列检查 e23 ≤ e12（差值低于 1e-4 时不作要求）。非负截断能把发散掩盖成“看似正常”的结果，"
             "这种情况在细化对比下会表现为 e12 大或 e23 不降，因此收敛项是决定性的判据。")
    L.append("")
    L.append(f"总耗时 {elapsed/60:.1f} min。")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
