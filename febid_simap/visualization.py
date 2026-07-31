#!/usr/bin/env python3
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无显示环境下也可保存图像
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D


def visualize_profile(simulator, snapshots, scan_pattern, plot_config, output_path="."):
    """
    高级可视化函数：支持正负高度混合、光影渲染及自适应布局
    """
    print("\n[Visualization] 正在生成 3D 渲染形貌图...")

    # 1. 数据准备：计算相对初始平面的高度差 delta_H
    # z_data > 0 代表沉积，z_data < 0 代表刻蚀，0 代表初始表面
    z_data = snapshots["h_material"][-1] - simulator.params.h_initial

    X, Y = simulator.X, simulator.Y
    z_min, z_max = np.min(z_data), np.max(z_data)
    z_range = max(z_max - z_min, 1e-12)

    # 2. 颜色与归一化逻辑 (针对正负值自适应)
    # 使用 RdBu_r (红蓝) 这种发散色系：蓝色代表沉积(正)，红色代表刻蚀(负)，白色为0
    if z_min < -1e-9 and z_max > 1e-9:
        # 数据跨越了 0 点
        cmap_3d = "RdBu_r"
        norm = TwoSlopeNorm(vmin=z_min, vcenter=0, vmax=z_max)
    elif z_max <= 1e-9:
        # 纯刻蚀或平整
        cmap_3d = "coolwarm_r"
        norm = Normalize(vmin=z_min, vmax=max(z_max, 0))
    else:
        # 纯沉积
        cmap_3d = "viridis"
        norm = Normalize(vmin=min(z_min, 0), vmax=z_max)

    # 3. 3D 光影渲染配置 (解决“全是一个颜色”的问题)
    ls = LightSource(azdeg=315, altdeg=45)  # 设置光源角度
    # 计算光照阴影后的颜色映射
    rgb = ls.shade(
        z_data, cmap=plt.get_cmap(cmap_3d), norm=norm, vert_exag=1, blend_mode="soft"
    )

    # 4. 坐标轴比例自适应 (防止微米级平面在视觉上太扁)
    x_lim = (np.min(X), np.max(X))
    y_lim = (np.min(Y), np.max(Y))
    xy_span = max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])
    # 动态拉伸 Z 轴比例，确保起伏清晰可见 (范围 0.3 到 1.2)
    z_aspect = np.clip(0.6 * (xy_span / z_range), 0.3, 1.2)

    # 5. 创建画布
    fig = plt.figure(figsize=(22, 14), constrained_layout=True)

    # 6. 绘制 4 个视角的 3D 子图
    angles = [
        (30, 45, "Northeast"),
        (30, 135, "Southeast"),
        (60, 225, "Southwest"),
        (60, 315, "Northwest"),
    ]
    for idx, (elev, azim, name) in enumerate(angles):
        pos = idx + 1 if idx < 2 else idx + 2  # 跳过中间位置留给 2D 图
        ax = fig.add_subplot(2, 3, pos, projection="3d")

        # 使用 facecolors 参数传入渲染后的 RGB 数据
        surf = ax.plot_surface(
            X,
            Y,
            z_data,
            rcount=100,
            ccount=100,
            facecolors=rgb,
            antialiased=True,
            shade=False,
        )

        ax.set(xlabel="X [nm]", ylabel="Y [nm]", zlabel="Delta H [nm]")
        ax.set_title(
            f"{name} View\n(elev={elev}, azim={azim})", fontsize=12, fontweight="bold"
        )
        ax.view_init(elev, azim)

        try:
            ax.set_box_aspect([1, 1, z_aspect])
        except:
            pass

        # 在第一个子图旁添加 Colorbar
        if idx == 0:
            # 手动创建与 norm 匹配的 scalar mappable
            sm = plt.cm.ScalarMappable(cmap=cmap_3d, norm=norm)
            cb = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=15, pad=0.1)
            cb.set_label("Height Change [nm]", fontsize=10)

    # 7. 2D 等高线图 (Top View)
    ax_2d = fig.add_subplot(2, 3, 3)
    levels = np.linspace(z_min, z_max, 30) if z_range > 1e-12 else 10
    cont = ax_2d.contourf(X, Y, z_data, levels=levels, cmap=cmap_3d, norm=norm)

    # 绘制扫描起点
    if len(scan_pattern.scan_path) > 0:
        ax_2d.plot(
            scan_pattern.scan_path[0][0],
            scan_pattern.scan_path[0][1],
            "go",
            ms=10,
            mew=2,
            mec="white",
            label="Start",
        )
        ax_2d.legend(loc="upper right", fontsize=8)

    ax_2d.set(xlabel="X [nm]", ylabel="Y [nm]", title="2D Top View", aspect="equal")
    fig.colorbar(cont, ax=ax_2d, label="Delta H [nm]")

    # 8. 截面图 (Cross-sections)
    ax_cross = fig.add_subplot(2, 3, 6)
    cy, cx = simulator.ny // 2, simulator.nx // 2

    ax_cross.plot(simulator.x, z_data[cy, :], "b-", lw=2, label="X-Profile (ctr Y)")
    ax_cross.plot(simulator.y, z_data[:, cx], "r-", lw=2, label="Y-Profile (ctr X)")
    ax_cross.fill_between(simulator.x, 0, z_data[cy, :], color="blue", alpha=0.1)
    ax_cross.axhline(0, color="black", ls="--", lw=1.0, alpha=0.5)

    ax_cross.set(
        xlabel="Position [nm]", ylabel="Delta H [nm]", title="Cross-section Profiles"
    )
    ax_cross.grid(True, linestyle=":", alpha=0.6)
    ax_cross.legend(fontsize=9)

    # 9. 保存与输出
    process_info = simulator.model.process_type.capitalize()
    fig.suptitle(
        f"{process_info} Profile - {simulator.model.system_type}\n"
        f"Max Change: {z_max:.2f} nm | Min Change: {z_min:.2f} nm",
        fontsize=18,
        fontweight="bold",
    )

    save_path = os.path.join(output_path, f"profile_{simulator.model.process_type}.png")
    plt.savefig(save_path, dpi=getattr(plot_config, "DPI", 300), bbox_inches="tight")
    print(f"  ✓ 图像已保存至: {save_path}")
    plt.close()
