import os
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def parse_gnu_file(filename, max_residue=16):
    residues = {}
    data = []

    print(f"正在读取文件: {filename}")

    with open(filename, 'r') as f:
        lines = f.readlines()
        print(f"文件行数: {len(lines)}")

        # 从 gnuplot 的 ytics 定义里提取“残基位置 -> 残基名称”的映射
        for line in lines:
            if line.startswith('set ytics('):
                matches = re.findall(r'"([^"]+)"\s+([\d.]+)', line)
                print(f"找到的残基标签: {len(matches)}个")
                for label, pos in matches:
                    if float(pos) <= max_residue:
                        residues[float(pos)] = label.split(':')[0]
                break

        data_start = False
        data_count = 0
        # 从 splot 后面的数据块中读取三列数值：时间/帧、残基位置、结构状态
        for line in lines:
            if 'splot "-"' in line:
                data_start = True
                continue
            if not data_start or line.strip() == 'e' or line.strip() == 'EOF':
                continue

            parts = line.split()
            if len(parts) == 3:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = int(float(parts[2]))
                    if y <= max_residue:
                        data.append([x, y, z])
                        data_count += 1
                except ValueError:
                    continue

    print(f"读取到的数据点数: {data_count}")
    print(f"找到的残基数: {len(residues)}")

    if not data:
        raise ValueError("没有读取到任何有效数据！")

    return residues, np.array(data).reshape(-1, 3)


def generate_plot(
    residues,
    data,
    output_file='dssp_plot.png',
    colors=('black', 'blue', 'green', 'red'),
    fig_size=(15, 10),
    dpi=150,
    point_size=1450,
    y_fontsize=10,
    x_fontsize=12,
    title_fontsize=14,
    cbar_fontsize=10,
    title='Protein Secondary Structure Dynamics (DSSP)',
    x_label='Frame',
    y_label='Residue',
    cbar_label='Secondary Structure',
    x_tick_step=2000,
    x_tick_labels=None,
    x_offset=0,
    colorbar_side='right',
    x_tick_shift=0,
):
    if len(data) == 0:
        raise ValueError("输入数据为空！")

    # DSSP 状态编码对应的颜色顺序：None、Beta、Alpha、Turn
    cmap = ListedColormap(list(colors))
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    norm = BoundaryNorm(bounds, cmap.N)
    cbar_colors = list(dict.fromkeys(colors))
    cbar_bounds = np.arange(len(cbar_colors) + 1)
    cbar_ticks = np.arange(len(cbar_colors)) + 0.5

    # 根据色条位置选择布局：右侧或左侧
    if colorbar_side == 'right':
        fig, (ax, cax) = plt.subplots(
            ncols=2,
            gridspec_kw={'width_ratios': [30, 1]},
            figsize=fig_size,
            dpi=dpi,
        )
    else:
        fig, (cax, ax) = plt.subplots(
            ncols=2,
            gridspec_kw={'width_ratios': [1, 30]},
            figsize=fig_size,
            dpi=dpi,
        )

    ax.scatter(
        data[:, 0],
        data[:, 1],
        c=data[:, 2],
        cmap=cmap,
        norm=norm,
        s=point_size,
        marker='s',
        linewidths=0,
        alpha=1.0,
    )

    y_ticks = sorted(residues.keys())
    y_labels = [residues[pos] for pos in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=y_fontsize)
    # ax.set_ylabel(y_label, fontsize=x_fontsize)

    max_frame = int(np.max(data[:, 0]))
    min_frame = int(np.min(data[:, 0]))
    ax.set_xticks(np.arange(0, max_frame + 1, x_tick_step))
    ax.set_xlabel(x_label, fontsize=x_fontsize)
    ax.grid(True, alpha=0.2, linestyle='--')

    # 按固定间隔生成横轴刻度；如果传入自定义标签且数量一致，就替换默认刻度文本
    first_tick = min_frame - (min_frame % x_tick_step)
    x_ticks = np.arange(first_tick + x_tick_shift, max_frame + x_tick_step + x_tick_shift, x_tick_step)
    ax.set_xticks(x_ticks)

    if x_tick_labels is not None and len(x_tick_labels) == len(x_ticks):
        ax.set_xticklabels(x_tick_labels, fontsize=x_fontsize)

    ax.set_xlim(min_frame - 0.5 + x_offset, max_frame + 0.5 + x_offset)

    y_min = min(data[:, 1]) - 0.5
    y_max = max(data[:, 1]) + 0.5
    ax.set_ylim(y_min, y_max)

    cbar = fig.colorbar(
        cm.ScalarMappable(
            cmap=ListedColormap(cbar_colors),
            norm=BoundaryNorm(cbar_bounds, len(cbar_colors)),
        ),
        cax=cax,
        ticks=cbar_ticks,
    )
    cbar.solids.set_edgecolor('face')
    cbar.solids.set_linewidth(0)
    cbar.minorticks_off()
    cbar.ax.tick_params(which='major', length=4, width=1)
    cbar.ax.tick_params(which='minor', length=0)
    cbar.set_ticklabels(['None', 'Beta', 'Alpha', 'Turn'])
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=x_fontsize)

    plt.suptitle(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    return output_file


def run_heatmap(input_file, output_file, **plot_kwargs):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到文件: {input_file}")

    residues, data = parse_gnu_file(input_file)
    return generate_plot(residues, data, output_file=output_file, **plot_kwargs)

if __name__ == '__main__':
    try:
        input_file = r"3hh2-dssp/replic1/dssp159-174.gnu"
        print(f"处理文件: {input_file}")
        output = run_heatmap(
            input_file=input_file,
            output_file='3hh2-dssp/replic1/heatmap_dssp159-174.png',
            colors=('#ffffff', '#faa26f', '#faa26f', '#b0d6a9', '#b0d6a9', '#b0d6a9', '#3c9bc9', '#3c9bc9'),
            fig_size=(15, 10),
            dpi=150,
            point_size=1650,
            y_fontsize=40,
            x_fontsize=40,
            title_fontsize=42,
            cbar_fontsize=40,
            title='Protein Secondary Structure Dynamics (DSSP)',
            x_label='Time(ns)',
            y_label='Residue',
            cbar_label='',
            x_tick_step=5000,
            x_tick_labels=[0, 200, 400, 600, 800, 1000, 600],
            x_offset=-500,
            colorbar_side='left',
            x_tick_shift=-500,
        )
        print(f"图表已生成: {output}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
