import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageColor import colormap
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, RectBivariateSpline
import sys

# ==================== 参数自定义区 ====================

# 【输入文件】
DATA_FILE = ('129-se.dat')  # 修改为您的数据文件名

# 【自由能颜色范围】
VMIN = 0
VMAX = 16

# 【XY轴范围】(None=自动，从数据中获取)
XLIM = [5, 25]  # 例如: [0, 20] - 超出数据范围的部分将用最高能量填充
YLIM = [0, 35]  # 例如: [70, 90] - 超出数据范围的部分将用最高能量填充
#YLIM =[5, 34.8]
#XLIM = [5, 30.2]


# 【XY轴刻度】(None=自动)
XTICKS = 5, 10,15, 20,25
YTICKS = None

# 【平滑处理参数】
SMOOTH = True
SIGMA = -1  # 高斯平滑参数

# 【插值参数】新增：提高图像分辨率
INTERPOLATE = True  # 是否进行插值
INTERPOLATION_METHOD = 'spline'  # 插值方法：'linear', 'cubic', 'spline'
INTERPOLATION_FACTOR = 8  # 插值倍数：原始网格点数的倍数（1=不插值，2=双倍分辨率，3=三倍分辨率等）

# 【字体大小】
TITLE_SIZE = 36
LABEL_SIZE = 36
TICK_SIZE = 30
COLORBAR_SIZE = 30
COLORBAR_TICK_SIZE = 30
CONTOUR_LABEL_SIZE = 30

# 【等高线设置】
CONTOUR_INTERVAL = 1.0
SHOW_CONTOUR_LABELS = False
CONTOUR_LEVELS = 0  # 自定义等高线等级，如: [0, 2, 4, 6, 8, 10, 12, 14]

# 【输出设置】
OUTPUT_FIG = 'FEL_from_grid.png'
SHOW_PLOT = True

# ======================================================

def read_grid_data(filename):
    """读取网格格式的FEL数据"""
    print(f"正在读取文件: {filename}")
    
    # 读取整个文件
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 分离文件头和数据行
    header_lines = []
    data_lines = []
    
    for line in lines:
        if line.startswith('#'):
            header_lines.append(line.strip())
        else:
            data_lines.append(line.strip())
    
    # 打印文件头信息
    print("文件头信息:")
    for line in header_lines:
        print(line)
    
    # 解析网格参数
    nx = ny = None
    x0 = y0 = None
    dx = dy = None
    
    for line in header_lines:
        if 'Nx=' in line:
            parts = line.replace('#', '').strip().split()
            for part in parts:
                if 'Nx=' in part:
                    nx = int(part.split('=')[1])
                elif 'Ny=' in part:
                    ny = int(part.split('=')[1])
                elif 'X0=' in part:
                    x0 = float(part.split('=')[1])
                elif 'dX=' in part:
                    dx = float(part.split('=')[1])
                elif 'Y0=' in part:
                    y0 = float(part.split('=')[1])
                elif 'dY=' in part:
                    dy = float(part.split('=')[1])
    
    # 如果文件头中没有网格参数，从数据中推断
    if None in [nx, ny, x0, dx, y0, dy]:
        print("警告: 无法从文件头解析所有网格参数，尝试自动检测...")
        
        # 解析数据行
        data = []
        for line in data_lines:
            if line.strip():  # 跳过空行
                parts = line.split()
                if len(parts) >= 3:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        if not data:
            print("错误: 没有找到有效数据")
            sys.exit(1)
        
        data = np.array(data)
        
        # 从数据中推断网格参数
        x_vals = np.unique(data[:, 0])
        y_vals = np.unique(data[:, 1])
        nx = len(x_vals)
        ny = len(y_vals)
        x0 = x_vals[0]
        y0 = y_vals[0]
        dx = x_vals[1] - x_vals[0] if nx > 1 else 1.0
        dy = y_vals[1] - y_vals[0] if ny > 1 else 1.0
        
        print(f"从数据推断的网格参数: Nx={nx}, Ny={ny}, X0={x0:.4f}, dX={dx:.4f}, Y0={y0:.4f}, dY={dy:.4f}")
    
    # 解析数据行
    data = []
    for line in data_lines:
        if line.strip():  # 跳过空行
            parts = line.split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    if not data:
        print("错误: 没有找到有效数据")
        sys.exit(1)
    
    data = np.array(data)
    
    # 检查数据点数是否与网格大小匹配
    expected_points = nx * ny
    if len(data) != expected_points:
        print(f"警告: 数据点数({len(data)})与网格大小({nx}×{ny}={expected_points})不匹配")
        
        # 尝试重塑
        if len(data) % nx == 0:
            ny = len(data) // nx
            print(f"调整Ny为: {ny}")
        elif len(data) % ny == 0:
            nx = len(data) // ny
            print(f"调整Nx为: {nx}")
        else:
            print("警告: 无法将数据重塑为规则网格，尝试使用最小边界框")
            # 按x值排序，然后按y值排序
            data = data[data[:, 0].argsort()]
            data = data[data[:, 1].argsort(kind='mergesort')]
    
    # 重塑为网格
    try:
        x_data = data[:, 0].reshape(ny, nx)
        y_data = data[:, 1].reshape(ny, nx)
        z_data = data[:, 2].reshape(ny, nx)
    except ValueError as e:
        print(f"错误: 无法重塑数据为网格: {e}")
        print(f"数据形状: {data.shape}, 期望的网格: {ny}×{nx}")
        sys.exit(1)
    
    # 检查网格一致性
    x_unique = np.unique(x_data[0, :])
    y_unique = np.unique(y_data[:, 0])
    
    print(f"X轴范围: {x_unique.min():.4f} - {x_unique.max():.4f}")
    print(f"Y轴范围: {y_unique.min():.4f} - {y_unique.max():.4f}")
    print(f"Z值范围: {z_data.min():.4f} - {z_data.max():.4f}")
    
    return x_data, y_data, z_data, nx, ny, x0, dx, y0, dy

def extend_grid_to_range(X, Y, Z, target_x_range, target_y_range, fill_value=None):
    """将网格扩展到目标范围，超出部分用fill_value填充"""
    if fill_value is None:
        fill_value = VMAX  # 默认用最高能量填充
    
    # 获取原始网格参数
    dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
    dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
    
    # 创建目标网格
    x_new = np.arange(target_x_range[0], target_x_range[1] + dx/2, dx)
    y_new = np.arange(target_y_range[0], target_y_range[1] + dy/2, dy)
    
    # 确保网格点数量正确
    if x_new[-1] > target_x_range[1]:
        x_new = x_new[:-1]
    if y_new[-1] > target_y_range[1]:
        y_new = y_new[:-1]
    
    X_new, Y_new = np.meshgrid(x_new, y_new)
    
    # 创建新的Z矩阵，初始值为fill_value
    Z_new = np.full_like(X_new, fill_value, dtype=float)
    
    # 创建掩码矩阵：原始数据区域为True，填充区域为False
    mask = np.zeros_like(X_new, dtype=bool)
    
    # 将原始数据复制到新网格中对应位置
    for i in range(X.shape[0]):  # y方向
        for j in range(X.shape[1]):  # x方向
            # 找到原始点在目标网格中的位置
            x_idx = np.argmin(np.abs(x_new - X[i, j]))
            y_idx = np.argmin(np.abs(y_new - Y[i, j]))
            
            # 如果点在目标范围内，复制值
            if (target_x_range[0] <= X[i, j] <= target_x_range[1] and 
                target_y_range[0] <= Y[i, j] <= target_y_range[1]):
                Z_new[y_idx, x_idx] = Z[i, j]
                mask[y_idx, x_idx] = True
    
    print(f"扩展网格: X范围 [{X.min():.2f}, {X.max():.2f}] -> [{X_new.min():.2f}, {X_new.max():.2f}]")
    print(f"扩展网格: Y范围 [{Y.min():.2f}, {Y.max():.2f}] -> [{Y_new.min():.2f}, {Y_new.max():.2f}]")
    print(f"扩展网格: 新网格大小 {Z_new.shape}")
    print(f"扩展网格: 原始数据区域点数 {np.sum(mask)} / 总点数 {Z_new.size}")
    
    return X_new, Y_new, Z_new, mask

def interpolate_grid(X, Y, Z, mask, factor=2, method='spline'):
    """对网格进行插值，提高分辨率"""
    if factor <= 1:
        print(f"插值倍数 {factor} <= 1，跳过插值")
        return X, Y, Z, mask
    
    print(f"\n正在应用插值，方法: {method}, 倍数: {factor}")
    
    # 获取原始网格参数
    nx_orig, ny_orig = X.shape[1], X.shape[0]
    nx_new = int((nx_orig - 1) * factor + 1)
    ny_new = int((ny_orig - 1) * factor + 1)
    
    print(f"原始网格: {ny_orig}×{nx_orig} = {ny_orig*nx_orig} 个点")
    print(f"插值网格: {ny_new}×{nx_new} = {ny_new*nx_new} 个点")
    print(f"分辨率提高: {factor}倍")
    
    # 创建新的网格坐标
    x_orig = X[0, :]
    y_orig = Y[:, 0]
    
    x_new = np.linspace(x_orig[0], x_orig[-1], nx_new)
    y_new = np.linspace(y_orig[0], y_orig[-1], ny_new)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    
    # 创建掩码的浮点版本，用于插值
    mask_float = mask.astype(float)
    
    # 根据选择的方法进行插值
    if method == 'linear':
        # 线性插值
        Z_new = griddata(
            (X.flatten(), Y.flatten()), 
            Z.flatten(), 
            (X_new, Y_new), 
            method='linear'
        )
        mask_new_float = griddata(
            (X.flatten(), Y.flatten()), 
            mask_float.flatten(), 
            (X_new, Y_new), 
            method='linear'
        )
    
    elif method == 'cubic':
        # 三次插值
        Z_new = griddata(
            (X.flatten(), Y.flatten()), 
            Z.flatten(), 
            (X_new, Y_new), 
            method='cubic'
        )
        mask_new_float = griddata(
            (X.flatten(), Y.flatten()), 
            mask_float.flatten(), 
            (X_new, Y_new), 
            method='linear'  # 掩码使用线性插值
        )
    
    elif method == 'spline':
        # 样条插值（更平滑）
        # 首先检查数据是否在矩形网格上
        if np.allclose(np.diff(x_orig), x_orig[1] - x_orig[0]) and np.allclose(np.diff(y_orig), y_orig[1] - y_orig[0]):
            try:
                # 使用矩形网格样条插值（更精确）
                spline = RectBivariateSpline(y_orig, x_orig, Z)
                Z_new = spline(y_new, x_new)
                
                spline_mask = RectBivariateSpline(y_orig, x_orig, mask_float)
                mask_new_float = spline_mask(y_new, x_new)
            except:
                print("警告: 矩形样条插值失败，回退到线性插值")
                Z_new = griddata(
                    (X.flatten(), Y.flatten()), 
                    Z.flatten(), 
                    (X_new, Y_new), 
                    method='linear'
                )
                mask_new_float = griddata(
                    (X.flatten(), Y.flatten()), 
                    mask_float.flatten(), 
                    (X_new, Y_new), 
                    method='linear'
                )
        else:
            print("警告: 非均匀网格，使用线性插值")
            Z_new = griddata(
                (X.flatten(), Y.flatten()), 
                Z.flatten(), 
                (X_new, Y_new), 
                method='linear'
            )
            mask_new_float = griddata(
                (X.flatten(), Y.flatten()), 
                mask_float.flatten(), 
                (X_new, Y_new), 
                method='linear'
            )
    
    else:
        print(f"警告: 未知的插值方法 '{method}'，使用线性插值")
        Z_new = griddata(
            (X.flatten(), Y.flatten()), 
            Z.flatten(), 
            (X_new, Y_new), 
            method='linear'
        )
        mask_new_float = griddata(
            (X.flatten(), Y.flatten()), 
            mask_float.flatten(), 
            (X_new, Y_new), 
            method='linear'
        )
    
    # 处理插值产生的NaN值（边缘区域）
    nan_mask = np.isnan(Z_new)
    if np.any(nan_mask):
        print(f"警告: 插值产生 {np.sum(nan_mask)} 个NaN值，用邻近值填充")
        
        # 用最近的非NaN值填充NaN
        from scipy import ndimage
        Z_new = ndimage.generic_filter(Z_new, lambda x: x[~np.isnan(x)][0] if np.any(~np.isnan(x)) else np.nan, size=3)
        
        # 如果还有NaN，用平均值填充
        if np.any(np.isnan(Z_new)):
            Z_new[np.isnan(Z_new)] = np.nanmean(Z_new)
    
    # 将掩码二值化
    mask_new = mask_new_float > 0.5
    
    # 将填充区域的值设为VMAX
    Z_new[~mask_new] = VMAX
    
    print(f"插值完成: 新网格大小 {Z_new.shape}")
    
    return X_new, Y_new, Z_new, mask_new

def main():
    print("="*70)
    print("网格FEL图像绘制脚本 (简化版，无能量井标记)")
    print("="*70)
    
    try:
        # 读取数据
        X, Y, Z, nx, ny, x0, dx, y0, dy = read_grid_data(DATA_FILE)
    except Exception as e:
        print(f"读取数据时发生错误: {e}")
        print("请检查数据文件格式是否正确")
        sys.exit(1)
    
    # 自由能数据（假设Z已经是自由能）
    free_energy = Z.copy()
    
    # 将最小值设为0（可选）
    free_energy = free_energy - free_energy.min()
    
    print(f"\n自由能范围: {free_energy.min():.3f} - {free_energy.max():.3f} kcal/mol")
    
    # 确定目标范围
    if XLIM is not None:
        target_x_range = XLIM
    else:
        target_x_range = [X.min(), X.max()]
    
    if YLIM is not None:
        target_y_range = YLIM
    else:
        target_y_range = [Y.min(), Y.max()]
    
    print(f"目标X范围: [{target_x_range[0]:.2f}, {target_x_range[1]:.2f}]")
    print(f"目标Y范围: [{target_y_range[0]:.2f}, {target_y_range[1]:.2f}]")
    
    # 将网格扩展到目标范围，超出部分用最高能量填充
    X_ext, Y_ext, free_energy_ext, mask_ext = extend_grid_to_range(
        X, Y, free_energy, 
        target_x_range, target_y_range, 
        fill_value=VMAX
    )
    
    # 平滑处理
    if SMOOTH:
        print(f"\n应用高斯平滑，sigma={SIGMA}")
        # 注意：只平滑扩展后网格中的数据部分，避免平滑填充的边界
        free_energy_smoothed = gaussian_filter(free_energy_ext, sigma=SIGMA)
        
        # 恢复填充区域为VMAX
        free_energy_smoothed[~mask_ext] = VMAX
        
        free_energy_ext = free_energy_smoothed
    
    # 插值处理（提高分辨率）
    if INTERPOLATE and INTERPOLATION_FACTOR > 1:
        X_final, Y_final, free_energy_final, mask_final = interpolate_grid(
            X_ext, Y_ext, free_energy_ext, mask_ext,
            factor=INTERPOLATION_FACTOR,
            method=INTERPOLATION_METHOD
        )
    else:
        X_final, Y_final, free_energy_final, mask_final = X_ext, Y_ext, free_energy_ext, mask_ext
    
    # 绘图
    print("\n正在绘制FEL图...")
    
    fig, ax = plt.subplots(figsize=(14, 11))
    colormap="jet"
    # 绘制填色图
    im = ax.pcolormesh(X_final, Y_final, free_energy_final, 
                      cmap=colormap,
                      vmin=VMIN, 
                      vmax=VMAX,
                      shading='auto')
    
    # 添加等高线
    if CONTOUR_LEVELS is None:
        contour_levels = np.arange(VMIN, VMAX + CONTOUR_INTERVAL, CONTOUR_INTERVAL)
    else:
        contour_levels = CONTOUR_LEVELS
    
    CS = ax.contour(X_final, Y_final, free_energy_final, 
                    levels=contour_levels, 
                    colors='black', 
                    linewidths=1.2, 
                    alpha=0.7)
    
    # 等高线标签
    if SHOW_CONTOUR_LABELS:
        ax.clabel(CS, inline=True, fontsize=CONTOUR_LABEL_SIZE, 
                  fmt='%g', inline_spacing=10)
    
    # 设置坐标轴范围
    ax.set_xlim(target_x_range[0], target_x_range[1])
    ax.set_ylim(target_y_range[0], target_y_range[1])
    
    # 设置刻度
    if XTICKS:
        ax.set_xticks(XTICKS)
    if YTICKS:
        ax.set_yticks(YTICKS)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, extend='neither', pad=0.02, fraction=0.046)
    cbar.set_label('Free Energy (kcal/mol)', 
                  fontsize=COLORBAR_SIZE, fontweight='bold')
    colormap=("coolwarm")
    
    # 设置颜色条刻度
    cbar_ticks = np.arange(VMIN, VMAX+1, 1)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    
    # 标签
    #ax.set_xlabel('Distance 1 (nm)', fontsize=LABEL_SIZE, fontweight='bold')
    #ax.set_ylabel('Angle 1 (°)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Distance 1 (nm)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_xlabel('Distance 2 (nm)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Free Energy Landscape', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    
    # 刻度
    ax.tick_params(axis='both', labelsize=TICK_SIZE, width=1.5, length=8)
    
    # 布局调整
    plt.tight_layout(pad=1.0)
    
    # 保存图像
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight')
    print(f"\nFEL图已保存为: {OUTPUT_FIG}")
    
    # 显示图像
    if SHOW_PLOT:
        plt.show()
    
    # 打印处理摘要
    print("\n" + "="*70)
    print("处理摘要:")
    print("-"*70)
    print(f"原始网格大小: {nx} × {ny}")
    print(f"最终网格大小: {free_energy_final.shape[1]} × {free_energy_final.shape[0]}")
    print(f"插值倍数: {INTERPOLATION_FACTOR if INTERPOLATE else 1}")
    print(f"平滑处理: {'是' if SMOOTH else '否'} (sigma={SIGMA})")
    print(f"自由能范围: {free_energy_final.min():.3f} - {free_energy_final.max():.3f} kcal/mol")
    print(f"输出文件: {OUTPUT_FIG}")
    print("="*70)

if __name__ == "__main__":
    main()