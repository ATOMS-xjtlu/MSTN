import numpy as np
import matplotlib.pyplot as plt
import re

def process_fel_file(filename):
    """处理FEL文件，提取最小自由能剖面"""
    x_values = []
    y_values = []
    
    # 读取文件
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 解析数据
    data = []
    for line in lines:
        # 跳过注释行和空行
        if line.startswith('#') or line.strip() == '':
            continue
        
        # 分割行
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                # 注意：文件中第一列是x，第二列是y，第三列是z
                # 根据问题描述，我们要用第二列作为横坐标，第三列作为纵坐标
                x = float(parts[1])  # 第二列作为横坐标
                y = float(parts[2])  # 第三列作为纵坐标
                data.append((x, y))
            except ValueError:
                continue
    
    if not data:
        return np.array([]), np.array([])
    
    # 转换为numpy数组
    data = np.array(data)
    
    # 按x值分组，取每组y的最小值
    unique_x = np.unique(data[:, 0])
    min_y_for_x = []
    
    for x_val in unique_x:
        # 找出所有具有相同x值的数据点
        mask = data[:, 0] == x_val
        y_vals = data[mask, 1]
        # 取y的最小值
        min_y = np.min(y_vals)
        min_y_for_x.append(min_y)
    
    return unique_x, np.array(min_y_for_x)

def plot_fel_profiles(file1, file2, labels=None):
    """绘制两个FEL文件的自由能剖面曲线"""
    if labels is None:
        labels = ['File 1', 'File 2']
    
    # 处理第一个文件
    x1, y1 = process_fel_file(file1)
    x2, y2 = process_fel_file(file2)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    if len(x1) > 0:
        # 对x1排序，确保曲线绘制正确
        sort_idx1 = np.argsort(x1)
        plt.plot(x1[sort_idx1], y1[sort_idx1], 'b-', linewidth=2, markersize=4, label=labels[0])
    
    if len(x2) > 0:
        # 对x2排序，确保曲线绘制正确
        sort_idx2 = np.argsort(x2)
        plt.plot(x2[sort_idx2], y2[sort_idx2], 'r-', linewidth=2, markersize=4, label=labels[1])
    
    # 添加标签和标题
    plt.xlabel('Angle(°)', fontsize=25)
    plt.ylabel('PMF (kcal/mol)', fontsize=25)
    #plt.title('Minimum Free Energy Profiles', fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=20)  # 设置坐标轴刻度字体大小
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(fontsize=18)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()

    # 返回数据以便进一步分析
    return (x1, y1), (x2, y2)

def plot_with_smoothing(file1, file2, labels=None, smooth_window=5):
    """绘制带有平滑处理的自由能剖面曲线"""
    if labels is None:
        labels = ['File 1', 'File 2']
    
    # 处理第一个文件
    x1, y1 = process_fel_file(file1)
    x2, y2 = process_fel_file(file2)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始数据
    if len(x1) > 0:
        sort_idx1 = np.argsort(x1)
        ax1.plot(x1[sort_idx1], y1[sort_idx1], 'b-', linewidth=2, marker='o', markersize=4, label=labels[0])
    
    if len(x2) > 0:
        sort_idx2 = np.argsort(x2)
        ax1.plot(x2[sort_idx2], y2[sort_idx2], 'r-', linewidth=2, marker='s', markersize=4, label=labels[1])
    
    ax1.set_xlabel('Angle', fontsize=18)
    ax1.set_ylabel('PMF', fontsize=18)
    #ax1.set_title('Original Minimum Free Energy Profiles', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=12)  # 主要刻度
    ax1.tick_params(axis='both', which='minor', labelsize=10)  # 次要刻度
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=18)
    plt.xticks(fontsize=18)  # x轴刻度字体大小
    plt.yticks(fontsize=18)  # y轴刻度字体大小
    # 平滑处理后的数据
    if len(x1) > 0 and len(y1) > smooth_window:
        from scipy.ndimage import uniform_filter1d
        y1_smooth = uniform_filter1d(y1[sort_idx1], size=smooth_window)
        ax2.plot(x1[sort_idx1], y1_smooth, 'b-', linewidth=2, marker='o', markersize=4, label=f'{labels[0]} (smoothed)')
    
    if len(x2) > 0 and len(y2) > smooth_window:
        from scipy.ndimage import uniform_filter1d
        y2_smooth = uniform_filter1d(y2[sort_idx2], size=smooth_window)
        ax2.plot(x2[sort_idx2], y2_smooth, 'r-', linewidth=2, marker='s', markersize=4, label=f'{labels[1]} (smoothed)')
    
    ax2.set_xlabel('Reaction Coordinate (y)', fontsize=18)
    ax2.set_ylabel('Free Energy (z)', fontsize=18)
    ax2.set_title(f'Smoothed Profiles (window={smooth_window})', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=18)

    plt.tight_layout()
    plt.show()
    
    return (x1, y1), (x2, y2)

# 主程序
if __name__ == "__main__":
    # 文件路径
    file1 = "gibbs-an.txt"
    file2 = "gibbs-an-apo.txt"
    
    # 可选：自定义标签
    labels = ['APO-MSTN:ACTRII', 'APO-MSTN']
    
    print("正在处理数据...")
    
    # 绘制原始数据
    data1, data2 = plot_fel_profiles(file1, file2, labels)
    
    # 如果需要平滑处理，取消下面一行的注释
    # data1, data2 = plot_with_smoothing(file1, file2, labels, smooth_window=3)
    
    # 打印统计信息
    if len(data1[0]) > 0:
        print(f"\n文件1 ({labels[0]}) 统计:")
        print(f"  数据点数量: {len(data1[0])}")
        print(f"  X范围: [{np.min(data1[0]):.2f}, {np.max(data1[0]):.2f}]")
        print(f"  Y范围: [{np.min(data1[1]):.2f}, {np.max(data1[1]):.2f}]")
        print(f"  Y平均值: {np.mean(data1[1]):.4f}")
        print(f"  Y最小值位置: {data1[0][np.argmin(data1[1])]:.2f}")
    
    if len(data2[0]) > 0:
        print(f"\n文件2 ({labels[1]}) 统计:")
        print(f"  数据点数量: {len(data2[0])}")
        print(f"  X范围: [{np.min(data2[0]):.2f}, {np.max(data2[0]):.2f}]")
        print(f"  Y范围: [{np.min(data2[1]):.2f}, {np.max(data2[1]):.2f}]")
        print(f"  Y平均值: {np.mean(data2[1]):.4f}")
        print(f"  Y最小值位置: {data2[0][np.argmin(data2[1])]:.2f}")