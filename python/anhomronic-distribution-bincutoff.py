import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import simps

def calculate_anharmonicity(data, kBT=0.592, bin_width=None, cutoff=None):
    """
    计算ΔV分布的非谐性值γ
    
    参数:
    data: ΔV数据数组 (kcal/mol)
    kBT: 玻尔兹曼常数乘以温度 (kcal/mol)，默认298K时约为0.592
    bin_width: 直方图分箱宽度 (kcal/mol)，如果为None则自动计算
    cutoff: 截断值，低于此值的数据不参与计算 (kcal/mol)
    """
    # 应用截断
    if cutoff is not None:
        data = data[data <= cutoff]
        if len(data) == 0:
            raise ValueError("应用截断后没有剩余数据，请调整cutoff值")
    
    # 将ΔV转换为无量纲
    deltaV_dimensionless = data / kBT
    
    # 计算标准差
    sigma = np.std(deltaV_dimensionless)
    
    # 计算最大熵 S_max
    S_max = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    
    # 确定分箱参数
    if bin_width is None:
        # 使用自动分箱，基于数据范围和标准差
        data_range = np.max(deltaV_dimensionless) - np.min(deltaV_dimensionless)
        bins = max(30, int(data_range / (0.5 * sigma)))
    else:
        # 使用指定的分箱宽度
        bin_width_dimensionless = bin_width / kBT
        min_val = np.min(deltaV_dimensionless)
        max_val = np.max(deltaV_dimensionless)
        bins = np.arange(min_val, max_val + bin_width_dimensionless, bin_width_dimensionless)
    
    # 计算概率密度函数
    hist, bin_edges = np.histogram(deltaV_dimensionless, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 计算实际分布的熵 S_ΔV = -∫p(ΔV)ln(p(ΔV))dΔV
    # 避免log(0)的情况
    mask = hist > 0
    if np.sum(mask) == 0:
        raise ValueError("没有有效的直方图数据点，请调整bin_width或cutoff参数")
    
    S_deltaV = -simps(hist[mask] * np.log(hist[mask]), bin_centers[mask])
    
    # 计算非谐性 γ = S_max - S_ΔV
    gamma = S_max - S_deltaV
    
    return gamma, hist, bin_centers, sigma, S_max, S_deltaV

def read_gamd_log(filename):
    """
    读取GaMD日志文件，提取Boost能量数据
    """
    boost_potential = []
    boost_dihedral = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                try:
                    boost_pot = float(parts[5])
                    boost_dih = float(parts[6])
                    boost_potential.append(boost_pot)
                    boost_dihedral.append(boost_dih)
                except ValueError:
                    continue
    
    # 总ΔV = Boost-Energy-Potential + Boost-Energy-Dihedral
    total_deltaV = np.array(boost_potential) + np.array(boost_dihedral)
    
    return total_deltaV

def plot_distribution(data, kBT=0.592, bin_width=None, cutoff=None):
    """
    绘制ΔV分布图
    """
    # 计算非谐性值
    gamma, hist, bin_centers, sigma, S_max, S_deltaV = calculate_anharmonicity(
        data, kBT, bin_width, cutoff)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制实际分布
    ax.plot(bin_centers * kBT, hist / kBT, 'b-', linewidth=2, label='Potential energy distribution')
    
    # 绘制高斯分布拟合
    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    gaussian_fit = stats.norm.pdf(x_fit, loc=np.mean(data/kBT), scale=sigma)
    ax.plot(x_fit * kBT, gaussian_fit / kBT, 'r--', linewidth=2, label='Fit Gaussian distribution')
    
    # 设置坐标轴标签
    ax.set_xlabel('ΔV (kcal/mol)', fontsize=18)
    ax.set_ylabel('p(ΔV)', fontsize=18)
    
    # 设置标题
    #ax.set_title('ΔV分布与非谐性分析', fontsize=14)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(fontsize=14)
    
    # 添加统计信息文本框
    stats_text = f'γ = {gamma:.6f}\n'
    stats_text += f'σ = {sigma*kBT:.3f} kcal/mol\n'
    stats_text += f'S_max = {S_max:.3f}\n'
    stats_text += f'S_ΔV = {S_deltaV:.3f}'
    
    if cutoff is not None:
        stats_text += f'\ncutoff = {cutoff:.3f} kcal/mol'
    
    ax.text(0.63, 0.85, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#DCDCDC'),
            fontsize=14)
    
    # 设置y轴范围
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig, gamma

# 主程序
if __name__ == "__main__":
    # 读取数据
    try:
        deltaV_data = read_gamd_log('gamd-1us.log')
        print(f"成功读取 {len(deltaV_data)} 个数据点")
        
        # 计算基本统计量
        print(f"ΔV统计信息:")
        print(f"  均值: {np.mean(deltaV_data):.3f} kcal/mol")
        print(f"  标准差: {np.std(deltaV_data):.3f} kcal/mol")
        print(f"  最小值: {np.min(deltaV_data):.3f} kcal/mol")
        print(f"  最大值: {np.max(deltaV_data):.3f} kcal/mol")
        
        # 设置参数
        bin_width = 0.17  # 分箱宽度 (kcal/mol)
        cutoff = 17  # 截断值 (kcal/mol)，低于此值的数据不参与计算
        
        print(f"\n使用参数: bin_width={bin_width}, cutoff={cutoff}")
        
        # 绘制分布图
        fig, gamma = plot_distribution(deltaV_data, bin_width=bin_width, cutoff=cutoff)
        
        # 保存图像
        plt.savefig('deltaV_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nγ = {gamma:.6f}")
        
        # 判断分布性质
        if abs(gamma) < 0.01:
            print("ΔV分布接近高斯分布，重加权结果可靠")
        else:
            print("ΔV分布偏离高斯分布，重加权结果可能存在偏差")
            
    except FileNotFoundError:
        print("错误: 未找到gamd-1500.log文件")
        print("请确保文件路径正确")
    except ValueError as e:
        print(f"错误: {e}")
        
    # 使用示例数据测试（如果实际数据量不足）
    print("\n" + "="*50)
    print("示例: 使用生成的高斯数据测试")
    
    # 生成示例高斯数据
    np.random.seed(42)
    example_data = np.random.normal(6.0, 1.5, 1000)
    _, example_gamma = plot_distribution(example_data, bin_width=0.1, cutoff=15.0)
    plt.title('示例: 高斯分布测试')
    plt.show()
    print(f"示例高斯数据的非谐性值: {example_gamma:.6f}")