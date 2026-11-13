import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import simps

def calculate_anharmonicity(data, kBT=0.592, bin_width=None, cutoff=None):
    """
    Compute the anharmonicity gamma of the DeltaV distribution.
    
    Parameters:
    data: array of DeltaV values (kcal/mol)
    kBT: Boltzmann constant times temperature (kcal/mol); about 0.592 at 298 K by default
    bin_width: histogram bin width (kcal/mol); computed automatically if None
    cutoff: values below this cutoff are excluded (kcal/mol)
    """
    if cutoff is not None:
        data = data[data <= cutoff]
        if len(data) == 0:
            raise ValueError("No data left after applying the cutoff; please adjust cutoff")
    
    deltaV_dimensionless = data / kBT
    
    sigma = np.std(deltaV_dimensionless)
    
    S_max = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    
    if bin_width is None:
        data_range = np.max(deltaV_dimensionless) - np.min(deltaV_dimensionless)
        bins = max(30, int(data_range / (0.5 * sigma)))
    else:
        bin_width_dimensionless = bin_width / kBT
        min_val = np.min(deltaV_dimensionless)
        max_val = np.max(deltaV_dimensionless)
        bins = np.arange(min_val, max_val + bin_width_dimensionless, bin_width_dimensionless)
    
    hist, bin_edges = np.histogram(deltaV_dimensionless, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mask = hist > 0
    if np.sum(mask) == 0:
        raise ValueError("No valid histogram bins; please adjust bin_width or cutoff")
    
    S_deltaV = -simps(hist[mask] * np.log(hist[mask]), bin_centers[mask])
    
    gamma = S_max - S_deltaV
    
    return gamma, hist, bin_centers, sigma, S_max, S_deltaV

def read_gamd_log(filename):
    """
    Read a GaMD log file and extract the boost energy data.
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
    
    total_deltaV = np.array(boost_potential) + np.array(boost_dihedral)
    
    return total_deltaV

def plot_distribution(data, kBT=0.592, bin_width=None, cutoff=None):
    """
    Plot the DeltaV distribution.
    """
    gamma, hist, bin_centers, sigma, S_max, S_deltaV = calculate_anharmonicity(
        data, kBT, bin_width, cutoff)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(bin_centers * kBT, hist / kBT, 'b-', linewidth=2, label='Potential energy distribution')
    
    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 1000)
    gaussian_fit = stats.norm.pdf(x_fit, loc=np.mean(data/kBT), scale=sigma)
    ax.plot(x_fit * kBT, gaussian_fit / kBT, 'r--', linewidth=2, label='Fit Gaussian distribution')
    
    ax.set_xlabel('ΔV (kcal/mol)', fontsize=18)
    ax.set_ylabel('p(ΔV)', fontsize=18)
    
    
    ax.grid(True, alpha=0.3)
    
    ax.legend(fontsize=14)
    
    stats_text = f'γ = {gamma:.6f}\n'
    stats_text += f'σ = {sigma*kBT:.3f} kcal/mol\n'
    stats_text += f'S_max = {S_max:.3f}\n'
    stats_text += f'S_ΔV = {S_deltaV:.3f}'
    
    if cutoff is not None:
        stats_text += f'\ncutoff = {cutoff:.3f} kcal/mol'
    
    ax.text(0.63, 0.85, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#DCDCDC'),
            fontsize=14)
    
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig, gamma

if __name__ == "__main__":
    try:
        deltaV_data = read_gamd_log('gamd-1us.log')
        print(f"Read {len(deltaV_data)} data points")
        
        print(f"DeltaV statistics:")
        print(f"  Mean: {np.mean(deltaV_data):.3f} kcal/mol")
        print(f"  SD: {np.std(deltaV_data):.3f} kcal/mol")
        print(f"  Min: {np.min(deltaV_data):.3f} kcal/mol")
        print(f"  Max: {np.max(deltaV_data):.3f} kcal/mol")
        
        bin_width = 0.17
        cutoff = 17
        
        print(f"\nParameters used: bin_width={bin_width}, cutoff={cutoff}")
        
        fig, gamma = plot_distribution(deltaV_data, bin_width=bin_width, cutoff=cutoff)
        
        plt.savefig('deltaV_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nγ = {gamma:.6f}")
        
        if abs(gamma) < 0.01:
            print("DeltaV distribution is close to Gaussian; reweighting is reliable")
        else:
            print("DeltaV distribution deviates from Gaussian; reweighting may be biased")
            
    except FileNotFoundError:
        print("Error: gamd-1500.log not found")
        print("Please check the file path")
    except ValueError as e:
        print(f"Error: {e}")
        
    print("\n" + "="*50)
    print("Example: test with generated Gaussian data")
    
    np.random.seed(42)
    example_data = np.random.normal(6.0, 1.5, 1000)
    _, example_gamma = plot_distribution(example_data, bin_width=0.1, cutoff=15.0)
    plt.title('Example: Gaussian distribution test')
    plt.show()
    print(f"Anharmonicity of the example Gaussian data: {example_gamma:.6f}")