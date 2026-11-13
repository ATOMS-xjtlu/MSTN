import numpy as np
import matplotlib.pyplot as plt
import re

def process_fel_file(filename):
    """Process a FEL file and extract the minimum free-energy profile."""
    x_values = []
    y_values = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                x = float(parts[1])
                y = float(parts[2])
                data.append((x, y))
            except ValueError:
                continue
    
    if not data:
        return np.array([]), np.array([])
    
    data = np.array(data)
    
    unique_x = np.unique(data[:, 0])
    min_y_for_x = []
    
    for x_val in unique_x:
        mask = data[:, 0] == x_val
        y_vals = data[mask, 1]
        min_y = np.min(y_vals)
        min_y_for_x.append(min_y)
    
    return unique_x, np.array(min_y_for_x)

def plot_fel_profiles(file1, file2, labels=None):
    """Plot free-energy profiles from two FEL files."""
    if labels is None:
        labels = ['File 1', 'File 2']
    
    x1, y1 = process_fel_file(file1)
    x2, y2 = process_fel_file(file2)
    
    plt.figure(figsize=(10, 6))
    
    if len(x1) > 0:
        sort_idx1 = np.argsort(x1)
        plt.plot(x1[sort_idx1], y1[sort_idx1], 'b-', linewidth=2, markersize=4, label=labels[0])
    
    if len(x2) > 0:
        sort_idx2 = np.argsort(x2)
        plt.plot(x2[sort_idx2], y2[sort_idx2], 'r-', linewidth=2, markersize=4, label=labels[1])
    
    plt.xlabel('Angle(°)', fontsize=25)
    plt.ylabel('PMF (kcal/mol)', fontsize=25)
    #plt.title('Minimum Free Energy Profiles', fontsize=14, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.grid(True, alpha=0.3)
    
    plt.legend(fontsize=18)
    
    plt.tight_layout()
    
    plt.show()

    return (x1, y1), (x2, y2)

def plot_with_smoothing(file1, file2, labels=None, smooth_window=5):
    """Plot smoothed free-energy profile curves."""
    if labels is None:
        labels = ['File 1', 'File 2']
    
    x1, y1 = process_fel_file(file1)
    x2, y2 = process_fel_file(file2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    if len(x1) > 0:
        sort_idx1 = np.argsort(x1)
        ax1.plot(x1[sort_idx1], y1[sort_idx1], 'b-', linewidth=2, marker='o', markersize=4, label=labels[0])
    
    if len(x2) > 0:
        sort_idx2 = np.argsort(x2)
        ax1.plot(x2[sort_idx2], y2[sort_idx2], 'r-', linewidth=2, marker='s', markersize=4, label=labels[1])
    
    ax1.set_xlabel('Angle', fontsize=18)
    ax1.set_ylabel('PMF', fontsize=18)
    #ax1.set_title('Original Minimum Free Energy Profiles', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='minor', labelsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
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

if __name__ == "__main__":
    file1 = "gibbs-an.txt"
    file2 = "gibbs-an-apo.txt"
    
    labels = ['APO-MSTN:ACTRII', 'APO-MSTN']
    
    print("Processing data...")
    
    data1, data2 = plot_fel_profiles(file1, file2, labels)
    
    # data1, data2 = plot_with_smoothing(file1, file2, labels, smooth_window=3)
    
    if len(data1[0]) > 0:
        print(f"\nFile 1 ({labels[0]}) statistics:")
        print(f"  Data points: {len(data1[0])}")
        print(f"  X range: [{np.min(data1[0]):.2f}, {np.max(data1[0]):.2f}]")
        print(f"  Y range: [{np.min(data1[1]):.2f}, {np.max(data1[1]):.2f}]")
        print(f"  Mean Y: {np.mean(data1[1]):.4f}")
        print(f"  Y minimum at: {data1[0][np.argmin(data1[1])]:.2f}")
    
    if len(data2[0]) > 0:
        print(f"\nFile 2 ({labels[1]}) statistics:")
        print(f"  Data points: {len(data2[0])}")
        print(f"  X range: [{np.min(data2[0]):.2f}, {np.max(data2[0]):.2f}]")
        print(f"  Y range: [{np.min(data2[1]):.2f}, {np.max(data2[1]):.2f}]")
        print(f"  Mean Y: {np.mean(data2[1]):.4f}")
        print(f"  Y minimum at: {data2[0][np.argmin(data2[1])]:.2f}")