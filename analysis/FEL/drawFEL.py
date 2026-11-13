import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageColor import colormap
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, RectBivariateSpline
import sys


DATA_FILE = ('129-se.dat')

VMIN = 0
VMAX = 16

XLIM = [5, 25]
YLIM = [0, 35]
#YLIM =[5, 34.8]
#XLIM = [5, 30.2]


XTICKS = 5, 10,15, 20,25
YTICKS = None

SMOOTH = True
SIGMA = -1

INTERPOLATE = True
INTERPOLATION_METHOD = 'spline'
INTERPOLATION_FACTOR = 8

TITLE_SIZE = 36
LABEL_SIZE = 36
TICK_SIZE = 30
COLORBAR_SIZE = 30
COLORBAR_TICK_SIZE = 30
CONTOUR_LABEL_SIZE = 30

CONTOUR_INTERVAL = 1.0
SHOW_CONTOUR_LABELS = False
CONTOUR_LEVELS = 0

OUTPUT_FIG = 'FEL_from_grid.png'
SHOW_PLOT = True

# ======================================================

def read_grid_data(filename):
    """Read FEL data in grid format."""
    print(f"Reading file: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    header_lines = []
    data_lines = []
    
    for line in lines:
        if line.startswith('#'):
            header_lines.append(line.strip())
        else:
            data_lines.append(line.strip())
    
    print("File header:")
    for line in header_lines:
        print(line)
    
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
    
    if None in [nx, ny, x0, dx, y0, dy]:
        print("Warning: could not parse all grid parameters from the header; trying auto-detection...")
        
        data = []
        for line in data_lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        if not data:
            print("Error: no valid data found")
            sys.exit(1)
        
        data = np.array(data)
        
        x_vals = np.unique(data[:, 0])
        y_vals = np.unique(data[:, 1])
        nx = len(x_vals)
        ny = len(y_vals)
        x0 = x_vals[0]
        y0 = y_vals[0]
        dx = x_vals[1] - x_vals[0] if nx > 1 else 1.0
        dy = y_vals[1] - y_vals[0] if ny > 1 else 1.0
        
        print(f"Grid parameters inferred from data: Nx={nx}, Ny={ny}, X0={x0:.4f}, dX={dx:.4f}, Y0={y0:.4f}, dY={dy:.4f}")
    
    data = []
    for line in data_lines:
        if line.strip():
            parts = line.split()
            if len(parts) >= 3:
                data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    
    if not data:
        print("Error: no valid data found")
        sys.exit(1)
    
    data = np.array(data)
    
    expected_points = nx * ny
    if len(data) != expected_points:
        print(f"Warning: {len(data)} data points do not match grid size ({nx}x{ny}={expected_points})")
        
        if len(data) % nx == 0:
            ny = len(data) // nx
            print(f"Adjusted Ny to: {ny}")
        elif len(data) % ny == 0:
            nx = len(data) // ny
            print(f"Adjusted Nx to: {nx}")
        else:
            print("Warning: cannot reshape data into a regular grid; using the minimal bounding box")
            data = data[data[:, 0].argsort()]
            data = data[data[:, 1].argsort(kind='mergesort')]
    
    try:
        x_data = data[:, 0].reshape(ny, nx)
        y_data = data[:, 1].reshape(ny, nx)
        z_data = data[:, 2].reshape(ny, nx)
    except ValueError as e:
        print(f"Error: cannot reshape data into a grid: {e}")
        print(f"Data shape: {data.shape}, expected grid: {ny}x{nx}")
        sys.exit(1)
    
    x_unique = np.unique(x_data[0, :])
    y_unique = np.unique(y_data[:, 0])
    
    print(f"X range: {x_unique.min():.4f} - {x_unique.max():.4f}")
    print(f"Y range: {y_unique.min():.4f} - {y_unique.max():.4f}")
    print(f"Z range: {z_data.min():.4f} - {z_data.max():.4f}")
    
    return x_data, y_data, z_data, nx, ny, x0, dx, y0, dy

def extend_grid_to_range(X, Y, Z, target_x_range, target_y_range, fill_value=None):
    """Extend the grid to the target range, filling new cells with fill_value."""
    if fill_value is None:
        fill_value = VMAX
    
    dx = X[0, 1] - X[0, 0] if X.shape[1] > 1 else 1.0
    dy = Y[1, 0] - Y[0, 0] if Y.shape[0] > 1 else 1.0
    
    x_new = np.arange(target_x_range[0], target_x_range[1] + dx/2, dx)
    y_new = np.arange(target_y_range[0], target_y_range[1] + dy/2, dy)
    
    if x_new[-1] > target_x_range[1]:
        x_new = x_new[:-1]
    if y_new[-1] > target_y_range[1]:
        y_new = y_new[:-1]
    
    X_new, Y_new = np.meshgrid(x_new, y_new)
    
    Z_new = np.full_like(X_new, fill_value, dtype=float)
    
    mask = np.zeros_like(X_new, dtype=bool)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_idx = np.argmin(np.abs(x_new - X[i, j]))
            y_idx = np.argmin(np.abs(y_new - Y[i, j]))
            
            if (target_x_range[0] <= X[i, j] <= target_x_range[1] and 
                target_y_range[0] <= Y[i, j] <= target_y_range[1]):
                Z_new[y_idx, x_idx] = Z[i, j]
                mask[y_idx, x_idx] = True
    
    print(f"Extending grid: X range [{X.min():.2f}, {X.max():.2f}] -> [{X_new.min():.2f}, {X_new.max():.2f}]")
    print(f"Extending grid: Y range [{Y.min():.2f}, {Y.max():.2f}] -> [{Y_new.min():.2f}, {Y_new.max():.2f}]")
    print(f"Extending grid: new grid size {Z_new.shape}")
    print(f"Extending grid: original cells {np.sum(mask)} / total {Z_new.size}")
    
    return X_new, Y_new, Z_new, mask

def interpolate_grid(X, Y, Z, mask, factor=2, method='spline'):
    """Interpolate the grid to increase resolution."""
    if factor <= 1:
        print(f"Interpolation factor {factor} <= 1; skipping interpolation")
        return X, Y, Z, mask
    
    print(f"\nApplying interpolation: method={method}, factor={factor}")
    
    nx_orig, ny_orig = X.shape[1], X.shape[0]
    nx_new = int((nx_orig - 1) * factor + 1)
    ny_new = int((ny_orig - 1) * factor + 1)
    
    print(f"Original grid: {ny_orig}x{nx_orig} = {ny_orig*nx_orig} points")
    print(f"Interpolated grid: {ny_new}x{nx_new} = {ny_new*nx_new} points")
    print(f"Resolution increased by {factor}x")
    
    x_orig = X[0, :]
    y_orig = Y[:, 0]
    
    x_new = np.linspace(x_orig[0], x_orig[-1], nx_new)
    y_new = np.linspace(y_orig[0], y_orig[-1], ny_new)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    
    mask_float = mask.astype(float)
    
    if method == 'linear':
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
            method='linear'
        )
    
    elif method == 'spline':
        if np.allclose(np.diff(x_orig), x_orig[1] - x_orig[0]) and np.allclose(np.diff(y_orig), y_orig[1] - y_orig[0]):
            try:
                spline = RectBivariateSpline(y_orig, x_orig, Z)
                Z_new = spline(y_new, x_new)
                
                spline_mask = RectBivariateSpline(y_orig, x_orig, mask_float)
                mask_new_float = spline_mask(y_new, x_new)
            except:
                print("Warning: rectbivariant spline interpolation failed; falling back to linear")
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
            print("Warning: non-uniform grid; using linear interpolation")
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
        print(f"Warning: unknown interpolation method '{method}'; using linear")
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
    
    nan_mask = np.isnan(Z_new)
    if np.any(nan_mask):
        print(f"Warning: interpolation produced {np.sum(nan_mask)} NaN values; filling from neighbours")
        
        from scipy import ndimage
        Z_new = ndimage.generic_filter(Z_new, lambda x: x[~np.isnan(x)][0] if np.any(~np.isnan(x)) else np.nan, size=3)
        
        if np.any(np.isnan(Z_new)):
            Z_new[np.isnan(Z_new)] = np.nanmean(Z_new)
    
    mask_new = mask_new_float > 0.5
    
    Z_new[~mask_new] = VMAX
    
    print(f"Interpolation complete: new grid size {Z_new.shape}")
    
    return X_new, Y_new, Z_new, mask_new

def main():
    print("="*70)
    print("Grid FEL plotting script (simplified, no basin markers)")
    print("="*70)
    
    try:
        X, Y, Z, nx, ny, x0, dx, y0, dy = read_grid_data(DATA_FILE)
    except Exception as e:
        print(f"Error while reading data: {e}")
        print("Please check the data file format")
        sys.exit(1)
    
    free_energy = Z.copy()
    
    free_energy = free_energy - free_energy.min()
    
    print(f"\nFree-energy range: {free_energy.min():.3f} - {free_energy.max():.3f} kcal/mol")
    
    if XLIM is not None:
        target_x_range = XLIM
    else:
        target_x_range = [X.min(), X.max()]
    
    if YLIM is not None:
        target_y_range = YLIM
    else:
        target_y_range = [Y.min(), Y.max()]
    
    print(f"Target X range: [{target_x_range[0]:.2f}, {target_x_range[1]:.2f}]")
    print(f"Target Y range: [{target_y_range[0]:.2f}, {target_y_range[1]:.2f}]")
    
    X_ext, Y_ext, free_energy_ext, mask_ext = extend_grid_to_range(
        X, Y, free_energy, 
        target_x_range, target_y_range, 
        fill_value=VMAX
    )
    
    if SMOOTH:
        print(f"\nApplying Gaussian smoothing, sigma={SIGMA}")
        free_energy_smoothed = gaussian_filter(free_energy_ext, sigma=SIGMA)
        
        free_energy_smoothed[~mask_ext] = VMAX
        
        free_energy_ext = free_energy_smoothed
    
    if INTERPOLATE and INTERPOLATION_FACTOR > 1:
        X_final, Y_final, free_energy_final, mask_final = interpolate_grid(
            X_ext, Y_ext, free_energy_ext, mask_ext,
            factor=INTERPOLATION_FACTOR,
            method=INTERPOLATION_METHOD
        )
    else:
        X_final, Y_final, free_energy_final, mask_final = X_ext, Y_ext, free_energy_ext, mask_ext
    
    print("\nPlotting FEL...")
    
    fig, ax = plt.subplots(figsize=(14, 11))
    colormap="jet"
    im = ax.pcolormesh(X_final, Y_final, free_energy_final, 
                      cmap=colormap,
                      vmin=VMIN, 
                      vmax=VMAX,
                      shading='auto')
    
    if CONTOUR_LEVELS is None:
        contour_levels = np.arange(VMIN, VMAX + CONTOUR_INTERVAL, CONTOUR_INTERVAL)
    else:
        contour_levels = CONTOUR_LEVELS
    
    CS = ax.contour(X_final, Y_final, free_energy_final, 
                    levels=contour_levels, 
                    colors='black', 
                    linewidths=1.2, 
                    alpha=0.7)
    
    if SHOW_CONTOUR_LABELS:
        ax.clabel(CS, inline=True, fontsize=CONTOUR_LABEL_SIZE, 
                  fmt='%g', inline_spacing=10)
    
    ax.set_xlim(target_x_range[0], target_x_range[1])
    ax.set_ylim(target_y_range[0], target_y_range[1])
    
    if XTICKS:
        ax.set_xticks(XTICKS)
    if YTICKS:
        ax.set_yticks(YTICKS)
    
    cbar = plt.colorbar(im, ax=ax, extend='neither', pad=0.02, fraction=0.046)
    cbar.set_label('Free Energy (kcal/mol)', 
                  fontsize=COLORBAR_SIZE, fontweight='bold')
    colormap=("coolwarm")
    
    cbar_ticks = np.arange(VMIN, VMAX+1, 1)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    
    #ax.set_xlabel('Distance 1 (nm)', fontsize=LABEL_SIZE, fontweight='bold')
    #ax.set_ylabel('Angle 1 (°)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Distance 1 (nm)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_xlabel('Distance 2 (nm)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Free Energy Landscape', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
    
    ax.tick_params(axis='both', labelsize=TICK_SIZE, width=1.5, length=8)
    
    plt.tight_layout(pad=1.0)
    
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches='tight')
    print(f"\nFEL figure saved as: {OUTPUT_FIG}")
    
    if SHOW_PLOT:
        plt.show()
    
    print("\n" + "="*70)
    print("Processing summary:")
    print("-"*70)
    print(f"Original grid size: {nx} x {ny}")
    print(f"Final grid size: {free_energy_final.shape[1]} x {free_energy_final.shape[0]}")
    print(f"Interpolation factor: {INTERPOLATION_FACTOR if INTERPOLATE else 1}")
    print(f"Smoothing: {'yes' if SMOOTH else 'no'} (sigma={SIGMA})")
    print(f"Free-energy range: {free_energy_final.min():.3f} - {free_energy_final.max():.3f} kcal/mol")
    print(f"Output file: {OUTPUT_FIG}")
    print("="*70)

if __name__ == "__main__":
    main()