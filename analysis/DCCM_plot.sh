#!/bin/bash

set -euo pipefail

top="../md-initial.pdb"
traj="../md-nW1000.xtc"
start=0
end=5000
stride=1
residues="1-218"
outfile="dccm.csv"
outfigure="dccm.tif"
cmap="bwr"  

usage() {
    echo "Usage: $0 --top TOPOLOGY --traj TRAJECTORY --start START --end END --stride STRIDE --residues 1-999 --matrix OUTFILE --figure OUTFIG --cmap CMAP"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --top) top="$2"; shift 2 ;;
        --traj) traj="$2"; shift 2 ;;
        --start) start="$2"; shift 2 ;;
        --end) end="$2"; shift 2 ;;
        --stride) stride="$2"; shift 2 ;;
        --residues) residues="$2"; shift 2 ;;
        --matrix) outfile="$2"; shift 2 ;;
        --figure) outfigure="$2"; shift 2 ;;
        --cmap) cmap="$2"; shift 2 ;;  
        *) echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$top" || -z "$traj" ]] && usage
[[ ! -f "$top" ]] && { echo "Error: $top not found."; exit 1; }
[[ ! -f "$traj" ]] && { echo "Error: $traj not found."; exit 1; }

res_start=$(echo "$residues" | cut -d- -f1)
res_end=$(echo "$residues" | cut -d- -f2)

cpp_input=$(mktemp)
py_script=$(mktemp)

cat > "$cpp_input" <<EOF
parm $top
trajin $traj $start $end $stride
matrix :$residues@CA correl out $outfile byres
run
quit
EOF

cpptraj -i "$cpp_input"
rm -f "$cpp_input"

sed 's/[[:space:]]\+/,/g' "$outfile" | sed 's/^,//' > "${outfile}.tmp"
mv "${outfile}.tmp" "$outfile"

cat > "$py_script" <<EOF
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

try:
    data = pd.read_csv("$outfile", header=None)
    matrix = data.astype(float).values


    res_start = $res_start
    res_end = $res_end
    residues = np.arange(res_start, res_end + 1)

    mpl.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.figsize': (10, 9),
        'image.cmap': '$cmap',
    })

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="$cmap", interpolation='nearest')

    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")
    ax.set_title("Dynamic Cross-Correlation Matrix (DCCM)")

    num_ticks = 20
    ticks = np.round(np.linspace(0, matrix.shape[0] - 1, num_ticks - 1)).astype(int)
    ticks = np.append(ticks, matrix.shape[0] - 1)
    ticks = np.unique(ticks)
    
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(residues[ticks])
    ax.set_yticklabels(residues[ticks])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation Coefficient", size=12)

    plt.tight_layout()
    plt.savefig("$outfigure", dpi=300)
    plt.close()

except Exception as e:
    print(f"Plotting failed: {e}")
    exit(1)
EOF

python "$py_script"
rm -f "$py_script"
