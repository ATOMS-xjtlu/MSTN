# -*- coding: utf-8 -*-
"""Pre-active MSTN: Ca RMSD to the first frame and to the 1 us structure, for the
three replicas. Quantifies the plateau between 1000 and 2000 ns."""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import MDAnalysis as mda
from MDAnalysis.analysis import rms

ROOT = r"C:\path\to\project"
SYS = "Pre-active-MSTN"
OUT = os.path.join(ROOT, "response_letter", "_workbench", "preactive_md")
os.makedirs(OUT, exist_ok=True)

REPS = [("main", "#1f77b4", "Replica 1"),
        ("replica1", "#d62728", "Replica 2"),
        ("replica2", "#2ca02c", "Replica 3")]


def biggest(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)), key=os.path.getsize, reverse=True)
    return hits[0] if hits else None


series = {}
for rep, colour, label in REPS:
    seg1 = os.path.join(ROOT, "data", "data", "cMD", SYS, rep, "1us")
    seg2 = os.path.join(ROOT, "data", "data", "cMD", SYS, rep, "2us")
    xtc1, xtc2 = biggest(seg1, "*.xtc"), biggest(seg2, "*.xtc")
    pdb = biggest(seg1, "md-*initial.pdb")
    cwd = os.getcwd()
    os.chdir(seg1)
    try:
        u = mda.Universe(os.path.basename(pdb),
                         [os.path.basename(xtc1), os.path.join("..", "2us", os.path.basename(xtc2))])
        sel = u.select_atoms("resid 1-218 and name CA")
        R = rms.RMSD(sel, sel, select="name CA").run()
        t = R.results.rmsd[:, 1] / 1000.0          # ns
        d0 = R.results.rmsd[:, 2]                  # vs frame 0
        ref = sel.positions.copy()                 # frame 0 coordinates

        # trajectory has 40 ps per frame -> 1 us = frame 25000
        step = 40.0
        idx1us = int(round(1000.0 / step))
        u.trajectory[idx1us]
        ref1 = sel.positions.copy()

        d1 = np.empty(len(u.trajectory))
        for i, ts in enumerate(u.trajectory):
            mobile = sel.positions
            d1[i] = rms.rmsd(mobile, ref1, superposition=True, center=False)
    finally:
        os.chdir(cwd)
    series[rep] = (t, d0, d1)
    m = (t >= 1000)
    print("%-9s plateau RMSD(0-ref) = %.2f +- %.2f A   RMSD(1us-ref) = %.2f +- %.2f A"
          % (rep, d0[m].mean(), d0[m].std(), d1[m].mean(), d1[m].std()))

csv = os.path.join(OUT, "rmsd_preactive_plateau.csv")
grid = np.arange(0, 2000.1, 1.0)
cols = {k: (np.interp(grid, v[0], v[1]), np.interp(grid, v[0], v[2])) for k, v in series.items()}
with open(csv, "w", encoding="utf-8") as fh:
    fh.write("time_ns,rmsd0_rep1,rmsd0_rep2,rmsd0_rep3,rmsd1us_rep1,rmsd1us_rep2,rmsd1us_rep3\n")
    for i, tt in enumerate(grid):
        fh.write("%.1f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n"
                 % (tt, cols["main"][0][i], cols["replica1"][0][i], cols["replica2"][0][i],
                    cols["main"][1][i], cols["replica1"][1][i], cols["replica2"][1][i]))
print("saved", csv)

rcParams.update({"font.family": "sans-serif",
                 "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
                 "axes.linewidth": 0.8, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.4, 2.9))
for rep, colour, label in REPS:
    t, d0, _ = series[rep]
    ax.plot(t, d0, color=colour, lw=0.55, label=label)
ax.axvspan(1000, 1500, color="0.85", alpha=0.55, lw=0, zorder=0)
ax.text(1250, 11.3, "plateau\n1000\u20131500 ns", ha="center", va="top", fontsize=6.5, color="0.25")
ax.set_xlim(0, 2000)
ax.set_ylim(0, 12)
ax.set_xlabel("time (ns)")
ax.set_ylabel("C\u03b1 RMSD (\u00c5)")
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "Fig_Pre-active-MSTN_rmsd." + ext), dpi=600)
print("saved figure")
