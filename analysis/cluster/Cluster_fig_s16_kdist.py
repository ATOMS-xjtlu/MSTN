# -*- coding: utf-8 -*-
"""Figure S16 (a): k-distance curves of the six cMD ensembles (k = minPts = 25),
used to justify the neighbourhood radius eps = 1.5 A."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\path\to\project"
CL = os.path.join(ROOT, "response_letter", "_workbench", "clustering")
OUT = os.path.join(ROOT, "response_letter", "_workbench", "figures")
os.makedirs(OUT, exist_ok=True)

SYSTEMS = [("Pre-active-MSTN", "Pre-active MSTN", "#1f77b4"),
           ("Traj-MSTN", "Traj-MSTN", "#2ca02c"),
           ("Traj-MSTN-ActRII", "Traj-MSTN:ActRII", "#d62728"),
           ("APO-MSTN", "APO-MSTN", "#9467bd"),
           ("APO-MSTN-ActRII", "APO-MSTN:ActRII", "#ff7f0e"),
           ("Active-MSTN", "Active-MSTN", "#4d4d4d")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 2.7))
for key, label, colour in SYSTEMS:
    p = os.path.join(CL, "%s_kdist.npy" % key)
    if not os.path.exists(p):
        continue
    kd = np.load(p)
    ax.plot(np.linspace(0, 100, len(kd)), kd, color=colour, lw=0.9, label=label)
ax.axhline(1.5, color="0.25", lw=0.8, ls=(0, (4, 2)))
ax.text(2, 1.56, r"cut-off 1.5 $\AA$", fontsize=6.2, color="0.25")
ax.set_xlabel("frames, sorted by distance to the 25th neighbour (%)")
ax.set_ylabel("distance to 25th neighbour (\u00c5)")
ax.set_xlim(0, 100)
ax.set_ylim(0, 4.0)
ax.legend(fontsize=5.6, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "Fig_S16_kdistance.png"), dpi=600)
fig.savefig(os.path.join(OUT, "Fig_S16_kdistance.pdf"))

from PIL import Image
Image.open(os.path.join(OUT, "Fig_S16_kdistance.png")).convert("RGB").save(
    os.path.join(OUT, "Fig_S16_kdistance.png"), dpi=(600, 600))
print("saved")
