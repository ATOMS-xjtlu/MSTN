# -*- coding: utf-8 -*-
"""Figure S14 (rebuilt at print size, same four panels as the delivered version)
with the crystal structures and the clustered states marked on panels A and B."""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

ROOT = r"C:\path\to\project"
PROJ = os.path.join(ROOT, "_work", "pca")
OUT = os.path.join(ROOT, "response_letter", "_workbench", "figures")
os.makedirs(OUT, exist_ok=True)

SYSTEMS = [("Pre-active-MSTN", "Pre-active MSTN", "#1f77b4"),
           ("Traj-MSTN", "Traj-MSTN", "#2ca02c"),
           ("Traj-MSTN-ActRII", "Traj-MSTN:ActRII", "#d62728"),
           ("APO-MSTN", "APO-MSTN", "#9467bd"),
           ("APO-MSTN-ActRII", "APO-MSTN:ActRII", "#ff7f0e"),
           ("Active-MSTN", "Active-MSTN", "#4d4d4d")]
COLOUR = {k: c for k, _, c in SYSTEMS}
REPLICAS = ["main", "replica1", "replica2"]

# crystal-derived starting models: (name, PC1, PC2, Angle-1, D1)
CRYSTALS = [("5JI1", 93.6, -71.9, 51.6, 17.44),
            ("5NTU", -66.7, -77.0, 118.7, 27.54),
            ("3HH2", -59.0, -34.2, 117.9, 5.94)]

OFF = {"1": (3.0, -8.5), "2": (3.0, 2.5), "4": (3.0, 2.5), "5": (-9.0, 2.0),
       "6": (3.0, 2.5), "7": (3.0, -8.0), "8": (-9.5, -7.0), "9": (-8.5, 2.5),
       "10": (2.5, 2.5), "11": (-5.0, -9.0), "12": (-10.0, 2.0)}
CRYSTAL_OFF = {"5JI1": (3.5, 2.5), "5NTU": (3.5, 2.5), "3HH2": (-16.0, -9.0)}
OFF_A = {"1": (3.0, 2.5), "2": (3.0, 2.5), "4": (3.0, 2.5), "5": (3.0, 2.5),
         "6": (3.0, 2.5), "7": (3.0, -8.0), "8": (-3.0, -9.0), "9": (2.5, -9.0),
         "10": (3.0, 2.5), "11": (-9.0, 3.0), "12": (-9.5, 3.0)}

states = []
with open(os.path.join(ROOT, "response_letter", "_workbench", "landmarks", "state_matches.csv"),
          encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        states.append(row)

data = {}
for key, label, colour in SYSTEMS:
    data[key] = np.vstack([np.loadtxt(p, comments="#")
                           for p in [os.path.join(PROJ, "proj_%s__%s.dat" % (key, r))
                                     for r in REPLICAS]])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "pdf.fonttype": 42,
})


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(pad=1.5)


def mark(ax, x, y, colour=None):
    """Landmark markers: black square for crystals, open circle for states."""
    if colour is None:
        ax.plot(x, y, "s", ms=3.6, mfc="black", mec="white", mew=0.5, zorder=8)
    else:
        ax.plot(x, y, "o", ms=3.4, mfc="white", mec=colour, mew=1.0, zorder=8)


fig = plt.figure(figsize=(6.5, 4.3))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.32,
                       left=0.085, right=0.985, top=0.945, bottom=0.185)

# ---------------------------------------------------------------- A: PC1 vs D1
ax = fig.add_subplot(gs[0, 0])
for key, label, colour in SYSTEMS:
    d = data[key]
    ax.scatter(d[::3, 1], d[::3, 12], s=0.5, alpha=0.20, color=colour, linewidths=0,
               rasterized=True)
for name, pc1, pc2, ang, d1v in CRYSTALS:
    mark(ax, pc1, d1v)
    ax.annotate(name, (pc1, d1v), xytext=CRYSTAL_OFF[name], textcoords="offset points",
                fontsize=5.4)
for s in states:
    x, y = float(s["pc1"]), float(s["d1"])
    mark(ax, x, y, COLOUR[s["system"]])
    ax.annotate(s["state"].split()[-1], (x, y), xytext=OFF_A[s["state"].split()[-1]],
                textcoords="offset points", fontsize=5.2, color=COLOUR[s["system"]])
ax.set_xlabel("PC1")
ax.set_ylabel("Distance-1 (\u00c5)")
ax.set_title("PC1 vs Distance-1  (pooled r = \u22120.70)", fontsize=7.4, pad=3)
ax.text(-0.19, 1.06, "A", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")
style(ax)

# ---------------------------------------------------------------- B: PC2 vs angle
ax = fig.add_subplot(gs[0, 1])
for key, label, colour in SYSTEMS:
    d = data[key]
    ax.scatter(d[::3, 2], d[::3, 11], s=0.5, alpha=0.20, color=colour, linewidths=0,
               rasterized=True)
for name, pc1, pc2, ang, d1v in CRYSTALS:
    mark(ax, pc2, ang)
    ax.annotate(name, (pc2, ang), xytext=(3.5, 2.5), textcoords="offset points", fontsize=5.4)
for s in states:
    x, y = float(s["pc2"]), float(s["angle1"])
    mark(ax, x, y, COLOUR[s["system"]])
    ax.annotate(s["state"].split()[-1], (x, y), xytext=(2.8, 2.4), textcoords="offset points",
                fontsize=5.2, color=COLOUR[s["system"]])
ax.set_xlabel("PC2")
ax.set_ylabel("Angle-1 (deg)")
ax.set_title("PC2 vs Angle-1  (pooled r = +0.62)", fontsize=7.4, pad=3)
ax.text(-0.19, 1.06, "B", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")
style(ax)

# ---------------------------------------------------------------- C: R2 bars
ax = fig.add_subplot(gs[1, 0])
labels, a_r2, d1_r2 = [], [], []
pooled_all = {k: np.concatenate([data[s][:, i] for s in data]) for k, i in
              [("angle", 11), ("d1", 12), ("d2", 13), ("pc1", 1), ("pc2", 2)]}
groups = [(lab, np.column_stack([data[k][:, 1], data[k][:, 2]]),
           data[k][:, 11], data[k][:, 12]) for k, lab, _ in SYSTEMS]
groups.append(("pooled", np.column_stack([pooled_all["pc1"], pooled_all["pc2"]]),
               pooled_all["angle"], pooled_all["d1"]))
for lab, pc, ang, d1v in groups:
    X = np.column_stack([np.ones(len(pc)), pc])
    out = []
    for y in (ang, d1v):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.append(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    labels.append(lab)
    a_r2.append(out[0])
    d1_r2.append(out[1])
x = np.arange(len(labels))
ax.bar(x - 0.19, a_r2, width=0.36, color="#4c72b0", label="Angle-1", linewidth=0)
ax.bar(x + 0.19, d1_r2, width=0.36, color="#dd8452", label="Distance-1", linewidth=0)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=5.8)
ax.set_ylabel("R$^2$ from PC1\u2013PC2")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=1.0,
          columnspacing=0.8, borderaxespad=0.2)
ax.set_title("CV content of the PC1\u2013PC2 plane", fontsize=7.4, pad=3)
ax.text(-0.19, 1.06, "C", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")
style(ax)

# ---------------------------------------------------------------- D: PC1 vs time
ax = fig.add_subplot(gs[1, 1])
for key, label, colour in SYSTEMS:
    for rep in REPLICAS:
        p = os.path.join(PROJ, "proj_%s__%s.dat" % (key, rep))
        if os.path.exists(p):
            dd = np.loadtxt(p, comments="#")
            ax.plot(dd[:, 0], dd[:, 1], lw=0.5, color=colour, alpha=0.85)
ax.set_xlabel("time (ns)")
ax.set_ylabel("PC1")
ax.set_title("PC1 along the cMD trajectories", fontsize=7.4, pad=3)
ax.text(-0.19, 1.06, "D", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")
style(ax)

# ---------------------------------------------------------------- legends
h_sys = [Line2D([], [], marker="o", ls="", ms=3.0, color=c, label=l) for _, l, c in SYSTEMS]
h_mark = [Line2D([], [], marker="s", ls="", ms=3.4, mfc="black", mec="white", mew=0.5,
                 label="crystal (start)"),
          Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec="0.4", mew=1.0,
                 label="state 1\u201312")]
leg1 = fig.legend(handles=h_sys + h_mark, loc="lower center", ncol=4, frameon=False,
                  fontsize=5.9, bbox_to_anchor=(0.5, -0.02), columnspacing=0.9,
                  handletextpad=0.25)

fig.savefig(os.path.join(OUT, "Fig_S14_pca_landmarks.png"), dpi=600)
fig.savefig(os.path.join(OUT, "Fig_S14_pca_landmarks.pdf"))

from PIL import Image
Image.open(os.path.join(OUT, "Fig_S14_pca_landmarks.png")).convert("RGB").save(
    os.path.join(OUT, "Fig_S14_pca_landmarks.png"), dpi=(600, 600))
print("saved", os.path.join(OUT, "Fig_S14_pca_landmarks.png"))




