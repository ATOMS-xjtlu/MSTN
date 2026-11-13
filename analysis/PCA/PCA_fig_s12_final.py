# -*- coding: utf-8 -*-
"""Figure S12, final layout (2 x 4, reading order A-H):

  A variance | B non-APO projection | C GaMD, non-APO | D APO projection (merged)
  E GaMD, APO | F Pre-active FEL | G Traj-MSTN:ActRII FEL | H Active-MSTN FEL
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as _pe
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

ROOT = r"C:\path\to\project"
PROJ = os.path.join(ROOT, "_work", "pca")
OUT = os.path.join(ROOT, "response_letter", "_workbench", "figures")
os.makedirs(OUT, exist_ok=True)

KT = 0.0019872041 * 310.15
PC1 = dict(lo=-92.0, hi=132.0, step=4.0)
PC2 = dict(lo=-116.0, hi=100.0, step=4.0)
MIN_COUNT, DG_CAP, DG_VMAX = 5, 6.0, 3.0

SYSTEMS = [("Pre-active-MSTN", "Pre-active MSTN", "#1f77b4"),
           ("Traj-MSTN", "Traj-MSTN", "#2ca02c"),
           ("Traj-MSTN-ActRII", "Traj-MSTN:ActRII", "#d62728"),
           ("APO-MSTN", "APO-MSTN", "#9467bd"),
           ("APO-MSTN-ActRII", "APO-MSTN:ActRII", "#ff7f0e"),
           ("Active-MSTN", "Active-MSTN", "#4d4d4d")]
NON_APO = [s for s in SYSTEMS if not s[0].startswith("APO")]
APO = [s for s in SYSTEMS if s[0].startswith("APO")]
GAMD = [("Pre-active-MSTN", "Pre-active MSTN", "#1f77b4"),
        ("Pre-active-MSTN-ActRII", "Pre-active MSTN:ActRII", "#17becf"),
        ("APO-MSTN", "APO-MSTN", "#9467bd"),
        ("APO-MSTN-ActRII", "APO-MSTN:ActRII", "#ff7f0e")]
GAMD_NON_APO = [g for g in GAMD if not g[0].startswith("APO")]
GAMD_APO = [g for g in GAMD if g[0].startswith("APO")]
REPLICAS = ["main", "replica1", "replica2"]

cmd = {k: np.vstack([np.loadtxt(p, comments="#")
                     for p in [os.path.join(PROJ, "proj_%s__%s.dat" % (k, r)) for r in REPLICAS]])
       for k, _, _ in SYSTEMS}
gamd = {k: np.vstack([np.loadtxt(p, comments="#")
                      for p in sorted(glob.glob(os.path.join(PROJ, "proj_gamd_%s*.dat" % k)))])
        for k, _, _ in GAMD}
model = np.load(os.path.join(ROOT, "R3-1_delivery", "02_PCA_results", "model.npz"), allow_pickle=True)
evr = model["evr"]

COLOURS = {k: c for k, _, c in SYSTEMS}
STATE_SYSTEM = {"1": "APO-MSTN", "2": "APO-MSTN", "4": "APO-MSTN-ActRII",
                "5": "Pre-active-MSTN", "6": "Pre-active-MSTN", "7": "Pre-active-MSTN",
                "8": "Pre-active-MSTN", "9": "Traj-MSTN-ActRII", "10": "Traj-MSTN-ActRII",
                "11": "Traj-MSTN-ActRII", "12": "Traj-MSTN-ActRII"}
LM = [("5JI1", 93.6, -71.9, None), ("5NTU", -66.7, -77.0, None), ("3HH2", -59.0, -34.2, None)]
with open(os.path.join(ROOT, "response_letter", "_workbench", "landmarks", "state_matches.csv"),
          encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        lab = row["state"].split()[-1]
        LM.append((lab, float(row["pc1"]), float(row["pc2"]),
                   COLOURS[STATE_SYSTEM.get(lab, row["system"])]))
OFF = {"5JI1": (3.2, 2.6), "5NTU": (-5.0, -9.0), "3HH2": (-5.0, -9.0),
       "1": (6.0, -12.0), "2": (3.2, 2.6), "4": (3.2, 2.6), "5": (3.2, -7.6),
       "6": (3.2, 2.6), "7": (3.2, 2.6), "8": (-7.0, 2.2), "9": (3.2, -7.6),
       "10": (3.2, 2.6), "11": (-8.0, 2.2), "12": (-7.0, 2.2)}


def edges(g):
    n = int(round((g["hi"] - g["lo"]) / g["step"]))
    return np.linspace(g["lo"], g["hi"], n + 1)


XE, YE = edges(PC1), edges(PC2)
XC = 0.5 * (XE[1:] + XE[:-1])
YC = 0.5 * (YE[1:] + YE[:-1])


def hist(x, y):
    h, _, _ = np.histogram2d(x, y, bins=(XE, YE))
    return h


def density(h, sigma=1.5):
    hs = gaussian_filter(h, sigma)
    return hs / max(hs.sum(), 1e-300)


def free_energy(h, sigma=1.0):
    p = density(h, sigma)
    m = h >= MIN_COUNT
    with np.errstate(divide="ignore"):
        dg = -KT * np.log(np.where(p > 0, p, 1.0))
    dg[~m] = np.nan
    if np.isfinite(dg).any():
        dg -= np.nanmin(dg)
        dg = np.minimum(dg, DG_CAP)
    return dg


def landmarks(ax, states=None, crystals=("5JI1", "5NTU", "3HH2"), fs=4.6, size=3.0):
    for lab, x, y, col in LM:
        if col is None:
            if lab not in crystals:
                continue
            ax.plot(x, y, "s", ms=size + 0.6, mfc="black", mec="white", mew=0.4, zorder=9)
        else:
            if states is not None and lab not in states:
                continue
            ax.plot(x, y, "o", ms=size, mfc=col, mec="white", mew=0.4, zorder=9)
        ax.annotate(lab, (x, y), xytext=OFF.get(lab, (3, 3)), textcoords="offset points",
                    fontsize=fs, color="black", zorder=10,
                    path_effects=[_pe.withStroke(linewidth=1.0, foreground="white")])


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 1.8, "ytick.major.size": 1.8, "pdf.fonttype": 42,
})
cmap = LinearSegmentedColormap.from_list("fel", ["#00007f", "#1f8fff", "#8fe0c0",
                                                 "#ffe066", "#ff8c1a", "#b30000"])

fig = plt.figure(figsize=(7.0, 3.55))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.62, wspace=0.42,
                       left=0.055, right=0.955, top=0.845, bottom=0.115)


def tight(*arrs, pad=0.12):
    """Axis limits that enclose the given frame sets with a small margin."""
    x0 = min(a[:, 1].min() for a in arrs); x1 = max(a[:, 1].max() for a in arrs)
    y0 = min(a[:, 2].min() for a in arrs); y1 = max(a[:, 2].max() for a in arrs)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - dx, x1 + dx), (y0 - dy, y1 + dy)


def base(ax, letter, title=None, lim=None):
    if lim is None:
        ax.set_xlim(PC1["lo"], PC1["hi"]); ax.set_ylim(PC2["lo"], PC2["hi"])
    else:
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1])
    ax.set_xlabel("PC1", labelpad=1); ax.set_ylabel("PC2", labelpad=1)
    ax.tick_params(labelsize=5.2, pad=1)
    ax.set_title("(%s)%s" % (letter, "" if not title else " " + title),
                 loc="left", fontsize=7, fontweight="bold", pad=2)


# ---------------------------------------------------------------- A
ax = fig.add_subplot(gs[0, 0])
ax.bar(np.arange(1, 11), 100 * evr[:10], width=0.72, color="#4a7ebb", lw=0)
ax2 = ax.twinx()
ax2.plot(np.arange(1, 11), 100 * np.cumsum(evr[:10]), color="#c0392b", lw=0.8, marker="o", ms=1.6)
ax2.set_ylim(0, 100)
ax2.tick_params(labelsize=5.2, pad=1)
ax.set_xticks([1, 2, 4, 6, 8, 10])
ax.tick_params(labelsize=5.2, pad=1)
ax.set_xlabel("PC", labelpad=1); ax.set_ylabel("variance (%)", labelpad=1)
ax.text(0.40, 0.55, "62.8%", transform=ax.transAxes, fontsize=5.4)
ax.set_title("(A)", loc="left", fontsize=7, fontweight="bold", pad=2)

# ---------------------------------------------------------------- B non-APO projection
ax = fig.add_subplot(gs[0, 1])
for key, label, colour in NON_APO:
    p = density(hist(cmd[key][:, 1], cmd[key][:, 2]))
    p = p / p.max()
    ax.contour(XC, YC, p.T, levels=[0.2, 0.5, 0.8], colors=[colour], linewidths=0.6)
base(ax, "B", lim=tight(*[cmd[k] for k, _, _ in NON_APO]))
landmarks(ax, states={"5", "6", "7", "8", "9", "10", "11", "12"}, crystals=("5NTU", "3HH2"))

# ---------------------------------------------------------------- C GaMD non-APO
ax = fig.add_subplot(gs[0, 2])
h = np.zeros_like(hist(cmd["Pre-active-MSTN"][:, 1], cmd["Pre-active-MSTN"][:, 2]))
for key, _, _ in NON_APO:
    h += hist(cmd[key][:, 1], cmd[key][:, 2])
p_cmd = density(h, 1.5)
flat = np.sort(p_cmd.ravel())[::-1]
cut = flat[np.searchsorted(np.cumsum(flat) / flat.sum(), 0.90)]
ax.contourf(XC, YC, p_cmd.T, levels=[cut, p_cmd.max() * 1.01], colors=["#dcdcdc"], alpha=0.85, zorder=0)
for key, label, colour in GAMD_NON_APO:
    p = density(hist(gamd[key][:, 1], gamd[key][:, 2]), 1.5)
    p = p / p.max()
    ax.contour(XC, YC, p.T, levels=[0.2, 0.5], colors=[colour], linewidths=0.6, linestyles="--")
base(ax, "C", "GaMD, non-APO", lim=tight(*([cmd[k] for k, _, _ in NON_APO] + [gamd[k] for k, _, _ in GAMD_NON_APO])))
landmarks(ax, states=set(), crystals=("5NTU", "3HH2"))

# ---------------------------------------------------------------- D APO projection (merged)
ax = fig.add_subplot(gs[0, 3])
for key, label, colour in APO:
    p = density(hist(cmd[key][:, 1], cmd[key][:, 2]))
    p = p / p.max()
    ax.contour(XC, YC, p.T, levels=[0.2, 0.5, 0.8], colors=[colour], linewidths=0.6)
base(ax, "D", "APO systems", lim=tight(*[cmd[k] for k, _, _ in APO]))
landmarks(ax, states={"1", "2", "4"}, crystals=("5JI1",))

# ---------------------------------------------------------------- E GaMD APO
ax = fig.add_subplot(gs[1, 0])
h = np.zeros_like(hist(cmd["APO-MSTN"][:, 1], cmd["APO-MSTN"][:, 2]))
for key, _, _ in APO:
    h += hist(cmd[key][:, 1], cmd[key][:, 2])
p_cmd = density(h, 1.5)
flat = np.sort(p_cmd.ravel())[::-1]
cut = flat[np.searchsorted(np.cumsum(flat) / flat.sum(), 0.90)]
ax.contourf(XC, YC, p_cmd.T, levels=[cut, p_cmd.max() * 1.01], colors=["#dcdcdc"], alpha=0.85, zorder=0)
for key, label, colour in GAMD_APO:
    p = density(hist(gamd[key][:, 1], gamd[key][:, 2]), 1.5)
    p = p / p.max()
    ax.contour(XC, YC, p.T, levels=[0.2, 0.5], colors=[colour], linewidths=0.6, linestyles="--")
base(ax, "E", "GaMD, APO", lim=tight(*([cmd[k] for k, _, _ in APO] + [gamd[k] for k, _, _ in GAMD_APO])))
landmarks(ax, states=set(), crystals=("5JI1",))

# ---------------------------------------------------------------- F-H FELs
FEL_KEYS = ["Pre-active-MSTN", "Traj-MSTN-ActRII", "Active-MSTN"]
FEL_STATES = [{"5", "6", "7", "8"}, {"9", "10", "11", "12"}, set()]
for col, (key, states, letter) in enumerate(zip(FEL_KEYS, FEL_STATES, "FGH")):
    ax = fig.add_subplot(gs[1, col + 1])
    dg = free_energy(hist(cmd[key][:, 1], cmd[key][:, 2]))
    im = ax.pcolormesh(XE, YE, np.ma.masked_invalid(dg.T), cmap=cmap,
                       vmin=0, vmax=DG_VMAX, shading="flat", rasterized=True)
    ax.set_facecolor("#e8e8e8")
    base(ax, letter)
    landmarks(ax, states=states)

cax = fig.add_axes([0.963, 0.135, 0.010, 0.24])
cb = fig.colorbar(im, cax=cax)
cb.set_label("\u0394G (kcal mol$^{-1}$)", fontsize=5.6, labelpad=1)
cb.ax.tick_params(labelsize=5.2)

h1 = [Line2D([], [], color=c, lw=0.8, label=l) for _, l, c in SYSTEMS]
h2 = [Line2D([], [], color=c, lw=0.8, ls="--", label="GaMD " + l) for _, l, c in GAMD]
fig.legend(handles=h1 + h2, loc="upper center", ncol=5, frameon=False, fontsize=5.2,
           bbox_to_anchor=(0.5, 0.995), columnspacing=0.7, handlelength=1.0,
           handletextpad=0.2, labelspacing=0.3)

fig.savefig(os.path.join(OUT, "Fig_S12_pca_final.png"), dpi=600)
fig.savefig(os.path.join(OUT, "Fig_S12_pca_final.pdf"))

from PIL import Image
Image.open(os.path.join(OUT, "Fig_S12_pca_final.png")).convert("RGB").save(
    os.path.join(OUT, "Fig_S12_pca_final.png"), dpi=(600, 600))
print("saved")
