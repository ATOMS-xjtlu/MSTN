"""Figure S18: stability of the eight docked complexes over 200 ns of unrestrained cMD."""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats

D = r"C:\path\to\project\delivery_cMD\_analysis"
OUT = os.path.join(D, "Fig_S18_docked_complexes.png")

ORDER = ["state12_CNP0022627_1", "state12_CNP0606043_1", "state12_CNP0183427_0", "state12_CNP0604857_3",
         "state8_CNP0022627_1", "state8_CNP0183427_0", "state8_CNP0604857_3", "state8_CNP0606043_1"]
LBL = {"state12_CNP0022627_1": "L1/12", "state12_CNP0606043_1": "L4/12", "state12_CNP0183427_0": "L2/12",
       "state12_CNP0604857_3": "L3/12", "state8_CNP0022627_1": "L1/8", "state8_CNP0183427_0": "L2/8",
       "state8_CNP0604857_3": "L3/8", "state8_CNP0606043_1": "L4/8"}


def load_ts(name):
    rows = list(csv.DictReader(open(os.path.join(D, f"{name}_timeseries.csv"), encoding="utf-8")))
    t = np.array([float(r["time_ns"]) for r in rows])
    rmsd = np.array([float(r["lig_rmsd_A"]) for r in rows])
    mind = np.array([float(r["min_dist_A"]) for r in rows])
    b = np.array([np.nan if r["buried_frac"] in ("", "None") else float(r["buried_frac"]) for r in rows])
    return t, rmsd, mind, b


def load_occ(name):
    out = {}
    for r in csv.DictReader(open(os.path.join(D, f"{name}_residue_occupancy.csv"), encoding="utf-8")):
        out[int(r["topology_resid"])] = float(r["occupancy_pct"])
    return out


fig = plt.figure(figsize=(11.5, 7.4))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

# A: RMSD traces
axA = fig.add_subplot(gs[0, 0])
axA.axvspan(0, 50, color="0.92", zorder=0)
for name in ORDER:
    t, rmsd, mind, b = load_ts(name)
    c = "tab:blue" if name.startswith("state12") else "tab:red"
    ls = "-" if name.split("_")[1] in ("CNP0022627_1", "CNP0606043_1") else "--"
    axA.plot(t, rmsd, c=c, ls=ls, lw=0.9, label=LBL[name])
axA.set_xlabel("time (ns)", fontsize=8); axA.set_ylabel("ligand RMSD (Å)", fontsize=8)
axA.set_title("A  heavy-atom RMSD to the docked pose", fontsize=9)
axA.legend(fontsize=5.5, ncol=2, frameon=False, loc="upper left")
axA.tick_params(labelsize=7)

# B: contact-occupancy heat map
axB = fig.add_subplot(gs[0, 1])
occ = {n: load_occ(n) for n in ORDER}
resids = sorted({r for n in ORDER for r, v in occ[n].items() if v >= 40.0})
mat = np.array([[occ[n].get(r, 0.0) for r in resids] for n in ORDER])
im = axB.imshow(mat, aspect="auto", cmap="magma_r", vmin=0, vmax=100)
axB.set_yticks(range(len(ORDER)))
axB.set_yticklabels([LBL[n] for n in ORDER], fontsize=7)
labels = []
for r in resids:
    num = r + 266 if r <= 109 else (r - 109) + 266
    labels.append(f"{num} {'I' if r <= 109 else 'II'}")
axB.set_xticks(range(len(resids)))
axB.set_xticklabels(labels, fontsize=5, rotation=90)
axB.set_title("B  contact occupancy, cavity-lining residues (%)", fontsize=9)
cb = fig.colorbar(im, ax=axB, fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=6)
axB.tick_params(axis="x", labelsize=5)

# C: buried fraction and closest approach
axC = fig.add_subplot(gs[1, 0])
bmean, bsd, dmean = [], [], []
for n in ORDER:
    t, rmsd, mind, b = load_ts(n)
    sel = (t >= 50) & np.isfinite(b)
    bmean.append(np.nanmean(b[sel]) * 100); bsd.append(np.nanstd(b[sel]) * 100)
    dmean.append(mind[t >= 50].mean())
x = np.arange(len(ORDER))
axC.bar(x, bmean, yerr=bsd, color=["tab:blue"] * 4 + ["tab:red"] * 4, alpha=0.8, width=0.65)
axC.set_xticks(x); axC.set_xticklabels([LBL[n] for n in ORDER], fontsize=7, rotation=45, ha="right")
axC.set_ylabel("buried SASA (%)", fontsize=8); axC.set_ylim(0, 100)
ax2 = axC.twinx()
ax2.plot(x, dmean, "k^", ms=4, lw=0)
ax2.set_ylabel("closest approach (Å)", fontsize=8); ax2.set_ylim(2.5, 3.5)
axC.set_title("C  burial and closest approach (50–200 ns)", fontsize=9)
axC.tick_params(labelsize=7); ax2.tick_params(labelsize=6)

# D: Vina vs Amber Score
axD = fig.add_subplot(gs[1, 1])
data = [("state12_CNP0022627_1", -28.68, -8.59), ("state12_CNP0606043_1", -25.55, -8.80),
        ("state12_CNP0183427_0", -24.35, -9.02), ("state12_CNP0604857_3", -19.99, -8.87),
        ("state8_CNP0022627_1", -21.38, -8.04), ("state8_CNP0183427_0", -21.81, -8.26),
        ("state8_CNP0604857_3", -16.27, -8.18), ("state8_CNP0606043_1", -17.11, -8.20)]
for name, a, v in data:
    c = "tab:blue" if name.startswith("state12") else "tab:red"
    axD.scatter(v, a, c=c, s=42, zorder=3)
    axD.annotate(LBL[name], (v, a), fontsize=6, xytext=(4, 3), textcoords="offset points")
rho = stats.spearmanr([d[1] for d in data], [d[2] for d in data]).statistic
axD.set_xlabel("Vina score", fontsize=8); axD.set_ylabel("Amber Score (kcal mol$^{-1}$)", fontsize=8)
axD.set_title(f"D  Vina vs Amber Score (Spearman ρ = {rho:.2f})", fontsize=9)
axD.tick_params(labelsize=7)

fig.savefig(OUT, dpi=220)
print("saved", OUT)


