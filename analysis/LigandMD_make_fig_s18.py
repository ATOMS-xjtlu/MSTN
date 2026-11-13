"""Figure S18 (draft): stability of the eight docked complexes over 200 ns."""
import os, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

D = r"C:\path\to\project\delivery_cMD\_analysis"
OUT = os.path.join(D, "Fig_S18_draft.png")

ORDER = ["state12_CNP0022627_1", "state12_CNP0606043_1", "state12_CNP0183427_0", "state12_CNP0604857_3",
         "state8_CNP0022627_1", "state8_CNP0183427_0", "state8_CNP0604857_3", "state8_CNP0606043_1"]
LBL = {"state12_CNP0022627_1": "L1 / state 12", "state12_CNP0606043_1": "L4 / state 12",
       "state12_CNP0183427_0": "L2 / state 12", "state12_CNP0604857_3": "L3 / state 12",
       "state8_CNP0022627_1": "L1 / state 8", "state8_CNP0183427_0": "L2 / state 8",
       "state8_CNP0604857_3": "L3 / state 8", "state8_CNP0606043_1": "L4 / state 8"}


def load(name):
    t, r, q, b, m = [], [], [], [], []
    with open(os.path.join(D, f"{name}_timeseries.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t.append(float(row["time_ns"])); r.append(float(row["lig_rmsd_A"]))
            q.append(float(row["native_Q"])); m.append(float(row["min_dist_A"]))
            b.append(np.nan if row["buried_frac"] in ("", "None") else float(row["buried_frac"]))
    return map(np.array, (t, r, q, m, np.array(b, float)))


fig, ax = plt.subplots(2, 2, figsize=(11, 8))

t, r, q, m, b = load("state12_CNP0022627_1")
ax[0, 0].axvspan(0, 50, color="0.9", zorder=0)
ax[0, 1].axvspan(0, 50, color="0.9", zorder=0)
ax[1, 0].axvspan(0, 50, color="0.9", zorder=0)
for name in ORDER:
    t, r, q, m, b = load(name)
    c = "tab:blue" if name.startswith("state12") else "tab:red"
    ls = "-" if name.endswith(("0022627_1", "0606043_1")) else "--"
    ax[0, 0].plot(t, r, c=c, ls=ls, lw=1.0, label=LBL[name])
    ax[0, 1].plot(t, q, c=c, ls=ls, lw=1.0)
ax[0, 0].set_ylabel("ligand RMSD (Å)")
ax[0, 0].set_xlabel("time (ns)")
ax[0, 0].legend(fontsize=6, ncol=2, frameon=False)
ax[0, 0].set_title("A  heavy-atom RMSD after fit on 218 Cα")
ax[0, 1].set_ylabel("native contact fraction Q")
ax[0, 1].set_xlabel("time (ns)")
ax[0, 1].set_title("B  retention of the docked contact set")

# panel C: RMSD vs Q scatter over 50-200 ns
for name in ORDER:
    t, r, q, m, b = load(name)
    sel = t >= 50
    c = "tab:blue" if name.startswith("state12") else "tab:red"
    ax[1, 0].scatter(r[sel][::10], q[sel][::10], s=3, c=c, alpha=0.4)
ax[1, 0].set_xlabel("ligand RMSD (Å)")
ax[1, 0].set_ylabel("native contact fraction Q")
ax[1, 0].set_title("C  states separate: state 12 (blue) vs state 8 (red)")

# panel D: Vina vs Amber Score
data = [("state12_CNP0022627_1", -28.68, -8.59), ("state12_CNP0606043_1", -25.55, -8.80),
        ("state12_CNP0183427_0", -24.35, -9.02), ("state12_CNP0604857_3", -19.99, -8.87),
        ("state8_CNP0022627_1", -21.38, -8.04), ("state8_CNP0183427_0", -21.81, -8.26),
        ("state8_CNP0604857_3", -16.27, -8.18), ("state8_CNP0606043_1", -17.11, -8.20)]
for name, a, v in data:
    c = "tab:blue" if name.startswith("state12") else "tab:red"
    ax[1, 1].scatter(v, a, c=c, s=40)
    ax[1, 1].annotate(name.split("_")[1], (v, a), fontsize=6, xytext=(3, 3), textcoords="offset points")
rho = stats.spearmanr([x[1] for x in data], [x[2] for x in data])
ax[1, 1].set_xlabel("Vina score")
ax[1, 1].set_ylabel("Amber Score (kcal mol$^{-1}$)")
ax[1, 1].set_title(f"D  Vina vs Amber Score (Spearman ρ = {rho.statistic:.2f})")
for a_ in ax.ravel():
    a_.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("saved", OUT)
