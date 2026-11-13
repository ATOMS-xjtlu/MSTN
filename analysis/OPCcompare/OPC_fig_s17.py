"""Figure S17: water-model control (OPC vs the three TIP3P replicas)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

D = r"C:\path\to\project\delivery_cMD_MSTN\_analysis"
CVS = r"C:\path\to\project\R2-4_delivery\02_cMD_results\cvs"
OUT = os.path.join(D, "Fig_S17_OPC.png")

SYS = {"Pre-active MSTN": "Pre-active-MSTN", "Traj-MSTN:ActRII": "Traj-MSTN-ActRII"}


def load_tip(system, rep):
    a = np.loadtxt(os.path.join(CVS, f"{system}__{rep}.dat"), comments="#")
    return a[a[:, 0] <= 1000.0]


def load_opc(system):
    return np.loadtxt(os.path.join(D, f"{system}__OPC.dat"), comments="#")


def fel(ax, angle, d1, title):
    ea = np.linspace(30, 175, 49)
    ed = np.linspace(0, 45, 46)
    H, _, _ = np.histogram2d(angle, d1, bins=[ea, ed])
    p = H / H.sum()
    with np.errstate(divide="ignore"):
        g = -np.log(p)
    g[~np.isfinite(g)] = np.nan
    g = g - np.nanmin(g)
    g[g > 3] = np.nan
    im = ax.imshow(g.T, origin="lower", extent=[30, 175, 0, 45], aspect="auto", cmap="viridis", vmin=0, vmax=3)
    ax.set_xlabel("Angle-1 (°)", fontsize=8)
    ax.set_ylabel("Distance-1 (Å)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return im


def bhat(p, q):
    return float(np.sum(np.sqrt(p * q)))


def hist2d(d, ea, ed):
    H, _, _ = np.histogram2d(d[:, 1], d[:, 2], bins=[ea, ed])
    return H / H.sum()


fig = plt.figure(figsize=(11, 6.6))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.3)

# A, B: FELs of the OPC runs
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
op = load_opc("Pre-active-MSTN"); ot = load_opc("Traj-MSTN-ActRII")
im = fel(axA, op[:, 1], op[:, 2], "A  Pre-active MSTN, OPC")
fel(axB, ot[:, 1], ot[:, 2], "B  Traj-MSTN:ActRII, OPC")
cb = fig.colorbar(im, ax=[axA, axB], fraction=0.03, pad=0.02)
cb.set_label("ΔG (kcal mol$^{-1}$)", fontsize=7); cb.ax.tick_params(labelsize=6)

# C: Angle-1 distributions, Pre-active
axC = fig.add_subplot(gs[0, 2])
for i, r in enumerate(("main", "replica1", "replica2")):
    d = load_tip("Pre-active-MSTN", r)
    axC.hist(d[:, 1], bins=np.linspace(30, 175, 60), density=True, histtype="step", lw=0.8,
             color="0.6", label=f"TIP3P rep{i+1}")
axC.hist(op[:, 1], bins=np.linspace(30, 175, 60), density=True, histtype="step", lw=1.6,
         color="tab:blue", label="OPC")
axC.set_xlabel("Angle-1 (°)", fontsize=8); axC.set_ylabel("density", fontsize=8)
axC.set_title("C  Angle-1, Pre-active MSTN", fontsize=9)
axC.legend(fontsize=6, frameon=False); axC.tick_params(labelsize=7)

# D: Distance-2 distributions, Traj
axD = fig.add_subplot(gs[1, 0])
for i, r in enumerate(("main", "replica1", "replica2")):
    d = load_tip("Traj-MSTN-ActRII", r)
    axD.hist(d[:, 3], bins=np.linspace(0, 25, 50), density=True, histtype="step", lw=0.8,
             color="0.6", label=f"TIP3P rep{i+1}")
axD.hist(ot[:, 3], bins=np.linspace(0, 25, 50), density=True, histtype="step", lw=1.6,
         color="tab:blue", label="OPC")
axD.set_xlabel("Distance-2 (Å)", fontsize=8); axD.set_ylabel("density", fontsize=8)
axD.set_title("D  Distance-2, Traj-MSTN:ActRII", fontsize=9)
axD.legend(fontsize=6, frameon=False); axD.tick_params(labelsize=7)

# E: Bhattacharyya overlaps
axE = fig.add_subplot(gs[1, 1])
ea = np.linspace(30, 175, 49); ed = np.linspace(0, 45, 46)
labels, vals, cols = [], [], []
for sysname, tag in (("Pre-active-MSTN", "Pre-active"), ("Traj-MSTN-ActRII", "Traj")):
    Hs = {r: hist2d(load_tip(sysname, r), ea, ed) for r in ("main", "replica1", "replica2")}
    Ho = hist2d(load_opc(sysname), ea, ed)
    intra = [bhat(Hs[a], Hs[b]) for i, a in enumerate(Hs) for b in list(Hs)[i+1:]]
    cross = [bhat(Ho, Hs[r]) for r in Hs]
    labels += [f"{tag}\nTIP3P-TIP3P", f"{tag}\nOPC-TIP3P"]
    vals += [intra, cross]
    cols += ["0.6", "tab:blue"]
xs = [0, 1]
for j, (v, c, lab) in enumerate(zip(vals, ["0.6", "tab:blue"], ["TIP3P-TIP3P", "OPC-TIP3P"])):
    jitter = np.linspace(-0.12, 0.12, len(v))
    axE.scatter(np.full(len(v), xs[j]) + jitter, v, s=26, color=c, zorder=3, label=lab)
    axE.hlines(np.mean(v), xs[j] - 0.18, xs[j] + 0.18, color=c, lw=1.4)
axE.set_xticks(xs)
axE.set_xticklabels(["Pre-active", "Traj:ActRII"], fontsize=7)
axE.set_ylim(0, 1)
axE.set_ylabel("Bhattacharyya overlap", fontsize=8)
axE.set_title("E  Landscape overlap (Angle-1 × D1)", fontsize=9)
axE.tick_params(labelsize=7)
axE.legend(fontsize=6, frameon=False, loc="lower left")

# F: block means of Angle-1
axF = fig.add_subplot(gs[1, 2])
for sysname, tag, mk in (("Pre-active-MSTN", "Pre-active", "o"), ("Traj-MSTN-ActRII", "Traj", "s")):
    d = load_opc(sysname)
    b = [d[(d[:, 0] >= k * 100) & (d[:, 0] < (k + 1) * 100), 1].mean() for k in range(10)]
    axF.plot(np.arange(10) * 100 + 50, b, marker=mk, ms=3, lw=1.2, label=f"{tag}, OPC")
    m = load_tip(sysname, "main")
    bm = [m[(m[:, 0] >= k * 100) & (m[:, 0] < (k + 1) * 100), 1].mean() for k in range(10)]
    axF.plot(np.arange(10) * 100 + 50, bm, ls="--", lw=0.9, color="0.5", label=f"{tag}, TIP3P main")
axF.set_xlabel("time (ns)", fontsize=8); axF.set_ylabel("Angle-1 (°)", fontsize=8)
axF.set_title("F  Block means", fontsize=9)
axF.legend(fontsize=6, frameon=False); axF.tick_params(labelsize=7)

fig.savefig(OUT, dpi=220)
print("saved", OUT)


