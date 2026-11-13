"""Compare the OPC runs with the three TIP3P replicas of the same system.

For every descriptor (Angle-1, Distance-1, Distance-2):
  * per-replica mean over the first 1 us of the TIP3P runs
  * mu, sigma across the three TIP3P replicas  ->  z = |OPC - mu| / sigma
  * two-sample Kolmogorov-Smirnov and Wasserstein distance, OPC vs each TIP3P replica,
    with the TIP3P replica-to-replica distances as the reference scale
"""
import os, glob
import numpy as np
from scipy import stats

CVS = r"C:\path\to\project\R2-4_delivery\02_cMD_results\cvs"
OPC = r"C:\path\to\project\delivery_cMD_MSTN\_analysis"
OUT = OPC

SYSTEMS = {"Pre-active-MSTN": "Pre-active-MSTN", "Traj-MSTN-ActRII": "Traj-MSTN-ActRII"}
COLS = {"angle1_deg": 1, "d1_A": 2, "d2_A": 3}


def load_tip3p(system, replica, tmax=1000.0):
    p = os.path.join(CVS, f"{system}__{replica}.dat")
    d = np.loadtxt(p, comments="#")
    d = d[d[:, 0] <= tmax]
    return {k: d[:, i] for k, i in COLS.items()}


def load_opc(system):
    p = os.path.join(OPC, f"{system}__OPC.dat")
    d = np.loadtxt(p, comments="#")
    return {k: d[:, i] for k, i in COLS.items()}, d[:, 0]


def main():
    lines = []
    for system in SYSTEMS:
        reps = ["main", "replica1", "replica2"]
        tip = {r: load_tip3p(system, r) for r in reps}
        opc, t = load_opc(system)
        lines.append(f"\n== {system} ==")
        lines.append(f"   OPC sampling: {t.min():.0f}-{t.max():.0f} ns, {len(t)} frames")
        for key in COLS:
            mu_rep = [tip[r][key].mean() for r in reps]
            sd_rep = [tip[r][key].std(ddof=1) for r in reps]
            mu = float(np.mean(mu_rep)); sg = float(np.std(mu_rep, ddof=1))
            opc_m = float(opc[key].mean()); opc_s = float(opc[key].std(ddof=1))
            z = abs(opc_m - mu) / sg if sg > 0 else float("nan")
            # distribution distances
            d_intra, d_cross = [], []
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = tip[reps[i]][key], tip[reps[j]][key]
                    d_intra.append(stats.wasserstein_distance(a, b))
            for r in reps:
                d_cross.append(stats.wasserstein_distance(opc[key], tip[r][key]))
            ks = min(stats.ks_2samp(opc[key], tip[r][key]).pvalue for r in reps)
            lines.append(f"   {key:11s} TIP3P {mu:7.2f} ± {sg:5.2f}  (reps {', '.join(f'{v:.2f}' for v in mu_rep)})"
                         f"   OPC {opc_m:7.2f} ± {opc_s:5.2f}   z = {z:4.2f}")
            lines.append(f"               W1: intra-TIP3P {min(d_intra):.3f}-{max(d_intra):.3f};  "
                         f"OPC-TIP3P {min(d_cross):.3f}-{max(d_cross):.3f};  KS p(min) {ks:.2g}")
        # block stability of the OPC run
        for key in COLS:
            v = opc[key]
            blocks = [v[i:i + 1000].mean() for i in range(0, len(v) - 999, 1000)]
            lines.append(f"   OPC {key:11s} 100-ns blocks: " + ", ".join(f"{b:.1f}" for b in blocks))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "OPC_vs_TIP3P_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
