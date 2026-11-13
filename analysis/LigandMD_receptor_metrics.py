"""Receptor-side metrics for the eight docked complexes: C-alpha RMSD of the MSTN dimer
and per-residue C-alpha RMSF of the cavity-lining residues (50-200 ns)."""
import os, sys, csv
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

ROOT = r"C:\path\to\project\delivery_cMD"
OUT = os.path.join(ROOT, "_analysis")
SYSTEMS = ["state8_CNP0022627_1", "state8_CNP0183427_0", "state8_CNP0604857_3", "state8_CNP0606043_1",
           "state12_CNP0022627_1", "state12_CNP0183427_0", "state12_CNP0604857_3", "state12_CNP0606043_1"]


def qcp_fit(P, Q):
    """returns rotation-free RMSD after optimal superposition (QCP-style via SVD)"""
    Pc = P.mean(0); Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    P2 = (P - Pc) @ R.T
    return P2, (Q - Qc), R


rows = []
for name in SYSTEMS:
    os.chdir(os.path.join(ROOT, name))
    u = mda.Universe("complex.prmtop", "md-200ns.trj", format="NCDF")
    ca = u.select_atoms("protein and name CA and resid 1-218")
    prot_h = u.select_atoms("protein and not name H*")
    lig = u.select_atoms("resname MOL")
    u.trajectory[0]
    ref = ca.positions.astype(float).copy()
    dd0 = distance_array(lig.positions, prot_h.positions, box=u.dimensions)
    pocket_res = sorted(set(int(x) for x in prot_h[np.unique(np.where(dd0 < 4.0)[1])].resids))
    pmask = np.array([r.resid in pocket_res for r in ca.residues])
    rmsd = []
    traj_frames = []
    n = len(u.trajectory)
    for fi in range(1000, n, 25):          # 40 ns onwards, every 1 ns
        u.trajectory[fi]
        P = ca.positions.astype(float)
        P2, Q2, R = qcp_fit(P, ref)
        rmsd.append(float(np.sqrt(((P2 - Q2) ** 2).sum(1).mean())))
        traj_frames.append(P2 - Q2)
    A = np.array(traj_frames)
    rmsf = np.sqrt((A ** 2).mean(0).sum(1))   # per-atom fluctuation after fitting to ref
    rows.append(dict(system=name, ca_rmsd_mean=round(float(np.mean(rmsd)), 2),
                     ca_rmsd_sd=round(float(np.std(rmsd, ddof=1)), 2),
                     pocket_rmsf_mean=round(float(rmsf[pmask].mean()), 2),
                     pocket_rmsf_max=round(float(rmsf[pmask].max()), 2),
                     global_rmsf_mean=round(float(rmsf.mean()), 2)))
    print(rows[-1], flush=True)

with open(os.path.join(OUT, "receptor_metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("DONE")

