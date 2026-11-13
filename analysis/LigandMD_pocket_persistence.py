"""Pocket persistence, measured independently of the ligand.

Probe points = the heavy atoms of the docked ligand in the first frame (they mark the volume
the predicted region had to provide).  At later frames the protein is superposed on the
218 C-alpha atoms and, for every probe point, we ask

  * is the point still free of protein (no heavy atom within 2.5 A)?   -> the cavity has not collapsed
  * how many water oxygens lie within 3.0 A of the point?             -> the cavity is open to solvent

Usage:  python pocket_persistence.py
"""
import os, csv
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

ROOT = r"C:\path\to\project\delivery_cMD"
OUT = os.path.join(ROOT, "_analysis")
SYSTEMS = ["state8_CNP0022627_1", "state8_CNP0183427_0", "state8_CNP0604857_3", "state8_CNP0606043_1",
           "state12_CNP0022627_1", "state12_CNP0183427_0", "state12_CNP0604857_3", "state12_CNP0606043_1"]
FRAMES = [1250, 2500, 3750, 4999]


def fit(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


rows = []
for name in SYSTEMS:
    os.chdir(os.path.join(ROOT, name))
    u = mda.Universe("complex.prmtop", "md-200ns.trj", format="NCDF")
    ca = u.select_atoms("protein and name CA and resid 1-218")
    prot_h = u.select_atoms("protein and not name H*")
    lig = u.select_atoms("resname MOL")
    heavy = ~np.char.startswith(np.asarray(lig.names, dtype=str), "H")
    wat = u.select_atoms("resname WAT and name O")
    u.trajectory[0]
    ref_ca = ca.positions.copy()
    probes = lig.positions[heavy].astype(float).copy()
    rec = {"system": name}
    for fi in FRAMES:
        u.trajectory[fi]
        R, t = fit(ca.positions, ref_ca)
        prot = prot_h.positions.astype(float) @ R.T + t
        wat_xyz = wat.positions.astype(float) @ R.T + t
        # minimum distance from every probe point to any protein heavy atom
        d_prot = distance_array(probes, prot).min(axis=1)
        free = float((d_prot > 2.5).mean())
        # minimum distance from every probe point to any water oxygen
        d_wat = distance_array(probes, wat_xyz).min(axis=1)
        nw = float(np.mean(d_wat < 3.0) * len(probes))
        rec[f"{fi*0.04:.0f}ns"] = (round(100 * free, 0), round(nw, 1), round(float(d_prot.min()), 2))
    rows.append(rec)
    print(name, {k: v for k, v in rec.items() if k != "system"}, flush=True)

with open(os.path.join(OUT, "pocket_persistence.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("DONE  (values: % of docked volume still free of protein | mean number of waters per probe point | closest protein approach)")
