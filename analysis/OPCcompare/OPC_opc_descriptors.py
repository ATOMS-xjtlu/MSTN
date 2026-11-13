"""OPC vs TIP3P: Angle-1, wrist distances, wrist-helix content (DSSP), inter-chain contacts,
C-alpha RMSD and Rg for the two systems that were re-simulated in OPC water.

OPC   : delivery_cMD_MSTN/ preactive_0ns, traj   (2 x 500 ns, 50 ps/frame)
TIP3P : supplementary/cMD/cMD/{Pre-active-MSTN, Traj-MSTN-ActRII}/{main,replica1,replica2}/1us
        (protein-only xtc, 40 ps/frame)

Wrist helix: DSSP states H/G/I anywhere in P322-A330 (resid 56-64) of chain I and the
corresponding segment of chain II (resid 165-173).
"""
import os, glob, sys
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dssp import DSSP
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.lib.mdamath import triclinic_vectors

ROOT = r"C:\path\to\project"
OUT = os.path.join(ROOT, "delivery_cMD_MSTN", "_analysis")
WRIST_I = list(range(56, 65))        # P322-A330
WRIST_II = list(range(165, 174))
CONTACT_RES = {"W295": 29, "Y304": 38, "Y321": 55, "M350": 84}


def reimage(xyz, box, ref_com):
    M = triclinic_vectors(box)
    inv = np.linalg.inv(M)
    frac = xyz @ inv
    shift = -np.round(frac - (ref_com @ inv))
    return xyz + shift @ M


def fit(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


def analyse(universe, protein, stride, label, boxed):
    """returns dict of per-frame series"""
    u = universe
    ca = u.select_atoms("protein and name CA and resid 1-218")
    prot_idx = u.select_atoms("protein").indices
    sg1 = u.select_atoms("resid 73 and name SG").indices[0]
    sg2 = u.select_atoms("resid 182 and name SG").indices[0]
    k1 = u.select_atoms("resid 90 and name CA").indices[0]
    k2 = u.select_atoms("resid 199 and name CA").indices[0]
    p_ = u.select_atoms("resid 56 and name CA").indices[0]
    m_ = u.select_atoms("resid 210 and name CA").indices[0]
    a_ = u.select_atoms("resid 64 and name CA").indices[0]
    l_ = u.select_atoms("resid 129 and name CA").indices[0]
    ch2 = u.select_atoms("resid 110-218 and not name H*")
    res_atoms = {k: u.select_atoms(f"resid {v} and not name H*") for k, v in CONTACT_RES.items()}

    n = len(u.trajectory)
    ref = None
    out = {k: [] for k in ("t", "angle", "d1", "d2", "rg", "rmsd",
                           "occ_W295", "occ_Y304", "occ_Y321", "occ_M350")}
    for fi in range(0, n, stride):
        u.trajectory[fi]
        xyz = u.atoms.positions.astype(float)
        if boxed:
            xyz = reimage(xyz, u.dimensions, xyz[prot_idx].mean(0))
        apex = 0.5 * (xyz[sg1] + xyz[sg2])
        v1, v2 = xyz[k1] - apex, xyz[k2] - apex
        ang = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))
        cas = xyz[ca.indices]
        out["t"].append(fi * 0.05 if boxed else fi * 0.04)
        out["angle"].append(ang)
        out["d1"].append(float(np.linalg.norm(xyz[p_] - xyz[m_])))
        out["d2"].append(float(np.linalg.norm(xyz[a_] - xyz[l_])))
        out["rg"].append(float(np.sqrt(((cas - cas.mean(0)) ** 2).sum(1).mean())))
        if ref is None:
            ref = cas.copy(); out["rmsd"].append(0.0)
        else:
            R, t = fit(cas, ref)
            out["rmsd"].append(float(np.sqrt((((cas @ R.T + t) - ref) ** 2).sum(1).mean())))
        c2 = u.atoms.positions[ch2.indices].astype(float)
        for k, ag in res_atoms.items():
            a1 = u.atoms.positions[ag.indices].astype(float)
            if boxed:
                a1 = reimage(a1, u.dimensions, xyz[prot_idx].mean(0))
                c2b = reimage(c2, u.dimensions, xyz[prot_idx].mean(0))
            else:
                c2b = c2
            out[f"occ_{k}"].append(float(distance_array(a1, c2b).min() < 4.5))
    return {k: np.array(v) for k, v in out.items()}


def helix_series(universe, protein, stride):
    """DSSP over the trajectory; returns per-frame helix counts and per-residue occupancy"""
    u2 = mda.Merge(protein)
    u2.trajectory = universe.trajectory
    res = DSSP(u2).run(step=stride)
    codes = res.results.dssp            # (frames, n_res)
    resids = u2.residues.resids
    idxI = [np.where(resids == r)[0][0] for r in WRIST_I]
    idxII = [np.where(resids == r)[0][0] for r in WRIST_II]
    helical = np.isin(codes, list("HGI"))
    nI = helical[:, idxI].sum(1)
    nII = helical[:, idxII].sum(1)
    return codes, nI, nII, idxI, idxII


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    # OPC
    for folder, top, label in (("preactive_0ns", "preactive_0ns.prmtop", "Pre-active-MSTN"),
                               ("traj", "traj.prmtop", "Traj-MSTN-ActRII")):
        jobs.append((os.path.join(ROOT, "delivery_cMD_MSTN", folder), top,
                     ["md500a.trj", "md500b.trj"], "NCDF", 20, f"{label}__OPC", True))
    # TIP3P
    for system in ("Pre-active-MSTN", "Traj-MSTN-ActRII"):
        base = os.path.join(ROOT, "supplementary", "cMD", "cMD", system)
        for rep in ("main", "replica1", "replica2"):
            d = os.path.join(base, rep, "1us")
            xtcs = [f for f in os.listdir(d) if f.endswith(".xtc")]
            jobs.append((d, "md-1usinitial.pdb", xtcs[0], "XTC", 25, f"{system}__{rep}", False))

    summary = []
    for wd, top, trj, fmt, stride, label, boxed in jobs:
        os.chdir(wd)
        u = mda.Universe(top, trj if isinstance(trj, list) else trj, format=fmt) if not isinstance(trj, list) \
            else mda.Universe(top, trj, format=fmt)
        prot = u.select_atoms("protein")
        met = analyse(u, prot, stride, label, boxed)
        codes, nI, nII, idxI, idxII = helix_series(u, prot, stride)
        row = dict(label=label,
                   n_frames=len(met["t"]),
                   angle_mean=float(met["angle"].mean()), angle_sd=float(met["angle"].std(ddof=1)),
                   d1_mean=float(met["d1"].mean()), d1_sd=float(met["d1"].std(ddof=1)),
                   d2_mean=float(met["d2"].mean()), d2_sd=float(met["d2"].std(ddof=1)),
                   rg_mean=float(met["rg"].mean()),
                   rmsd_mean=float(met["rmsd"].mean()),
                   helix_I_mean=float(nI.mean()), helix_II_mean=float(nII.mean()),
                   helix_I_ge6=float((nI >= 6).mean() * 100), helix_II_ge6=float((nII >= 6).mean() * 100),
                   helix_I_ge4=float((nI >= 4).mean() * 100), helix_II_ge4=float((nII >= 4).mean() * 100),
                   occ_W295=float(met["occ_W295"].mean() * 100), occ_Y304=float(met["occ_Y304"].mean() * 100),
                   occ_Y321=float(met["occ_Y321"].mean() * 100), occ_M350=float(met["occ_M350"].mean() * 100))
        summary.append(row)
        print(f"{label:32s} angle {row['angle_mean']:6.1f} d1 {row['d1_mean']:5.1f} d2 {row['d2_mean']:5.1f} "
              f"helixI {row['helix_I_mean']:4.1f} helixII {row['helix_II_mean']:4.1f} "
              f"occ {row['occ_W295']:.0f}/{row['occ_Y304']:.0f}/{row['occ_Y321']:.0f}/{row['occ_M350']:.0f} "
              f"rmsd {row['rmsd_mean']:4.1f}", flush=True)

    import csv
    keys = list(summary[0].keys())
    with open(os.path.join(OUT, "OPC_TIP3P_descriptors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
    print("saved", os.path.join(OUT, "OPC_TIP3P_descriptors.csv"))


if __name__ == "__main__":
    main()
