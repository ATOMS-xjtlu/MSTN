"""
Analysis of the eight docked MSTN-ligand complexes over 200 ns unrestrained cMD.

Writes, into delivery_cMD/_analysis/:
  <system>_timeseries.csv          time series at the chosen stride
  <system>_residue_occupancy.csv   contact occupancy per topology residue
  summary.csv / summary.md         statistics over 50-200 ns

Definitions (identical for every system):
  protein           all protein residues (MSTN dimer; ActRII where present)
  alignment         least squares fit on the 218 C-alpha atoms of the MSTN dimer
  ligand            residue name MOL, heavy atoms only
  ligand RMSD       heavy-atom RMSD to the first production frame, after the protein fit,
                    with the ligand re-imaged into the frame of the protein
  COM shift         |ligand COM - COM of the cavity-lining C-alphas| , cavity-lining residues
                    being those within 4.0 A of the ligand in the first frame
  native contacts   heavy-atom pairs < 4.0 A in the first frame; Q = fraction still < 4.0 A
  contacts          any heavy-atom pair < 4.0 A (PBC aware)
  buried fraction   (SASA_lig,free - (SASA_complex - SASA_protein)) / SASA_lig,free

Usage:  python analyze_complex_md.py [stride]     stride in frames, 2 = 100 ps
"""
import os, sys, csv
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from MDAnalysis.lib.mdamath import triclinic_vectors
from sasa import sasa_targets, radii_from_names

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "_analysis")
SYSTEMS = ["state8_CNP0022627_1", "state8_CNP0183427_0", "state8_CNP0604857_3", "state8_CNP0606043_1",
           "state12_CNP0022627_1", "state12_CNP0183427_0", "state12_CNP0604857_3", "state12_CNP0606043_1"]
KEY = {"W295": 295, "Y304": 304, "Y321": 321, "K320": 320, "M350": 350, "P365": 365, "M367": 367}


def top_resid(mstn_num):
    i = mstn_num - 266
    return i, i + 109


def fit(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


def place_near(g, target_com, box):
    best, bestd = g, np.linalg.norm(g.mean(0) - target_com)
    if box is not None and np.isfinite(box).all():
        M = triclinic_vectors(box)
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    cand = g + np.array([i, j, k]) @ M
                    d = np.linalg.norm(cand.mean(0) - target_com)
                    if d < bestd:
                        best, bestd = cand, d
    return best


def main(stride=2):
    os.makedirs(OUT, exist_ok=True)
    summary = []
    for name in SYSTEMS:
        os.chdir(os.path.join(ROOT, name))
        u = mda.Universe("complex.prmtop", "md-200ns.trj", format="NCDF")
        prot = u.select_atoms("protein")
        prot_h = u.select_atoms("protein and not name H*")
        ca = u.select_atoms("protein and name CA and resid 1-218")
        lig = u.select_atoms("resname MOL")
        heavy = ~np.char.startswith(np.asarray(lig.names, dtype=str), "H")
        nframes = len(u.trajectory)
        lig_rad = radii_from_names(lig.names[heavy])
        prot_rad = radii_from_names(prot_h.names)

        u.trajectory[0]
        ref_ca = ca.positions.copy()
        ref_lig = lig.positions.astype(float)[heavy].copy()
        dd0 = distance_array(lig.positions[heavy], prot_h.positions, box=u.dimensions)
        ii, jj = np.where(dd0 < 4.0)
        pocket_resids = sorted(set(int(x) for x in prot_h[np.unique(jj)].resids))
        pocket_ca = u.select_atoms("protein and name CA and resid " + " ".join(map(str, pocket_resids)))
        sasa_free = sasa_targets(lig.positions.astype(float)[heavy], lig_rad)
        nfr = 0

        rows = []
        occ = np.zeros(219)
        occI = {k: 0 for k in KEY}
        nfr = 0
        for fi in range(0, nframes, stride):
            u.trajectory[fi]
            box = u.dimensions
            t_ns = fi * 0.04
            lx = place_near(lig.positions.astype(float), prot_h.positions.mean(0), box)
            R, t = fit(ca.positions, ref_ca)
            lhr = lx[heavy] @ R.T + t
            rmsd = float(np.sqrt(((lhr - ref_lig) ** 2).sum(1).mean()))
            dcom = float(np.linalg.norm(lhr.mean(0) - (pocket_ca.positions.astype(float) @ R.T + t).mean(0)))
            dd = distance_array(lig.positions[heavy], prot_h.positions, box=box)
            ncont = int((dd < 4.0).sum())
            mind = float(dd.min())
            q = float((dd[ii, jj] < 4.0).mean())
            cres = prot_h[np.unique(np.where(dd < 4.0)[1])]
            rs = set(int(x) for x in cres.resids)
            for r in rs:
                occ[r] += 1
            for k, num in KEY.items():
                a, b = top_resid(num)
                if a in rs or b in rs:
                    occI[k] += 1
            bury = float("nan")
            if (fi // stride) % 25 == 0:
                lpos = place_near(lig.positions.astype(float)[heavy], prot_h.positions.mean(0), box)
                ppos = place_near(prot_h.positions.astype(float), lpos.mean(0), box)
                s_com = sasa_targets(lpos, lig_rad, ppos, prot_rad)
                bury = (sasa_free - s_com) / sasa_free
            rows.append((round(t_ns, 2), round(rmsd, 3), round(dcom, 3), round(mind, 3), ncont, round(q, 3),
                         None if np.isnan(bury) else round(bury, 3)))
            nfr += 1
            if fi % 500 == 0:
                print(f"  {name} {t_ns:6.1f} ns  rmsd {rmsd:5.2f}  dcom {dcom:5.2f}  Q {q:4.2f}  ncont {ncont}", flush=True)
        arr = np.array([r[1:6] for r in rows if r[0] >= 50.0], dtype=float)
        bur = np.array([r[6] for r in rows if r[0] >= 50.0 and r[6] is not None], dtype=float)
        st = dict(system=name, state=8 if name.startswith("state8") else 12, ligand=name.split("_", 1)[1],
                  frames=len(rows),
                  lig_rmsd_mean=round(float(arr[:, 0].mean()), 2), lig_rmsd_sd=round(float(arr[:, 0].std(ddof=1)), 2),
                  rmsd_median=round(float(np.median(arr[:, 0])), 2), rmsd_max=round(float(arr[:, 0].max()), 2),
                  com_mean=round(float(arr[:, 1].mean()), 2), com_sd=round(float(arr[:, 1].std(ddof=1)), 2),
                  mind_mean=round(float(arr[:, 2].mean()), 2),
                  ncont_mean=round(float(arr[:, 3].mean()), 1), q_mean=round(float(arr[:, 4].mean()), 3),
                  buried_mean=round(float(bur.mean()), 1) if bur.size else float("nan"),
                  **{f"occ_{k}": round(100.0 * occI[k] / nfr, 1) for k in KEY})
        summary.append(st)
        with open(os.path.join(OUT, f"{name}_timeseries.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["time_ns", "lig_rmsd_A", "com_shift_A", "min_dist_A", "n_contacts", "native_Q", "buried_frac"])
            wr.writerows(rows)
        with open(os.path.join(OUT, f"{name}_residue_occupancy.csv"), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["topology_resid", "mstn_number_chainI", "occupancy_pct"])
            for r in range(1, 219):
                if occ[r]:
                    num = r + 266 if r <= 109 else (r - 109) + 266
                    wr.writerow([r, num, round(100.0 * occ[r] / nfr, 1)])
        print(name, "done", flush=True)

    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        wr.writeheader()
        wr.writerows(summary)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("| system | RMSD (A) | COM shift (A) | min d (A) | contacts | native Q | buried (%) | Y321 | W295 | M367 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for s in summary:
            f.write(f"| {s['system']} | {s['lig_rmsd_mean']} ± {s['lig_rmsd_sd']} | {s['com_mean']} ± {s['com_sd']} "
                    f"| {s['mind_mean']} | {s['ncont_mean']} | {s['q_mean']} | {s['buried_mean']} "
                    f"| {s['occ_Y321']} | {s['occ_W295']} | {s['occ_M367']} |\n")
    print("ALL DONE")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

