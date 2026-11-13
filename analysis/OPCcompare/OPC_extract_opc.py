"""Extract the three collective variables from the OPC trajectories (1 us per system).

Same definitions as the TIP3P analysis:
  Angle-1 : angle at the midpoint of the two C339 SG atoms, between the two K356 CA atoms
  Distance-1 : P322(CA, chain I) - M367(CA, chain II)
  Distance-2 : A330(CA, chain I) - L286(CA, chain II)
plus Rg of the 218 CA and the CA RMSD to the first production frame.

All geometry is computed from numpy indices, after putting every atom in the image nearest
the protein centre of mass (the protein occupies most of the box, so per-atom wrapping in the
file is not reliable).
"""
import os, sys
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.mdamath import triclinic_vectors

ROOT = r"C:\path\to\project\delivery_cMD_MSTN"
OUT = os.path.join(ROOT, "_analysis")
SYSTEMS = [("preactive_0ns", "preactive_0ns.prmtop", "Pre-active-MSTN"),
           ("traj", "traj.prmtop", "Traj-MSTN-ActRII")]

RES = {"L286": 20, "P322": 56, "A330": 64, "C339": 73, "K356": 90, "M367": 101}


def reimage(xyz, box, ref_com):
    """put every atom in the image nearest ref_com (protein centre of mass)"""
    M = triclinic_vectors(box)
    inv = np.linalg.inv(M)
    frac = xyz @ inv
    fcom = ref_com @ inv
    shift = -np.round(frac - fcom)
    return xyz + shift @ M


def fit(P, Q):
    Pc = P.mean(0); Qc = Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


def main(stride=2):
    os.makedirs(OUT, exist_ok=True)
    for folder, top, label in SYSTEMS:
        os.chdir(os.path.join(ROOT, folder))
        u = mda.Universe(top, ["md500a.trj", "md500b.trj"], format="NCDF")
        n = len(u.trajectory)
        print(f"{label}: {len(u.atoms)} atoms, {n} frames", flush=True)

        def idx(mask):
            a = u.select_atoms(mask)
            assert len(a) == 1, mask
            return a.indices[0]

        i_sg1 = idx(f"resid {RES['C339']} and name SG")
        i_sg2 = idx(f"resid {RES['C339'] + 109} and name SG")
        i_k1 = idx(f"resid {RES['K356']} and name CA")
        i_k2 = idx(f"resid {RES['K356'] + 109} and name CA")
        i_p = idx(f"resid {RES['P322']} and name CA")
        i_m = idx(f"resid {RES['M367'] + 109} and name CA")
        i_a = idx(f"resid {RES['A330']} and name CA")
        i_l = idx(f"resid {RES['L286'] + 109} and name CA")
        ca_idx = u.select_atoms("protein and name CA and resid 1-218").indices
        prot_idx = u.select_atoms("protein").indices

        rows, ref = [], None
        for fi in range(0, n, stride):
            u.trajectory[fi]
            xyz_all = u.atoms.positions.astype(float)
            xyz = reimage(xyz_all, u.dimensions, xyz_all[prot_idx].mean(0))
            apex = 0.5 * (xyz[i_sg1] + xyz[i_sg2])
            v1 = xyz[i_k1] - apex
            v2 = xyz[i_k2] - apex
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            ang = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
            d1 = float(np.linalg.norm(xyz[i_p] - xyz[i_m]))
            d2 = float(np.linalg.norm(xyz[i_a] - xyz[i_l]))
            cas = xyz[ca_idx]
            rg = float(np.sqrt(((cas - cas.mean(0)) ** 2).sum(1).mean()))
            if ref is None:
                ref = cas.copy(); rmsd = 0.0
            else:
                R, t = fit(cas, ref)
                rmsd = float(np.sqrt((((cas @ R.T + t) - ref) ** 2).sum(1).mean()))
            rows.append((fi * 0.05, ang, d1, d2, rg, rmsd))
        out = os.path.join(OUT, f"{label}__OPC.dat")
        with open(out, "w") as f:
            f.write("# time_ns angle1_deg d1_A d2_A rg_A rmsd_ca_A\n")
            f.write(f"# system={label} replica=OPC\n")
            for r in rows:
                f.write("%.4f %.4f %.4f %.4f %.3f %.3f\n" % r)
        arr = np.array(rows)
        print(f"  -> {out}  ({len(rows)} frames)  angle {arr[:,1].mean():.1f}, d1 {arr[:,2].mean():.1f}, "
              f"d2 {arr[:,3].mean():.1f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)

