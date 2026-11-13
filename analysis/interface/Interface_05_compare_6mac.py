#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSTN-ActRII simulated interface versus the 6MAC crystal interface (GDF-11:ActRII)
==========================================================
Metrics:
  (i) Interface-residue overlap (Jaccard)
  (ii) Contact-pair agreement (intersection of residue-residue contact pairs)
  (iii) Interface RMSD (backbone RMSD of interface residues after superposition on ActRII C-alpha atoms)
  (iv) Interface area DeltaSASA
"""
import os
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances, align
import mdtraj as md

# ============ Paths ============
CIF   = '/mnt/c/Users/p2600254/Desktop/project/supplementary/interface_analysis/main/comparison/6MAC.cif'
BASE  = '/mnt/c/Users/p2600254/Desktop/project/supplementary/interface_analysis/main/main'
PRMTOP= '/home/p2600254/interface_analysis/complex_protein.parm7'
XTRAJ = f'{BASE}/1us/md1us.xtc'
OUT   = '/mnt/c/Users/p2600254/Desktop/project/interface_comparison.txt'
CUT   = 4.5
TMPDIR= '/home/p2600254/interface_analysis/tmp_compare'
os.makedirs(TMPDIR, exist_ok=True)
# =============================

lines = []
def w(s=''):
    lines.append(str(s))

# ---------- 1. Load 6MAC ----------
cry = mda.Universe(CIF)
gdf = cry.select_atoms('segid A')   # GDF-11
act = cry.select_atoms('segid C')   # ActRII
gdf_h = gdf.select_atoms('not name H*')
act_h = act.select_atoms('not name H*')
w("=== 6MAC crystal ===")
w(f"GDF-11(A): {len(gdf.residues)} residues (resid {gdf.residues.resids.min()}-{gdf.residues.resids.max()}), {len(gdf_h)} heavy atoms")
w(f"ActRII(C): {len(act.residues)} residues (resid {act.residues.resids.min()}-{act.residues.resids.max()}), {len(act_h)} heavy atoms")

d = distances.distance_array(gdf_h.positions, act_h.positions)
mask = d < CUT
cry_pairs = set()
for i, j in zip(*np.where(mask)):
    cry_pairs.add((int(gdf_h[i].resid), int(act_h[j].resid)))
w(f"Crystal contacts: {mask.sum()} atom pairs, {len(cry_pairs)} residue pairs")

# ---------- 2. Representative trajectory structure (averaged after alignment on ActRII C-alpha) ----------
u = mda.Universe(PRMTOP, XTRAJ)
mst = u.select_atoms('resid 1:109')       # MSTN
act_sim = u.select_atoms('resid 314:408') # ActRII
ref = u.copy()
aligner = align.AlignTraj(u, ref, select='resid 314:408 and name CA', in_memory=True)
aligner.run()
# Average structure (frame-by-frame accumulation)
avg = np.zeros_like(u.atoms.positions)
for ts in u.trajectory:
    avg += u.atoms.positions.copy()
avg /= len(u.trajectory)
u.trajectory[0]
u.atoms.positions = avg

mst_h = mst.select_atoms('not name H*')
act_sim_h = act_sim.select_atoms('not name H*')
d2 = distances.distance_array(mst_h.positions, act_sim_h.positions)
mask2 = d2 < CUT
sim_pairs = set()
for i, j in zip(*np.where(mask2)):
    sim_pairs.add((int(mst_h[i].resid), int(act_sim_h[j].resid)))
w("\n=== Representative trajectory structure (1-us average, aligned on ActRII) ===")
w(f"Trajectory contacts: {mask2.sum()} atom pairs, {len(sim_pairs)} residue pairs")

# ---------- 3. Residue mapping (from sequence alignment) ----------
# ActRII: simulation 314-408 -> 6MAC chain C resid 26-120 (offset -288)
# MSTN/GDF-11: simulation 1-109 -> 6MAC chain A resid 1-109 (offset 0; resid 1 missing in the crystal)
ACT_OFF = -288
sim_pairs_mapped = {(m, a + ACT_OFF) for (m, a) in sim_pairs}

# ---------- 4. Metric (i): residue overlap ----------
cry_mst = {p[0] for p in cry_pairs}
cry_act = {p[1] for p in cry_pairs}
sim_mst = {p[0] for p in sim_pairs_mapped}
sim_act = {p[1] for p in sim_pairs_mapped}
def jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0
w("\n=== Metric (i): interface-residue overlap (Jaccard) ===")
w(f"GDF-11/MSTN side: crystal {len(cry_mst)} vs trajectory {len(sim_mst)}, intersection {len(cry_mst&sim_mst)}, Jaccard={jac(cry_mst,sim_mst):.2f}")
w(f"ActRII side: crystal {len(cry_act)} vs trajectory {len(sim_act)}, intersection {len(cry_act&sim_act)}, Jaccard={jac(cry_act,sim_act):.2f}")

# ---------- 5. Metric (ii): contact-pair agreement ----------
overlap = cry_pairs & sim_pairs_mapped
w("\n=== Metric (ii): contact-pair agreement ===")
w(f"Crystal contact pairs: {len(cry_pairs)}")
w(f"Trajectory contact pairs (mapped): {len(sim_pairs_mapped)}")
w(f"Shared contact pairs: {len(overlap)}  ({len(overlap)/len(cry_pairs)*100:.0f}% of crystal)")
w("\nShared contact pairs (GDF-11 resid <-> ActRII resid):")
for m, a in sorted(overlap):
    w(f"  GDF-11 {m}  <->  ActRII {a}")
w("\nCrystal-only: " + ", ".join(f"{m}-{a}" for m, a in sorted(cry_pairs - sim_pairs_mapped)))
w("Trajectory-only: " + ", ".join(f"{m}-{a}" for m, a in sorted(sim_pairs_mapped - cry_pairs)))

# ---------- 6. Metric (iii): interface RMSD ----------
act_sim_ca = act_sim.select_atoms('name CA')
act_ca = act.select_atoms('name CA')
assert len(act_sim_ca) == len(act_ca), f"ActRII C-alpha count mismatch {len(act_sim_ca)} vs {len(act_ca)}"

def kabsch(mobile, reference):
    """Least-squares superposition: return R such that reference ~= (mobile - m_center) @ R.T + r_center."""
    m_center = mobile.mean(axis=0)
    r_center = reference.mean(axis=0)
    mc = mobile - m_center
    rc = reference - r_center
    H = mc.T @ rc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, m_center, r_center

def superimpose(coords, R, m_center, r_center):
    return (coords - m_center) @ R.T + r_center

R, mc, rc = kabsch(act_sim_ca.positions, act_ca.positions)
act_rmsd = np.sqrt(((superimpose(act_sim_ca.positions, R, mc, rc) - act_ca.positions)**2).sum(axis=1).mean())
common_res = [r for r in sorted(cry_mst) if 2 <= r <= 109]
sim_coords, cry_coords = [], []
for r in common_res:
    a1 = mst.select_atoms(f'resid {r} and name CA')
    a2 = cry.select_atoms(f'segid A and resid {r} and name CA')
    if len(a1) == 1 and len(a2) == 1:
        sim_coords.append(superimpose(a1.positions, R, mc, rc)[0])
        cry_coords.append(a2.positions[0])
sim_coords = np.array(sim_coords); cry_coords = np.array(cry_coords)
irmsd = np.sqrt(((sim_coords - cry_coords)**2).sum(axis=1).mean())
mst_ca_all = mst.select_atoms('name CA')
gdf_ca_all = cry.select_atoms('segid A and name CA')
n = min(len(mst_ca_all), len(gdf_ca_all))
full_rmsd = np.sqrt(((superimpose(mst_ca_all.positions, R, mc, rc)[:n] - gdf_ca_all.positions[:n])**2).sum(axis=1).mean())
w("\n=== Metric (iii): interface RMSD ===")
w(f"ActRII C-alpha RMSD (after superposition, should be ~0): {act_rmsd:.2f} A")
w(f"GDF-11/MSTN interface-residue C-alpha RMSD: {irmsd:.2f} A  ({len(sim_coords)} residues)")
w(f"Full GDF-11/MSTN C-alpha RMSD (aligned on ActRII): {full_rmsd:.2f} A")

# ---------- 7. Metric (iv): interface area DeltaSASA ----------
def sasa(ag, label):
    pdb = f'{TMPDIR}/{label}.pdb'
    ag.write(pdb)
    t = md.load(pdb)
    return md.shrake_rupley(t, mode='atom').sum() * 100  # nm² -> Å²

w("\n=== Metric (iv): interface area DeltaSASA (shrake_rupley, probe 1.4 A) ===")
# Crystal structure
cry_comp = cry.select_atoms('all')
s_comp = sasa(cry_comp, 'cry_comp')
s_gdf = sasa(gdf, 'cry_gdf')
s_act = sasa(act, 'cry_act')
cry_delta = (s_gdf + s_act - s_comp) / 2
w(f"Crystal: complex={s_comp:.0f}, GDF-11={s_gdf:.0f}, ActRII={s_act:.0f} -> DeltaSASA={cry_delta:.0f} A^2")
# Trajectory
sim_comp = u.select_atoms('resid 1:109 or resid 314:408')
s_comp2 = sasa(sim_comp, 'sim_comp')
s_mst = sasa(mst, 'sim_mst')
s_act2 = sasa(act_sim, 'sim_act')
sim_delta = (s_mst + s_act2 - s_comp2) / 2
w(f"Trajectory: complex={s_comp2:.0f}, MSTN={s_mst:.0f}, ActRII={s_act2:.0f} -> DeltaSASA={sim_delta:.0f} A^2")
w(f"\nInterface-area comparison: crystal {cry_delta:.0f} A^2 vs trajectory {sim_delta:.0f} A^2 (trajectory/crystal = {sim_delta/cry_delta:.2f})")

with open(OUT, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print("\n".join(lines))
print("\nDone. Written to:", OUT)
