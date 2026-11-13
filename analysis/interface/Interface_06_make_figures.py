#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate comparison figures: side-by-side contact maps and an interface-overlay PDB."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import distances, align

CIF   = '/mnt/c/Users/p2600254/Desktop/project/supplementary/interface_analysis/main/comparison/6MAC.cif'
BASE  = '/mnt/c/Users/p2600254/Desktop/project/supplementary/interface_analysis/main/main'
PRMTOP= '/home/p2600254/interface_analysis/complex_protein.parm7'
XTRAJ = f'{BASE}/1us/md1us.xtc'
OUTDIR= '/mnt/c/Users/p2600254/Desktop/project/interface_delivery/results'
CUT   = 4.5

# Load crystal
cry = mda.Universe(CIF)
gdf = cry.select_atoms('segid A'); act = cry.select_atoms('segid C')
gdf_h = gdf.select_atoms('not name H*'); act_h = act.select_atoms('not name H*')
d = distances.distance_array(gdf_h.positions, act_h.positions)
cry_pairs = {(int(gdf_h[i].resid), int(act_h[j].resid)) for i, j in zip(*np.where(d < CUT))}

# Load representative trajectory structure
u = mda.Universe(PRMTOP, XTRAJ)
mst = u.select_atoms('resid 1:109'); act_sim = u.select_atoms('resid 314:408')
ref = u.copy()
align.AlignTraj(u, ref, select='resid 314:408 and name CA', in_memory=True).run()
avg = np.zeros_like(u.atoms.positions)
for ts in u.trajectory:
    avg += u.atoms.positions.copy()
avg /= len(u.trajectory)
u.trajectory[0]; u.atoms.positions = avg

mst_h = mst.select_atoms('not name H*'); act_sim_h = act_sim.select_atoms('not name H*')
d2 = distances.distance_array(mst_h.positions, act_sim_h.positions)
sim_pairs = {(int(mst_h[i].resid), int(act_sim_h[j].resid) - 288) for i, j in zip(*np.where(d2 < CUT))}

# Superposition
act_sim_ca = act_sim.select_atoms('name CA'); act_ca = act.select_atoms('name CA')
def kabsch(mob, ref):
    mc_, rc_ = mob.mean(0), ref.mean(0)
    H = (mob - mc_).T @ (ref - rc_)
    U, S, Vt = np.linalg.svd(H)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1,1,dd]) @ U.T, mc_, rc_
R, mc, rc = kabsch(act_sim_ca.positions, act_ca.positions)
def sup(coords):
    return (coords - mc) @ R.T + rc

# ===== 1. Side-by-side contact maps =====
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, pairs, title, color in [
    (axes[0], cry_pairs, 'Crystal 6MAC (GDF-11 : ActRII)', 'tab:red'),
    (axes[1], sim_pairs, 'Simulation (MSTN : ActRII, 1μs avg)', 'tab:blue'),
]:
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    ax.scatter(xs, ys, s=45, c=color, alpha=0.8)
    ax.set_xlabel('GDF-11 / MSTN residue'); ax.set_ylabel('ActRII residue')
    ax.set_title(f'{title}\n({len(pairs)} contact pairs)')
    ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUTDIR}/contact_map_comparison.png', dpi=150)
print("Contact map saved")

# ===== 2. Interface-overlay PDB =====
# Write the superimposed trajectory MSTN+ActRII to PDB (chains B=D), crystal GDF-11+ActRII (chains A=C)
with open(f'{OUTDIR}/interface_overlay.pdb', 'w') as f:
    # Crystal structure
    for a in cry.atoms:
        x, y, z = a.position
        f.write(f"ATOM  {a.index+1:5d} {a.name:>4s} {a.resname:>3s} {a.segid:1s}{a.resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}\n")
    f.write("TER\n")
    # Trajectory (after superposition), chains B (MSTN) and D (ActRII)
    for a in mst.atoms:
        p = sup(a.position[None,:])[0]
        f.write(f"ATOM  {a.index+1:5d} {a.name:>4s} {a.resname:>3s} B{a.resid:4d}    "
                f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n")
    f.write("TER\n")
    for a in act_sim.atoms:
        p = sup(a.position[None,:])[0]
        f.write(f"ATOM  {a.index+1:5d} {a.name:>4s} {a.resname:>3s} D{a.resid:4d}    "
                f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n")
    f.write("END\n")
print("Interface-overlay PDB saved")
