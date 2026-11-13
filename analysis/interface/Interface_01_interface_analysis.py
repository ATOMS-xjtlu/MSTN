#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSTN-ActRII interface interaction analysis
==============================
Analyse the interactions between ActRII (residues 1-109) and MSTN monomer 2 (residues 314-408):
  1) heavy-atom contacts (<4.5 A) and their persistence
  2) salt bridges (acid-base distance <4.0 A)
  3) hydrogen bonds (N/O distance <3.5 A, geometric approximation)
  4) list of interface residues

Input: water-stripped prmtop (complex_protein.parm7) + GROMACS xtc trajectory
Output: interface_analysis_results.txt
"""
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances

# ============ Input paths (adjust as needed) ============
BASE   = '/mnt/c/Users/p2600254/Desktop/project/supplementary/interface_analysis/main/main'
PRMTOP = '/home/p2600254/interface_analysis/complex_protein.parm7'  # 6318-atom topology after removing water and ions
OUT    = '/mnt/c/Users/p2600254/Desktop/project/interface_analysis_results.txt'

ACT_SEL = 'resid 1:109'       # ActRII receptor
MST_SEL = 'resid 314:408'     # MSTN monomer 2
CUT_CONTACT = 4.5             # contact distance cutoff (A)
CUT_SALT    = 4.0             # salt-bridge distance cutoff (A)
CUT_HBOND   = 3.5             # hydrogen-bond distance cutoff (A)
# ===========================================

def rkey(atom):
    r = atom.residue
    return (int(r.resid), str(r.resname))

def fmt(k):
    return f"{k[1]}{k[0]}"

lines = []
def w(s=''):
    lines.append(str(s))

for tag in ['1us', '2us']:
    xtc = f'{BASE}/{tag}/md{tag}.xtc'
    u = mda.Universe(PRMTOP, xtc)
    act = u.select_atoms(ACT_SEL)
    mst = u.select_atoms(MST_SEL)
    act_h = act.select_atoms('not name H*')
    mst_h = mst.select_atoms('not name H*')

    n_frames = len(u.trajectory)
    pair_pers, salt_pers, hbond_pers = {}, {}, {}
    n_contacts = []

    acid_sel = act.select_atoms('resname ASP GLU and name OD1 OD2 OE1 OE2')
    base_sel = act.select_atoms('resname LYS ARG HIE HID HIP and name NZ NH1 NH2 ND1 NE2')
    mst_acid = mst.select_atoms('resname ASP GLU and name OD1 OD2 OE1 OE2')
    mst_base = mst.select_atoms('resname LYS ARG HIE HID HIP and name NZ NH1 NH2 ND1 NE2')
    act_polar = act.select_atoms('not name H* and (name N* or name O*)')
    mst_polar = mst.select_atoms('not name H* and (name N* or name O*)')

    for ts in u.trajectory:
        # 1) Heavy-atom contacts
        d = distances.distance_array(act_h.positions, mst_h.positions)
        mask = d < CUT_CONTACT
        n_contacts.append(int(mask.sum()))
        pairs = set()
        for i, j in zip(*np.where(mask)):
            pairs.add((rkey(act_h[i]), rkey(mst_h[j])))
        for p in pairs:
            pair_pers[p] = pair_pers.get(p, 0) + 1

        # 2) Salt bridges (ActRII acidic-MSTN basic and ActRII basic-MSTN acidic)
        if len(acid_sel) and len(mst_base):
            d1 = distances.distance_array(acid_sel.positions, mst_base.positions)
            for i, j in zip(*np.where(d1 < CUT_SALT)):
                k = (rkey(acid_sel[i]), rkey(mst_base[j]))
                salt_pers[k] = salt_pers.get(k, 0) + 1
        if len(base_sel) and len(mst_acid):
            d2 = distances.distance_array(base_sel.positions, mst_acid.positions)
            for i, j in zip(*np.where(d2 < CUT_SALT)):
                k = (rkey(base_sel[i]), rkey(mst_acid[j]))
                salt_pers[k] = salt_pers.get(k, 0) + 1

        # 3) Hydrogen bonds (N/O heavy atoms <3.5 A, approximation)
        if len(act_polar) and len(mst_polar):
            dh = distances.distance_array(act_polar.positions, mst_polar.positions)
            for i, j in zip(*np.where(dh < CUT_HBOND)):
                k = (rkey(act_polar[i]), rkey(mst_polar[j]))
                hbond_pers[k] = hbond_pers.get(k, 0) + 1

    n_contacts = np.array(n_contacts)
    w(f"================ Trajectory {tag} ================")
    w(f"Frames: {n_frames}  Total time: {n_frames*u.trajectory.dt/1000:.0f} ns")
    w(f"Interface contact atom pairs (mean +/- SD): {n_contacts.mean():.1f} +/- {n_contacts.std():.1f}  range {n_contacts.min()}-{n_contacts.max()}")

    pers = sorted([(p, c/n_frames) for p, c in pair_pers.items() if c/n_frames >= 0.5], key=lambda x: -x[1])
    w(f"\n--- Persistent contact residue pairs (>50% of frames): {len(pers)} ---")
    for (ra, rb), frac in pers:
        w(f"  ActRII {fmt(ra)}  <->  MSTN {fmt(rb)}   {frac*100:.0f}%")

    sp = sorted([(p, c/n_frames) for p, c in salt_pers.items() if c/n_frames >= 0.3], key=lambda x: -x[1])
    w(f"\n--- Salt bridges (>30% of frames): {len(sp)} ---")
    for (ra, rb), frac in sp:
        w(f"  {fmt(ra)} (ActRII) <-> {fmt(rb)} (MSTN)   {frac*100:.0f}%")

    hp = sorted([(p, c/n_frames) for p, c in hbond_pers.items() if c/n_frames >= 0.3], key=lambda x: -x[1])
    w(f"\n--- Hydrogen bonds (N/O < 3.5 A, >30% of frames): {len(hp)} ---")
    for (ra, rb), frac in hp:
        w(f"  {fmt(ra)} (ActRII) <-> {fmt(rb)} (MSTN)   {frac*100:.0f}%")

    act_res = sorted({ra for (ra, rb) in pair_pers})
    mst_res = sorted({rb for (ra, rb) in pair_pers})
    w(f"\n--- Interface residues ---")
    w(f"  ActRII interface residues ({len(act_res)}): " + " ".join(fmt(r) for r in act_res))
    w(f"  MSTN interface residues ({len(mst_res)}): " + " ".join(fmt(r) for r in mst_res))
    w("")

with open(OUT, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print("Done. Results written to:", OUT)
