#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xtc -> Amber mdcrd trajectory conversion (for MM-GBSA)
============================================
1. Load the water- and ion-stripped prmtop (6318 atoms) with the xtc trajectory
2. Select the ActRII (1-109) and MSTN (314-408) chains (3159 atoms in total)
3. Take every 25th frame (100 frames in total) and write an Amber ASCII mdcrd

Key format: Amber mdcrd stores 10 coordinate values per line (%8.3f), not 10 atoms (30 values).
"""
import numpy as np
import MDAnalysis as mda

BASE   = '/mnt/c/Users/p2600254/Desktop/project/supplementary/interface_analysis/main/main'
PRMTOP = '/home/p2600254/interface_analysis/complex_protein.parm7'  # water- and ion-stripped topology
WORK   = '/home/p2600254/interface_analysis'

def write_mdcrd(path, u, sel, stride=25):
    n = sel.n_atoms
    frames = list(range(0, len(u.trajectory), stride))
    with open(path, 'w') as f:
        for idx in frames:
            u.trajectory[idx]
            flat = sel.positions.reshape(-1)          # (3n,)
            for i in range(0, 3*n, 10):               # 10 values per line
                line = ''.join(f'{v:8.3f}' for v in flat[i:i+10])
                f.write(line + '\n')
    return len(frames)

for tag in ['1us', '2us']:
    xtc = f'{BASE}/{tag}/md{tag}.xtc'
    u = mda.Universe(PRMTOP, xtc)
    sel = u.select_atoms('resid 1:109 or resid 314:408')
    mdcrd = f'{WORK}/traj_{tag}.mdcrd'
    nf = write_mdcrd(mdcrd, u, sel)
    print(f"{tag}: {sel.n_atoms} atoms -> {mdcrd}, {nf} frames")
print("Done")
