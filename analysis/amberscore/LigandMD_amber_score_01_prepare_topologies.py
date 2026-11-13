#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare topologies for Amber Score: strip water and ions from the complex prmtop and split receptor and ligand."""
import parmed as pmd
import os

MDIR   = '/mnt/c/Users/p2600254/Desktop/project/supplementary/docking/md_prep/systems'
ASDIR  = '/mnt/c/Users/p2600254/Desktop/project/supplementary/docking/amber_score/systems'
os.makedirs(ASDIR, exist_ok=True)

systems = ['state8_CNP0022627_1', 'state8_CNP0183427_0', 'state8_CNP0604857_3', 'state8_CNP0606043_1',
           'state12_CNP0022627_1', 'state12_CNP0183427_0', 'state12_CNP0604857_3', 'state12_CNP0606043_1']

for sys in systems:
    src = f'{MDIR}/{sys}'
    dst = f'{ASDIR}/{sys}'
    os.makedirs(dst, exist_ok=True)
    print(f"===== {sys} =====")

    # Read the complex (with crystal waters/solvent)
    p = pmd.load_file(f'{src}/complex.prmtop', f'{src}/complex.inpcrd')

    # Dry complex (protein + ligand)
    dry = p.copy(pmd.Structure)
    dry.strip(':WAT,Na+,Cl-')
    dry.save(f'{dst}/complex_dry.prmtop', overwrite=True)
    dry.save(f'{dst}/complex_dry.inpcrd', overwrite=True)

    # Receptor (protein only; the number of protein residues differs between systems)
    rec = dry.copy(pmd.Structure)
    rec.strip(':MOL')
    rec.save(f'{dst}/receptor.prmtop', overwrite=True)
    rec.save(f'{dst}/receptor.inpcrd', overwrite=True)

    # Ligand
    lig = dry.copy(pmd.Structure)
    lig.strip('!(:MOL)')
    lig.save(f'{dst}/ligand.prmtop', overwrite=True)
    lig.save(f'{dst}/ligand.inpcrd', overwrite=True)

    print(f"  dry: {len(dry.atoms)} atoms / {len(dry.residues)} residues")
    print(f"  rec: {len(rec.atoms)} atoms / {len(rec.residues)} residues")
    print(f"  lig: {len(lig.atoms)} atoms")
print("\nDone")
