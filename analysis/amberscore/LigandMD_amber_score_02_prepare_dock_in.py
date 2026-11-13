#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Organise files for DOCK6 Amber Score and generate dock.in

DOCK6 Amber Score file naming convention (following prepare_amber.pl output):
  Receptor:  <rec>.prmtop / <rec>.inpcrd
  Ligand:  <lig>.prmtop / <lig>.inpcrd / <lig>.amber_score.mol2
  Complex: <rec>.<lig>.prmtop / <rec>.<lig>.inpcrd
"""
import os, shutil

ASDIR = '/mnt/c/Users/p2600254/Desktop/project/supplementary/docking/amber_score/systems'
LIGDIR = '/mnt/c/Users/p2600254/Desktop/project/supplementary/docking/md_prep/ligands'

systems = ['state8_CNP0022627_1', 'state8_CNP0183427_0', 'state8_CNP0604857_3', 'state8_CNP0606043_1',
           'state12_CNP0022627_1', 'state12_CNP0183427_0', 'state12_CNP0604857_3', 'state12_CNP0606043_1']

for sys in systems:
    rec, lig = sys.split('_', 1)   # state8 / CNP0022627_1
    d = f'{ASDIR}/{sys}'
    print(f"===== {sys} (rec={rec}, lig={lig}) =====")

    # Receptor
    shutil.copy(f'{d}/receptor.prmtop', f'{d}/{rec}.prmtop')
    shutil.copy(f'{d}/receptor.inpcrd', f'{d}/{rec}.inpcrd')
    # Ligand
    shutil.copy(f'{d}/ligand.prmtop', f'{d}/{lig}.prmtop')
    shutil.copy(f'{d}/ligand.inpcrd', f'{d}/{lig}.inpcrd')
    # Ligand GAFF mol2 (required by DOCK6 Amber Score)
    shutil.copy(f'{LIGDIR}/{sys}_gaff.mol2', f'{d}/{lig}.amber_score.mol2')
    # Complex
    shutil.copy(f'{d}/complex_dry.prmtop', f'{d}/{rec}.{lig}.prmtop')
    shutil.copy(f'{d}/complex_dry.inpcrd', f'{d}/{rec}.{lig}.inpcrd')

    # dock.in (Amber Score options)
    with open(f'{d}/dock.in', 'w') as f:
        f.write(f"""# DOCK6 Amber Score input - {sys}
ligand_atom_file                        {lig}.amber_score.mol2
amber_score_primary                     yes
amber_score_secondary                   no
amber_score_receptor_file_prefix        {rec}
amber_score_movable_region              ligand
amber_score_minimization_rmsgrad        0.01
amber_score_before_md_minimization_cycles  100
amber_score_md_steps                    3000
amber_score_after_md_minimization_cycles   100
amber_score_gb_model                    5
amber_score_nonbonded_cutoff            18.0
amber_score_temperature                 300.0
amber_score_abort_on_unprepped_ligand   yes
""")
    print(f"  generated dock.in and the standardised file names")

print("\nDone")
