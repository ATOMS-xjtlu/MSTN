# -*- coding: utf-8 -*-
"""Count Na+/Cl- ions and box volume for each cMD system, to report the exact
salt concentration achieved in the Setup section."""
import glob
import os
import re

import numpy as np

ROOT = r"C:\path\to\project"
BASE = os.path.join(ROOT, "data", "data", "cMD")
NA = 6.02214076e23

systems = ["Pre-active-MSTN", "APO-MSTN", "APO-MSTN-ActRII",
           "Traj-MSTN", "Traj-MSTN-ActRII", "Active-MSTN"]

print("%-20s %8s %6s %6s %10s %9s" % ("system", "waters", "Na+", "Cl-", "V (nm^3)", "mM"))
for s in systems:
    top = os.path.join(BASE, s, "main", "complex-amber.top")
    pdb = os.path.join(BASE, s, "main", "1us", "md-1usinitial.pdb")
    if not (os.path.exists(top) and os.path.exists(pdb)):
        print("%-20s  missing" % s)
        continue
    txt = open(top, errors="ignore").read()
    m = re.search(r"%FLAG RESIDUE_LABEL\s*\n%FORMAT\(20a4\)\s*\n(.*?)%FLAG", txt, re.S)
    labels = []
    for line in m.group(1).splitlines():
        labels += [line[i:i + 4].strip() for i in range(0, len(line), 4)]
    labels = [l for l in labels if l]
    n_na = sum(1 for l in labels if l in ("Na+", "Na", "SOD"))
    n_cl = sum(1 for l in labels if l in ("Cl-", "Cl", "CLA"))
    n_wat = sum(1 for l in labels if l in ("WAT", "HOH", "SOL"))

    box = None
    for line in open(pdb, errors="ignore"):
        if line.startswith("CRYST1"):
            box = [float(line[6:15]), float(line[15:24]), float(line[24:33])]
            break
    if box is None:
        print("%-20s  no CRYST1" % s)
        continue
    vol_nm3 = box[0] * box[1] * box[2] / 1000.0
    # activity-free estimate: molarity of each ion species
    mM = (n_cl / (NA * vol_nm3 * 1e-24)) * 1000 if vol_nm3 else float("nan")
    print("%-20s %8d %6d %6d %10.1f %9.1f" % (s, n_wat, n_na, n_cl, vol_nm3, mM))
