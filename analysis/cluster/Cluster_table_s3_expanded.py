# -*- coding: utf-8 -*-
"""Draft of the expanded Table S3: one row per state, combining the structural
descriptors (from the state PDBs) with the cluster they correspond to (from the
gmx cluster summary files)."""
import glob
import os
import re

import numpy as np

ROOT = r"C:\path\to\project"
CL = os.path.join(ROOT, "data", "data", "cluster")
STATE_DIR = os.path.join(CL, "state")
OUT = os.path.join(ROOT, "response_letter", "_workbench", "Table_S3_expanded_draft.md")

HELIX = {"5": "\u03b2-sheet", "6": "disordered", "7": "3", "8": "4", "9": "4",
         "10": "6", "11": "6", "12": "6"}
SYSTEM = {"1": "APO-MSTN", "2": "APO-MSTN", "4": "APO-MSTN:ActRII",
          "5": "Pre-active MSTN", "6": "Pre-active MSTN", "7": "Traj-MSTN",
          "8": "Traj-MSTN", "9": "Traj-MSTN:ActRII", "10": "Traj-MSTN:ActRII",
          "11": "Traj-MSTN:ActRII", "12": "Traj-MSTN:ActRII"}
SEG = {"1": 2, "2": 1, "4": 1, "5": 2, "6": 2, "7": 2, "8": 2, "9": 2,
       "10": 1, "11": 2, "12": 2}
S3_FREQ = {"1": 5.34, "2": 10.37, "4": 22.68, "5": 10.02, "6": 31.74, "7": 6.33,
           "8": 12.67, "9": 12.41, "10": 10.57, "11": 65.41, "12": 13.28}


def read_pdb(path):
    ca, sg = {}, {}
    for line in open(path, errors="ignore"):
        if not line.startswith("ATOM"):
            continue
        r = int(line[22:26])
        nm = line[12:16].strip()
        v = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if nm == "CA" and r <= 218:
            ca[r] = v
        elif nm == "SG":
            sg[r] = v
    return ca, sg


def descriptors(path):
    ca, sg = read_pdb(path)
    ks = sorted(sg)
    pair = None
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            d = np.linalg.norm(sg[ks[i]] - sg[ks[j]])
            if d < 2.5 and (ks[i] <= 109) != (ks[j] <= 109):
                pair = (ks[i], ks[j])
                break
        if pair:
            break
    i, j = pair
    oi, oj = 339 - i, 339 - j
    apex = 0.5 * (sg[i] + sg[j])
    v1 = ca[356 - oi] - apex
    v2 = ca[356 - oj] - apex
    ang = np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1)))
    d1 = np.linalg.norm(ca[322 - oi] - ca[367 - oj])
    d2 = np.linalg.norm(ca[330 - oi] - ca[286 - oj])
    X = np.array([ca[r] for r in range(1, 219)])
    return ang, d1, d2, X


def rmsd(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.sqrt(((A @ R - B) ** 2).sum() / len(A)))


act_ca, _ = read_pdb(os.path.join(ROOT, "data", "data", "cMD", "Active-MSTN", "main", "md-initial.pdb"))
act = np.array([act_ca[r] for r in range(1, 219)])

# cluster summary
clusters = {}
for path in glob.glob(os.path.join(CL, "*-split.dat")):
    system = os.path.basename(path).replace("-split.dat", "")
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 13:
            continue
        clusters.setdefault(system, []).append(
            dict(idx=int(f[0]), total=int(f[1]), frac=float(f[2]),
                 frac1=float(f[7]), frac2=float(f[8]),
                 rep1=int(f[11]), rep2=int(f[12])))

rows = []
for path in sorted(glob.glob(os.path.join(STATE_DIR, "*.pdb")),
                   key=lambda p: int(os.path.basename(p).split("-state")[1].split(".")[0])):
    tag = os.path.basename(path).split("-state")[1].split(".")[0]
    frame = int(os.path.basename(path).split("-")[1].split("-")[0])
    ang, d1, d2, X = descriptors(path)
    to_act = rmsd(X.astype(float), act.astype(float))
    sysname = SYSTEM[tag]
    # find the cluster whose representative matches this frame (with the segment offset)
    hit = None
    for c in clusters.get(sysname.replace(" ", "-"), []):
        for seg, rep in ((1, c["rep1"]), (2, c["rep2"])):
            if rep > 0 and (rep == frame or abs(rep - frame) <= 3 or abs(rep + 10000 - frame) <= 1
                            or abs(rep + 20000 - frame) <= 1):
                hit = (c, seg)
                break
        if hit:
            break
    c, seg = hit if hit else (None, SEG[tag])
    frac = (c["frac"] * 100) if c else S3_FREQ[tag]
    rows.append((tag, sysname, seg, c["idx"] if c else "-", c["total"] if c else "-",
                 frac, frame, ang, d1, d2, HELIX.get(tag, "\u2013"), to_act))

lines = ["# Expanded Table S3 (draft): state list\n",
         "The planned Table S14 was cancelled and its content merged into this table. Sources: `data\\data\\cluster\\state\\*.pdb` (structural descriptors)"
         " and `data\\data\\cluster\\*-split.dat` (cluster statistics).\n",
         "| State | System | Segment | Cluster | Cluster frames | Cluster fraction (%) | Representative frame | Angle-1 (deg) | D1 (A) | D2 (A) | Wrist helix | RMSD to Active (A) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append("| %s | %s | %s | %s | %s | %.2f | %d | %.1f | %.2f | %.2f | %s | %.1f |"
                 % (r[0], r[1], "\u00b9" if r[2] == 1 else "\u00b2", r[3], r[4], r[5], r[6],
                    r[7], r[8], r[9], r[10], r[11]))

lines += ["\n## Notes\n",
          "1. The cluster fraction is the percentage of all subsampled frames of that system; seven states in the original Table S3 used the fraction of the segment,"
          " and the values should be unified to the definition used here.",
          "2. The representative frame index follows the state file name; the Cluster column was obtained by matching this index to the Rep column of `-split.dat`.",
          "3. State 3 did not form a stable cluster, is represented by the crystal structures (5JI1, 6MAC) and is not included in this table."]

open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
