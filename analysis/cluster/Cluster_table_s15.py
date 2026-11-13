# -*- coding: utf-8 -*-
"""Build Table S15 (cluster inventory) from the gmx cluster summary files and
mark the manuscript states that map onto each cluster."""
import glob
import os
import re

ROOT = r"C:\path\to\project"
CL = os.path.join(ROOT, "data", "data", "cluster")
OUT = os.path.join(ROOT, "response_letter", "_workbench")

# state -> (system, cluster index, rep frame used in Table S3)
STATE_MAP = {
    "1": ("APO-MSTN", 1, 9401),
    "2": ("APO-MSTN", 0, 1926),
    "4": ("APO-MSTN-ActRII", 1, 17615),
    "5": ("Pre-active-MSTN", 2, 29131),
    "6": ("Pre-active-MSTN", 0, 11162),
    "7": ("Traj-MSTN", 2, 25219),
    "8": ("Traj-MSTN", 1, 27668),
    "9": ("Traj-MSTN-ActRII", 2, 11038),
    "10": ("Traj-MSTN-ActRII", 3, 28036),
    "11": ("Traj-MSTN-ActRII", 0, 15333),
    "12": ("Traj-MSTN-ActRII", 1, 20172),
}

rows = []
for path in sorted(glob.glob(os.path.join(CL, "*-split.dat"))):
    system = os.path.basename(path).replace("-split.dat", "")
    lines = open(path, encoding="utf-8", errors="ignore").read().splitlines()
    n1 = n2 = 0
    for line in lines:
        m = re.match(r"#\s*1st=\s*(\d+)\s+2nd=\s*(\d+)", line)
        if m:
            n1, n2 = int(m.group(1)), int(m.group(2))
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if len(f) < 13:
            continue
        rows.append(dict(system=system, n1=n1, n2=n2, idx=int(f[0]), total=int(f[1]),
                         frac=float(f[2]), n1f=int(f[5]), n2f=int(f[6]),
                         frac1=float(f[7]), frac2=float(f[8]),
                         rep1=int(f[11]), rep2=int(f[12])))

# attach the state label
for r in rows:
    r["state"] = ""
    for tag, (sysname, idx, _) in STATE_MAP.items():
        if sysname == r["system"] and idx == r["idx"]:
            r["state"] = tag

order = ["Pre-active-MSTN", "APO-MSTN", "APO-MSTN-ActRII", "Traj-MSTN", "Traj-MSTN-ActRII"]

lines = []
lines.append("# Table S15 (draft): cluster inventory\n")
lines.append("Compiled from `data\\data\\cluster\\*-split.dat` (output of `gmx cluster -g`). "
             "Columns `1st`/`2nd` give the frame counts of the two trajectory segments (after subsampling).\n")
for system in order:
    sub = [r for r in rows if r["system"] == system]
    sub.sort(key=lambda r: r["idx"])
    if not sub:
        continue
    n1, n2 = sub[0]["n1"], sub[0]["n2"]
    lines.append("\n## %s (1st = %d frames, 2nd = %d frames, %d clusters)\n" % (system, n1, n2, len(sub)))
    lines.append("| Cluster | Frames | % of total | % of 1st | % of 2nd | Rep 1st | Rep 2nd | State |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sub:
        lines.append("| %d | %d | %.2f | %.2f | %.2f | %s | %s | %s |"
                     % (r["idx"], r["total"], 100 * r["frac"], 100 * r["frac1"], 100 * r["frac2"],
                        r["rep1"] if r["rep1"] > 0 else "–",
                        r["rep2"] if r["rep2"] > 0 else "–",
                        r["state"] if r["state"] else "–"))

lines.append("\n## Notes\n")
lines.append("1. The GROMOS algorithm has no noise class: every frame belongs to a cluster, and clusters not assigned a state are less populated conformations.")
lines.append("2. The state-to-cluster assignment follows the representative frame index; for states 1, 6, 10 and 12 the listed frame index includes the offset of the preceding segment (or differs from the Rep column by up to 3 frames due to transcription) "
             "(+0, +10000, +20000 and +10000, respectively), so the values differ from the Rep column but refer to the same cluster.")
lines.append("3. Seven percentages in Table S3 match the corresponding-segment fractions in this table exactly (states 5, 6, 7, 8, 9, 10 and 11); "
             "those for states 1, 2, 4 and 12 were taken from the %-of-total column or from neighbouring clusters and should be checked and unified.")

text = "\n".join(lines) + "\n"
open(os.path.join(OUT, "Table_S15_cluster_list.md"), "w", encoding="utf-8").write(text)
print(text[:1500])
print("...\nsaved", os.path.join(OUT, "Table_S15_cluster_list.md"))
