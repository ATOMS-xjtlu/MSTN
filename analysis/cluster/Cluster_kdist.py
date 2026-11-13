# -*- coding: utf-8 -*-
"""k-distance analysis of the sampled cMD ensembles (justification for eps and
minPts of the density clustering), plus the cluster-count vs eps sweep.

For every frame the distance to its k-th nearest neighbour (k = minPts) is
computed; the sorted curve is the standard k-distance plot and its elbow region
is the value used as the neighbourhood radius.  Runs on the subsampled
trajectories that were used for the clustering.
"""
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\path\to\project\_libs")
os.chdir(r"C:\path\to\project\response_letter\_build")
import cluster_gromos as CG          # noqa: E402

OUT = os.path.join(CG.OUT)
MINPTS = int(os.environ.get("CL_MINPTS", "25"))
rows = []

for system in CG.CMD:
    X, meta = CG.load_system(system)
    Xa = CG.kabsch(X.astype(np.float64), X[0].astype(np.float64))
    Xf = Xa.reshape(len(Xa), -1)
    g = Xf @ Xf.T
    sq = np.diag(g).copy()
    d2 = (sq[:, None] + sq[None, :] - 2 * g) / (218 * 3)
    np.maximum(d2, 0, out=d2)
    D = np.sqrt(d2)
    np.fill_diagonal(D, np.inf)
    k = min(MINPTS, len(X) - 1)
    kdist = np.sort(np.partition(D, k - 1, axis=1)[:, k - 1])
    np.save(os.path.join(OUT, "%s_kdist.npy" % system), kdist)
    # elbow: largest curvature of the sorted curve (normalised axes)
    xs = np.linspace(0, 1, len(kdist))
    ys = (kdist - kdist.min()) / (kdist.max() - kdist.min() + 1e-9)
    d2y = np.gradient(np.gradient(ys, xs), xs)
    elbow = kdist[int(np.argmax(np.abs(d2y)))]
    for q in (5, 25, 50, 75, 95):
        pass
    rows.append((system, len(X), float(np.percentile(kdist, 95)),
                 float(np.percentile(kdist, 50)), float(elbow)))
    print("%-18s n=%4d  k=%2d  kdist p50=%.2f  p95=%.2f  elbow=%.2f A"
          % (system, len(X), k, rows[-1][3], rows[-1][2], rows[-1][4]), flush=True)

with open(os.path.join(OUT, "kdist_summary.csv"), "w", encoding="utf-8") as fh:
    fh.write("system,frames,kdist_p95,kdist_p50,elbow\n")
    for r in rows:
        fh.write("%s,%d,%.3f,%.3f,%.3f\n" % r)
print("saved", OUT)
