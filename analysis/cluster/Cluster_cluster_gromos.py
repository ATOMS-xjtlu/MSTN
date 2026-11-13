# -*- coding: utf-8 -*-
"""GROMOS clustering (gmx cluster -method gromos) of the MSTN cMD ensembles.

Protocol: C-alpha atoms of the 218-residue MSTN dimer, every frame superposed on
a common reference, RMSD cut-off 1.5 A, every 50th saved frame (2 ns).

MDAnalysis' XDR reader cannot open paths containing non-ASCII characters, so the
process changes into each segment directory and opens the files by bare name.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\path\to\project\_libs")
import MDAnalysis as mda          # noqa: E402

ROOT = r"C:\path\to\project"
CMD_ROOT = os.path.join(ROOT, "supplementary", "cMD", "cMD")
OUT = os.path.join(ROOT, "response_letter", "_workbench", "clustering")
os.makedirs(OUT, exist_ok=True)

MSTN_NRES = 218
STRIDE = int(os.environ.get("CL_STRIDE", "50"))
CUTOFF = float(os.environ.get("CL_CUTOFF", "1.5"))
MIN_SIZE = 25

CMD = {
    'Pre-active-MSTN': {
        'main': [('Pre-active-MSTN/main/1us', 'md-1usinitial.pdb', 'md-nW1000.xtc'),
                 ('Pre-active-MSTN/main/2us', 'md-2usinitial.pdb', 'md-nW2000.xtc')],
        'replica1': [('Pre-active-MSTN/replica1/1us', 'md-1usinitial.pdb', 'md-nW1us.xtc'),
                     ('Pre-active-MSTN/replica1/2us', 'md-2usinitial.pdb', 'md-nW2us.xtc')],
        'replica2': [('Pre-active-MSTN/replica2/1us', 'md-1usinitial.pdb', 'md-nW1000.xtc'),
                     ('Pre-active-MSTN/replica2/2us', 'md-2usinitial.pdb', 'md-nW2000.xtc')],
    },
    'Traj-MSTN': {
        'main': [('Traj-MSTN/main/1us', 'md-1usinitial.pdb', 'md-nW1000.xtc'),
                 ('Traj-MSTN/main/2us', 'md-2usinitial.pdb', 'md-nW2000.xtc')],
        'replica1': [('Traj-MSTN/replica1/1us', 'md-1usinitial.pdb', 'md-nW1us.xtc'),
                     ('Traj-MSTN/replica1/2us', 'md-2usinitial.pdb', 'md-nW2000.xtc')],
        'replica2': [('Traj-MSTN/replica2/1us', 'md-1usinitial.pdb', 'md-nW1us.xtc'),
                     ('Traj-MSTN/replica2/2us', 'md-2usinitial.pdb', 'md-nW2000.xtc')],
    },
    'Traj-MSTN-ActRII': {
        'main': [('Traj-MSTN-ActRII/main/1us', 'md-1usinitial.pdb', 'nW1000.xtc'),
                 ('Traj-MSTN-ActRII/main/2us', 'md-2usinitial.pdb', 'md-nW2000.xtc')],
        'replica1': [('Traj-MSTN-ActRII/replica1/1us', 'md-1usinitial.pdb', 'md-nW1000.xtc'),
                     ('Traj-MSTN-ActRII/replica1/2us', '../1us/md-1usinitial.pdb', 'md-nW2000.xtc')],
        'replica2': [('Traj-MSTN-ActRII/replica2/1us', 'md-1usinitial.pdb', 'md-nW1us.xtc'),
                     ('Traj-MSTN-ActRII/replica2/2us', 'md-2usinitial.pdb', 'md-nW2000.xtc')],
    },
    'APO-MSTN': {
        'main': [('APO-MSTN/main/1us', 'md-1usinitial.pdb', 'md-nW1000.xtc'),
                 ('APO-MSTN/main/2us', 'md-2usinitial.pdb', 'md-nW2us.xtc')],
        'replica1': [('APO-MSTN/replica1/1us', 'md-1usinitial.pdb', 'md-nW1us-extract.xtc'),
                     ('APO-MSTN/replica1/2us', 'md-2usinitial.pdb', 'md-nW-2us-extract.xtc')],
        'replica2': [('APO-MSTN/replica2/1us', 'md-1usinitial.pdb', 'md-nW1us-extract.xtc'),
                     ('APO-MSTN/replica2/2us', 'md-2usinitial.pdb', 'md-nW2us-extract.xtc')],
    },
    'APO-MSTN-ActRII': {
        'main': [('APO-MSTN-ActRII/main/1us', 'md-1usinitial.pdb', 'md1us.xtc'),
                 ('APO-MSTN-ActRII/main/2us', 'md-2usinitial.pdb', 'md2us.xtc')],
        'replica1': [('APO-MSTN-ActRII/replica1/1us', 'md-1usinitial.pdb', 'md1us.xtc'),
                     ('APO-MSTN-ActRII/replica1/2us', 'md-2usinitial.pdb', 'md-nW2000.xtc')],
        'replica2': [('APO-MSTN-ActRII/replica2/1us', 'md-1usinitial.pdb', 'md-nW1us.xtc'),
                     ('APO-MSTN-ActRII/replica2/2us', 'md-2usinitial.pdb', 'md-nW2us.xtc')],
    },
    'Active-MSTN': {
        'main': [('Active-MSTN/main', 'md-initial.pdb', 'md-nW1000.xtc')],
        'replica1': [('Active-MSTN/replica1', 'md-initial.pdb', 'md-nW1000.xtc')],
        'replica2': [('Active-MSTN/replica2', 'md-initial.pdb', 'md-nW1000.xtc')],
    },
}
REPLICAS = ["main", "replica1", "replica2"]


def load_system(system):
    frames, meta = [], []
    for rep in REPLICAS:
        for seg_idx, (d, top, trj) in enumerate(CMD[system][rep]):
            seg_dir = os.path.normpath(os.path.join(CMD_ROOT, d))
            top_path = os.path.normpath(os.path.join(seg_dir, top))
            trj_path = os.path.normpath(os.path.join(seg_dir, trj))
            os.chdir(seg_dir)
            top_arg = os.path.relpath(top_path, seg_dir)      # e.g. ..\1us\md-1usinitial.pdb
            u = mda.Universe(top_arg, os.path.basename(trj_path))
            ca = u.select_atoms("name CA and resid 1:%d" % MSTN_NRES)
            if len(ca) != MSTN_NRES:
                raise SystemExit("expected %d CA, got %d (%s)" % (MSTN_NRES, len(ca), d))
            idx = ca.indices
            n = len(u.trajectory)
            for i in range(0, n, STRIDE):
                u.trajectory[i]
                frames.append(u.atoms.positions[idx].astype(np.float32))
                meta.append((rep, seg_idx, i, u.trajectory.time / 1000.0))
    return np.asarray(frames, dtype=np.float32), meta


def kabsch(X, ref):
    Xc = X - X.mean(1, keepdims=True)
    Rc = ref - ref.mean(0)
    H = np.einsum("nki,kj->ij", Xc, Rc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return np.einsum("nki,ij->nkj", Xc, R)


def gromos(D, cutoff):
    n = D.shape[0]
    labels = -np.ones(n, dtype=int)
    centres = []
    for i in range(n):
        if centres:
            d = D[i, centres]
            k = int(np.argmin(d))
            if d[k] < cutoff:
                labels[i] = k
                members = np.where(labels == k)[0]
                sub = D[np.ix_(members, members)]
                centres[k] = int(members[int(np.argmin(sub.mean(1)))])
                continue
        labels[i] = len(centres)
        centres.append(i)
    return labels, np.array(centres)


def analyse(system):
    t0 = time.time()
    X, meta = load_system(system)
    Xa = kabsch(X.astype(np.float64), X[0].astype(np.float64))
    Xf = Xa.reshape(len(Xa), -1)
    g = Xf @ Xf.T
    sq = np.diag(g).copy()
    d2 = (sq[:, None] + sq[None, :] - 2 * g) / (MSTN_NRES * 3)
    np.maximum(d2, 0, out=d2)
    D = np.sqrt(d2).astype(np.float32)
    print("  %s: %d frames, matrix %.0f MB, %.0fs"
          % (system, len(X), D.nbytes / 1e6, time.time() - t0), flush=True)

    sweep = {}
    for cut in (1.0, 1.25, 1.5, 2.0):
        lab, cen = gromos(D, cut)
        sizes = np.bincount(lab)
        sweep[cut] = (len(sizes), int((sizes >= MIN_SIZE).sum()),
                      float(100.0 * sizes[sizes >= MIN_SIZE].sum() / len(lab)))

    labels, centres = gromos(D, CUTOFF)
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    rows = []
    for rank, k in enumerate(order, 1):
        members = np.where(labels == k)[0]
        med = centres[k]
        sub = D[np.ix_(members, members)]
        rows.append((rank, int(k), len(members), 100.0 * len(members) / len(labels),
                     int(med), meta[med][0], round(meta[med][3], 1),
                     float(sub.mean()), float(sub.max())))
    with open(os.path.join(OUT, "%s_clusters.csv" % system), "w", encoding="utf-8") as fh:
        fh.write("rank,cluster,size,pct,medoid_index,replica,time_ns,mean_rmsd,max_rmsd\n")
        for r in rows:
            fh.write("%d,%d,%d,%.2f,%d,%s,%.1f,%.3f,%.3f\n" % r)
    np.savez_compressed(os.path.join(OUT, "%s_frames.npz" % system),
                        labels=labels, centres=centres,
                        time=np.array([m[3] for m in meta]),
                        replica=np.array([m[0] for m in meta]))
    print("  clusters=%d  >=%d structures: %d (%.1f%%)  total %.0fs"
          % (len(sizes), MIN_SIZE, sweep[CUTOFF][1], sweep[CUTOFF][2], time.time() - t0),
          flush=True)
    for r in rows[:6]:
        print("   #%-2d n=%-4d %5.1f%%  medoid %s %7.1f ns  <RMSD> %.2f"
              % (r[0], r[2], r[3], r[5], r[6], r[7]), flush=True)
    return dict(system=system, n=len(labels), clusters=len(sizes), rows=rows, sweep=sweep)


if __name__ == "__main__":
    only = sys.argv[1:] or list(CMD)
    summary = []
    for system in only:
        summary.append(analyse(system))
    with open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8") as fh:
        for s in summary:
            fh.write("%s: %d frames, %d clusters\n" % (s["system"], s["n"], s["clusters"]))
            for cut, (nc, nmaj, pct) in sorted(s["sweep"].items()):
                fh.write("   cutoff %.2f A -> %d clusters, %d with >=%d structures (%.1f%% of frames)\n"
                         % (cut, nc, nmaj, MIN_SIZE, pct))
            for r in s["rows"][:12]:
                fh.write("   #%d n=%d (%.1f%%) medoid %s %.1f ns meanRMSD %.2f\n"
                         % (r[0], r[2], r[3], r[5], r[6], r[7]))
    print("done")
