"""pca_extract.py -- common-basis PCA of the MSTN dimer (both chains, C-alpha only).

Why this script exists
----------------------
Reviewer 3.1 asks for a PCA/global conformational-space analysis of the cMD
ensemble, with every system and every replica projected separately onto one
common coordinate basis.  This script:

  1. reads the C-alpha coordinates of the 218-residue MSTN dimer (resid 1:218)
     from every cMD replica and from the four GaMD ensembles;
  2. removes global translation/rotation by least-squares superposition onto a
     single common reference structure (first frame of Pre-active MSTN main),
     refined once against the ensemble mean;
  3. builds ONE covariance matrix from a frame-balanced pool of all six cMD
     systems (equal number of frames per system, so the 0.04 ns Pre-active
     trajectories do not dominate), diagonalises it, and fixes the signs of
     PC1/PC2;
  4. projects every cMD replica and every GaMD trajectory onto the resulting
     eigenvectors, and stores the projections together with the three
     manuscript collective variables (Angle-1, Distance-1, Distance-2) matched
     frame by frame in time.

The eigenvectors are computed from cMD only.  GaMD is projected a posteriori
onto that cMD basis, so the basis is independent of the boosted ensemble.

MDAnalysis' XDR reader cannot open paths containing non-ASCII characters, so
the trajectories are addressed through ASCII junctions created in %TEMP%
(pca_cmd -> supplementary/cMD/cMD, pca_gamd -> GaMD/GaMD).

Outputs (in _work/pca/):
    model.npz                    reference/mean structures, eigenvectors,
                                 eigenvalues and variance ratios
    proj_<system>__<replica>.dat time, PC1..PC10, Angle-1, D1, D2, segment
    proj_gamd_<system>.dat       same columns for the GaMD ensembles
    extract_log.json             frame counts, strides, timings
"""

import json
import os
import sys
import time

import numpy as np
import MDAnalysis as mda

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT = os.path.join(PROJ, '_work', 'pca')

TEMP = os.environ.get('TEMP', '/tmp')
CMD_ROOT = os.environ.get('PCA_CMD_ROOT', os.path.join(TEMP, 'pca_cmd'))
GAMD_ROOT = os.environ.get('PCA_GAMD_ROOT', os.path.join(TEMP, 'pca_gamd'))
CV_ROOT = os.path.join(PROJ, '_work', 'cvs')

MSTN_NRES = 218          # residues 1-218 = the two MSTN chains
N_PC = 10
FRAMES_PER_REPLICA = 2000
FRAMES_PER_SYSTEM_PCA = 1000
FRAMES_GAMD = 3000

# ---------------------------------------------------------------------------
# manifest: system -> replica -> [(segment dir, topology, trajectory), ...]
# The only topology that does not match its own trajectory is
# Traj-MSTN-ActRII/replica1/2us (PDB carries water, XTC does not); the solute
# PDB of the same replica is used there instead.
# ---------------------------------------------------------------------------
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

GAMD = {
    'Pre-active-MSTN': [('Pre-active-MSTN', 'md-1usinitial.pdb', 'full1us.xtc'),
                        ('Pre-active-MSTN', 'md-2usinitial.pdb', 'full2us.xtc')],
    'Pre-active-MSTN-ActRII': [('Pre-active-MSTN-ActRII', 'md-initial.pdb', 'md-nW1000.xtc')],
    'APO-MSTN': [('APO-MSTN', 'md-initial.pdb', 'md-full-1us.xtc')],
    'APO-MSTN-ActRII': [('APO-MSTN-ActRII', 'md-initial.pdb', 'md-full1us.xtc')],
}

SYSTEM_ORDER = ['Pre-active-MSTN', 'Traj-MSTN', 'Traj-MSTN-ActRII',
                'APO-MSTN', 'APO-MSTN-ActRII', 'Active-MSTN']
REPLICAS = ['main', 'replica1', 'replica2']


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def read_ca(pairs, stride):
    """Return (frames[N,218,3], times[N], segments[N]) for one replica.

    Each segment is loaded with its own topology because the atom count differs
    between segments of a few systems (some continuation runs carry water).
    Segment 0 is loaded first and its CA order is used as the reference order;
    every later segment is checked against it.
    """
    frames, times, segs = [], [], []
    order = None
    for k, (top, trj) in enumerate(pairs):
        u = mda.Universe(top, trj)
        ca = u.select_atoms('name CA and resid 1:%d' % MSTN_NRES)
        if len(ca) != MSTN_NRES:
            raise SystemExit('expected %d CA, got %d (%s)' % (MSTN_NRES, len(ca), top))
        key = (tuple(ca.resids), tuple(ca.names))
        if order is None:
            order = key
        elif key != order:
            raise SystemExit('CA order differs between segments (%s)' % top)
        idx = ca.indices
        n = len(u.trajectory)
        for i in range(0, n, stride):
            u.trajectory[i]
            frames.append(u.atoms.positions[idx].astype(np.float32))
            times.append(u.trajectory.time / 1000.0)
            segs.append(k)
    return np.asarray(frames), np.asarray(times), np.asarray(segs)


def segment_pairs(root, segs):
    return [(os.path.join(root, d, top), os.path.join(root, d, trj))
            for d, top, trj in segs]


def count_frames(pairs):
    return sum(len(mda.Universe(top, trj).trajectory) for top, trj in pairs)


def kabsch(frames, ref):
    """Superpose every frame onto ref (least squares over all CA)."""
    ref = np.asarray(ref, np.float64)
    q = ref - ref.mean(axis=0)
    P = frames.astype(np.float64) - frames.mean(axis=1, keepdims=True)
    H = np.einsum('tni,nj->tij', P, q)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum('tij,tjk->tik', Vt.transpose(0, 2, 1),
                                        U.transpose(0, 2, 1))))
    D = np.zeros((len(frames), 3, 3))
    D[:, 0, 0] = 1.0
    D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = Vt.transpose(0, 2, 1) @ D @ U.transpose(0, 2, 1)
    return np.einsum('tni,tji->tnj', P, R) + ref.mean(axis=0)


def load_cv(system, replica):
    """Existing Angle-1 / D1 / D2 time series (validated in R2-4)."""
    path = os.path.join(CV_ROOT, '%s__%s.dat' % (system, replica))
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, comments='#', ndmin=2)


def load_gamd_cv(system):
    path = os.path.join(PROJ, '_work', 'gamd_cvs', '%s.dat' % system)
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, comments='#', ndmin=2)


def match_cv(cv, times):
    """Nearest-time match of the CV table for each PCA frame; NaN if >2 ns away."""
    out = np.full((len(times), 3), np.nan)
    if cv is None or len(cv) == 0:
        return out
    t = cv[:, 0]
    j = np.searchsorted(t, times)
    for i, (tt, jj) in enumerate(zip(times, j)):
        cand = [k for k in (jj - 1, jj) if 0 <= k < len(t)]
        k = min(cand, key=lambda k: abs(t[k] - tt))
        if abs(t[k] - tt) <= 2.0:
            out[i] = cv[k, 1:4]
    return out


def find_stride(n_frames, target):
    return max(1, int(round(n_frames / float(target))))


def write_table(path, times, pc, cv, segs):
    arr = np.column_stack([times, pc, cv, segs])
    np.savetxt(path, arr, fmt='%.4f',
               header='time_ns ' + ' '.join('PC%d' % (i + 1) for i in range(pc.shape[1]))
                      + ' angle1_deg d1_A d2_A segment')


# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    log = {'cmd': {}, 'gamd': {}}

    # ---- 1. cMD coordinates ------------------------------------------------
    data = {}
    for system in SYSTEM_ORDER:
        for rep in REPLICAS:
            pairs = segment_pairs(CMD_ROOT, CMD[system][rep])
            n_total = count_frames(pairs)
            stride = find_stride(n_total, FRAMES_PER_REPLICA)
            t0 = time.time()
            X, T, S = read_ca(pairs, stride)
            data[(system, rep)] = dict(X=X, T=T, S=S)
            log['cmd']['%s__%s' % (system, rep)] = dict(
                stride=stride, frames=int(len(X)), n_total=int(n_total),
                seconds=round(time.time() - t0, 1))
            print('%-20s %-9s stride=%-3d frames=%-5d (of %d) in %.1fs'
                  % (system, rep, stride, len(X), n_total, time.time() - t0))

    # ---- 2. common reference: first frame of Pre-active main ---------------
    ref0 = data[('Pre-active-MSTN', 'main')]['X'][0]
    keys = list(data)
    for k in keys:
        data[k]['X'] = kabsch(data[k]['X'], ref0)
    mean = np.mean(np.concatenate([data[k]['X'] for k in keys]), axis=0)
    for k in keys:
        data[k]['X'] = kabsch(data[k]['X'], mean)
    mean = np.mean(np.concatenate([data[k]['X'] for k in keys]), axis=0)
    print('reference refined; mean Rg = %.2f A'
          % np.sqrt(((mean - mean.mean(0)) ** 2).sum(1).mean()))

    # ---- 3. covariance from a frame-balanced pool of all cMD systems -------
    rng = np.random.default_rng(20261029)
    pool = []
    for system in SYSTEM_ORDER:
        per_rep = FRAMES_PER_SYSTEM_PCA // 3
        chosen = []
        for rep in REPLICAS:
            X = data[(system, rep)]['X']
            idx = np.sort(rng.choice(len(X), size=min(per_rep, len(X)), replace=False))
            chosen.append(X[idx])
        pool.append(np.concatenate(chosen))
    pool = np.concatenate(pool).astype(np.float64)       # (N, 218, 3)
    Xf = (pool - mean).reshape(len(pool), -1)
    cov = np.cov(Xf.T)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    evr = w / w.sum()
    print('variance explained: ' + ', '.join('PC%d %.1f%%' % (i + 1, 100 * evr[i])
                                             for i in range(6)))

    # ---- 4. sign conventions ------------------------------------------------
    # PC1 increases with the inter-monomer separation (Distance-1);
    # PC2 increases with Angle-1.  Correlations are taken over all cMD frames.
    cv_all = np.concatenate([match_cv(load_cv(s, r), data[(s, r)]['T'])
                             for s in SYSTEM_ORDER for r in REPLICAS])
    pc_all = np.concatenate([(data[(s, r)]['X'] - mean).reshape(len(data[(s, r)]['X']), -1)
                             @ V[:, :N_PC] for s in SYSTEM_ORDER for r in REPLICAS])
    ok = np.isfinite(cv_all[:, 1])
    r1 = np.corrcoef(pc_all[ok, 0], cv_all[ok, 1])[0, 1]
    r2 = np.corrcoef(pc_all[ok, 1], cv_all[ok, 0])[0, 1]
    if r1 < 0:
        V[:, 0] *= -1
    if r2 < 0:
        V[:, 1] *= -1
    print('sign check: corr(PC1,D1)=%.2f  corr(PC2,Angle-1)=%.2f' % (r1, r2))

    # ---- 5. project every cMD replica, attach CVs --------------------------
    for system in SYSTEM_ORDER:
        for rep in REPLICAS:
            d = data[(system, rep)]
            A = (d['X'].astype(np.float64) - mean).reshape(len(d['X']), -1)
            pc = A @ V[:, :N_PC]
            cv = match_cv(load_cv(system, rep), d['T'])
            write_table(os.path.join(OUT, 'proj_%s__%s.dat' % (system, rep)),
                        d['T'], pc, cv, d['S'])
            d['pc'] = pc
            d['cv'] = cv
            d['X'] = None
            print('%-20s %-9s projected: PC1 %7.1f .. %7.1f  PC2 %7.1f .. %7.1f'
                  % (system, rep, pc[:, 0].min(), pc[:, 0].max(),
                     pc[:, 1].min(), pc[:, 1].max()))

    # ---- 6. GaMD projected a posteriori onto the cMD basis ----------------
    for system, segs in GAMD.items():
        pairs = segment_pairs(GAMD_ROOT, segs)
        n_total = count_frames(pairs)
        stride = find_stride(n_total, FRAMES_GAMD)
        X, T, S = read_ca(pairs, stride)
        A = (kabsch(X, mean) - mean).reshape(len(X), -1)
        pc = A @ V[:, :N_PC]
        cv = match_cv(load_gamd_cv(system), T)
        write_table(os.path.join(OUT, 'proj_gamd_%s.dat' % system), T, pc, cv, S)
        log['gamd'][system] = dict(stride=stride, frames=int(len(X)),
                                   n_total=int(n_total))
        print('GaMD %-22s stride=%-3d frames=%-5d (of %d) PC1 %7.1f .. %7.1f'
              % (system, stride, len(X), n_total, pc[:, 0].min(), pc[:, 0].max()))

    np.savez_compressed(os.path.join(OUT, 'model.npz'),
                        mean=mean, ref=ref0, components=V[:, :N_PC].T,
                        eigenvalues=w, evr=evr)
    log['variance_pct'] = (100 * evr[:N_PC]).round(3).tolist()
    log['seconds'] = round(time.time() - t_start, 1)
    with open(os.path.join(OUT, 'extract_log.json'), 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print('done in %.1fs' % (time.time() - t_start))


if __name__ == '__main__':
    main()
