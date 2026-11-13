"""extract_cvs.py -- extract Angle-1, Distance-1, Distance-2 from a trajectory.

Definitions follow the manuscript:
    Angle-1     apex = midpoint of the C339-C339 interchain disulfide (SG-SG);
                endpoints = CA of K356 in chain I and chain II
    Distance-1  CA(P322, chain I)  - CA(M367, chain II)
    Distance-2  CA(A330, chain I)  - CA(L286, chain II)

Residue indices are not hard-coded: topologies number the GF domain 1..109 (chain I)
and 110..218 (chain II), and any receptor follows.  The interchain C339-C339
disulfide is located geometrically (single SG-SG contact across the chain-I/II
boundary), which fixes both chain offsets.

Output columns: time_ns  angle1_deg  d1_A  d2_A
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_angles

MSTN_NRES = 218
CHAIN_I_END = 109
MANUSCRIPT_CYS = 339


def stage(path, tag=''):
    """Copy a file into an ASCII temp dir (the XDR reader cannot open non-ASCII paths)."""
    root = os.path.join(tempfile.gettempdir(), 'mdstage')
    os.makedirs(root, exist_ok=True)
    d = os.path.join(root, tag.replace(' ', '_') or 'stage')
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(path).encode('ascii', 'replace').decode('ascii'))
    shutil.copyfile(path, dst)
    return dst, d


def resolve(u, log=print):
    """Return a dict of atom indices for the three collective variables."""
    sg = u.select_atoms('resname CYX and name SG and resid 1:%d' % MSTN_NRES)
    resids = np.array([int(a.resid) for a in sg])
    pos = sg.positions
    pair = None
    for i in range(len(resids)):
        for j in range(i + 1, len(resids)):
            if (resids[i] <= CHAIN_I_END) == (resids[j] <= CHAIN_I_END):
                continue
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 2.5:
                a, b = (resids[i], resids[j]) if resids[i] <= CHAIN_I_END else (resids[j], resids[i])
                pair = (a, b, d)
    if pair is None:
        raise SystemExit('interchain C339-C339 disulfide not found')
    a, b, d = pair
    off_i = MANUSCRIPT_CYS - a
    off_j = MANUSCRIPT_CYS - b
    log('disulfide: chain I resid %d <-> chain II resid %d (%.2f A); offsets +%d / +%d'
        % (a, b, d, off_i, off_j))
    idx = dict(
        k356_i=356 - off_i, k356_ii=356 - off_j,
        c339_i=a, c339_ii=b,
        p322_i=322 - off_i, m367_ii=367 - off_j,
        a330_i=330 - off_i, l286_ii=286 - off_j,
    )

    def one(sel, what):
        ag = u.select_atoms(sel)
        if len(ag) != 1:
            raise SystemExit('%s: expected 1 atom, got %d (%s)' % (what, len(ag), sel))
        return ag[0].index

    out = dict(
        ca_k356_i=one('name CA and resid %d' % idx['k356_i'], 'K356 chain I'),
        ca_k356_ii=one('name CA and resid %d' % idx['k356_ii'], 'K356 chain II'),
        sg_c339_i=one('resname CYX and name SG and resid %d' % idx['c339_i'], 'C339 SG chain I'),
        sg_c339_ii=one('resname CYX and name SG and resid %d' % idx['c339_ii'], 'C339 SG chain II'),
        ca_p322=one('name CA and resid %d' % idx['p322_i'], 'P322 chain I'),
        ca_m367=one('name CA and resid %d' % idx['m367_ii'], 'M367 chain II'),
        ca_a330=one('name CA and resid %d' % idx['a330_i'], 'A330 chain I'),
        ca_l286=one('name CA and resid %d' % idx['l286_ii'], 'L286 chain II'),
    )
    log('residues used: K356 %d/%d  C339 %d/%d  P322 %d  M367 %d  A330 %d  L286 %d'
        % (idx['k356_i'], idx['k356_ii'], idx['c339_i'], idx['c339_ii'],
           idx['p322_i'], idx['m367_ii'], idx['a330_i'], idx['l286_ii']))
    return out


def extract(top, traj, out, max_frames=None, tag='job', log=print):
    top_c, dtop = stage(top, tag + '_top')
    traj_c, dtraj = stage(traj, tag + '_traj')
    try:
        u = mda.Universe(top_c, traj_c)
        log('atoms=%d frames=%d' % (len(u.atoms), len(u.trajectory)))
        ix = resolve(u, log)
        rows = []
        for i, ts in enumerate(u.trajectory):
            p = u.atoms.positions
            apex = (p[ix['sg_c339_i']] + p[ix['sg_c339_ii']]) / 2.0
            v1 = p[ix['ca_k356_i']] - apex
            v2 = p[ix['ca_k356_ii']] - apex
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            d1 = np.linalg.norm(p[ix['ca_p322']] - p[ix['ca_m367']])
            d2 = np.linalg.norm(p[ix['ca_a330']] - p[ix['ca_l286']])
            rows.append((ts.time / 1000.0, ang, d1, d2))
            if max_frames and i + 1 >= max_frames:
                break
        arr = np.asarray(rows, dtype=float)
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        np.savetxt(out, arr, fmt='%.4f', header='time_ns angle1_deg d1_A d2_A')
        log('wrote %s (%d frames)' % (out, len(arr)))
        return arr
    finally:
        shutil.rmtree(dtop, ignore_errors=True)
        shutil.rmtree(dtraj, ignore_errors=True)


if __name__ == '__main__':
    base = os.path.join('supplementary', 'cMD', 'cMD', 'APO-MSTN', 'main', '1us')
    top = os.path.join(base, 'md-1usinitial.pdb')
    traj = os.path.join(base, 'md-nW1000.xtc')
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    arr = extract(top, traj, os.path.join('_work', 'cvs', 'APO-MSTN_main_1us.dat'),
                  max_frames=n, tag='apo_main_1us')

    ref = np.loadtxt(os.path.join(base, 'angle.dat'))
    print('\nreference first 5:', ref[:5, 1])
    print('ours      first 5:', np.round(arr[:5, 1], 4))
    for stride in (1, 2, 4):
        a = arr[::stride, 1]
        m = min(len(a), len(ref))
        d = np.abs(a[:m] - ref[:m, 1])
        print('stride %d: mean|diff|=%.4f max|diff|=%.4f (n=%d)' % (stride, d.mean(), d.max(), m))
