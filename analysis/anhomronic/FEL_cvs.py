"""cvs.py -- collective-variable extraction shared by the R2-4 analyses.

Angle-1     apex = midpoint of the C339-C339 interchain disulfide (SG-SG, per author
            decision); endpoints = CA of K356 in chain I and chain II
Distance-1  CA(P322, chain I) - CA(M367, chain II)
Distance-2  CA(A330, chain I) - CA(L286, chain II)

Residue indices are resolved geometrically (interchain C339-C339 SG-SG contact), so
the same code works for the apo dimer and for the ActRII complexes, whose topologies
append the receptor after the 218 GF-domain residues.

MDAnalysis' XDR reader cannot open non-ASCII paths, so every reader call changes into
the trajectory directory and passes bare file names.
"""
import os

import numpy as np

import MDAnalysis as mda

MSTN_NRES = 218
CHAIN_I_END = 109
MANUSCRIPT_CYS = 339
CV_COLUMNS = ('time_ns', 'angle1_deg', 'd1_A', 'd2_A')


def _one(u, sel, what):
    ag = u.select_atoms(sel)
    if len(ag) != 1:
        raise RuntimeError('%s: expected 1 atom, got %d (%s)' % (what, len(ag), sel))
    return ag[0].index


def resolve_indices(u):
    """Locate the eight atoms defining Angle-1, Distance-1 and Distance-2."""
    sg = u.select_atoms('resname CYX and name SG and resid 1:%d' % MSTN_NRES)
    resids = np.array([int(a.resid) for a in sg])
    pos = sg.positions
    pair = None
    for i in range(len(resids)):
        for j in range(i + 1, len(resids)):
            if (resids[i] <= CHAIN_I_END) == (resids[j] <= CHAIN_I_END):
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < 2.5:
                a, b = (resids[i], resids[j]) if resids[i] <= CHAIN_I_END else (resids[j], resids[i])
                pair = (a, b, d)
    if pair is None:
        raise RuntimeError('interchain C339-C339 disulfide not found')
    a, b, d = pair
    off_i = MANUSCRIPT_CYS - a
    off_j = MANUSCRIPT_CYS - b
    idx = dict(
        ca_k356_i=_one(u, 'name CA and resid %d' % (356 - off_i), 'K356 chain I'),
        ca_k356_ii=_one(u, 'name CA and resid %d' % (356 - off_j), 'K356 chain II'),
        sg_c339_i=_one(u, 'resname CYX and name SG and resid %d' % a, 'C339 SG chain I'),
        sg_c339_ii=_one(u, 'resname CYX and name SG and resid %d' % b, 'C339 SG chain II'),
        ca_p322=_one(u, 'name CA and resid %d' % (322 - off_i), 'P322 chain I'),
        ca_m367=_one(u, 'name CA and resid %d' % (367 - off_j), 'M367 chain II'),
        ca_a330=_one(u, 'name CA and resid %d' % (330 - off_i), 'A330 chain I'),
        ca_l286=_one(u, 'name CA and resid %d' % (286 - off_j), 'L286 chain II'),
    )
    note = 'C339-C339 SG %.2f A; chain offsets +%d/+%d; K356 %d/%d' % (
        d, off_i, off_j, 356 - off_i, 356 - off_j)
    return idx, note


def extract_segment(dirpath, top, traj, max_frames=None):
    """Return (array[N,4], info dict) of time / Angle-1 / D1 / D2 for one segment."""
    cwd = os.getcwd()
    os.chdir(dirpath)
    try:
        # the XDR reader needs an ASCII path: a path relative to the trajectory folder
        # contains only ASCII components even when the workspace root does not
        here = os.getcwd()
        top_arg = top if not os.path.isabs(top) else os.path.relpath(top, here)
        u = mda.Universe(top_arg, traj)
        idx, note = resolve_indices(u)
        rows = []
        speeds = []
        for i, ts in enumerate(u.trajectory):
            p = u.atoms.positions
            apex = (p[idx['sg_c339_i']] + p[idx['sg_c339_ii']]) / 2.0
            v1 = p[idx['ca_k356_i']] - apex
            v2 = p[idx['ca_k356_ii']] - apex
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            d1 = float(np.linalg.norm(p[idx['ca_p322']] - p[idx['ca_m367']]))
            d2 = float(np.linalg.norm(p[idx['ca_a330']] - p[idx['ca_l286']]))
            rows.append((ts.time / 1000.0, ang, d1, d2))
            if max_frames and i + 1 >= max_frames:
                break
        arr = np.asarray(rows, dtype=float)
        info = dict(atoms=len(u.atoms), frames=len(arr), note=note,
                    t0=float(arr[0, 0]) if len(arr) else None,
                    t1=float(arr[-1, 0]) if len(arr) else None)
        return arr, info
    finally:
        os.chdir(cwd)


def xtc_natoms(path):
    import struct
    with open(path, 'rb') as f:
        magic, na = struct.unpack('>ii', f.read(8))
    if magic != 1995:
        raise RuntimeError('not an xtc file: ' + path)
    return na


def pdb_natoms(path):
    n = 0
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                n += 1
    return n


def pick_topology(dirpath, top, traj, extra_dirs=()):
    """Choose a topology whose atom count matches the trajectory.

    Some folders store a solvated PDB next to a stripped XTC (or vice versa); a
    mismatch is fatal for MDAnalysis, so fall back to any PDB of the same system
    with the right atom count.
    """
    if not os.path.isabs(traj):
        traj = os.path.join(dirpath, traj)
    want = xtc_natoms(traj)
    cands = []
    for d in [dirpath] + list(extra_dirs):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.pdb', '.ent')):
                cands.append(os.path.join(d, f))
    # prefer the declared topology
    cands.sort(key=lambda p: (os.path.basename(p) != top, len(p)))
    for p in cands:
        try:
            if pdb_natoms(p) == want:
                return os.path.abspath(p)
        except Exception:
            continue
    raise RuntimeError('no topology with %d atoms for %s' % (want, traj))
