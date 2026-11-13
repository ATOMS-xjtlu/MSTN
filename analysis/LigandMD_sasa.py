"""Shrake-Rupley solvent-accessible surface, computed for a subset of atoms.

Only the ligand surface is needed (buried-fraction metric), so the scored atoms
are the ligand heavy atoms and the occluders are the protein heavy atoms plus the
ligand atoms themselves.  A KD-tree per atom group keeps this cheap.
"""
import numpy as np
from scipy.spatial import cKDTree

PROBE = 1.4
RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20, "F": 1.47, "CL": 1.75, "P": 1.80}


def radii_from_names(names):
    out = []
    for n in names:
        s = str(n).upper()
        el = "CL" if s.startswith("CL") else s[0]
        out.append(RADII.get(el, 1.70))
    return np.array(out, dtype=float)


def _sphere_points(n=240):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


_PTS = _sphere_points(240)


def sasa_targets(target_centers, target_radii, occluder_centers=None, occluder_radii=None):
    """total SASA (A^2) of the target atoms, with the given occluders (may be None)."""
    tc = np.asarray(target_centers, float)
    tr = np.asarray(target_radii, float)
    if occluder_centers is not None and len(occluder_centers):
        allc = np.vstack([tc, np.asarray(occluder_centers, float)])
        allr = np.concatenate([tr, np.asarray(occluder_radii, float)])
    else:
        allc, allr = tc, tr
    tree = cKDTree(allc)
    rmax = allr.max()
    total = 0.0
    for k, (c, r) in enumerate(zip(tc, tr)):
        R = r + PROBE
        idx = tree.query_ball_point(c, R + rmax + 0.1)
        idx = [j for j in idx if j != k]
        if idx:
            sub = allc[idx]
            rr = allr[idx] + PROBE
            pts = c + _PTS * R
            d2 = ((pts[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
            frac = 1.0 - (d2 < rr[None, :] ** 2).any(axis=1).mean()
        else:
            frac = 1.0
        total += 4.0 * np.pi * R * R * frac
    return total
