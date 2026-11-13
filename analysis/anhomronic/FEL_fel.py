"""fel.py -- free-energy-landscape statistics for the R2-4 convergence analyses.

Everything here works on the CV tables written by _tools/extract_all.py:
    time_ns  angle1_deg  d1_A  d2_A  segment

Design notes
------------
* Fixed grids are used for every system, replica, block and cumulative subset, so
  landscapes are directly comparable (a point the reviewer can check).
* Bins holding fewer than MIN_COUNT frames are masked instead of being reported as
  high free energy; masking is recorded so the mask can be shown in figures.
* Basin definitions come from k-means on the pooled ensemble of a system (centres are
  then frozen), so bootstrap and block analyses only resample frames, not definitions.
* Blocks are resampled with a moving-block bootstrap; the block length defaults to
  max(2 * integrated autocorrelation time, 50 ns).
"""
import os

import numpy as np

KT = 0.0019872041 * 310.15          # kcal/mol at 310.15 K

GRID = {
    'angle1': dict(lo=20.0, hi=170.0, step=3.0),
    'd1': dict(lo=3.0, hi=40.0, step=1.0),
    'd2': dict(lo=5.0, hi=30.0, step=1.0),
}
MIN_COUNT = 5
DG_CAP = 6.0


# --------------------------------------------------------------------------- io
def load(path):
    d = np.loadtxt(path)
    if d.ndim == 1:
        d = d[None, :]
    return dict(time=d[:, 0], angle1=d[:, 1], d1=d[:, 2], d2=d[:, 3], seg=d[:, 4].astype(int))


def decimate(data, dt_max=0.25):
    """Thin each contiguous segment to at most dt_max resolution (indices preserved)."""
    out = []
    for seg in np.unique(data['seg']):
        m = data['seg'] == seg
        t = data['time'][m]
        idx = np.where(m)[0]
        if len(t) < 3:
            out.append(idx)
            continue
        dt = np.median(np.diff(t))
        k = max(1, int(round(dt_max / dt))) if dt > 0 else 1
        out.append(idx[::k])
    keep = np.sort(np.concatenate(out))
    return {k: v[keep] for k, v in data.items()}


def subset_time(data, t0=None, t1=None):
    m = np.ones(len(data['time']), bool)
    if t0 is not None:
        m &= data['time'] >= t0
    if t1 is not None:
        m &= data['time'] <= t1
    return {k: v[m] for k, v in data.items()}


def stack(datas):
    keys = ('time', 'angle1', 'd1', 'd2')
    out = {k: np.concatenate([d[k] for d in datas]) for k in keys}
    out['rep'] = np.concatenate([np.full(len(d['time']), i) for i, d in enumerate(datas)])
    out['seg'] = np.zeros(len(out['time']), int)
    return out


# ------------------------------------------------------------------ histograms
def edges(name):
    g = GRID[name]
    n = int(round((g['hi'] - g['lo']) / g['step']))
    return np.linspace(g['lo'], g['hi'], n + 1)


def hist2d(x, y, xname, yname):
    xe, ye = edges(xname), edges(yname)
    h, _, _ = np.histogram2d(x, y, bins=(xe, ye))
    return h


def free_energy(h, min_count=MIN_COUNT, cap=DG_CAP):
    """Histogram -> free energy (kcal/mol), min at 0, masked sparse bins -> nan."""
    p = h / max(h.sum(), 1)
    out = np.full(h.shape, np.nan)
    m = h >= min_count
    with np.errstate(divide='ignore'):
        dg = -KT * np.log(np.where(m, p, 1.0))
    dg[~m] = np.nan
    if np.isfinite(dg).any():
        dg = dg - np.nanmin(dg)
        dg = np.minimum(dg, cap)
    out[:] = dg
    return out


def smooth(h, sigma=1.0):
    """Small Gaussian smoothing (separable), edge-aware."""
    if sigma <= 0:
        return h.astype(float)
    r = int(max(1, round(3 * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    a = h.astype(float)
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 0, a)
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 1, a)
    return a


def prob2d(h, min_count=MIN_COUNT):
    p = h / max(h.sum(), 1)
    p = np.where(h >= min_count, p, 0.0)
    s = p.sum()
    return p / s if s > 0 else p


def bhattacharyya(p, q):
    return float(np.sum(np.sqrt(p * q)))


def js_divergence(p, q):
    p = p / max(p.sum(), 1e-300)
    q = q / max(q.sum(), 1e-300)
    m = 0.5 * (p + q)
    def kl(a, b):
        msk = a > 0
        return float(np.sum(a[msk] * np.log2(a[msk] / b[msk])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def fel_rms(f1, f2, cap=DG_CAP):
    """RMS difference of two free-energy surfaces over bins where both are defined."""
    m = np.isfinite(f1) & np.isfinite(f2)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((np.clip(f1[m], 0, cap) - np.clip(f2[m], 0, cap)) ** 2)))


# ------------------------------------------------------------------ timescales
def integrated_act(x, dt):
    """Integrated autocorrelation time (in frames) with the standard sum-to-zero rule."""
    x = np.asarray(x, float)
    x = x - x.mean()
    v = np.dot(x, x) / len(x)
    if v <= 0:
        return 1.0
    acf = np.correlate(x, x, mode='full')[len(x) - 1:] / (v * len(x))
    acf = acf[:max(2, len(acf) // 2)]
    s = 0.0
    for k in range(1, len(acf)):
        if acf[k] <= 0:
            break
        s += acf[k]
    return max(1.0, 1.0 + 2.0 * s)


# ---------------------------------------------------------------------- basins
def kmeans(X, k, seed=0, iters=200):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), k, replace=False)]
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        newc = np.array([X[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(newc, c, atol=1e-8):
            break
        c = newc
    return c, lab


def define_basins(data, k=5, seed=0, max_fit=20000):
    """Pool frames, fit k-means in z-scored CV space, return centres + zscaler."""
    X = np.column_stack([data['angle1'], data['d1'], data['d2']])
    if len(X) > max_fit:
        idx = np.linspace(0, len(X) - 1, max_fit).astype(int)
        Xf = X[idx]
    else:
        Xf = X
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    c, _ = kmeans((Xf - mu) / sd, k, seed=seed)
    centres = c * sd + mu
    # order basins by mean Distance-1 (compact -> open), which follows helix maturation
    order = np.argsort(centres[:, 1])
    centres = centres[order]
    return dict(centres=centres, mu=mu, sd=sd)


def assign(data, basins):
    X = np.column_stack([data['angle1'], data['d1'], data['d2']])
    Z = (X - basins['mu']) / basins['sd']
    C = (basins['centres'] - basins['mu']) / basins['sd']
    d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    return d.argmin(1)


def populations(labels, k):
    n = np.bincount(labels, minlength=k).astype(float)
    return n / max(n.sum(), 1)


def delta_g(pop):
    """Free energies relative to the most populated basin (kcal/mol)."""
    with np.errstate(divide='ignore'):
        g = -KT * np.log(np.where(pop > 0, pop, np.nan))
    return g - np.nanmin(g)


def transitions(labels, k):
    """Directed transition counts between basins for a contiguous frame sequence."""
    M = np.zeros((k, k), int)
    for a, b in zip(labels[:-1], labels[1:]):
        if a != b:
            M[a, b] += 1
    return M


# ------------------------------------------------------------------- bootstrap
def moving_block_indices(n, block, rng):
    """Indices for one moving-block bootstrap resample of a length-n sequence."""
    nblocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), nblocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


def block_bootstrap(labels, n_samples=1000, block=200, seed=0, k=None):
    """Return arrays of resampled basin populations."""
    rng = np.random.default_rng(seed)
    k = k or int(labels.max()) + 1
    out = np.zeros((n_samples, k))
    n = len(labels)
    for i in range(n_samples):
        idx = moving_block_indices(n, block, rng)
        out[i] = np.bincount(labels[idx], minlength=k) / len(idx)
    return out


def ci95(samples, axis=0):
    return np.percentile(samples, 2.5, axis=axis), np.percentile(samples, 97.5, axis=axis)

def auto_grid(data, pad_deg=6.0, pad_a=1.5):
    """Set the module grid from the pooled data of one system (fixed for all subsets).

    Called once per system so that replicas, blocks and cumulative subsets are all
    binned identically, while each system uses its own actual range.
    """
    import math
    def span(v, pad, step):
        lo = math.floor((float(np.min(v)) - pad) / step) * step
        hi = math.ceil((float(np.max(v)) + pad) / step) * step
        return lo, hi
    a_lo, a_hi = span(data['angle1'], pad_deg, GRID['angle1']['step'])
    d1_lo, d1_hi = span(data['d1'], pad_a, 1.0)
    d2_lo, d2_hi = span(data['d2'], pad_a, 1.0)
    GRID['angle1'].update(lo=a_lo, hi=a_hi)
    GRID['d1'].update(lo=d1_lo, hi=d1_hi)
    GRID['d2'].update(lo=d2_lo, hi=d2_hi)
    return {k: dict(v) for k, v in GRID.items()}
