"""pca_analyze.py -- PC-space free-energy landscapes, overlaps and interpretation.

Reads the common-basis projections written by pca_extract.py and produces:

  Tables (in _work/pca/):
    pca_variance.csv          eigenvalue, % variance, cumulative %
    pca_correlations.csv      Pearson r between PC1-PC3 and Angle-1/D1/D2,
                              per system and pooled
    pca_overlap_replicas.csv  Bhattacharyya coefficient between the three
                              replicas of each system in PC1-PC2 space
    pca_overlap_systems.csv   Bhattacharyya coefficient between the six systems
    pca_basins.csv            CV-space basins mapped into PC1-PC2 space
    pca_gamd_coverage.csv     GaMD coverage of the cMD-occupied PC space
    summary.json              every scalar used in the manuscript text

  Figures (in _work/pca/fig/):
    fig_pca_overview.png/pdf  variance, global map, FELs, GaMD coverage
    fig_pca_replicas.png/pdf  replica-wise FELs + reproducibility
    fig_pca_interpretation.png/pdf  PC-CV correlations and time traces
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fel as felmod  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PCA = os.path.join(PROJ, '_work', 'pca')
FIG = os.path.join(PCA, 'fig')
os.makedirs(FIG, exist_ok=True)

KT = 0.0019872041 * 310.15          # kcal/mol at 310.15 K

SYSTEMS = ['Pre-active-MSTN', 'Traj-MSTN', 'Traj-MSTN-ActRII',
           'APO-MSTN', 'APO-MSTN-ActRII', 'Active-MSTN']
REPLICAS = ['main', 'replica1', 'replica2']
LABEL = {
    'Pre-active-MSTN': 'Pre-active MSTN',
    'Traj-MSTN': 'Traj-MSTN',
    'Traj-MSTN-ActRII': 'Traj-MSTN:ActRII',
    'APO-MSTN': 'APO-MSTN',
    'APO-MSTN-ActRII': 'APO-MSTN:ActRII',
    'Active-MSTN': 'Active-MSTN',
    'Pre-active-MSTN-ActRII': 'Pre-active MSTN:ActRII',
}
COLOR = {
    'Pre-active-MSTN': '#1f77b4',
    'Traj-MSTN': '#2ca02c',
    'Traj-MSTN-ActRII': '#d62728',
    'APO-MSTN': '#9467bd',
    'APO-MSTN-ActRII': '#ff7f0e',
    'Active-MSTN': '#4d4d4d',
}
GAMD_SYSTEMS = ['Pre-active-MSTN', 'Pre-active-MSTN-ActRII', 'APO-MSTN', 'APO-MSTN-ActRII']

# common grid for every panel (same bin width for all systems)
PC1 = dict(lo=-92.0, hi=132.0, step=4.0)
PC2 = dict(lo=-116.0, hi=100.0, step=4.0)
MIN_COUNT = 5
DG_CAP = 6.0
DG_VMAX = 3.0


# --------------------------------------------------------------------- loading
def load_table(path):
    d = np.loadtxt(path, comments='#', ndmin=2)
    return dict(time=d[:, 0], pc=d[:, 1:11], angle=d[:, 11], d1=d[:, 12],
                d2=d[:, 13], seg=d[:, 14].astype(int))


def load_all():
    cmd = {(s, r): load_table(os.path.join(PCA, 'proj_%s__%s.dat' % (s, r)))
           for s in SYSTEMS for r in REPLICAS}
    gamd = {s: load_table(os.path.join(PCA, 'proj_gamd_%s.dat' % s))
            for s in GAMD_SYSTEMS}
    return cmd, gamd


def pooled(cmd, system):
    parts = [cmd[(system, r)] for r in REPLICAS]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ------------------------------------------------------------------ histograms
def edges(g):
    n = int(round((g['hi'] - g['lo']) / g['step']))
    return np.linspace(g['lo'], g['hi'], n + 1)


XE, YE = edges(PC1), edges(PC2)


def hist(x, y):
    h, _, _ = np.histogram2d(x, y, bins=(XE, YE))
    return h


def prob(h, min_count=MIN_COUNT):
    p = h / max(h.sum(), 1)
    p = np.where(h >= min_count, p, 0.0)
    s = p.sum()
    return p / s if s > 0 else p


def prob_smooth(h, sigma=1.5):
    """KDE-like probability density used for overlap coefficients."""
    hs = felmod.smooth(h, sigma)
    return hs / max(hs.sum(), 1e-300)


def free_energy(h, sigma=1.0):
    hs = felmod.smooth(h, sigma)
    p = hs / max(hs.sum(), 1)
    m = h >= MIN_COUNT
    with np.errstate(divide='ignore'):
        dg = -KT * np.log(np.where(p > 0, p, 1.0))
    dg[~m] = np.nan
    if np.isfinite(dg).any():
        dg -= np.nanmin(dg)
        dg = np.minimum(dg, DG_CAP)
    return dg


def bhatt(p, q):
    return float(np.sum(np.sqrt(p * q)))


def rval(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# --------------------------------------------------------------------- analyse
def main():
    model = np.load(os.path.join(PCA, 'model.npz'))
    evr = model['evr']
    cmd, gamd = load_all()
    summary = {}

    # ---- variance ---------------------------------------------------------
    var_rows = []
    for i in range(10):
        var_rows.append(dict(pc='PC%d' % (i + 1), eigenvalue=model['eigenvalues'][i],
                             percent=100 * evr[i],
                             cumulative=100 * evr[:i + 1].sum()))
    write_csv(os.path.join(PCA, 'pca_variance.csv'), var_rows,
              ['pc', 'eigenvalue', 'percent', 'cumulative'])
    summary['variance'] = {r['pc']: round(r['percent'], 2) for r in var_rows}
    summary['variance_cumulative_pc2'] = round(var_rows[1]['cumulative'], 2)
    summary['variance_cumulative_pc3'] = round(var_rows[2]['cumulative'], 2)
    print('PC1+PC2 = %.1f%%   PC1-PC3 = %.1f%%'
          % (var_rows[1]['cumulative'], var_rows[2]['cumulative']))

    # ---- PC-CV correlations ----------------------------------------------
    cor_rows = []
    for system in SYSTEMS:
        d = pooled(cmd, system)
        row = dict(system=system, n=len(d['time']))
        for j in (1, 2, 3):
            for cv, col in (('angle1', 'angle'), ('d1', 'd1'), ('d2', 'd2')):
                row['r_PC%d_%s' % (j, cv)] = rval(d['pc'][:, j - 1], d[col])
        cor_rows.append(row)
    allf = {k: np.concatenate([pooled(cmd, s)[k] for s in SYSTEMS])
            for k in ('time', 'pc', 'angle', 'd1', 'd2')}
    row = dict(system='pooled', n=len(allf['time']))
    for j in (1, 2, 3):
        for cv, col in (('angle1', 'angle'), ('d1', 'd1'), ('d2', 'd2')):
            row['r_PC%d_%s' % (j, cv)] = rval(allf['pc'][:, j - 1], allf[col])
    cor_rows.append(row)
    write_csv(os.path.join(PCA, 'pca_correlations.csv'), cor_rows, list(cor_rows[0]))
    summary['correlations'] = cor_rows
    for r in cor_rows:
        print('%-20s r(PC1,D1)=%+.2f r(PC1,A1)=%+.2f r(PC2,A1)=%+.2f r(PC2,D2)=%+.2f'
              % (r['system'], r['r_PC1_d1'], r['r_PC1_angle1'],
                 r['r_PC2_angle1'], r['r_PC2_d2']))

    # ---- how much of each CV is contained in PC1-PC2 ----------------------
    rec_rows = []
    for system in SYSTEMS + ['pooled']:
        d = pooled(cmd, system) if system != 'pooled' else {
            k: np.concatenate([pooled(cmd, ss)[k] for ss in SYSTEMS])
            for k in ('angle', 'd1', 'd2', 'pc', 'time')}
        X = np.column_stack([np.ones(len(d['time'])), d['pc'][:, :2]])
        row = dict(system=system, n=len(d['time']))
        for name, y in (('angle1', d['angle']), ('d1', d['d1']), ('d2', d['d2'])):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            row['R2_' + name] = float(1 - ((y - X @ beta) ** 2).sum()
                                      / ((y - y.mean()) ** 2).sum())
        rec_rows.append(row)
    write_csv(os.path.join(PCA, 'pca_cv_reconstruction.csv'), rec_rows,
              list(rec_rows[0]))
    summary['cv_reconstruction'] = rec_rows

    # ---- replica overlap --------------------------------------------------
    rep_rows = []
    for system in SYSTEMS:
        ps = {}
        for r in REPLICAS:
            d = cmd[(system, r)]
            ps[r] = prob_smooth(hist(d['pc'][:, 0], d['pc'][:, 1]))
        for i, a in enumerate(REPLICAS):
            for b in REPLICAS[i + 1:]:
                rep_rows.append(dict(system=system, pair='%s/%s' % (a, b),
                                     bhattacharyya=bhatt(ps[a], ps[b])))
    write_csv(os.path.join(PCA, 'pca_overlap_replicas.csv'), rep_rows,
              ['system', 'pair', 'bhattacharyya'])

    # ---- system overlap ---------------------------------------------------
    sys_p = {s: prob_smooth(hist(*[pooled(cmd, s)['pc'][:, i] for i in (0, 1)]))
             for s in SYSTEMS}
    sys_rows = []
    for i, a in enumerate(SYSTEMS):
        for b in SYSTEMS[i + 1:]:
            sys_rows.append(dict(system_a=a, system_b=b,
                                 bhattacharyya=bhatt(sys_p[a], sys_p[b])))
    write_csv(os.path.join(PCA, 'pca_overlap_systems.csv'), sys_rows,
              ['system_a', 'system_b', 'bhattacharyya'])
    summary['overlap_replicas'] = rep_rows
    summary['overlap_systems'] = sys_rows

    # ---- CV basins mapped onto PC1-PC2 -----------------------------------
    basin_rows = []
    for system in SYSTEMS:
        path = os.path.join(PROJ, '_work', 'analysis', '%s_basins.csv' % system)
        if not os.path.exists(path):
            continue
        centres = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(3, 4, 5),
                             ndmin=2)
        d = pooled(cmd, system)
        X = np.column_stack([d['angle'], d['d1'], d['d2']])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        C = (centres - mu) / sd
        lab = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        for j, c in enumerate(centres):
            m = lab == j
            basin_rows.append(dict(
                system=system, basin='B%d' % j, n=int(m.sum()),
                angle1=c[0], d1=c[1], d2=c[2],
                pc1=float(d['pc'][m, 0].mean()) if m.any() else np.nan,
                pc1_sd=float(d['pc'][m, 0].std()) if m.any() else np.nan,
                pc2=float(d['pc'][m, 1].mean()) if m.any() else np.nan,
                pc2_sd=float(d['pc'][m, 1].std()) if m.any() else np.nan))
    write_csv(os.path.join(PCA, 'pca_basins.csv'), basin_rows, list(basin_rows[0]))
    summary['basins'] = basin_rows

    # ---- GaMD coverage ----------------------------------------------------
    # region occupied by the cMD ensemble: bins holding 90 % of its probability
    cmd_all = {k: np.concatenate([pooled(cmd, s)[k] for s in SYSTEMS])
               for k in ('pc', 'angle', 'd1', 'd2')}
    h_cmd = hist(cmd_all['pc'][:, 0], cmd_all['pc'][:, 1])
    p_cmd = felmod.smooth(h_cmd, 1.5)
    p_cmd = p_cmd / p_cmd.sum()
    flat = np.argsort(p_cmd.ravel())[::-1]
    csum = np.cumsum(p_cmd.ravel()[flat])
    keep = flat[:np.searchsorted(csum, 0.90) + 1]
    inside = np.zeros(p_cmd.size, bool)
    inside[keep] = True
    inside = inside.reshape(p_cmd.shape)
    gamd_rows = []
    for system in GAMD_SYSTEMS:
        d = gamd[system]
        h = hist(d['pc'][:, 0], d['pc'][:, 1])
        idx = np.clip(np.digitize(d['pc'][:, 0], XE) - 1, 0, len(XE) - 2)
        idy = np.clip(np.digitize(d['pc'][:, 1], YE) - 1, 0, len(YE) - 2)
        frac = float(inside[idx, idy].mean())
        p = prob_smooth(h)
        gamd_rows.append(dict(
            system=system, n=len(d['time']),
            pc1_mean=float(d['pc'][:, 0].mean()), pc1_min=float(d['pc'][:, 0].min()),
            pc1_max=float(d['pc'][:, 0].max()),
            pc2_mean=float(d['pc'][:, 1].mean()), pc2_min=float(d['pc'][:, 1].min()),
            pc2_max=float(d['pc'][:, 1].max()),
            frac_in_cmd90=frac,
            bhattacharyya_vs_cmd=bhatt(p, sys_p[system])
            if system in sys_p else np.nan))
    write_csv(os.path.join(PCA, 'pca_gamd_coverage.csv'), gamd_rows, list(gamd_rows[0]))
    summary['gamd_coverage'] = gamd_rows
    for r in gamd_rows:
        print('GaMD %-22s PC1 %6.1f..%6.1f  in cMD90 %.2f  B=%.2f'
              % (r['system'], r['pc1_min'], r['pc1_max'], r['frac_in_cmd90'],
                 r['bhattacharyya_vs_cmd']))

    # ---- system means -----------------------------------------------------
    means = []
    for system in SYSTEMS:
        d = pooled(cmd, system)
        means.append(dict(system=system, n=len(d['time']),
                          pc1=float(d['pc'][:, 0].mean()), pc1_sd=float(d['pc'][:, 0].std()),
                          pc2=float(d['pc'][:, 1].mean()), pc2_sd=float(d['pc'][:, 1].std()),
                          angle1=float(d['angle'].mean()), d1=float(d['d1'].mean()),
                          d2=float(d['d2'].mean())))
    write_csv(os.path.join(PCA, 'pca_system_means.csv'), means, list(means[0]))
    summary['system_means'] = means
    for m in means:
        print('%-20s PC1 %6.1f+-%4.1f  PC2 %6.1f+-%4.1f  A1 %5.1f D1 %5.2f D2 %5.2f'
              % (m['system'], m['pc1'], m['pc1_sd'], m['pc2'], m['pc2_sd'],
                 m['angle1'], m['d1'], m['d2']))

    with open(os.path.join(PCA, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    make_figures(cmd, gamd, summary)
    make_main_figure(cmd, summary)


def write_csv(path, rows, cols):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append('%.4f' % v)
                else:
                    vals.append(str(v))
            f.write(','.join(vals) + '\n')


# ============================================================ figure helpers
XC = 0.5 * (XE[:-1] + XE[1:])
YC = 0.5 * (YE[:-1] + YE[1:])
GAMD_COLOR = {
    'Pre-active-MSTN': '#1f77b4',
    'Pre-active-MSTN-ActRII': '#17becf',
    'APO-MSTN': '#9467bd',
    'APO-MSTN-ActRII': '#ff7f0e',
}


def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def panel_fel(ax, h, title, basins=None, letters=None):
    dg = free_energy(h)
    ax.set_facecolor('#dedede')
    ax.pcolormesh(XE, YE, dg.T, cmap='jet', vmin=0, vmax=DG_VMAX, shading='flat')
    ax.set_xlim(XE[0], XE[-1])
    ax.set_ylim(YE[0], YE[-1])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title, fontsize=10)
    if basins is not None:
        import matplotlib.patheffects as pe
        for b in basins:
            if not np.isfinite(b['pc1']):
                continue
            ax.plot(b['pc1'], b['pc2'], 'x', color='white', ms=8, mew=2.0)
            ax.plot(b['pc1'], b['pc2'], 'x', color='black', ms=8, mew=0.7)
            ax.annotate(b['basin'], (b['pc1'], b['pc2']), xytext=(4, 4),
                        textcoords='offset points', color='white', fontsize=8,
                        fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    if letters:
        ax.text(-0.16, 1.06, letters, transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='top')
    _style(ax)


def panel_contours(ax, cmd, letter=None):
    from matplotlib.lines import Line2D
    handles = []
    for s in SYSTEMS:
        d = pooled(cmd, s)
        p = felmod.smooth(hist(d['pc'][:, 0], d['pc'][:, 1]), 1.5)
        p = p / max(p.max(), 1e-300)
        ax.contourf(XC, YC, p.T, levels=[0.12, 1.1], colors=[COLOR[s]], alpha=0.12)
        ax.contour(XC, YC, p.T, levels=[0.2, 0.5, 0.8], colors=[COLOR[s]],
                   linewidths=[0.7, 1.1, 1.6])
        ax.plot(d['pc'][:, 0].mean(), d['pc'][:, 1].mean(), 'o', color=COLOR[s],
                ms=7, mec='white', mew=0.8)
        handles.append(Line2D([], [], color=COLOR[s], lw=1.6, label=LABEL[s]))
    ax.set_xlim(XE[0], XE[-1])
    ax.set_ylim(YE[0], YE[-1])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Common-basis projection of the six cMD ensembles', fontsize=10)
    ax.legend(handles=handles, fontsize=6.6, loc='lower right', frameon=False, ncol=2)
    if letter:
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='top')
    _style(ax)


def make_figures(cmd, gamd, summary):
    plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})

    # ------------------------------------------------------------- overview
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.38)

    ax = fig.add_subplot(gs[0, 0])
    v = summary['variance']
    pct = [v['PC%d' % (i + 1)] for i in range(10)]
    ax.bar(range(1, 11), pct, color='#4c72b0', width=0.7, label='individual PC')
    ax.plot(range(1, 11), np.cumsum(pct), 'o-', color='#c44e52', ms=4, lw=1.4,
            label='cumulative')
    ax.set_ylim(0, 100)
    ax.set_xlabel('principal component')
    ax.set_ylabel('variance explained (%)')
    ax.set_title('Variance explained by the common-basis PCA', fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc='center right')
    ax.text(0.30, 0.55, 'PC1+PC2 = %.1f%%' % summary['variance_cumulative_pc2'],
            transform=ax.transAxes, fontsize=9)
    ax.text(-0.16, 1.06, 'A', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')
    _style(ax)

    ax = fig.add_subplot(gs[0, 1])
    panel_contours(ax, cmd, 'B')

    ax = fig.add_subplot(gs[0, 2])
    allpc = np.concatenate([pooled(cmd, s)['pc'][:, :2] for s in SYSTEMS])
    h_cmd = hist(allpc[:, 0], allpc[:, 1])
    p_cmd = felmod.smooth(h_cmd, 1.5)
    p_cmd = p_cmd / p_cmd.sum()
    flat = np.argsort(p_cmd.ravel())[::-1]
    csum = np.cumsum(p_cmd.ravel()[flat])
    inside = np.zeros(p_cmd.size, bool)
    inside[flat[:np.searchsorted(csum, 0.90) + 1]] = True
    ax.contourf(XC, YC, inside.reshape(p_cmd.shape).T.astype(float),
                levels=[0.5, 1.5], colors=['#dedede'])
    for s in GAMD_SYSTEMS:
        d = gamd[s]
        p = felmod.smooth(hist(d['pc'][:, 0], d['pc'][:, 1]), 1.5)
        p = p / max(p.max(), 1e-300)
        ax.contour(XC, YC, p.T, levels=[0.2, 0.5], colors=[GAMD_COLOR[s]],
                   linewidths=[0.9, 1.6], linestyles='--')
        ax.plot(np.nan, np.nan, '--', color=GAMD_COLOR[s], lw=1.4, label=LABEL[s])
    ax.legend(fontsize=7, loc='upper left', frameon=False)
    ax.set_xlim(XE[0], XE[-1])
    ax.set_ylim(YE[0], YE[-1])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('GaMD sampling on the same PC basis', fontsize=10)
    ax.text(0.02, 0.02, 'grey: region containing 90 % of the cMD frames',
            transform=ax.transAxes, fontsize=7)
    ax.text(-0.16, 1.06, 'F', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')
    _style(ax)

    show = [('Pre-active-MSTN', 'C'), ('Traj-MSTN-ActRII', 'D'),
            ('Active-MSTN', 'E')]
    for k, (system, letter) in enumerate(show):
        ax = fig.add_subplot(gs[1, k])
        d = pooled(cmd, system)
        panel_fel(ax, hist(d['pc'][:, 0], d['pc'][:, 1]),
                  '%s  (PCA-based FEL)' % LABEL[system], None, letter)

    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, DG_VMAX))
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.30])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('ΔG (kcal mol$^{-1}$)')

    fig.savefig(os.path.join(FIG, 'fig_pca_overview.png'), dpi=220,
                bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig_pca_overview.pdf'), bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------- replicas
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 11))
    rows = ['Pre-active-MSTN', 'Traj-MSTN-ActRII', 'Active-MSTN']
    for i, system in enumerate(rows):
        for j, rep in enumerate(REPLICAS):
            ax = axes[i, j]
            d = cmd[(system, rep)]
            panel_fel(ax, hist(d['pc'][:, 0], d['pc'][:, 1]),
                      '%s — %s' % (LABEL[system], rep))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_pca_replicas.png'), dpi=200)
    fig.savefig(os.path.join(FIG, 'fig_pca_replicas.pdf'))
    plt.close(fig)

    # ------------------------------------------------------- interpretation
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    pooled_cor = summary['correlations'][-1]
    ax = fig.add_subplot(gs[0, 0])
    for s in SYSTEMS:
        d = pooled(cmd, s)
        ax.scatter(d['pc'][::3, 0], d['d1'][::3], s=2, alpha=0.25, color=COLOR[s],
                   label=LABEL[s], rasterized=True)
    ax.set_xlabel('PC1')
    ax.set_ylabel('Distance-1 (Å)')
    ax.set_title('PC1 vs Distance-1  (pooled r = %.2f)' % pooled_cor['r_PC1_d1'],
                 fontsize=10)
    ax.legend(fontsize=7, markerscale=4, frameon=False, loc='upper left')
    ax.text(-0.14, 1.05, 'A', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')
    _style(ax)

    ax = fig.add_subplot(gs[0, 1])
    for s in SYSTEMS:
        d = pooled(cmd, s)
        ax.scatter(d['pc'][::3, 1], d['angle'][::3], s=2, alpha=0.25, color=COLOR[s],
                   rasterized=True)
    ax.set_xlabel('PC2')
    ax.set_ylabel('Angle-1 (deg)')
    ax.set_title('PC2 vs Angle-1  (pooled r = %.2f)' % pooled_cor['r_PC2_angle1'],
                 fontsize=10)
    ax.text(-0.14, 1.05, 'B', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')
    _style(ax)

    ax = fig.add_subplot(gs[1, 0])
    labels, a_r2, d1_r2 = [], [], []
    groups = [(s, pooled(cmd, s)) for s in SYSTEMS]
    groups.append(('pooled', {k: np.concatenate([pooled(cmd, ss)[k] for ss in SYSTEMS])
                              for k in ('angle', 'd1', 'd2', 'pc', 'time')}))
    for s, d in groups:
        X = np.column_stack([np.ones(len(d['time'])), d['pc'][:, :2]])
        out = []
        for y in (d['angle'], d['d1']):
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            out.append(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        labels.append('pooled' if s == 'pooled' else LABEL[s])
        a_r2.append(out[0])
        d1_r2.append(out[1])
    x = np.arange(len(labels))
    ax.bar(x - 0.19, a_r2, width=0.36, color='#4c72b0', label='Angle-1')
    ax.bar(x + 0.19, d1_r2, width=0.36, color='#dd8452', label='Distance-1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('R$^2$ reconstructed from PC1–PC2')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title('Collective-variable content of the PC1–PC2 plane', fontsize=10)
    ax.text(-0.14, 1.05, 'C', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')
    _style(ax)

    ax = fig.add_subplot(gs[1, 1])
    for s in SYSTEMS:
        for r in REPLICAS:
            d = cmd[(s, r)]
            ax.plot(d['time'], d['pc'][:, 0], lw=0.7, color=COLOR[s], alpha=0.8)
        ax.plot(np.nan, np.nan, color=COLOR[s], lw=1.4, label=LABEL[s])
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('PC1')
    ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.22))
    ax.set_title('PC1 along the cMD trajectories', fontsize=10)
    ax.text(-0.14, 1.05, 'D', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')
    _style(ax)

    fig.savefig(os.path.join(FIG, 'fig_pca_interpretation.png'), dpi=220,
                bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig_pca_interpretation.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('figures written to %s' % FIG)


def make_main_figure(cmd, summary):
    """Compact 2x2 version intended for the main text (new Figure 7 candidate)."""
    plt.rcParams.update({'font.size': 9.5, 'axes.linewidth': 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.6))
    panel_contours(axes[0, 0], cmd, 'A')
    for ax, system, letter in ((axes[0, 1], 'Pre-active-MSTN', 'B'),
                               (axes[1, 0], 'Traj-MSTN-ActRII', 'C')):
        d = pooled(cmd, system)
        panel_fel(ax, hist(d['pc'][:, 0], d['pc'][:, 1]),
                  '%s: PCA-based FEL' % LABEL[system], None, letter)
    ax = axes[1, 1]
    for s in SYSTEMS:
        d = pooled(cmd, s)
        ax.scatter(d['pc'][::3, 1], d['angle'][::3], s=2, alpha=0.25,
                   color=COLOR[s], rasterized=True)
    cor = summary['correlations'][-1]
    ax.set_xlabel('PC2')
    ax.set_ylabel('Angle-1 (deg)')
    ax.set_title('PC2 recovers the hinge angle', fontsize=10)
    ax.text(0.03, 0.95, 'pooled r = %.2f' % cor['r_PC2_angle1'],
            transform=ax.transAxes, fontsize=9, va='top')
    ax.text(-0.14, 1.05, 'D', transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top')
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig_pca_main.png'), dpi=250, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, 'fig_pca_main.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('main-text candidate written')


if __name__ == '__main__':
    main()
