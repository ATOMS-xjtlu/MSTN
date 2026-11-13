"""make_si_figures.py -- compose the proposed SI figures S8-S11 for R2-4.

    Fig. S8   replica-wise landscapes: Traj-MSTN:ActRII (both projections) + Active-MSTN reference
    Fig. S9   cumulative landscapes of Traj-MSTN:ActRII at 0.5 / 1.0 / 1.5 / 2.0 us
    Fig. S10  free energy vs trajectory length, landscape overlap vs length, replica overlap matrices
    Fig. S11  GaMD sampling densities (four systems) and their cumulative overlap

Outputs go to _work/si_figures/ as PNG (preview) and PDF (vector, for submission).
"""
import csv
import os
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import fel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CVDIR = os.path.join('_work', 'cvs')
GCVDIR = os.path.join('_work', 'gamd_cvs')
ANALYSIS = os.path.join('_work', 'analysis')
GANAL = os.path.join('_work', 'gamd_analysis')
OUT = os.path.join('_work', 'si_figures')
REPS = ('main', 'replica1', 'replica2')
PROJ = {'a1d1': ('angle1', 'd1', 'Angle-1 (deg)', 'Distance-1 (A)'),
        'd1d2': ('d1', 'd2', 'Distance-1 (A)', 'Distance-2 (A)')}
CMAP = 'jet'


def set_grid(angle, d1, d2):
    fel.GRID['angle1'].update(lo=angle[0], hi=angle[1], step=angle[2])
    fel.GRID['d1'].update(lo=d1[0], hi=d1[1], step=d1[2])
    fel.GRID['d2'].update(lo=d2[0], hi=d2[1], step=d2[2])


def load_cmd(system):
    out = {}
    for rep in REPS:
        p = os.path.join(CVDIR, '%s__%s.dat' % (system, rep))
        if os.path.exists(p):
            out[rep] = fel.decimate(fel.load(p))
    return out


def load_gamd(system):
    d = np.loadtxt(os.path.join(GCVDIR, system + '.dat'))
    return dict(time=d[:, 0], angle1=d[:, 1], d1=d[:, 2], d2=d[:, 3], seg=d[:, 4].astype(int))


def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def panel(ax, data, proj, vmax, letter, title='', show_xlabel=True, show_ylabel=True,
          centres=None):
    x, y, xl, yl = PROJ[proj]
    xe, ye = fel.edges(x), fel.edges(y)
    f = fel.free_energy(fel.hist2d(data[x], data[y], x, y))
    ax.set_facecolor('#ececec')
    im = ax.pcolormesh(xe, ye, f.T, cmap=CMAP, vmin=0, vmax=vmax, shading='auto')
    if centres is not None:
        for j, c in enumerate(centres):
            cx = c[0] if x == 'angle1' else c[1]
            cy = c[1] if y == 'd1' else c[2]
            ax.plot(cx, cy, 'x', color='w', ms=5, mew=1.3)
            ax.annotate('B%d' % j, (cx, cy), textcoords='offset points',
                        xytext=(4, -9) if j % 2 == 0 else (4, 6),
                        fontsize=6.5, color='w', weight='bold')
    if show_xlabel:
        ax.set_xlabel(xl, fontsize=7.5)
    if show_ylabel:
        ax.set_ylabel(yl, fontsize=7.5)
    ax.tick_params(labelsize=6.5)
    ax.set_title(title, fontsize=8)
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=10, weight='bold',
            va='bottom', ha='left')
    return im


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  wrote', name + '.png / .pdf')


# --------------------------------------------------------------------- Fig. S8
def fig_s8():
    print('[Fig. S8] replica-wise landscapes')
    tma = load_cmd('Traj-MSTN-ActRII')
    act = load_cmd('Active-MSTN')
    # shared grid across the systems shown here
    allframes = fel.stack([tma[r] for r in tma] + [act[r] for r in act])
    set_grid((60, 170, 3), (3, 20, 1), (6, 26, 1))
    pooled_tma = fel.stack([tma[r] for r in tma])
    pooled_act = fel.stack([act[r] for r in act])
    basins_tma = fel.define_basins(pooled_tma, k=4, seed=0)['centres']
    basins_act = fel.define_basins(pooled_act, k=3, seed=0)['centres']

    f_pool = fel.free_energy(fel.hist2d(pooled_tma['angle1'], pooled_tma['d1'], 'angle1', 'd1'))
    vmax = 6.0

    rows = [('Traj-MSTN:ActRII', tma, 'a1d1', pooled_tma, basins_tma),
            ('Traj-MSTN:ActRII', tma, 'd1d2', pooled_tma, basins_tma),
            ('Active-MSTN', act, 'a1d1', pooled_act, basins_act)]
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 8.0))
    L = 0
    for r, (name, reps, proj, pooled, centres) in enumerate(rows):
        for c, rep in enumerate(list(REPS) + ['pooled']):
            data = pooled if rep == 'pooled' else reps[rep]
            title = name if c == 0 else ''
            im = panel(axes[r][c], data, proj, vmax, string.ascii_uppercase[L],
                       title=title, centres=centres if rep == 'pooled' else None)
            if c == 0:
                axes[r][c].set_ylabel(PROJ[proj][3], fontsize=7.5)
            L += 1
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('dG (kcal/mol)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for r, y in enumerate((0.86, 0.55, 0.24)):
        fig.text(0.012, y, ['Traj-MSTN:ActRII', 'Traj-MSTN:ActRII', 'Active-MSTN'][r],
                 rotation=90, fontsize=8, va='center', weight='bold')
    fig.subplots_adjust(left=0.09, right=0.90, top=0.95, bottom=0.06, wspace=0.25, hspace=0.18)
    save(fig, 'Fig_S8_replica_landscapes')


# --------------------------------------------------------------------- Fig. S9
def fig_s9():
    print('[Fig. S9] cumulative landscapes')
    tma = load_cmd('Traj-MSTN-ActRII')
    data = tma['main']
    fel.auto_grid(fel.stack([tma[r] for r in tma]))
    basins = fel.define_basins(fel.stack([tma[r] for r in tma]), k=4, seed=0)['centres']
    p_full = {p: fel.prob2d(fel.hist2d(data[PROJ[p][0]], data[PROJ[p][1]], PROJ[p][0], PROJ[p][1]))
              for p in PROJ}
    times = (0.5, 1.0, 1.5, 2.0)
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.4))
    L = 0
    for r, proj in enumerate(('a1d1', 'd1d2')):
        for c, T in enumerate(times):
            sub = fel.subset_time(data, None, T * 1000.0)
            x, y = PROJ[proj][0], PROJ[proj][1]
            p = fel.prob2d(fel.hist2d(sub[x], sub[y], x, y))
            B = fel.bhattacharyya(p, p_full[proj])
            im = panel(axes[r][c], sub, proj, 6.0, string.ascii_uppercase[L],
                       title='%.1f us  (B = %.2f)' % (T, B),
                       centres=basins if c == 3 else None)
            if r == 0:
                axes[r][c].set_xlabel('')
            L += 1
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('dG (kcal/mol)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.93, bottom=0.10, wspace=0.25, hspace=0.28)
    save(fig, 'Fig_S9_cumulative_landscapes')


# -------------------------------------------------------------------- Fig. S10
def fig_s10():
    print('[Fig. S10] convergence metrics')
    tma = load_cmd('Traj-MSTN-ActRII')
    act = load_cmd('Active-MSTN')
    rows = read_csv(os.path.join(ANALYSIS, 'Traj-MSTN-ActRII_cumulative.csv'))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    L = 0

    for k, (system, reps) in enumerate((('Traj-MSTN:ActRII', tma), ('Active-MSTN', act))):
        ax = axes[k]
        fel.auto_grid(fel.stack([reps[r] for r in reps]))
        names = list(reps)
        n = len(names)
        probs = {rep: fel.prob2d(fel.hist2d(reps[rep]['angle1'], reps[rep]['d1'], 'angle1', 'd1'))
                 for rep in names}
        M = np.ones((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = fel.bhattacharyya(probs[names[i]], probs[names[j]])
        ax.imshow(M, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(n)); ax.set_xticklabels(names, fontsize=7, rotation=30)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, '%.2f' % M[i, j], ha='center', va='center', fontsize=8,
                        color='w' if M[i, j] < 0.6 else 'k')
        ax.set_title('%s: replica overlap' % system, fontsize=8.5)
        ax.text(-0.20, 1.04, string.ascii_uppercase[L], transform=ax.transAxes,
                fontsize=10, weight='bold')
        L += 1

    ax = axes[2]
    for rep in REPS:
        rr = [r for r in rows if r['replica'] == rep]
        T = np.array([float(r['T']) for r in rr]) / 1000.0
        B = np.array([float(r['bhattacharyya_vs_full']) for r in rr])
        ax.plot(T, B, 'o-', ms=4, lw=1.1, label=rep)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('trajectory length (us)', fontsize=8)
    ax.set_ylabel('Bhattacharyya vs full landscape', fontsize=8)
    ax.set_title('Traj-MSTN:ActRII: overlap vs length', fontsize=8.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.text(-0.16, 1.04, string.ascii_uppercase[L], transform=ax.transAxes,
            fontsize=10, weight='bold')
    L += 1

    fig.suptitle('Statistical basis of the projected landscapes (cMD replicas)', fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'Fig_S10_convergence_metrics')


# -------------------------------------------------------------------- Fig. S11
def fig_s11():
    print('[Fig. S11] GaMD sampling densities')
    systems = ['Pre-active-MSTN', 'Pre-active-MSTN-ActRII', 'APO-MSTN', 'APO-MSTN-ActRII']
    datas = {s: load_gamd(s) for s in systems}
    pooled = {k: np.concatenate([datas[s][k] for s in systems]) for k in ('angle1', 'd1', 'd2')}
    set_grid((36, 165, 3), (2, 39, 1), (4, 26, 1))
    fig = plt.figure(figsize=(11.0, 5.2))
    gs = GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.32, left=0.06, right=0.92,
                  top=0.90, bottom=0.10)
    L = 0
    for c, s in enumerate(systems):
        ax = fig.add_subplot(gs[0, c])
        im = panel(ax, datas[s], 'a1d1', 6.0, string.ascii_uppercase[L],
                   title='%s\n%.0f ns, %d frames' % (s, datas[s]['time'].max(), len(datas[s]['time'])))
        ax.set_xlabel('Angle-1 (deg)', fontsize=7.5) if c >= 0 else None
        L += 1
    cax = fig.add_axes([0.935, 0.55, 0.011, 0.34])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('-kT ln p (kcal/mol)', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # cumulative overlap of the 2 us system
    ax = fig.add_subplot(gs[1, 0:2])
    rows = read_csv(os.path.join(GANAL, 'gamd_cumulative.csv'))
    rr = [r for r in rows if r['system'] == 'Pre-active-MSTN']
    T = np.array([float(r['T']) for r in rr]) / 1000.0
    B1 = np.array([float(r['bhattacharyya_a1d1']) for r in rr])
    B2 = np.array([float(r['bhattacharyya_d1d2']) for r in rr])
    ax.plot(np.r_[0, T], np.r_[0, B1], 'o-', ms=4, label='Angle-1 - D1')
    ax.plot(np.r_[0, T], np.r_[0, B2], 's-', ms=4, label='D1 - D2')
    ax.set_xlabel('GaMD trajectory length (us)', fontsize=8)
    ax.set_ylabel('Bhattacharyya vs 2 us', fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.set_title('sampling density stabilises within ~1 us', fontsize=8.5)
    ax.text(-0.12, 1.05, 'E', transform=ax.transAxes, fontsize=10, weight='bold')

    ax = fig.add_subplot(gs[1, 2:4])
    ax.axis('off')
    txt = ('GaMD ensembles are shown as sampling density\n'
           '(-k_BT ln p), not as free energy.\n\n'
           'The harmonicity factor gamma is used only to check\n'
           'the applicability of second-order cumulant reweighting;\n'
           'it is not a convergence measure.\n\n'
           'Quantitative landscape statistics are derived from the\n'
           'independent cMD replicas (Fig. S8-S10).')
    ax.text(0.0, 0.95, txt, fontsize=7.5, va='top')
    ax.text(-0.05, 1.05, 'F', transform=ax.transAxes, fontsize=10, weight='bold')
    save(fig, 'Fig_S11_gamd_density')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    fig_s8()
    fig_s9()
    fig_s10()
    fig_s11()
    print('done ->', OUT)


if __name__ == '__main__':
    main()
