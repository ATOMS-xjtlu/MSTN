"""gamd_final.py -- final GaMD deliverables for R2-4.

Produces, in _work/gamd_analysis/:
  gamd_diagnostics.csv            sigma_dV/kT, gamma, ESS, weight concentration (whole run and 500 ns blocks)
  gamd_basins.csv                 basin centres with weighted and raw populations
  gamd_blocks.csv                 raw basin populations per 500 ns block (with bootstrap CI)
  gamd_cumulative.csv             block/cumulative landscape overlap (Bhattacharyya)
  gamd_vs_cmd.csv                 overlap between the GaMD landscape and the pooled cMD landscape
  fig_density_a1d1.png            4 systems, Angle-1 - Distance-1 sampling density
  fig_density_d1d2.png            4 systems, Distance-1 - Distance-2 sampling density
  fig_reweighting_diagnostic.png  weight concentration and ESS (why reweighting is not usable)
  fig_cumulative_preactive.png    cumulative density + overlap for the 2 us system
  fig_gamd_vs_cmd.png             GaMD vs cMD landscape overlap per system
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import fel
import gamd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CVDIR = os.path.join('_work', 'gamd_cvs')
CMDDIR = os.path.join('_work', 'cvs')
OUT = os.path.join('_work', 'gamd_analysis')
PROJ = {'a1d1': ('angle1', 'd1', 'Angle-1 (deg)', 'Distance-1 (A)'),
        'd1d2': ('d1', 'd2', 'Distance-1 (A)', 'Distance-2 (A)')}
SYSTEMS = ['Pre-active-MSTN', 'Pre-active-MSTN-ActRII', 'APO-MSTN', 'APO-MSTN-ActRII']
REPS = ('main', 'replica1', 'replica2')


def load(system):
    d = np.loadtxt(os.path.join(CVDIR, system + '.dat'))
    dv = np.loadtxt(os.path.join(CVDIR, system + '_dv.dat'))
    return dict(time=d[:, 0], angle1=d[:, 1], d1=d[:, 2], d2=d[:, 3],
                seg=d[:, 4].astype(int), dv=dv[:, 1])


def wprob(data, x, y, w):
    xe, ye = fel.edges(x), fel.edges(y)
    h, _, _ = np.histogram2d(data[x], data[y], bins=(xe, ye), weights=w)
    p = h / max(h.sum(), 1e-300)
    m = p > 0
    dg = np.full(p.shape, np.nan)
    dg[m] = np.minimum(-gamd.KT * np.log(p[m]) + gamd.KT * np.log(p[m].max()), fel.DG_CAP)
    return dg, p


def basin_pop(data, centres, w=None):
    X = np.column_stack([data['angle1'], data['d1'], data['d2']])
    d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    k = len(centres)
    if w is None:
        w = np.ones(len(lab)) / len(lab)
    pop = np.array([w[lab == j].sum() for j in range(k)])
    return pop / max(pop.sum(), 1e-300), lab


def block_ci(lab, k, nboot=400, block=500, seed=0):
    rng = np.random.default_rng(seed)
    out = np.zeros((nboot, k))
    n = len(lab)
    nb = max(1, n // block)
    for i in range(nboot):
        starts = rng.integers(0, max(1, n - block), nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out[i] = np.bincount(lab[idx], minlength=k) / len(idx)
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)

    diag_rows, basin_rows, block_rows, cum_rows, cmpmd_rows = [], [], [], [], []
    store = {}

    # one grid for every GaMD system, so panels and overlaps are comparable
    pooled = load(SYSTEMS[0])
    for s in SYSTEMS[1:]:
        d = load(s)
        pooled = {k: np.concatenate([pooled[k], d[k]]) for k in ('angle1', 'd1', 'd2')}
    fel.auto_grid(pooled)
    print('grid:', {k: (v['lo'], v['hi'], v['step']) for k, v in fel.GRID.items()})

    for system in SYSTEMS:
        data = load(system)
        w = gamd.weights(data['dv'])
        diag = gamd.diagnostics(data['dv'], w)
        basins = fel.define_basins(fel.stack([data]), k=4, seed=0)
        centres = basins['centres']
        pop_w, lab = basin_pop(data, centres, w)
        pop_r, _ = basin_pop(data, centres, None)

        diag_rows.append(dict(system=system, frames=diag['n_frames'],
                              dv_mean=diag['dv_mean'], dv_sd=diag['dv_sd'],
                              sigma_over_kT=diag['sigma_over_kT'], gamma=diag['gamma'],
                              ess=diag['ess'], ess_percent=100 * diag['ess_frac'],
                              top1_weight=diag['top1_weight'], top10_weight=diag['top10_weight']))
        for j in range(4):
            basin_rows.append(dict(system=system, basin='B%d' % j,
                                   angle1=float(centres[j][0]), d1=float(centres[j][1]),
                                   d2=float(centres[j][2]),
                                   pop_raw=float(pop_r[j]), pop_weighted=float(pop_w[j])))

        # 500 ns blocks: raw populations with bootstrap CI
        tmax = float(data['time'].max())
        for b0 in np.arange(0.0, tmax, 500.0):
            b1 = min(b0 + 500.0, tmax)
            m = (data['time'] >= b0) & (data['time'] < b1)
            if m.sum() < 100:
                continue
            sub = {k: v[m] for k, v in data.items()}
            pb, lb = basin_pop(sub, centres, None)
            lo, hi = block_ci(lb, 4)
            row = dict(system=system, t0=b0, t1=b1, frames=int(m.sum()))
            for j in range(4):
                row['pop_B%d' % j] = pb[j]
                row['pop_B%d_lo' % j] = lo[j]
                row['pop_B%d_hi' % j] = hi[j]
            block_rows.append(row)

        # cumulative overlap of the *density* landscape (reweighting cannot be used)
        p_full = {proj: wprob(data, *PROJ[proj][:2], np.ones(len(data['time'])))[1]
                  for proj in PROJ}
        for frac in (0.25, 0.5, 0.75, 1.0):
            m = data['time'] <= tmax * frac
            sub = {k: v[m] for k, v in data.items()}
            row = dict(system=system, T=float(sub['time'].max()), frames=int(m.sum()))
            for proj in PROJ:
                p = wprob(sub, *PROJ[proj][:2], np.ones(len(sub['time'])))[1]
                row['bhattacharyya_%s' % proj] = fel.bhattacharyya(p, p_full[proj])
            cum_rows.append(row)

        # overlap with the pooled cMD landscape of the same system
        cmd_files = [os.path.join(CMDDIR, '%s__%s.dat' % (system, r)) for r in REPS]
        cmd_files = [p for p in cmd_files if os.path.exists(p)]
        if cmd_files:
            cmd = fel.stack([fel.decimate(fel.load(p)) for p in cmd_files])
            row = dict(system=system, cmd_replicas=len(cmd_files), frames_cmd=int(len(cmd['time'])))
            for proj, (x, y, _, _) in PROJ.items():
                p_g = wprob(data, x, y, np.ones(len(data['time'])))[1]
                p_c = fel.prob2d(fel.hist2d(cmd[x], cmd[y], x, y))
                row['bhattacharyya_%s' % proj] = fel.bhattacharyya(p_g, p_c)
            cmpmd_rows.append(row)

        store[system] = dict(data=data, w=w, diag=diag, centres=centres)
        print('[%s] ESS=%.1f (%.3f%%) top1=%.2f sigma/kT=%.2f gamma=%.3f'
              % (system, diag['ess'], 100 * diag['ess_frac'], diag['top1_weight'],
                 diag['sigma_over_kT'], diag['gamma']))

    # ---------------------------------------------------------------- figures
    def density_figure(proj, fname):
        x, y, xl, yl = PROJ[proj]
        fig, axes = plt.subplots(1, 4, figsize=(13.2, 2.9))
        im = None
        for ax, system in zip(axes, SYSTEMS):
            s = store[system]
            dg, _ = wprob(s['data'], x, y, np.ones(len(s['data']['time'])))
            ax.set_facecolor('#eeeeee')
            im = ax.pcolormesh(fel.edges(x), fel.edges(y), dg.T, cmap='jet',
                               vmin=0, vmax=fel.DG_CAP, shading='auto')
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.set_title('%s\n%.0f ns, %d frames' % (system, s['data']['time'].max(),
                                                     len(s['data']['time'])), fontsize=8)
            ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.01)
        cb.set_label('-kT ln p  (kcal/mol, sampling density)', fontsize=7)
        fig.savefig(os.path.join(OUT, fname), dpi=180, bbox_inches='tight')
        plt.close(fig)

    density_figure('a1d1', 'fig_density_a1d1.png')
    density_figure('d1d2', 'fig_density_d1d2.png')

    # reweighting diagnostic: sorted cumulative weight curves
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for system in SYSTEMS:
        s = store[system]
        ws = np.sort(s['w'])[::-1]
        cum = np.cumsum(ws)
        axes[0].plot(np.arange(1, len(cum) + 1), cum, label=system, lw=1.2)
        axes[1].plot(np.arange(1, 201), cum[:200], label=system, lw=1.2)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('frames (sorted by weight)')
    axes[0].set_ylabel('cumulative weight')
    axes[0].set_title('weights are carried by a handful of frames', fontsize=9)
    axes[1].set_xlabel('top-N frames')
    axes[1].set_title('zoom: first 200 frames', fontsize=9)
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle('GaMD exponential reweighting: effective sample size = 1-3 frames', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_reweighting_diagnostic.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    # cumulative figure for the 2 us system
    system = 'Pre-active-MSTN'
    s = store[system]
    data = s['data']
    tmax = float(data['time'].max())
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 2.9))
    p_full = wprob(data, 'angle1', 'd1', np.ones(len(data['time'])))[1]
    im = None
    for ax, frac in zip(axes[:4], (0.25, 0.5, 0.75, 1.0)):
        m = data['time'] <= tmax * frac
        sub = {k: v[m] for k, v in data.items()}
        dg, p = wprob(sub, 'angle1', 'd1', np.ones(len(sub['time'])))
        ax.set_facecolor('#eeeeee')
        im = ax.pcolormesh(fel.edges('angle1'), fel.edges('d1'), dg.T, cmap='jet',
                           vmin=0, vmax=fel.DG_CAP, shading='auto')
        ax.set_title('%.1f us  B=%.2f' % (sub['time'].max() / 1000.0,
                                          fel.bhattacharyya(p, p_full)), fontsize=8)
        ax.set_xlabel('Angle-1 (deg)', fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel('Distance-1 (A)', fontsize=8)
    T = np.array([r['T'] for r in cum_rows if r['system'] == system]) / 1000.0
    B = np.array([r['bhattacharyya_a1d1'] for r in cum_rows if r['system'] == system])
    axes[4].plot(np.r_[0, T], np.r_[0, B], 'o-', color='C0')
    axes[4].set_xlabel('time (us)')
    axes[4].set_ylabel('Bhattacharyya vs 2 us')
    axes[4].set_ylim(0, 1.02)
    axes[4].grid(alpha=0.3)
    axes[4].set_title('density overlap', fontsize=8)
    fig.colorbar(im, ax=list(axes[:4]), fraction=0.015, pad=0.01).set_label('-kT ln p', fontsize=7)
    fig.suptitle('%s   cumulative sampling density' % system, fontsize=9)
    fig.savefig(os.path.join(OUT, 'fig_cumulative_preactive.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    # GaMD vs cMD overlap
    if cmpmd_rows:
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        xs = np.arange(len(cmpmd_rows))
        ax.bar(xs - 0.2, [r['bhattacharyya_a1d1'] for r in cmpmd_rows], 0.4, label='Angle-1 - D1')
        ax.bar(xs + 0.2, [r['bhattacharyya_d1d2'] for r in cmpmd_rows], 0.4, label='D1 - D2')
        ax.set_xticks(xs)
        ax.set_xticklabels([r['system'].replace('-MSTN', '').replace('ActRII', 'ActRII')
                            for r in cmpmd_rows], fontsize=6, rotation=20)
        ax.set_ylabel('Bhattacharyya (GaMD vs cMD)')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'fig_gamd_vs_cmd.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)

    # ---------------------------------------------------------------- tables
    def dump(name, rows):
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(os.path.join(OUT, name), 'w', encoding='utf-8') as f:
            f.write(','.join(keys) + '\n')
            for r in rows:
                f.write(','.join('%.6g' % r[k] if isinstance(r[k], (float, np.floating))
                                 else str(r[k]) for k in keys) + '\n')

    dump('gamd_diagnostics.csv', diag_rows)
    dump('gamd_basins.csv', basin_rows)
    dump('gamd_blocks.csv', block_rows)
    dump('gamd_cumulative.csv', cum_rows)
    dump('gamd_vs_cmd.csv', cmpmd_rows)
    print('\noutputs in', OUT)


if __name__ == '__main__':
    main()
