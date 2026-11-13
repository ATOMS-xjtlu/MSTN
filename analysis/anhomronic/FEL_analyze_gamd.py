"""analyze_gamd.py -- reweighted GaMD landscapes and their statistical support.

For every GaMD system this writes into _work/gamd_analysis/:
    <system>_diagnostics.csv    sigma_dV/kT, gamma, ESS, weight concentration (whole run + 500 ns blocks)
    <system>_blocks.csv         weighted basin populations per 500 ns block
    <system>_cumulative.csv     weighted populations / overlap vs trajectory length
    fig_<system>_fel_<proj>.png reweighted FEL next to the raw sampling density
    fig_<system>_cumulative.png cumulative reweighted landscape (2 us system)

The point of the density panel is honesty: with ESS of a few frames the reweighted
surface is not a converged free-energy landscape, and the figure shows both.
"""
import argparse
import json
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
OUT = os.path.join('_work', 'gamd_analysis')
PROJ = {'a1d1': ('angle1', 'd1', 'Angle-1 (deg)', 'Distance-1 (A)'),
        'd1d2': ('d1', 'd2', 'Distance-1 (A)', 'Distance-2 (A)')}
SYSTEMS = ['Pre-active-MSTN', 'Pre-active-MSTN-ActRII', 'APO-MSTN', 'APO-MSTN-ActRII']


def load(system):
    d = np.loadtxt(os.path.join(CVDIR, system + '.dat'))
    dv = np.loadtxt(os.path.join(CVDIR, system + '_dv.dat'))
    assert len(d) == len(dv), (system, len(d), len(dv))
    out = dict(time=d[:, 0], angle1=d[:, 1], d1=d[:, 2], d2=d[:, 3], seg=d[:, 4].astype(int))
    out['dv'] = dv[:, 1]
    return out


def wfel(data, x, y, w):
    """Weighted 2-D free energy on the global fel grid, plus the normalised probability."""
    xe, ye = fel.edges(x), fel.edges(y)
    h, _, _ = np.histogram2d(data[x], data[y], bins=(xe, ye), weights=w)
    p = h / max(h.sum(), 1e-300)
    out = np.full(h.shape, np.nan)
    m = h > 0
    dg = -gamd.KT * np.log(np.where(m, p, 1.0))
    dg[~m] = np.nan
    if np.isfinite(dg).any():
        dg = np.minimum(dg - np.nanmin(dg), fel.DG_CAP)
    out[:] = dg
    return out, p


def wpop(data, centres, w):
    X = np.column_stack([data['angle1'], data['d1'], data['d2']])
    d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    k = len(centres)
    pop = np.array([w[lab == j].sum() for j in range(k)])
    return pop / max(pop.sum(), 1e-300), lab


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', type=float, default=500.0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    summary = {}
    for system in SYSTEMS:
        if not os.path.exists(os.path.join(CVDIR, system + '.dat')):
            continue
        data = load(system)
        w = gamd.weights(data['dv'])
        diag = gamd.diagnostics(data['dv'], w)
        print('\n[%s] frames=%d  sigma/kT=%.2f  gamma=%.3f  ESS=%.1f (%.3f%%)  '
              'top1=%.2f top10=%.3f'
              % (system, diag['n_frames'], diag['sigma_over_kT'], diag['gamma'],
                 diag['ess'], 100 * diag['ess_frac'], diag['top1_weight'], diag['top10_weight']))

        fel.auto_grid(data)
        # basins from the sampled ensemble (unweighted), then weighted populations
        basins = fel.define_basins(fel.stack([data]), k=4, seed=0)
        centres = basins['centres']
        pop, lab = wpop(data, centres, w)
        pop_u = np.bincount(fel.assign(data, basins), minlength=4) / len(data)
        print('  basins:' + '  '.join('B%d(a=%.0f,d1=%.1f,d2=%.1f) w=%.3f raw=%.3f'
                                     % (j, centres[j][0], centres[j][1], centres[j][2],
                                        pop[j], pop_u[j]) for j in range(4)))

        # per-block diagnostics
        block_rows = []
        tmax = float(data['time'].max())
        for b0 in np.arange(0.0, tmax, args.blocks):
            b1 = min(b0 + args.blocks, tmax)
            m = (data['time'] >= b0) & (data['time'] < b1)
            if m.sum() < 50:
                continue
            sub = {k: v[m] for k, v in data.items()}
            wb = gamd.weights(sub['dv'])
            db = gamd.diagnostics(sub['dv'], wb)
            pb, _ = wpop(sub, centres, wb)
            row = dict(system=system, t0=b0, t1=b1, frames=db['n_frames'],
                       sigma_over_kT=db['sigma_over_kT'], gamma=db['gamma'],
                       ess=db['ess'], ess_frac=db['ess_frac'], top1=db['top1_weight'])
            for j in range(4):
                row['pop_B%d' % j] = pb[j]
            block_rows.append(row)
            print('  block %4.0f-%4.0f ns: ESS=%5.1f top1=%.2f  w-pop=%s'
                  % (b0, b1, db['ess'], db['top1_weight'],
                     ' '.join('%.2f' % x for x in pb)))

        # cumulative
        cum_rows = []
        wf_full = wfel(data, 'angle1', 'd1', w)[1]
        for frac in (0.25, 0.5, 0.75, 1.0):
            m = data['time'] <= tmax * frac
            sub = {k: v[m] for k, v in data.items()}
            wc = gamd.weights(sub['dv'])
            pc = wfel(sub, 'angle1', 'd1', wc)[1]
            dc = gamd.diagnostics(sub['dv'], wc)
            row = dict(system=system, T=float(sub['time'].max()), frames=dc['n_frames'],
                       ess=dc['ess'], bhattacharyya_vs_full=fel.bhattacharyya(pc, wf_full))
            cum_rows.append(row)

        # figures: reweighted FEL vs sampling density
        for proj, (x, y, xl, yl) in PROJ.items():
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
            f_w, _ = wfel(data, x, y, w)
            f_u, _ = wfel(data, x, y, np.ones_like(w) / len(w))
            xe, ye = fel.edges(x), fel.edges(y)
            for ax, f, title in ((axes[0], f_u, 'sampling density (-kT ln p)'),
                                 (axes[1], f_w, 'reweighted (ESS=%.1f frames)' % diag['ess'])):
                ax.set_facecolor('#eeeeee')
                im = ax.pcolormesh(xe, ye, f.T, cmap='jet', vmin=0, vmax=fel.DG_CAP, shading='auto')
                ax.set_xlabel(xl, fontsize=8)
                ax.set_ylabel(yl, fontsize=8)
                ax.set_title(title, fontsize=8)
                ax.tick_params(labelsize=7)
            cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
            cb.set_label('dG (kcal/mol)', fontsize=7)
            fig.suptitle('%s   %s   sigma_dV/kT=%.1f  gamma=%.3f'
                         % (system, proj, diag['sigma_over_kT'], diag['gamma']), fontsize=9)
            fig.savefig(os.path.join(OUT, 'fig_%s_fel_%s.png' % (system, proj)),
                        dpi=180, bbox_inches='tight')
            plt.close(fig)

        # cumulative figure for the 2 us system
        if tmax > 1500:
            fig, axes = plt.subplots(1, 4, figsize=(13.0, 2.9))
            wf = wfel(data, 'angle1', 'd1', w)[1]
            for ax, frac in zip(axes, (0.25, 0.5, 0.75, 1.0)):
                m = data['time'] <= tmax * frac
                sub = {k: v[m] for k, v in data.items()}
                wc = gamd.weights(sub['dv'])
                f, p = wfel(sub, 'angle1', 'd1', wc)
                ax.set_facecolor('#eeeeee')
                im = ax.pcolormesh(fel.edges('angle1'), fel.edges('d1'), f.T,
                                   cmap='jet', vmin=0, vmax=fel.DG_CAP, shading='auto')
                ax.set_title('%.1f us  B=%.2f  ESS=%.0f'
                             % (sub['time'].max() / 1000.0, fel.bhattacharyya(p, wf),
                                gamd.diagnostics(sub['dv'], wc)['ess']), fontsize=8)
                ax.set_xlabel('Angle-1 (deg)', fontsize=8)
                ax.tick_params(labelsize=7)
            axes[0].set_ylabel('Distance-1 (A)', fontsize=8)
            fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.01).set_label('dG (kcal/mol)', fontsize=7)
            fig.suptitle('%s   cumulative reweighted landscape' % system, fontsize=9)
            fig.savefig(os.path.join(OUT, 'fig_%s_cumulative.png' % system),
                        dpi=180, bbox_inches='tight')
            plt.close(fig)

        summary[system] = dict(diagnostics=diag,
                               basins=[dict(basin='B%d' % j,
                                            angle1=float(centres[j][0]),
                                            d1=float(centres[j][1]),
                                            d2=float(centres[j][2]),
                                            pop_weighted=float(pop[j]),
                                            pop_raw=float(pop_u[j])) for j in range(4)],
                               blocks=block_rows, cumulative=cum_rows)

    with open(os.path.join(OUT, 'gamd_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # combined diagnostics table
    rows = []
    for system, s in summary.items():
        d = s['diagnostics']
        rows.append(dict(system=system, frames=d['n_frames'], dv_mean=d['dv_mean'],
                         dv_sd=d['dv_sd'], sigma_over_kT=d['sigma_over_kT'],
                         gamma=d['gamma'], ess=d['ess'], ess_percent=100 * d['ess_frac'],
                         top1_weight=d['top1_weight'], top10_weight=d['top10_weight']))
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, 'gamd_diagnostics.csv'), 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join('%.6g' % r[k] if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
    print('\noutputs in', OUT)


if __name__ == '__main__':
    main()
