"""analyze.py (v2) -- convergence statistics for the projected cMD landscapes.

Per system it writes into _work/analysis/:
    <system>_basins.csv        basin centres + physical descriptors
    <system>_per_replica.csv   populations / dG / CI / transitions / dwell / ACT
    <system>_across_replica.csv mean+-SD across the three replicas
    <system>_blocks.csv        populations per 500 ns block
    <system>_cumulative.csv    populations, dG and landscape metrics vs length
    <system>_overlap.csv       pairwise Bhattacharyya / JS between replicas
    fig_<system>_fel_<proj>_replicas.png
    fig_<system>_fel_<proj>_cumulative.png
    fig_<system>_dg_vs_time.png
    fig_<system>_blocks.png
    fig_<system>_overlap.png

Conventions (documented in the paper text):
  * fixed CV grids for every system / replica / block / cumulative subset;
  * basins = k-means on the pooled ensemble of a system, centres frozen afterwards;
  * transitions and dwell times are counted on a 1 ns resampling to suppress
    bin-boundary flicker;
  * bootstrap uses moving blocks of max(2*ACT(Distance-1), 50 ns), capped at T/5,
    and the across-replica spread is reported separately.
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import fel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CVDIR = os.path.join('_work', 'cvs')
OUT = os.path.join('_work', 'analysis')
REPLICAS = ('main', 'replica1', 'replica2')
PROJ = {'a1d1': ('angle1', 'd1', 'Angle-1 (deg)', 'Distance-1 (A)'),
        'd1d2': ('d1', 'd2', 'Distance-1 (A)', 'Distance-2 (A)')}
CT = 'jet'


def load_replicas(system, dt_max=0.25):
    out = {}
    for rep in REPLICAS:
        p = os.path.join(CVDIR, '%s__%s.dat' % (system, rep))
        if os.path.exists(p):
            out[rep] = fel.decimate(fel.load(p), dt_max=dt_max)
    return out


def resample_labels(data, basins, dt=1.0):
    """Basin labels on a coarse grid, used for transition / dwell statistics."""
    lab = fel.assign(data, basins)
    t = data['time']
    grid = np.arange(t.min(), t.max() + 1e-9, dt)
    idx = np.searchsorted(t, grid).clip(0, len(t) - 1)
    return lab[idx], grid


def dwell_stats(lab, grid, k):
    """Mean dwell time (ns) and number of visits per basin on a coarse label track."""
    out = []
    if len(lab) < 2:
        return [dict(basin=j, visits=0, mean_dwell=np.nan, total_ns=0.0) for j in range(k)]
    starts = np.r_[0, np.where(lab[1:] != lab[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(lab)]
    dt = grid[1] - grid[0] if len(grid) > 1 else 1.0
    for j in range(k):
        durs = [(ends[i] - starts[i]) * dt for i in range(len(starts)) if lab[starts[i]] == j]
        out.append(dict(basin=j, visits=len(durs), total_ns=float(np.sum(durs)),
                        mean_dwell=float(np.mean(durs)) if durs else np.nan))
    return out


def analyse(system, reps_data, k, blocks_ns=500.0, nboot=1000, seed=0, log=print):
    pooled = fel.stack([reps_data[r] for r in reps_data])
    fel.auto_grid(pooled)
    basins = fel.define_basins(pooled, k=k, seed=seed)
    centres = basins['centres']
    lab_pool = fel.assign(pooled, basins)

    basin_rows = []
    for j in range(k):
        m = lab_pool == j
        basin_rows.append(dict(
            basin='B%d' % j, n_frames=int(m.sum()),
            pop_overall=float(m.mean()) if len(m) else 0.0,
            angle1=float(pooled['angle1'][m].mean()) if m.any() else np.nan,
            d1=float(pooled['d1'][m].mean()) if m.any() else np.nan,
            d2=float(pooled['d2'][m].mean()) if m.any() else np.nan,
            angle1_sd=float(pooled['angle1'][m].std()) if m.any() else np.nan,
            d1_sd=float(pooled['d1'][m].std()) if m.any() else np.nan))

    per_rep = []
    probs_a1d1 = {}
    probs_d1d2 = {}
    for rep, data in reps_data.items():
        lab = fel.assign(data, basins)
        pop = fel.populations(lab, k)
        dg = fel.delta_g(pop)
        dt = float(np.median(np.diff(data['time'])))
        act_a1 = fel.integrated_act(data['angle1'], dt) * dt
        act_d1 = fel.integrated_act(data['d1'], dt) * dt
        T = float(data['time'].max() - data['time'].min())
        block_ns = float(np.clip(2 * act_d1, 50.0, max(50.0, T / 5)))
        block = max(10, int(round(block_ns / dt)))
        bs = fel.block_bootstrap(lab, n_samples=nboot, block=block, seed=seed, k=k)
        lo, hi = fel.ci95(bs)
        dg_bs = np.array([fel.delta_g(row) for row in bs])
        dg_lo, dg_hi = fel.ci95(dg_bs)

        clab, grid = resample_labels(data, basins, dt=1.0)
        M = fel.transitions(clab, k)
        dwell = dwell_stats(clab, grid, k)

        row = dict(replica=rep, frames=int(len(lab)), T_ns=T, dt_ns=dt,
                   act_angle1_ns=act_a1, act_d1_ns=act_d1, block_ns=block_ns,
                   n_transitions=int(M.sum() - np.trace(M)))
        for j in range(k):
            row['pop_B%d' % j] = pop[j]
            row['pop_B%d_lo' % j] = lo[j]
            row['pop_B%d_hi' % j] = hi[j]
            row['dG_B%d' % j] = dg[j]
            row['dG_B%d_lo' % j] = dg_lo[j]
            row['dG_B%d_hi' % j] = dg_hi[j]
            row['visits_B%d' % j] = dwell[j]['visits']
            row['dwell_B%d_ns' % j] = dwell[j]['mean_dwell']
        for j, l in itertools.permutations(range(k), 2):
            row['trans_B%d_B%d' % (j, l)] = int(M[j, l])
        per_rep.append(row)
        probs_a1d1[rep] = fel.prob2d(fel.hist2d(data['angle1'], data['d1'], 'angle1', 'd1'))
        probs_d1d2[rep] = fel.prob2d(fel.hist2d(data['d1'], data['d2'], 'd1', 'd2'))

    # across-replica spread
    across = []
    for j in range(k):
        p = np.array([r['pop_B%d' % j] for r in per_rep])
        g = np.array([r['dG_B%d' % j] for r in per_rep])
        across.append(dict(basin='B%d' % j, pop_mean=p.mean(), pop_sd=p.std(ddof=1) if len(p) > 1 else 0.0,
                           pop_min=p.min(), pop_max=p.max(),
                           dG_mean=g.mean(), dG_sd=g.std(ddof=1) if len(g) > 1 else 0.0))

    overlap = []
    for a, b in itertools.combinations(probs_a1d1, 2):
        overlap.append(dict(proj='a1d1', rep_a=a, rep_b=b,
                            bhattacharyya=fel.bhattacharyya(probs_a1d1[a], probs_a1d1[b]),
                            js=fel.js_divergence(probs_a1d1[a], probs_a1d1[b])))
        overlap.append(dict(proj='d1d2', rep_a=a, rep_b=b,
                            bhattacharyya=fel.bhattacharyya(probs_d1d2[a], probs_d1d2[b]),
                            js=fel.js_divergence(probs_d1d2[a], probs_d1d2[b])))

    # blocks
    block_rows = []
    for rep, data in reps_data.items():
        tmax = float(data['time'].max())
        for b0 in np.arange(0.0, tmax + 1e-9, blocks_ns):
            b1 = min(b0 + blocks_ns, tmax)
            sub = fel.subset_time(data, b0, b1)
            if len(sub['time']) < 50:
                continue
            lab = fel.assign(sub, basins)
            pop = fel.populations(lab, k)
            row = dict(replica=rep, t0=b0, t1=b1, frames=int(len(sub['time'])))
            for j in range(k):
                row['pop_B%d' % j] = pop[j]
            block_rows.append(row)

    # cumulative
    cum_rows = []
    for rep, data in reps_data.items():
        tmax = float(data['time'].max())
        f_full = fel.free_energy(fel.hist2d(data['angle1'], data['d1'], 'angle1', 'd1'))
        for frac in (0.25, 0.5, 0.75, 1.0):
            sub = fel.subset_time(data, None, tmax * frac)
            if len(sub['time']) < 100:
                continue
            lab = fel.assign(sub, basins)
            pop = fel.populations(lab, k)
            dg = fel.delta_g(pop)
            p = fel.prob2d(fel.hist2d(sub['angle1'], sub['d1'], 'angle1', 'd1'))
            f_sub = fel.free_energy(fel.hist2d(sub['angle1'], sub['d1'], 'angle1', 'd1'))
            row = dict(replica=rep, T=tmax * frac, frames=int(len(sub['time'])),
                       bhattacharyya_vs_full=fel.bhattacharyya(p, probs_a1d1[rep]),
                       js_vs_full=fel.js_divergence(p, probs_a1d1[rep]),
                       fel_rms_vs_full=fel.fel_rms(f_sub, f_full))
            for j in range(k):
                row['pop_B%d' % j] = pop[j]
                row['dG_B%d' % j] = dg[j]
            cum_rows.append(row)

    return dict(basins=basins, centres=centres, basin_rows=basin_rows,
                per_replica=per_rep, across=across, overlap=overlap,
                blocks=block_rows, cumulative=cum_rows, pooled=pooled,
                probs=dict(a1d1=probs_a1d1, d1d2=probs_d1d2))


# --------------------------------------------------------------------- outputs
def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(('%.6g' % r[k]) if isinstance(r[k], (float, np.floating)) else str(r[k])
                             for k in keys) + '\n')


def _fel_panel(ax, data, x, y, vmax, xl, yl, title):
    xe, ye = fel.edges(x), fel.edges(y)
    h = fel.hist2d(data[x], data[y], x, y)
    f = fel.free_energy(h)
    ax.set_facecolor('#eeeeee')
    im = ax.pcolormesh(xe, ye, f.T, cmap=CT, vmin=0, vmax=vmax, shading='auto')
    ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]),
               np.nan_to_num(f.T, nan=vmax), levels=[1, 2, 4], colors='w',
               linewidths=0.5, alpha=0.8)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return im


def plot_replicas(system, res, proj, reps_data, outdir):
    x, y, xl, yl = PROJ[proj]
    names = list(reps_data)
    pooled = res['pooled']
    f_pool = fel.free_energy(fel.hist2d(pooled[x], pooled[y], x, y))
    vmax = float(6.0)
    fig = plt.figure(figsize=(3.1 * (len(names) + 1) + 0.8, 3.0))
    gs = fig.add_gridspec(1, len(names) + 2, width_ratios=[1] * (len(names) + 1) + [0.05],
                          wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(names) + 1)]
    im = None
    for ax, name in zip(axes, names):
        im = _fel_panel(ax, reps_data[name], x, y, vmax, xl, yl, name)
    im = _fel_panel(axes[-1], pooled, x, y, vmax, xl, yl, 'pooled')
    for j, c in enumerate(res['centres']):
        cx = c[0] if x == 'angle1' else c[1]
        cy = c[1] if y == 'd1' else c[2]
        axes[-1].plot(cx, cy, 'x', color='w', ms=6, mew=1.4)
        axes[-1].annotate('B%d' % j, (cx, cy), textcoords='offset points',
                          xytext=(5, -9), fontsize=7, color='w', weight='bold')
    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('dG (kcal/mol)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle('%s   %s projection   (basins: k-means on pooled frames, k=%d)'
                 % (system, proj, len(res['centres'])), fontsize=9)
    p = os.path.join(outdir, 'fig_%s_fel_%s_replicas.png' % (system, proj))
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return p


def plot_cumulative(system, res, proj, reps_data, outdir):
    x, y, xl, yl = PROJ[proj]
    rep = 'main' if 'main' in reps_data else list(reps_data)[0]
    data = reps_data[rep]
    tmax = float(data['time'].max())
    f_full = fel.free_energy(fel.hist2d(data[x], data[y], x, y))
    vmax = float(6.0)
    fracs = (0.25, 0.5, 0.75, 1.0)
    fig = plt.figure(figsize=(3.0 * 4 + 0.8, 2.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1] * 4 + [0.05], wspace=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    im = None
    for ax, fr in zip(axes, fracs):
        sub = fel.subset_time(data, None, tmax * fr)
        im = _fel_panel(ax, sub, x, y, vmax, xl, yl, '%.1f us' % (sub['time'].max() / 1000.0))
    cax = fig.add_subplot(gs[0, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('dG (kcal/mol)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle('%s   cumulative %s landscape (%s)' % (system, proj, rep), fontsize=9)
    p = os.path.join(outdir, 'fig_%s_fel_%s_cumulative.png' % (system, proj))
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return p


def plot_dg(system, res, outdir):
    k = len(res['centres'])
    reps = sorted({r['replica'] for r in res['cumulative']})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2))
    cmap = plt.get_cmap('tab10')
    for j in range(k):
        for rep in reps:
            rows = [r for r in res['cumulative'] if r['replica'] == rep]
            if not rows:
                continue
            T = np.array([r['T'] for r in rows]) / 1000.0
            dg = np.array([r['dG_B%d' % j] for r in rows])
            axes[0].plot(T, dg, 'o-', ms=3, lw=1, color=cmap(j % 10), alpha=0.9,
                         label='B%d' % j if rep == reps[0] else None)
    axes[0].set_xlabel('trajectory length (us)')
    axes[0].set_ylabel('dG (kcal/mol)')
    axes[0].set_title('basin free energy vs length', fontsize=9)
    axes[0].legend(fontsize=7)
    for rep in reps:
        rows = [r for r in res['cumulative'] if r['replica'] == rep]
        T = np.array([r['T'] for r in rows]) / 1000.0
        axes[1].plot(T, [r['bhattacharyya_vs_full'] for r in rows], 'o-', ms=3, lw=1, label=rep)
        axes[2].plot(T, [r['fel_rms_vs_full'] for r in rows], 'o-', ms=3, lw=1, label=rep)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title('Bhattacharyya overlap with full landscape', fontsize=9)
    axes[2].set_title('FEL RMS difference vs full (kcal/mol)', fontsize=9)
    for ax in axes[1:]:
        ax.set_xlabel('trajectory length (us)')
        ax.legend(fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.suptitle(system, fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, 'fig_%s_dg_vs_time.png' % system)
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return p


def plot_blocks(system, res, outdir):
    k = len(res['centres'])
    rows = res['blocks']
    if not rows:
        return None
    reps = sorted({r['replica'] for r in rows})
    fig, axes = plt.subplots(1, len(reps), figsize=(3.2 * len(reps), 2.8), squeeze=False)
    cmap = plt.get_cmap('tab10')
    for ax, rep in zip(axes[0], reps):
        rr = [r for r in rows if r['replica'] == rep]
        centers = [(r['t0'] + r['t1']) / 2 / 1000.0 for r in rr]
        bottom = np.zeros(len(rr))
        for j in range(k):
            vals = np.array([r['pop_B%d' % j] for r in rr])
            ax.bar(centers, vals, width=0.4, bottom=bottom, color=cmap(j % 10), label='B%d' % j)
            bottom += vals
        ax.set_title(rep, fontsize=9)
        ax.set_xlabel('time (us)')
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel('basin population')
    axes[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle('%s   500 ns blocks' % system, fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, 'fig_%s_blocks.png' % system)
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return p


def plot_overlap(system, res, outdir):
    reps = sorted({r['rep_a'] for r in res['overlap']} | {r['rep_b'] for r in res['overlap']})
    n = len(reps)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, proj in zip(axes, ('a1d1', 'd1d2')):
        M = np.ones((n, n))
        for r in res['overlap']:
            if r['proj'] != proj:
                continue
            i, j = reps.index(r['rep_a']), reps.index(r['rep_b'])
            M[i, j] = M[j, i] = r['bhattacharyya']
        im = ax.imshow(M, vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(n)); ax.set_xticklabels(reps, fontsize=7, rotation=45)
        ax.set_yticks(range(n)); ax.set_yticklabels(reps, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, '%.2f' % M[i, j], ha='center', va='center',
                        fontsize=7, color='w' if M[i, j] < 0.6 else 'k')
        ax.set_title('%s Bhattacharyya' % proj, fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02).set_label('overlap', fontsize=7)
    fig.suptitle('%s   replica-to-replica landscape overlap' % system, fontsize=9)
    p = os.path.join(outdir, 'fig_%s_overlap.png' % system)
    fig.savefig(p, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return p


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--nboot', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    systems = sorted({f.split('__')[0] for f in os.listdir(CVDIR) if f.endswith('.dat')})
    if args.only:
        want = set(args.only.split(','))
        systems = [s for s in systems if s in want]

    summary = {}
    for system in systems:
        reps_data = load_replicas(system)
        if not reps_data:
            continue
        print('\n[%s] replicas: %s' % (system, ', '.join(reps_data)))
        res = analyse(system, reps_data, args.k, nboot=args.nboot, seed=args.seed)
        write_csv(os.path.join(OUT, '%s_basins.csv' % system), res['basin_rows'])
        write_csv(os.path.join(OUT, '%s_per_replica.csv' % system), res['per_replica'])
        write_csv(os.path.join(OUT, '%s_across_replica.csv' % system), res['across'])
        write_csv(os.path.join(OUT, '%s_blocks.csv' % system), res['blocks'])
        write_csv(os.path.join(OUT, '%s_cumulative.csv' % system), res['cumulative'])
        write_csv(os.path.join(OUT, '%s_overlap.csv' % system), res['overlap'])
        for proj in PROJ:
            plot_replicas(system, res, proj, reps_data, OUT)
            plot_cumulative(system, res, proj, reps_data, OUT)
        plot_dg(system, res, OUT)
        plot_blocks(system, res, OUT)
        plot_overlap(system, res, OUT)
        summary[system] = dict(basins=res['basin_rows'], per_replica=res['per_replica'],
                               across=res['across'], overlap=res['overlap'])

        for b in res['basin_rows']:
            print('  %s  angle=%5.1f  d1=%5.2f  d2=%5.2f  pop=%5.3f  n=%d'
                  % (b['basin'], b['angle1'], b['d1'], b['d2'], b['pop_overall'], b['n_frames']))
        for r in res['per_replica']:
            print('  %-9s T=%5.0f ns  ACT(angle)=%5.0f  ACT(d1)=%4.0f  block=%4.0f ns  trans=%3d  %s'
                  % (r['replica'], r['T_ns'], r['act_angle1_ns'], r['act_d1_ns'],
                     r['block_ns'], r['n_transitions'],
                     ' '.join('B%d=%.2f[%.2f,%.2f]' % (j, r['pop_B%d' % j],
                                                       r['pop_B%d_lo' % j], r['pop_B%d_hi' % j])
                              for j in range(args.k))))
        print('  across replicas: ' + '  '.join(
            'B%d=%.2f+-%.2f' % (i, a['pop_mean'], a['pop_sd']) for i, a in enumerate(res['across'])))
        print('  overlap: ' + '  '.join('%s/%s %s=%.2f' % (o['rep_a'], o['rep_b'], o['proj'],
                                                           o['bhattacharyya'])
                                        for o in res['overlap'] if o['proj'] == 'a1d1'))

    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print('\noutputs in', OUT)


if __name__ == '__main__':
    main()
