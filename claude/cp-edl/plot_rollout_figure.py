"""
Figure for the long-horizon rollout experiment: shows that predicted uncertainty and field
roughness collapse together while error grows -- the mechanism behind the sigma<->error
anti-correlation. Reads exp_rollout/rollout_drift_results.json (needs the roughness diagnostic).

Panel (a): the three quantities indexed to their first rollout step (one axis, no dual scale),
           averaged over the six schemes with a +/-1 std band.
Panel (b): predicted uncertainty vs field roughness, pooled over schemes and steps, showing the
           near-deterministic positive coupling.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (validated), + linestyle as secondary encoding
C_ERR, C_UNC, C_ROUGH = '#D55E00', '#0072B2', '#009E73'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.linewidth': 0.6,
    'axes.edgecolor': '#444444', 'xtick.color': '#444444', 'ytick.color': '#444444',
    'axes.labelcolor': '#222222', 'text.color': '#222222',
})


def load(path):
    rows = json.load(open(path))
    err = np.array([r['err_curve'] for r in rows])      # (6, T-1)
    unc = np.array([r['unc_curve'] for r in rows])
    rough = np.array([r['rough_curve'] for r in rows])
    return rows, err, unc, rough


def idx(a):
    return a / a[:, :1]                                 # index each scheme's curve to step 1


def main(out_dir):
    rows, err, unc, rough = load('exp_rollout/rollout_drift_results.json')
    steps = np.arange(1, err.shape[1] + 1)
    ei, ui, ri = idx(err), idx(unc), idx(rough)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # --- Panel (a): indexed curves, mean +/- std over schemes ---
    # y-offsets (in points) keep the two coincident bottom labels from colliding
    for data, c, ls, lab, dy in [(ei, C_ERR, '-', 'rel. error', 0),
                                  (ui, C_UNC, '--', 'uncertainty', 9),
                                  (ri, C_ROUGH, ':', 'roughness', -9)]:
        m, s = data.mean(0), data.std(0)
        ax.plot(steps, m, color=c, linestyle=ls, linewidth=2.0, label=lab)
        ax.fill_between(steps, m - s, m + s, color=c, alpha=0.15, linewidth=0)
        ax.annotate(lab, (steps[-1], m[-1]), color=c, fontsize=8.5,
                    xytext=(5, dy), textcoords='offset points', va='center')
    ax.axhline(1.0, color='#999999', linewidth=0.6, linestyle='-', zorder=0)
    ax.set_xlabel('rollout step'); ax.set_ylabel('value (indexed to step 1)')
    ax.set_title('(a) Error rises as uncertainty and roughness fall', fontsize=9, loc='left')
    ax.set_xlim(1, steps[-1] + 2.2); ax.grid(True, color='#e8e8e8', linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    # --- Panel (b): uncertainty vs roughness, pooled, indexed ---
    xr, yu = ri.ravel(), ui.ravel()
    ax2.scatter(xr, yu, s=14, color=C_UNC, alpha=0.55, edgecolor='white', linewidth=0.3)
    r = np.corrcoef(xr, yu)[0, 1]
    lo, hi = xr.min(), xr.max()
    a, b = np.polyfit(xr, yu, 1)
    ax2.plot([lo, hi], [a * lo + b, a * hi + b], color='#444444', linewidth=1.2, linestyle='-')
    ax2.annotate(f'$r = {r:+.2f}$', (0.06, 0.9), xycoords='axes fraction', fontsize=10)
    ax2.set_xlabel('field roughness (indexed)'); ax2.set_ylabel('predicted uncertainty (indexed)')
    ax2.set_title('(b) Uncertainty tracks input roughness', fontsize=9, loc='left')
    ax2.grid(True, color='#e8e8e8', linewidth=0.5); ax2.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax2.spines[sp].set_visible(False)

    fig.tight_layout(w_pad=2.0)
    out = Path(out_dir)
    fig.savefig(out / 'rollout_drift.pdf', bbox_inches='tight')
    fig.savefig(out / 'rollout_drift.png', dpi=300, bbox_inches='tight')
    print('mean corr(sigma,roughness) =', round(float(np.mean([r for r in
          [np.corrcoef(ri[i], ui[i])[0, 1] for i in range(len(rows))]])), 3))
    print('rough_growth per scheme:', [round(float(ri[i, -1]), 3) for i in range(len(rows))])
    print(f'wrote {out}/rollout_drift.pdf and .png')


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else '../tex/img')
