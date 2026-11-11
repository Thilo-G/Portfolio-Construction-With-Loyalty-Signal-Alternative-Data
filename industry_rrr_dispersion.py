"""industry_rrr_dispersion.py -- RRR level and dispersion by sector over time.

Requested in research_revision_notes.md line 179: a graph of industry \\RRR{}
mean and median over time with dispersion bands at the 10th and 90th
percentiles.

The figure serves the Data section. It shows two things the summary table
cannot: that the central tendency of RRR moves with the cycle (visibly at the
2020 demand shock), and that the cross-firm spread within a sector is wide
enough at every date for a within-sector sort to have something to sort on.
That second point matters because the paper's signal is industry-time
adjusted, so it is the WITHIN-sector spread, not the level, that the sort
exploits.

Output
    Paper_LaTeX/figures/industry_rrr_dispersion.pdf
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
FIGURE_DIR = os.path.join(_PROJECT, 'Paper_LaTeX', 'figures')

COLORS = sns.color_palette('colorblind', 10)

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
})


def build(df_filtered):
    d = df_filtered.reset_index()[['DATE', 'SECTOR', 'RRR_PCT']].copy()
    d['DATE'] = pd.to_datetime(d['DATE'])
    d['RRR_PCT'] = pd.to_numeric(d['RRR_PCT'], errors='coerce')
    d = d.dropna(subset=['RRR_PCT', 'SECTOR'])

    g = d.groupby(['SECTOR', 'DATE'])['RRR_PCT']
    stats = g.agg(mean='mean', median='median',
                  p10=lambda s: s.quantile(0.10),
                  p90=lambda s: s.quantile(0.90),
                  n='size').reset_index()
    # A percentile band on three firms is noise, not dispersion.
    stats = stats[stats['n'] >= 5]
    return stats


def make_figure(stats):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    sectors = sorted(stats['SECTOR'].unique())
    n = len(sectors)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.1 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, sec in zip(axes, sectors):
        s = stats[stats['SECTOR'] == sec].sort_values('DATE')
        ax.fill_between(s['DATE'], s['p10'], s['p90'],
                        color=COLORS[0], alpha=0.18,
                        label='10th to 90th percentile')
        ax.plot(s['DATE'], s['median'], color=COLORS[0], linewidth=2.0,
                label='Median')
        ax.plot(s['DATE'], s['mean'], color=COLORS[3], linewidth=1.4,
                linestyle='--', label='Mean')
        ax.axhline(100, color='gray', linewidth=0.7, linestyle=':')

        # RRR has no upper bound, so a single firm rebuilding from a near-zero
        # base can push a sector's 90th percentile into the thousands and
        # flatten every other series in the panel. Clip the axis to a robust
        # range and say so, rather than dropping the observation.
        hi = float(np.nanpercentile(s['p90'], 90)) * 1.15
        lo = float(np.nanmin(s['p10'])) - 5.0
        clipped = int((s['p90'] > hi).sum())
        ax.set_ylim(lo, hi)
        title = sec if not clipped else f'{sec} (axis clipped, {clipped} quarter(s) above)'
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    for ax in axes[n:]:
        ax.set_visible(False)

    for ax in axes[max(0, n - ncols):n]:
        ax.set_xlabel('Quarter')
    for i in range(0, n, ncols):
        axes[i].set_ylabel('Revenue Retention Rate')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    path = os.path.join(FIGURE_DIR, 'industry_rrr_dispersion.pdf')
    fig.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f'  wrote {path}')
    return path


def main():
    print('=' * 78)
    print('  INDUSTRY RRR LEVEL AND DISPERSION OVER TIME')
    print('=' * 78)

    ff = av2.phase1b_load_ff_factors()
    _df_long, df_filtered, _ = av2.phase1_load_and_diagnose(ff)

    stats = build(df_filtered)
    print(f'\n  Sectors: {sorted(stats["SECTOR"].unique())}')
    print(f'  Quarters per sector (>=5 firms): '
          f'{stats.groupby("SECTOR")["DATE"].nunique().to_dict()}')
    print('\n  Pooled within-sector spread (p90 minus p10), percentage points:')
    spread = (stats['p90'] - stats['p10']).describe()
    print(f'    median {spread["50%"]:.1f}, mean {spread["mean"]:.1f}, '
          f'min {spread["min"]:.1f}, max {spread["max"]:.1f}')

    make_figure(stats)
    print('\n  DONE')


if __name__ == '__main__':
    main()
