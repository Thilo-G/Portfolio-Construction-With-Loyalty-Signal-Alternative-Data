"""robustness_figures.py -- RRR Financial Implications
=====================================================

Two figures for the paper, drawn only from outputs other scripts saved:

  Paper_LaTeX/figures/robustness_alphas.pdf
      Every robustness check of the Appendix robustness table as a coefficient
      plot: the five-factor alpha of the long-short, Q1 and Q4 portfolios, each
      with its 95% Newey-West interval (alpha +/- 1.96 SE, author decision
      2026-10-20), and the exact p-value printed at the end of each row.
      Source: output/robustness_summary.xlsx (asymmetry_across_variants.py).

  Paper_LaTeX/figures/ls_monthly_hist.pdf
      Histogram of the monthly long-short returns of the headline portfolio,
      with the mean and the median marked.
      Source: sheet ls_monthly of output/signal_ic_diagnostics.xlsx
      (signal_ic_diagnostics.py).

Design: one hue per figure (reference blue), a darker step of the same ramp for
the headline row, text in ink and muted gray, no in-figure titles (the LaTeX
caption carries them). No significance stars; p-values are printed exactly.

Run (from Code/RRR-FI-IM):
    ..\\venv\\Scripts\\python.exe robustness_figures.py
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                 # noqa: E402
import matplotlib.transforms as mtransforms     # noqa: E402
from matplotlib.ticker import MaxNLocator      # noqa: E402
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

# =============================================================================
# Constants
# =============================================================================
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / 'output'
FIG_DIR = HERE.parent.parent / 'Paper_LaTeX' / 'figures'

Z_95 = 1.959964             # two-sided 95% interval
EXPECTED_ROWS = 16          # checks in the robustness table
EXPECTED_MONTHS = 107       # months of the headline long-short series

BLUE = '#2a78d6'            # reference categorical slot 1
BLUE_DARK = '#184f95'       # darker step of the same ramp: the headline row
BAND = '#eaf1fb'            # light step of the same ramp: headline row background
INK = '#2b2b2b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'

PANELS = [('LS', 'Long-short (Q1 minus Q4)'),
          ('Q1', 'Q1 (highest RRR)'),
          ('Q4', 'Q4 (lowest RRR)')]
GROUPS = [('A', 'A. Signal construction'), ('B', 'B. Estimation and sample')]


# =============================================================================
# Helpers
# =============================================================================
# Shorter row labels for the figure; the Appendix table keeps the full ones.
SHORT_LABELS = {
    'Two-quarter RRR (t-2 denominator)': 'Two-quarter RRR (t-2)',
    'Four-quarter RRR (t-4 denominator)': 'Four-quarter RRR (t-4)',
    'RCB4, rolling customer base (4 quarters)': 'RCB4, rolling customer base',
    'Four-quarter moving average (t-4 to t-1)': 'Moving average, t-4 to t-1',
    'Excluding COVID-19 (2020Q1-2021Q2)': 'Excluding COVID-19',
    'Two largest contributors removed from both legs': 'Two largest contributors removed',
}


def plain_label(latex_label):
    """'Two-quarter \\RRR{} ($t-2$ denominator)' -> 'Two-quarter RRR (t-2)'."""
    out = latex_label.replace('\\RRR{}', 'RRR').replace('$', '').replace('--', '-')
    out = out.replace('\\', '')
    return SHORT_LABELS.get(out, out)


def p_text(p):
    """Exact p-value, AMA style: [.034], [< .001]."""
    return '[< .001]' if p < 0.001 else '[' + f'{p:.3f}'.lstrip('0') + ']'


def style_axis(ax):
    ax.set_facecolor('white')
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7.5, length=2.5)
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# =============================================================================
# Figure 1: robustness coefficient plot
# =============================================================================
def robustness_plot():
    rows = pd.read_excel(OUTPUT_DIR / 'robustness_summary.xlsx')
    assert len(rows) == EXPECTED_ROWS, f'{len(rows)} robustness rows, expected {EXPECTED_ROWS}'
    headline_check = rows['Check'].iloc[0]

    # One y position per check, plus a heading line per group, top to bottom.
    items = []
    for code, heading in GROUPS:
        items.append(('heading', heading, None))
        for _, r in rows[rows['Panel'] == code].iterrows():
            items.append(('check', plain_label(r['Check']), r))
    n = len(items)
    ypos = {i: n - 1 - i for i in range(n)}

    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 0.235 * n + 0.95), sharey=True)
    fig.subplots_adjust(left=0.26, right=0.92, wspace=0.62, top=0.93, bottom=0.10)
    for ax, (leg, title) in zip(axes, PANELS):
        style_axis(ax)
        a, se = rows[f'{leg} alpha (%/mo)'], rows[f'{leg} SE']
        lo, hi = (a - Z_95 * se).min(), (a + Z_95 * se).max()
        span = max(hi, 0) - min(lo, 0)
        ax.set_xlim(min(lo, 0) - 0.06 * span, max(hi, 0) + 0.06 * span)
        ax.axvline(0, color=MUTED, linewidth=0.9, linestyle=(0, (3, 2)))
        to_right = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for i, (kind, _label, r) in enumerate(items):
            if kind != 'check':
                continue
            y = ypos[i]
            is_headline = r['Check'] == headline_check
            colour = BLUE_DARK if is_headline else BLUE
            if is_headline:
                ax.axhspan(y - 0.5, y + 0.5, color=BAND, zorder=0)
            alpha, half = r[f'{leg} alpha (%/mo)'], Z_95 * r[f'{leg} SE']
            ax.plot([alpha - half, alpha + half], [y, y], color=colour, linewidth=1.4,
                    solid_capstyle='round', zorder=2)
            ax.plot(alpha, y, 'o', markersize=4.6, color=colour, markeredgecolor='white',
                    markeredgewidth=0.8, zorder=3)
            ax.text(1.03, y, p_text(r[f'{leg} p']), transform=to_right, fontsize=6.8,
                    color=INK_2, va='center', ha='left')
        ax.tick_params(axis='y', length=0)
        ax.set_title(title, loc='left', fontsize=8.5, color=INK)
        ax.set_xlabel('Five-factor alpha (% per month)', fontsize=7.5, color=INK)

    axes[0].set_yticks([ypos[i] for i in range(n)])
    axes[0].set_yticklabels([label for _, label, _ in items])
    axes[0].tick_params(axis='y', length=0)
    for tick, (kind, _, _) in zip(axes[0].get_yticklabels(), items):
        if kind == 'heading':
            tick.set_fontweight('bold')
    axes[0].set_ylim(-0.7, n - 0.3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / 'robustness_alphas.pdf'
    fig.savefig(path)
    fig.savefig(OUTPUT_DIR / 'robustness_alphas.png', dpi=200)
    plt.close(fig)
    print(f'  wrote {path}')


# =============================================================================
# Figure 2: histogram of monthly long-short returns
# =============================================================================
def monthly_histogram():
    m = pd.read_excel(OUTPUT_DIR / 'signal_ic_diagnostics.xlsx', sheet_name='ls_monthly')
    x = m['Long-short return (%)']
    assert len(x) == EXPECTED_MONTHS, f'{len(x)} months, expected {EXPECTED_MONTHS}'
    mean, median = x.mean(), x.median()

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    style_axis(ax)
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color(MUTED)
    bins = np.arange(2 * np.floor(x.min() / 2), 2 * np.ceil(x.max() / 2) + 2, 2)
    ax.hist(x, bins=bins, color=BLUE, edgecolor='white', linewidth=1.0, zorder=2)
    ax.axvline(0, color=MUTED, linewidth=0.9, zorder=3)
    ax.axvline(mean, color=BLUE_DARK, linewidth=1.5, zorder=3)
    ax.axvline(median, color=BLUE_DARK, linewidth=1.5, linestyle=(0, (1, 1.5)), zorder=3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.text(0.98, 0.96, f'Mean {mean:.2f}% (solid line)\nMedian {median:.2f}% (dotted line)',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.5, color=INK)
    ax.set_xlabel('Monthly return of the long-short portfolio (%)', fontsize=7.5, color=INK)
    ax.set_ylabel('Number of months', fontsize=7.5, color=INK)
    fig.tight_layout()

    path = FIG_DIR / 'ls_monthly_hist.pdf'
    fig.savefig(path)
    fig.savefig(OUTPUT_DIR / 'ls_monthly_hist.png', dpi=200)
    plt.close(fig)
    print(f'  wrote {path}')


# =============================================================================
# Main
# =============================================================================
def main():
    robustness_plot()
    monthly_histogram()


if __name__ == '__main__':
    main()
