"""double_sort_size.py -- RRR Financial Implications
====================================================

RRR x Size double sort, requested by the author on 2026-10-12 and since
2026-10-20 an Appendix table of the paper (Paper_LaTeX/tables/tab_double_sort_size.tex,
written by write_size_table below). An RRR x revenue-growth double sort that this
script also computed was removed on 2026-11-11; it was never reported, because
the growth question is answered by the growth sort and the spanning regression.

Design: 3x3 CONDITIONAL (dependent) sorts, the project's bivariate convention.
  1. Each signal QUARTER, firms are split into size tertiles with
     av2.safe_tercile (T1 = largest): size = MCAP_FORM (formation-date market
     cap, also the portfolio weight).
  2. Within each (quarter, control tertile), firms are split into raw RRR_PCT
     tertiles (T1 = highest RRR).
  3. Nine cell portfolios, value-weighted on MCAP_FORM exactly as
     av2.build_portfolio_returns weights (simple returns), and an
     equal-weighted version on the same firms as a concentration check.
  4. The RRR long-short (T1-T3) inside each control tertile, and the
     control-neutral long-short = the average of the three.
  5. FF5 alphas with Newey-West errors (signal_ic_diagnostics.ff5_alpha, the
     same alignment as av2.run_factor_regressions); long-shorts are zero-cost
     and take no RF subtraction, cells are regressed in excess of RF.

It is conditional rather than independent because RRR is correlated with size
(Q1 averages $56.7B market cap against $5.2B for Q4), which leaves independent
high-RRR/small-size cells thin; the independent cell counts are written out to
show this. The unconditional RRR tercile spread on the same sample is reported
next to the neutral spread so the two compare like for like.

Added 2026-10-19: a probe of the small-size third (sheets size_small_*), because
the RRR long-short reverses there. It reports both legs, the firms that carry
them, and the alpha without the largest contributors or the worst months.

No significance stars; exact p-values.

Outputs
    output/double_sort_size.xlsx   (sheet size_long_short is read by numbers_tex.py)
    Paper_LaTeX/tables/tab_double_sort_size.tex

Run (from Code/RRR-FI-IM):
    ..\\venv\\Scripts\\python.exe double_sort_size.py
"""

# =============================================================================
# Cell 1: imports
# =============================================================================
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import analysis_v2 as av2                 # noqa: E402
import robustness_diagnostics as rd       # noqa: E402
from signal_ic_diagnostics import (       # noqa: E402
    ff5_alpha, guard_headline, quiet, to_month_end, FF5, EXPECTED_T_MONTHS)

# =============================================================================
# Cell 2: constants
# =============================================================================
OUTPUT_DIR = av2.OUTPUT_DIR
XLSX_PATH = os.path.join(OUTPUT_DIR, 'double_sort_size.xlsx')

# key, label, column, label of control tertile T1 / T3
CONTROLS = [
    ('size', 'Size', 'MCAP_FORM', 'large', 'small'),
]
TERTILES = ['T1', 'T2', 'T3']
MIN_CELL_MONTHS = 100   # every cell must have returns in at least this many factor months


# =============================================================================
# Cell 3: sorting and portfolio construction
# =============================================================================
def assign_tertiles(panel, ctrl_col):
    """Control tertile, conditional RRR tertile within it, and the independent
    (unconditional) RRR tertile, all within each signal quarter."""
    p = panel.copy()
    for c in [ctrl_col, 'RRR_PCT', 'MCAP_FORM', 'RET_SIMPLE']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    p = p.dropna(subset=[ctrl_col, 'RRR_PCT']).copy()
    p['C_T'] = p.groupby('QUARTER')[ctrl_col].transform(av2.safe_tercile).astype(object)
    p['R_T'] = (p.dropna(subset=['C_T']).groupby(['QUARTER', 'C_T'])['RRR_PCT']
                .transform(av2.safe_tercile).astype(object))
    p['R_T_IND'] = p.groupby('QUARTER')['RRR_PCT'].transform(av2.safe_tercile).astype(object)
    return p


def cell_returns(p, rrr_col, weighting):
    """Monthly cell returns, value- or equal-weighted, on firms with a formation
    market cap so both weightings hold the same firms. Columns '<C_T>_<RRR T>'."""
    df = p.dropna(subset=['C_T', rrr_col, 'MCAP_FORM', 'RET_SIMPLE']).copy()
    df['CELL'] = df['C_T'] + '_' + df[rrr_col]
    grp = df.groupby(['Date', 'CELL'])
    if weighting == 'VW':
        df['w'] = df['MCAP_FORM'] / grp['MCAP_FORM'].transform('sum')
    else:
        df['w'] = 1.0 / grp['MCAP_FORM'].transform('size')
    df['wr'] = df['w'] * df['RET_SIMPLE']
    port = df.groupby(['Date', 'CELL'])['wr'].sum().unstack('CELL').sort_index()
    port.index = to_month_end(port.index)
    return port


def single_sort_returns(p, weighting):
    """Unconditional RRR tercile portfolios on the same sample (T1-T3 spread)."""
    df = p.dropna(subset=['R_T_IND', 'MCAP_FORM', 'RET_SIMPLE']).copy()
    grp = df.groupby(['Date', 'R_T_IND'])
    if weighting == 'VW':
        df['w'] = df['MCAP_FORM'] / grp['MCAP_FORM'].transform('sum')
    else:
        df['w'] = 1.0 / grp['MCAP_FORM'].transform('size')
    df['wr'] = df['w'] * df['RET_SIMPLE']
    port = df.groupby(['Date', 'R_T_IND'])['wr'].sum().unstack('R_T_IND').sort_index()
    port.index = to_month_end(port.index)
    return port['T1'] - port['T3']


def _alpha_row(series, ff, excess):
    r = ff5_alpha(series, ff, excess=excess)
    s = pd.Series(series).copy()
    s.index = to_month_end(s.index)
    comb = s.to_frame('y').join(ff, how='inner').dropna(subset=FF5 + ['y'])
    mean = (comb['y'] - comb['RF']).mean() if excess else comb['y'].mean()
    return {'Mean return (%/mo)': mean * 100, 'FF5 alpha (%/mo)': r['alpha'] * 100,
            'SE': r['se'] * 100, 'p': r['p'], 'N months': r['n']}


# =============================================================================
# Cell 4: one control variable
# =============================================================================
def run_control(panel, ff, key, label, ctrl_col, hi_word, lo_word):
    print('\n' + '=' * 78)
    print(f'  RRR x {label}: 3x3 conditional sort ({ctrl_col})')
    print('=' * 78)
    ctrl_name = {'T1': f'{label} T1 ({hi_word})', 'T2': f'{label} T2', 'T3': f'{label} T3 ({lo_word})'}
    rrr_name = {'T1': 'RRR T1 (high)', 'T2': 'RRR T2', 'T3': 'RRR T3 (low)'}

    p = assign_tertiles(panel, ctrl_col)
    cells, longs = [], []
    grids_alpha, grids_p = [], []
    for weighting in ['VW', 'EW']:
        port = cell_returns(p, 'R_T', weighting)
        expected = {f'{c}_{r}' for c in TERTILES for r in TERTILES}
        assert set(port.columns) == expected, f'{key} {weighting}: cells {sorted(port.columns)}'

        ga, gp = {}, {}
        for c in TERTILES:
            for r in TERTILES:
                row = _alpha_row(port[f'{c}_{r}'], ff, excess=True)
                assert row['N months'] >= MIN_CELL_MONTHS, (
                    f'{key} {weighting} cell {c}_{r} has only {row["N months"]} months')
                cells.append({'Weighting': weighting, 'Control tertile': ctrl_name[c],
                              'RRR tertile': rrr_name[r], **row})
                ga[(ctrl_name[c], rrr_name[r])] = row['FF5 alpha (%/mo)']
                gp[(ctrl_name[c], rrr_name[r])] = row['p']

        ls = {c: port[f'{c}_T1'] - port[f'{c}_T3'] for c in TERTILES}
        neutral = pd.concat(ls.values(), axis=1).mean(axis=1, skipna=False).dropna()
        for c in TERTILES:
            row = _alpha_row(ls[c], ff, excess=False)
            longs.append({'Weighting': weighting, 'Portfolio': f'RRR T1-T3 within {ctrl_name[c]}', **row})
            ga[(ctrl_name[c], 'RRR T1-T3')] = row['FF5 alpha (%/mo)']
            gp[(ctrl_name[c], 'RRR T1-T3')] = row['p']
        row = _alpha_row(neutral, ff, excess=False)
        longs.append({'Weighting': weighting, 'Portfolio': f'{label}-neutral RRR T1-T3 (average)', **row})
        ga[(f'{label}-neutral average', 'RRR T1-T3')] = row['FF5 alpha (%/mo)']
        gp[(f'{label}-neutral average', 'RRR T1-T3')] = row['p']
        row = _alpha_row(single_sort_returns(p, weighting), ff, excess=False)
        longs.append({'Weighting': weighting,
                      'Portfolio': 'Unconditional RRR tercile T1-T3 (same sample)', **row})
        ga[('Unconditional RRR tercile spread', 'RRR T1-T3')] = row['FF5 alpha (%/mo)']
        gp[('Unconditional RRR tercile spread', 'RRR T1-T3')] = row['p']

        cols = [rrr_name[r] for r in TERTILES] + ['RRR T1-T3']
        rows_order = [ctrl_name[c] for c in TERTILES] + [f'{label}-neutral average',
                                                          'Unconditional RRR tercile spread']
        to_grid = lambda d: (pd.Series(d).unstack().reindex(index=rows_order, columns=cols))  # noqa: E731
        a_grid, p_grid = to_grid(ga), to_grid(gp)
        a_grid.insert(0, 'Weighting', weighting)
        p_grid.insert(0, 'Weighting', weighting)
        grids_alpha.append(a_grid)
        grids_p.append(p_grid)

        print(f'\n  {weighting}: FF5 alpha %/mo [p]')
        fmt = a_grid.drop(columns='Weighting').copy().astype(object)
        for i in fmt.index:
            for j in fmt.columns:
                a, pv = a_grid.loc[i, j], p_grid.loc[i, j]
                fmt.loc[i, j] = '' if pd.isna(a) else f'{a:.4f} [{pv:.3f}]'
        with pd.option_context('display.width', 250, 'display.max_columns', 10):
            print(fmt.to_string())

    # Cell diagnostics on firm-quarters (value-weighted sample).
    fq = (p.dropna(subset=['C_T', 'R_T', 'MCAP_FORM'])
          .drop_duplicates(['FIRM', 'QUARTER']).copy())
    fq['CELL'] = fq['C_T'] + '_' + fq['R_T']
    diag = []
    for (c, r), g in fq.groupby(['C_T', 'R_T']):
        per_q = g.groupby('QUARTER')
        eff = per_q['MCAP_FORM'].apply(lambda w: rd.hhi_effective_n(w.values)[1])
        diag.append({'Control tertile': ctrl_name[c], 'RRR tertile': rrr_name[r],
                     'Avg firms per quarter': per_q.size().mean(),
                     'Min firms in a quarter': int(per_q.size().min()),
                     'Avg market cap ($M)': per_q['MCAP_FORM'].mean().mean(),
                     'Avg RRR (%)': per_q['RRR_PCT'].mean().mean(),
                     f'Avg {label.lower()} control': per_q[ctrl_col].mean().mean(),
                     'Avg effective N (VW)': eff.mean(),
                     'Min effective N (VW)': eff.min()})
    diag = pd.DataFrame(diag)
    print('\n  Cell diagnostics (firm-quarters):')
    print(diag.round(2).to_string(index=False))

    # Independent 3x3: average firms per quarter in each cell.
    fqi = p.dropna(subset=['C_T', 'R_T_IND', 'MCAP_FORM']).drop_duplicates(['FIRM', 'QUARTER'])
    indep = (fqi.groupby(['QUARTER', 'C_T', 'R_T_IND']).size()
             .groupby(['C_T', 'R_T_IND']).mean().unstack('R_T_IND')
             .reindex(index=TERTILES, columns=TERTILES))
    indep.index = [ctrl_name[c] for c in indep.index]
    indep.columns = [rrr_name[r] for r in indep.columns]
    print('\n  Independent 3x3 sort, average firms per quarter (why the conditional design is used):')
    print(indep.round(1).to_string())

    return {f'{key}_grid_alpha': pd.concat(grids_alpha).reset_index().rename(columns={'index': 'Row'}),
            f'{key}_grid_p': pd.concat(grids_p).reset_index().rename(columns={'index': 'Row'}),
            f'{key}_cells': pd.DataFrame(cells),
            f'{key}_long_short': pd.DataFrame(longs),
            f'{key}_cell_diag': diag,
            f'{key}_indep_counts': indep.reset_index().rename(columns={'index': 'Control tertile'})}


# =============================================================================
# Cell 5: what drives the small-firm reversal (added 2026-10-19)
# =============================================================================
PROBE_TOP_K = [1, 2, 3]          # largest contributors removed from the sample
PROBE_WORST_MONTHS = [3, 5]      # worst months of the small-firm long-short dropped


def leg_contributions(p, ctrl_t, rrr_t):
    """Each firm's summed monthly contribution (weight x return, in percentage
    points) to one value-weighted cell, with its months in the cell."""
    df = p.dropna(subset=['C_T', 'R_T', 'MCAP_FORM', 'RET_SIMPLE']).copy()
    df = df[(df['C_T'] == ctrl_t) & (df['R_T'] == rrr_t)].copy()
    df['w'] = df['MCAP_FORM'] / df.groupby('Date')['MCAP_FORM'].transform('sum')
    df['contrib'] = df['w'] * df['RET_SIMPLE'] * 100
    out = df.groupby('FIRM').agg(**{'Months in cell': ('contrib', 'size'),
                                    'Summed contribution (pp)': ('contrib', 'sum'),
                                    'Avg RRR (%)': ('RRR_PCT', 'mean'),
                                    'Avg market cap ($M)': ('MCAP_FORM', 'mean')})
    return out, df['contrib'].sum()


def small_tertile_probe(panel, ff):
    """The RRR long-short inside the small-size third: its two legs, the firms
    that carry them, and how the alpha changes without the largest contributors
    or the worst months. The contributors are found from the data, not named."""
    print('\n' + '=' * 78)
    print('  Small-size third: what drives the RRR long-short')
    print('=' * 78)
    p = assign_tertiles(panel, 'MCAP_FORM')
    port = cell_returns(p, 'R_T', 'VW')
    ls = port['T3_T1'] - port['T3_T3']
    ls = ls.reindex(ls.index.intersection(ff.dropna(subset=FF5).index)).dropna()

    rows = []
    hi, lo = ff5_alpha(port['T3_T1'], ff, excess=True), ff5_alpha(port['T3_T3'], ff, excess=True)
    for name, r in [('Small high-RRR cell (excess)', hi), ('Small low-RRR cell (excess)', lo)]:
        rows.append({'Item': f'{name}: FF5 alpha', 'Value': r['alpha'] * 100, 'SE': r['se'] * 100, 'p': r['p']})
        rows.append({'Item': f'{name}: SMB loading', 'Value': r['beta_SMB'], 'SE': r['se_SMB'], 'p': r['p_SMB']})
    legs = port[['T3_T1', 'T3_T3']].reindex(ls.index)
    rows.append({'Item': 'Correlation of the two small-firm cells', 'Value': legs.corr().iloc[0, 1]})
    rows.append({'Item': 'Share of months with small-firm long-short > 0', 'Value': float((ls > 0).mean())})
    base = ff5_alpha(ls, ff)
    rows.append({'Item': 'Small-firm RRR long-short: FF5 alpha, full sample', 'Value': base['alpha'] * 100,
                 'SE': base['se'] * 100, 'p': base['p'], 'N months': base['n']})

    contrib_lo, total_lo = leg_contributions(p, 'T3', 'T3')
    contrib_hi, total_hi = leg_contributions(p, 'T3', 'T1')
    top_lo = contrib_lo['Summed contribution (pp)'].nlargest(max(PROBE_TOP_K)).index.tolist()
    for k in PROBE_TOP_K:
        removed = top_lo[:k]
        pk = assign_tertiles(panel[~panel['FIRM'].isin(removed)], 'MCAP_FORM')
        portk = cell_returns(pk, 'R_T', 'VW')
        r = ff5_alpha(portk['T3_T1'] - portk['T3_T3'], ff)
        rows.append({'Item': f'Without the top {k} contributor(s) to the small low-RRR cell: '
                             + ', '.join(removed),
                     'Value': r['alpha'] * 100, 'SE': r['se'] * 100, 'p': r['p'], 'N months': r['n']})
    for k in PROBE_WORST_MONTHS:
        worst = ls.nsmallest(k)
        r = ff5_alpha(ls.drop(worst.index), ff)
        rows.append({'Item': f'Without the {k} worst months: '
                             + ', '.join(worst.index.strftime('%Y-%m')),
                     'Value': r['alpha'] * 100, 'SE': r['se'] * 100, 'p': r['p'], 'N months': r['n']})
    probe = pd.DataFrame(rows)

    contrib = pd.concat([
        contrib_lo.sort_values('Summed contribution (pp)', ascending=False).head(10)
                  .assign(Cell='Small low-RRR', **{'Cell total (pp)': total_lo}),
        contrib_hi.sort_values('Summed contribution (pp)', ascending=True).head(10)
                  .assign(Cell='Small high-RRR', **{'Cell total (pp)': total_hi}),
    ]).reset_index()
    by_year = (ls.groupby(ls.index.year).agg(['size', 'mean']) * [1, 100]).reset_index()
    by_year.columns = ['Year', 'Months', 'Mean small-firm long-short (%/mo)']

    with pd.option_context('display.width', 250, 'display.max_colwidth', 120):
        print(probe.round(4).to_string(index=False))
        print('\n  Largest contributors (summed monthly contribution, pp):')
        print(contrib[['Cell', 'FIRM', 'Months in cell', 'Summed contribution (pp)', 'Cell total (pp)',
                       'Avg RRR (%)']].groupby('Cell').head(5).round(1).to_string(index=False))
    return {'size_small_probe': probe, 'size_small_contrib': contrib, 'size_small_by_year': by_year}


# =============================================================================
# Cell 6: Appendix table (added 2026-10-20, author decision: size double sort in
# the Appendix only)
# =============================================================================
TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(_HERE)), 'Paper_LaTeX', 'tables',
                          'tab_double_sort_size.tex')


def _ama(v):
    """AMA number format: three decimals, no zero before the decimal point."""
    s = f'{abs(v):.3f}'
    s = s[1:] if s.startswith('0.') else s
    return ('$-$' + s) if v < 0 else s


def _ama_p(p):
    return '[$<$ .001]' if p < 0.001 else '[' + f'{p:.3f}'.lstrip('0') + ']'


def write_size_table(long_short, cell_diag):
    """RRR long-short (top minus bottom RRR tercile) within each size tercile,
    the size-neutral average and the unconditional tercile spread, value-weighted."""
    vw = long_short[long_short['Weighting'] == 'VW'].set_index('Portfolio')
    cols = [('Large', 'RRR T1-T3 within Size T1 (large)'),
            ('Middle', 'RRR T1-T3 within Size T2'),
            ('Small', 'RRR T1-T3 within Size T3 (small)'),
            ('Size-neutral', 'Size-neutral RRR T1-T3 (average)'),
            ('All firms', 'Unconditional RRR tercile T1-T3 (same sample)')]
    firms = cell_diag.groupby('Control tertile')['Avg firms per quarter'].sum()
    firms_by_col = {'Large': firms.get('Size T1 (large)'), 'Middle': firms.get('Size T2'),
                    'Small': firms.get('Size T3 (small)')}
    a = [vw.loc[key, 'FF5 alpha (%/mo)'] for _, key in cols]
    se = [vw.loc[key, 'SE'] for _, key in cols]
    p = [vw.loc[key, 'p'] for _, key in cols]
    n = [int(vw.loc[key, 'N months']) for _, key in cols]
    L = ['\\begin{threeparttable}', '\\footnotesize',
         '\\begin{tabular}{lccccc}', '\\toprule',
         ' & \\multicolumn{3}{c}{Within size tercile} & & \\\\',
         '\\cmidrule(lr){2-4}',
         ' & ' + ' & '.join(lbl for lbl, _ in cols) + ' \\\\', '\\midrule',
         'FF5 $\\alpha$ (\\%/mo) & ' + ' & '.join(_ama(v) for v in a) + ' \\\\',
         ' & ' + ' & '.join(f'({_ama(v)})' for v in se) + ' \\\\',
         ' & ' + ' & '.join(_ama_p(v) for v in p) + ' \\\\', '\\midrule',
         'Firms per quarter & ' + ' & '.join(
             f'{firms_by_col[lbl]:.1f}' if lbl in firms_by_col else '' for lbl, _ in cols) + ' \\\\',
         '$N$ (months) & ' + ' & '.join(str(v) for v in n) + ' \\\\',
         '\\bottomrule', '\\end{tabular}', '\\begin{tablenotes}', '\\small',
         '\\item \\textit{Notes:} Each quarter, firms are split into terciles of formation-date '
         'market capitalization and, within each size tercile, into terciles of \\RRR{}; each '
         'column holds the value-weighted portfolio long the top and short the bottom \\RRR{} '
         'tercile, the size-neutral column averages the three, and the last column sorts on '
         '\\RRR{} without conditioning on size. Five-factor alphas in percent per month, with '
         'Newey-West standard errors in parentheses and exact $p$-values in brackets.',
         '\\end{tablenotes}', '\\end{threeparttable}']
    with open(TABLE_PATH, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'  wrote {TABLE_PATH}')


# =============================================================================
# Cell 7: main
# =============================================================================
def main():
    _df_filtered, _returns, ff, panel, market_ret = rd.load_base_data()
    with quiet():
        port, reg = rd.baseline_portfolio_and_regressions(panel, ff, market_ret)
    guard_headline(port, reg, ff)

    sheets = {}
    for key, label, col, hi, lo in CONTROLS:
        sheets.update(run_control(panel, ff, key, label, col, hi, lo))
    sheets.update(small_tertile_probe(panel, ff))

    for name, df in sheets.items():
        if name.endswith('_long_short'):
            assert len(df) == 2 * 5, f'{name}: {len(df)} long-short rows'
            neutral = df[df['Portfolio'].str.contains('neutral')]
            assert (neutral['N months'] <= EXPECTED_T_MONTHS).all()
        if name.endswith('_cells'):
            assert len(df) == 2 * 9, f'{name}: {len(df)} cells'

    with pd.ExcelWriter(XLSX_PATH, engine='openpyxl') as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print(f'\n  wrote {XLSX_PATH}')
    write_size_table(sheets['size_long_short'], sheets['size_cell_diag'])


if __name__ == '__main__':
    main()
