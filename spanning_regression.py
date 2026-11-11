"""spanning_regression.py -- is the RRR long-short spanned by the growth
long-short?

Motivation stated plainly, confirmed with the author 2026-09-16: returning-
customer revenue is a large part of total revenue, and therefore of revenue
growth, so RRR and revenue growth are substantively related by construction.
A natural further check is whether either return series is explained by the
other once standard risk factors are held fixed.

Both return series already exist with full 108-month overlap:
  RRR long-short:    output/ta4_signal_analysis.xlsx, sheet Baseline_Returns
  Growth long-short:  output/signal_sort_robustness.xlsx, sheet revgrowth_raw_ret

SIGNAL BASIS: the RAW baseline long-short, matching the headline specification
(author decision, 2026-09-16). Previously this test ran on the industry-time
adjusted series. Both were estimated and the answer is the same either way:
RRR alpha 1.8245%/mo [p=.000] raw against 1.8343%/mo [p=.000] adjusted, growth
alpha -0.0976%/mo [p=.851] raw against 0.0531%/mo [p=.921] adjusted. Raw is
used so the test applies to the signal the paper actually leads with, not a
different one.

Two directions, both estimated with av2.newey_west_ols(), the SAME Newey-West
implementation (bandwidth max(1, floor(4*(T/100)**(2/9)))) every other
regression in this paper uses -- not a second, independent HAC implementation,
so these standard errors are on identical footing with every other table:
  (1) r_RRR    = alpha + beta * r_growth + FF factors + e
  (2) r_growth = alpha + beta * r_RRR    + FF factors + e

Outputs
    Paper_LaTeX/tables/tab_spanning_regression.tex
    output/spanning_regression.xlsx   one sheet per regression (var, coef, se, t, p),
                                      read by numbers_tex.py for the text. Restored
                                      2026-11-11: the file had not been rewritten
                                      since 2026-09-16, so the text cited stale
                                      numbers while the table was current.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
OUTPUT_DIR = av2.OUTPUT_DIR
TABLE_DIR = os.path.join(_PROJECT, 'Paper_LaTeX', 'tables')


def align_monthly(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp('M')
    return s


def spanning_reg(y, x_span, ff, factor_cols, span_label):
    """Regress y on [x_span] + factor_cols via av2.newey_west_ols (the paper's
    single canonical HAC implementation, not a second one)."""
    y = align_monthly(y)
    x_span = align_monthly(x_span).rename(span_label)
    ff2 = ff.copy()
    ff2.index = pd.to_datetime(ff2.index).to_period('M').to_timestamp('M')

    combined = pd.concat([y.rename('y'), x_span, ff2[factor_cols]], axis=1).dropna()
    n = len(combined)
    X = combined[[span_label] + factor_cols]   # newey_west_ols adds the constant itself
    fit = av2.newey_west_ols(combined['y'], X)

    names = ['const', span_label] + factor_cols
    out = {}
    for name in names:
        out[name] = {'coef': fit.params[name], 'se': fit.bse[name],
                     't': fit.tvalues[name], 'p': fit.pvalues[name]}
    out['_n'] = n
    out['_r2'] = fit.rsquared
    return out


def _fmt(v, dp=3, pct=False):
    if v is None or (isinstance(v, float) and v != v):
        return '--'
    x = v * 100 if pct else v
    # AMA rules (JMR/JoM): at most three decimals, no zero before the decimal point.
    out = format(float(x), f'.{min(dp, 3)}f')
    neg = out.startswith('-')
    if neg:
        out = out[1:]
    if out.startswith('0.'):
        out = out[1:]
    return ('$-$' + out) if neg else out


def _pv(p):
    if p is None or (isinstance(p, float) and p != p):
        return ''
    return '[$<$ .001]' if p < 0.001 else '[' + f'{p:.3f}'.lstrip('0') + ']'


def make_table(r1_ff5, r2_ff5, r1_ff3, r2_ff3):
    os.makedirs(TABLE_DIR, exist_ok=True)
    L = []
    L.append('\\begin{threeparttable}')
    L.append('\\footnotesize')
    L.append('\\begin{tabular}{lrrrr}')
    L.append('\\toprule')
    L.append(' & \\multicolumn{2}{c}{FF3} & \\multicolumn{2}{c}{FF5} \\\\')
    L.append('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}')
    L.append(' & $\\alpha$ (\\%/mo) & $\\beta$ on span & $\\alpha$ (\\%/mo) & $\\beta$ on span \\\\')
    L.append('\\midrule')
    L.append('\\multicolumn{5}{l}{\\textit{Panel A: RRR long-short, spanned by growth long-short}} \\\\')
    a3, a5 = r1_ff3, r1_ff5
    L.append(f"$r^{{\\text{{RRR}}}}_t$ & {_fmt(a3['const']['coef'],4,True)} & "
             f"{_fmt(a3['growth_LS']['coef'])} & {_fmt(a5['const']['coef'],4,True)} & "
             f"{_fmt(a5['growth_LS']['coef'])} \\\\")
    L.append(f" & ({_fmt(a3['const']['se'],4,True)}) {_pv(a3['const']['p'])} & "
             f"({_fmt(a3['growth_LS']['se'])}) {_pv(a3['growth_LS']['p'])} & "
             f"({_fmt(a5['const']['se'],4,True)}) {_pv(a5['const']['p'])} & "
             f"({_fmt(a5['growth_LS']['se'])}) {_pv(a5['growth_LS']['p'])} \\\\")
    L.append('\\midrule')
    L.append('\\multicolumn{5}{l}{\\textit{Panel B: growth long-short, spanned by RRR long-short}} \\\\')
    b3, b5 = r2_ff3, r2_ff5
    L.append(f"$r^{{\\text{{growth}}}}_t$ & {_fmt(b3['const']['coef'],4,True)} & "
             f"{_fmt(b3['RRR_LS']['coef'])} & {_fmt(b5['const']['coef'],4,True)} & "
             f"{_fmt(b5['RRR_LS']['coef'])} \\\\")
    L.append(f" & ({_fmt(b3['const']['se'],4,True)}) {_pv(b3['const']['p'])} & "
             f"({_fmt(b3['RRR_LS']['se'])}) {_pv(b3['RRR_LS']['p'])} & "
             f"({_fmt(b5['const']['se'],4,True)}) {_pv(b5['const']['p'])} & "
             f"({_fmt(b5['RRR_LS']['se'])}) {_pv(b5['RRR_LS']['p'])} \\\\")
    L.append('\\midrule')
    L.append(f"$N$ (months) & \\multicolumn{{4}}{{c}}{{{r1_ff5['_n']}}} \\\\")
    L.append('\\bottomrule')
    L.append('\\end{tabular}')
    L.append('\\begin{tablenotes}')
    L.append('\\small')
    # Note as the author shortened it by hand on 2026-11-10.
    L.append('\\item \\textit{Notes:} Panel A regresses the \\RRR{} long-short return '
             'on the revenue-growth long-short return plus FF3 or FF5 factors; '
             'Panel B reverses the two series. $\\beta$ on span is the loading on '
             "the other signal's long-short return.")
    L.append('\\end{tablenotes}')
    L.append('\\end{threeparttable}')

    path = os.path.join(TABLE_DIR, 'tab_spanning_regression.tex')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'\n  wrote {path}')


def save_workbook(results):
    """output/spanning_regression.xlsx: one sheet per regression with the rows
    numbers_tex.py reads (var, coef, se, t, p), plus an Info sheet with N and R2."""
    path = os.path.join(OUTPUT_DIR, 'spanning_regression.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for sheet, r in results.items():
            rows = [{'var': k, **v} for k, v in r.items() if not k.startswith('_')]
            pd.DataFrame(rows).to_excel(xw, sheet_name=sheet, index=False)
        pd.DataFrame([{'sheet': s, 'n_months': r['_n'], 'r2': r['_r2']}
                      for s, r in results.items()]).to_excel(xw, sheet_name='Info', index=False)
    print(f'  wrote {path}')


def main():
    print('=' * 78)
    print('  SPANNING REGRESSION: RRR LONG-SHORT vs GROWTH LONG-SHORT')
    print('=' * 78)

    ff = av2.phase1b_load_ff_factors()
    xlsx = os.path.join(OUTPUT_DIR, 'signal_sort_robustness.xlsx')
    raw_xlsx = os.path.join(OUTPUT_DIR, 'ta4_signal_analysis.xlsx')
    rrr = pd.read_excel(raw_xlsx, sheet_name='Baseline_Returns', index_col=0)['Q1-Q4']
    growth = pd.read_excel(xlsx, sheet_name='revgrowth_raw_ret', index_col=0)['Q1-Q4']
    rrr.index = pd.to_datetime(rrr.index)
    growth.index = pd.to_datetime(growth.index)

    overlap = rrr.index.intersection(growth.index)
    print(f'\n  RRR series: {len(rrr)} months. Growth series: {len(growth)} months. '
          f'Overlap: {len(overlap)} months.')

    ff3 = ['Mkt-RF', 'SMB', 'HML']
    ff5 = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    r1_ff3 = spanning_reg(rrr, growth, ff, ff3, 'growth_LS')
    r1_ff5 = spanning_reg(rrr, growth, ff, ff5, 'growth_LS')
    r2_ff3 = spanning_reg(growth, rrr, ff, ff3, 'RRR_LS')
    r2_ff5 = spanning_reg(growth, rrr, ff, ff5, 'RRR_LS')

    print('\n  --- Direction 1: RRR spanned by growth? ---')
    for label, r in [('FF3', r1_ff3), ('FF5', r1_ff5)]:
        print(f'    {label}: alpha={r["const"]["coef"]*100:.4f}%/mo '
              f'(p={r["const"]["p"]:.4f})  beta_growth={r["growth_LS"]["coef"]:.4f} '
              f'(p={r["growth_LS"]["p"]:.4f})  R2={r["_r2"]:.4f}')

    print('\n  --- Direction 2: growth spanned by RRR? ---')
    for label, r in [('FF3', r2_ff3), ('FF5', r2_ff5)]:
        print(f'    {label}: alpha={r["const"]["coef"]*100:.4f}%/mo '
              f'(p={r["const"]["p"]:.4f})  beta_RRR={r["RRR_LS"]["coef"]:.4f} '
              f'(p={r["RRR_LS"]["p"]:.4f})  R2={r["_r2"]:.4f}')

    print('\n  --- For reference: the headline long-short alone ---')
    print(f'    RRR long-short (raw, headline): FF5 {av2.EXPECTED_HEADLINE_FF5_ALPHA_PCT:.4f}%/mo '
          '(config.EXPECTED_HEADLINE_FF5_ALPHA_PCT)')

    make_table(r1_ff5, r2_ff5, r1_ff3, r2_ff3)
    save_workbook({'RRR_on_growth_FF3': r1_ff3, 'RRR_on_growth_FF5': r1_ff5,
                   'growth_on_RRR_FF3': r2_ff3, 'growth_on_RRR_FF5': r2_ff5})

    print()
    print('=' * 78)
    print('  DONE')
    print('=' * 78)


if __name__ == '__main__':
    main()
