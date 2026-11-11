"""h2_asymmetry_test.py -- direct test of H2 (asymmetry).

H2 claims the RRR return spread is driven by low-RRR firms, not high-RRR firms.
The paper's evidence for this so far is indirect: Q4's alpha clears significance
and Q1's does not (Table 5). That is a comparison of two significance verdicts,
not a test of the asymmetry itself, and a referee correctly asked for the direct
version.

Direct test: if the two extreme quartiles were true mirror images (Q1 alpha = +a,
Q4 alpha = -a), an equally-weighted combination of the two, (Q1+Q4)/2, would have
zero alpha by construction. If that combined portfolio's alpha is significantly
negative, that is direct evidence the negative pull from Q4 is not offset by an
equal positive pull from Q1 -- exactly the asymmetry H2 claims, tested rather than
inferred.

This is algebraically a Wald test of the linear restriction alpha_Q1 = -alpha_Q4:
regressing (r_Q1 + r_Q4)/2 on the same factors gives an intercept that is exactly
(alpha_Q1 + alpha_Q4)/2, with a directly interpretable standard error, using only
the return series the paper already computes. No new data, no new signal.

Outputs
    output/h2_asymmetry_test.xlsx
    Paper_LaTeX/tables/tab_h2_asymmetry.tex
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402
import robustness_diagnostics as rd       # noqa: E402

RUN_ID = datetime.now().strftime('%Y%m%d-%H%M%S')

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
OUTPUT_DIR = av2.OUTPUT_DIR
TABLE_DIR = os.path.join(_PROJECT, 'Paper_LaTeX', 'tables')


def _fmt(v, dp=3):
    if v is None or (isinstance(v, float) and v != v):
        return '--'
    # AMA rules (JMR/JoM): at most three decimals, no zero before the decimal point.
    out = format(float(v), f'.{min(dp, 3)}f')
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


def make_table(rows):
    os.makedirs(TABLE_DIR, exist_ok=True)
    L = []
    L.append('\\begin{threeparttable}')
    L.append('\\footnotesize')
    # 2026-09-29 (proofreading): the stray empty fourth column is removed.
    L.append('\\begin{tabular}{lrr}')
    L.append('\\toprule')
    L.append(' & FF3 & FF5 \\\\')
    L.append('\\midrule')
    # Row label and note as the author shortened them by hand on 2026-11-05.
    for label, key in [('$\\alpha_{Q1}$ (Table~\\ref{tab:portfolio_alphas_raw})', 'Q1'),
                       ('$\\alpha_{Q4}$ (Table~\\ref{tab:portfolio_alphas_raw})', 'Q4'),
                       ('$(\\alpha_{Q1}+\\alpha_{Q4})/2$', 'avg')]:
        r3, r5 = rows[key]['FF3'], rows[key]['FF5']
        L.append(f"{label} & {_fmt(r3['alpha']*100)} & {_fmt(r5['alpha']*100)} \\\\")
        L.append(f" & ({_fmt(r3['se']*100)}) {_pv(r3['p'])} & "
                 f"({_fmt(r5['se']*100)}) {_pv(r5['p'])} \\\\")
    L.append('\\bottomrule')
    L.append('\\end{tabular}')
    L.append('\\begin{tablenotes}')
    L.append('\\small')
    L.append('\\item \\textit{Notes:} The third row regresses the equally weighted '
             'combination of the two extreme legs, $(r_{Q1}+r_{Q4})/2$, on the same '
             'factors. Newey-West standard errors in parentheses, exact $p$-values '
             'in brackets.')
    L.append('\\end{tablenotes}')
    L.append('\\end{threeparttable}')

    path = os.path.join(TABLE_DIR, 'tab_h2_asymmetry.tex')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'\n  wrote {path}')


def main():
    print('=' * 78)
    print('  H2 ASYMMETRY -- DIRECT TEST')
    print(f'  RUN ID: {RUN_ID}')
    print('=' * 78)

    df_filtered, returns, ff_factors, panel, market_ret = rd.load_base_data()
    port, reg = rd.baseline_portfolio_and_regressions(panel, ff_factors, market_ret)

    port2 = port.copy()
    port2['avg'] = (port2['Q1'] + port2['Q4']) / 2.0
    reg2 = av2.run_factor_regressions(port2, ff_factors, label='H2_asymmetry')

    rows = {'Q1': {}, 'Q4': {}, 'avg': {}}
    for spec in ['FF3', 'FF5']:
        for leg, source in [('Q1', reg), ('Q4', reg)]:
            r = source.get(f'{leg}_{spec}', {})
            rows[leg][spec] = {'alpha': r.get('alpha', np.nan),
                               'se': r.get('se_alpha', np.nan),
                               'p': r.get('alpha_p', np.nan)}
        r = reg2.get(f'avg_{spec}', {})
        rows['avg'][spec] = {'alpha': r.get('alpha', np.nan),
                             'se': r.get('se_alpha', np.nan),
                             'p': r.get('alpha_p', np.nan)}

    print('\n  Direct test of H2 -- (alpha_Q1 + alpha_Q4)/2:')
    for spec in ['FF3', 'FF5']:
        a = rows['avg'][spec]
        print(f'    {spec}: {a["alpha"]*100:7.4f}%/mo  SE={a["se"]*100:.4f}  '
              f'p={a["p"]:.4f}')

    make_table(rows)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx = os.path.join(OUTPUT_DIR, 'h2_asymmetry_test.xlsx')
    with pd.ExcelWriter(xlsx, engine='openpyxl') as xw:
        port2.to_excel(xw, sheet_name='Portfolio_Returns')
        pd.DataFrame({f'{leg}_{spec}': rows[leg][spec]
                      for leg in rows for spec in ['FF3', 'FF5']}).T.to_excel(
            xw, sheet_name='Results')
    print(f'  wrote {xlsx}')

    with open(os.path.join(OUTPUT_DIR, 'h2_asymmetry_manifest.json'), 'w',
              encoding='utf-8') as fh:
        json.dump({'run_id': RUN_ID, 'script': 'h2_asymmetry_test.py',
                   'rows': {k: {s: {kk: (None if vv != vv else vv)
                                    for kk, vv in v.items()}
                                for s, v in rows[k].items()}
                           for k in rows}},
                  fh, indent=2, default=float)

    print()
    print('=' * 78)
    print('  DONE')
    print('=' * 78)


if __name__ == '__main__':
    main()
