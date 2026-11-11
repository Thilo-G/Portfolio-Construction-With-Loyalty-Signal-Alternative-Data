"""concentration_trim_robustness.py -- the premium without its largest positions
and without its largest contributors.

Two trims, both applied to BOTH legs at once, both on the baseline quartile
boundaries (nothing is re-quartiled; the k=0 row reproduces the headline), and
both re-weighting the remaining names so each leg stays fully invested and the
book dollar-neutral (av2.build_portfolio_returns re-weights by MCAP_FORM within
whatever names remain in the bucket).

1. Market-cap trim (Appendix table tab_concentration_trim). Each quarter, the
   top-k names by formation weight leave Q1 and the top-k leave Q4. It answers
   "is the spread a few very large positions?", the objection raised against
   the value-weighted satisfaction portfolios.

2. Contributor trim (author decision 2026-11-10; Appendix table
   tab_contributor_trim and row 16 of the robustness table). A firm's
   contribution to a leg is its formation weight times its monthly return,
   summed over every month it is in that leg. The k firms that added most to
   the long leg's return and the k firms that subtracted most from the short
   leg's return over the whole sample are removed from those legs in every
   quarter they are in them. It answers "is the spread a handful of stocks?"
   (Jacobson and Mizik 2009 traced the satisfaction return to ten stocks).
   Because the firms are chosen on realized returns, the trim lowers the alpha
   by construction; the text says so.

Every data-loading and portfolio step is imported from robustness_diagnostics
and analysis_v2 rather than reimplemented, so the timing convention cannot
drift from the canonical chain.

Outputs
    output/concentration_trim_robustness.xlsx
        Summary            market-cap trim, k = 0..3
        Summary_contrib    contributor trim, k = 0..3
        Contrib_Q1/Q4      every firm's summed contribution to each leg
        Contrib_removed    the firms removed at k = 3, in order, with names
        <portfolio sheets> monthly returns of every trimmed book
    output/concentration_trim_manifest.json
    Paper_LaTeX/tables/tab_concentration_trim.tex
    Paper_LaTeX/tables/tab_contributor_trim.tex
"""

# =============================================================================
# Imports
# =============================================================================
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402
import performance_metrics as pm          # noqa: E402
import robustness_diagnostics as rd       # noqa: E402

# =============================================================================
# Constants
# =============================================================================
RUN_ID = datetime.now().strftime('%Y%m%d-%H%M%S')

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
OUTPUT_DIR = av2.OUTPUT_DIR
TABLE_DIR = os.path.join(_PROJECT, 'Paper_LaTeX', 'tables')

K_VALUES = [1, 2, 3]
BUCKET = 'RRR_Q_MAIN'


# =============================================================================
# Market-cap trim
# =============================================================================
def trim_both_legs(panel, k, market_ret, ff_factors):
    """Drop the top-k formation-weight names from Q1 and from Q4 each quarter."""
    w_q1 = rd.leg_weights_by_quarter(panel, BUCKET, 'Q1')
    w_q4 = rd.leg_weights_by_quarter(panel, BUCKET, 'Q4')

    drops = []
    excluded_weight = {}
    for leg, w in [('Q1', w_q1), ('Q4', w_q4)]:
        ranked = w.copy()
        ranked['RANK'] = ranked.groupby('QUARTER')['MCAP_FORM'].rank(
            method='first', ascending=False)
        hit = ranked.loc[ranked['RANK'] <= k, ['FIRM', 'QUARTER']].copy()
        hit['DROP_LEG'] = leg
        drops.append(hit)
        excluded_weight[leg] = (
            ranked.loc[ranked['RANK'] <= k]
            .groupby('QUARTER')['WEIGHT'].sum()
            .reindex(ranked['QUARTER'].unique()).mean() * 100
        )

    drop_keys = pd.concat(drops, ignore_index=True)

    df_k = panel.merge(drop_keys, on=['FIRM', 'QUARTER'], how='left')
    # Drop a name only from the leg it was ranked in, never from the other leg.
    df_k['BUCKET_TRIM'] = df_k[BUCKET].where(~(df_k[BUCKET] == df_k['DROP_LEG']), np.nan)

    port = av2.build_portfolio_returns(df_k, 'BUCKET_TRIM', market_ret)
    reg = av2.run_factor_regressions(port, ff_factors, f'Trim top-{k} both legs')

    return port, reg, excluded_weight, df_k


# =============================================================================
# Contributor trim
# =============================================================================
def contributions_by_firm(panel, leg):
    """Each firm's contribution to the leg's value-weighted return, summed over
    every holding month in which it is in the leg: sum_m w_{i,m} r_{i,m}, with
    the formation-cap weights build_portfolio_returns uses (normalized within
    each holding month's bucket). Positive values raised the leg's return,
    negative values lowered it."""
    df = panel.loc[panel[BUCKET] == leg,
                   ['FIRM', 'Date', 'QUARTER', 'MCAP_FORM', 'RET_SIMPLE']].copy()
    df['MCAP_FORM'] = pd.to_numeric(df['MCAP_FORM'], errors='coerce')
    df['RET_SIMPLE'] = pd.to_numeric(df['RET_SIMPLE'], errors='coerce')
    df = df.dropna(subset=['MCAP_FORM', 'RET_SIMPLE'])
    df['WEIGHT'] = df['MCAP_FORM'] / df.groupby('Date')['MCAP_FORM'].transform('sum')
    df['CONTRIB'] = df['WEIGHT'] * df['RET_SIMPLE']
    out = df.groupby('FIRM').agg(CONTRIB=('CONTRIB', 'sum'),
                                 QUARTERS_IN_LEG=('QUARTER', 'nunique'),
                                 MEAN_WEIGHT=('WEIGHT', 'mean'))
    out['CONTRIB_PCT_SUM'] = out['CONTRIB'] * 100          # summed monthly contribution, in percent
    return out.sort_values('CONTRIB', ascending=False)


def firm_names():
    """Ticker -> listed name (without legal suffixes), from the sample list
    firm_table.py wrote (columns FIRM, LISTED_NAME); falls back to the ticker
    without the exchange suffix when the file is missing."""
    path = os.path.join(OUTPUT_DIR, 'sample_firms.xlsx')
    names = {}
    if os.path.exists(path):
        firms = pd.read_excel(path)
        if {'FIRM', 'LISTED_NAME'} <= set(firms.columns):
            names = dict(zip(firms['FIRM'].astype(str).str.upper(), firms['LISTED_NAME'].astype(str)))
    return names


def display_name(firm, names):
    if firm in names and names[firm] and names[firm] != 'nan':
        return names[firm]
    return firm.replace(' US EQUITY', '').replace(' EQUITY', '')


def trim_top_contributors(panel, k, market_ret, ff_factors, contrib_q1, contrib_q4):
    """Remove the k largest positive contributors from Q1 and the k largest
    negative contributors from Q4, in every quarter those firms are in the leg."""
    drop_q1 = list(contrib_q1.index[:k])                               # added most to the long leg
    drop_q4 = list(contrib_q4.sort_values('CONTRIB').index[:k])        # subtracted most from the short leg

    df_k = panel.copy()
    in_drop = (((df_k[BUCKET] == 'Q1') & df_k['FIRM'].isin(drop_q1))
               | ((df_k[BUCKET] == 'Q4') & df_k['FIRM'].isin(drop_q4)))
    df_k['BUCKET_TRIM'] = df_k[BUCKET].where(~in_drop, np.nan)

    port = av2.build_portfolio_returns(df_k, 'BUCKET_TRIM', market_ret)
    reg = av2.run_factor_regressions(port, ff_factors, f'Contributor trim top-{k} both legs')

    excluded_weight = {}
    for leg, drop in (('Q1', drop_q1), ('Q4', drop_q4)):
        w = rd.leg_weights_by_quarter(panel, BUCKET, leg)
        excluded_weight[leg] = (
            w.loc[w['FIRM'].isin(drop)].groupby('QUARTER')['WEIGHT'].sum()
            .reindex(w['QUARTER'].unique()).fillna(0.0).mean() * 100
        )
    return port, reg, excluded_weight, df_k, drop_q1, drop_q4


# =============================================================================
# Formatting and tables
# =============================================================================
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
    # House style drops the leading zero: [.174], not [0.174].
    return '[$<$ .001]' if p < 0.001 else '[' + f'{p:.3f}'.lstrip('0') + ']'


def _table_lines(rows, note):
    L = ['\\begin{threeparttable}', '\\footnotesize',
         '\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrrr@{}}',
         '\\toprule',
         'Specification & FF3 $\\alpha$ (\\%/mo) & FF5 $\\alpha$ (\\%/mo) & $N$ '
         '& Weight removed (\\%) \\\\',
         '\\midrule']
    for r in rows:
        L.append(f"{r['label']} & {_fmt(r['ff3_alpha'])} & {_fmt(r['ff5_alpha'])} & "
                 f"{int(r['n'])} & {r['weight_note']} \\\\")
        L.append(f" & ({_fmt(r['ff3_se'])}) {_pv(r['ff3_p'])} & "
                 f"({_fmt(r['ff5_se'])}) {_pv(r['ff5_p'])} & & \\\\")
    L += ['\\bottomrule', '\\end{tabularx}', '\\begin{tablenotes}', '\\small',
          '\\item \\textit{Notes:} ' + note, '\\end{tablenotes}', '\\end{threeparttable}']
    return L


def make_table(rows):
    note = ('The $k$ largest names by formation market cap are removed from each leg '
            'after quartile assignment, and both legs are re-weighted across the '
            'remaining names, so the book stays dollar-neutral on baseline quartile '
            "boundaries. The final column reports the average share of each leg's weight "
            'removed; Newey-West standard errors in parentheses, exact $p$-values in brackets.')
    path = os.path.join(TABLE_DIR, 'tab_concentration_trim.tex')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(_table_lines(rows, note)) + '\n')
    print(f'\n  wrote {path}')


def _name_list(names):
    if len(names) <= 1:
        return ''.join(names)
    return ', '.join(names[:-1]) + (',' if len(names) > 2 else '') + ' and ' + names[-1]


def make_contrib_table(rows, removed_q1, removed_q4):
    note = ("A firm's contribution to a leg is its formation weight times its monthly return, "
            'summed over the months it is in that leg; the $k$ firms that added most to the long '
            "leg's return and the $k$ that subtracted most from the short leg's are removed from "
            'those legs in every quarter they are in them, and the remaining names are re-weighted. '
            f'Removed, in order: {_name_list(removed_q1)} from Q1 and {_name_list(removed_q4)} from Q4. '
            "The final column reports the average share of each leg's weight removed; Newey-West "
            'standard errors in parentheses, exact $p$-values in brackets.')
    path = os.path.join(TABLE_DIR, 'tab_contributor_trim.tex')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(_table_lines(rows, note)) + '\n')
    print(f'  wrote {path}')


def _row(label, reg, n, weight_note, panel_k=None):
    r3, r5 = reg.get('Q1-Q4_FF3', {}), reg.get('Q1-Q4_FF5', {})
    row = {
        'label': label,
        'ff3_alpha': r3.get('alpha', np.nan) * 100,
        'ff3_se': r3.get('se_alpha', np.nan) * 100,
        'ff3_p': r3.get('alpha_p', np.nan),
        'ff5_alpha': r5.get('alpha', np.nan) * 100,
        'ff5_se': r5.get('se_alpha', np.nan) * 100,
        'ff5_p': r5.get('alpha_p', np.nan),
        'n': n,
        'weight_note': weight_note,
    }
    if panel_k is not None:
        # Turnover of the TRIMMED book, not the baseline's: removing names changes
        # what has to be traded at each reformation.
        _a, _n, _d, _s, to_ann = pm.compute_portfolio_turnover(
            panel_k, 'BUCKET_TRIM', quarters_between=1,
            reformations_per_year=pm.REFORMATIONS_PER_YEAR)
        row['turnover_ann_ls'] = float(to_ann.get('Q1-Q4', np.nan)) * 100
    # Per-leg alphas: the paper reports Q1 and Q4 for every robustness row.
    for leg in ('Q1', 'Q4'):
        for mdl in ('FF3', 'FF5'):
            rr = reg.get(f'{leg}_{mdl}', {}) or {}
            tag = leg + '_' + mdl.lower()
            row[tag + '_alpha'] = rr.get('alpha', np.nan) * 100
            row[tag + '_se'] = rr.get('se_alpha', np.nan) * 100
            row[tag + '_p'] = rr.get('alpha_p', np.nan)
    return row


# =============================================================================
# Main
# =============================================================================
def main():
    print('=' * 78)
    print('  CONCENTRATION TRIMS: LARGEST POSITIONS AND LARGEST CONTRIBUTORS')
    print(f'  RUN ID: {RUN_ID}')
    print('=' * 78)

    df_filtered, returns, ff_factors, panel, market_ret = rd.load_base_data()
    base_port, base_reg = rd.baseline_portfolio_and_regressions(
        panel, ff_factors, market_ret)
    n_base = base_reg['Q1-Q4_FF3']['n_obs']
    base_row = _row('Baseline ($k=0$)', base_reg, n_base, '0.0')

    ports = {'baseline': base_port}

    # ---- 1. market-cap trim --------------------------------------------------
    rows = [base_row]
    print('\n  Long-short alpha after a SYMMETRIC trim of the top-k names by market cap in each leg:')
    for k in K_VALUES:
        port, reg, exw, panel_k = trim_both_legs(panel, k, market_ret, ff_factors)
        ports[f'trim_k{k}'] = port
        rows.append(_row(f'Trim top-{k} from both legs', reg, reg.get('Q1-Q4_FF3', {}).get('n_obs', 0),
                         f"Q1 {exw['Q1']:.1f} / Q4 {exw['Q4']:.1f}", panel_k))
        print(f"    k={k}: FF3 {rows[-1]['ff3_alpha']:7.4f}%/mo [p={rows[-1]['ff3_p']:.3f}]  "
              f"FF5 {rows[-1]['ff5_alpha']:7.4f}%/mo [p={rows[-1]['ff5_p']:.3f}]  "
              f"(removed Q1 {exw['Q1']:.1f}%, Q4 {exw['Q4']:.1f}% of leg weight)")
    make_table(rows)

    # ---- 2. contributor trim -------------------------------------------------
    contrib_q1 = contributions_by_firm(panel, 'Q1')
    contrib_q4 = contributions_by_firm(panel, 'Q4')
    names = firm_names()
    print(f'\n  Firm names found for {len(names)} tickers (sample_firms.xlsx)')
    print('\n  Largest contributors to the long leg (summed weight x return, %):')
    for firm, r in contrib_q1.head(5).iterrows():
        print(f"    {display_name(firm, names):<32} {r['CONTRIB_PCT_SUM']:8.2f}  "
              f"({int(r['QUARTERS_IN_LEG'])} quarters in Q1, mean weight {r['MEAN_WEIGHT']:.1%})")
    print('  Largest negative contributors to the short leg:')
    for firm, r in contrib_q4.sort_values('CONTRIB').head(5).iterrows():
        print(f"    {display_name(firm, names):<32} {r['CONTRIB_PCT_SUM']:8.2f}  "
              f"({int(r['QUARTERS_IN_LEG'])} quarters in Q4, mean weight {r['MEAN_WEIGHT']:.1%})")

    crows = [dict(base_row)]
    removed_q1, removed_q4 = [], []
    print('\n  Long-short alpha after removing the top-k contributors from each leg:')
    for k in K_VALUES:
        port, reg, exw, panel_k, drop_q1, drop_q4 = trim_top_contributors(
            panel, k, market_ret, ff_factors, contrib_q1, contrib_q4)
        ports[f'contrib_k{k}'] = port
        removed_q1, removed_q4 = drop_q1, drop_q4
        crows.append(_row(f'Top-{k} contributors from both legs', reg,
                          reg.get('Q1-Q4_FF3', {}).get('n_obs', 0),
                          f"Q1 {exw['Q1']:.1f} / Q4 {exw['Q4']:.1f}", panel_k))
        print(f"    k={k}: FF3 {crows[-1]['ff3_alpha']:7.4f}%/mo [p={crows[-1]['ff3_p']:.3f}]  "
              f"FF5 {crows[-1]['ff5_alpha']:7.4f}%/mo [p={crows[-1]['ff5_p']:.3f}]  "
              f"Q1 {crows[-1]['Q1_ff5_alpha']:7.4f} [p={crows[-1]['Q1_ff5_p']:.3f}]  "
              f"Q4 {crows[-1]['Q4_ff5_alpha']:7.4f} [p={crows[-1]['Q4_ff5_p']:.3f}]  "
            f"(removed Q1 {exw['Q1']:.1f}%, Q4 {exw['Q4']:.1f}% of leg weight; "
              f"Q1: {', '.join(display_name(f, names) for f in drop_q1)}; "
              f"Q4: {', '.join(display_name(f, names) for f in drop_q4)})")

    removed = pd.DataFrame(
        [{'leg': 'Q1', 'rank': i + 1, 'firm': f, 'name': display_name(f, names),
          'contrib_pct_sum': contrib_q1.loc[f, 'CONTRIB_PCT_SUM'],
          'quarters_in_leg': int(contrib_q1.loc[f, 'QUARTERS_IN_LEG'])}
         for i, f in enumerate(removed_q1)]
        + [{'leg': 'Q4', 'rank': i + 1, 'firm': f, 'name': display_name(f, names),
            'contrib_pct_sum': contrib_q4.loc[f, 'CONTRIB_PCT_SUM'],
            'quarters_in_leg': int(contrib_q4.loc[f, 'QUARTERS_IN_LEG'])}
           for i, f in enumerate(removed_q4)])
    make_contrib_table(crows, [display_name(f, names) for f in removed_q1],
                       [display_name(f, names) for f in removed_q4])

    # ---- 3. persist ----------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx = os.path.join(OUTPUT_DIR, 'concentration_trim_robustness.xlsx')
    with pd.ExcelWriter(xlsx, engine='openpyxl') as xw:
        pd.DataFrame(rows).to_excel(xw, sheet_name='Summary', index=False)
        pd.DataFrame(crows).to_excel(xw, sheet_name='Summary_contrib', index=False)
        contrib_q1.reset_index().to_excel(xw, sheet_name='Contrib_Q1', index=False)
        contrib_q4.sort_values('CONTRIB').reset_index().to_excel(xw, sheet_name='Contrib_Q4', index=False)
        removed.to_excel(xw, sheet_name='Contrib_removed', index=False)
        for name, p in ports.items():
            p.to_excel(xw, sheet_name=name[:31])
    print(f'  wrote {xlsx}')

    with open(os.path.join(OUTPUT_DIR, 'concentration_trim_manifest.json'), 'w',
              encoding='utf-8') as fh:
        json.dump({'run_id': RUN_ID,
                   'script': 'concentration_trim_robustness.py',
                   'k_values': K_VALUES,
                   'rows': rows,
                   'contrib_rows': crows,
                   'contrib_removed': removed.to_dict(orient='records')},
                  fh, indent=2, default=float)

    print()
    print('=' * 78)
    print('  DONE')
    print('=' * 78)


if __name__ == '__main__':
    main()
