"""asymmetry_across_variants.py -- the paper's single robustness table.

Every robustness check in the paper appears here in one table, in one column
set: long-short, Q1 and Q4 five-factor alphas with standard errors and exact
p-values, the estimation month count, and annualized turnover. It replaces the
six separate robustness tables earlier drafts carried (author instruction,
2026-09-18).

WHAT THIS SCRIPT DOES NOT DO: it computes no regression and rebuilds no
portfolio. Every number is read back from a workbook or manifest another script
already wrote and already verified. This file is an aggregator and a table
writer. That is deliberate -- a summary table whose numbers were recomputed
here could silently disagree with the analysis that produced them.

SIGNAL BASIS. Raw is the paper's primary specification (author decision,
2026-09-18) and every row is raw unless its label says otherwise. Raw therefore
carries no qualifier in the labels; only the industry-time adjusted row is
marked, because marking the default would be noise.

TWO ROWS THAT ARE EASY TO CONFLATE, and the table groups them apart for that
reason. The five-month formation lag holds the signal fixed and moves the trade
date, so it measures how fast the information decays out of the price. The
two-quarter RRR holds the trade date fixed and moves the denominator, so it
measures how far back retention stays informative. One varies when you act, the
other varies what you measure.

UNIT TRAP, deliberately handled per source: the Turnover sheet in
annual_rrr_robustness.xlsx is already in percent (its row label says so),
while those in ta4_signal_analysis.xlsx and rcb4_signal_analysis.xlsx hold
decimals (2.4115 meaning 241.15%). The two manifests are in percent. Each
source declares its own scaling below and a range assertion catches any 100x
slip.

ORIENTATION TRAP: the two families of Turnover sheet are transposed relative to
each other. The annual sheet indexes by metric name with legs as columns;
TA4/RCB4 index by leg with metrics as columns. Each row states its own
orientation rather than relying on a guess.

Outputs
    Paper_LaTeX/tables/tab_robustness_summary.tex
"""

import json
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
OUTPUT_DIR = av2.OUTPUT_DIR
TABLE_DIR = os.path.join(_PROJECT, 'Paper_LaTeX', 'tables')

SSM = 'signal_sort_manifest.json'
EVM = 'estimation_variants_manifest.json'
TA4 = 'ta4_signal_analysis.xlsx'
RCB4 = 'rcb4_signal_analysis.xlsx'
ANN = 'annual_rrr_robustness.xlsx'
TRIM = 'concentration_trim_robustness.xlsx'

# (label, reason, source)
#   source = ('man', manifest_file, key)
#            ('xlsx', workbook, sheet, turnover_spec)
#            ('trim', sheet, row_label)      sheet of concentration_trim_robustness.xlsx
# turnover_spec = (workbook, sheet, key, scale, orientation)

PANEL_A = [
    ('\\RRR{} (headline)',
     'The signal as defined, with no further processing.',
     ('man', SSM, 'baseline')),

    ('\\RRR{}, industry-adjusted',
     'Does the premium survive removing sector-quarter means, so it is not a '
     'sector bet in disguise?',
     ('man', SSM, 'rrr_adj')),

    ('Two-quarter \\RRR{} ($t-2$ denominator)',
     'How far back does retention stay informative when the window widens from '
     'one quarter to two?',
     ('man', SSM, 'tq_rrr_raw')),

    ('Four-quarter \\RRR{} ($t-4$ denominator)',
     'The same question at a one-year window: retention measured against total '
     'revenue four quarters earlier.',
     ('man', SSM, 'lr_rrr_raw')),

    ('Share of returning revenue',
     'Does bounding the denominator at current revenue, rather than leaving it '
     'unbounded at the prior period, change the result?',
     ('man', SSM, 'sorr_raw')),

    ('TA4, trailing-mean denominator',
     'Dilutes the term the denominator shares with the lagged numerator: a '
     'four-quarter mean in place of the single prior quarter.',
     ('xlsx', TA4, 'TA4_FactorReg',
      (TA4, 'Turnover', 'TA4_annualized', 100.0, 'metric_col'))),

    # RCB3 removed 2026-09-27 (author: "out, wrong signal"), replaced by the
    # moving average over quarters t-4 to t-1 below.
    ('RCB4, rolling customer base (4 quarters)',
     'Rebuilds the denominator from total revenue four quarters back plus the '
     'new-customer revenue acquired since, so it shares no term with the '
     'lagged numerator.',
     ('xlsx', RCB4, 'RCB4_FactorReg',
      (RCB4, 'Turnover', 'RCB4_annualized', 100.0, 'metric_col'))),

    # Author decision 2026-09-27: mean RRR over the four quarters before the
    # measurement quarter. Replaces the earlier t-3..t moving average
    # (ma4_signal_robustness.py), which is no longer reported.
    ('Four-quarter moving average ($t-4$ to $t-1$)',
     'Rewards firms whose retention has been durably high; averaging four '
     'readings dilutes any single-quarter spike.',
     ('man', SSM, 'ma4lag_raw')),

    ('Annual \\RRR{}, annual re-formation',
     'Does the premium survive re-sorting once a year on an annual signal, at a '
     'fraction of the trading the quarterly book requires?',
     ('xlsx', ANN, 'Factor_Regressions',
      (ANN, 'Turnover', 'Turnover (annualized, %)', 1.0, 'metric_row'))),

    ('Prior-period revenue growth',
     'Is this simply a revenue-growth premium, given that returning-customer '
     'revenue is a large part of revenue growth?',
     ('man', SSM, 'revgrowth_raw')),
]

PANEL_B = [
    ('First-half subperiod',
     'Is the premium present in the early sample, or concentrated in one era?',
     ('man', EVM, 'first_half')),

    ('Second-half subperiod',
     'The same question for the later sample, when the cross-section is smaller.',
     ('man', EVM, 'second_half')),

    ('Excluding COVID-19 (2020Q1--2021Q2)',
     'Does the pandemic-era swing in consumer behaviour drive the result? '
     'Dropped from formation and estimation, not absorbed by a dummy.',
     ('man', EVM, 'no_covid')),

    ('Equal-weighted',
     'Is the premium a large-firm phenomenon that value weighting magnifies?',
     ('man', EVM, 'equal_weighted')),

    ('Five-month formation lag',
     'How fast does the information decay? The signal is unchanged; only the '
     'trade date moves, from two months after quarter-end to five.',
     ('man', EVM, 'lag5')),

    # Author decision 2026-11-10: the robustness row removes the largest
    # CONTRIBUTORS (summed weight x return over the sample), not the largest
    # names by market cap; the market-cap trim stays in the Appendix only
    # (concentration_trim_robustness.py, sheets Summary_contrib and Summary).
    ('Two largest contributors removed from both legs',
     'Does the premium depend on a handful of stocks? The two firms that added '
     'most to the long leg and the two that subtracted most from the short leg '
     'over the sample are removed from their legs.',
     ('trim', 'Summary_contrib', 'Top-2 contributors from both legs')),
]


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def _load_manifest(name):
    with open(os.path.join(OUTPUT_DIR, name), 'r', encoding='utf-8') as fh:
        return json.load(fh)


def read_manifest_row(manifest_file, key):
    m = _load_manifest(manifest_file)
    block = m.get('signals', m.get('variants', {}))
    if key not in block:
        raise KeyError(f'{manifest_file}: missing key {key}')
    v = block[key]
    if v is None:
        raise KeyError(f'{manifest_file}: key {key} is null')
    out = {}
    for leg, tag in [('LS', 'Q1Q4'), ('Q1', 'Q1'), ('Q4', 'Q4')]:
        out[leg] = {
            'alpha': float(v[f'{tag}_ff5_alpha_pct_mo']) * 100.0,
            'se': float(v[f'{tag}_ff5_se']) * 100.0,
            'p': float(v[f'{tag}_ff5_p']),
        }
    out['n'] = int(v['Q1Q4_ff5_n'])
    out['turnover'] = float(v['turnover_ann_ls'])
    return out


def read_turnover(spec):
    workbook, sheet, key, scale, orient = spec
    d = pd.read_excel(os.path.join(OUTPUT_DIR, workbook), sheet_name=sheet, index_col=0)
    if orient == 'metric_row':
        if key not in d.index:
            raise KeyError(f'{workbook}/{sheet}: missing row {key}')
        return float(d.loc[key, 'Q1-Q4']) * scale
    if key not in d.columns:
        raise KeyError(f'{workbook}/{sheet}: missing column {key}')
    return float(d.loc['Q1-Q4', key]) * scale


def read_xlsx_row(workbook, sheet, turnover_spec):
    d = pd.read_excel(os.path.join(OUTPUT_DIR, workbook), sheet_name=sheet, index_col=0)
    out = {}
    for leg, idx in [('LS', 'Q1-Q4_FF5'), ('Q1', 'Q1_FF5'), ('Q4', 'Q4_FF5')]:
        if idx not in d.index:
            raise KeyError(f'{workbook}/{sheet}: missing {idx}')
        r = d.loc[idx]
        out[leg] = {'alpha': float(r['alpha']) * 100.0,
                    'se': float(r['se_alpha']) * 100.0,
                    'p': float(r['alpha_p'])}
    out['n'] = int(d.loc['Q1-Q4_FF5', 'n_obs'])
    out['turnover'] = read_turnover(turnover_spec)
    return out


def read_trim_row(sheet, row_label):
    d = pd.read_excel(os.path.join(OUTPUT_DIR, TRIM), sheet_name=sheet, index_col=0)
    if row_label not in d.index:
        raise KeyError(f'{TRIM}/{sheet}: missing row {row_label}')
    r = d.loc[row_label]
    out = {'LS': {'alpha': float(r['ff5_alpha']), 'se': float(r['ff5_se']),
                  'p': float(r['ff5_p'])}}
    for leg in ('Q1', 'Q4'):
        out[leg] = {'alpha': float(r[f'{leg}_ff5_alpha']),
                    'se': float(r[f'{leg}_ff5_se']),
                    'p': float(r[f'{leg}_ff5_p'])}
    out['n'] = int(r['n'])
    out['turnover'] = float(r['turnover_ann_ls'])
    return out


def read_row(source):
    kind = source[0]
    if kind == 'man':
        return read_manifest_row(source[1], source[2])
    if kind == 'xlsx':
        return read_xlsx_row(source[1], source[2], source[3])
    if kind == 'trim':
        return read_trim_row(source[1], source[2])
    raise ValueError(f'unknown source kind {kind}')


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _fmt(v, dp=3):
    # AMA rules (JMR/JoM): at most three decimals, no zero before the decimal point.
    out = format(float(v), f'.{min(dp, 3)}f')
    neg = out.startswith('-')
    if neg:
        out = out[1:]
    if out.startswith('0.'):
        out = out[1:]
    return ('$-$' + out) if neg else out


def _pv(p):
    # House style drops the leading zero: [.174], not [0.174].
    return '[$<$ .001]' if p < 0.001 else '[' + f'{p:.3f}'.lstrip('0') + ']'


def make_table(rows_a, rows_b):
    os.makedirs(TABLE_DIR, exist_ok=True)
    L = []

    # The reason for each check is explained in the Methodology section, not in
    # the table (author decision 2026-09-27), so the table has six columns and
    # fits \textwidth at \footnotesize.
    head = [
        '{\\footnotesize',
        '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{longtable}{@{}p{4.6cm}rrrrr@{}}',
        '\\caption{Robustness Checks: Long-Short, Q1, and Q4 Five-Factor Alphas, '
        'Estimation Months, and Turnover}\\label{tab:robustness_summary}\\\\',
        '\\toprule',
        'Check & Q1$-$Q4 $\\alpha$ & Q1 $\\alpha$ & Q4 $\\alpha$ & $N$ & Turn. \\\\',
        ' & (\\%/mo) & (\\%/mo) & (\\%/mo) & & (ann., \\%) \\\\',
        '\\midrule',
        '\\endfirsthead',
        '\\multicolumn{6}{@{}l}{\\textit{Table \\ref{tab:robustness_summary} continued}}\\\\',
        '\\toprule',
        'Check & Q1$-$Q4 $\\alpha$ & Q1 $\\alpha$ & Q4 $\\alpha$ & $N$ & Turn. \\\\',
        ' & (\\%/mo) & (\\%/mo) & (\\%/mo) & & (ann., \\%) \\\\',
        '\\midrule',
        '\\endhead',
    ]
    L.extend(head)

    def block(rows):
        for r in rows:
            L.append(
                f"{r['label']} & {_fmt(r['LS']['alpha'])} & "
                f"{_fmt(r['Q1']['alpha'])} & {_fmt(r['Q4']['alpha'])} & "
                f"{r['n']} & {r['turnover']:.2f} \\\\")
            L.append(
                f" & ({_fmt(r['LS']['se'])}) {_pv(r['LS']['p'])} & "
                f"({_fmt(r['Q1']['se'])}) {_pv(r['Q1']['p'])} & "
                f"({_fmt(r['Q4']['se'])}) {_pv(r['Q4']['p'])} & & \\\\")
            L.append('\\addlinespace')

    L.append('\\multicolumn{6}{@{}l}{\\textit{Panel A: Signal construction}} \\\\')
    L.append('\\addlinespace')
    block(rows_a)
    L.append('\\midrule')
    L.append('\\multicolumn{6}{@{}l}{\\textit{Panel B: Estimation and sample}} \\\\')
    L.append('\\addlinespace')
    block(rows_b)
    L.append('\\bottomrule')
    L.append('\\multicolumn{6}{@{}p{0.98\\textwidth}@{}}{\\footnotesize '
             '\\textit{Notes:} Value-weighted quartile portfolios; five-factor alphas '
             'in percent per month, with Newey-West standard errors in parentheses and '
             'exact $p$-values in brackets. $N$ is the number of estimation months, '
             'turnover is the annualized one-way turnover of the long-short book, and '
             'every row uses the raw signal except the industry-adjusted one.} \\\\')
    L.append('\\end{longtable}')
    L.append('}')

    path = os.path.join(TABLE_DIR, 'tab_robustness_summary.tex')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'\n  wrote {path}')



# ---------------------------------------------------------------------------
# Appendix table: what each construction did to the signal itself
# ---------------------------------------------------------------------------

DIAG_ROWS = [
    ('\\RRR{} (headline)', ('man', SSM, 'baseline')),
    ('\\RRR{}, industry-adjusted', ('man', SSM, 'rrr_adj')),
    ('Two-quarter \\RRR{}', ('man', SSM, 'tq_rrr_raw')),
    ('Four-quarter \\RRR{}', ('man', SSM, 'lr_rrr_raw')),
    ('Share of returning revenue', ('man', SSM, 'sorr_raw')),
    ('TA4, trailing-mean denominator', ('pers', TA4, 'TA4')),
    ('RCB4, rolling customer base', ('pers', RCB4, 'RCB4')),
    ('Four-quarter moving average ($t-4$ to $t-1$)', ('man', SSM, 'ma4lag_raw')),
    ('Prior-period revenue growth', ('man', SSM, 'revgrowth_raw')),
]


def read_diag(source):
    """Stay-rate, lag-1 autocorrelation and turnover for one construction."""
    if source[0] == 'man':
        v = _load_manifest(source[1])['signals'][source[2]]
        return {'stay': float(v['stay_rate']),
                'ac1': float(v['lag1_autocorr']),
                'turnover': float(v['turnover_ann_ls'])}
    # TA4 and RCB4 keep their diagnostics in a Persistence sheet of their own,
    # indexed by metric name with the variant as a column.
    workbook, col = source[1], source[2]
    d = pd.read_excel(os.path.join(OUTPUT_DIR, workbook),
                      sheet_name='Persistence', index_col=0)
    t = pd.read_excel(os.path.join(OUTPUT_DIR, workbook),
                      sheet_name='Turnover', index_col=0)
    return {'stay': float(d.loc['Overall stay-rate', col]),
            'ac1': float(d.loc['Lag-1 mean', col]),
            'turnover': float(t.loc['Q1-Q4', col + '_annualized']) * 100.0}


def make_diag_table(rows):
    L = ['\\begin{threeparttable}', '\\footnotesize',
         '\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrr@{}}',
         '\\toprule',
         'Signal construction & Quartile stay-rate & Lag-1 autocorr. '
         '& Turnover (ann., \\%) \\\\',
         '\\midrule']
    for r in rows:
        L.append("%s & %.1f\\%% & %s & %.2f \\\\"
                 % (r['label'], r['stay'] * 100, _fmt(r['ac1']), r['turnover']))
    L += ['\\bottomrule', '\\end{tabularx}',
          '\\begin{tablenotes}', '\\small',
          '\\item \\textit{Notes:} Quartile stay-rate is the share of firms in the same '
          'quartile at the next rebalancing, and lag-1 autocorrelation is the mean across '
          'firms of the within-firm correlation between consecutive quarterly signal '
          'values. Turnover is annualized one-way turnover of the long-short book.',
          '\\end{tablenotes}', '\\end{threeparttable}']
    path = os.path.join(TABLE_DIR, 'tab_signal_diagnostics.tex')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print('  wrote ' + path)

def save_rows(rows_a, rows_b):
    """Write the table's rows to output/robustness_summary.xlsx (added 2026-10-20),
    so robustness_figures.py draws the robustness graph from the same numbers."""
    recs = []
    for panel, rows in (('A', rows_a), ('B', rows_b)):
        for r in rows:
            rec = {'Panel': panel, 'Check': r['label'], 'N months': r['n'],
                   'Turnover (ann., %)': r['turnover']}
            for leg in ('LS', 'Q1', 'Q4'):
                rec[f'{leg} alpha (%/mo)'] = r[leg]['alpha']
                rec[f'{leg} SE'] = r[leg]['se']
                rec[f'{leg} p'] = r[leg]['p']
            recs.append(rec)
    path = os.path.join(OUTPUT_DIR, 'robustness_summary.xlsx')
    pd.DataFrame(recs).to_excel(path, index=False)
    print('  wrote ' + path)


def main():
    print('=' * 78)
    print('  COMBINED ROBUSTNESS TABLE')
    print('=' * 78)

    rows_a, rows_b = [], []
    for target, spec in ((rows_a, PANEL_A), (rows_b, PANEL_B)):
        for label, reason, source in spec:
            rec = read_row(source)
            target.append({'label': label, 'reason': reason, **rec})

    allrows = rows_a + rows_b
    print(f"\n  {'Check':<44}{'Q1-Q4':>18}{'Q1':>18}{'Q4':>18}{'N':>6}{'Turn':>10}")
    print('  ' + '-' * 114)
    for r in allrows:
        lbl = r['label'].replace('\\RRR{}', 'RRR').replace('$t-2$', 't-2') \
                        .replace('$t-4$', 't-4').replace('--', '-')
        print(f"  {lbl:<44}"
              f"{r['LS']['alpha']:>10.4f} [{r['LS']['p']:.3f}]"
              f"{r['Q1']['alpha']:>10.4f} [{r['Q1']['p']:.3f}]"
              f"{r['Q4']['alpha']:>10.4f} [{r['Q4']['p']:.3f}]"
              f"{r['n']:>6}{r['turnover']:>9.2f}%")

    # --- guards ------------------------------------------------------------
    # 1. unit slip: every annualized turnover must be a plausible percentage
    for r in allrows:
        assert 10.0 < r['turnover'] < 1000.0, \
            f"{r['label']}: turnover {r['turnover']} outside plausible percent range"

    # 2. the headline row must reproduce the known raw baseline figure
    head = allrows[0]
    assert abs(head['LS']['alpha'] - av2.EXPECTED_HEADLINE_FF5_ALPHA_PCT) < 0.01, \
        f"headline long-short alpha is {head['LS']['alpha']}, expected {av2.EXPECTED_HEADLINE_FF5_ALPHA_PCT}"

    # 3. no row may silently carry a missing estimate
    for r in allrows:
        for leg in ('LS', 'Q1', 'Q4'):
            v = r[leg]['alpha']
            assert v == v, f"{r['label']}: {leg} alpha is NaN"

    make_table(rows_a, rows_b)
    save_rows(rows_a, rows_b)

    diag = [dict(label=lbl, **read_diag(src)) for lbl, src in DIAG_ROWS]
    make_diag_table(diag)
    # Saved next to the robustness rows (added 2026-10-20) for numbers_tex.py.
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, 'robustness_summary.xlsx'), engine='openpyxl',
                        mode='a', if_sheet_exists='replace') as xw:
        pd.DataFrame(diag).assign(stay=lambda d: d['stay'] * 100).rename(
            columns={'label': 'Construction', 'stay': 'Stay-rate (%)',
                                           'ac1': 'Lag-1 autocorrelation',
                                           'turnover': 'Turnover (ann., %)'}).to_excel(
            xw, sheet_name='signal_diagnostics', index=False)

    print()
    print('=' * 78)
    print('  DONE')
    print('=' * 78)


if __name__ == '__main__':
    main()
