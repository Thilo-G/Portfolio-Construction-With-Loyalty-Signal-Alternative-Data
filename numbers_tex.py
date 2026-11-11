"""numbers_tex.py -- RRR Financial Implications
==============================================

Writes Paper_LaTeX/numbers.tex: one LaTeX macro for every result number the
paper's prose cites, read from the outputs the pipeline saved. No number is
typed into the text by hand (python-code-conventions: all output numbers come
from code). main.tex inputs numbers.tex; the section files cite the macros,
for example "earns \\valAlphaLS\\% per month (\\valPLS)".

Conventions
-----------
* Every macro starts with "val" and uses letters only (LaTeX's rule).
* Number format follows the AMA rules the paper uses: at most three decimals,
  no zero before the decimal point (.526), a typeset minus ($-$1.647).
* p-value macros (valP...) carry the whole expression, e.g. "$p=.034$" or
  "$p<.001$". Two-sided throughout.
* After a run, the script prints every macro with its source, so a reader can
  trace each number to a file.

Run (from Code/RRR-FI-IM), after run_pipeline.py:
    ..\\venv\\Scripts\\python.exe numbers_tex.py
"""

# =============================================================================
# Imports
# =============================================================================
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analysis_v2 as av2
import industry_rrr_dispersion as ird

# =============================================================================
# Constants
# =============================================================================
HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'
TEX_PATH = HERE.parent.parent / 'Paper_LaTeX' / 'numbers.tex'

MACROS = {}   # name -> (LaTeX text, source)

# Freshness guard (added 2026-11-11). Every file the macros come from must be
# from the same pipeline run as the canonical panel (step 2); step 1 writes its
# outputs a few minutes before the panel, hence the tolerance. On 2026-10-20 a
# spanning_regression.xlsx from September fed stale numbers into the text while
# the table next to them was current.
FRESHNESS_TOLERANCE_S = 3 * 3600


# =============================================================================
# Formatting
# =============================================================================
def num(v, dp=3):
    """AMA style: dp decimals, no zero before the decimal point, typeset minus."""
    s = f'{abs(v):.{dp}f}'
    if s.startswith('0.'):
        s = s[1:]
    return ('$-$' + s) if round(v, dp) < 0 else s


def signed(v, dp=3):
    """As num(), with an explicit plus sign for positive values."""
    return num(v, dp) if round(v, dp) < 0 else '$+$' + num(v, dp)


def pexpr(p):
    """Whole p-value expression, AMA style: $p=.034$, $p<.001$."""
    return '$p<.001$' if p < 0.001 else f'$p={num(p)}$'


def integer(v):
    return f'{int(round(v)):,}'


NUMBER_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']


def count(v):
    """A count in running text: spelled out below 10, digits from 10 (style guide)."""
    n = int(round(v))
    return NUMBER_WORDS[n] if 0 <= n < 10 else f'{n:,}'


def put(name, text, source):
    assert name.isalpha(), f'macro name {name} must be letters only'
    assert name not in MACROS, f'macro {name} defined twice'
    MACROS[name] = (text, source)


# =============================================================================
# Readers
# =============================================================================
def newest_canonical_panel():
    found = sorted(OUT.glob('canonical_panel_*.xlsx'), key=lambda p: p.stat().st_mtime)
    return found[-1]


def fresh(name):
    """Path of an output file, after checking that it is not older than the
    canonical panel of the current run (minus FRESHNESS_TOLERANCE_S)."""
    path = OUT / name
    panel = newest_canonical_panel()
    if path.stat().st_mtime < panel.stat().st_mtime - FRESHNESS_TOLERANCE_S:
        raise SystemExit(f'{name} is older than {panel.name}: rerun the pipeline step that '
                         f'writes it before writing numbers.tex')
    return path


def p_from_t(t):
    """Two-sided normal p-value from a Newey-West t-statistic (as the FMB table)."""
    return 2 * stats.norm.sf(abs(t))


# =============================================================================
# Sections of numbers
# =============================================================================
def sample_numbers():
    funnel = pd.read_excel(fresh('sample_funnel.xlsx'), sheet_name='Funnel').set_index('Step')['Firms']

    def step(prefix):
        hits = [v for k, v in funnel.items() if k.startswith(prefix)]
        assert len(hits) == 1, f'funnel step {prefix!r}: {len(hits)} matches'
        return int(hits[0])

    screened = step('Firms in the screened revenue panel')
    in_sectors = step('Firms in the four GICS sectors')
    panel = step('Firms after excluding GameStop')
    sorts = step('Firms entering portfolio formation')
    src = 'sample_funnel.xlsx'
    put('valNScreened', integer(screened), src)
    put('valNSectorDropped', count(screened - in_sectors), src)
    put('valNSectorKept', integer(in_sectors), src)
    put('valNPanel', integer(panel), src)
    put('valNNoReturns', count(panel - sorts), src)
    put('valNSorts', integer(sorts), src)

    firms = pd.read_excel(fresh('sample_firms.xlsx'))
    tick_col = [c for c in firms.columns if c.lower() in ('ticker', 'firm')][0]
    sort_col = [c for c in firms.columns if 'sort' in c.lower()][0]
    in_sorts = firms[firms[sort_col].astype(bool)]
    assert len(in_sorts) == sorts, f'{len(in_sorts)} firms flagged in sorts, funnel says {sorts}'
    foreign = int((~in_sorts[tick_col].str.upper().str.endswith(' US EQUITY')).sum())
    put('valNForeign', count(foreign), 'sample_firms.xlsx')
    put('valNSortsUS', integer(sorts - foreign), 'sample_firms.xlsx')

    cp_path = newest_canonical_panel()
    cp = pd.read_excel(cp_path, sheet_name=0, engine='openpyxl')
    assert cp['FIRM'].nunique() == panel, f'{cp_path.name}: {cp["FIRM"].nunique()} firms, funnel says {panel}'
    src = cp_path.name
    per_firm = cp.groupby('FIRM').size()
    put('valNQuarters', integer(cp['DATE'].nunique()), src)
    put('valNFirmQuarters', integer(len(cp)), src)
    put('valNFirmQuartersRRR', integer(pd.to_numeric(cp['RRR_PCT'], errors='coerce').notna().sum()), src)
    put('valMedianQuartersPerFirm', integer(per_firm.median()), src)
    put('valMinQuartersPerFirm', integer(per_firm.min()), src)
    sectors = cp.drop_duplicates('FIRM')['SECTOR'].value_counts()
    for key, name in [('Consumer Discretionary', 'ConsDisc'), ('Industrials', 'Industrials'),
                      ('Consumer Staples', 'Staples'), ('Communication Services', 'CommSvc')]:
        put(f'valN{name}', integer(sectors[key]), src)
        put(f'valShare{name}', num(100 * sectors[key] / panel, 1), src)

    disp = ird.build(cp.set_index(['FIRM', 'DATE']))
    put('valDispersionMedian', integer((disp['p90'] - disp['p10']).median()),
        f'{src} via industry_rrr_dispersion.build')


def descriptive_numbers():
    d = pd.read_excel(fresh('pooled_descriptives.xlsx')).set_index('Variable')
    src = 'pooled_descriptives.xlsx'
    put('valRRRMean', num(d.loc['RRR (%)', 'Mean'], 2), src)
    put('valRRRMedian', num(d.loc['RRR (%)', 'Median'], 2), src)
    put('valRRRSD', num(d.loc['RRR (%)', 'SD'], 2), src)
    put('valRRRPNinety', num(d.loc['RRR (%)', 'P90'], 2), src)
    put('valNRRRObs', integer(d.loc['RRR (%)', 'N']), src)
    put('valBTMMean', num(d.loc['BTM', 'Mean']), src)
    put('valBTMMedian', num(d.loc['BTM', 'Median']), src)

    c = pd.read_excel(fresh('capm_vs_sp500.xlsx')).iloc[0]
    src = 'capm_vs_sp500.xlsx'
    put('valCapmBeta', num(c['beta']), src)
    put('valCapmBetaSE', num(c['beta_se']), src)
    put('valCapmBetaPOne', pexpr(p_from_t((c['beta'] - 1) / c['beta_se'])), src + ' (test of beta = 1)')
    put('valCapmRsq', num(c['r2']), src)
    put('valCapmN', integer(c['n_obs']), src)
    put('valCapmAlpha', num(100 * c['alpha_month']), src)
    put('valCapmAlphaSE', num(100 * c['alpha_se']), src)
    put('valCapmAlphaP', pexpr(c['alpha_p']), src)

    pc = pd.read_excel(fresh('portfolio_characteristics.xlsx')).set_index('Quartile')
    src = 'portfolio_characteristics.xlsx'
    for q, name in [('Q1', 'QOne'), ('Q4', 'QFour')]:
        put(f'valRev{name}', integer(pc.loc[q, 'Revenue (M)']), src)
        put(f'valMcap{name}', integer(pc.loc[q, 'Market Cap (M)']), src)
        put(f'valBTM{name}', num(pc.loc[q, 'BTM']), src)
        put(f'valROA{name}', num(pc.loc[q, 'Op. ROA (%)'], 2), src)
        put(f'valSoNR{name}', num(pc.loc[q, 'SoNR (%)'], 2), src)


def headline_numbers():
    r = pd.read_excel(fresh('headline_factor_regressions.xlsx'), index_col=0)
    src = 'headline_factor_regressions.xlsx (generate_tables_figures.py)'
    ls5 = r.loc['Q1-Q4_FF5']
    assert abs(100 * ls5['alpha'] - av2.EXPECTED_HEADLINE_FF5_ALPHA_PCT) < 1e-4, \
        f'headline {100 * ls5["alpha"]:.4f} differs from the guard {av2.EXPECTED_HEADLINE_FF5_ALPHA_PCT}'
    for key, name in [('Q1-Q4_FF3', 'LSThree'), ('Q1-Q4_FF5', 'LS'), ('Q1-Q4_FF3+Mom', 'LSCarhart'),
                      ('Q1_FF5', 'QOne'), ('Q2_FF5', 'QTwo'), ('Q3_FF5', 'QThree'), ('Q4_FF5', 'QFour')]:
        row = r.loc[key]
        put(f'valAlpha{name}', num(100 * row['alpha']), src)
        put(f'valSE{name}', num(100 * row['se_alpha']), src)
        put(f'valP{name}', pexpr(row['alpha_p']), src)
    put('valNMonths', integer(ls5['n_obs']), src)
    put('valSMBLoadLS', num(ls5['beta_SMB']), src)
    put('valCMALoadLS', num(ls5['beta_CMA']), src)
    put('valMomLoadLS', num(r.loc['Q1-Q4_FF3+Mom', 'beta_Mom']), src)
    for q, name in [('Q1', 'QOne'), ('Q2', 'QTwo'), ('Q3', 'QThree'), ('Q4', 'QFour')]:
        put(f'valBeta{name}', num(r.loc[f'{q}_FF5', 'beta_Mkt-RF'], 2), src)
        put(f'valSMB{name}', num(r.loc[f'{q}_FF5', 'beta_SMB']), src)


def risk_numbers():
    m = pd.read_excel(fresh('performance_metrics_summary.xlsx'), sheet_name='Metrics', index_col=0)
    src = 'performance_metrics_summary.xlsx'
    cols = [('Q1', 'QOne'), ('Q2', 'QTwo'), ('Q3', 'QThree'), ('Q4', 'QFour'), ('MKT', 'Bench')]
    for col, name in cols:
        put(f'valRet{name}', num(m.loc['Ann Return (%)', col], 2), src)
        put(f'valVol{name}', num(m.loc['Ann Vol (%)', col], 2), src)
        put(f'valSharpe{name}', num(m.loc['Sharpe Ratio', col]), src)
        put(f'valDrawdown{name}', num(m.loc['Max Drawdown (%)', col], 2), src)


def cross_section_numbers():
    f = pd.read_excel(fresh('fama_macbeth.xlsx'))
    f = f[f['Variable'] == 'RRR_PCT'].set_index('Spec')
    src = 'fama_macbeth.xlsx'
    for spec, name in [('(1) RRR only', 'One'), ('(2) RRR + Controls', 'Two'),
                       ('(3) RRR only, value-weighted', 'Three'),
                       ('(4) RRR + Controls, value-weighted', 'Four')]:
        put(f'valFMB{name}', num(f.loc[spec, 'Coefficient']), src)
        put(f'valPFMB{name}', pexpr(p_from_t(f.loc[spec, 't-stat'])), src)
    put('valFMBT', integer(f['T'].iloc[0]), src)

    h = pd.read_excel(fresh('h2_asymmetry_test.xlsx'), sheet_name='Results', index_col=0)
    src = 'h2_asymmetry_test.xlsx'
    for key, name in [('avg_FF3', 'Three'), ('avg_FF5', 'Five')]:
        put(f'valHTwo{name}', num(100 * h.loc[key, 'alpha']), src)
        put(f'valHTwo{name}SE', num(100 * h.loc[key, 'se']), src)
        put(f'valPHTwo{name}', pexpr(h.loc[key, 'p']), src)

    src = 'spanning_regression.xlsx'
    for sheet, stem, load_var in [('RRR_on_growth_FF3', 'SpanRRRThree', 'growth_LS'),
                                  ('RRR_on_growth_FF5', 'SpanRRRFive', 'growth_LS'),
                                  ('growth_on_RRR_FF3', 'SpanGrowthThree', 'RRR_LS'),
                                  ('growth_on_RRR_FF5', 'SpanGrowthFive', 'RRR_LS')]:
        s = pd.read_excel(fresh('spanning_regression.xlsx'), sheet_name=sheet).set_index('var')
        put(f'val{stem}', num(100 * s.loc['const', 'coef']), src)
        put(f'valP{stem}', pexpr(s.loc['const', 'p']), src)
        put(f'val{stem}Load', num(s.loc[load_var, 'coef'], 2), src)


ROBUSTNESS_STEMS = [
    ('(headline)', 'Base'), ('industry-adjusted', 'IndAdj'), ('Two-quarter', 'TwoQ'),
    ('Four-quarter \\RRR{}', 'FourQ'), ('Share of returning', 'SoRR'), ('TA4', 'TAFour'),
    ('RCB4', 'RCBFour'), ('moving average', 'MAFour'), ('Annual', 'Annual'),
    ('revenue growth', 'Growth'), ('First-half', 'FirstHalf'), ('Second-half', 'SecondHalf'),
    ('COVID', 'NoCovid'), ('Equal-weighted', 'EW'), ('Five-month', 'LagFive'),
    ('contributors removed', 'ContribTwo'),
]

# How each check is named in running text, for generated lists of checks.
CHECK_PHRASE = {
    'Base': 'the headline specification', 'IndAdj': 'the industry-adjusted \\RRR{}',
    'TwoQ': 'the two-quarter \\RRR{}', 'FourQ': 'the four-quarter \\RRR{}',
    'SoRR': 'the share of returning revenue', 'TAFour': 'TA4', 'RCBFour': 'RCB4',
    'MAFour': 'the four-quarter moving average', 'Annual': 'the annual \\RRR{}',
    'Growth': 'the revenue-growth sort', 'FirstHalf': 'the first-half subperiod',
    'SecondHalf': 'the second-half subperiod', 'NoCovid': 'the sample without COVID-19',
    'EW': 'equal weighting', 'LagFive': 'the five-month formation lag',
    'ContribTwo': 'the removal of the two largest contributors',
}


def prose_list(items):
    """'A', 'A and B', 'A, B, and C' (Oxford comma)."""
    items = list(items)
    if len(items) <= 1:
        return ''.join(items)
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


def robustness_numbers():
    rows = pd.read_excel(fresh('robustness_summary.xlsx'), sheet_name='Sheet1')
    src = 'robustness_summary.xlsx (asymmetry_across_variants.py)'
    stem_of = {}
    for _, row in rows.iterrows():
        hits = [stem for key, stem in ROBUSTNESS_STEMS if key in row['Check']]
        assert len(hits) == 1, f'robustness row {row["Check"]!r} matched {hits}'
        stem_of[row['Check']] = hits[0]
    assert len(stem_of) == len(ROBUSTNESS_STEMS), 'robustness rows and stems differ'
    rows['stem'] = rows['Check'].map(stem_of)
    rows = rows.set_index('stem')
    for stem, r in rows.iterrows():
        put(f'valAlpha{stem}', num(r['LS alpha (%/mo)']), src)
        put(f'valP{stem}', pexpr(r['LS p']), src)
        put(f'valAlphaQOne{stem}', num(r['Q1 alpha (%/mo)']), src)
        put(f'valPQOne{stem}', pexpr(r['Q1 p']), src)
        put(f'valAlphaQFour{stem}', num(r['Q4 alpha (%/mo)']), src)
        put(f'valPQFour{stem}', pexpr(r['Q4 p']), src)
        put(f'valN{stem}', integer(r['N months']), src)
        put(f'valTurn{stem}', num(r['Turnover (ann., %)'], 2), src)

    base = rows.loc['Base']
    others = rows.drop(index='Base')
    put('valNChecks', integer(len(rows)), src)
    put('valNChecksQFourNeg', count((rows['Q4 alpha (%/mo)'] < 0).sum()), src)
    put('valNChecksQFourBelowOne', count((rows['Q4 alpha (%/mo)'] <= -1).sum()), src)
    put('valNChecksQOneAboveBase', count((others['Q1 alpha (%/mo)'] > base['Q1 alpha (%/mo)']).sum()), src)
    put('valNChecksQOnePAboveThree', count((rows['Q1 p'] > 0.3).sum()), src)
    # Q4 alphas with p below .10, and the checks where p is .10 or above, as prose.
    put('valNChecksQFourPBelowTen', count((rows['Q4 p'] < 0.10).sum()), src)
    put('valChecksQFourPAboveTen',
        prose_list(CHECK_PHRASE[s] for s in rows.index[rows['Q4 p'] >= 0.10]), src)
    alt = rows[rows['Panel'] == 'A'].drop(index=['Base', 'Growth'])
    put('valNAltConstr', count(len(alt)), src)
    put('valNAltConstrPos', count((alt['LS alpha (%/mo)'] > 0).sum()), src)
    overlap = rows.loc[['TAFour', 'RCBFour', 'MAFour']]
    put('valPMinOverlap', num(overlap['LS p'].min()), src)
    put('valPMaxOverlap', num(overlap['LS p'].max()), src)
    short = base['N months'] - overlap['N months']
    put('valMonthsShorterMin', integer(short.min()), src)
    put('valMonthsShorterMax', integer(short.max()), src)
    put('valAnnualAlphaSharePct', integer(100 * rows.loc['Annual', 'LS alpha (%/mo)'] / base['LS alpha (%/mo)']), src)
    put('valAnnualTurnSharePct', integer(100 * rows.loc['Annual', 'Turnover (ann., %)'] / base['Turnover (ann., %)']), src)

    d = pd.read_excel(fresh('robustness_summary.xlsx'), sheet_name='signal_diagnostics')
    d['stem'] = [[stem for key, stem in ROBUSTNESS_STEMS if key in c][0] for c in d['Construction']]
    d = d.set_index('stem')
    src = 'robustness_summary.xlsx, sheet signal_diagnostics'
    for stem in ['Base', 'SoRR', 'TAFour', 'RCBFour', 'MAFour']:
        put(f'valStay{stem}', num(d.loc[stem, 'Stay-rate (%)'], 1), src)
        put(f'valAC{stem}', signed(d.loc[stem, 'Lag-1 autocorrelation']), src)


def other_check_numbers():
    diag = fresh('robustness_diagnostics.xlsx')
    pooled = pd.read_excel(diag, sheet_name='T6_DivYield_Pooled').set_index('RRR_Q_MAIN')
    firm = pd.read_excel(diag, sheet_name='T6_DivYield_FirmLevel').set_index('RRR_Q_MAIN')
    put('valDivGapPooled', num(100 * (pooled.loc['Q1', 'mean'] - pooled.loc['Q4', 'mean']), 2), diag.name)
    put('valDivGapFirm', num(100 * (firm.loc['Q1', 'mean'] - firm.loc['Q4', 'mean']), 2), diag.name)

    # Market-cap trim (Appendix table): the k largest names by formation weight.
    trim = fresh('concentration_trim_robustness.xlsx')
    t = pd.read_excel(trim, sheet_name='Summary').set_index('label')
    src_name = 'concentration_trim_robustness.xlsx, sheet Summary'
    for k, name in [(1, 'One'), (2, 'Two'), (3, 'Three')]:
        row = t.loc[f'Trim top-{k} from both legs']
        put(f'valAlphaCut{name}', num(row['ff5_alpha']), src_name)
        put(f'valPCut{name}', pexpr(row['ff5_p']), src_name)
    # weight_note reads "Q1 55.4 / Q4 20.2": average formation weight removed from each leg, in %.
    note = t.loc['Trim top-2 from both legs', 'weight_note'].replace('/', ' ').split()
    put('valTrimWeightQOne', note[note.index('Q1') + 1], src_name + ' (weight_note)')
    put('valTrimWeightQFour', note[note.index('Q4') + 1], src_name + ' (weight_note)')

    # Contributor trim (robustness row and Appendix table, added 2026-11-11): the k
    # firms that added most to the long leg and the k that subtracted most from the short leg.
    t = pd.read_excel(trim, sheet_name='Summary_contrib').set_index('label')
    src_name = 'concentration_trim_robustness.xlsx, sheet Summary_contrib'
    # (The k = 2 row's long-short, Q1 and Q4 alphas are also the robustness row
    # 'ContribTwo' above; these macros add k = 1 and k = 3, the weights and the names.)
    for k, name in [(1, 'One'), (2, 'Two'), (3, 'Three')]:
        row = t.loc[f'Top-{k} contributors from both legs']
        put(f'valAlphaContribTop{name}', num(row['ff5_alpha']), src_name)
        put(f'valPContribTop{name}', pexpr(row['ff5_p']), src_name)
    row = t.loc['Top-2 contributors from both legs']
    note = row['weight_note'].replace('/', ' ').split()
    put('valContribWeightQOne', note[note.index('Q1') + 1], src_name + ' (weight_note)')
    put('valContribWeightQFour', note[note.index('Q4') + 1], src_name + ' (weight_note)')
    removed = pd.read_excel(trim, sheet_name='Contrib_removed')
    src_name = 'concentration_trim_robustness.xlsx, sheet Contrib_removed'
    for leg, name in [('Q1', 'QOne'), ('Q4', 'QFour')]:
        top2 = removed[(removed['leg'] == leg) & (removed['rank'] <= 2)].sort_values('rank')
        put(f'valContribFirms{name}', prose_list(top2['name'].astype(str)), src_name)

    src_name = 'robustness_diagnostics.xlsx'
    for leg, name in [('Q1', 'QOne'), ('Q4', 'QFour')]:
        eff = pd.read_excel(diag, sheet_name=f'T1_EffN_{leg}')['EFFECTIVE_N']
        put(f'valEffN{name}', num(eff.mean(), 2), src_name)
        put(f'valEffN{name}Min', num(eff.min(), 2), src_name)

    ic = fresh('signal_ic_diagnostics.xlsx')
    c = pd.read_excel(ic, sheet_name='ls_month_concentration').set_index('Statistic')['Value']

    def stat(prefix):
        hits = [v for k, v in c.items() if k.startswith(prefix)]
        assert len(hits) == 1, f'{prefix!r}: {len(hits)} matches'
        return float(hits[0])

    src_name = ic.name
    put('valLSMean', num(stat('Mean monthly long-short return'), 2), src_name)
    put('valLSMedian', num(stat('Median monthly long-short return'), 2), src_name)
    put('valLSSD', num(stat('SD (%)'), 2), src_name)
    put('valSharePos', num(100 * stat('Share of months with long-short > 0'), 1), src_name)
    put('valPSharePos', pexpr(stat('Binomial p of that share')), src_name)
    put('valBestTenShare', integer(100 * stat('Best 10% of months')), src_name)
    put('valBestTenNormal', integer(100 * stat('Same share expected under a normal')), src_name)
    drops = pd.read_excel(ic, sheet_name='ls_drop_best_months').set_index('Sample')
    for key, name in [('Drop best 5 month(s)', 'DropBestFive'), ('Drop best 5 and worst 5', 'DropFiveFive')]:
        put(f'valAlpha{name}', num(drops.loc[key, 'FF5 alpha (%/mo)']), src_name)
        put(f'valP{name}', pexpr(drops.loc[key, 'p']), src_name)

    ds = pd.read_excel(fresh('double_sort_size.xlsx'), sheet_name='size_long_short')
    ds = ds[ds['Weighting'] == 'VW'].set_index('Portfolio')
    src_name = 'double_sort_size.xlsx'
    for key, name in [('RRR T1-T3 within Size T1 (large)', 'Large'), ('RRR T1-T3 within Size T2', 'Middle'),
                      ('RRR T1-T3 within Size T3 (small)', 'Small'),
                      ('Size-neutral RRR T1-T3 (average)', 'Neutral'),
                      ('Unconditional RRR tercile T1-T3 (same sample)', 'Uncond')]:
        put(f'valDS{name}', num(ds.loc[key, 'FF5 alpha (%/mo)']), src_name)
        put(f'valPDS{name}', pexpr(ds.loc[key, 'p']), src_name)


def example_numbers():
    """The hypothetical firm of Tables 3 and 4 (signal_example_table.py): the cells
    the Methodology text cites, so text and tables cannot diverge."""
    ex = json.loads(fresh('signal_example.json').read_text(encoding='utf-8'))
    src_name = 'signal_example.json (signal_example_table.py)'
    quarters = ex['quarters']
    cited = [('rrr', 'RRR', 1), ('rrr', 'RRR', 3), ('rrr', 'RRR', 4),
             ('tq', 'TwoQ', 5), ('fq', 'FourQ', 7),
             ('rcb4', 'RCBFour', 1), ('rcb4', 'RCBFour', 4), ('rcb4', 'RCBFour', 7),
             ('sorr', 'SoRR', 3), ('ta4', 'TAFour', 4),
             ('ma4', 'MAFour', 4), ('ma4', 'MAFour', 5), ('ma4', 'MAFour', 8)]
    for key, stem, q in cited:
        value = ex['signals'][key][quarters.index(q)]
        put(f'valEx{stem}Q{NUMBER_WORDS[q].capitalize()}', num(value, 1), src_name)
    put('valExSpikeQuarter', count(ex['spike_quarter']), src_name)
    put('valExLastQuarter', count(quarters[-1]), src_name)


# =============================================================================
# Main
# =============================================================================
def main():
    sample_numbers()
    descriptive_numbers()
    headline_numbers()
    risk_numbers()
    cross_section_numbers()
    robustness_numbers()
    other_check_numbers()
    example_numbers()

    lines = ['% numbers.tex -- written by Code/RRR-FI-IM/numbers_tex.py. Do not edit by hand.',
             '% Every result number cited in the prose; rerun the pipeline and numbers_tex.py to refresh.']
    for name, (text, source) in MACROS.items():
        lines.append(f'\\newcommand{{\\{name}}}{{{text}}}% {source}')
    TEX_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    width = max(len(n) for n in MACROS)
    for name, (text, source) in MACROS.items():
        print(f'  \\{name:<{width}}  {text:<16}  {source}')
    print(f'\n  {len(MACROS)} macros written to {TEX_PATH}')


if __name__ == '__main__':
    main()
