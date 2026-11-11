"""signal_ic_diagnostics.py -- RRR Financial Implications
=========================================================

Diagnostics requested by the author on 2026-10-12. They are for the author's
reading of the signal, not paper tables: nothing in Paper_LaTeX/ is written.

  Q1    Correlation of the RRR signal with the Fama-French five factors.
        Panel A: monthly time-series correlation of the headline long-short
                 (and its two legs) with Mkt-RF, SMB, HML, RMW, CMA (+ Mom),
                 next to the FF5 loadings of the same series.
        Panel B: per-quarter Spearman correlation of firm RRR with the
                 characteristic behind each factor (market beta, size,
                 book-to-market, operating profitability, investment).
  Q2/Q3 Monthly information coefficient IC_m = Corr(RRR_i, R_i,m) on exactly
        the cross-sections of the monthly Fama-MacBeth (Pearson, Spearman and
        a cap-weighted rank IC), with mean, SD, ICIR, Newey-West t and hit
        rate, plus a non-overlapping quarterly check on 3-month holding returns.
  Q5    Consistency versus outliers: winsorized Pearson ICs, IC by calendar
        year, IC within size thirds (added 2026-10-19), concentration of the
        long-short return in its best months, FF5 alpha after dropping the best
        months, and a leave-one-firm-out jackknife of the long-short alpha.

Data and timing come from robustness_diagnostics.load_base_data(), the
raw-signal chokepoint, so every number is on the headline sample (signal at
quarter-end, formed two months later, held three months, value-weighted on
formation-date market cap). Two guards stop the run if the sample drifts: the
headline FF5 long-short alpha must reproduce 2.2511%/mo on 107 months, and the
univariate monthly Fama-MacBeth slope must reproduce output/fama_macbeth.xlsx.

No significance stars; exact p-values throughout.

Outputs
    output/signal_ic_diagnostics.xlsx
    output/ic_timeseries.png, output/ic_timeseries.pdf

Run (from Code/RRR-FI-IM):
    ..\\venv\\Scripts\\python.exe signal_ic_diagnostics.py
"""

# =============================================================================
# Cell 1: imports
# =============================================================================
import contextlib
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402

warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import analysis_v2 as av2                 # noqa: E402
import robustness_diagnostics as rd       # noqa: E402

# =============================================================================
# Cell 2: constants
# =============================================================================
OUTPUT_DIR = av2.OUTPUT_DIR
XLSX_PATH = os.path.join(OUTPUT_DIR, 'signal_ic_diagnostics.xlsx')
FIG_BASE = os.path.join(OUTPUT_DIR, 'ic_timeseries')
FM_XLSX_PATH = os.path.join(OUTPUT_DIR, 'fama_macbeth.xlsx')

FF5 = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
EXPECTED_FF5_ALPHA_PCT = av2.EXPECTED_HEADLINE_FF5_ALPHA_PCT   # headline, defined once in analysis_v2
EXPECTED_T_MONTHS = 107             # monthly Fama-MacBeth cross-sections (D18)
MIN_CROSS_SECTION = 10              # same floor as analysis_v2.run_fama_macbeth
BETA_WINDOW, BETA_MIN_OBS = 36, 24  # rolling market beta for the characteristic panel
WINSOR_LEVELS = [(0.01, 0.99), (0.05, 0.95)]
DROP_BEST_K = [1, 3, 5]
ROLLING_IC_MONTHS = 12

# Categorical slots 1 and 2 of the dataviz reference palette, validated for
# colour-vision deficiency on a white surface (validate_palette.js, all pass).
COL_SPEARMAN = '#2a78d6'
COL_SPEARMAN_DARK = '#184f95'       # darker step of the same ramp: rolling mean
COL_PEARSON = '#eb6834'
COL_MUTED = '#898781'
COL_GRID = '#e1e0d9'
COL_INK = '#2b2b2b'

IC_LABELS = {
    'IC_PEARSON': 'Pearson IC',
    'IC_SPEARMAN': 'Spearman rank IC',
    'IC_CAPW_RANK': 'Cap-weighted rank IC (non-standard)',
    'IC_PEARSON_W01': 'Pearson IC, RRR winsorized 1/99',
    'IC_PEARSON_W05': 'Pearson IC, RRR winsorized 5/95',
    'IC_PEARSON_RETW01': 'Pearson IC, returns winsorized 1/99',
    'IC_PEARSON_BOTHW01': 'Pearson IC, RRR and returns winsorized 1/99',
}


# =============================================================================
# Cell 3: helpers
# =============================================================================
@contextlib.contextmanager
def quiet():
    """Silence the console output of analysis_v2 helpers inside loops."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def to_month_end(index):
    return pd.to_datetime(index).to_period('M').to_timestamp('M')


def nw_mean_test(x, max_lag=None):
    """Mean of a time series with a Bartlett Newey-West standard error.

    Same formula and lag rule as analysis_v2.run_fama_macbeth, floor(4(T/100)^(2/9)),
    so a mean IC and a Fama-MacBeth slope are tested identically. The p-value
    uses the normal reference distribution, as the paper's Fama-MacBeth table does.
    """
    x = pd.Series(x).dropna().astype(float)
    T = len(x)
    mu = x.mean()
    if max_lag is None:
        max_lag = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    d = x - mu
    gamma = (d ** 2).mean()
    for j in range(1, max_lag + 1):
        gamma += 2 * (1 - j / (max_lag + 1)) * (d.iloc[j:].values * d.iloc[:-j].values).mean()
    se = np.sqrt(gamma / T)
    t = mu / se if se > 0 else np.nan
    p = float(2 * sps.norm.sf(abs(t))) if np.isfinite(t) else np.nan
    return {'mean': mu, 'se_nw': se, 't_nw': t, 'p_nw': p, 'nw_lag': max_lag, 'T': T}


def binom_p(k, n):
    """Exact two-sided binomial p-value against a 50% hit rate."""
    try:
        return float(sps.binomtest(int(k), int(n), 0.5).pvalue)
    except AttributeError:                       # scipy < 1.7
        return float(sps.binom_test(int(k), int(n), 0.5))


def ff5_alpha(y, ff, excess=False):
    """FF5 regression of one monthly series, aligned exactly as
    analysis_v2.run_factor_regressions aligns it (month-end index, inner join
    with the factors, Newey-West lag from the series length). A long-short is
    already zero-cost, so RF is subtracted only when excess=True (a leg)."""
    s = pd.Series(y).copy()
    s.index = to_month_end(s.index)
    comb = s.to_frame('y').join(ff, how='inner')
    yy = comb['y'] - comb['RF'] if excess else comb['y']
    d = pd.concat([yy.rename('y'), comb[FF5]], axis=1).dropna()
    m = av2.newey_west_ols(d['y'], d[FF5])
    out = {'alpha': float(m.params['const']), 'se': float(m.bse['const']),
           't': float(m.tvalues['const']), 'p': float(m.pvalues['const']),
           'r2': float(m.rsquared), 'n': int(m.nobs)}
    for f in FF5:
        out[f'beta_{f}'] = float(m.params[f])
        out[f'se_{f}'] = float(m.bse[f])
        out[f'p_{f}'] = float(m.pvalues[f])
    return out


def wcorr(x, y, w):
    """Weighted Pearson correlation."""
    w = np.asarray(w, float) / np.sum(w)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx, vy = np.sum(w * (x - mx) ** 2), np.sum(w * (y - my) ** 2)
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


def summarize_ic(ic, periods_per_year, label):
    """Mean, SD, ICIR, Newey-West test, persistence and hit rate of one IC series."""
    ic = pd.Series(ic).dropna()
    nw = nw_mean_test(ic)
    sd = ic.std(ddof=1)
    icir = nw['mean'] / sd if sd > 0 else np.nan
    k = int((ic > 0).sum())
    return {
        'IC type': label,
        'T': nw['T'],
        'Mean IC': nw['mean'],
        'Median IC': ic.median(),
        'SD of IC': sd,
        'ICIR (per period)': icir,
        'ICIR annualized': icir * np.sqrt(periods_per_year),
        't naive (ICIR x sqrt T)': icir * np.sqrt(nw['T']),
        'SE Newey-West': nw['se_nw'],
        't Newey-West': nw['t_nw'],
        'p Newey-West': nw['p_nw'],
        'NW lag': nw['nw_lag'],
        'AR(1) of IC': ic.autocorr(1),
        'Hit rate (IC > 0)': k / len(ic),
        'Hit-rate binomial p': binom_p(k, len(ic)),
        'Min IC': ic.min(),
        'Max IC': ic.max(),
    }


def _hdr(title):
    print('\n' + '=' * 78)
    print('  ' + title)
    print('=' * 78)


# =============================================================================
# Cell 4: guards (the sample must be the headline sample)
# =============================================================================
def guard_headline(port, reg, ff):
    r = reg['Q1-Q4_FF5']
    got = round(r['alpha'] * 100, 4)
    assert got == EXPECTED_FF5_ALPHA_PCT and int(r['n_obs']) == EXPECTED_T_MONTHS, (
        f"Headline FF5 long-short is {got}%/mo on {int(r['n_obs'])} months, expected "
        f"{EXPECTED_FF5_ALPHA_PCT}% on {EXPECTED_T_MONTHS}. The sample has drifted; stop.")
    mine = ff5_alpha(port['Q1-Q4'], ff)
    assert abs(mine['alpha'] - r['alpha']) < 1e-12, 'ff5_alpha() does not match run_factor_regressions'
    print(f"  [GUARD] headline FF5 long-short {got:.4f}%/mo, N={int(r['n_obs'])}: reproduced")


def guard_fama_macbeth(mdf):
    """The univariate monthly FM slope on this cross-section must equal the
    paper's (output/fama_macbeth.xlsx), which proves the IC is computed on the
    same 107 cross-sections with the same Newey-West rule."""
    slopes = []
    for _, g in mdf.groupby('MONTH_P'):
        x, y = g['RRR_PCT'], g['EXCESS_RET_M_PCT']
        slopes.append(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    nw = nw_mean_test(pd.Series(slopes))
    fm = pd.read_excel(FM_XLSX_PATH)
    row = fm[(fm['Spec'] == '(1) RRR only') & (fm['Variable'] == 'RRR_PCT')].iloc[0]
    assert int(row['T']) == nw['T'] == EXPECTED_T_MONTHS, (
        f"FM T mismatch: file {int(row['T'])}, here {nw['T']}")
    assert abs(row['Coefficient'] - nw['mean']) < 1e-8 and abs(row['t-stat'] - nw['t_nw']) < 1e-6, (
        f"FM slope mismatch: file {row['Coefficient']:.6f} (t={row['t-stat']:.4f}), "
        f"here {nw['mean']:.6f} (t={nw['t_nw']:.4f})")
    print(f"  [GUARD] univariate monthly FM slope {nw['mean']:.4f} (t={nw['t_nw']:.4f}, "
          f"T={nw['T']}): matches fama_macbeth.xlsx")


# =============================================================================
# Cell 5: Q1 -- correlation with the five factors
# =============================================================================
def q1_factor_corr_ts(port, ff):
    _hdr('Q1 PANEL A  Time-series correlation with the factors (monthly)')
    p = port[['Q1', 'Q4', 'Q1-Q4']].copy()
    p.index = to_month_end(p.index)
    d = p.join(ff, how='inner')
    d = pd.DataFrame({
        'Q1-Q4 long-short': d['Q1-Q4'],
        'Q1 leg (excess)': d['Q1'] - d['RF'],
        'Q4 leg (excess)': d['Q4'] - d['RF'],
        **{f: d[f] for f in FF5 + ['Mom']},
    }).dropna()
    assert len(d) == EXPECTED_T_MONTHS, f'{len(d)} months in the factor sample'

    rows = []
    for s in ['Q1-Q4 long-short', 'Q1 leg (excess)', 'Q4 leg (excess)']:
        for f in FF5 + ['Mom']:
            r, pv = sps.pearsonr(d[s], d[f])
            rows.append({'Series': s, 'Factor': f, 'Pearson r': r, 'p': pv, 'N months': len(d)})
    tidy = pd.DataFrame(rows)
    grid = tidy.pivot(index='Series', columns='Factor', values='Pearson r')[FF5 + ['Mom']]
    print(grid.round(4).to_string())

    load_rows = []
    for s, col, excess in [('Q1-Q4 long-short', 'Q1-Q4', False),
                           ('Q1 leg (excess)', 'Q1', True),
                           ('Q4 leg (excess)', 'Q4', True)]:
        r = ff5_alpha(port[col], ff, excess=excess)
        load_rows.append({'Series': s, 'Term': 'alpha (%/mo)', 'Coefficient': r['alpha'] * 100,
                          'SE': r['se'] * 100, 'p': r['p'], 'R2': r['r2'], 'N months': r['n']})
        for f in FF5:
            load_rows.append({'Series': s, 'Term': f, 'Coefficient': r[f'beta_{f}'],
                              'SE': r[f'se_{f}'], 'p': r[f'p_{f}'], 'R2': r['r2'], 'N months': r['n']})
    loadings = pd.DataFrame(load_rows)
    ls = loadings[loadings['Series'] == 'Q1-Q4 long-short']
    print('\n  FF5 loadings of the long-short (multivariate, for contrast):')
    for _, r in ls.iterrows():
        print(f"    {r['Term']:<14} {r['Coefficient']:>9.4f}  ({r['SE']:.4f})  [{r['p']:.3f}]")
    print(f"    R2 = {ls['R2'].iloc[0]:.4f}")
    return tidy, d.corr(), loadings


def rolling_market_beta(returns, ff):
    """Rolling 36-month (at least 24) market beta of each firm's monthly excess
    return, using months up to and including the month it is dated."""
    r = returns[['FIRM', 'Date', 'RET_SIMPLE']].copy()
    r['M'] = pd.to_datetime(r['Date']).dt.to_period('M')
    r = r.drop_duplicates(['FIRM', 'M'])
    wide = r.pivot(index='M', columns='FIRM', values='RET_SIMPLE').sort_index()
    ffm = ff.copy()
    ffm.index = ffm.index.to_period('M')
    common = wide.index.intersection(ffm.index)
    wide, ffm = wide.loc[common], ffm.loc[common]
    ex = wide.sub(ffm['RF'], axis=0)
    out = {}
    for firm in ex.columns:
        y = ex[firm]
        x = ffm['Mkt-RF'].where(y.notna())
        cov = y.rolling(BETA_WINDOW, min_periods=BETA_MIN_OBS).cov(x)
        var = x.rolling(BETA_WINDOW, min_periods=BETA_MIN_OBS).var()
        out[firm] = cov / var
    beta = pd.DataFrame(out)
    beta.index.name = 'M'
    return beta.stack().rename('BETA').reset_index().rename(columns={'level_1': 'FIRM'})


def q1_characteristic_corr(df_filtered, returns, ff, panel):
    _hdr('Q1 PANEL B  Cross-sectional correlation of RRR with factor characteristics')
    d = df_filtered.reset_index()[['FIRM', 'DATE', 'RRR_PCT', 'SIZE', 'BTM', 'IS_OPER_INC',
                                   'TOTAL_EQUITY', 'BS_TOT_ASSET']].copy()
    d['FIRM'] = d['FIRM'].str.upper()
    for c in ['RRR_PCT', 'SIZE', 'BTM', 'IS_OPER_INC', 'TOTAL_EQUITY', 'BS_TOT_ASSET']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['QP'] = pd.to_datetime(d['DATE']).dt.to_period('Q')
    d['M'] = pd.to_datetime(d['DATE']).dt.to_period('M')

    # Same firms and signal quarters as the sorts.
    sort_firms = set(panel['FIRM'].unique())
    sort_q = set(pd.to_datetime(panel['QUARTER']).dt.to_period('Q').unique())
    d = d[d['FIRM'].isin(sort_firms) & d['QP'].isin(sort_q)].copy()

    # Operating profitability (RMW): operating income / book equity, positive equity only.
    d['OP_BE'] = np.where(d['TOTAL_EQUITY'] > 0, d['IS_OPER_INC'] / d['TOTAL_EQUITY'], np.nan)
    # Investment (CMA): four-quarter total-asset growth, consecutive periods enforced by merge.
    lag = df_filtered.reset_index()[['FIRM', 'DATE', 'BS_TOT_ASSET']].copy()
    lag['FIRM'] = lag['FIRM'].str.upper()
    lag['QP'] = pd.to_datetime(lag['DATE']).dt.to_period('Q') + 4
    lag = lag.rename(columns={'BS_TOT_ASSET': 'ASSET_LAG4'})[['FIRM', 'QP', 'ASSET_LAG4']]
    lag['ASSET_LAG4'] = pd.to_numeric(lag['ASSET_LAG4'], errors='coerce')
    d = d.merge(lag, on=['FIRM', 'QP'], how='left')
    d['ASSET_GROWTH'] = np.where(d['ASSET_LAG4'] > 0, d['BS_TOT_ASSET'] / d['ASSET_LAG4'] - 1, np.nan)
    # Market beta (Mkt-RF) dated at the quarter-end month.
    d = d.merge(rolling_market_beta(returns, ff), on=['FIRM', 'M'], how='left')

    chars = [('BETA', 'Market beta, 36-month rolling', 'Mkt-RF'),
             ('SIZE', 'Size, log market cap', 'SMB (small minus big)'),
             ('BTM', 'Book-to-market', 'HML (high minus low)'),
             ('OP_BE', 'Operating profitability, op. income / book equity', 'RMW (robust minus weak)'),
             ('ASSET_GROWTH', 'Investment, 4-quarter asset growth', 'CMA (conservative minus aggressive)')]
    per_q, rows = [], []
    for col, label, factor in chars:
        rhos = []
        for qp, g in d.groupby('QP'):
            sub = g[['RRR_PCT', col]].dropna()
            if len(sub) < MIN_CROSS_SECTION:
                continue
            rho = sub['RRR_PCT'].corr(sub[col], method='spearman')
            rhos.append(rho)
            per_q.append({'Quarter': str(qp), 'Characteristic': label, 'Spearman rho': rho, 'N': len(sub)})
        s = pd.Series(rhos)
        nw = nw_mean_test(s)
        sign = np.sign(nw['mean'])
        n_q = pd.DataFrame(per_q)
        avg_n = n_q.loc[n_q['Characteristic'] == label, 'N'].mean()
        rows.append({'Characteristic': label, 'Factor it underlies': factor,
                     'Mean Spearman rho': nw['mean'], 'Median rho': s.median(),
                     'Share of quarters with same sign as mean': float((np.sign(s) == sign).mean()),
                     't Newey-West': nw['t_nw'], 'p Newey-West': nw['p_nw'],
                     'T quarters': nw['T'], 'Avg N firms': avg_n})
        print(f"  {label:<52} mean rho {nw['mean']:>8.4f}  [p={nw['p_nw']:.3f}]  "
              f"T={nw['T']}  avg N={avg_n:.0f}")
    return pd.DataFrame(rows), pd.DataFrame(per_q)


# =============================================================================
# Cell 6: Q2/Q3 -- information coefficients
# =============================================================================
def build_ic_cross_sections(panel, ff):
    """One row per firm and holding month, as analysis_v2.run_fama_macbeth builds it."""
    df = panel.copy()
    rf = ff['RF'].copy()
    rf.index = pd.to_datetime(rf.index).to_period('M')
    df['MONTH_P'] = pd.to_datetime(df['Date']).dt.to_period('M')
    df['RF_M'] = df['MONTH_P'].map(rf)
    for c in ['RET_SIMPLE', 'RRR_PCT', 'MCAP_FORM']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    mdf = df.drop_duplicates(['FIRM', 'MONTH_P']).copy()
    mdf['EXCESS_RET_M_PCT'] = (mdf['RET_SIMPLE'] - mdf['RF_M']) * 100.0
    mdf = mdf.dropna(subset=['EXCESS_RET_M_PCT', 'RRR_PCT'])
    n = mdf.groupby('MONTH_P')['FIRM'].transform('size')
    return mdf[n >= MIN_CROSS_SECTION].copy()


def monthly_ic(mdf):
    rows = []
    for mon, g in mdf.groupby('MONTH_P'):
        x, y = g['RRR_PCT'], g['RET_SIMPLE']
        rec = {'MONTH': mon.to_timestamp('M'), 'N': len(g),
               'IC_PEARSON': x.corr(y),
               'IC_SPEARMAN': x.corr(y, method='spearman')}
        for lo, hi in WINSOR_LEVELS:
            xl, xh = x.quantile([lo, hi])
            rec[f'IC_PEARSON_W{int(round(lo * 100)):02d}'] = x.clip(xl, xh).corr(y)
        yl, yh = y.quantile([0.01, 0.99])
        xl, xh = x.quantile([0.01, 0.99])
        rec['IC_PEARSON_RETW01'] = x.corr(y.clip(yl, yh))
        rec['IC_PEARSON_BOTHW01'] = x.clip(xl, xh).corr(y.clip(yl, yh))
        gw = g.dropna(subset=['MCAP_FORM'])
        rec['IC_CAPW_RANK'] = wcorr(gw['RRR_PCT'].rank(pct=True), gw['RET_SIMPLE'].rank(pct=True),
                                    gw['MCAP_FORM'])
        rows.append(rec)
    ic = pd.DataFrame(rows).set_index('MONTH').sort_index()
    for c in ['IC_PEARSON', 'IC_SPEARMAN']:
        ic[f'{c}_ROLL{ROLLING_IC_MONTHS}'] = ic[c].rolling(ROLLING_IC_MONTHS).mean()
        ic[f'{c}_CUM'] = ic[c].cumsum()
    return ic


def quarterly_ic(panel, ff):
    """One IC per signal quarter on the compounded 3-month holding return; the
    quarters do not overlap. Only quarters whose whole holding window lies in
    the factor sample are kept, so the window matches the 107 monthly ICs."""
    hw = rd.compound_holding_return(panel, ff, ['RRR_PCT'])
    last_m = pd.to_datetime(ff.dropna(subset=['RF']).index.max()).to_period('M')
    end_m = (panel.assign(M=pd.to_datetime(panel['Date']).dt.to_period('M'))
             .groupby('QUARTER')['M'].max())
    keep = set(end_m[end_m <= last_m].index)
    hw = hw[hw['QUARTER'].isin(keep)]
    rows = []
    for q, g in hw.groupby('QUARTER'):
        sub = g[['RRR_PCT', 'EXCESS_RET_HW']].dropna()
        if len(sub) < MIN_CROSS_SECTION:
            continue
        rows.append({'QUARTER': pd.Timestamp(q), 'N': len(sub),
                     'IC_PEARSON': sub['RRR_PCT'].corr(sub['EXCESS_RET_HW']),
                     'IC_SPEARMAN': sub['RRR_PCT'].corr(sub['EXCESS_RET_HW'], method='spearman')})
    return pd.DataFrame(rows).set_index('QUARTER').sort_index()


def q23_ic(panel, ff):
    _hdr('Q2/Q3  Monthly information coefficients')
    mdf = build_ic_cross_sections(panel, ff)
    guard_fama_macbeth(mdf)
    ic = monthly_ic(mdf)
    assert len(ic) == EXPECTED_T_MONTHS, f'{len(ic)} monthly ICs, expected {EXPECTED_T_MONTHS}'

    summ = []
    for col, label in IC_LABELS.items():
        rec = summarize_ic(ic[col], 12, label)
        rec['Frequency'] = 'Monthly'
        rec['Avg N firms'] = ic['N'].mean()
        summ.append(rec)
    icq = quarterly_ic(panel, ff)
    for col in ['IC_PEARSON', 'IC_SPEARMAN']:
        rec = summarize_ic(icq[col], 4, IC_LABELS[col])
        rec['Frequency'] = 'Quarterly (3-month holding return)'
        rec['Avg N firms'] = icq['N'].mean()
        summ.append(rec)
    summ = pd.DataFrame(summ)
    lead = ['Frequency', 'IC type']
    summ = summ[lead + [c for c in summ.columns if c not in lead]]

    show = ['Frequency', 'IC type', 'T', 'Mean IC', 'SD of IC', 'ICIR (per period)',
            'ICIR annualized', 't Newey-West', 'p Newey-West', 'Hit rate (IC > 0)', 'Hit-rate binomial p']
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(summ[show].round(4).to_string(index=False))
    return ic, icq, summ


# =============================================================================
# Cell 7: Q5 -- consistency versus outliers
# =============================================================================
def q5_ic_by_year(ic):
    g = ic.groupby(ic.index.year)
    out = pd.DataFrame({
        'Months': g.size(),
        'Mean Pearson IC': g['IC_PEARSON'].mean(),
        'Mean Spearman IC': g['IC_SPEARMAN'].mean(),
        'Spearman hit rate': g['IC_SPEARMAN'].apply(lambda s: (s > 0).mean()),
        'Mean cap-weighted rank IC': g['IC_CAPW_RANK'].mean(),
    })
    out.index.name = 'Year'
    _hdr('Q5  IC by calendar year')
    print(out.round(4).to_string())
    return out.reset_index()


def q5_month_concentration(port, ff):
    _hdr('Q5  Is the long-short return concentrated in a few months?')
    ls = port['Q1-Q4'].copy()
    ls.index = to_month_end(ls.index)
    ls = ls.reindex(ls.index.intersection(ff.dropna(subset=FF5).index)).dropna()
    T = len(ls)
    k_pos = int((ls > 0).sum())
    n_top = int(np.ceil(0.10 * T))
    top = ls.nlargest(n_top)
    # Benchmark for the best-months share: what a normal series with the same
    # mean and SD would give, E[X | X > q] = mu + sigma * phi(z_q) / (1 - q).
    # A share near this benchmark is what noise around a constant mean produces.
    frac = n_top / T
    z = sps.norm.ppf(1 - frac)
    top_mean_normal = ls.mean() + ls.std() * sps.norm.pdf(z) / frac
    share_normal = frac * top_mean_normal / ls.mean()
    stats = pd.DataFrame([
        ('Months', T),
        ('Mean monthly long-short return (%)', ls.mean() * 100),
        ('Median monthly long-short return (%)', ls.median() * 100),
        ('SD (%)', ls.std() * 100),
        ('Skewness', sps.skew(ls)),
        ('Share of months with long-short > 0', k_pos / T),
        ('Binomial p of that share against 50%', binom_p(k_pos, T)),
        (f'Best 10% of months ({n_top}): share of summed long-short return', top.sum() / ls.sum()),
        ('Same share expected under a normal distribution with this mean and SD', share_normal),
        ('Mean long-short return without the best 10% of months (%)', ls.drop(top.index).mean() * 100),
    ], columns=['Statistic', 'Value'])
    print(stats.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    rows = []
    full = ff5_alpha(ls, ff)
    rows.append({'Sample': 'Full sample', **{k: full[k] for k in ['alpha', 'se', 'p', 'n']}})
    for k in DROP_BEST_K:
        r = ff5_alpha(ls.drop(ls.nlargest(k).index), ff)
        rows.append({'Sample': f'Drop best {k} month(s)', **{c: r[c] for c in ['alpha', 'se', 'p', 'n']}})
    for k in DROP_BEST_K:
        drop = ls.nlargest(k).index.union(ls.nsmallest(k).index)
        r = ff5_alpha(ls.drop(drop), ff)
        rows.append({'Sample': f'Drop best {k} and worst {k}', **{c: r[c] for c in ['alpha', 'se', 'p', 'n']}})
    drops = pd.DataFrame(rows)
    drops['alpha'] *= 100
    drops['se'] *= 100
    drops = drops.rename(columns={'alpha': 'FF5 alpha (%/mo)', 'se': 'SE', 'n': 'N months'})
    print('\n' + drops.round(4).to_string(index=False))
    stats['Value'] = stats['Value'].astype(float)
    return stats, drops, top


def q5_jackknife(panel, ff, market_ret):
    _hdr('Q5  Leave-one-firm-out jackknife of the FF5 long-short alpha')
    firms = sorted(panel['FIRM'].unique())
    base = panel.copy()
    q1_months = base[base['RRR_Q_MAIN'] == 'Q1'].groupby('FIRM').size()
    q4_months = base[base['RRR_Q_MAIN'] == 'Q4'].groupby('FIRM').size()
    rows = []
    for i, f in enumerate(firms, 1):
        p = base[base['FIRM'] != f].copy()
        p['Q_JK'] = p.groupby('QUARTER')['RRR_PCT'].transform(av2.safe_quartile)
        with quiet():
            port = av2.build_portfolio_returns(p, 'Q_JK', market_ret)
        r = ff5_alpha(port['Q1-Q4'], ff)
        rows.append({'Firm removed': f, 'FF5 alpha (%/mo)': r['alpha'] * 100, 'SE': r['se'] * 100,
                     'p': r['p'], 'N months': r['n'],
                     'Holding months in Q1 (full sort)': int(q1_months.get(f, 0)),
                     'Holding months in Q4 (full sort)': int(q4_months.get(f, 0))})
        if i % 25 == 0:
            print(f'    {i}/{len(firms)} firms done')
    jk = pd.DataFrame(rows)
    jk['Change vs full sample (pp)'] = jk['FF5 alpha (%/mo)'] - EXPECTED_FF5_ALPHA_PCT
    jk = jk.sort_values('Change vs full sample (pp)', key=np.abs, ascending=False).reset_index(drop=True)
    a = jk['FF5 alpha (%/mo)']
    print(f'  {len(jk)} re-sorts. FF5 alpha min {a.min():.4f}, median {a.median():.4f}, '
          f'max {a.max():.4f}; largest p {jk["p"].max():.3f}')
    print('  Five most influential firms:')
    print(jk.head(5)[['Firm removed', 'FF5 alpha (%/mo)', 'p', 'Change vs full sample (pp)',
                      'Holding months in Q1 (full sort)', 'Holding months in Q4 (full sort)']]
          .round(4).to_string(index=False))
    return jk


SIZE_THIRDS = {'T1': 'Large third', 'T2': 'Middle third', 'T3': 'Small third'}


def assign_size_tertile(panel):
    """Formation-quarter size tertiles on MCAP_FORM (T1 = largest), on the same
    firm set and breakpoints as the RRR x Size double sort
    (double_sort_size_growth.assign_tertiles with MCAP_FORM)."""
    p = panel.copy()
    for c in ['MCAP_FORM', 'RRR_PCT']:
        p[c] = pd.to_numeric(p[c], errors='coerce')
    p = p.dropna(subset=['MCAP_FORM', 'RRR_PCT']).copy()
    p['SIZE_T'] = p.groupby('QUARTER')['MCAP_FORM'].transform(av2.safe_tercile).astype(object)
    return p


def q5_ic_by_size(panel, ff):
    """Monthly Pearson and Spearman IC computed separately inside each size third.
    Added 2026-10-19: the whole-sample IC averages a positive relation among large
    firms with a negative one among small firms, which this table shows directly."""
    _hdr('Q5  Monthly IC within size thirds')
    mdf = build_ic_cross_sections(assign_size_tertile(panel), ff)
    rows = []
    for t, lab in SIZE_THIRDS.items():
        recs = []
        for _, g in mdf[mdf['SIZE_T'] == t].groupby('MONTH_P'):
            if len(g) < MIN_CROSS_SECTION:
                continue
            recs.append({'N': len(g), 'IC_PEARSON': g['RRR_PCT'].corr(g['RET_SIMPLE']),
                         'IC_SPEARMAN': g['RRR_PCT'].corr(g['RET_SIMPLE'], method='spearman')})
        d = pd.DataFrame(recs)
        assert len(d) == EXPECTED_T_MONTHS, f'{lab}: {len(d)} monthly ICs, expected {EXPECTED_T_MONTHS}'
        for col in ['IC_PEARSON', 'IC_SPEARMAN']:
            rec = summarize_ic(d[col], 12, IC_LABELS[col])
            rec['Size third'] = lab
            rec['Avg N firms'] = d['N'].mean()
            rows.append(rec)
    out = pd.DataFrame(rows)
    lead = ['Size third', 'IC type']
    out = out[lead + [c for c in out.columns if c not in lead]]
    show = lead + ['T', 'Mean IC', 'SD of IC', 'ICIR (per period)', 't Newey-West',
                   'p Newey-West', 'Hit rate (IC > 0)', 'Avg N firms']
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(out[show].round(4).to_string(index=False))
    return out


def q5_outlier_ic_table(summ):
    keep = summ[summ['Frequency'] == 'Monthly']
    return keep[['IC type', 'Mean IC', 'Median IC', 'SD of IC', 'ICIR (per period)',
                 't Newey-West', 'p Newey-West', 'Hit rate (IC > 0)']].reset_index(drop=True)


# =============================================================================
# Cell 8: figure
# =============================================================================
def plot_ic(ic):
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': COL_MUTED, 'axes.labelcolor': COL_INK,
                         'xtick.color': COL_INK, 'ytick.color': COL_INK})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True,
                                   gridspec_kw={'height_ratios': [1.15, 1]})
    x = ic.index
    for ax in (ax1, ax2):
        ax.set_facecolor('white')
        ax.grid(axis='y', color=COL_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        ax.axhline(0, color=COL_MUTED, linewidth=0.8)

    ax1.bar(x, ic['IC_SPEARMAN'], width=20, color=COL_SPEARMAN, alpha=0.55,
            label='Monthly Spearman rank IC')
    ax1.plot(x, ic[f'IC_SPEARMAN_ROLL{ROLLING_IC_MONTHS}'], color=COL_SPEARMAN_DARK, linewidth=2,
             label=f'{ROLLING_IC_MONTHS}-month rolling mean')
    ax1.set_ylabel('Rank IC')
    ax1.set_title('A. Monthly Spearman rank IC of RRR with holding-month returns',
                  loc='left', fontsize=10, color=COL_INK)
    ax1.legend(frameon=False, loc='upper left', ncol=2, fontsize=8)

    ax2.plot(x, ic['IC_SPEARMAN_CUM'], color=COL_SPEARMAN, linewidth=2, label='Spearman rank IC')
    ax2.plot(x, ic['IC_PEARSON_CUM'], color=COL_PEARSON, linewidth=2, label='Pearson IC')
    for col, colr, lab in [('IC_SPEARMAN_CUM', COL_SPEARMAN, 'Spearman'),
                           ('IC_PEARSON_CUM', COL_PEARSON, 'Pearson')]:
        ax2.annotate(f'{lab} {ic[col].iloc[-1]:.2f}', xy=(x[-1], ic[col].iloc[-1]),
                     xytext=(8, 0), textcoords='offset points', va='center', fontsize=8, color=COL_INK)
    ax2.set_ylabel('Cumulative IC')
    ax2.set_title('B. Cumulative sum of monthly ICs', loc='left', fontsize=10, color=COL_INK)
    ax2.legend(frameon=False, loc='upper left', fontsize=8)
    ax2.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(FIG_BASE + '.png', dpi=200)
    fig.savefig(FIG_BASE + '.pdf')
    plt.close(fig)
    print(f'\n  wrote {FIG_BASE}.png / .pdf')


# =============================================================================
# Cell 9: main
# =============================================================================
def main():
    df_filtered, returns, ff, panel, market_ret = rd.load_base_data()
    with quiet():
        port, reg = rd.baseline_portfolio_and_regressions(panel, ff, market_ret)
    guard_headline(port, reg, ff)

    corr_tidy, corr_matrix, loadings = q1_factor_corr_ts(port, ff)
    char_summary, char_by_q = q1_characteristic_corr(df_filtered, returns, ff, panel)
    ic, icq, ic_summary = q23_ic(panel, ff)
    by_year = q5_ic_by_year(ic)
    by_size = q5_ic_by_size(panel, ff)
    ls_stats, ls_drops, _top = q5_month_concentration(port, ff)
    jk = q5_jackknife(panel, ff, market_ret)
    plot_ic(ic)

    # Monthly long-short returns on the factor months (added 2026-10-20): the
    # input of the consistency histogram drawn by robustness_figures.py.
    ls_series = port['Q1-Q4'].copy()
    ls_series.index = to_month_end(ls_series.index)
    ls_series = ls_series.reindex(ls_series.index.intersection(ff.dropna(subset=FF5).index)).dropna()
    assert len(ls_series) == EXPECTED_T_MONTHS, f'{len(ls_series)} long-short months, expected {EXPECTED_T_MONTHS}'
    ls_monthly = pd.DataFrame({'Month': ls_series.index.strftime('%Y-%m'),
                               'Long-short return (%)': ls_series.values * 100})

    ic_out = ic.copy()
    ic_out.index = ic_out.index.strftime('%Y-%m')
    icq_out = icq.copy()
    icq_out.index = pd.to_datetime(icq_out.index).to_period('Q').astype(str)
    with pd.ExcelWriter(XLSX_PATH, engine='openpyxl') as xw:
        corr_tidy.to_excel(xw, sheet_name='factor_corr_ts', index=False)
        corr_matrix.to_excel(xw, sheet_name='factor_corr_matrix')
        loadings.to_excel(xw, sheet_name='factor_loadings_ff5', index=False)
        char_summary.to_excel(xw, sheet_name='char_corr_cs', index=False)
        char_by_q.to_excel(xw, sheet_name='char_corr_by_quarter', index=False)
        ic_out.to_excel(xw, sheet_name='ic_monthly')
        ic_summary.to_excel(xw, sheet_name='ic_summary', index=False)
        icq_out.to_excel(xw, sheet_name='ic_quarterly')
        by_year.to_excel(xw, sheet_name='ic_by_year', index=False)
        by_size.to_excel(xw, sheet_name='ic_by_size', index=False)
        q5_outlier_ic_table(ic_summary).to_excel(xw, sheet_name='outlier_ic', index=False)
        ls_stats.to_excel(xw, sheet_name='ls_month_concentration', index=False)
        ls_drops.to_excel(xw, sheet_name='ls_drop_best_months', index=False)
        ls_monthly.to_excel(xw, sheet_name='ls_monthly', index=False)
        jk.to_excel(xw, sheet_name='jackknife_firm', index=False)
    print(f'  wrote {XLSX_PATH}')


if __name__ == '__main__':
    main()
