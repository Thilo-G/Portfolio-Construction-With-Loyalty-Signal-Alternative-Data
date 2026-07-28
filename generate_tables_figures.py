"""
generate_tables_figures.py — RRR Financial Implications
========================================================
Generates all LaTeX tables (.tex) and figures (.pdf) for the paper.

Re-runs the analysis_v2.py data pipeline to get raw DataFrames,
reads pre-computed Excel outputs, and produces publication-quality outputs.

Tables: 8 .tex files -> Paper_LaTeX/tables/
Figures: ~16 .pdf files -> Paper_LaTeX/figures/
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy import stats as _scipy_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication"
CODE_DIR = os.path.join(BASE_DIR, "Code", "RRR-FI-IM")
OUTPUT_DIR = os.path.join(CODE_DIR, "output")
TABLE_DIR = os.path.join(BASE_DIR, "Paper_LaTeX", "tables")
FIGURE_DIR = os.path.join(BASE_DIR, "Paper_LaTeX", "figures")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# Import analysis_v2 functions
# NOTE (Stage 5 canonical-run rebuild): value_weighted_market, run_factor_regressions
# and safe_quartile are imported here (not reimplemented locally) so that every
# portfolio built in this script uses the EXACT SAME construction as
# identification_battery.py / robustness_diagnostics.py / exclude_amazon_robustness.py:
# build_holding_panel() -> value_weighted_market() / build_portfolio_returns()
# (formation-date MCAP_FORM weights, SIMPLE returns) -> run_factor_regressions()
# (Newey-West OLS, automatic Newey-West 1994 bandwidth -- do not hardcode max_lags
# when calling newey_west_ols() for any of these tables; see the Stage 5 fix note
# inside _run_inline_factor_reg() below for the bug this caused). Prior to this fix, Step 0b below reconstructed portfolios
# inline using RETURN_LOG (log returns) and the stale quarter-end HISTORICAL_MARKET_CAP,
# which does NOT match the canonical build_portfolio_returns() convention (simple
# returns, formation-date market cap) and produced alphas that diverged from the
# canonical baseline (e.g. the old inline Q1-Q4 ADJ RRR FF3 alpha was ~2.36%/mo vs the
# canonical 1.9198%/mo baseline confirmed across identification_battery.py,
# robustness_diagnostics.py and exclude_amazon_robustness.py's own "[CHECK]" assertions).
sys.path.insert(0, CODE_DIR)
from analysis_v2 import (
    phase1_load_and_diagnose, phase1b_load_ff_factors, phase1c_load_monthly_returns,
    merge_signals_to_returns, safe_tercile, safe_quartile, build_portfolio_returns,
    newey_west_ols, value_weighted_market, run_factor_regressions, build_holding_panel,
)

# Import export_table tool (house style: booktabs + threeparttable, no significance
# stars, exact p-values in brackets)
sys.path.insert(0, os.path.expanduser('~/.claude/tools'))
from export_table import export_custom_table

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
COLORS = sns.color_palette("colorblind", 10)
C_T1, C_T2, C_T3, C_MKT = COLORS[3], COLORS[2], COLORS[0], COLORS[7]
C_Q1, C_Q2, C_Q3, C_Q4 = COLORS[3], COLORS[2], COLORS[1], COLORS[0]

# Fixed sector color map — ensures sectors always use the same color across all figures
SECTOR_COLORS = {
    'Communication Services': COLORS[0],
    'Consumer Discretionary':  COLORS[1],
    'Consumer Staples':         COLORS[2],
    'Industrials':              COLORS[3],
    # Extra entries in case additional sectors appear in the data
    'Energy':                   COLORS[4],
    'Financials':               COLORS[5],
    'Health Care':              COLORS[6],
    'Information Technology':   COLORS[7],
    'Materials':                COLORS[8],
    'Real Estate':              COLORS[9],
    'Utilities':                COLORS[9],
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def fmt(v, dp=2):
    """Format float to dp decimal places."""
    if pd.isna(v):
        return ""
    return f"{v:.{dp}f}"


def fmt_int(v):
    """Format integer with comma separator."""
    if pd.isna(v):
        return ""
    return f"{int(v):,}"


# stars_from_t() (legacy significance-stars helper) has been removed as part of the
# Stage 5 house-style pass. No function in this file may produce '*'/'**'/'***' —
# house style is exact p-values in brackets only (see _fmt_pval above).


# ---------------------------------------------------------------------------
# Revenue_Growth-style table helpers
# ---------------------------------------------------------------------------

# House style (style_guide.md, CLAUDE.md hard constraint): NO significance stars,
# anywhere, ever. Exact p-values are reported in brackets instead. _stars() has been
# removed on purpose -- do not reintroduce it. _fmt_coef() below intentionally has no
# pval parameter so it cannot silently grow stars back in.

def _fmt_coef(coef, decimals=4):
    """Format coefficient with LaTeX minus sign. No significance stars (house style)."""
    if pd.isna(coef):
        return ''
    sign = '$-$' if coef < 0 else ''
    return f"{sign}{abs(coef):.{decimals}f}"


def _fmt_se(se, decimals=4):
    """Format standard error in parentheses."""
    if pd.isna(se):
        return ''
    return f"({se:.{decimals}f})"


def _fmt_pval(pval, decimals=3):
    """Format an exact p-value in brackets, no leading zero, '<.001'-style floor
    below the display precision. Matches this paper's existing table convention
    (e.g. [.003], [<.001]) and style_guide.md's 3-decimal p-value rule."""
    if pd.isna(pval):
        return ''
    floor = 10 ** (-decimals)
    if pval < floor:
        return f"[<{floor:.{decimals}f}]".replace('0.', '.', 1)
    s = f"{pval:.{decimals}f}"
    if s.startswith('0.'):
        s = s[1:]
    elif s.startswith('-0.'):
        s = '-' + s[2:]
    return f"[{s}]"


def _write_tabular(lines, output_path):
    """Write a list of LaTeX lines to a .tex file (UTF-8)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  Written: {os.path.basename(output_path)}")


# safe_quartile is imported from analysis_v2 (removed the local duplicate that used
# to live here) so quartile assignment is byte-identical to every other script in the
# canonical pipeline.


def _se_from_t(coef, t_stat):
    """Compute SE = |coef / t_stat|; returns NaN when t_stat is zero or NaN."""
    if pd.isna(t_stat) or t_stat == 0:
        return np.nan
    return abs(coef / t_stat)


# ============================================================================
# STEP 0: Re-run data pipeline
# ============================================================================
print("=" * 80)
print("  GENERATING TABLES AND FIGURES")
print("=" * 80)

print("\n[Step 0] Loading data via analysis_v2 pipeline...")
df_long, df_filtered, industry_stats = phase1_load_and_diagnose()
ff_factors = phase1b_load_ff_factors()
returns = phase1c_load_monthly_returns()
# merge_signals_to_returns() is a backward-compat alias for build_holding_panel():
# 2-month formation lag, 3-month hold, one row per (firm, holding-month). This IS the
# canonical timing (analysis_v2.py commit 1fce67f) -- unchanged from before.
panel = merge_signals_to_returns(df_filtered, returns)

# Also read pre-computed Excel files
pooled_desc = pd.read_excel(os.path.join(OUTPUT_DIR, 'pooled_descriptives.xlsx'))
industry_desc = pd.read_excel(os.path.join(OUTPUT_DIR, 'industry_descriptives.xlsx'))
risk_analysis = pd.read_excel(os.path.join(OUTPUT_DIR, 'risk_analysis.xlsx'))
fama_macbeth = pd.read_excel(os.path.join(OUTPUT_DIR, 'fama_macbeth.xlsx'))
portfolio_char_xlsx = pd.read_excel(os.path.join(OUTPUT_DIR, 'portfolio_characteristics.xlsx'))

print("\n  Data loaded successfully.")


# ============================================================================
# BUILD PORTFOLIO RETURN TIME SERIES (for figures and inline factor regressions)
# ============================================================================
# STAGE 5 FIX (canonical-run rebuild): every portfolio below is now built with
#   panel.groupby('QUARTER')[<contemporaneous signal column>].transform(safe_quartile)
#   -> build_portfolio_returns(panel, bucket_col, market_ret)
# i.e. the EXACT SAME two calls identification_battery.py / robustness_diagnostics.py /
# exclude_amazon_robustness.py use for the confirmed canonical baseline
# (ADJ RRR Q1-Q4: FF3=1.9198%/mo t=2.94, FF5=2.1547%/mo t=3.29, N=88).
#
# This replaces a prior version of this block that had TWO bugs relative to that
# baseline: (1) it value-weighted using the stale quarter-end HISTORICAL_MARKET_CAP
# instead of the formation-date MCAP_FORM, and aggregated RETURN_LOG (log returns)
# instead of RET_SIMPLE -- build_portfolio_returns()'s own docstring is explicit that
# a value-weighted mean of log returns is biased and "deliberately not used anywhere
# here"; and (2) it sorted on '<signal>_LAG1' columns (an extra .shift(1) computed in
# phase1_load_and_diagnose for a pre-timing-fix design), which stacks a SECOND lag on
# top of the formation gap that build_holding_panel() already applies -- exactly the
# "stacked double lag" bug commit 1fce67f fixed in analysis_v2.py itself. Sorts now use
# the CONTEMPORANEOUS quarter-end columns (ADJ_RRR_PCT, RRR_PCT, ...) already present on
# `panel`, matching build_holding_panel()'s own docstring convention.
print("\n[Step 0b] Building portfolio return time series (canonical timing)...")

market_ret = value_weighted_market(panel)

# Keep `ret` as an alias for `panel` -- Figures 17/18 further below still reference
# `ret[['RRR_Q', 'REV_GROWTH_PCT']]` at firm-holding-month granularity.
ret = panel

# --- Quartile portfolios (adjusted RRR) — MAIN RESULT / headline ---
panel['RRR_Q'] = panel.groupby('QUARTER')['ADJ_RRR_PCT'].transform(safe_quartile)
port_rrr_q = build_portfolio_returns(panel, 'RRR_Q', market_ret)
port_rrr_q['MKT'] = market_ret.reindex(port_rrr_q.index)

# --- Quartile portfolios (adjusted AR) ---
panel['AR_Q'] = panel.groupby('QUARTER')['ADJ_ACQ_RATE_PCT'].transform(safe_quartile)
port_ar_q = build_portfolio_returns(panel, 'AR_Q', market_ret)
port_ar_q['MKT'] = market_ret.reindex(port_ar_q.index)

# --- Quartile portfolios (raw RRR) — for Table 4 Panel A ---
panel['RRR_RAW_Q'] = panel.groupby('QUARTER')['RRR_PCT'].transform(safe_quartile)
port_rrr_raw_q = build_portfolio_returns(panel, 'RRR_RAW_Q', market_ret)
port_rrr_raw_q['MKT'] = market_ret.reindex(port_rrr_raw_q.index)

# --- Quartile portfolios (raw AR) — for Table 5 Panel A ---
panel['AR_RAW_Q'] = panel.groupby('QUARTER')['ACQ_RATE_PCT'].transform(safe_quartile)
port_ar_raw_q = build_portfolio_returns(panel, 'AR_RAW_Q', market_ret)
port_ar_raw_q['MKT'] = market_ret.reindex(port_ar_raw_q.index)

# --- No-COVID quartile portfolios (adjusted RRR) — for Table 8 Panel A ---
panel_nc = panel[~((panel['Date'] >= '2020-01-01') & (panel['Date'] <= '2021-06-30'))].copy()
market_ret_nc = value_weighted_market(panel_nc)
panel_nc['RRR_Q_NC'] = panel_nc.groupby('QUARTER')['ADJ_RRR_PCT'].transform(safe_quartile)
port_nc_q = build_portfolio_returns(panel_nc, 'RRR_Q_NC', market_ret_nc)
port_nc_q['MKT'] = market_ret_nc.reindex(port_nc_q.index)

# --- Equal-weighted quartile portfolios (adjusted RRR) — for Table 8 / consolidated
# robustness table. Mirrors analysis_v2.py phase4_additional_tests() section 4.3
# EXACTLY: equal-weight RET_SIMPLE (not log returns) within (Date, bucket) on the
# canonical holding panel. That function computes this same series but does not
# persist a factor-regression alpha to disk, so this is re-derived here (same
# unmodified inputs / same two lines of logic) and saved below for traceability.
panel['RRR_Q_EW'] = panel.groupby('QUARTER')['ADJ_RRR_PCT'].transform(safe_quartile)
_df_ew_q = panel.dropna(subset=['RRR_Q_EW']).copy()
_df_ew_q['RET_SIMPLE'] = pd.to_numeric(_df_ew_q['RET_SIMPLE'], errors='coerce')
port_ew_q = (
    _df_ew_q.groupby(['Date', 'RRR_Q_EW'])['RET_SIMPLE']
    .mean()
    .unstack('RRR_Q_EW')
    .sort_index()
)
if 'Q1' in port_ew_q.columns and 'Q4' in port_ew_q.columns:
    port_ew_q['Q1-Q4'] = port_ew_q['Q1'] - port_ew_q['Q4']
_ew_mkt = _df_ew_q.groupby('Date')['RET_SIMPLE'].mean().sort_index()
port_ew_q['MKT'] = _ew_mkt.reindex(port_ew_q.index)

# --- Port-forward robustness: 1-year-lag (4-quarter) RRR definition ---
# "Variant A" in analysis_v2.py phase1_load_and_diagnose (Phase 1.4b): LR_RRR_t =
# Rev_ret_t / Total_Revenue_{t-4}, i.e. retained revenue measured against the revenue
# base from four quarters (one year) prior, industry-time adjusted the same way as
# ADJ_RRR_PCT. This column (ADJ_LR_RRR_PCT) already exists on `panel` (it is one of
# build_holding_panel()'s alt_cols) and was previously only ever compared via a
# matplotlib PDF table (run_alt_metric_analysis(), output/alt_metric/portfolio_alphas_
# ff3.pdf) with no persisted spreadsheet -- ported forward here using the exact same
# canonical build_portfolio_returns()/run_factor_regressions() pipeline as every other
# table, and persisted to output/lr_rrr_robustness_stage5.xlsx for traceability.
panel['LR_RRR_Q'] = panel.groupby('QUARTER')['ADJ_LR_RRR_PCT'].transform(safe_quartile)
port_lr_rrr_q = build_portfolio_returns(panel, 'LR_RRR_Q', market_ret)

# --- Double sort portfolios (3x3, for Figure 9 heatmap; RRR x AR, NOT the excluded
# RRR x Size / RRR x BTM double sorts) ---
panel['RRR_T_DS'] = panel.groupby('QUARTER')['ADJ_RRR_PCT'].transform(safe_tercile)
panel['AR_T_DS'] = panel.groupby('QUARTER')['ADJ_ACQ_RATE_PCT'].transform(safe_tercile)
_df_ds = panel.dropna(subset=['RRR_T_DS', 'AR_T_DS']).copy()
_df_ds['PORT'] = _df_ds['RRR_T_DS'].astype(str) + '_' + _df_ds['AR_T_DS'].astype(str)
_df_ds['MCAP_FORM'] = pd.to_numeric(_df_ds['MCAP_FORM'], errors='coerce')
_df_ds['RET_SIMPLE'] = pd.to_numeric(_df_ds['RET_SIMPLE'], errors='coerce')
_df_ds['w'] = _df_ds['MCAP_FORM'] / _df_ds.groupby(['Date', 'PORT'])['MCAP_FORM'].transform('sum')
_df_ds['w_return'] = _df_ds['w'] * _df_ds['RET_SIMPLE']
port_double = (
    _df_ds.groupby(['Date', 'PORT'])['w_return']
    .sum()
    .unstack('PORT')
    .sort_index()
)

# --- Placebo median-split portfolios — within top quartile of revenue growth ---
# Mirrors analysis_v2.py run_placebo_median_split() exactly (contemporaneous
# ADJ_RRR_PCT, formation-date MCAP_FORM weights, simple returns).
panel['REV_GROWTH_PCT'] = pd.to_numeric(panel['REV_GROWTH_PCT'], errors='coerce')

def assign_growth_quartile(x):
    """Assign revenue-growth quartiles: Q1=highest growth."""
    x = x.dropna()
    if len(x) < 8:
        return pd.Series([np.nan]*len(x), index=x.index)
    try:
        return pd.qcut(x, 4, labels=['Q4_G', 'Q3_G', 'Q2_G', 'Q1_G'])
    except ValueError:
        return pd.Series([np.nan]*len(x), index=x.index)

def safe_median_split(x):
    """Median split: High = above median, Low = below median."""
    x = x.dropna()
    if len(x) < 4:
        return pd.Series([np.nan]*len(x), index=x.index)
    med = x.median()
    return pd.Series(np.where(x >= med, 'High', 'Low'), index=x.index)

panel['GROWTH_Q'] = panel.groupby('QUARTER')['REV_GROWTH_PCT'].transform(assign_growth_quartile)
# Keep only top-quartile revenue growth firms
high_growth_q = panel[panel['GROWTH_Q'] == 'Q1_G'].copy()
# Within this subsample, median-split by contemporaneous adjusted RRR
high_growth_q['RRR_M_PL'] = high_growth_q.groupby('QUARTER')['ADJ_RRR_PCT'].transform(safe_median_split)
df_pl = high_growth_q.dropna(subset=['RRR_M_PL']).copy()
df_pl['MCAP_FORM'] = pd.to_numeric(df_pl['MCAP_FORM'], errors='coerce')
df_pl['RET_SIMPLE'] = pd.to_numeric(df_pl['RET_SIMPLE'], errors='coerce')
df_pl['w'] = df_pl['MCAP_FORM'] / df_pl.groupby(['Date', 'RRR_M_PL'])['MCAP_FORM'].transform('sum')
df_pl['w_return'] = df_pl['w'] * df_pl['RET_SIMPLE']
port_placebo_m = (
    df_pl.groupby(['Date', 'RRR_M_PL'])['w_return']
    .sum()
    .unstack('RRR_M_PL')
    .sort_index()
)
if 'High' in port_placebo_m.columns and 'Low' in port_placebo_m.columns:
    port_placebo_m['High-Low'] = port_placebo_m['High'] - port_placebo_m['Low']
port_placebo_m['MKT'] = market_ret.reindex(port_placebo_m.index)

# Approximate firms per placebo group (for footnote)
_pl_firms_per_grp = df_pl.groupby(['QUARTER', 'RRR_M_PL'])['FIRM'].nunique().groupby('RRR_M_PL').mean()
print(f"  Placebo median-split avg firms/group: {_pl_firms_per_grp.to_dict()}")

print("  Portfolio time series built (canonical build_holding_panel / build_portfolio_returns pipeline).")


# ============================================================================
# INLINE FACTOR REGRESSION HELPERS
# ============================================================================

# Factor labels: map raw column names to display labels used in tables
FACTOR_LABELS = {
    'Mkt-RF': 'Market',
    'SMB':    'SMB',
    'HML':    'HML',
    'Mom':    'Mom',
    'RMW':    'RMW',
    'CMA':    'CMA',
}

# Ordered factor sequence for all regression tables
FACTOR_ORDER = ['Mkt-RF', 'SMB', 'HML', 'Mom', 'RMW', 'CMA']

# Factors active in each model
MODEL_FACTORS = {
    'FF3':     ['Mkt-RF', 'SMB', 'HML'],
    'FF3+Mom': ['Mkt-RF', 'SMB', 'HML', 'Mom'],
    'FF5':     ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
}

# Standard table footnote for factor regression tables
FACTOR_REG_NOTE = (
    r'\textit{Note: OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). '
    r'Standard errors in parentheses; exact $p$-values in brackets for alpha estimates. No significance stars are used.}'
)

# NW_LAGS is intentionally UNUSED as of the Stage 5 fix below (kept only so any old
# call site elsewhere that might still reference it does not raise NameError). Do not
# pass max_lags=NW_LAGS to newey_west_ols() -- see the fix note in
# _run_inline_factor_reg() for why (it silently produced a standard error that did
# not match the canonical identification_battery.py / robustness_diagnostics.py /
# exclude_amazon_robustness.py baseline, even though the coefficient matched).
NW_LAGS = 4

# Months per year (annualization factor) -- used by Figure 11's risk-comparison bars.
# Previously defined inline inside the old Table 7 _risk_row() helper; restored here
# as a module-level constant now that Table 7 has been rebuilt from
# performance_metrics.py and no longer defines it.
_MONTHS_PER_YEAR = 12


def _run_inline_factor_reg(port_ts, ff_factors_df):
    """Run FF3, FF3+Mom, FF5 regressions on each portfolio column.

    Parameters
    ----------
    port_ts : DataFrame
        Portfolio return time series. Columns include Q1, Q2, Q3, Q4, Q1-Q4, MKT.
        Index must be date-like (aligned or alignable to ff_factors_df).
    ff_factors_df : DataFrame
        Fama-French factors with columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF.

    Returns
    -------
    dict keyed by '{port}_{model}', each value is a dict with:
        alpha, alpha_t, alpha_p, se_alpha, r2, n_obs,
        beta_{factor}, se_{factor}, tstat_{factor}, pval_{factor}
    """
    # Align portfolio dates to month-end timestamps matching ff_factors_df index
    port = port_ts.copy()
    port.index = pd.to_datetime(port.index).to_period('M').to_timestamp('M')

    results = {}
    # BUG FIX (Stage 5): this list used to be hardcoded to ['Q1','Q2','Q3','Q4','Q1-Q4'],
    # so any caller with a differently-named portfolio set (e.g. port_placebo_m, whose
    # columns are High/Low/High-Low) silently matched NOTHING and got an empty
    # `results` dict back -- no error, but _build_panel_table() then rendered every
    # cell blank for that table. Auto-detect instead: every column except MKT is a
    # portfolio to regress. Verified this is equivalent to the old hardcoded list for
    # every existing Q1/Q2/Q3/Q4/Q1-Q4/MKT caller.
    portfolio_cols = [c for c in port.columns if c != 'MKT']

    for port_col in portfolio_cols:
        # For individual (single-leg) portfolios subtract RF to get excess return;
        # long-short / zero-cost spreads (name contains '-', e.g. 'Q1-Q4', 'High-Low')
        # are already zero-cost and must NOT have RF subtracted.
        if '-' in port_col:
            y_raw = port[port_col]
        else:
            rf_aligned = ff_factors_df['RF'].reindex(port.index)
            y_raw = port[port_col] - rf_aligned

        for model, factors in MODEL_FACTORS.items():
            # Build regressor matrix from aligned ff_factors
            X_df = ff_factors_df[factors].reindex(port.index)
            combined = pd.concat([y_raw, X_df], axis=1).dropna()
            if len(combined) < len(factors) + 5:
                continue

            y = combined.iloc[:, 0]
            X = combined.iloc[:, 1:]

            # BUG FIX (Stage 5): this used to pass max_lags=NW_LAGS (hardcoded to 4),
            # but analysis_v2.py's own run_factor_regressions() -- the function
            # identification_battery.py / robustness_diagnostics.py /
            # exclude_amazon_robustness.py all use for the canonical baseline -- calls
            # newey_west_ols(y, X) with NO max_lags, letting it fall through to the
            # Newey-West (1994) automatic-bandwidth rule (4*(T/100)**(2/9), which
            # evaluates to 3 for this sample's N). The hardcoded 4 produced a alpha
            # that matched the canonical baseline exactly but a standard error that
            # quietly did not (e.g. baseline ADJ RRR FF3: SE=0.6433 here vs the
            # canonical 0.6533 / t=2.94 confirmed across all four locked scripts).
            # Not passing max_lags reproduces the canonical SE exactly.
            ols_result = newey_west_ols(y, X)
            params = ols_result.params
            tvals  = ols_result.tvalues
            pvals  = ols_result.pvalues
            bses   = ols_result.bse

            key = f'{port_col}_{model}'
            entry = {
                'alpha':   params['const'],
                'alpha_t': tvals['const'],
                'alpha_p': pvals['const'],
                'se_alpha': bses['const'],
                'r2':      ols_result.rsquared,
                'n_obs':   int(ols_result.nobs),
            }
            for factor in factors:
                if factor in params.index:
                    entry[f'beta_{factor}']  = params[factor]
                    entry[f'se_{factor}']    = bses[factor]
                    entry[f'tstat_{factor}'] = tvals[factor]
                    entry[f'pval_{factor}']  = pvals[factor]
            results[key] = entry

    return results


def _build_panel_table(reg_results, port_labels, note_text):
    """Build a 3-panel factor regression table (one panel per model).

    Layout per panel:
        Columns: Q1, Q2, Q3, Q4, Q1-Q4
        Rows:    alpha (+SE), each factor beta (+SE), R2, N

    Parameters
    ----------
    reg_results : dict
        Output of _run_inline_factor_reg: keys are '{port}_{model}'.
    port_labels : list of str
        Portfolio column names in display order, e.g. ['Q1','Q2','Q3','Q4','Q1-Q4'].
    note_text : str
        LaTeX note string to place at the bottom of the table.

    Returns
    -------
    list of str — LaTeX lines from \\begin{tabular} to \\end{tabular}.
    """
    n_ports = len(port_labels)
    col_spec = 'l' + 'c' * n_ports
    note_span = n_ports + 1
    header_str = ' & '.join([''] + port_labels)

    panel_titles = {
        'FF3':     'Fama--French Three-Factor Model',
        'FF3+Mom': 'Carhart Four-Factor Model',
        'FF5':     'Fama--French Five-Factor Model',
    }

    # House style / project convention (CLAUDE.md "Storage" rule, confirmed against
    # every section file's own `\begin{table}[htbp]\caption{}\label{}\input{...}
    # \end{table}` wrapper): each tables/tab_*.tex fragment is self-contained
    # threeparttable + tabular + tablenotes ONLY -- no outer `table` float, no
    # \caption/\label (those live in the section file, one level up). This matches
    # every existing on-disk table in this project. Do not add a `table` float or
    # \caption/\label here -- it would nest a second `table` environment inside the
    # section file's own float once Stage 6 wires these in, which LaTeX does not
    # support (this is also why this function does NOT call the global
    # export_custom_table() -- that helper bundles its own `table` float, which
    # would conflict with this project's structure).
    lines = []
    lines.append('\\begin{threeparttable}')
    lines.append('\\footnotesize')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    lines.append(header_str + ' \\\\')

    for model_idx, model in enumerate(['FF3', 'FF3+Mom', 'FF5']):
        active_factors = MODEL_FACTORS[model]

        lines.append('\\midrule')
        lines.append(
            f'\\multicolumn{{{note_span}}}{{l}}'
            f'{{\\textbf{{{panel_titles[model]}}}}} \\\\'
        )
        lines.append('\\midrule')
        # NOTE: the portfolio header row (Q1/Q2/.../Q1-Q4) is NOT repeated here.
        # A single header row lives once at the top of the table (see above). Repeating
        # it after every panel title used to produce a duplicate-header-row bug that was
        # already flagged and fixed by hand in the committed .tex files; do not
        # reintroduce it here.

        # Alpha row: coefficient / SE / exact p-value (3 rows, no stars -- house style)
        alpha_cells = []
        se_alpha_cells = []
        p_alpha_cells = []
        for port in port_labels:
            key = f'{port}_{model}'
            if key in reg_results:
                r = reg_results[key]
                a_pct = r['alpha'] * 100     # convert to %/month
                se_pct = r['se_alpha'] * 100
                alpha_cells.append(_fmt_coef(a_pct))
                se_alpha_cells.append(_fmt_se(se_pct))
                p_alpha_cells.append(_fmt_pval(r['alpha_p']))
            else:
                alpha_cells.append('')
                se_alpha_cells.append('')
                p_alpha_cells.append('')
        lines.append('  $\\alpha$ (\\%/mo) & ' + ' & '.join(alpha_cells) + ' \\\\')
        lines.append('  & ' + ' & '.join(se_alpha_cells) + ' \\\\')
        lines.append('  & ' + ' & '.join(p_alpha_cells) + ' \\\\')

        # Factor beta rows (coefficient / SE only, no p-value row -- matches the
        # existing convention where only the focal alpha gets an exact p-value row)
        for factor in active_factors:
            beta_cells = []
            se_beta_cells = []
            for port in port_labels:
                key = f'{port}_{model}'
                b_key = f'beta_{factor}'
                s_key = f'se_{factor}'
                if key in reg_results and b_key in reg_results[key]:
                    r = reg_results[key]
                    beta_cells.append(_fmt_coef(r[b_key]))
                    se_beta_cells.append(_fmt_se(r[s_key]))
                else:
                    beta_cells.append('')
                    se_beta_cells.append('')
            label = FACTOR_LABELS[factor]
            lines.append(f'  {label} & ' + ' & '.join(beta_cells) + ' \\\\')
            lines.append('  & ' + ' & '.join(se_beta_cells) + ' \\\\')

        # R2 and N rows
        lines.append('\\midrule')
        r2_cells = []
        n_cells = []
        for port in port_labels:
            key = f'{port}_{model}'
            if key in reg_results:
                r2_cells.append(fmt(reg_results[key]['r2'], 4))
                n_cells.append(fmt_int(reg_results[key]['n_obs']))
            else:
                r2_cells.append('')
                n_cells.append('')
        lines.append('  $R^2$ & ' + ' & '.join(r2_cells) + ' \\\\')
        lines.append('  $N$ & ' + ' & '.join(n_cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\begin{tablenotes}')
    lines.append('\\small')
    lines.append(f'\\item {note_text}')
    lines.append('\\end{tablenotes}')
    lines.append('\\end{threeparttable}')
    return lines


# ============================================================================
# TABLE GENERATION HELPERS (legacy — used by FMB table and risk table)
# ============================================================================

# Standard column header for three-model tables
_THREE_MODEL_HEADER = '& (1) FF3 & (2) FF3+Mom & (3) FF5'
_THREE_MODEL_NCOLS = 4   # label + 3 model columns


# ============================================================================
# TABLE 1: Summary Statistics
# ============================================================================
print("\n[Table 1] Summary statistics...")

def _latex_escape_row(s):
    """Escape % and replace leading - with $-$ in table cells."""
    s = s.replace('%', '\\%')
    # Replace negative numbers: ' -X.XX' -> ' $-$X.XX'
    import re
    s = re.sub(r'(?<=& )-(\d)', r'$-$\1', s)
    # Also handle comma-separated thousands with negatives
    s = re.sub(r'(?<=& )-(\d)', r'$-$\1', s)
    return s

# House style: dollar-magnitude columns get 2 decimal places; every rate/ratio
# column gets 4 (style_guide.md; feedback_decimal_places_dollar_columns.md). Assets,
# Market Cap and Revenue are the only dollar-magnitude variables in pooled_desc.
_DOLLAR_VARS = {'Assets', 'Market Cap', 'Revenue'}

_tab1_lines = []
_tab1_lines.append('\\begin{threeparttable}')
_tab1_lines.append('\\footnotesize')
_tab1_lines.append('\\begin{tabular}{lrrrrrrrr}')
_tab1_lines.append('\\toprule')
_tab1_lines.append('Variable & N & Mean & Median & SD & P10 & P25 & P75 & P90 \\\\')
_tab1_lines.append('\\midrule')

for _, r in pooled_desc.iterrows():
    _dp = 2 if r['Variable'] in _DOLLAR_VARS else 4
    _row = (
        f"{r['Variable']} & {fmt_int(r['N'])} & {fmt(r['Mean'],_dp)} & "
        f"{fmt(r['Median'],_dp)} & {fmt(r['SD'],_dp)} & {fmt(r['P10'],_dp)} & "
        f"{fmt(r['P25'],_dp)} & {fmt(r['P75'],_dp)} & {fmt(r['P90'],_dp)} \\\\"
    )
    _tab1_lines.append(_latex_escape_row(_row))
_tab1_lines.append('\\bottomrule')
_tab1_lines.append('\\end{tabular}')
_tab1_lines.append('\\begin{tablenotes}')
_tab1_lines.append('\\small')
_tab1_lines.append(
    r'\item \textit{Notes:} Pooled summary statistics for the 124 sample firms over '
    r'2017Q1--2024Q3. RRR, AR, and Revenue Growth are expressed as percentages. '
    r'Assets, Market Cap and Revenue are in millions of USD (2 decimal places); all '
    r'other statistics are rates or ratios (4 decimal places). BTM is book-to-market '
    r'ratio. PM is operating profit margin (\%).'
)
_tab1_lines.append('\\end{tablenotes}')
_tab1_lines.append('\\end{threeparttable}')
_write_tabular(_tab1_lines, os.path.join(TABLE_DIR, 'tab_summary_stats.tex'))


# ============================================================================
# TABLE 2: Sample Composition
# ============================================================================
print("[Table 2] Sample composition...")

_tab2_lines = []
_tab2_lines.append('\\begin{threeparttable}')
_tab2_lines.append('\\resizebox{\\textwidth}{!}{%')
_tab2_lines.append('\\begin{tabular}{llrrrrrrrr}')
_tab2_lines.append('\\toprule')
_tab2_lines.append(
    'Sector & Metric & N & Mean & Median & SD & P10 & P25 & P75 & P90 \\\\'
)
_tab2_lines.append('\\midrule')
_sectors = industry_desc['Sector'].unique()
for _s_idx, _sector in enumerate(_sectors):
    _sd = industry_desc[industry_desc['Sector'] == _sector]
    _n_firms = df_filtered[df_filtered['SECTOR'] == _sector].reset_index()['FIRM'].nunique()
    _first = True
    for _, _r in _sd.iterrows():
        _sec_cell = f'\\textbf{{{_sector}}} ({_n_firms} firms)' if _first else ''
        _first = False
        _metric = str(_r['Metric']).replace('%', '\\%')
        # All columns here are percentage rates (RG/AR/RRR) -- 4dp per house style
        # (no dollar-magnitude columns in this table). Row is run through
        # _latex_escape_row() so negative values render with the proper math-mode
        # minus sign ($-$), matching every other table -- the pre-Stage-5 code this
        # block was based on built the row string directly without that step, which
        # would have left negative cells (e.g. Industrials RG median) with a plain
        # text hyphen instead.
        _row = (
            f"{_sec_cell} & {_metric} & {fmt_int(_r['N'])} & "
            f"{fmt(_r['Mean'],4)} & {fmt(_r['Median'],4)} & {fmt(_r['SD'],4)} & "
            f"{fmt(_r['P10'],4)} & {fmt(_r['P25'],4)} & {fmt(_r['P75'],4)} & "
            f"{fmt(_r['P90'],4)} \\\\"
        )
        _tab2_lines.append(_latex_escape_row(_row))
    if _s_idx < len(_sectors) - 1:
        _tab2_lines.append('\\midrule')
_tab2_lines.append('\\bottomrule')
_tab2_lines.append('\\end{tabular}%')
_tab2_lines.append('}')
_tab2_lines.append('\\begin{tablenotes}')
_tab2_lines.append('\\small')
_tab2_lines.append(
    r'\item \textit{Notes:} Number of firms and descriptive statistics for Revenue '
    r'Growth (RG), Acquisition Rate (AR), and Revenue Retention Rate (RRR) by GICS '
    r'sector. All values in percentages.'
)
_tab2_lines.append('\\end{tablenotes}')
_tab2_lines.append('\\end{threeparttable}')
_write_tabular(_tab2_lines, os.path.join(TABLE_DIR, 'tab_sample_composition.tex'))


# ============================================================================
# TABLE 3: (REMOVED — was Signal Persistence)
# ============================================================================
# Table 3 (tab_persistence.tex) has been removed from the paper.


# ============================================================================
# TABLE 3a/3b: Portfolio Alphas (RRR) — split into raw + adjusted
# ============================================================================
print("[Table 3a/3b] Portfolio alphas (RRR)...")

_rrr_port_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4']

_tab3a_note = (
    r'\textit{Note: Value-weighted raw RRR quartile portfolio factor regressions. '
    r'Q1 = highest RRR, Q4 = lowest RRR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). '
    r'Standard errors in parentheses; exact $p$-values in brackets for alpha estimates. No significance stars are used.}'
)
_tab3b_note = (
    r'\textit{Note: Value-weighted industry-time-adjusted RRR quartile portfolio factor regressions. '
    r'Q1 = highest adjusted RRR, Q4 = lowest adjusted RRR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). '
    r'Standard errors in parentheses; exact $p$-values in brackets for alpha estimates. No significance stars are used.}'
)

print("  Computing RRR regressions (raw)...")
_reg_rrr_raw = _run_inline_factor_reg(port_rrr_raw_q, ff_factors)
print("  Computing RRR regressions (adjusted)...")
_reg_rrr_adj = _run_inline_factor_reg(port_rrr_q, ff_factors)

# Write two separate files — each is a standalone _build_panel_table output
_write_tabular(
    _build_panel_table(_reg_rrr_raw, _rrr_port_labels, _tab3a_note),
    os.path.join(TABLE_DIR, 'tab_portfolio_alphas_raw.tex')
)
_write_tabular(
    _build_panel_table(_reg_rrr_adj, _rrr_port_labels, _tab3b_note),
    os.path.join(TABLE_DIR, 'tab_portfolio_alphas_adj.tex')
)


# ============================================================================
# TABLE 4a/4b: Portfolio Alphas (AR) — split into raw + adjusted
# ============================================================================
print("[Table 4a/4b] Portfolio alphas (AR)...")

_ar_port_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4']

_tab4a_ar_note = (
    r'\textit{Note: Value-weighted raw AR quartile portfolio factor regressions. '
    r'Q1 = highest AR, Q4 = lowest AR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). '
    r'Standard errors in parentheses; exact $p$-values in brackets for alpha estimates. No significance stars are used.}'
)
_tab4b_ar_note = (
    r'\textit{Note: Value-weighted industry-time-adjusted AR quartile portfolio factor regressions. '
    r'Q1 = highest adjusted AR, Q4 = lowest adjusted AR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). '
    r'Standard errors in parentheses; exact $p$-values in brackets for alpha estimates. No significance stars are used.}'
)

print("  Computing AR regressions (raw)...")
_reg_ar_raw = _run_inline_factor_reg(port_ar_raw_q, ff_factors)
print("  Computing AR regressions (adjusted)...")
_reg_ar_adj = _run_inline_factor_reg(port_ar_q, ff_factors)

_write_tabular(
    _build_panel_table(_reg_ar_raw, _ar_port_labels, _tab4a_ar_note),
    os.path.join(TABLE_DIR, 'tab_portfolio_alphas_ar_raw.tex')
)
_write_tabular(
    _build_panel_table(_reg_ar_adj, _ar_port_labels, _tab4b_ar_note),
    os.path.join(TABLE_DIR, 'tab_portfolio_alphas_ar_adj.tex')
)
# (old single-file AR table removed — now split into raw + adj above)


# ============================================================================
# TABLE 6: Fama-MacBeth Regressions
# ============================================================================
print("[Table 6] Fama-MacBeth regressions...")

# Reorganize FMB data into columns — derive SEs from coefficient / t-stat
_fmb_specs = list(fama_macbeth['Spec'].unique())
_fmb_vars_order = list(fama_macbeth['Variable'].unique())   # preserves original order

_fmb_var_labels = {
    'ADJ_RRR_PCT':      'Adj.~RRR (\\%)',
    'ADJ_ACQ_RATE_PCT': 'Adj.~AR (\\%)',
    'RRR_PCT':          'Raw RRR (\\%)',
    'ACQ_RATE_PCT':     'Raw AR (\\%)',
    'SIZE_LAG':         'Size',
    'BTM_LAG':          'BTM',
    'PM_LAG':           'Profit Margin',
    'MKT_BETA':         'Market $\\beta$',
}

_n_fmb_cols = len(_fmb_specs)
_fmb_col_labels = [f'({i+1})' for i in range(_n_fmb_cols)]
_fmb_col_spec = 'l' + 'c' * _n_fmb_cols

# Group header: "Adjusted signal" vs "Raw signal", spanning however many consecutive
# columns share a group (run-length encoded so this does not hardcode a 2+2 split).
_fmb_group_labels = ['Adjusted signal' if 'Adj' in _s else 'Raw signal' for _s in _fmb_specs]
_fmb_groups = []  # list of (label, span)
for _lbl in _fmb_group_labels:
    if _fmb_groups and _fmb_groups[-1][0] == _lbl:
        _fmb_groups[-1] = (_lbl, _fmb_groups[-1][1] + 1)
    else:
        _fmb_groups.append((_lbl, 1))
_fmb_group_row = ['']
_fmb_cmidrules = []
_col = 2  # first data column in LaTeX 1-indexing (col 1 is the row-label column)
for _lbl, _span in _fmb_groups:
    _fmb_group_row.append(f'\\multicolumn{{{_span}}}{{c}}{{{_lbl}}}')
    _fmb_cmidrules.append(f'\\cmidrule(lr){{{_col}-{_col + _span - 1}}}')
    _col += _span
_fmb_header = ' & '.join([''] + _fmb_col_labels)

_tab6_lines = []
_tab6_lines.append('\\begin{threeparttable}')
_tab6_lines.append('\\footnotesize')
_tab6_lines.append(f'\\begin{{tabular}}{{{_fmb_col_spec}}}')
_tab6_lines.append('\\toprule')
_tab6_lines.append(' & '.join(_fmb_group_row) + ' \\\\')
_tab6_lines.append(''.join(_fmb_cmidrules))
_tab6_lines.append(_fmb_header + ' \\\\')
_tab6_lines.append('\\midrule')

for _var in _fmb_vars_order:
    _var_data = fama_macbeth[fama_macbeth['Variable'] == _var]
    _display = _fmb_var_labels.get(_var, _var.replace('_', '\\_'))
    _coef_cells = []
    _se_cells = []
    _p_cells = []
    for _spec in _fmb_specs:
        _sv = _var_data[_var_data['Spec'] == _spec]
        if len(_sv) > 0:
            _c = _sv.iloc[0]['Coefficient']
            _t = _sv.iloc[0]['t-stat']
            # Derive p-value from t-stat (two-tailed normal approximation; the FMB
            # coefficients here are time-series averages with Newey-West t-stats, so a
            # normal reference distribution is the standard choice).
            _p = 2 * _scipy_stats.norm.sf(abs(_t))
            _se = _se_from_t(_c, _t)
            _coef_cells.append(_fmt_coef(_c))
            _se_cells.append(_fmt_se(_se) if not np.isnan(_se) else '')
            _p_cells.append(_fmt_pval(_p))
        else:
            _coef_cells.append('')
            _se_cells.append('')
            _p_cells.append('')
    _tab6_lines.append(f'  {_display} & ' + ' & '.join(_coef_cells) + ' \\\\')
    _tab6_lines.append('  \\quad & ' + ' & '.join(_se_cells) + ' \\\\')
    _tab6_lines.append('  \\quad & ' + ' & '.join(_p_cells) + ' \\\\')

# Summary rows: T (quarters) and Avg N
_tab6_lines.append('\\midrule')
_t_cells = []
_n_avg_cells = []
for _spec in _fmb_specs:
    _sd = fama_macbeth[fama_macbeth['Spec'] == _spec]
    if len(_sd) > 0:
        _t_cells.append(fmt_int(_sd.iloc[0]['T']))
        _n_avg_cells.append(fmt(_sd.iloc[0]['Avg N'], 0))
    else:
        _t_cells.append('')
        _n_avg_cells.append('')
_tab6_lines.append('  T (quarters) & ' + ' & '.join(_t_cells) + ' \\\\')
_tab6_lines.append('  Avg.\\ N & ' + ' & '.join(_n_avg_cells) + ' \\\\')

_tab6_lines.append('\\bottomrule')
_tab6_lines.append('\\end{tabular}')
_tab6_lines.append('\\begin{tablenotes}')
_tab6_lines.append('\\small')
_tab6_lines.append(
    r'\item \textit{Notes:} Source: analysis\_v2.py (fama\_macbeth.xlsx), run\_fama\_macbeth(). '
    r'Time-series averages of quarterly cross-sectional regression coefficients (Fama-MacBeth); '
    r'dependent variable is each firm'"'"'s compounded excess return over the same 2-month-formation, '
    r'3-month-hold window as the portfolio sorts (identical timing, not an independent check). '
    r'Columns (1)--(2) use the industry-time-adjusted \RRR{} signal; columns (3)--(4) use the raw '
    r'\RRR{} signal. Controls (columns 2 and 4) are contemporaneous: Size = log market '
    r'capitalization, BTM = book-to-market, Profit Margin = operating income / revenue. '
    r'Standard errors in parentheses, derived from the reported $t$-statistic '
    r'(Newey-West automatic bandwidth, Newey and West 1994, applied to the time series of quarterly '
    r'cross-sectional coefficients); exact $p$-values in brackets. No significance stars are used.'
)
_tab6_lines.append('\\end{tablenotes}')
_tab6_lines.append('\\end{threeparttable}')
_write_tabular(_tab6_lines, os.path.join(TABLE_DIR, 'tab_fmb.tex'))


# ============================================================================
# TABLE 7 (Stage 5 REPLACEMENT): Performance / Risk Metrics — full battery
# ============================================================================
# Author decision (Stage 5): performance_metrics.py's numbers are authoritative for
# this table, not an inline recomputation. performance_metrics.py was purpose-built
# for this paper with explicit, documented metric choices (sample's own market
# portfolio as the benchmark rather than FF Mkt-RF; a stricter textbook Sortino
# definition than what's used elsewhere in the repo, by deliberate design) and has
# already been independently verified bit-identical to the main portfolio
# construction. The PRIOR version of this table (kept in git history) recomputed
# these statistics inline from port_rrr_q using cumsum()-based drawdown, which
# silently assumed LOG returns; that assumption stopped holding once Step 0b above
# switched portfolio construction to SIMPLE returns (build_portfolio_returns()),
# so the inline version was quietly wrong for Max Drawdown / cumulative-return-based
# statistics. Retired in favor of reading the already-computed, already-verified
# battery below. Same output filename (tab_risk_stats.tex) is kept so the existing
# \input{tables/tab_risk_stats} in 06_results.tex continues to work unmodified.
print("[Table 7] Performance / risk metrics battery (performance_metrics.py)...")

_perf_metrics = pd.read_excel(
    os.path.join(OUTPUT_DIR, 'performance_metrics_summary.xlsx'), sheet_name='Metrics'
).set_index('Unnamed: 0')
_perf_cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4', 'MKT']

# (row label in source, display label, decimal places, suffix)
_perf_rows_spec = [
    ('Ann Return (%)',          'Ann.\\ Return',        2, '\\%'),
    ('Ann Vol (%)',             'Ann.\\ Volatility',    2, '\\%'),
    ('Sharpe Ratio',            'Sharpe Ratio',         4, ''),
    ('Sortino Ratio',           'Sortino Ratio',        4, ''),
    ('Tracking Error (%)',      'Tracking Error',       2, '\\%'),
    ('Information Ratio',       'Information Ratio',    4, ''),
    ('Downside Beta',           'Downside $\\beta$',    4, ''),
    ('Upside Beta',             'Upside $\\beta$',      4, ''),
    ('Max Drawdown (%)',        'Max Drawdown',         2, '\\%'),
    ('Turnover (avg qtrly, %)', 'Turnover (qtrly avg)', 2, '\\%'),
    ('Hit Rate vs Market (%)',  'Hit Rate vs.\\ Market',1, '\\%'),
    ('Hit Rate vs Q4 (%)',      'Hit Rate vs.\\ Q4',    1, '\\%'),
]

_tab7_lines = []
_tab7_lines.append('\\begin{threeparttable}')
_tab7_lines.append('\\resizebox{\\textwidth}{!}{%')
_tab7_lines.append('\\begin{tabular}{lrrrrrr}')
_tab7_lines.append('\\toprule')
_tab7_lines.append(' & ' + ' & '.join(_perf_cols) + ' \\\\')
_tab7_lines.append('\\midrule')
for _src_label, _disp_label, _dp, _suffix in _perf_rows_spec:
    _cells = []
    for _c in _perf_cols:
        _v = _perf_metrics.loc[_src_label, _c]
        _cells.append(fmt(_v, _dp) + _suffix if pd.notna(_v) else '')
    _row = f'{_disp_label} & ' + ' & '.join(_cells) + ' \\\\'
    _tab7_lines.append(_latex_escape_row(_row))
_tab7_lines.append('\\midrule')
_n_cells = [fmt_int(_perf_metrics.loc['N (months)', _c]) for _c in _perf_cols]
_tab7_lines.append('$N$ (months) & ' + ' & '.join(_n_cells) + ' \\\\')
_tab7_lines.append('\\bottomrule')
_tab7_lines.append('\\end{tabular}%')
_tab7_lines.append('}')
_tab7_lines.append('\\begin{tablenotes}')
_tab7_lines.append('\\small')
_tab7_lines.append(
    r'\item \textit{Notes:} Full performance/risk battery for value-weighted adjusted-\RRR{} '
    r'quartile portfolios (source: performance\_metrics.py). Q1 = highest \RRR{}, Q4 = lowest, '
    r'Q1$-$Q4 = long-short, MKT = the sample'"'"'s own value-weighted market portfolio (not FF Mkt-RF). '
    r'Tracking Error and Information Ratio are computed against MKT. Downside/Upside Beta are betas on '
    r'MKT estimated only in months MKT fell/rose, respectively. Sortino Ratio uses a strict textbook '
    r'downside-deviation definition (0\% threshold), which need not numerically match Sortino figures '
    r'computed elsewhere in the underlying codebase under a different convention. '
    r'Hit Rate vs.\ Q4 reports, for each column portfolio, the fraction of months it outperforms Q4 '
    r'(trivially 0\% in the Q4 column itself; not meaningfully defined for the Q1$-$Q4 spread, shown '
    r'blank). Turnover is 0.5 $\times$ the sum of absolute formation-weight changes at each quarterly '
    r're-formation.'
)
_tab7_lines.append('\\end{tablenotes}')
_tab7_lines.append('\\end{threeparttable}')
_write_tabular(_tab7_lines, os.path.join(TABLE_DIR, 'tab_risk_stats.tex'))


# ============================================================================
# TABLE 8: Robustness — 3-panel layout, computed inline
# Panel A: No-COVID quartile portfolios (adjusted RRR)
# Panel B: Equal-weighted quartile portfolios (adjusted RRR)
# ============================================================================
print("[Table 8] Robustness...")

_rob_port_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4']
_tab8_ncols = len(_rob_port_labels) + 1
_tab8_note = (
    r'\textit{Note: Robustness checks for adjusted RRR quartile portfolios. '
    r'Q1 = highest RRR, Q4 = lowest RRR, Q1$-$Q4 = long-short. '
    r'Panel A excludes 2020Q1--2021Q2 (COVID period). '
    r'Panel B uses equal-weighted returns. '
    r'OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). '
    r'Standard errors in parentheses; exact $p$-values in brackets for alpha estimates. No significance stars are used.}'
)

_tab_nc_note = (
    r'\textit{Note: Excluding COVID-19 period (2020Q1--2021Q2). '
    r'Value-weighted adjusted RRR quartile portfolios. '
    r'Q1 = highest RRR, Q4 = lowest RRR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). '
    r'Standard errors in parentheses; exact $p$-values in brackets for alpha estimates. No significance stars are used.}'
)
_tab_ew_note = (
    r'\textit{Note: Equal-weighted adjusted RRR quartile portfolios. '
    r'Q1 = highest RRR, Q4 = lowest RRR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). '
    r'Standard errors in parentheses; exact $p$-values in brackets for alpha estimates. No significance stars are used.}'
)

print("  Computing no-COVID robustness regressions...")
_reg_nc = _run_inline_factor_reg(port_nc_q, ff_factors)
print("  Computing equal-weighted robustness regressions...")
_reg_ew = _run_inline_factor_reg(port_ew_q, ff_factors)

_write_tabular(
    _build_panel_table(_reg_nc, _rob_port_labels, _tab_nc_note),
    os.path.join(TABLE_DIR, 'tab_robustness_nocovid.tex')
)
_write_tabular(
    _build_panel_table(_reg_ew, _rob_port_labels, _tab_ew_note),
    os.path.join(TABLE_DIR, 'tab_robustness_ew.tex')
)


# ============================================================================
# TABLE: Same-Growth Placebo — median split within top-quartile revenue growth
# ============================================================================
print("[Table] Placebo (median split)...")

_pl_port_labels = ['High', 'Low', 'High-Low']
_pl_avg_firms = int(_pl_firms_per_grp.mean()) if len(_pl_firms_per_grp) > 0 else 15
_tab_pl_note = (
    r'\textit{Note: Median split by adjusted RRR within the top quartile of revenue growth. '
    r'All firms have similarly high headline growth; the sort isolates revenue composition. '
    r'High = above-median RRR, Low = below-median RRR, High$-$Low = long-short. '
    rf'Average group size is approximately {_pl_avg_firms} firms per quarter. '
    r'OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). '
    r'Standard errors in parentheses; exact $p$-values in brackets for alpha estimates. No significance stars are used.}'
)

print("  Computing placebo regressions...")
_reg_placebo = _run_inline_factor_reg(port_placebo_m, ff_factors)

_write_tabular(
    _build_panel_table(_reg_placebo, _pl_port_labels, _tab_pl_note),
    os.path.join(TABLE_DIR, 'tab_placebo.tex')
)

print("\n  8 legacy tables regenerated under corrected timing.")


# ============================================================================
# STAGE 5: CANONICAL-RUN TABLE REBUILD
# ----------------------------------------------------------------------------
# Everything below is new for Stage 5 (canonical-run rebuild / house-style pass).
# All tables below are pulled from the already-committed, already-run output of
# identification_battery.py, robustness_diagnostics.py, exclude_amazon_robustness.py,
# future_beta_retry.py and performance_metrics.py -- no analysis is recomputed here,
# only read, reorganized and formatted. Where a script's own committed .xlsx omits a
# figure it demonstrably computed (found during this stage; see
# canonical_run_manifest.json and the per-table notes below), the figure is taken
# from a clean, verified re-run of the UNMODIFIED script's console output rather than
# either recomputing it differently or silently dropping it.
# ============================================================================

def _xlsx_to_reg_results(df):
    """Convert a factor-regression xlsx already on the _build_panel_table() key
    convention (first column holds keys like 'Q1_FF3', 'Q1-Q4_FF5', ...; other
    columns are alpha/alpha_t/alpha_p/se_alpha/r2/n_obs/beta_<factor>/se_<factor>/
    tstat_<factor>/pval_<factor>) into the same {key: {...}} dict shape
    _build_panel_table() expects, so xlsx output already produced by the locked
    analysis scripts can be rendered with the exact same table-building code used
    for the portfolios computed inline in this script."""
    key_col = df.columns[0]
    results = {}
    for _, row in df.iterrows():
        entry = {
            'alpha': row['alpha'], 'alpha_t': row['alpha_t'], 'alpha_p': row['alpha_p'],
            'se_alpha': row['se_alpha'], 'r2': row['r2'], 'n_obs': row['n_obs'],
        }
        for factor in FACTOR_ORDER:
            beta_col = f'beta_{factor}'
            if beta_col in df.columns and pd.notna(row[beta_col]):
                entry[f'beta_{factor}'] = row[beta_col]
                entry[f'se_{factor}'] = row[f'se_{factor}']
        results[row[key_col]] = entry
    return results


DIAG_XLSX = os.path.join(OUTPUT_DIR, 'robustness_diagnostics.xlsx')
IDBAT_XLSX = os.path.join(OUTPUT_DIR, 'identification_battery.xlsx')


# ============================================================================
# TABLE: Portfolio Characteristics (reuse — was orphaned; portfolio_char_xlsx was
# already loaded at Step 0 but never used by any table below it)
# ============================================================================
print("[Table] Portfolio characteristics (fixing precision: dollar cols 2dp, rate/ratio cols 4dp)...")

_pc = portfolio_char_xlsx.set_index('Quartile')
_PC_DOLLAR_COLS = ['Revenue (M)', 'Market Cap (M)']
_PC_RATE_COLS = ['Rev Growth (%)', 'SoNR (%)', 'RRR (%)', 'BTM', 'Op. ROA (%)']

_tab_pc_lines = []
_tab_pc_lines.append('\\begin{threeparttable}')
_tab_pc_lines.append('\\footnotesize')
_tab_pc_lines.append('\\setlength{\\tabcolsep}{4pt}')
_tab_pc_lines.append('\\begin{tabular}{lrrrrrrrr}')
_tab_pc_lines.append('\\toprule')
_tab_pc_lines.append(
    'Quartile & Avg $N$ & Revenue (M) & Mkt Cap (M) & Rev Growth (\\%) & SoNR (\\%) & RRR (\\%) & BTM & Op.\\ ROA (\\%) \\\\'
)
_tab_pc_lines.append('\\midrule')
for _q in ['Q1', 'Q2', 'Q3', 'Q4']:
    _r = _pc.loc[_q]
    _qlabel = f'{_q} (High RRR)' if _q == 'Q1' else f'{_q} (Low RRR)' if _q == 'Q4' else _q
    _cells = [fmt(_r['Avg N / quarter'], 1)]
    _cells += [fmt(_r[_c], 2) for _c in _PC_DOLLAR_COLS]
    _cells += [fmt(_r[_c], 4) for _c in _PC_RATE_COLS]
    _tab_pc_lines.append(_latex_escape_row(f'{_qlabel} & ' + ' & '.join(_cells) + ' \\\\'))
_diff = _pc.loc['Q1'] - _pc.loc['Q4']
_diff_cells = [''] + [fmt(_diff[_c], 2) for _c in _PC_DOLLAR_COLS] + [fmt(_diff[_c], 4) for _c in _PC_RATE_COLS]
_tab_pc_lines.append('\\midrule')
_tab_pc_lines.append(_latex_escape_row('$Q1-Q4$ & ' + ' & '.join(_diff_cells) + ' \\\\'))
_tab_pc_lines.append('\\bottomrule')
_tab_pc_lines.append('\\end{tabular}')
_tab_pc_lines.append('\\begin{tablenotes}')
_tab_pc_lines.append('\\small')
_tab_pc_lines.append(
    r'\item \textit{Notes:} Time-series averages of cross-sectional mean characteristics for '
    r'value-weighted industry-time-adjusted \RRR{} quartile portfolios. Q1 = highest adjusted \RRR{}, '
    r'Q4 = lowest. Avg $N$ = average number of firms per quarter. Rev Growth = quarterly revenue growth '
    r'rate (\%). SoNR = Share of New Revenue = '
    r'$\text{Rev}^{\text{new}}_{i,t} / \text{Rev}_{i,t} \times 100$, the fraction of current-period '
    r'revenue attributable to newly acquired customers (\%). BTM = (total assets $-$ total liabilities) '
    r'/ market capitalization. Op.\ ROA = operating income / total assets (\%). Dollar-magnitude '
    r'columns (Revenue, Market Cap) are reported to 2 decimal places; all rate and ratio columns to 4. '
    r'The Q1$-$Q4 row reports the difference in cross-sectional means.'
)
_tab_pc_lines.append('\\end{tablenotes}')
_tab_pc_lines.append('\\end{threeparttable}')
_write_tabular(_tab_pc_lines, os.path.join(TABLE_DIR, 'tab_portfolio_characteristics.tex'))


# ============================================================================
# TABLE: Signal Persistence / Autocorrelation (reuse — matches task item 6:
# firm-level AR(1) alongside the quartile transition-matrix diagonal)
# ============================================================================
print("[Table] Signal persistence & autocorrelation...")

_sp_df = pd.read_excel(os.path.join(OUTPUT_DIR, 'signal_persistence_comparison.xlsx')).set_index('Metric')
_SP_COLS = ['Raw RRR', 'Adj RRR', 'Raw AR', 'Adj AR']

_tab_sp_lines = []
_tab_sp_lines.append('\\begin{threeparttable}')
_tab_sp_lines.append('\\resizebox{\\textwidth}{!}{%')
_tab_sp_lines.append('\\begin{tabular}{lrrrr}')
_tab_sp_lines.append('\\toprule')
_tab_sp_lines.append(' & Raw \\RRR{} & Adj.\\ \\RRR{} & Raw \\AR{} & Adj.\\ \\AR{} \\\\')
_tab_sp_lines.append('\\midrule')
_tab_sp_lines.append(_latex_escape_row(
    'Cross-sect SD (%/quarter) & ' +
    ' & '.join(fmt(_sp_df.loc['Cross-sect SD (%/quarter)', c], 2) for c in _SP_COLS) + ' \\\\'
))
_tab_sp_lines.append('\\midrule')
_tab_sp_lines.append('\\multicolumn{5}{l}{\\textit{Autocorrelation $\\mathrm{corr}(t,\\,t-k)$, firm-level Pearson}} \\\\')
_AC_ROWS = [
    ('Lag-1 corr mean', 'Lag $k=1$ mean', 4),
    ('Lag-1 corr median', 'Lag $k=1$ median', 4),
    ('Lag-1 % positive', 'Lag $k=1$ % pos.', 1),
    ('Lag-4 corr mean', 'Lag $k=4$ mean', 4),
    ('Lag-4 corr median', 'Lag $k=4$ median', 4),
    ('Lag-4 % positive', 'Lag $k=4$ % pos.', 1),
]
for _metric, _label, _dp in _AC_ROWS:
    _row = f'\\quad {_label} & ' + ' & '.join(fmt(_sp_df.loc[_metric, c], _dp) for c in _SP_COLS) + ' \\\\'
    _tab_sp_lines.append(_latex_escape_row(_row))
_tab_sp_lines.append('\\midrule')
_tab_sp_lines.append('\\multicolumn{5}{l}{\\textit{Quartile stay-rate (fraction staying in same quartile, quarter to quarter)}} \\\\')
_SR_ROWS = [
    ('Overall stay-rate (%)', 'Overall'),
    ('Q1 stay-rate (%)', 'Q1 (highest)'),
    ('Q2 stay-rate (%)', 'Q2'),
    ('Q3 stay-rate (%)', 'Q3'),
    ('Q4 stay-rate (%)', 'Q4 (lowest)'),
]
for _metric, _label in _SR_ROWS:
    _cells = ' & '.join(f"{_sp_df.loc[_metric, c]:.1f}\\%" for c in _SP_COLS)
    _tab_sp_lines.append(f'\\quad {_label} & {_cells} \\\\')
_tab_sp_lines.append('\\bottomrule')
_tab_sp_lines.append('\\end{tabular}%')
_tab_sp_lines.append('}')
_tab_sp_lines.append('\\begin{tablenotes}')
_tab_sp_lines.append('\\small')
_tab_sp_lines.append(
    r'\item \textit{Notes:} Source: analysis\_v2.py (signal\_persistence\_comparison.xlsx, '
    r'quartile\_persistence.xlsx). Raw signals are unadjusted firm-level quarterly values; adjusted '
    r'signals are demeaned within GICS sector $\times$ quarter. Cross-sectional SD is the time-series '
    r'mean of the quarterly within-cross-section standard deviation. $\mathrm{corr}(t,\,t-k)$ is the '
    r'firm-level Pearson correlation between the signal in quarter $t$ and quarter $t-k$ ($N=124$ '
    r'firms); $k=4$ corresponds to the same calendar quarter one year prior. Lag-1 correlations are '
    r'negative for all four signals (quarter-to-quarter mean-reversion); lag-4 correlations are '
    r'positive for all four, with adjusted \RRR{} showing the highest annual persistence (mean '
    r'$=0.444$). Quartile stay-rate is the fraction of firm-quarter transitions where a firm remains in '
    r'the same quartile as the prior quarter (Q1 = highest signal, Q4 = lowest); the Q4 stay-rate is '
    r'markedly higher than Q1--Q3 for both \RRR{} definitions, indicating the low-retention tail is the '
    r'most persistent group.'
)
_tab_sp_lines.append('\\end{tablenotes}')
_tab_sp_lines.append('\\end{threeparttable}')
_write_tabular(_tab_sp_lines, os.path.join(TABLE_DIR, 'tab_signal_persistence.tex'))


# ============================================================================
# TABLE: Robustness — Two-Quarter Lag (reuse — was orphaned; factor_reg_Lag2_adj.xlsx
# refreshed in the prior commit but no code in this file rebuilt the .tex from it)
# ============================================================================
print("[Table] Robustness: two-quarter (extra formation-gap) lag...")

_lag2_df = pd.read_excel(os.path.join(OUTPUT_DIR, 'factor_reg_Lag2_adj.xlsx'))
_reg_lag2 = _xlsx_to_reg_results(_lag2_df)
_tab_lag2_note = (
    r'\textit{Notes:} Source: analysis\_v2.py (factor\_reg\_Lag2\_adj.xlsx). Returns measured in months '
    r'$t+4$ through $t+6$ (an additional one-quarter formation gap beyond the main 2-month '
    r'specification). Value-weighted industry-time-adjusted \RRR{} quartile portfolios. Q1 = highest '
    r'adj.\ \RRR{}, Q4 = lowest, Q1$-$Q4 = long-short. All other construction is identical to the main '
    r'specification. OLS with Newey-West standard errors (automatic bandwidth, Newey and West 1994). Standard errors in parentheses; '
    r'exact $p$-values in brackets for alpha estimates. No significance stars are used.'
)
_write_tabular(
    _build_panel_table(_reg_lag2, _rrr_port_labels, _tab_lag2_note),
    os.path.join(TABLE_DIR, 'tab_robustness_lag2.tex')
)


# ============================================================================
# TABLE (NEW, item 1): Headline Summary — ADJ RRR long-short + CAPM sample-market check
# ============================================================================
print("[Table] Headline summary (ADJ RRR long-short alpha + CAPM sample-market check)...")

_capm = pd.read_excel(os.path.join(OUTPUT_DIR, 'capm_vs_sp500.xlsx')).iloc[0]
_hl3, _hl5 = _reg_rrr_adj['Q1-Q4_FF3'], _reg_rrr_adj['Q1-Q4_FF5']

_tab_hl_lines = []
_tab_hl_lines.append('\\begin{threeparttable}')
_tab_hl_lines.append('\\footnotesize')
_tab_hl_lines.append('\\begin{tabular}{lrrr}')
_tab_hl_lines.append('\\toprule')
_tab_hl_lines.append(
    '\\multicolumn{4}{l}{\\textbf{Panel A: Adjusted \\RRR{} Long-Short Portfolio (Q1$-$Q4, $k=2$-month formation lag)}} \\\\'
)
_tab_hl_lines.append('\\midrule')
_tab_hl_lines.append(' & $\\alpha$ (\\%/mo) & $R^2$ & $N$ \\\\')
_tab_hl_lines.append(f"FF3 & {_fmt_coef(_hl3['alpha']*100)} & {fmt(_hl3['r2'],4)} & {fmt_int(_hl3['n_obs'])} \\\\")
_tab_hl_lines.append(f" & ({fmt(_hl3['se_alpha']*100,4)}) {_fmt_pval(_hl3['alpha_p'])} & & \\\\")
_tab_hl_lines.append(f"FF5 & {_fmt_coef(_hl5['alpha']*100)} & {fmt(_hl5['r2'],4)} & {fmt_int(_hl5['n_obs'])} \\\\")
_tab_hl_lines.append(f" & ({fmt(_hl5['se_alpha']*100,4)}) {_fmt_pval(_hl5['alpha_p'])} & & \\\\")
_tab_hl_lines.append('\\midrule')
_tab_hl_lines.append(
    '\\multicolumn{4}{l}{\\textbf{Panel B: CAPM, Sample Value-Weighted Market vs.\\ S\\&P 500}} \\\\'
)
_tab_hl_lines.append('\\midrule')
_tab_hl_lines.append(' & Coefficient & $t$-stat & \\\\')
_tab_hl_lines.append(f"$\\alpha$ (\\%/mo) & {_fmt_coef(_capm['alpha_month']*100)} & ${fmt(_capm['alpha_t'],2)}$ & \\\\")
_tab_hl_lines.append(f" & ({fmt(_capm['alpha_se']*100,4)}) & {_fmt_pval(_capm['alpha_p'])} & \\\\")
_tab_hl_lines.append(f"$\\beta$ & {_fmt_coef(_capm['beta'])} & ${fmt(_capm['beta_t'],2)}$ & \\\\")
_tab_hl_lines.append(f" & ({fmt(_capm['beta_se'],4)}) & & \\\\")
_tab_hl_lines.append(f"$R^2$ & {fmt(_capm['r2'],4)} & & \\\\")
_tab_hl_lines.append(f"$N$ (months) & {fmt_int(_capm['n_obs'])} & & \\\\")
_tab_hl_lines.append('\\bottomrule')
_tab_hl_lines.append('\\end{tabular}')
_tab_hl_lines.append('\\begin{tablenotes}')
_tab_hl_lines.append('\\small')
_tab_hl_lines.append(
    r'\item \textit{Notes:} Panel A source: analysis\_v2.py canonical portfolio pipeline '
    r'(build\_holding\_panel $\to$ build\_portfolio\_returns $\to$ run\_factor\_regressions). '
    r'Value-weighted, industry-time-adjusted \RRR{} quartile long-short portfolio, formed 2 months '
    r'after quarter-end and held 3 months. OLS with Newey-West standard errors (automatic bandwidth, '
    r'Newey and West 1994); standard '
    r'errors in parentheses, exact $p$-values in brackets, no significance stars. '
    r'Panel B source: analysis\_v2.py (capm\_vs\_sp500.xlsx). OLS of the sample'"'"'s own '
    r'value-weighted market portfolio (excess return) on the S\&P 500 (excess return), Newey-West (HAC) '
    r'standard errors, 4 lags; reported to verify the sample portfolio tracks a standard broad-market '
    r'benchmark. The CAPM alpha is small and statistically indistinguishable from zero, as expected for '
    r'a diversified market-cap-weighted benchmark portfolio; the beta is close to but somewhat above 1, '
    r'consistent with the sample tilting toward a subset of (four) GICS sectors.'
)
_tab_hl_lines.append('\\end{tablenotes}')
_tab_hl_lines.append('\\end{threeparttable}')
_write_tabular(_tab_hl_lines, os.path.join(TABLE_DIR, 'tab_headline_summary.tex'))


# ============================================================================
# TABLE (NEW, item 2): QMJ / BAB Factor Regressions — the decisive confound test
# ============================================================================
print("[Table] QMJ / BAB factor regressions (identification_battery.py Task 1)...")

_qmj_bab = pd.read_excel(IDBAT_XLSX, sheet_name='T1_QMJ_BAB')
_QMJ_BAB_LABELS = {
    'FF3 (ref)': 'FF3 (reference)', 'FF5 (ref)': 'FF5 (reference)',
    'FF3+QMJ': 'FF3 + QMJ', 'FF5+QMJ': 'FF5 + QMJ',
    'FF5+BAB': 'FF5 + BAB', 'FF5+QMJ+BAB': 'FF5 + QMJ + BAB',
}
_tab_qmj_lines = []
_tab_qmj_lines.append('\\begin{threeparttable}')
_tab_qmj_lines.append('\\footnotesize')
_tab_qmj_lines.append('\\begin{tabular}{lrrrrr}')
_tab_qmj_lines.append('\\toprule')
_tab_qmj_lines.append('Specification & $\\alpha$ (\\%/mo) & $\\beta_{QMJ}$ & $\\beta_{BAB}$ & $R^2$ & $N$ \\\\')
_tab_qmj_lines.append('\\midrule')
for _, _r in _qmj_bab.iterrows():
    _label = _QMJ_BAB_LABELS.get(_r['Spec'], _r['Spec'])
    _qmj_cell = _fmt_coef(_r['beta_QMJ']) if pd.notna(_r['beta_QMJ']) else '--'
    _bab_cell = _fmt_coef(_r['beta_BAB']) if pd.notna(_r['beta_BAB']) else '--'
    _tab_qmj_lines.append(
        f"{_label} & {_fmt_coef(_r['Alpha_pct_mo'])} & {_qmj_cell} & {_bab_cell} & "
        f"{fmt(_r['R2'],4)} & {fmt_int(_r['N'])} \\\\"
    )
    _qmj_t = f"(${fmt(_r['t_QMJ'],2)}$)" if pd.notna(_r['t_QMJ']) else ''
    _bab_t = f"(${fmt(_r['t_BAB'],2)}$)" if pd.notna(_r['t_BAB']) else ''
    _tab_qmj_lines.append(f" & ({fmt(_r['Alpha_se_pct'],4)}) & {_qmj_t} & {_bab_t} & & \\\\")
    _tab_qmj_lines.append(f" & {_fmt_pval(_r['Alpha_p'])} & & & & \\\\")
_tab_qmj_lines.append('\\bottomrule')
_tab_qmj_lines.append('\\end{tabular}')
_tab_qmj_lines.append('\\begin{tablenotes}')
_tab_qmj_lines.append('\\small')
_qmj_bab_headline_t = _qmj_bab.loc[_qmj_bab['Spec'] == 'FF5+QMJ+BAB', 'Alpha_t'].values[0]
_tab_qmj_lines.append(
    r'\item \textit{Notes:} Source: identification\_battery.py Task 1. Value-weighted adjusted-\RRR{} '
    r'long-short portfolio (Q1$-$Q4). QMJ = AQR Quality-Minus-Junk factor; BAB = AQR '
    r'Betting-Against-Beta factor (both US, monthly). $\alpha$ row: standard error in parentheses '
    r'below, exact $p$-value in brackets on the next row; no significance stars. $\beta_{QMJ}$/'
    r'$\beta_{BAB}$ rows show $t$-statistics in parentheses (loadings are controls here, not the focal '
    r'test). Decision rule (pre-registered): the long-short alpha survives this confound battery if '
    r'the FF5+QMJ+BAB $\alpha$ $t$-statistic remains $\geq 2.0$; it does '
    rf'($t={_qmj_bab_headline_t:.2f}$), so the RRR premium is not subsumed by quality or low-beta '
    r'exposure.'
)
_tab_qmj_lines.append('\\end{tablenotes}')
_tab_qmj_lines.append('\\end{threeparttable}')
_write_tabular(_tab_qmj_lines, os.path.join(TABLE_DIR, 'tab_qmj_bab.tex'))


# ============================================================================
# TABLE (NEW, item 3): Consolidated Robustness Battery
# ============================================================================
print("[Table] Consolidated robustness battery...")

_t2_boot = pd.read_excel(DIAG_XLSX, sheet_name='T2_Bootstrap')
_t2_sub = pd.read_excel(DIAG_XLSX, sheet_name='T2_Subperiod')
_t7_cost = pd.read_excel(DIAG_XLSX, sheet_name='T7_NetOfCost')

# --- 4-quarter-trailing RRR (LR_RRR): port_lr_rrr_q was built in Step 0b above (the
# comment there promises persistence to lr_rrr_robustness_stage5.xlsx) but was never
# actually regressed or written -- completed here.
print("  Computing 4-quarter-trailing RRR (LR_RRR) robustness regressions...")
_reg_lr_rrr = _run_inline_factor_reg(port_lr_rrr_q, ff_factors)
_lr_rrr_export_rows = [
    {'key': _key, 'alpha': _r['alpha'], 'alpha_t': _r['alpha_t'], 'alpha_p': _r['alpha_p'],
     'se_alpha': _r['se_alpha'], 'r2': _r['r2'], 'n_obs': _r['n_obs']}
    for _key, _r in _reg_lr_rrr.items()
]
pd.DataFrame(_lr_rrr_export_rows).to_excel(
    os.path.join(OUTPUT_DIR, 'lr_rrr_robustness_stage5.xlsx'), index=False
)
print("  Written: lr_rrr_robustness_stage5.xlsx")

# --- Consumer-Discretionary-only: robustness_diagnostics.py computes this (Task 3,
# function task3_consumer_discretionary_only(), called at line ~1231) but its result
# is DROPPED from the committed workbook -- every other task (1,2,4,5,6,7,8,9) has a
# matching entry in the `sheets = {...}` export dict (robustness_diagnostics.py lines
# ~1245-1263); task3's `results["task3"]` is computed but never added to that dict,
# so there is no T3_* sheet. This looks like an oversight in the already-committed,
# locked script, not an intentional exclusion, and robustness_diagnostics.py must not
# be edited in this stage. fix_cd_only_export.py re-runs Task 3's exact sort and
# factor-regression logic (imports load_base_data + task3_consumer_discretionary_only
# read-only; does not reimplement anything) and persists the result to
# output/cd_only_alpha.xlsx (sheet CD_only_alpha), with a guardrail assert that the
# persisted FF3/FF5 alpha and t-stat match the originally observed console figures.
# Read that workbook here rather than hand-typing the figures.
_cd_only_alpha = pd.read_excel(os.path.join(OUTPUT_DIR, 'cd_only_alpha.xlsx'), sheet_name='CD_only_alpha')
_cd_only_ff3_row = _cd_only_alpha[_cd_only_alpha['Model'] == 'FF3'].iloc[0]
_cd_only_ff5_row = _cd_only_alpha[_cd_only_alpha['Model'] == 'FF5'].iloc[0]
_CD_ONLY_FF3_ALPHA = _cd_only_ff3_row['Alpha_pct_per_month']
_CD_ONLY_FF3_T = _cd_only_ff3_row['t_stat']
_CD_ONLY_FF3_P = _cd_only_ff3_row['p_value']
_CD_ONLY_N = int(_cd_only_ff3_row['N_months'])
_CD_ONLY_FF5_ALPHA = _cd_only_ff5_row['Alpha_pct_per_month']
_CD_ONLY_FF5_T = _cd_only_ff5_row['t_stat']
_CD_ONLY_FF5_P = _cd_only_ff5_row['p_value']
_cd_only_ff3_se = _se_from_t(_CD_ONLY_FF3_ALPHA, _CD_ONLY_FF3_T)
_cd_only_ff5_se = _se_from_t(_CD_ONLY_FF5_ALPHA, _CD_ONLY_FF5_T)


def _rob_row(label, ff3_alpha, ff3_se, ff3_p, ff5_alpha, ff5_se, ff5_p, n):
    """Build a 2-line (coef row + SE/p row) entry for the consolidated robustness
    table. ff3_p/ff5_p may be a float p-value or a pre-formatted display string
    (e.g. '[<.001]') when only a rounded value is available from source."""
    p3 = ff3_p if isinstance(ff3_p, str) else _fmt_pval(ff3_p)
    p5 = ff5_p if isinstance(ff5_p, str) else _fmt_pval(ff5_p)
    n_cell = fmt_int(n) if n is not None else ''
    line1 = f"{label} & {_fmt_coef(ff3_alpha)} & {_fmt_coef(ff5_alpha)} & {n_cell} \\\\"
    line2 = f" & ({fmt(ff3_se,4)}) {p3} & ({fmt(ff5_se,4)}) {p5} & \\\\"
    return [line1, line2]


_rob_lines = []
_rob_lines.append('\\begin{threeparttable}')
_rob_lines.append('\\footnotesize')
_rob_lines.append('\\resizebox{\\textwidth}{!}{%')
_rob_lines.append('\\begin{tabular}{lrrr}')
_rob_lines.append('\\toprule')
_rob_lines.append('Specification & FF3 $\\alpha$ (\\%/mo) & FF5 $\\alpha$ (\\%/mo) & $N$ \\\\')
_rob_lines.append('\\midrule')

_b3, _b5 = _reg_rrr_adj['Q1-Q4_FF3'], _reg_rrr_adj['Q1-Q4_FF5']
_rob_lines += _rob_row('Baseline (main specification)',
                        _b3['alpha']*100, _b3['se_alpha']*100, _b3['alpha_p'],
                        _b5['alpha']*100, _b5['se_alpha']*100, _b5['alpha_p'], _b3['n_obs'])

_boot3 = _t2_boot[(_t2_boot['spec'] == 'FF3') & (_t2_boot['block_length_months'] == 6)].iloc[0]
_boot5 = _t2_boot[(_t2_boot['spec'] == 'FF5') & (_t2_boot['block_length_months'] == 6)].iloc[0]
_rob_lines.append(
    f"Block bootstrap (6-month blocks, $n=5{{,}}000$) & {_fmt_coef(_boot3['boot_mean_pct'])} & "
    f"{_fmt_coef(_boot5['boot_mean_pct'])} & {fmt_int(_boot3['n_obs'])} \\\\"
)
_rob_lines.append(
    f" & 95\\% CI $[{fmt(_boot3['ci_lo_pct'],4)},\\,{fmt(_boot3['ci_hi_pct'],4)}]$ & "
    f"95\\% CI $[{fmt(_boot5['ci_lo_pct'],4)},\\,{fmt(_boot5['ci_hi_pct'],4)}]$ & \\\\"
)

for _period, _label in [('first_half', 'First-half subperiod (2017-09 to 2021-04)'),
                         ('second_half', 'Second-half subperiod (2021-05 to 2024-12)')]:
    _s3 = _t2_sub[(_t2_sub['period'] == _period) & (_t2_sub['spec'] == 'FF3')].iloc[0]
    _s5 = _t2_sub[(_t2_sub['period'] == _period) & (_t2_sub['spec'] == 'FF5')].iloc[0]
    _rob_lines += _rob_row(_label,
                            _s3['alpha_pct'], _se_from_t(_s3['alpha_pct'], _s3['t']), _s3['p'],
                            _s5['alpha_pct'], _se_from_t(_s5['alpha_pct'], _s5['t']), _s5['p'], _s3['n_obs'])

_rob_lines += _rob_row('Consumer Discretionary sector only (re-quartiled within sector)',
                        _CD_ONLY_FF3_ALPHA, _cd_only_ff3_se, _CD_ONLY_FF3_P,
                        _CD_ONLY_FF5_ALPHA, _cd_only_ff5_se, _CD_ONLY_FF5_P, _CD_ONLY_N)

# --- Excluding COVID-19 period: this row (and the 5-month-formation-lag row below)
# existed in the committed tab_robustness_consolidated.tex (added by hand, Paper_LaTeX
# commit "Add missing COVID-exclusion and 5-month-lag rows...") with NO corresponding
# code in this script -- the same class of gap as the firm+quarter-FE Sloan row fixed
# above, discovered while verifying that fix. The underlying regression already exists
# in this script (_reg_nc, computed above for tab_robustness_nocovid.tex, Table 8);
# reuse it here instead of hand-typing the figures a second time.
_nc3, _nc5 = _reg_nc['Q1-Q4_FF3'], _reg_nc['Q1-Q4_FF5']
_rob_lines += _rob_row('Excluding COVID-19 period (2020Q1--2021Q2)',
                        _nc3['alpha']*100, _nc3['se_alpha']*100, _nc3['alpha_p'],
                        _nc5['alpha']*100, _nc5['se_alpha']*100, _nc5['alpha_p'], _nc3['n_obs'])

_e3, _e5 = _reg_ew['Q1-Q4_FF3'], _reg_ew['Q1-Q4_FF5']
_rob_lines += _rob_row('Equal-weighted (vs.\\ value-weighted baseline)',
                        _e3['alpha']*100, _e3['se_alpha']*100, _e3['alpha_p'],
                        _e5['alpha']*100, _e5['se_alpha']*100, _e5['alpha_p'], _e3['n_obs'])

# --- 5-month formation lag: same gap as the COVID-exclusion row above (hand-added to
# the committed .tex, no code path in this script). _reg_lag2 (factor_reg_Lag2_adj.xlsx,
# already loaded above for tab_robustness_lag2.tex) is reused here rather than
# hand-typing the figures a second time.
_lg3, _lg5 = _reg_lag2['Q1-Q4_FF3'], _reg_lag2['Q1-Q4_FF5']
_rob_lines += _rob_row('5-month formation lag (vs.\\ 2-month baseline)',
                        _lg3['alpha']*100, _lg3['se_alpha']*100, _lg3['alpha_p'],
                        _lg5['alpha']*100, _lg5['se_alpha']*100, _lg5['alpha_p'], _lg3['n_obs'])

for _bps in [25, 50, 100]:
    _c3 = _t7_cost[(_t7_cost['Spec'] == 'FF3') & (_t7_cost['Cost_bps'] == _bps)].iloc[0]
    _c5 = _t7_cost[(_t7_cost['Spec'] == 'FF5') & (_t7_cost['Cost_bps'] == _bps)].iloc[0]
    _tag = ' (primary)' if _bps == 50 else ''
    _rob_lines += _rob_row(f'Net of {_bps}bp round-trip trading cost{_tag}',
                            _c3['Net_alpha_pct'], _se_from_t(_c3['Net_alpha_pct'], _c3['Net_t']), _c3['Net_p'],
                            _c5['Net_alpha_pct'], _se_from_t(_c5['Net_alpha_pct'], _c5['Net_t']), _c5['Net_p'], None)

_pp3, _pp5 = _reg_placebo['High-Low_FF3'], _reg_placebo['High-Low_FF5']
_rob_lines += _rob_row('Same-growth placebo (High$-$Low \\RRR{}, top-growth-quartile firms only)',
                        _pp3['alpha']*100, _pp3['se_alpha']*100, _pp3['alpha_p'],
                        _pp5['alpha']*100, _pp5['se_alpha']*100, _pp5['alpha_p'], _pp3['n_obs'])

_l3, _l5 = _reg_lr_rrr['Q1-Q4_FF3'], _reg_lr_rrr['Q1-Q4_FF5']
_rob_lines += _rob_row('4-quarter-trailing \\RRR{} definition (industry-time adjusted)',
                        _l3['alpha']*100, _l3['se_alpha']*100, _l3['alpha_p'],
                        _l5['alpha']*100, _l5['se_alpha']*100, _l5['alpha_p'], _l3['n_obs'])

_rob_lines.append('\\bottomrule')
_rob_lines.append('\\end{tabular}%')
_rob_lines.append('}')
_rob_lines.append('\\begin{tablenotes}')
_rob_lines.append('\\small')
_rob_lines.append(
    r'\item \textit{Notes:} Value-weighted, industry-time-adjusted \RRR{} quartile long-short '
    r'portfolio (Q1$-$Q4) under a battery of robustness variants; the baseline row reproduces the main '
    r'specification for reference. Each row after the label shows the coefficient on top and, on the '
    r'row below, the standard error in parentheses followed by the exact $p$-value in brackets (no '
    r'significance stars), except the bootstrap row, which reports a percentile 95\% confidence '
    r'interval in place of a parametric SE/$p$-value (the point estimate shown is the bootstrap mean; '
    r'results are qualitatively identical using 3- or 12-month blocks in place of the 6-month primary '
    r'block; source: robustness\_diagnostics.py Task 2). Consumer-Discretionary-only re-quartiles '
    r'\RRR{} within that sector'"'"'s cross-section each quarter (90 of 124 firms; 82.1 avg firms/quarter '
    r'in the resulting sort) rather than merely filtering the all-sector quartile assignment; source: '
    r'fix\_cd\_only\_export.py, output/cd\_only\_alpha.xlsx (re-runs robustness\_diagnostics.py Task 3'"'"'s '
    r'exact sort and factor-regression logic via a read-only import; that task computes this result but '
    r'never writes it to robustness\_diagnostics.py'"'"'s own committed workbook, so this companion '
    r'script persists it separately, with a guardrail assert against the originally observed figures; '
    r'see canonical\_run\_manifest.json). Excluding COVID-19 period drops 2020Q1--2021Q2 from portfolio '
    r'formation and factor regressions entirely (not merely a dummy control), reducing $N$ from 88 to 70 '
    r'months (source: this script'"'"'s _reg_nc, the same regression underlying '
    r'tab\_robustness\_nocovid.tex). Equal-weighted source: this script'"'"'s own inline reconstruction, matching '
    r'analysis\_v2.py phase4\_additional\_tests() section 4.3 exactly. 5-month formation lag pushes '
    r'portfolio formation to five months after the signal quarter-end (one additional quarter beyond the '
    r'2-month baseline), holding all other construction fixed; $N$ falls to 85 months because the extra '
    r'quarter'"'"'s lag shortens the usable return series at both ends of the sample (source: this script'"'"'s '
    r'_reg_lag2, the same regression underlying tab\_robustness\_lag2.tex, factor\_reg\_Lag2\_adj.xlsx). '
    r'Trading-cost assumption follows '
    r'Novy-Marx and Velikov (2016, \textit{Review of Financial Studies} 29(1):104--147); 50bp round-trip '
    r'is the primary assumption for "large, liquid stocks," 25bp/100bp are sensitivity bounds; net '
    r'alpha nets a cost drag against the gross (baseline) alpha using the strategy'"'"'s realized '
    r'quarterly turnover (source: robustness\_diagnostics.py Task 7). Same-growth placebo restricts to '
    r'the top revenue-growth quartile each quarter and splits by \RRR{} at the median within that '
    r'subsample; reported as a single row per author decision (demoted from a standalone table). '
    r'4-quarter-trailing \RRR{} redefines the retention signal against the revenue base four quarters '
    r'(one year) prior rather than the prior quarter, industry-time adjusted the same way as the main '
    r'signal; all other construction (timing, weighting) is unchanged (source: this script, '
    r'lr\_rrr\_robustness\_stage5.xlsx, ported forward from the LR\_RRR column already computed in the '
    r'canonical panel).'
)
_rob_lines.append('\\end{tablenotes}')
_rob_lines.append('\\end{threeparttable}')
_write_tabular(_rob_lines, os.path.join(TABLE_DIR, 'tab_robustness_consolidated.tex'))


# ============================================================================
# TABLE (NEW, item 4): Concentration — effective-N/HHI, exclude-top-K, Amazon exclusion
# ============================================================================
print("[Table] Concentration diagnostics (effective-N/HHI, exclude-top-k, Amazon exclusion)...")

_t1_effn_q1 = pd.read_excel(DIAG_XLSX, sheet_name='T1_EffN_Q1')
_t1_effn_q4 = pd.read_excel(DIAG_XLSX, sheet_name='T1_EffN_Q4')
_t1_excl = pd.read_excel(DIAG_XLSX, sheet_name='T1_ExcludeTopK')
_amzn = pd.read_excel(os.path.join(OUTPUT_DIR, 'exclude_amazon_robustness.xlsx'), sheet_name='Summary_Comparison')

_tab_conc_lines = []
_tab_conc_lines.append('\\begin{threeparttable}')
_tab_conc_lines.append('\\footnotesize')
_tab_conc_lines.append('\\begin{tabular}{lrrrr}')
_tab_conc_lines.append('\\toprule')
_tab_conc_lines.append(
    '\\multicolumn{5}{l}{\\textbf{Panel A: Effective-$N$ / Herfindahl Concentration by Quarter (long and short legs)}} \\\\'
)
_tab_conc_lines.append('\\midrule')
_tab_conc_lines.append(' & Mean & Median & Min & Max \\\\')
for _label, _series in [
    ('Q1 (long leg) HHI', _t1_effn_q1['HHI']),
    ('Q1 (long leg) Effective $N$', _t1_effn_q1['EFFECTIVE_N']),
    ('Q4 (short leg) HHI', _t1_effn_q4['HHI']),
    ('Q4 (short leg) Effective $N$', _t1_effn_q4['EFFECTIVE_N']),
]:
    _tab_conc_lines.append(
        f"{_label} & {fmt(_series.mean(),4)} & {fmt(_series.median(),4)} & "
        f"{fmt(_series.min(),4)} & {fmt(_series.max(),4)} \\\\"
    )
_tab_conc_lines.append('\\midrule')
_tab_conc_lines.append(
    '\\multicolumn{5}{l}{\\textbf{Panel B: Long-Short Alpha Excluding the Top-$K$ Formation-Cap Names from Q1}} \\\\'
)
_tab_conc_lines.append('\\midrule')
_tab_conc_lines.append(' & FF3 $\\alpha$ (\\%/mo) & FF5 $\\alpha$ (\\%/mo) & Avg.\\ \\% of Q1 excl.\\ & \\\\')
for _, _r in _t1_excl.iterrows():
    _tab_conc_lines.append(
        f"{_r['Specification']} & {_fmt_coef(_r['FF3_alpha_pct'])} & {_fmt_coef(_r['FF5_alpha_pct'])} & "
        f"{fmt(_r['avg_pct_weight_excluded'],2)}\\% & \\\\"
    )
    _tab_conc_lines.append(
        f" & ({fmt(_se_from_t(_r['FF3_alpha_pct'], _r['FF3_t']),4)}) {_fmt_pval(_r['FF3_p'])} & "
        f"({fmt(_se_from_t(_r['FF5_alpha_pct'], _r['FF5_t']),4)}) {_fmt_pval(_r['FF5_p'])} & & \\\\"
    )
_tab_conc_lines.append('\\midrule')
_tab_conc_lines.append(
    '\\multicolumn{5}{l}{\\textbf{Panel C: Amazon-Specific Full-Period Exclusion (separate from Panel B)}} \\\\'
)
_tab_conc_lines.append('\\midrule')
for _idx in [0, 1]:
    _a = _amzn.iloc[_idx]
    _tab_conc_lines.append(
        f"{_a['Specification']} & {_fmt_coef(_a['FF3_alpha_pct'])} & {_fmt_coef(_a['FF5_alpha_pct'])} & & \\\\"
    )
    _tab_conc_lines.append(
        f" & ({fmt(_se_from_t(_a['FF3_alpha_pct'], _a['FF3_t']),4)}) {_fmt_pval(_a['FF3_p'])} & "
        f"({fmt(_se_from_t(_a['FF5_alpha_pct'], _a['FF5_t']),4)}) {_fmt_pval(_a['FF5_p'])} & & \\\\"
    )
_tab_conc_lines.append('\\bottomrule')
_tab_conc_lines.append('\\end{tabular}')
_tab_conc_lines.append('\\begin{tablenotes}')
_tab_conc_lines.append('\\small')
_tab_conc_lines.append(
    r'\item \textit{Notes:} Source: robustness\_diagnostics.py (Panels A, B) and '
    r'exclude\_amazon\_robustness.py (Panel C). Panel A summarizes, across the 30 sample quarters, the '
    r'Herfindahl index (HHI, sum of squared formation-cap weights) and effective $N$ ($1/\text{HHI}$) '
    r'of the Q1 (long) and Q4 (short) legs. Panel B drops the top-$K$ formation-cap names from the long '
    r'leg each quarter and re-forms the long-short alpha; standard error in parentheses, exact '
    r'$p$-value in brackets, no significance stars. Panel C is a separate, full-period robustness check '
    r'that removes AMZN US EQUITY from the investable universe entirely before any sorting (not merely '
    r'excluded when it lands in Q1); it is not a per-quarter top-$K$ exclusion and should not be read as '
    r'directly comparable to Panel B. The FF3 alpha and its $t$-statistic move modestly ($t$: 2.94 '
    r'$\to$ 2.80), but the FF5 alpha and its $t$-statistic move more ($t$: 3.29 $\to$ 2.59) -- a '
    r'meaningfully closer call than the baseline, not a robustly unchanged result.'
)
_tab_conc_lines.append('\\end{tablenotes}')
_tab_conc_lines.append('\\end{threeparttable}')
_write_tabular(_tab_conc_lines, os.path.join(TABLE_DIR, 'tab_concentration.tex'))


# ============================================================================
# TABLE (NEW, item 5): Full-Quartile Monotonicity
# ============================================================================
print("[Table] Full-quartile monotonicity...")

_t5_mono = pd.read_excel(DIAG_XLSX, sheet_name='T5_QuartileAlphas')
_t5_mono = _t5_mono[_t5_mono['Portfolio'] != 'MONOTONIC_Q1_TO_Q4']

_tab_mono_lines = []
_tab_mono_lines.append('\\begin{threeparttable}')
_tab_mono_lines.append('\\footnotesize')
_tab_mono_lines.append('\\begin{tabular}{lrr}')
_tab_mono_lines.append('\\toprule')
_tab_mono_lines.append(' & FF3 $\\alpha$ (\\%/mo) & FF5 $\\alpha$ (\\%/mo) \\\\')
_tab_mono_lines.append('\\midrule')
for _q in ['Q1', 'Q2', 'Q3', 'Q4']:
    _r3 = _t5_mono[(_t5_mono['Spec'] == 'FF3') & (_t5_mono['Portfolio'] == _q)].iloc[0]
    _r5 = _t5_mono[(_t5_mono['Spec'] == 'FF5') & (_t5_mono['Portfolio'] == _q)].iloc[0]
    _tab_mono_lines.append(f"{_q} & {_fmt_coef(_r3['Alpha_pct'])} & {_fmt_coef(_r5['Alpha_pct'])} \\\\")
    _se3 = _se_from_t(_r3['Alpha_pct'], _r3['t'])
    _se5 = _se_from_t(_r5['Alpha_pct'], _r5['t'])
    _tab_mono_lines.append(f" & ({fmt(_se3,4)}) {_fmt_pval(_r3['p'])} & ({fmt(_se5,4)}) {_fmt_pval(_r5['p'])} \\\\")
_tab_mono_lines.append('\\bottomrule')
_tab_mono_lines.append('\\end{tabular}')
_tab_mono_lines.append('\\begin{tablenotes}')
_tab_mono_lines.append('\\small')
_tab_mono_lines.append(
    r'\item \textit{Notes:} Source: robustness\_diagnostics.py Task 5. Individual-quartile factor-model '
    r'alphas for the value-weighted, industry-time-adjusted \RRR{} sort; the long-short row (Q1$-$Q4) '
    r'is reported separately in the paper'"'"'s main adjusted-\RRR{} long-short alpha table and not '
    r'repeated here. Standard error in parentheses, exact $p$-value in brackets, no significance stars. '
    r'Point estimates decline monotonically from Q1 to Q4 under both FF3 and FF5; statistical '
    r'significance concentrates in Q4 (and, under FF5, marginally in Q1), with Q2 and Q3 individually '
    r'indistinguishable from zero.'
)
_tab_mono_lines.append('\\end{tablenotes}')
_tab_mono_lines.append('\\end{threeparttable}')
_write_tabular(_tab_mono_lines, os.path.join(TABLE_DIR, 'tab_quartile_monotonicity.tex'))


# ============================================================================
# TABLE (NEW, item 7): H2a / H2b Asymmetry
# ============================================================================
print("[Table] H2a/H2b asymmetry (downside-risk bootstrap + up/down-market interaction)...")

_t8a = pd.read_excel(DIAG_XLSX, sheet_name='T8_H2a_Bootstrap')
_t8b_cont = pd.read_excel(DIAG_XLSX, sheet_name='T8_H2b_Continuous')
_t8b_dummy = pd.read_excel(DIAG_XLSX, sheet_name='T8_H2b_Dummy')

_H2A_LABELS = {
    'downside_beta (Q1-Q4)': 'Downside $\\beta$ (Q1$-$Q4)',
    'ann_vol (Q1-Q4)': 'Annualized volatility (Q1$-$Q4)',
    'ann_sharpe (Q1-Q4)': 'Annualized Sharpe (Q1$-$Q4)',
}
_tab_h2_lines = []
_tab_h2_lines.append('\\begin{threeparttable}')
_tab_h2_lines.append('\\footnotesize')
_tab_h2_lines.append('\\textbf{Panel A: H2a, Bootstrap Q1 vs.\\ Q4 Differences (downside/insurance)}')
_tab_h2_lines.append('')
_tab_h2_lines.append('\\begin{tabular}{lrrrl}')
_tab_h2_lines.append('\\toprule')
_tab_h2_lines.append('Metric & Point est.\\ & Boot.\\ SE & 95\\% CI & Distinguishable from 0 \\\\')
_tab_h2_lines.append('\\midrule')
for _, _r in _t8a.iterrows():
    _label = _H2A_LABELS.get(_r['Metric'], _r['Metric'])
    _sig = 'Yes' if _r['Distinguishable_from_zero_95pct'] else 'No'
    _tab_h2_lines.append(
        f"{_label} & {_fmt_coef(_r['Point_estimate'])} & {fmt(_r['Boot_SE'],4)} & "
        f"$[{fmt(_r['CI_lo'],4)},\\,{fmt(_r['CI_hi'],4)}]$ & {_sig} \\\\"
    )
_tab_h2_lines.append('\\bottomrule')
_tab_h2_lines.append('\\end{tabular}')
_tab_h2_lines.append('')
_tab_h2_lines.append('\\vspace{6pt}')
_tab_h2_lines.append('')
_tab_h2_lines.append('\\textbf{Panel B: H2b, Up-Market / Down-Market Signal Sensitivity (panel regression)}')
_tab_h2_lines.append('')
_tab_h2_lines.append('\\begin{tabular}{lrr}')
_tab_h2_lines.append('\\toprule')
_tab_h2_lines.append(' & (1) Continuous \\RRR{} & (2) Q1-vs-Q4 dummy \\\\')
_tab_h2_lines.append('\\midrule')
_H2B_ROLE_MAP = [
    ('const', 'const', 'Constant'),
    ('MKT_RF', 'MKT_RF', 'Market excess return'),
    ('ADJ_RRR_PCT', 'Q1_DUMMY', 'Signal (Adj.\\ RRR\\ /\\ Q1 dummy)'),
    ('UP', 'UP', 'Up-market dummy'),
    ('ADJ_RRR_PCT_x_MKT', 'Q1_DUMMY_x_MKT', 'Signal $\\times$ Market'),
    ('ADJ_RRR_PCT_x_MKT_x_UP', 'Q1_DUMMY_x_MKT_x_UP', 'Signal $\\times$ Market $\\times$ Up (focal)'),
]
for _cont_var, _dummy_var, _disp in _H2B_ROLE_MAP:
    _rc = _t8b_cont[_t8b_cont['Variable'] == _cont_var]
    _rd = _t8b_dummy[_t8b_dummy['Variable'] == _dummy_var]
    _cc = _fmt_coef(_rc.iloc[0]['Coef']) if len(_rc) else ''
    _dc = _fmt_coef(_rd.iloc[0]['Coef']) if len(_rd) else ''
    _tab_h2_lines.append(f"{_disp} & {_cc} & {_dc} \\\\")
    _cse = f"({fmt(_rc.iloc[0]['SE'],4)}) {_fmt_pval(_rc.iloc[0]['p'])}" if len(_rc) else ''
    _dse = f"({fmt(_rd.iloc[0]['SE'],4)}) {_fmt_pval(_rd.iloc[0]['p'])}" if len(_rd) else ''
    _tab_h2_lines.append(f" & {_cse} & {_dse} \\\\")
_tab_h2_lines.append('\\bottomrule')
_tab_h2_lines.append('\\end{tabular}')
_tab_h2_lines.append('\\begin{tablenotes}')
_tab_h2_lines.append('\\small')
_tab_h2_lines.append(
    r'\item \textit{Notes:} Source: robustness\_diagnostics.py Task 8. Panel A: block-bootstrap ($n=5{,}000$) '
    r'Q1-minus-Q4 differences in downside beta, annualized volatility and annualized Sharpe ratio; '
    r'reported honestly including the non-significant downside-beta difference (95\% CI includes 0). '
    r'Panel B: panel regressions of monthly excess return on market excess return, an \RRR{} signal '
    r'(continuous industry-time-adjusted \RRR{}, or a Q1-vs-Q4 dummy), an up-market dummy (UP), and '
    r'their interactions; two-way (firm, month) clustered standard errors '
    r'(Cameron-Gelbach-Miller 2011). The focal H2b term is the triple interaction '
    r'(Signal $\times$ Market $\times$ Up); it is small and statistically indistinguishable from zero '
    r'in both specifications, reported here in full rather than omitted. Standard errors in '
    r'parentheses, exact $p$-values in brackets, no significance stars.'
)
_tab_h2_lines.append('\\end{tablenotes}')
_tab_h2_lines.append('\\end{threeparttable}')
_write_tabular(_tab_h2_lines, os.path.join(TABLE_DIR, 'tab_h2ab_asymmetry.tex'))


# ============================================================================
# TABLE (NEW, item 8): Sloan-Style Persistence — both legs, honest mixed result
# ============================================================================
print("[Table] Sloan-style persistence of revenue components (both legs)...")

_sloan = pd.read_excel(IDBAT_XLSX, sheet_name='T4_sloan_persistence')

# Fama-MacBeth confirmation of the diff test (identification_battery.py Task 4
# computes this via _persistence_regression's `fm` dict and prints it to console,
# but never appends it to the exported T4_sloan_persistence sheet).
# fix_fm_persistence_export.py re-runs the exact FM computation read-only and
# persists it to output/fm_persistence.xlsx, with a guardrail assert matching the
# originally observed console figures (diff=1.7845, t=5.92, T=30).
_fm_persist = pd.read_excel(os.path.join(OUTPUT_DIR, 'fm_persistence.xlsx'), sheet_name='T4_FM_persistence')

# Firm+quarter-FE robustness of the persistence test (firm_fe_persistence.py; see
# that script's docstring). This is a standalone companion script whose output has
# no code path into this table prior to this fix -- it was previously wired in by
# hand-editing the committed .tex file directly.
_firm_fe_persist = pd.read_excel(os.path.join(OUTPUT_DIR, 'firm_fe_persistence.xlsx'), sheet_name='firm_fe_persistence')

_SLOAN_DV_LABELS = {
    'RG_LEAD': 'Future revenue growth $RG_{t+1}$ (\\%)',
    'OPINC_ROA_LEAD': 'Future operating ROA $OpInc_{t+1}/Assets_t$ (\\%)',
}


def _pval_inline(pval, decimals=3):
    """Inline (bracket-free) p-value for embedding inside already-open $...$ math,
    e.g. 'p<.001' or 'p=.004'. Mirrors _fmt_pval's precision/floor rule."""
    if pd.isna(pval):
        return ''
    floor = 10 ** (-decimals)
    if pval < floor:
        return f"p<{floor:.{decimals}f}".replace('0.', '.', 1)
    s = f"{pval:.{decimals}f}"
    if s.startswith('0.'):
        s = s[1:]
    return f"p={s}"


def _signed_num(v, decimals=4):
    """Plain (non-LaTeX-macro) signed number for embedding inside already-open
    $...$ math, where a literal '-' renders correctly as a minus sign (unlike
    _fmt_coef's '$-$' macro, which is meant for use OUTSIDE an existing $...$ span)."""
    if pd.isna(v):
        return ''
    sign = '-' if v < 0 else ''
    return f"{sign}{abs(v):.{decimals}f}"


def _sloan_fe_verdict(b_ret, b_acq, diff_p, alpha=0.05):
    """Descriptive verdict for the firm+quarter-FE persistence row, derived from
    its own coefficients and firm-clustered diff p-value. Mirrors
    firm_fe_persistence.py's PASS / sign-only / FAIL logic (its survives_firmFE_gap
    column) but adds a 'REVERSES' category for a statistically significant sign
    flip (b_ret < b_acq with diff_p < alpha); firm_fe_persistence.py's own column
    folds that case into a plain 'FAIL' without flagging that the reversal is
    itself significant, which is exactly the paper's point about this row."""
    significant = diff_p < alpha
    if b_ret > b_acq:
        return 'PASS (ret>acq, diff sig.)' if significant else 'sign-only (ret>acq, diff n.s.)'
    return r'REVERSES (ret\textless{}acq, diff sig.)' if significant else 'FAIL (ret<=acq)'

_tab_sloan_lines = []
_tab_sloan_lines.append('\\begin{threeparttable}')
_tab_sloan_lines.append('\\footnotesize')
_tab_sloan_lines.append('\\resizebox{\\textwidth}{!}{%')
_tab_sloan_lines.append('\\begin{tabular}{llrrrl}')
_tab_sloan_lines.append('\\toprule')
_tab_sloan_lines.append(
    'Dependent variable & Sample & Retention $b$ & Acquisition $b$ & Diff.\\ (firm-cl.\\ $t$) & Criterion \\\\'
)
_tab_sloan_lines.append('\\midrule')
for _dv in ['RG_LEAD', 'OPINC_ROA_LEAD']:
    _tab_sloan_lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{_SLOAN_DV_LABELS[_dv]}}}}} \\\\")
    _sub = _sloan[_sloan['DV'] == _dv]
    for _, _r in _sub.iterrows():
        _tab_sloan_lines.append(
            f"\\quad & {_r['Winsor']} & {_fmt_coef(_r['b_retention'])} & {_fmt_coef(_r['b_acquisition'])} & "
            f"{_fmt_coef(_r['diff_ret_minus_acq_firmcl'])} (${fmt(_r['diff_t_firmcl'],2)}$) & {_r['criterion']} \\\\"
        )
        _se_ret = _se_from_t(_r['b_retention'], _r['t_retention'])
        _se_acq = _se_from_t(_r['b_acquisition'], _r['t_acquisition'])
        _tab_sloan_lines.append(
            f"\\quad & & ({fmt(_se_ret,4)}) & ({fmt(_se_acq,4)}) & "
            f"two-way $t={fmt(_r['diff_t_twoway'],2)}$ & \\\\"
        )
        _tab_sloan_lines.append(
            f"\\quad & & {_fmt_pval(_r['p_retention'])} & {_fmt_pval(_r['p_acquisition'])} & "
            f"{_fmt_pval(_r['diff_p_firmcl'])} & \\\\"
        )
    if _dv == 'OPINC_ROA_LEAD':
        # Firm+quarter-FE robustness row: strips out all time-invariant firm
        # characteristics, identifying the retention-vs-acquisition gap from
        # within-firm, quarter-to-quarter variation only (firm_fe_persistence.py).
        _fe_row = _firm_fe_persist[
            (_firm_fe_persist['DV'] == 'OPINC_ROA_LEAD') &
            (_firm_fe_persist['Winsor'] == '1/99-winsorized') &
            (_firm_fe_persist['FE'] == 'Firm + Quarter FE')
        ].iloc[0]
        _fe_verdict = _sloan_fe_verdict(_fe_row['b_retention'], _fe_row['b_acquisition'], _fe_row['diff_p_firmcl'])
        _tab_sloan_lines.append(
            f"\\quad & 1/99-w., firm + quarter FE & {_fmt_coef(_fe_row['b_retention'])} & "
            f"{_fmt_coef(_fe_row['b_acquisition'])} & "
            f"{_fmt_coef(_fe_row['diff_ret_minus_acq_firmcl'])} (${fmt(_fe_row['diff_t_firmcl'],2)}$) & {_fe_verdict} \\\\"
        )
        _se_ret_fe = _se_from_t(_fe_row['b_retention'], _fe_row['t_retention'])
        _se_acq_fe = _se_from_t(_fe_row['b_acquisition'], _fe_row['t_acquisition'])
        _tab_sloan_lines.append(
            f"\\quad & & ({fmt(_se_ret_fe,4)}) & ({fmt(_se_acq_fe,4)}) & "
            f"two-way $t={fmt(_fe_row['diff_t_twoway'],2)}$ & \\\\"
        )
        _tab_sloan_lines.append(
            f"\\quad & & {_fmt_pval(_fe_row['p_retention'])} & {_fmt_pval(_fe_row['p_acquisition'])} & "
            f"{_fmt_pval(_fe_row['diff_p_firmcl'])} & \\\\"
        )
    _tab_sloan_lines.append('\\midrule')
_tab_sloan_lines[-1] = '\\bottomrule'  # replace the trailing extra midrule
_tab_sloan_lines.append('\\end{tabular}%')
_tab_sloan_lines.append('}')
_fm_focal = _fm_persist[(_fm_persist['DV'] == 'OPINC_ROA_LEAD') & (_fm_persist['Winsor'] == '1/99-winsorized')].iloc[0]
_fm_T = int(_fm_focal['T_quarters'])
_fm_diff_disp = _signed_num(_fm_focal['fm_diff_coef'], 4)
_fm_t_disp = fmt(_fm_focal['fm_diff_t'], 2)
_fm_p_disp = _pval_inline(_fm_focal['fm_diff_p'])

_fe_diff_disp = _signed_num(_fe_row['diff_ret_minus_acq_firmcl'], 4)
_fe_t_firmcl_disp = fmt(_fe_row['diff_t_firmcl'], 2)
_fe_p_firmcl_disp = _pval_inline(_fe_row['diff_p_firmcl'])
_fe_t_twoway_disp = fmt(_fe_row['diff_t_twoway'], 2)
_fe_p_twoway_disp = _pval_inline(_fe_row['diff_p_twoway'])

_tab_sloan_lines.append('\\begin{tablenotes}')
_tab_sloan_lines.append('\\small')
_tab_sloan_lines.append(
    r'\item \textit{Notes:} Source: identification\_battery.py Task 4. Pooled OLS of the future outcome '
    r'on current-quarter retention and acquisition revenue-composition intensities (both scaled by '
    r'lagged total revenue, so the retention intensity equals contemporaneous \RRR{}) plus quarter '
    r'fixed effects; firm-clustered standard errors. Diff.\ column reports the firm-clustered test of '
    r'(Retention $-$ Acquisition) with its $t$-statistic; the two-way (firm $\times$ quarter) clustered '
    r'$t$-statistic is shown on the row below for comparison. Exact $p$-values (firm-clustered) in '
    r'brackets on the third row; no significance stars. Pre-registered criterion: PASS requires '
    r'retention $b >$ acquisition $b$ and a significant firm-clustered diff. A Fama-MacBeth '
    rf'(cross-sectional-by-quarter) version of the diff.\ test, $T={_fm_T}$ quarters, gives diff.\ '
    rf'$={_fm_diff_disp}$ ($t={_fm_t_disp}$, ${_fm_p_disp}$) for the operating-ROA, winsorized row; '
    r'identification\_battery.py (Task 4) computes this statistic but never writes it to its own '
    r'committed workbook, so fix\_fm\_persistence\_export.py persists it separately to '
    r'output/fm\_persistence.xlsx (sheet T4\_FM\_persistence), with a guardrail assert confirming the '
    r'value matches the originally observed console figure (see canonical\_run\_manifest.json). '
    r'The two dependent variables give an honestly mixed result: the operating-ROA leg is '
    r'sign-consistent with the pre-registered criterion in both raw and winsorized form (retained '
    r'revenue predicts future profitability more than acquired revenue does) but earns a PASS only in '
    r'the winsorized specification, where the diff is also statistically significant; the raw-data diff '
    r'carries the correct sign but is not statistically significant. The revenue-growth leg is '
    r'wrong-signed once winsorized (acquisition $b$ exceeds retention $b$) -- this is presented plainly '
    r'as a mixed result, not a clean pass. '
    r'The firm $+$ quarter FE row (source: firm\_fe\_persistence.xlsx) adds firm fixed effects to the '
    r'winsorized operating-ROA specification, absorbing all cross-firm variation and identifying the '
    r'retention-acquisition comparison purely from within-firm, quarter-to-quarter changes; the result '
    rf'reverses sign and remains statistically significant (diff.\ $={_fe_diff_disp}$, firm-clustered '
    rf'$t={_fe_t_firmcl_disp}$, ${_fe_p_firmcl_disp}$; two-way $t={_fe_t_twoway_disp}$, ${_fe_p_twoway_disp}$), so the '
    r'quarter-FE-only PASS above reflects a between-firm pattern, larger, more established firms both '
    r'retaining more revenue and earning more persistently profitable operating income, rather than a '
    r'within-firm dynamic in which a given firm'"'"'s own profitability persistence tracks its own '
    r'retention intensity over time.'
)
_tab_sloan_lines.append('\\end{tablenotes}')
_tab_sloan_lines.append('\\end{threeparttable}')
_write_tabular(_tab_sloan_lines, os.path.join(TABLE_DIR, 'tab_sloan_persistence.tex'))


# ============================================================================
# TABLE (NEW, item 9): Future-Beta Retry — minor, heavily caveated supporting note
# ============================================================================
print("[Table] Future-beta retry (rank-based, outlier-robust; minor/caveated)...")

_fb = pd.read_excel(os.path.join(OUTPUT_DIR, 'future_beta_retry.xlsx'), sheet_name='robust_rank_winsor_FM')
_fb_outlier = pd.read_excel(os.path.join(OUTPUT_DIR, 'future_beta_retry.xlsx'), sheet_name='outlier_diagnosis')
_fb_headline = _fb[_fb['Spec'] == 'rank(0-1) panel [headline]']
_fb_vw_adj = _fb_headline[(_fb_headline['Beta_def'] == 'sample VW market') &
                          (_fb_headline['Signal'] == 'ADJ_RRR_PCT')].iloc[0]
_fb_outlier_share = _fb_outlier.iloc[0]['top_Sxx_share'] * 100

_tab_fb_lines = []
_tab_fb_lines.append('\\begin{threeparttable}')
_tab_fb_lines.append('\\footnotesize')
_tab_fb_lines.append('\\begin{tabular}{llrrr}')
_tab_fb_lines.append('\\toprule')
_tab_fb_lines.append('Beta definition & Signal & Coef.\\ & $t$ & $N$ \\\\')
_tab_fb_lines.append('\\midrule')
for _, _r in _fb_headline.iterrows():
    _tab_fb_lines.append(
        f"{_r['Beta_def']} & {_r['Signal']} & {_fmt_coef(_r['coef'])} & ${fmt(_r['t'],2)}$ & {fmt_int(_r['N'])} \\\\"
    )
    _fb_se = _se_from_t(_r['coef'], _r['t'])
    _tab_fb_lines.append(f" & & ({fmt(_fb_se,4)}) {_fmt_pval(_r['p'])} & & \\\\")
_tab_fb_lines.append('\\bottomrule')
_tab_fb_lines.append('\\end{tabular}')
_tab_fb_lines.append('\\begin{tablenotes}')
_tab_fb_lines.append('\\small')
_tab_fb_lines.append(
    r'\item \textit{Notes:} Source: future\_beta\_retry.py, sheet robust\_rank\_winsor\_FM. '
    r'Rank(0--1)-transformed panel regression of a firm'"'"'s own trailing (12-month, $\geq$8 valid '
    r'months) market beta on the rank-transformed \RRR{} signal; two-way (firm, quarter) clustered '
    r'standard errors. This is a minor, heavily caveated supporting result, presented here as a note '
    r'rather than a headline finding; its placement in the main text versus an appendix is left to '
    r'author judgment at the prose-writing stage. '
    r'\textbf{Caveat 1 (outlier dependence):} an outlier-robust (rank-based) specification is required '
    rf'-- one quarter, 2020Q3, accounts for {_fb_outlier_share:.1f}\% of the weight in the '
    r'$S_{xx}$-weighted slope for adjusted \RRR{} and otherwise dominates the untransformed '
    r'level-panel version of this same regression (coef.\ $\approx 0.00003$, $t\approx 0.11$, not '
    r'significant; not tabulated). '
    r'\textbf{Caveat 2 (benchmark dependence):} this result does not extend to the sample'"'"'s own '
    r'value-weighted market beta -- substituting "sample VW market" for "FF market" as the dependent '
    rf'variable weakens the ADJ\_RRR\_PCT coefficient to ${_fb_vw_adj["coef"]:.4f}$ '
    rf'($t={_fb_vw_adj["t"]:.2f}$, not significant at conventional levels).'
)
_tab_fb_lines.append('\\end{tablenotes}')
_tab_fb_lines.append('\\end{threeparttable}')
_write_tabular(_tab_fb_lines, os.path.join(TABLE_DIR, 'tab_future_beta_retry.tex'))


# ============================================================================
# TABLE (NEW, item 10): SoRR — alpha, turnover/stability comparison, correlation
# ============================================================================
print("[Table] SoRR (Share of Retained Revenue): alpha, turnover-stability, correlation with RRR...")

_sorr_xlsx = os.path.join(OUTPUT_DIR, 'performance_sorr_summary.xlsx')
_sorr_alpha = pd.read_excel(_sorr_xlsx, sheet_name='SoRR_LongShort_Alpha')
_sorr_turnover = pd.read_excel(_sorr_xlsx, sheet_name='Signal_Turnover_Comparison').set_index('signal')
_sorr_corr = pd.read_excel(_sorr_xlsx, sheet_name='RRR_SoRR_Correlation').iloc[0]

_tab_sorr_lines = []
_tab_sorr_lines.append('\\begin{threeparttable}')
_tab_sorr_lines.append('\\footnotesize')
_tab_sorr_lines.append('\\begin{tabular}{lrrr}')
_tab_sorr_lines.append('\\toprule')
_tab_sorr_lines.append('\\multicolumn{4}{l}{\\textbf{Panel A: SoRR Long-Short Alpha (Q1$-$Q4, adjusted signal)}} \\\\')
_tab_sorr_lines.append('\\midrule')
_tab_sorr_lines.append('Model & $\\alpha$ (\\%/mo) & $R^2$ & $N$ \\\\')
for _, _r in _sorr_alpha.iterrows():
    _tab_sorr_lines.append(
        f"{_r['Spec']} & {_fmt_coef(_r['alpha_pct_month'])} & {fmt(_r['r2'],4)} & {fmt_int(_r['n_obs'])} \\\\"
    )
    _tab_sorr_lines.append(f" & ({fmt(_r['se_pct'],4)}) {_fmt_pval(_r['p_value'])} & & \\\\")
_tab_sorr_lines.append('\\midrule')
_tab_sorr_lines.append('\\multicolumn{4}{l}{\\textbf{Panel B: Signal Turnover / Stability Comparison}} \\\\')
_tab_sorr_lines.append('\\midrule')
_tab_sorr_lines.append(' & Mean abs.\\ chg.\\ (pp/qtr) & Cross-sect.\\ SD (pp) & Quartile stay-rate \\\\')
_rrr_turn = _sorr_turnover.loc['ADJ_RRR_PCT']
_srr_turn = _sorr_turnover.loc['ADJ_SRR_PCT']
_tab_sorr_lines.append(
    f"Adj.\\ \\RRR{{}} & {fmt(_rrr_turn['mean_abs_change_pp'],2)} & {fmt(_rrr_turn['cross_sectional_sd_pp'],2)} & "
    f"{fmt(_rrr_turn['quartile_stay_rate']*100,2)}\\% \\\\"
)
_tab_sorr_lines.append(
    f"Adj.\\ SoRR & {fmt(_srr_turn['mean_abs_change_pp'],2)} & {fmt(_srr_turn['cross_sectional_sd_pp'],2)} & "
    f"{fmt(_srr_turn['quartile_stay_rate']*100,2)}\\% \\\\"
)
_turnover_ratio = _rrr_turn['mean_abs_change_pp'] / _srr_turn['mean_abs_change_pp']
_tab_sorr_lines.append(f"Ratio (\\RRR{{}} / SoRR) & {fmt(_turnover_ratio,1)}$\\times$ & & \\\\")
_tab_sorr_lines.append('\\midrule')
_tab_sorr_lines.append('\\multicolumn{4}{l}{\\textbf{Panel C: \\RRR{}--SoRR Correlation (firm-quarter pooled)}} \\\\')
_tab_sorr_lines.append('\\midrule')
_tab_sorr_lines.append(
    f"Pearson (raw / adjusted) & \\multicolumn{{3}}{{l}}{{{fmt(_sorr_corr['pearson_raw'],4)} / "
    f"{fmt(_sorr_corr['pearson_adj'],4)}}} \\\\"
)
_tab_sorr_lines.append(
    f"Spearman (raw / adjusted) & \\multicolumn{{3}}{{l}}{{{fmt(_sorr_corr['spearman_raw'],4)} / "
    f"{fmt(_sorr_corr['spearman_adj'],4)}}} \\\\"
)
_tab_sorr_lines.append('\\bottomrule')
_tab_sorr_lines.append('\\end{tabular}')
_tab_sorr_lines.append('\\begin{tablenotes}')
_tab_sorr_lines.append('\\small')
_tab_sorr_lines.append(
    r'\item \textit{Notes:} Source: performance\_metrics.py (SoRR sub-analysis). SoRR = Share of '
    r'Retained Revenue = Returning\_Revenue$_t$ / Total\_Revenue$_t$ (normalized by \emph{current}-'
    r'period revenue; a composition share), distinct from \RRR{} = Returning\_Revenue$_t$ / '
    r'Total\_Revenue$_{t-1}$ (normalized by \emph{prior}-period revenue; a retention rate). Panel A: '
    r'value-weighted adjusted-SoRR long-short portfolio, OLS with Newey-West standard errors '
    r'(automatic bandwidth, Newey and West 1994); '
    r'standard error in parentheses, exact $p$-value in brackets, no significance stars; the alpha is '
    r'small and statistically indistinguishable from zero under every factor model. Panel B: SoRR is '
    r'roughly an order of magnitude more stable quarter-to-quarter than \RRR{} (smaller mean absolute '
    r'change, much higher quartile stay-rate). Panel C: \RRR{} and SoRR are positively but only '
    r'moderately correlated; the rank (Spearman) correlation is noticeably higher than the linear '
    r'(Pearson) correlation. Taken together, these three panels support reading SoRR as evidence that '
    r'\RRR{} reflects a stable underlying firm characteristic, not as a stronger return predictor in '
    r'its own right: SoRR is far more persistent than \RRR{} yet does not itself earn a long-short '
    r'return premium.'
)
_tab_sorr_lines.append('\\end{tablenotes}')
_tab_sorr_lines.append('\\end{threeparttable}')
_write_tabular(_tab_sorr_lines, os.path.join(TABLE_DIR, 'tab_sorr.tex'))


# ============================================================================
# TABLE (NEW, item 12): Dividend-Yield Check
# ============================================================================
print("[Table] Dividend-yield check (Q1 vs Q4; resolves price-only-returns concern)...")

_div_pooled = pd.read_excel(DIAG_XLSX, sheet_name='T6_DivYield_Pooled').set_index('RRR_Q_ADJ')
_div_firm = pd.read_excel(DIAG_XLSX, sheet_name='T6_DivYield_FirmLevel').set_index('RRR_Q_ADJ')
_div_fail = pd.read_excel(DIAG_XLSX, sheet_name='T6_Failures')
_n_div_covered = 124 - len(_div_fail)

_tab_div_lines = []
_tab_div_lines.append('\\begin{threeparttable}')
_tab_div_lines.append('\\footnotesize')
_tab_div_lines.append('\\begin{tabular}{lrrrr}')
_tab_div_lines.append('\\toprule')
_tab_div_lines.append(' & Q1 (\\%) & Q4 (\\%) & Q1 $-$ Q4 (pp) & $N$ (Q1 / Q4) \\\\')
_tab_div_lines.append('\\midrule')
_pooled_gap = (_div_pooled.loc['Q1', 'mean'] - _div_pooled.loc['Q4', 'mean']) * 100
_tab_div_lines.append(_latex_escape_row(
    f"Pooled (firm-quarter obs.) & {fmt(_div_pooled.loc['Q1','mean']*100,2)} & "
    f"{fmt(_div_pooled.loc['Q4','mean']*100,2)} & {fmt(_pooled_gap,2)} & "
    f"{fmt_int(_div_pooled.loc['Q1','count'])} / {fmt_int(_div_pooled.loc['Q4','count'])} \\\\"
))
_firm_gap = (_div_firm.loc['Q1', 'mean'] - _div_firm.loc['Q4', 'mean']) * 100
_tab_div_lines.append(_latex_escape_row(
    f"Firm-level (own-average first) & {fmt(_div_firm.loc['Q1','mean']*100,2)} & "
    f"{fmt(_div_firm.loc['Q4','mean']*100,2)} & {fmt(_firm_gap,2)} & "
    f"{fmt_int(_div_firm.loc['Q1','count'])} / {fmt_int(_div_firm.loc['Q4','count'])} \\\\"
))
_tab_div_lines.append('\\bottomrule')
_tab_div_lines.append('\\end{tabular}')
_tab_div_lines.append('\\begin{tablenotes}')
_tab_div_lines.append('\\small')
_tab_div_lines.append(
    rf'\item \textit{{Notes:}} Source: robustness\_diagnostics.py Task 6 (trailing dividend yield via '
    rf'yfinance; {_n_div_covered} of 124 sample firms have usable Yahoo Finance price/dividend history, '
    rf'{len(_div_fail)} failed to resolve, mostly delisted or unrecognized tickers). Trailing dividend '
    r'yield by adjusted-\RRR{} quartile, Q1 = highest \RRR{}, Q4 = lowest. The Q1$-$Q4 gap is negligible '
    r'in both pooled and firm-level form and, if anything, slightly negative (Q1 firms pay marginally '
    r'lower dividends), which resolves the concern that this paper'"'"'s price-only (non-dividend-'
    r'adjusted) monthly return convention could mechanically favor high-\RRR{} firms through an omitted '
    r'dividend channel.'
)
_tab_div_lines.append('\\end{tablenotes}')
_tab_div_lines.append('\\end{threeparttable}')
_write_tabular(_tab_div_lines, os.path.join(TABLE_DIR, 'tab_dividend_check.tex'))


# ============================================================================
# ITEM 13 (AR construct-validity): one sentence per author decision, NOT a
# dedicated table. Source data (identification_battery.xlsx, sheet T5_AR_winsor_FM)
# is already committed and unchanged; no table file is written here on purpose.
# Sentence for Stage 6 prose: "RRR retains a positive Fama-MacBeth coefficient
# (t=1.75 univariate) even against a 1st/99th-percentile winsorized version of AR
# (t=1.14 univariate for winsorized AR alone; RRR remains larger and, with controls,
# becomes significant at t=2.27 [p=.031] while winsorized AR stays insignistant and
# sign-flips), so RRR's predictive power is not an artifact of AR's heavier tails."
# ============================================================================

print("\n  Stage 5 canonical-run tables generated (headline, QMJ/BAB, consolidated "
      "robustness, concentration, monotonicity, signal persistence, H2a/H2b, Sloan "
      "persistence, future-beta, SoRR, performance/risk, portfolio characteristics, "
      "dividend check).")
print("\n  All tables generated.")


# ============================================================================
# FIGURES
# ============================================================================

def save_fig(fig, name):
    """Save figure as PDF."""
    path = os.path.join(FIGURE_DIR, f'{name}.pdf')
    fig.savefig(path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  Saved: {name}.pdf")


def prepend_zero_row(df, start_date='2017-01-01'):
    """Prepend a row of zeros at start_date so cumulative return graphs begin at 0.
    If start_date is already in the index, no row is prepended."""
    start = pd.Timestamp(start_date)
    if start in df.index:
        return df
    zero_row = pd.DataFrame([[0.0] * len(df.columns)], columns=df.columns, index=[start])
    return pd.concat([zero_row, df]).sort_index()


# ============================================================================
# FIGURE 1: Sector bar chart
# ============================================================================
print("\n[Figure 1] Sector bars...")

sector_counts = df_filtered.reset_index().groupby('SECTOR')['FIRM'].nunique().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4))
# Use the fixed SECTOR_COLORS map for consistent sector colour identity
bars = ax.barh(sector_counts.index, sector_counts.values,
               color=[SECTOR_COLORS.get(s, COLORS[0]) for s in sector_counts.index])
for bar, val in zip(bars, sector_counts.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, str(val),
            va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Number of Firms')
ax.set_title('Sample Firms by GICS Sector')
ax.set_xlim(0, sector_counts.max() * 1.15)
fig.tight_layout()
save_fig(fig, 'sector_bars')


# ============================================================================
# FIGURE 2: Firms per quarter
# ============================================================================
print("[Figure 2] Firms per quarter...")

firms_per_q = df_filtered.dropna(subset=['RRR_PCT']).reset_index().groupby('DATE')['FIRM'].nunique()
firms_per_q.index = pd.to_datetime(firms_per_q.index)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(firms_per_q.index, firms_per_q.values, marker='o', markersize=4, color=COLORS[0], linewidth=1.5)
ax.set_xlabel('Quarter')
ax.set_ylabel('Number of Firms with Valid RRR')
ax.set_title('Sample Coverage Over Time')
fig.tight_layout()
save_fig(fig, 'firms_per_quarter')


# ============================================================================
# FIGURE 3: RRR distribution
# ============================================================================
print("[Figure 3] RRR distribution...")

rrr_vals = pd.to_numeric(df_filtered['RRR_PCT'], errors='coerce').dropna()
# Winsorize for display
rrr_plot = rrr_vals.clip(-50, 200)

fig, ax = plt.subplots(figsize=(8, 4.5))
# density=False: y-axis shows raw observation counts, not probability density
ax.hist(rrr_plot, bins=80, density=False, alpha=0.5, color=COLORS[0], edgecolor='white', linewidth=0.5)
ax.axvline(rrr_plot.median(), color=COLORS[3], linestyle='--', linewidth=1.5,
           label=f'Median = {rrr_vals.median():.1f}%')
ax.axvline(rrr_vals.mean(), color=COLORS[1], linestyle='-', linewidth=1.5,
           label=f'Mean = {rrr_vals.mean():.1f}%')
ax.set_xlabel('RRR (%)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Revenue Retention Rate')
ax.legend()
fig.tight_layout()
save_fig(fig, 'rrr_distribution')


# ============================================================================
# FIGURE 4: AR distribution
# ============================================================================
print("[Figure 4] AR distribution...")

ar_vals = pd.to_numeric(df_filtered['ACQ_RATE_PCT'], errors='coerce').dropna()
ar_plot = ar_vals.clip(-100, 400)

fig, ax = plt.subplots(figsize=(8, 4.5))
# density=False: y-axis shows raw observation counts, not probability density
ax.hist(ar_plot, bins=80, density=False, alpha=0.5, color=COLORS[1], edgecolor='white', linewidth=0.5)
ax.axvline(ar_plot.median(), color=COLORS[3], linestyle='--', linewidth=1.5,
           label=f'Median = {ar_vals.median():.1f}%')
ax.axvline(ar_vals.mean(), color=COLORS[0], linestyle='-', linewidth=1.5,
           label=f'Mean = {ar_vals.mean():.1f}%')
ax.set_xlabel('AR (%)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Acquisition Rate')
ax.legend()
fig.tight_layout()
save_fig(fig, 'ar_distribution')


# ============================================================================
# FIGURE 5: RRR time series by sector
# ============================================================================
print("[Figure 5] RRR time series by sector...")

fig, ax = plt.subplots(figsize=(9, 5))
# Median lines only — IQR fill removed to avoid cross-sector shading overlap
for sector in sorted(df_filtered['SECTOR'].unique()):
    sd = df_filtered[df_filtered['SECTOR'] == sector].copy()
    sd['RRR_num'] = pd.to_numeric(sd['RRR_PCT'], errors='coerce')
    ts = sd.groupby(level='DATE')['RRR_num'].median()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()
    color = SECTOR_COLORS.get(sector, COLORS[0])
    ax.plot(ts.index, ts.values, label=sector, color=color, linewidth=1.5)

ax.set_xlabel('Quarter')
ax.set_ylabel('RRR (%)')
ax.set_title('Median RRR by Sector')
ax.legend(loc='best', fontsize=9)
fig.tight_layout()
save_fig(fig, 'rrr_ts')


# ============================================================================
# FIGURE 6: AR time series by sector
# ============================================================================
print("[Figure 6] AR time series by sector...")

fig, ax = plt.subplots(figsize=(9, 5))
# Median lines only — IQR fill removed to avoid cross-sector shading overlap
for sector in sorted(df_filtered['SECTOR'].unique()):
    sd = df_filtered[df_filtered['SECTOR'] == sector].copy()
    sd['AR_num'] = pd.to_numeric(sd['ACQ_RATE_PCT'], errors='coerce')
    ts = sd.groupby(level='DATE')['AR_num'].median()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()
    color = SECTOR_COLORS.get(sector, COLORS[0])
    ax.plot(ts.index, ts.values, label=sector, color=color, linewidth=1.5)

ax.set_xlabel('Quarter')
ax.set_ylabel('AR (%)')
ax.set_title('Median AR by Sector')
ax.legend(loc='best', fontsize=9)
fig.tight_layout()
save_fig(fig, 'ar_ts')


# ============================================================================
# FIGURE 7: RRR vs AR scatter
# ============================================================================
print("[Figure 7] RRR vs AR scatter...")

scatter_df = df_filtered[['RRR_PCT', 'ACQ_RATE_PCT', 'SECTOR']].copy()
scatter_df['RRR_PCT'] = pd.to_numeric(scatter_df['RRR_PCT'], errors='coerce')
scatter_df['ACQ_RATE_PCT'] = pd.to_numeric(scatter_df['ACQ_RATE_PCT'], errors='coerce')
scatter_df = scatter_df.dropna()
# Winsorize for display
scatter_df = scatter_df[(scatter_df['RRR_PCT'].between(-50, 200)) & (scatter_df['ACQ_RATE_PCT'].between(-100, 400))]

fig, ax = plt.subplots(figsize=(8, 6))
sectors_list = sorted(scatter_df['SECTOR'].unique())
for sector in sectors_list:
    sd = scatter_df[scatter_df['SECTOR'] == sector]
    color = SECTOR_COLORS.get(sector, COLORS[0])
    ax.scatter(sd['RRR_PCT'], sd['ACQ_RATE_PCT'], alpha=0.3, s=15, color=color, label=sector)

# Pooled (Pearson) correlation annotated directly on the plot
_pooled_corr = scatter_df['RRR_PCT'].corr(scatter_df['ACQ_RATE_PCT'])
ax.text(0.05, 0.95, f'$r = {_pooled_corr:.3f}$',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_xlabel('RRR (%)')
ax.set_ylabel('AR (%)')
ax.set_title('Revenue Retention Rate vs. Acquisition Rate')
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, 'rrr_ar_scatter')


# ============================================================================
# FIGURE 8: Cumulative returns by RRR QUARTILE (key figure)
# Y-axis: Cumulative Return (%)
# ============================================================================
print("[Figure 8] Cumulative returns by RRR quartile...")

_port_rrr_q_plot = prepend_zero_row(port_rrr_q)
fig, ax = plt.subplots(figsize=(9, 5.5))
# Q1=High RRR (best), Q4=Low RRR (worst) — consistent with analysis_v2 labelling
for col, color, ls, lw in [
    ('Q1', C_Q1, '-', 2),
    ('Q2', C_Q2, ':', 1.5),
    ('Q3', C_Q3, '-.', 1.5),
    ('Q4', C_Q4, '--', 1.5),
    ('MKT', C_MKT, '-.', 1.5),
]:
    if col in _port_rrr_q_plot.columns:
        cumret = _port_rrr_q_plot[col].cumsum() * 100  # cumulative log return in %
        label = f'{col} ({"High" if col=="Q1" else "Low" if col=="Q4" else "Mid"} RRR)' if col != 'MKT' else 'Market'
        ax.plot(cumret.index, cumret.values, label=label, color=color, linestyle=ls, linewidth=lw)

ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Cumulative Returns by RRR Quartile (Value-Weighted)')
ax.legend(loc='upper left', fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
fig.tight_layout()
save_fig(fig, 'cumret_rrr_q4_vw')


# ============================================================================
# FIGURE 9 (formerly 12): Double sort heatmap
# ============================================================================
print("[Figure 9] Double sort heatmap...")

# Build 3x3 matrix of annualized returns
heatmap_data = np.full((3, 3), np.nan)
rrr_labels = ['T1', 'T2', 'T3']
ar_labels = ['T1', 'T2', 'T3']

for i, rrr_t in enumerate(rrr_labels):
    for j, ar_t in enumerate(ar_labels):
        col = f'{rrr_t}_{ar_t}'
        if col in port_double.columns:
            heatmap_data[i, j] = port_double[col].mean() * 12 * 100

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(3))
ax.set_xticklabels(['T1 (High AR)', 'T2', 'T3 (Low AR)'])
ax.set_yticks(range(3))
ax.set_yticklabels(['T1 (High RRR)', 'T2', 'T3 (Low RRR)'])
ax.set_title('Annualized Returns (%) by RRR x AR Tercile')

# Annotate cells
for i in range(3):
    for j in range(3):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 10 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=12, fontweight='bold')

fig.colorbar(im, ax=ax, label='Annualized Return (%)')
fig.tight_layout()
save_fig(fig, 'double_sort_heatmap')


# ============================================================================
# FIGURE 10 (formerly 14): FMB coefficient plot
# ============================================================================
print("[Figure 10] FMB coefficient plot...")

# Plot RRR and AR coefficients across specifications
rrr_coefs = fama_macbeth[fama_macbeth['Variable'].isin(['ADJ_RRR_PCT', 'RRR_PCT'])]
ar_coefs = fama_macbeth[fama_macbeth['Variable'].isin(['ADJ_ACQ_RATE_PCT', 'ACQ_RATE_PCT'])]

fig, ax = plt.subplots(figsize=(9, 5))

y_pos = 0
labels = []
for _, row in rrr_coefs.iterrows():
    coef = row['Coefficient']
    se = coef / row['t-stat'] if row['t-stat'] != 0 else 0
    ci_lo = coef - 1.96 * abs(se)
    ci_hi = coef + 1.96 * abs(se)
    ax.plot([ci_lo, ci_hi], [y_pos, y_pos], color=COLORS[0], linewidth=2)
    ax.plot(coef, y_pos, 'o', color=COLORS[0], markersize=8)
    labels.append(f"RRR: {row['Spec'][:20]}")
    y_pos += 1

y_pos += 0.5  # gap
for _, row in ar_coefs.iterrows():
    coef = row['Coefficient']
    se = coef / row['t-stat'] if row['t-stat'] != 0 else 0
    ci_lo = coef - 1.96 * abs(se)
    ci_hi = coef + 1.96 * abs(se)
    ax.plot([ci_lo, ci_hi], [y_pos, y_pos], color=COLORS[1], linewidth=2)
    ax.plot(coef, y_pos, 'o', color=COLORS[1], markersize=8)
    labels.append(f"AR: {row['Spec'][:20]}")
    y_pos += 1

ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Fama-MacBeth Coefficient')
ax.set_title('Fama-MacBeth Coefficients with 95% CI')
fig.tight_layout()
save_fig(fig, 'fmb_coefs')


# ============================================================================
# FIGURE 11 (formerly 15): Risk comparison bars — uses quartile data
# ============================================================================
print("[Figure 11] Risk comparison (quartile)...")

# Compute risk metrics directly from quartile portfolio time series
_port_cols_risk = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4']
_risk_metrics_list = []
for _pc in _port_cols_risk:
    if _pc not in port_rrr_q.columns:
        continue
    _r = port_rrr_q[_pc].dropna()
    _ann_ret = _r.mean() * _MONTHS_PER_YEAR * 100
    _ann_vol = _r.std() * np.sqrt(_MONTHS_PER_YEAR) * 100
    _sharpe = (_r.mean() / _r.std() * np.sqrt(_MONTHS_PER_YEAR)) if _r.std() > 0 else np.nan
    _cumret = _r.cumsum()
    _max_dd = (_cumret - _cumret.cummax()).min() * 100
    _downside_ret = _r[_r < 0]
    _downside_dev = np.sqrt((_downside_ret ** 2).mean()) * np.sqrt(_MONTHS_PER_YEAR)
    _sortino = (_r.mean() * _MONTHS_PER_YEAR / _downside_dev) if _downside_dev > 0 else np.nan
    _risk_metrics_list.append({
        'Portfolio': _pc,
        'Sharpe': _sharpe,
        'Sortino': _sortino,
        'Max Drawdown (%)': _max_dd,
    })
_risk_metrics_q = pd.DataFrame(_risk_metrics_list)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
_risk_plot_items = [
    ('Sharpe', 'Sharpe Ratio'),
    ('Sortino', 'Sortino Ratio'),
    ('Max Drawdown (%)', 'Max Drawdown (%)'),
]

for ax, (col, title) in zip(axes, _risk_plot_items):
    if col in _risk_metrics_q.columns:
        _vals = _risk_metrics_q[col].tolist()
        _lbls = _risk_metrics_q['Portfolio'].tolist()
        # Color by quartile label: Q1=best color, Q4=worst, others intermediate
        _colors = [C_Q1 if l == 'Q1' else C_Q4 if l == 'Q4'
                   else C_Q2 if l == 'Q2' else C_Q3 if l == 'Q3'
                   else COLORS[7] for l in _lbls]
        ax.bar(_lbls, _vals, color=_colors, alpha=0.85)
        ax.set_title(title)
        ax.axhline(0, color='gray', linewidth=0.5)

fig.suptitle('Risk Metrics by RRR Quartile', fontsize=12, fontweight='bold')
fig.tight_layout()
save_fig(fig, 'risk_comparison')


# ============================================================================
# FIGURE 12 (formerly 16): Drawdown time series — uses quartile data
# ============================================================================
print("[Figure 12] Drawdown time series (quartile)...")

fig, ax = plt.subplots(figsize=(9, 5))
for col, color, ls in [('Q1', C_Q1, '-'), ('Q2', C_Q2, ':'), ('Q3', C_Q3, '-.'), ('Q4', C_Q4, '--')]:
    if col in port_rrr_q.columns:
        cum = np.exp(port_rrr_q[col].cumsum())
        running_max = cum.cummax()
        dd = (cum / running_max - 1) * 100
        ax.plot(dd.index, dd.values, label=col, color=color, linestyle=ls, linewidth=1.5)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.1, color=color)

ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.set_title('Rolling Drawdown by RRR Quartile')
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
fig.tight_layout()
save_fig(fig, 'drawdown_ts')


# ============================================================================
# FIGURE 13 (formerly 17): Up/down market returns — uses quartile data
# ============================================================================
print("[Figure 13] Up/down market returns (quartile)...")

port_q_aligned = port_rrr_q.copy()
port_q_aligned.index = pd.to_datetime(port_q_aligned.index).to_period('M').to_timestamp('M')
ff_aligned = ff_factors.reindex(port_q_aligned.index)
mkt_rf = ff_aligned.get('Mkt-RF', pd.Series(dtype=float))

up_months = mkt_rf[mkt_rf >= 0].index
down_months = mkt_rf[mkt_rf < 0].index

fig, ax = plt.subplots(figsize=(8, 5))
quartile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
x = np.arange(len(quartile_labels))
width = 0.35

# Compute average return in up and down months for each quartile
up_rets = [
    port_q_aligned.loc[port_q_aligned.index.isin(up_months), q].mean() * 100
    for q in quartile_labels if q in port_q_aligned.columns
]
down_rets = [
    port_q_aligned.loc[port_q_aligned.index.isin(down_months), q].mean() * 100
    for q in quartile_labels if q in port_q_aligned.columns
]

ax.bar(x - width/2, up_rets, width, label='Up-Market Months', color=COLORS[2], alpha=0.85)
ax.bar(x + width/2, down_rets, width, label='Down-Market Months', color=COLORS[3], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(quartile_labels)
ax.set_ylabel('Average Monthly Return (%)')
ax.set_title('Average Return in Up vs. Down Market Months by RRR Quartile')
ax.legend()
ax.axhline(0, color='gray', linewidth=0.5)
fig.tight_layout()
save_fig(fig, 'downside_upside')


# ============================================================================
# FIGURE 14 (formerly 19): Cumulative returns — placebo (quartile)
# ============================================================================
print("[Figure 14] Placebo cumulative returns (median split)...")

_port_pl_m_plot = prepend_zero_row(port_placebo_m)
fig, ax = plt.subplots(figsize=(9, 5.5))
for col, color, ls, lw in [
    ('High', C_Q1, '-', 2),
    ('Low', C_Q4, '--', 1.5),
]:
    if col in _port_pl_m_plot.columns:
        cumret = _port_pl_m_plot[col].cumsum() * 100
        label = f'{col} RRR'
        ax.plot(cumret.index, cumret.values, label=label, color=color, linestyle=ls, linewidth=lw)

if 'High-Low' in _port_pl_m_plot.columns:
    cumret_ls = _port_pl_m_plot['High-Low'].cumsum() * 100
    ax.plot(cumret_ls.index, cumret_ls.values, label='High$-$Low',
            color='black', linestyle='-', linewidth=2, alpha=0.7)

ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Same-Growth Placebo: Cumulative Returns (High-Growth Subsample)')
ax.legend(loc='upper left', fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
fig.tight_layout()
save_fig(fig, 'cumret_placebo')


# ============================================================================
# FIGURE 15 (formerly 22): Cumulative returns by adjusted RRR quartile (MAIN result)
# ============================================================================
print("[Figure 15] Cumulative returns by adjusted RRR quartile (MAIN)...")

_port_rrr_adj_q_plot = prepend_zero_row(port_rrr_q)
fig, ax = plt.subplots(figsize=(9, 5.5))
# Q1=High Adj. RRR (best), Q4=Low Adj. RRR (worst)
for col, color, ls, lw in [
    ('Q1', C_Q1, '-', 2),
    ('Q2', C_Q2, ':', 1.5),
    ('Q3', C_Q3, '-.', 1.5),
    ('Q4', C_Q4, '--', 1.5),
    ('MKT', C_MKT, '-.', 1.5),
]:
    if col in _port_rrr_adj_q_plot.columns:
        cumret = _port_rrr_adj_q_plot[col].cumsum() * 100
        label = f'{col} ({"High" if col=="Q1" else "Low" if col=="Q4" else "Mid"} Adj.~RRR)' if col != 'MKT' else 'Market'
        ax.plot(cumret.index, cumret.values, label=label, color=color, linestyle=ls, linewidth=lw)

ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Cumulative Returns by Adj. RRR Quartile (Value-Weighted) -- Main Result')
ax.legend(loc='upper left', fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
fig.tight_layout()
save_fig(fig, 'cumret_rrr_adj_q4_vw')


# ============================================================================
# FIGURE 16 (formerly 23): Cumulative returns by adjusted AR quartile (MAIN result)
# ============================================================================
print("[Figure 16] Cumulative returns by adjusted AR quartile (MAIN)...")

_port_ar_adj_q_plot = prepend_zero_row(port_ar_q)
fig, ax = plt.subplots(figsize=(9, 5.5))
# Q1=High Adj. AR (best), Q4=Low Adj. AR (worst)
for col, color, ls, lw in [
    ('Q1', C_Q1, '-', 2),
    ('Q2', C_Q2, ':', 1.5),
    ('Q3', C_Q3, '-.', 1.5),
    ('Q4', C_Q4, '--', 1.5),
    ('MKT', C_MKT, '-.', 1.5),
]:
    if col in _port_ar_adj_q_plot.columns:
        cumret = _port_ar_adj_q_plot[col].cumsum() * 100
        label = f'{col} ({"High" if col=="Q1" else "Low" if col=="Q4" else "Mid"} Adj.~AR)' if col != 'MKT' else 'Market'
        ax.plot(cumret.index, cumret.values, label=label, color=color, linestyle=ls, linewidth=lw)

ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Cumulative Returns by Adj. AR Quartile (Value-Weighted) -- Main Result')
ax.legend(loc='upper left', fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
fig.tight_layout()
save_fig(fig, 'cumret_ar_adj_q4_vw')


# ============================================================================
# FIGURE 17 (NEW): Revenue Growth by Adjusted RRR Quartile
# ============================================================================
print("[Figure 17] Revenue growth by adjusted RRR quartile...")

_rg_df = ret[['RRR_Q', 'REV_GROWTH_PCT']].copy()
_rg_df['REV_GROWTH_PCT'] = pd.to_numeric(_rg_df['REV_GROWTH_PCT'], errors='coerce')
_rg_df = _rg_df.dropna(subset=['RRR_Q', 'REV_GROWTH_PCT'])
# Winsorize at 1st/99th to remove outliers for display
_lo, _hi = _rg_df['REV_GROWTH_PCT'].quantile([0.01, 0.99])
_rg_df = _rg_df[_rg_df['REV_GROWTH_PCT'].between(_lo, _hi)]

_rg_stats = _rg_df.groupby('RRR_Q')['REV_GROWTH_PCT'].agg(['mean', 'median']).reindex(['Q1', 'Q2', 'Q3', 'Q4'])

fig, ax = plt.subplots(figsize=(8, 5))
_x = np.arange(4)
_w = 0.35
_labels = ['Q1\n(High RRR)', 'Q2', 'Q3', 'Q4\n(Low RRR)']
_colors_q = [C_Q1, C_Q2, C_Q3, C_Q4]
ax.bar(_x - _w/2, _rg_stats['mean'].values, _w, label='Mean', color=_colors_q, alpha=0.85)
ax.bar(_x + _w/2, _rg_stats['median'].values, _w, label='Median', color=_colors_q, alpha=0.45, edgecolor=_colors_q, linewidth=1.5)
ax.set_xticks(_x)
ax.set_xticklabels(_labels)
ax.set_xlabel('Adjusted RRR Quartile')
ax.set_ylabel('QoQ Revenue Growth (%)')
ax.set_title('Revenue Growth by Adjusted RRR Quartile')
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend()
fig.tight_layout()
save_fig(fig, 'revenue_growth_by_rrr_quartile')


# ============================================================================
# FIGURE 18 (NEW): Revenue VaR (5th Pct of QoQ Revenue Growth) by RRR Quartile
# ============================================================================
print("[Figure 18] Revenue VaR by adjusted RRR quartile...")

_var_stats = _rg_df.groupby('RRR_Q')['REV_GROWTH_PCT'].quantile(0.05).reindex(['Q1', 'Q2', 'Q3', 'Q4'])
_var_stats_10 = _rg_df.groupby('RRR_Q')['REV_GROWTH_PCT'].quantile(0.10).reindex(['Q1', 'Q2', 'Q3', 'Q4'])

fig, ax = plt.subplots(figsize=(8, 5))
_x = np.arange(4)
_w = 0.35
ax.bar(_x - _w/2, _var_stats.values, _w, label='5th Pct (VaR)', color=_colors_q, alpha=0.85)
ax.bar(_x + _w/2, _var_stats_10.values, _w, label='10th Pct', color=_colors_q, alpha=0.45, edgecolor=_colors_q, linewidth=1.5)
ax.set_xticks(_x)
ax.set_xticklabels(_labels)
ax.set_xlabel('Adjusted RRR Quartile')
ax.set_ylabel('Revenue Growth (%, Percentile)')
ax.set_title('Revenue VaR by Adjusted RRR Quartile (Downside Tail)')
ax.axhline(0, color='gray', linewidth=0.5)
ax.legend()
fig.tight_layout()
save_fig(fig, 'revenue_var_by_quartile')


# ============================================================================
# DONE
# ============================================================================
print("\n" + "=" * 80)
print("  ALL TABLES AND FIGURES GENERATED")
print(f"  Tables: {TABLE_DIR}")
print(f"  Figures: {FIGURE_DIR}")
print("=" * 80)

# List outputs
print("\nTables:")
for f in sorted(os.listdir(TABLE_DIR)):
    print(f"  {f}")
print("\nFigures:")
for f in sorted(os.listdir(FIGURE_DIR)):
    print(f"  {f}")
