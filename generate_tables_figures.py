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
sys.path.insert(0, CODE_DIR)
from analysis_v2 import (
    phase1_load_and_diagnose, phase1b_load_ff_factors, phase1c_load_monthly_returns,
    merge_signals_to_returns, safe_tercile, build_portfolio_returns, newey_west_ols
)

# Import export_table tool
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


def stars_from_t(t_stat):
    """Return significance stars from t-statistic (legacy — use _stars() for new tables)."""
    t = abs(t_stat)
    if t > 2.576:
        return "$^{***}$"
    elif t > 1.96:
        return "$^{**}$"
    elif t > 1.645:
        return "$^{*}$"
    return ""


# ---------------------------------------------------------------------------
# Revenue_Growth-style table helpers
# ---------------------------------------------------------------------------

def _stars(pval):
    """Return significance stars from p-value."""
    if pval < 0.01:
        return '***'
    if pval < 0.05:
        return '**'
    if pval < 0.10:
        return '*'
    return ''


def _fmt_coef(coef, pval, decimals=4):
    """Format coefficient with significance stars and LaTeX minus sign."""
    sign = '$-$' if coef < 0 else ''
    return f"{sign}{abs(coef):.{decimals}f}{_stars(pval)}"


def _fmt_se(se, decimals=4):
    """Format standard error in parentheses."""
    return f"({se:.{decimals}f})"


def _write_tabular(lines, output_path):
    """Write a list of LaTeX lines to a .tex file (UTF-8)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f"  Written: {os.path.basename(output_path)}")


def safe_quartile(x):
    """Assign quartiles: Q1=highest value, Q4=lowest value.
    Labels are reversed so the best (highest) firms receive Q1 — consistent with analysis_v2.py."""
    x = x.dropna()
    if x.nunique() < 4 or len(x) < 8:
        return pd.Series([np.nan]*len(x), index=x.index)
    try:
        return pd.qcut(x, 4, labels=['Q4', 'Q3', 'Q2', 'Q1'])
    except ValueError:
        return pd.Series([np.nan]*len(x), index=x.index)


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
returns_merged = merge_signals_to_returns(df_filtered, returns)

# Also read pre-computed Excel files
pooled_desc = pd.read_excel(os.path.join(OUTPUT_DIR, 'pooled_descriptives.xlsx'))
industry_desc = pd.read_excel(os.path.join(OUTPUT_DIR, 'industry_descriptives.xlsx'))
risk_analysis = pd.read_excel(os.path.join(OUTPUT_DIR, 'risk_analysis.xlsx'))
fama_macbeth = pd.read_excel(os.path.join(OUTPUT_DIR, 'fama_macbeth.xlsx'))

print("\n  Data loaded successfully.")


# ============================================================================
# BUILD PORTFOLIO RETURN TIME SERIES (for figures and inline factor regressions)
# ============================================================================
print("\n[Step 0b] Building portfolio return time series...")

ret = returns_merged.copy()
for col in ['RRR_PCT_LAG1', 'ACQ_RATE_PCT_LAG1', 'ADJ_RRR_PCT_LAG1',
            'ADJ_ACQ_RATE_PCT_LAG1', 'HISTORICAL_MARKET_CAP']:
    ret[col] = pd.to_numeric(ret[col], errors='coerce')

# Market portfolio (value-weighted, all sample firms)
ret_sample = ret[~ret['FIRM'].isin(['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX'])].copy()
ret_sample['MCAP'] = ret_sample['HISTORICAL_MARKET_CAP']
ret_sample['MCAP_TOTAL'] = ret_sample.groupby('Date')['MCAP'].transform('sum')
ret_sample['w_mkt'] = ret_sample['MCAP'] / ret_sample['MCAP_TOTAL']
ret_sample['w_ret_mkt'] = ret_sample['w_mkt'] * ret_sample['RETURN_LOG']
market_ret = ret_sample.groupby('Date')['w_ret_mkt'].sum().sort_index()

# --- Tercile portfolios (RRR adjusted) — kept for double-sort and legacy use ---
ret['RRR_T'] = ret.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_tercile)
port_rrr_t = build_portfolio_returns(ret, 'RRR_T', market_ret)

# --- Tercile portfolios (AR adjusted) ---
ret['AR_T'] = ret.groupby('QUARTER')['ADJ_ACQ_RATE_PCT_LAG1'].transform(safe_tercile)
port_ar_t = build_portfolio_returns(ret, 'AR_T', market_ret)

# --- Quartile portfolios (adjusted RRR) ---
ret['RRR_Q'] = ret.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_quartile)
df_q = ret.dropna(subset=['RRR_Q']).copy()
df_q['MCAP'] = pd.to_numeric(df_q['HISTORICAL_MARKET_CAP'], errors='coerce')
df_q['MCAP_SUM'] = df_q.groupby(['Date', 'RRR_Q'])['MCAP'].transform('sum')
df_q['w'] = df_q['MCAP'] / df_q['MCAP_SUM']
df_q['w_return'] = df_q['w'] * df_q['RETURN_LOG']
port_rrr_q = (
    df_q.groupby(['Date', 'RRR_Q'])['w_return']
    .sum()
    .unstack('RRR_Q')
    .sort_index()
)
# Q1=High adjusted RRR (best), Q4=Low adjusted RRR (worst)
if 'Q1' in port_rrr_q.columns and 'Q4' in port_rrr_q.columns:
    port_rrr_q['Q1-Q4'] = port_rrr_q['Q1'] - port_rrr_q['Q4']
port_rrr_q['MKT'] = market_ret.reindex(port_rrr_q.index)

# --- Quartile portfolios (adjusted AR) ---
ret['AR_Q'] = ret.groupby('QUARTER')['ADJ_ACQ_RATE_PCT_LAG1'].transform(safe_quartile)
df_q_ar = ret.dropna(subset=['AR_Q']).copy()
df_q_ar['MCAP'] = pd.to_numeric(df_q_ar['HISTORICAL_MARKET_CAP'], errors='coerce')
df_q_ar['MCAP_SUM'] = df_q_ar.groupby(['Date', 'AR_Q'])['MCAP'].transform('sum')
df_q_ar['w'] = df_q_ar['MCAP'] / df_q_ar['MCAP_SUM']
df_q_ar['w_return'] = df_q_ar['w'] * df_q_ar['RETURN_LOG']
port_ar_q = (
    df_q_ar.groupby(['Date', 'AR_Q'])['w_return']
    .sum()
    .unstack('AR_Q')
    .sort_index()
)
# Q1=High adjusted AR (best), Q4=Low adjusted AR (worst)
if 'Q1' in port_ar_q.columns and 'Q4' in port_ar_q.columns:
    port_ar_q['Q1-Q4'] = port_ar_q['Q1'] - port_ar_q['Q4']
port_ar_q['MKT'] = market_ret.reindex(port_ar_q.index)

# --- Quartile portfolios (raw RRR) — for Table 4 Panel A ---
ret['RRR_RAW_Q'] = ret.groupby('QUARTER')['RRR_PCT_LAG1'].transform(safe_quartile)
df_q_rrr_raw = ret.dropna(subset=['RRR_RAW_Q']).copy()
df_q_rrr_raw['MCAP'] = pd.to_numeric(df_q_rrr_raw['HISTORICAL_MARKET_CAP'], errors='coerce')
df_q_rrr_raw['MCAP_SUM'] = df_q_rrr_raw.groupby(['Date', 'RRR_RAW_Q'])['MCAP'].transform('sum')
df_q_rrr_raw['w'] = df_q_rrr_raw['MCAP'] / df_q_rrr_raw['MCAP_SUM']
df_q_rrr_raw['w_return'] = df_q_rrr_raw['w'] * df_q_rrr_raw['RETURN_LOG']
port_rrr_raw_q = (
    df_q_rrr_raw.groupby(['Date', 'RRR_RAW_Q'])['w_return']
    .sum()
    .unstack('RRR_RAW_Q')
    .sort_index()
)
if 'Q1' in port_rrr_raw_q.columns and 'Q4' in port_rrr_raw_q.columns:
    port_rrr_raw_q['Q1-Q4'] = port_rrr_raw_q['Q1'] - port_rrr_raw_q['Q4']
port_rrr_raw_q['MKT'] = market_ret.reindex(port_rrr_raw_q.index)

# --- Quartile portfolios (raw AR) — for Table 5 Panel A ---
ret['AR_RAW_Q'] = ret.groupby('QUARTER')['ACQ_RATE_PCT_LAG1'].transform(safe_quartile)
df_q_ar_raw = ret.dropna(subset=['AR_RAW_Q']).copy()
df_q_ar_raw['MCAP'] = pd.to_numeric(df_q_ar_raw['HISTORICAL_MARKET_CAP'], errors='coerce')
df_q_ar_raw['MCAP_SUM'] = df_q_ar_raw.groupby(['Date', 'AR_RAW_Q'])['MCAP'].transform('sum')
df_q_ar_raw['w'] = df_q_ar_raw['MCAP'] / df_q_ar_raw['MCAP_SUM']
df_q_ar_raw['w_return'] = df_q_ar_raw['w'] * df_q_ar_raw['RETURN_LOG']
port_ar_raw_q = (
    df_q_ar_raw.groupby(['Date', 'AR_RAW_Q'])['w_return']
    .sum()
    .unstack('AR_RAW_Q')
    .sort_index()
)
if 'Q1' in port_ar_raw_q.columns and 'Q4' in port_ar_raw_q.columns:
    port_ar_raw_q['Q1-Q4'] = port_ar_raw_q['Q1'] - port_ar_raw_q['Q4']
port_ar_raw_q['MKT'] = market_ret.reindex(port_ar_raw_q.index)

# --- No-COVID quartile portfolios (adjusted RRR) — for Table 8 Panel A ---
ret_nc = ret[~((ret['Date'] >= '2020-01-01') & (ret['Date'] <= '2021-06-30'))].copy()
ret_nc['RRR_Q_NC'] = ret_nc.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_quartile)
ret_nc_sample = ret_nc[~ret_nc['FIRM'].isin(['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX'])].copy()
ret_nc_sample['MCAP'] = ret_nc_sample['HISTORICAL_MARKET_CAP']
ret_nc_sample['MCAP_TOTAL'] = ret_nc_sample.groupby('Date')['MCAP'].transform('sum')
ret_nc_sample['w_mkt'] = ret_nc_sample['MCAP'] / ret_nc_sample['MCAP_TOTAL']
ret_nc_sample['w_ret_mkt'] = ret_nc_sample['w_mkt'] * ret_nc_sample['RETURN_LOG']
market_ret_nc = ret_nc_sample.groupby('Date')['w_ret_mkt'].sum().sort_index()
df_q_nc = ret_nc.dropna(subset=['RRR_Q_NC']).copy()
df_q_nc['MCAP'] = pd.to_numeric(df_q_nc['HISTORICAL_MARKET_CAP'], errors='coerce')
df_q_nc['MCAP_SUM'] = df_q_nc.groupby(['Date', 'RRR_Q_NC'])['MCAP'].transform('sum')
df_q_nc['w'] = df_q_nc['MCAP'] / df_q_nc['MCAP_SUM']
df_q_nc['w_return'] = df_q_nc['w'] * df_q_nc['RETURN_LOG']
port_nc_q = (
    df_q_nc.groupby(['Date', 'RRR_Q_NC'])['w_return']
    .sum()
    .unstack('RRR_Q_NC')
    .sort_index()
)
if 'Q1' in port_nc_q.columns and 'Q4' in port_nc_q.columns:
    port_nc_q['Q1-Q4'] = port_nc_q['Q1'] - port_nc_q['Q4']
port_nc_q['MKT'] = market_ret_nc.reindex(port_nc_q.index)

# --- Equal-weighted quartile portfolios (adjusted RRR) — for Table 8 Panel B ---
ret_ew = ret.copy()
ret_ew['RRR_Q_EW'] = ret_ew.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_quartile)
df_ew_q = ret_ew.dropna(subset=['RRR_Q_EW']).copy()
port_ew_q = (
    df_ew_q.groupby(['Date', 'RRR_Q_EW'])['RETURN_LOG']
    .mean()
    .unstack('RRR_Q_EW')
    .sort_index()
)
if 'Q1' in port_ew_q.columns and 'Q4' in port_ew_q.columns:
    port_ew_q['Q1-Q4'] = port_ew_q['Q1'] - port_ew_q['Q4']
ew_mkt = df_ew_q.groupby('Date')['RETURN_LOG'].mean().sort_index()
port_ew_q['MKT'] = ew_mkt.reindex(port_ew_q.index)

# --- Double sort portfolios (3x3, for Figure 12) ---
ret['RRR_T_DS'] = ret.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_tercile)
ret['AR_T_DS'] = ret.groupby('QUARTER')['ADJ_ACQ_RATE_PCT_LAG1'].transform(safe_tercile)
df_ds = ret.dropna(subset=['RRR_T_DS', 'AR_T_DS']).copy()
df_ds['PORT'] = df_ds['RRR_T_DS'].astype(str) + '_' + df_ds['AR_T_DS'].astype(str)
df_ds['MCAP'] = pd.to_numeric(df_ds['HISTORICAL_MARKET_CAP'], errors='coerce')
df_ds['MCAP_SUM'] = df_ds.groupby(['Date', 'PORT'])['MCAP'].transform('sum')
df_ds['w'] = df_ds['MCAP'] / df_ds['MCAP_SUM']
df_ds['w_return'] = df_ds['w'] * df_ds['RETURN_LOG']
port_double = (
    df_ds.groupby(['Date', 'PORT'])['w_return']
    .sum()
    .unstack('PORT')
    .sort_index()
)

# --- Placebo median-split portfolios — within top quartile of revenue growth ---
# Restrict to top quartile of revenue growth, then median-split by adjusted RRR
ret_pl = ret.copy()
ret_pl['REV_GROWTH_PCT'] = pd.to_numeric(ret_pl['REV_GROWTH_PCT'], errors='coerce')

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

ret_pl['GROWTH_Q'] = ret_pl.groupby('QUARTER')['REV_GROWTH_PCT'].transform(assign_growth_quartile)
# Keep only top-quartile revenue growth firms
high_growth_q = ret_pl[ret_pl['GROWTH_Q'] == 'Q1_G'].copy()
# Within this subsample, median-split by adjusted RRR
high_growth_q['RRR_M_PL'] = high_growth_q.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_median_split)
df_pl = high_growth_q.dropna(subset=['RRR_M_PL']).copy()
df_pl['MCAP'] = pd.to_numeric(df_pl['HISTORICAL_MARKET_CAP'], errors='coerce')
df_pl['MCAP_SUM'] = df_pl.groupby(['Date', 'RRR_M_PL'])['MCAP'].transform('sum')
df_pl['w'] = df_pl['MCAP'] / df_pl['MCAP_SUM']
df_pl['w_return'] = df_pl['w'] * df_pl['RETURN_LOG']
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

print("  Portfolio time series built.")


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
    r'\textit{Note: OLS with Newey-West standard errors (4 lags). '
    r'SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}'
)

# Newey-West lag count
NW_LAGS = 4


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
    portfolio_cols = [c for c in ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4'] if c in port.columns]

    for port_col in portfolio_cols:
        # For individual portfolios subtract RF to get excess return; long-short is already zero-cost
        if port_col == 'Q1-Q4':
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

            ols_result = newey_west_ols(y, X, max_lags=NW_LAGS)
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

    lines = []
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
        # Repeat portfolio header row so each model block is self-contained
        lines.append(header_str + ' \\\\')

        # Alpha row
        alpha_cells = []
        se_alpha_cells = []
        for port in port_labels:
            key = f'{port}_{model}'
            if key in reg_results:
                r = reg_results[key]
                a_pct = r['alpha'] * 100     # convert to %/month
                se_pct = r['se_alpha'] * 100
                alpha_cells.append(_fmt_coef(a_pct, r['alpha_p']))
                se_alpha_cells.append(_fmt_se(se_pct))
            else:
                alpha_cells.append('')
                se_alpha_cells.append('')
        lines.append('  $\\alpha$ (\\%/mo) & ' + ' & '.join(alpha_cells) + ' \\\\')
        lines.append('  & ' + ' & '.join(se_alpha_cells) + ' \\\\')

        # Factor beta rows
        for factor in active_factors:
            beta_cells = []
            se_beta_cells = []
            for port in port_labels:
                key = f'{port}_{model}'
                b_key = f'beta_{factor}'
                s_key = f'se_{factor}'
                p_key = f'pval_{factor}'
                if key in reg_results and b_key in reg_results[key]:
                    r = reg_results[key]
                    beta_cells.append(_fmt_coef(r[b_key], r[p_key]))
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
    lines.append('\\vspace{2pt}')
    lines.append(f'{{{note_text}}}')
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

_tab1_lines = []
_tab1_lines.append('\\begin{tabular}{lrrrrrrrr}')
_tab1_lines.append('\\toprule')
_tab1_lines.append('Variable & N & Mean & Median & SD & P10 & P25 & P75 & P90 \\\\')
_tab1_lines.append('\\midrule')
def _latex_escape_row(s):
    """Escape % and replace leading - with $-$ in table cells."""
    s = s.replace('%', '\\%')
    # Replace negative numbers: ' -X.XX' -> ' $-$X.XX'
    import re
    s = re.sub(r'(?<=& )-(\d)', r'$-$\1', s)
    # Also handle comma-separated thousands with negatives
    s = re.sub(r'(?<=& )-(\d)', r'$-$\1', s)
    return s

for _, r in pooled_desc.iterrows():
    _row = (
        f"{r['Variable']} & {fmt_int(r['N'])} & {fmt(r['Mean'],2)} & "
        f"{fmt(r['Median'],2)} & {fmt(r['SD'],2)} & {fmt(r['P10'],2)} & "
        f"{fmt(r['P25'],2)} & {fmt(r['P75'],2)} & {fmt(r['P90'],2)} \\\\"
    )
    _tab1_lines.append(_latex_escape_row(_row))
_tab1_lines.append('\\bottomrule')
_tab1_lines.append('\\end{tabular}')
_tab1_lines.append('\\vspace{2pt}')
_tab1_lines.append(
    r'{\footnotesize\textit{Note: '
    r'Pooled summary statistics for the 124 sample firms over 2017Q1--2024Q3. '
    r'RRR, AR, and Revenue Growth are expressed as percentages. '
    r'Market Cap is in millions of USD. BTM is book-to-market ratio. '
    r'PM is operating profit margin (\%).}}'
)
_write_tabular(_tab1_lines, os.path.join(TABLE_DIR, 'tab_summary_stats.tex'))


# ============================================================================
# TABLE 2: Sample Composition
# ============================================================================
print("[Table 2] Sample composition...")

_tab2_lines = []
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
        _tab2_lines.append(
            f"{_sec_cell} & {_metric} & {fmt_int(_r['N'])} & "
            f"{fmt(_r['Mean'],2)} & {fmt(_r['Median'],2)} & {fmt(_r['SD'],2)} & "
            f"{fmt(_r['P10'],2)} & {fmt(_r['P25'],2)} & {fmt(_r['P75'],2)} & "
            f"{fmt(_r['P90'],2)} \\\\"
        )
    if _s_idx < len(_sectors) - 1:
        _tab2_lines.append('\\midrule')
_tab2_lines.append('\\bottomrule')
_tab2_lines.append('\\end{tabular}%')
_tab2_lines.append('}')
_tab2_lines.append('\\vspace{2pt}')
_tab2_lines.append(
    r'{\footnotesize\textit{Note: '
    r'Number of firms and descriptive statistics for Revenue Growth (RG), '
    r'Acquisition Rate (AR), and Revenue Retention Rate (RRR) by GICS sector. '
    r'All values in percentages.}}'
)
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
    r'OLS with Newey-West standard errors (4 lags). '
    r'SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}'
)
_tab3b_note = (
    r'\textit{Note: Value-weighted industry-time-adjusted RRR quartile portfolio factor regressions. '
    r'Q1 = highest adjusted RRR, Q4 = lowest adjusted RRR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (4 lags). '
    r'SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}'
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
    r'OLS with Newey-West standard errors (4 lags). '
    r'SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}'
)
_tab4b_ar_note = (
    r'\textit{Note: Value-weighted industry-time-adjusted AR quartile portfolio factor regressions. '
    r'Q1 = highest adjusted AR, Q4 = lowest adjusted AR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (4 lags). '
    r'SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}'
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
_fmb_header = ' & '.join([''] + _fmb_col_labels)

_tab6_lines = []
_tab6_lines.append(f'\\begin{{tabular}}{{{_fmb_col_spec}}}')
_tab6_lines.append('\\toprule')
_tab6_lines.append(_fmb_header + ' \\\\')
_tab6_lines.append('\\midrule')

for _var in _fmb_vars_order:
    _var_data = fama_macbeth[fama_macbeth['Variable'] == _var]
    _display = _fmb_var_labels.get(_var, _var.replace('_', '\\_'))
    _coef_cells = []
    _se_cells = []
    for _spec in _fmb_specs:
        _sv = _var_data[_var_data['Spec'] == _spec]
        if len(_sv) > 0:
            _c = _sv.iloc[0]['Coefficient']
            _t = _sv.iloc[0]['t-stat']
            # Derive p-value from t-stat (two-tailed normal approximation)
            _p = 2 * _scipy_stats.norm.sf(abs(_t))
            _se = _se_from_t(_c, _t)
            _coef_cells.append(_fmt_coef(_c, _p))
            _se_cells.append(_fmt_se(_se) if not np.isnan(_se) else '')
        else:
            _coef_cells.append('')
            _se_cells.append('')
    _tab6_lines.append(f'  {_display} & ' + ' & '.join(_coef_cells) + ' \\\\')
    _tab6_lines.append('  \\quad & ' + ' & '.join(_se_cells) + ' \\\\')

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
_tab6_lines.append('\\vspace{2pt}')
_tab6_lines.append(
    r'{\footnotesize\textit{Note: Time-series averages of quarterly cross-sectional regression coefficients '
    r'(Fama-MacBeth). Dependent variable: average monthly excess return in quarter $t+1$. '
    r'Columns (1)--(4) use industry-time-adjusted signals; columns (5)--(8) use raw signals. '
    r'SEs in parentheses (Newey-West, 4 lags). $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}}'
)
_write_tabular(_tab6_lines, os.path.join(TABLE_DIR, 'tab_fmb.tex'))


# ============================================================================
# TABLE 7: Risk Statistics — Quartile portfolios (unchanged)
# ============================================================================
print("[Table 7] Risk statistics...")

# Build quartile-level risk statistics from the port_rrr_q time series
_MONTHS_PER_YEAR = 12

def _risk_row(ret_series, label):
    """Compute annualized risk metrics for a return series (log monthly returns)."""
    r = ret_series.dropna()
    ann_ret = r.mean() * _MONTHS_PER_YEAR * 100
    ann_vol = r.std() * np.sqrt(_MONTHS_PER_YEAR) * 100
    sharpe = (r.mean() / r.std() * np.sqrt(_MONTHS_PER_YEAR)) if r.std() > 0 else np.nan
    # Maximum drawdown on cumulative log-return series
    cumret = r.cumsum()
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) * 100
    max_dd = drawdown.min()
    # Downside beta: beta vs market on negative-market months
    mkt_aligned = port_rrr_q['MKT'].reindex(r.index).dropna()
    r_aligned = r.reindex(mkt_aligned.index)
    neg_mkt = mkt_aligned[mkt_aligned < 0]
    r_neg = r_aligned.reindex(neg_mkt.index)
    if len(r_neg) > 5 and neg_mkt.std() > 0:
        downside_beta = np.cov(r_neg.values, neg_mkt.values)[0, 1] / neg_mkt.var()
    else:
        downside_beta = np.nan
    # Sortino ratio (downside deviation using 0 as threshold)
    downside_ret = r[r < 0]
    downside_dev = np.sqrt((downside_ret ** 2).mean()) * np.sqrt(_MONTHS_PER_YEAR)
    sortino = (r.mean() * _MONTHS_PER_YEAR / downside_dev) if downside_dev > 0 else np.nan
    var5 = np.percentile(r, 5) * 100
    skew = r.skew()
    hit = (r > 0).mean() * 100
    return [label, fmt(ann_ret, 2), fmt(ann_vol, 2), fmt(sharpe, 4),
            fmt(max_dd, 2), fmt(downside_beta, 4), fmt(sortino, 4),
            fmt(var5, 2), fmt(skew, 4), fmt(hit, 1)]

_tab7_lines = []
_tab7_lines.append('\\begin{tabular}{lrrrrrrrrrr}')
_tab7_lines.append('\\toprule')
_tab7_lines.append(
    'Portfolio & Ret (\\%) & Vol (\\%) & Sharpe & MDD (\\%) & '
    'Down $\\beta$ & Sortino & VaR$_{5\\%}$ & Skew & Hit (\\%) \\\\'
)
_tab7_lines.append('\\midrule')

# Use port_rrr_q (quartile portfolios built in Step 0b)
for _port_col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4', 'MKT']:
    if _port_col in port_rrr_q.columns:
        _tab7_lines.append(' & '.join(_risk_row(port_rrr_q[_port_col], _port_col)) + ' \\\\')

_tab7_lines.append('\\bottomrule')
_tab7_lines.append('\\end{tabular}')
_tab7_lines.append('\\vspace{2pt}')
_tab7_lines.append(
    r'{\footnotesize\textit{Note: Annualized risk statistics for value-weighted RRR quartile portfolios '
    r'(industry-time adjusted signal). Q1 = high RRR, Q4 = low RRR, Q1-Q4 = long-short. '
    r'MDD = maximum drawdown. Down $\beta$ = beta on negative-market months. '
    r'VaR$_{5\%}$ = 5th percentile monthly return. Hit = fraction of positive months.}}'
)
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
    r'OLS with Newey-West standard errors (4 lags). '
    r'SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}'
)

_tab_nc_note = (
    r'\textit{Note: Excluding COVID-19 period (2020Q1--2021Q2). '
    r'Value-weighted adjusted RRR quartile portfolios. '
    r'Q1 = highest RRR, Q4 = lowest RRR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (4 lags). '
    r'SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}'
)
_tab_ew_note = (
    r'\textit{Note: Equal-weighted adjusted RRR quartile portfolios. '
    r'Q1 = highest RRR, Q4 = lowest RRR, Q1$-$Q4 = long-short. '
    r'OLS with Newey-West standard errors (4 lags). '
    r'SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}'
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
    r'OLS with Newey-West standard errors (4 lags). '
    r'SEs in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}'
)

print("  Computing placebo regressions...")
_reg_placebo = _run_inline_factor_reg(port_placebo_m, ff_factors)

_write_tabular(
    _build_panel_table(_reg_placebo, _pl_port_labels, _tab_pl_note),
    os.path.join(TABLE_DIR, 'tab_placebo.tex')
)

print("\n  All 8 tables generated.")


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
