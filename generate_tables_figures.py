"""
generate_tables_figures.py — RRR Financial Implications
========================================================
Writes the paper tables and the figure that the other scripts do not write
themselves (run_pipeline.py runs this as the second-to-last step).

It re-runs the analysis_v2.py data pipeline to rebuild the headline quartile
portfolios, reads the workbooks the earlier pipeline steps saved, and writes:

  Paper_LaTeX/tables/  tab_summary_stats, tab_portfolio_alphas_raw,
                       tab_growth_sort_alphas, tab_fmb, tab_risk_stats,
                       tab_portfolio_characteristics,
                       tab_headline_summary (CAPM check), tab_concentration,
                       tab_dividend_check
  Paper_LaTeX/figures/ cumret_rrr_q4_vw.pdf
  output/              headline_factor_regressions.xlsx (read by numbers_tex.py)

Trimmed 2026-10-20: the blocks for tables and figures the paper no longer uses
were removed; the full earlier version is in
archive/old_versions/generate_tables_figures.py.before-trim-20261020.
"""

import sys
import os
import re
import json
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
# Built from this file's location (Code/RRR-FI-IM), so the project can move.
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CODE_DIR))
OUTPUT_DIR = os.path.join(CODE_DIR, "output")
TABLE_DIR = os.path.join(BASE_DIR, "Paper_LaTeX", "tables")
FIGURE_DIR = os.path.join(BASE_DIR, "Paper_LaTeX", "figures")
DIAG_XLSX = os.path.join(OUTPUT_DIR, 'robustness_diagnostics.xlsx')   # robustness_diagnostics.py

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
    merge_signals_to_returns, safe_quartile, build_portfolio_returns,
    newey_west_ols, value_weighted_market,
)


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
COLORS = sns.color_palette("colorblind", 10)
C_T1, C_T2, C_T3, C_MKT = COLORS[3], COLORS[2], COLORS[0], COLORS[7]
C_Q1, C_Q2, C_Q3, C_Q4 = COLORS[3], COLORS[2], COLORS[1], COLORS[0]
C_LS = '#1a1a1a'   # long-short spread: near-black, deliberately dominant

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
    """Format float to dp decimal places, in the AMA style of JMR/JoM: at most three
    decimals (a larger dp is capped) and no zero before the decimal point (.123, -.045)."""
    if pd.isna(v):
        return ""
    s = f"{v:.{min(dp, 3)}f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


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

def _fmt_coef(coef, decimals=3):
    """Format coefficient with LaTeX minus sign. No significance stars (house style).
    AMA style: at most three decimals, no zero before the decimal point."""
    if pd.isna(coef):
        return ''
    sign = '$-$' if coef < 0 else ''
    s = f"{abs(coef):.{min(decimals, 3)}f}"
    if s.startswith('0.'):
        s = s[1:]
    return f"{sign}{s}"


def _fmt_se(se, decimals=3):
    """Format standard error in parentheses (AMA style, see fmt)."""
    if pd.isna(se):
        return ''
    return f"({fmt(se, decimals)})"


def _fmt_pval(pval, decimals=3):
    """Format an exact p-value in brackets, no leading zero, '< .001' floor below
    the display precision (e.g. [.003], [< .001]): the AMA rules of JMR/JoM, which
    require one space on either side of signs of relation."""
    if pd.isna(pval):
        return ''
    floor = 10 ** (-decimals)
    if pval < floor:
        return f"[< {floor:.{decimals}f}]".replace('0.', '.', 1)
    s = f"{pval:.{decimals}f}"
    if s.startswith('0.'):
        s = s[1:]
    elif s.startswith('-0.'):
        s = '-' + s[2:]
    return f"[{s}]"


# Tables whose first column carries long row labels and which therefore run past
# the text block. Converting that column to a tabularx X column lets the label
# wrap and absorb the slack, which keeps the body font the same size as every
# other table. Measured with scratchpad/measure_tables.py, which compiles each
# table on its own so the attribution is unambiguous.
_WRAP_LABEL_COLUMN = {
    'tab_concentration.tex',
    'tab_portfolio_characteristics.tex',
    'tab_summary_stats.tex',
    'tab_concentration_trim.tex',
    # Full width so the notes get the full text width: with three-decimal numbers
    # the narrower table made the notes wrap one line more and pushed the
    # growth-sort table past its page. Table 4 shares the layout, so both are listed.
    'tab_portfolio_alphas_raw.tex',
    'tab_growth_sort_alphas.tex',
}

_TABULAR_RE = re.compile(r'^(\s*)\\begin\{tabular\}\{([^}]*)\}\s*$')


def _wrap_first_column(lines):
    """Turn the leading 'l' column of a tabular into a wrapping tabularx X column."""
    out = []
    converted = False
    for ln in lines:
        m = _TABULAR_RE.match(ln)
        if m and not converted:
            indent, spec = m.group(1), m.group(2)
            body = spec.lstrip()
            if body.startswith('l'):
                rest = body[1:]
                out.append(indent + '\\begin{tabularx}{\\textwidth}'
                           '{@{}>{\\raggedright\\arraybackslash}X' + rest + '@{}}')
                converted = True
                continue
        # Some tables still close with "\end{tabular}%", a leftover trailing
        # comment from the old \resizebox wrapper, so match on the prefix.
        if converted and ln.strip().startswith('\\end{tabular}'):
            out.append(ln.replace('\\end{tabular}', '\\end{tabularx}'))
            continue
        out.append(ln)
    return out


def _write_tabular(lines, output_path):
    """Write a list of LaTeX lines to a .tex file (UTF-8)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.basename(output_path) in _WRAP_LABEL_COLUMN:
        lines = _wrap_first_column(lines)
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
# Factors first: phase 1 needs the FF risk-free rate.
ff_factors = phase1b_load_ff_factors()
_df_long, df_filtered, _industry_stats = phase1_load_and_diagnose(ff_factors)
returns = phase1c_load_monthly_returns()
# merge_signals_to_returns() is a backward-compat alias for build_holding_panel():
# 2-month formation lag, 3-month hold, one row per (firm, holding-month). This IS the
# canonical timing (analysis_v2.py commit 1fce67f) -- unchanged from before.
panel = merge_signals_to_returns(df_filtered, returns)

# Also read pre-computed Excel files
pooled_desc = pd.read_excel(os.path.join(OUTPUT_DIR, 'pooled_descriptives.xlsx'))
fama_macbeth = pd.read_excel(os.path.join(OUTPUT_DIR, 'fama_macbeth.xlsx'))
portfolio_char_xlsx = pd.read_excel(os.path.join(OUTPUT_DIR, 'portfolio_characteristics.xlsx'))

print("\n  Data loaded successfully.")


# ============================================================================
# BUILD THE HEADLINE PORTFOLIO RETURNS
# ============================================================================
# The raw-RRR quartile sort, built with the same two calls every pipeline script
# uses: a quartile per signal quarter (safe_quartile), then value-weighted
# monthly returns on formation-date market cap (build_portfolio_returns).
# It feeds the headline alpha table and the cumulative-return figure.
print("\n[Step 0b] Building the headline portfolio returns (canonical timing)...")

market_ret = value_weighted_market(panel)

panel['RRR_RAW_Q'] = panel.groupby('QUARTER')['RRR_PCT'].transform(safe_quartile)
port_rrr_raw_q = build_portfolio_returns(panel, 'RRR_RAW_Q', market_ret)
port_rrr_raw_q['MKT'] = market_ret.reindex(port_rrr_raw_q.index)

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
# 2026-09-29: the lag is the fixed rule floor(4(T/100)^(2/9)) in
# analysis_v2.newey_west_ols, not a data-dependent bandwidth; notes say so.
FACTOR_REG_NOTE = (
    r'\textit{Notes:} OLS with Newey-West standard errors (Bartlett kernel, '
    r'$\lfloor 4(T/100)^{2/9}\rfloor$ lags) in parentheses and exact $p$-values in '
    r'brackets for the alphas.'
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
        'FF3':     'Fama-French Three-Factor Model',
        'FF3+Mom': 'Carhart Four-Factor Model',
        'FF5':     'Fama-French Five-Factor Model',
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
    lines.append('\\scriptsize')
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
    """Escape % and replace leading - with $-$ in table cells.

    The % escape is idempotent. Several callers build their cells with a
    literal '\\%' suffix already attached (see _perf_rows_spec); escaping
    unconditionally turned those into '\\\\%', which LaTeX reads as a row
    break followed by a comment, silently truncating the row and swallowing
    the rest of the line. That corrupted tab_risk_stats.tex and
    tab_sample_composition.tex. Only escape a % that is not already escaped.
    """
    import re
    s = re.sub(r'(?<!\\)%', r'\\%', s)
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

# D6: the acquisition rate is removed as a construct, so its descriptive rows
# come out of the summary table too. Revenue growth stays: it is a plain
# descriptive here and the sorting variable for one robustness check.
_AR_VARS = {'Acq Rate (%)', 'Adj AR (%)'}
# 2026-09-27 (author): the industry-adjusted RRR is a robustness check only and
# is defined in the Methodology section, so it leaves the summary table.
_ADJ_RRR_VARS = {'Adj RRR (%)'}
# Row labels as the author set them by hand (2026-11-05).
_DISPLAY_LABELS = {'Rev Growth (%)': 'Rev. Growth (%)'}

for _, r in pooled_desc.iterrows():
    if r['Variable'] in _AR_VARS or r['Variable'] in _ADJ_RRR_VARS:
        continue
    _dp = 2 if r['Variable'] in _DOLLAR_VARS else 4
    _row = (
        f"{_DISPLAY_LABELS.get(r['Variable'], r['Variable'])} & {fmt_int(r['N'])} & {fmt(r['Mean'],_dp)} & "
        f"{fmt(r['Median'],_dp)} & {fmt(r['SD'],_dp)} & {fmt(r['P10'],_dp)} & "
        f"{fmt(r['P25'],_dp)} & {fmt(r['P75'],_dp)} & {fmt(r['P90'],_dp)} \\\\"
    )
    _tab1_lines.append(_latex_escape_row(_row))
_tab1_lines.append('\\bottomrule')
_tab1_lines.append('\\end{tabular}')
_tab1_lines.append('\\begin{tablenotes}')
_tab1_lines.append('\\small')
_tab1_lines.append(
    r'\item \textit{Notes:} Assets, Market Cap, and Revenue are in millions of USD; '
    r'\RRR{}, revenue growth, operating profit margin (PM), and operating ROA are '
    r'percentages. BTM is book-to-market, and MTE is market-to-equity.'
)
_tab1_lines.append('\\end{tablenotes}')
_tab1_lines.append('\\end{threeparttable}')
_write_tabular(_tab1_lines, os.path.join(TABLE_DIR, 'tab_summary_stats.tex'))


# ============================================================================
# TABLE 3a/3b: Portfolio Alphas (RRR) — split into raw + adjusted
# ============================================================================
print("[Table 3a/3b] Portfolio alphas (RRR)...")

_rrr_port_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4']

_tab3a_note = (
    r'\textit{Notes:} Value-weighted \RRR{} quartile portfolios; Q1 = highest \RRR{}, '
    r'Q4 = lowest, and Q1$-$Q4 = long-short. OLS with Newey-West standard errors '
    r'(Bartlett kernel, four lags) in parentheses and exact $p$-values in brackets '
    r'for the alphas.'
)

print("  Computing RRR regressions (raw)...")
_reg_rrr_raw = _run_inline_factor_reg(port_rrr_raw_q, ff_factors)
# Persist the headline regressions (added 2026-10-20): numbers_tex.py cites
# alphas and loadings from this file instead of retyping them into the text.
pd.DataFrame(_reg_rrr_raw).T.to_excel(os.path.join(OUTPUT_DIR, 'headline_factor_regressions.xlsx'))
_write_tabular(
    _build_panel_table(_reg_rrr_raw, _rrr_port_labels, _tab3a_note),
    os.path.join(TABLE_DIR, 'tab_portfolio_alphas_raw.tex')
)


# ============================================================================
# APPENDIX TABLE: factor regressions for the prior-period revenue-growth sort.
# Author note 2026-09-27: "I want the full FF table in the appendix". Reads the
# growth-sort portfolio returns persisted by signal_sort_robustness.py and
# applies the same regressions and layout as Table 4, so the two tables are
# directly comparable. The growth-sort results themselves stay in the body.
# ============================================================================
print("[Appendix] Factor regressions for the revenue-growth sort...")
_port_growth = pd.read_excel(
    os.path.join(OUTPUT_DIR, 'signal_sort_robustness.xlsx'),
    sheet_name='revgrowth_raw_ret', index_col=0, parse_dates=True)
_reg_growth = _run_inline_factor_reg(_port_growth, ff_factors)
_growth_ls_ff5 = _reg_growth['Q1-Q4_FF5']['alpha'] * 100
print(f"  Growth sort Q1-Q4 FF5 alpha: {_growth_ls_ff5:.4f}%/mo")
# Guard: must reproduce the growth-sort row of the robustness table, read from
# signal_sort_robustness.py's manifest (2026-10-20: was a hard-coded 1.0421).
with open(os.path.join(OUTPUT_DIR, 'signal_sort_manifest.json'), encoding='utf-8') as _fh:
    _ssm_growth = json.load(_fh)['signals']['revgrowth_raw']
_growth_expected = float(_ssm_growth['Q1Q4_ff5_alpha_pct_mo']) * 100
assert abs(_growth_ls_ff5 - _growth_expected) < 0.005, \
    f'growth-sort FF5 alpha {_growth_ls_ff5:.4f} does not reproduce {_growth_expected:.4f}'
_tab_growth_note = (
    r'\textit{Notes:} Value-weighted quartile portfolios sorted on prior-period revenue '
    r'growth; Q1 = highest growth, Q4 = lowest, and Q1$-$Q4 = long-short. Timing, '
    r'weighting, and estimation are identical to Table~\ref{tab:portfolio_alphas_raw}.'
)
_write_tabular(
    _build_panel_table(_reg_growth, _rrr_port_labels, _tab_growth_note),
    os.path.join(TABLE_DIR, 'tab_growth_sort_alphas.tex')
)


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
    'RRR_PCT':          '\\RRR{} (\\%)',
    'ACQ_RATE_PCT':     'Raw AR (\\%)',
    'SIZE_LAG':         'Size',
    'BTM_LAG':          'BTM',
    'PM_LAG':           'Profit Margin',
    'MKT_BETA':         'Market $\\beta$',
    # Monthly FM (2026-09-27) uses the quarter-end controls directly.
    'SIZE':             'Size (log market cap)',
    'BTM':              'Book-to-market',
    'PM_OPER_PCT':      'Operating profit margin (\\%)',
}

_n_fmb_cols = len(_fmb_specs)
_fmb_col_labels = [f'({i+1})' for i in range(_n_fmb_cols)]
_fmb_col_spec = 'l' + 'c' * _n_fmb_cols

# Group header: "Adjusted signal" vs "Raw signal", spanning however many consecutive
# columns share a group (run-length encoded so this does not hardcode a 2+2 split).
# 2026-10-20: columns (3)-(4) weight each firm by formation-date market cap.
_fmb_group_labels = ['Value-weighted' if 'value-weighted' in _s else 'Equal-weighted' for _s in _fmb_specs]
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
# The group header ("Adjusted signal" / "Raw signal") only earns its place when
# both groups are present; since 2026-09-27 the table is raw-only.
if len(_fmb_groups) > 1:
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
_tab6_lines.append('  $T$ (months) & ' + ' & '.join(_t_cells) + ' \\\\')
_tab6_lines.append('  Avg.\\ $N$ (firms) & ' + ' & '.join(_n_avg_cells) + ' \\\\')

_tab6_lines.append('\\bottomrule')
_tab6_lines.append('\\end{tabular}')
_tab6_lines.append('\\begin{tablenotes}')
_tab6_lines.append('\\small')
# Two sentences (D17): what the reader cannot infer from the table itself.
_tab6_lines.append(
    r'\item \textit{Notes:} Time-series averages of monthly cross-sectional slopes from '
    r'regressions of firm excess returns (percent per month) on \RRR{} and controls '
    r'measured at the quarter-end used for the portfolio sorts, with the same formation '
    r'and holding timing; columns (3) and (4) weight each firm by its formation-date market '
    r'capitalization. Newey-West standard errors in parentheses and exact $p$-values '
    r'in brackets.'
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
    ('Appraisal Ratio (FF3)',   'Appraisal Ratio (FF3)', 4, ''),
    ('Appraisal Ratio (FF5)',   'Appraisal Ratio (FF5)', 4, ''),
    ('Downside Beta',           'Downside $\\beta$',    4, ''),
    ('Upside Beta',             'Upside $\\beta$',      4, ''),
    ('Max Drawdown (%)',        'Max Drawdown',         2, '\\%'),
    ('Turnover (avg qtrly, %)', 'Turnover (per rebalancing)', 2, '\\%'),
    ('Turnover (annualized, %)', 'Turnover (annualized)', 2, '\\%'),
    ('Hit Rate vs Market (%)',  'Hit Rate vs.\\ Benchmark',1, '\\%'),
    # 'Hit Rate vs Q4 (%)' removed from the table by the author (hand edit, 2026-11-06).
]

_tab7_lines = []
_tab7_lines.append('\\begin{threeparttable}')
_tab7_lines.append('\\scriptsize')
_tab7_lines.append('\\begin{tabular}{lrrrrrr}')
_tab7_lines.append('\\toprule')
# 2026-09-27: the sample's value-weighted portfolio is called the "benchmark
# portfolio" throughout the paper (author note: "Market" was unclear).
_tab7_lines.append(' & ' + ' & '.join('Benchmark' if _c == 'MKT' else _c
                                     for _c in _perf_cols) + ' \\\\')
_tab7_lines.append('\\midrule')
for _src_label, _disp_label, _dp, _suffix in _perf_rows_spec:
    _cells = []
    for _c in _perf_cols:
        _v = _perf_metrics.loc[_src_label, _c]
        # Q4 against itself is not a statistic (it printed 0.0%); leave it blank.
        if _src_label == 'Hit Rate vs Q4 (%)' and _c == 'Q4':
            _cells.append('')
            continue
        _cells.append(fmt(_v, _dp) + _suffix if pd.notna(_v) else '')
    _row = f'{_disp_label} & ' + ' & '.join(_cells) + ' \\\\'
    _tab7_lines.append(_latex_escape_row(_row))
_tab7_lines.append('\\midrule')
_n_cells = [fmt_int(_perf_metrics.loc['N (months)', _c]) for _c in _perf_cols]
_tab7_lines.append('$N$ (months) & ' + ' & '.join(_n_cells) + ' \\\\')
_tab7_lines.append('\\bottomrule')
_tab7_lines.append('\\end{tabular}%')
_tab7_lines.append('\\begin{tablenotes}')
_tab7_lines.append('\\small')
_tab7_lines.append(
    # Note as the author shortened it by hand on 2026-11-06.
    r'\item \textit{Notes:} Value-weighted \RRR{} quartile portfolios; the benchmark is '
    r'the value-weighted portfolio of all sample firms, against which tracking error, '
    r'information ratio, and hit rate are measured for the long-only columns.'
)
_tab7_lines.append('\\end{tablenotes}')
_tab7_lines.append('\\end{threeparttable}')
_write_tabular(_tab7_lines, os.path.join(TABLE_DIR, 'tab_risk_stats.tex'))


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
_tab_pc_lines.append('\\scriptsize')
_tab_pc_lines.append('\\setlength{\\tabcolsep}{3pt}')
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
    r'\item \textit{Notes:} Time-series averages of cross-sectional means by \RRR{} '
    r'quartile, Q1 = highest. SoNR is the share of current-period revenue from newly '
    r'acquired customers, $\text{Rev}^{\text{new}}_{i,t} / \text{Rev}_{i,t} \times 100$; '
    r'revenue and market capitalization are in millions of USD.'
)
_tab_pc_lines.append('\\end{tablenotes}')
_tab_pc_lines.append('\\end{threeparttable}')
_write_tabular(_tab_pc_lines, os.path.join(TABLE_DIR, 'tab_portfolio_characteristics.tex'))


# ============================================================================
# TABLE (Appendix): CAPM regression of the benchmark portfolio on the S&P 500
# (file name tab_headline_summary kept from an earlier layout)
# ============================================================================
print("[Table] CAPM check: benchmark portfolio on the S&P 500...")

_capm = pd.read_excel(os.path.join(OUTPUT_DIR, 'capm_vs_sp500.xlsx')).iloc[0]

# 2026-09-29 (proofreading): coefficient / SE / exact p layout like every other
# regression table, no t-stat column, no empty column, rule under the header.
# The beta p-value tests beta = 1 (the Data text reads beta as "above one");
# normal reference distribution, as statsmodels' HAC alpha_p uses.
_beta_p_vs_one = 2 * _scipy_stats.norm.sf(abs((_capm['beta'] - 1.0) / _capm['beta_se']))
_tab_hl_lines = []
_tab_hl_lines.append('\\begin{threeparttable}')
_tab_hl_lines.append('\\footnotesize')
_tab_hl_lines.append('\\begin{tabular}{lr}')
_tab_hl_lines.append('\\toprule')
_tab_hl_lines.append(' & Estimate \\\\')
_tab_hl_lines.append('\\midrule')
_tab_hl_lines.append(f"$\\alpha$ (\\%/mo) & {_fmt_coef(_capm['alpha_month']*100)} \\\\")
_tab_hl_lines.append(f" & ({fmt(_capm['alpha_se']*100,4)}) \\\\")
_tab_hl_lines.append(f" & {_fmt_pval(_capm['alpha_p'])} \\\\")
_tab_hl_lines.append(f"$\\beta$ & {_fmt_coef(_capm['beta'])} \\\\")
_tab_hl_lines.append(f" & ({fmt(_capm['beta_se'],4)}) \\\\")
_tab_hl_lines.append(f" & {_fmt_pval(_beta_p_vs_one)} \\\\")
_tab_hl_lines.append('\\midrule')
_tab_hl_lines.append(f"$R^2$ & {fmt(_capm['r2'],4)} \\\\")
_tab_hl_lines.append(f"$N$ (months) & {fmt_int(_capm['n_obs'])} \\\\")
_tab_hl_lines.append('\\bottomrule')
_tab_hl_lines.append('\\end{tabular}')
_tab_hl_lines.append('\\begin{tablenotes}')
_tab_hl_lines.append('\\small')
_tab_hl_lines.append(
    # Note as the author shortened it by hand on 2026-11-10.
    r'\item \textit{Notes:} The excess return of the benchmark portfolio, the '
    r'value-weighted portfolio of all sample firms, regressed on that of the S\&P 500, over every '
    r'month the market series covers.'
)
_tab_hl_lines.append('\\end{tablenotes}')
_tab_hl_lines.append('\\end{threeparttable}')
_write_tabular(_tab_hl_lines, os.path.join(TABLE_DIR, 'tab_headline_summary.tex'))


# ============================================================================
# TABLE (Appendix): Concentration, effective N and HHI of each leg by quarter
# ============================================================================
print("[Table] Concentration (effective N and HHI by leg)...")

_t1_effn_q1 = pd.read_excel(DIAG_XLSX, sheet_name='T1_EffN_Q1')
_t1_effn_q4 = pd.read_excel(DIAG_XLSX, sheet_name='T1_EffN_Q4')

_tab_conc_lines = []
_tab_conc_lines.append('\\begin{threeparttable}')
_tab_conc_lines.append('\\footnotesize')
_tab_conc_lines.append('\\begin{tabular}{lrrrr}')
_tab_conc_lines.append('\\toprule')
_tab_conc_lines.append(
    '\\multicolumn{5}{l}{\\textbf{Effective-$N$ / Herfindahl Concentration by Quarter}} \\\\'
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
_tab_conc_lines.append('\\bottomrule')
_tab_conc_lines.append('\\end{tabular}')
_tab_conc_lines.append('\\begin{tablenotes}')
_tab_conc_lines.append('\\small')
_tab_conc_lines.append(
    r'\item \textit{Notes:} The Herfindahl-Hirschman index (HHI) of formation-cap weights and its '
    r'inverse, the effective number of names, for the long (Q1) and short (Q4) legs '
    r'across sample quarters. A leg holding $n$ equally weighted names has an '
    r'effective $N$ of $n$.'
)
_tab_conc_lines.append('\\end{tablenotes}')
_tab_conc_lines.append('\\end{threeparttable}')
_write_tabular(_tab_conc_lines, os.path.join(TABLE_DIR, 'tab_concentration.tex'))


# ============================================================================
# TABLE (NEW, item 12): Dividend-Yield Check
# ============================================================================
print("[Table] Dividend-yield check (Q1 vs Q4; resolves price-only-returns concern)...")

_div_pooled = pd.read_excel(DIAG_XLSX, sheet_name='T6_DivYield_Pooled').set_index('RRR_Q_MAIN')
_div_firm = pd.read_excel(DIAG_XLSX, sheet_name='T6_DivYield_FirmLevel').set_index('RRR_Q_MAIN')
_div_fail = pd.read_excel(DIAG_XLSX, sheet_name='T6_Failures')
# Count against the panel the dividend check actually ran on, not a hardcoded
# sample size from an earlier vintage.
_n_div_universe = int(len(_div_fail) + _div_firm['N_FIRMS'].sum()) \
    if 'N_FIRMS' in _div_firm.columns else 137
_n_div_covered = _n_div_universe - len(_div_fail)

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
# 2026-09-27 (author note "out"): the sentence on dividend-history coverage and
# unresolved tickers is removed from the note.
_tab_div_lines.append(
    r'\item \textit{Notes:} Trailing dividend yield by \RRR{} quartile, Q1 = highest. '
    r'Because the returns exclude dividends, the Q1$-$Q4 yield gap shows how far that '
    r'convention could shift the long-short return.'
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


def cumulative_wealth_pct(returns, scale=100.0):
    """Compounded cumulative return path in percent.

    Replaces the old `.cumsum() * 100`. The series carried here are SIMPLE
    monthly returns (build_portfolio_returns deliberately avoids log returns
    because a value-weighted mean of log returns is biased), so summing them
    is neither a log-cumulation nor a wealth path. It understates compounding
    on the winners and, for the long-short spread, misstates the shape of the
    curve outright. Compound properly instead.
    """
    r = returns.fillna(0.0)
    return ((1.0 + r).cumprod() - 1.0) * scale


def format_stats_box(stats_summary, col='Q1-Q4'):
    """Build the annotation text for a portfolio's key statistics.

    Values are READ from output/performance_metrics_summary.xlsx rather than
    recomputed here, so the figure and Table 7 can never disagree.
    """
    if stats_summary is None or col not in stats_summary.columns:
        return None

    def g(row, fmt, suffix=''):
        if row not in stats_summary.index:
            return 'n/a'
        v = stats_summary.loc[row, col]
        try:
            if v != v:
                return 'n/a'
            return format(float(v), fmt) + suffix
        except (TypeError, ValueError):
            return 'n/a'

    lines = [
        f'{col} long-short',
        f'Ann. return   {g("Ann Return (%)", ".2f", "%")}',
        f'Ann. vol      {g("Ann Vol (%)", ".2f", "%")}',
        f'Sharpe        {g("Sharpe Ratio", ".2f")}',
        f'Tracking err. {g("Tracking Error (%)", ".2f", "%")}',
        f'Info. ratio   {g("Information Ratio", ".2f")}',
        f'Max drawdown  {g("Max Drawdown (%)", ".2f", "%")}',
    ]
    return chr(10).join(lines)


def load_perf_summary():
    """Load the persisted performance battery, or None if it has not been run."""
    path = os.path.join(OUTPUT_DIR, 'performance_metrics_summary.xlsx')
    if not os.path.exists(path):
        print(f"  NOTE: {os.path.basename(path)} not found; stats boxes will be omitted.")
        return None
    try:
        return pd.read_excel(path, sheet_name='Metrics', index_col=0)
    except Exception as exc:
        print(f"  NOTE: could not read performance summary ({type(exc).__name__}); "
              f"stats boxes will be omitted.")
        return None


PERF_SUMMARY = None   # populated lazily on first use


def draw_cumret_panel(port, title, outname, label_suffix='RRR', include_ls=True):
    """Cumulative-return figure: quartiles, market, and the long-short spread.

    The long-short line is what a portfolio manager actually trades, so it is
    drawn on the same axis as the quartile legs, in near-black at full weight.
    If its range dwarfs the legs the axis is split into two stacked panels so
    neither is squashed.
    """
    global PERF_SUMMARY
    if PERF_SUMMARY is None:
        PERF_SUMMARY = load_perf_summary()

    plot_df = prepend_zero_row(port)
    # Author request 2026-09-27: the benchmark and the long-short portfolio are
    # the solid lines; the four quartile portfolios are dashed or dotted.
    legs = [
        ('Q1', C_Q1, '--', 1.8),
        ('Q2', C_Q2, (0, (5, 2, 1, 2)), 1.5),
        ('Q3', C_Q3, '-.', 1.5),
        ('Q4', C_Q4, ':', 1.8),
        ('MKT', C_MKT, '-', 1.8),
    ]

    def leg_label(col):
        if col == 'MKT':
            # The sample's own value-weighted portfolio of all firms, not the
            # Fama-French market factor (author note 2026-09-27: "Market" was
            # unclear).
            return 'Benchmark portfolio'
        tag = 'High' if col == 'Q1' else 'Low' if col == 'Q4' else 'Mid'
        return f'{col} ({tag} {label_suffix})'

    curves = {c: cumulative_wealth_pct(plot_df[c])
              for c, _, _, _ in legs if c in plot_df.columns}
    ls_col = 'Q1-Q4'
    has_ls = include_ls and ls_col in plot_df.columns
    ls_curve = cumulative_wealth_pct(plot_df[ls_col]) if has_ls else None

    # Split into two panels only when one series would flatten the other.
    split = False
    if has_ls and curves:
        leg_span = max(abs(c.max() - c.min()) for c in curves.values())
        ls_span = abs(ls_curve.max() - ls_curve.min())
        if leg_span > 0 and (ls_span / leg_span > 2.5 or leg_span / ls_span > 2.5):
            split = True

    if split:
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(9, 7.5), sharex=True,
            gridspec_kw={'height_ratios': [2.1, 1.0]})
    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax2 = None

    for col, color, ls, lw in legs:
        if col in curves:
            ax.plot(curves[col].index, curves[col].values,
                    label=leg_label(col), color=color, linestyle=ls, linewidth=lw)

    target = ax2 if split else ax
    if has_ls:
        target.plot(ls_curve.index, ls_curve.values,
                    label='Q1-Q4 (long-short)', color=C_LS, linestyle='-', linewidth=2.4)
        target.axhline(0, color='gray', linewidth=0.5)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax.set_ylabel('Cumulative Return (%)')
    if title:
        ax.set_title(title)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.legend(loc='upper left', fontsize=9)

    if split:
        ax2.set_ylabel('Cumulative Return (%)')
        ax2.set_xlabel('Date')
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
        ax2.legend(loc='upper left', fontsize=9)
    else:
        ax.set_xlabel('Date')

    # Notes line 194: no statistics box on the figure. Every number it carried
    # (return, volatility, Sharpe, alpha) is in Table~\ref{tab:risk_stats} and
    # Table~\ref{tab:headline_summary}, where it can be read at full precision.
    # format_stats_box() is left in place, unused, in case it is wanted again.

    fig.tight_layout()
    save_fig(fig, outname)


def prepend_zero_row(df, start_date='2017-01-01'):
    """Prepend a row of zeros at start_date so cumulative return graphs begin at 0.
    If start_date is already in the index, no row is prepended."""
    start = pd.Timestamp(start_date)
    if start in df.index:
        return df
    zero_row = pd.DataFrame([[0.0] * len(df.columns)], columns=df.columns, index=[start])
    return pd.concat([zero_row, df]).sort_index()


# ============================================================================
# FIGURE 8: Cumulative returns by RRR QUARTILE (key figure)
# Y-axis: Cumulative Return (%)
# ============================================================================
print("[Figure 8] Cumulative returns by RRR quartile...")

# NOTE: port_rrr_q is built from ADJ_RRR_PCT (Step 0b), so this is the
# industry x time adjusted sort. The old title said "RRR Quartile", which
# described the raw sort and was simply wrong. Corrected below.
# 2026-09-27: the paper's headline is the RAW signal (D12b), so Figure 2 now
# plots the raw sort, port_rrr_raw_q, the same series behind
# tab_portfolio_alphas_raw (Table 4). No in-figure title: the caption carries it.
draw_cumret_panel(
    port_rrr_raw_q,
    '',
    'cumret_rrr_q4_vw',
    label_suffix='RRR')


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


# ============================================================================
# ORPHAN CHECK
# ============================================================================
# A table that is generated but never \input anywhere is how deleted constructs
# survive a revision: the file keeps regenerating, keeps carrying whatever it
# carried, and nobody reads it because it never reaches the PDF. Three separate
# QA findings in the 2026-09 round traced back to exactly that. Report them on
# every run rather than relying on someone thinking to look.
_SECTION_DIR = os.path.join(os.path.dirname(TABLE_DIR), 'sections')
try:
    _section_text = ''
    for _f in sorted(os.listdir(_SECTION_DIR)):
        if _f.endswith('.tex'):
            with open(os.path.join(_SECTION_DIR, _f), encoding='utf-8') as _fh:
                _section_text += _fh.read()

    _orphans = [f for f in sorted(os.listdir(TABLE_DIR))
                if f.endswith('.tex') and f[:-4] not in _section_text]
    print("\nOrphan check:")
    if _orphans:
        print(f"  WARNING: {len(_orphans)} generated table(s) are never \\input "
              f"in sections/ and will not reach the PDF:")
        for _f in _orphans:
            print(f"    {_f}")
        print("  Either reference them or move them to tables/_removed_<date>/.")
    else:
        print("  OK: every generated table is referenced in sections/.")
except FileNotFoundError:
    print("\nOrphan check: skipped (sections/ not found)")
