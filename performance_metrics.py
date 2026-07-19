"""
performance_metrics.py -- RRR Financial Implications
=======================================================
Task 1: Full performance/risk metrics battery (Sharpe, Sortino, tracking error,
        information ratio, strategy turnover, hit rate, downside/upside beta,
        rolling drawdown, rolling upside-capture, max drawdown) for the RRR
        quartile portfolios (Q1-Q4) and the long-short, under the corrected
        k=2-month-formation-gap / 3-month-hold timing.
Task 2: Share of Retained Revenue (SoRR = ADJ_SRR_PCT) subsection -- verifies
        the signal is unaffected by the timing rebuild, sorts on it under the
        identical corrected timing, reports its long-short FF3/FF5 alpha, and
        directly quantifies the "SoRR is a low-turnover, stable signal" claim
        against RRR (signal-level turnover, quartile-transition stability, and
        the RRR-SoRR correlation implied by their shared numerator).

This script imports build_holding_panel(), build_portfolio_returns(),
value_weighted_market(), run_factor_regressions(), safe_quartile(), and
newey_west_ols() from analysis_v2.py -- the verified source of truth for
portfolio timing (commit 1fce67f) -- rather than reimplementing portfolio
construction. analysis_v2.py and the frozen 2025-06-04b main.py are NOT
modified by this file.

Modeling choices made explicit here (see also the printed run header):
  * "The market" for every relative metric (tracking error, information
    ratio, downside beta, upside beta, hit rate) is the SAMPLE's own
    value-weighted market portfolio (value_weighted_market(panel)), per the
    task instruction for tracking error / information ratio. This is
    extended to downside/upside beta for internal consistency across the
    whole battery. NOTE: this differs from analysis_v2.run_risk_analysis(),
    which uses the Fama-French Mkt-RF (broad CRSP) series for downside beta;
    the two are not directly comparable.
  * The headline RRR signal is ADJ_RRR_PCT (industry x time adjusted), the
    project's stated main-result signal (see project CLAUDE.md).
  * Sharpe and Sortino are computed on returns in excess of the Ken French
    monthly RF; the Q1-Q4 long-short is already zero-cost, so RF cancels out
    algebraically and is not separately subtracted (matches the existing
    convention in analysis_v2.run_factor_regressions()).
  * Sortino's downside deviation uses the Sortino & van der Meer (1991)
    definition: RMS of min(excess_return, 0) averaged over ALL periods (not
    just the negative subset) -- a stricter, more standard definition than
    either of the two ad hoc versions already in this repo
    (analysis_v2.run_risk_analysis and generate_tables_figures._risk_row).
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

CODE_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM"
sys.path.insert(0, CODE_DIR)

from analysis_v2 import (
    phase1_load_and_diagnose,
    phase1b_load_ff_factors,
    phase1c_load_monthly_returns,
    build_holding_panel,
    build_portfolio_returns,
    value_weighted_market,
    run_factor_regressions,
    run_portfolio_analysis,
    safe_quartile,
    OUTPUT_DIR,
    FORM_LAG_MONTHS,
    HOLD_MONTHS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
MONTHS_PER_YEAR = 12
ROLLING_WINDOW_MONTHS = 12          # trailing window for the rolling upside-capture series
MIN_OBS_CONDITIONAL_BETA = 6        # minimum down-/up-months required to fit a conditional beta
SRR_VERIFY_TOL = 1e-8               # tolerance for the SRR_PCT / ADJ_SRR_PCT recompute check

QUARTILE_COLS = ['Q1', 'Q2', 'Q3', 'Q4']
LONG_SHORT_COL = 'Q1-Q4'
MARKET_COL = 'MKT'
REPORT_COLS = QUARTILE_COLS + [LONG_SHORT_COL, MARKET_COL]

RRR_SIGNAL_COL = 'ADJ_RRR_PCT'      # headline signal: industry x time adjusted RRR
SRR_SIGNAL_COL = 'ADJ_SRR_PCT'      # headline signal: industry x time adjusted SoRR

ROW_ORDER = [
    'Ann Return (%)',
    'Ann Vol (%)',
    'Sharpe Ratio',
    'Sortino Ratio',
    'Tracking Error (%)',
    'Information Ratio',
    'Downside Beta',
    'Upside Beta',
    'Max Drawdown (%)',
    'Turnover (avg qtrly, %)',
    'Hit Rate vs Market (%)',
    'Hit Rate vs Q4 (%)',
    'N (months)',
]

METRIC_NOTES = {
    'Ann Return (%)': "Raw annualized value-weighted return actually earned holding the portfolio; context for the risk-adjusted ratios below.",
    'Ann Vol (%)': "Annualized standard deviation of monthly returns; total risk, the denominator of the Sharpe ratio.",
    'Sharpe Ratio': "Return per unit of total risk, in excess of the risk-free rate; the standard risk-adjusted-return benchmark.",
    'Sortino Ratio': "Like Sharpe, but the denominator counts only downside volatility, so large positive swings are not treated as 'risk'; isolates bad-state variance.",
    'Tracking Error (%)': "Volatility of the portfolio's return relative to the sample's own value-weighted market portfolio; how much the strategy deviates from the market, in either direction.",
    'Information Ratio': "Active return over tracking error relative to the sample's own market portfolio; distinguishes consistent skill from noisy market exposure.",
    'Downside Beta': "Beta on the sample market estimated only in months the sample market fell; how much of the portfolio's decline is systematic (market-driven) risk.",
    'Upside Beta': "Beta on the sample market estimated only in months the sample market rose; a portfolio with upside beta > downside beta has asymmetric, favorable market exposure.",
    'Max Drawdown (%)': "Largest peak-to-trough decline in cumulative value; a path-dependent tail-risk measure that volatility alone does not capture.",
    'Turnover (avg qtrly, %)': "Fraction of portfolio value traded at each quarterly re-formation (0.5 x sum of |weight changes|, new formation weights vs. prior weights drifted by realized returns); the implicit trading/implementability cost of the strategy.",
    'Hit Rate vs Market (%)': "Fraction of months the portfolio outperforms the sample market; a simple, distribution-free consistency measure.",
    'Hit Rate vs Q4 (%)': "Fraction of months Q1 outperforms Q4; tests directly whether the long-short edge is persistent or concentrated in a few months (headline number: the Q1 column).",
    'N (months)': "Sample size underlying every ratio above (months with both a valid sample-market return and a valid Ken French risk-free rate).",
}


# =============================================================================
# SECTION A -- CORE PER-SERIES RISK/PERFORMANCE METRICS
# =============================================================================

def excess_return(col_name, raw_series, rf_series):
    """Return raw_series minus the risk-free rate, EXCEPT for the Q1-Q4 long-short,
    which is already a zero-cost (self-financing) spread, so RF algebraically
    cancels: (R_Q1 - RF) - (R_Q4 - RF) = R_Q1 - R_Q4. Matches the convention already
    used in analysis_v2.run_factor_regressions()."""
    if col_name == LONG_SHORT_COL:
        return raw_series.copy()
    aligned = pd.concat([raw_series, rf_series], axis=1).dropna()
    return aligned.iloc[:, 0] - aligned.iloc[:, 1]


def sharpe_ratio(excess_ret):
    """Annualized Sharpe ratio = mean(excess return) / std(excess return), scaled to
    an annual basis. Business definition: risk-adjusted return per unit of TOTAL risk."""
    excess_ret = excess_ret.dropna()
    mu = excess_ret.mean() * MONTHS_PER_YEAR
    sigma = excess_ret.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    return mu / sigma if sigma > 0 else np.nan


def downside_deviation(excess_ret, mar=0.0):
    """Annualized downside deviation (Sortino & van der Meer 1991 definition): the
    root-mean-square shortfall below the minimum acceptable return (MAR = 0, i.e.
    the RF hurdle since excess_ret is already net of RF), averaged over ALL months
    (not just the negative subset) and annualized by sqrt(12)."""
    excess_ret = excess_ret.dropna()
    shortfall = np.minimum(excess_ret.values - mar, 0.0)
    monthly_dd = np.sqrt(np.mean(shortfall ** 2))
    return monthly_dd * np.sqrt(MONTHS_PER_YEAR)


def sortino_ratio(excess_ret):
    """Annualized Sortino ratio = mean(excess return) / downside deviation."""
    mu = excess_ret.dropna().mean() * MONTHS_PER_YEAR
    dd = downside_deviation(excess_ret)
    return mu / dd if dd > 0 else np.nan


def tracking_error(port_ret, mkt_ret):
    """Annualized standard deviation of the active return (portfolio - sample market)."""
    active = (port_ret - mkt_ret).dropna()
    return active.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)


def information_ratio(port_ret, mkt_ret):
    """Annualized mean active return / annualized tracking error."""
    active = (port_ret - mkt_ret).dropna()
    te = active.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    mu = active.mean() * MONTHS_PER_YEAR
    return mu / te if te > 0 else np.nan


def hit_rate(a, b):
    """Fraction of overlapping months a's return strictly exceeds b's return."""
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.empty:
        return np.nan
    return (aligned.iloc[:, 0] > aligned.iloc[:, 1]).mean()


def conditional_beta(port_ret, mkt_ret, condition):
    """OLS beta of port_ret on mkt_ret, restricted to months where the sample
    market return is negative (condition='down') or positive (condition='up').
    Symmetric definitions: downside beta uses down-market months, upside beta
    uses up-market months. Returns (beta, n_obs)."""
    aligned = pd.concat([port_ret, mkt_ret], axis=1).dropna()
    aligned.columns = ['p', 'm']
    if condition == 'down':
        sub = aligned[aligned['m'] < 0]
    elif condition == 'up':
        sub = aligned[aligned['m'] > 0]
    else:
        raise ValueError(f"condition must be 'down' or 'up', got {condition!r}")
    if len(sub) < MIN_OBS_CONDITIONAL_BETA or sub['m'].var() == 0:
        return np.nan, len(sub)
    model = sm.OLS(sub['p'], sm.add_constant(sub['m'])).fit()
    return model.params['m'], len(sub)


def drawdown_series(ret):
    """Running (expanding) drawdown from the highest prior wealth level, compounding
    SIMPLE monthly returns: wealth_t = prod(1+r), drawdown_t = wealth_t/running_max-1
    (<=0; e.g. -0.20 = 20% below the running peak). Applied identically to the
    Q1-Q4 long-short (as if $1 were compounded through the zero-cost spread) --
    standard practice for reporting long-short cumulative drawdown."""
    r = ret.dropna().sort_index()
    wealth = (1 + r).cumprod()
    running_max = wealth.cummax()
    return wealth / running_max - 1


def rolling_upside_capture(port_ret, mkt_ret, window=ROLLING_WINDOW_MONTHS):
    """Rolling trailing-`window`-month upside-capture ratio (%): among the months
    within the window where the sample MARKET return is positive, the ratio of the
    portfolio's compounded return to the market's compounded return over exactly
    those up-months. >100 means the portfolio gained more than the market during
    market rallies within that window; undefined (NaN) if the window has no
    up-months or the market's compounded up-month return is exactly zero."""
    aligned = pd.concat([port_ret, mkt_ret], axis=1).dropna()
    aligned.columns = ['p', 'm']
    aligned = aligned.sort_index()
    out = pd.Series(index=aligned.index, dtype=float)
    for i in range(len(aligned)):
        if i + 1 < window:
            out.iloc[i] = np.nan
            continue
        win = aligned.iloc[i + 1 - window: i + 1]
        up = win[win['m'] > 0]
        if up.empty:
            out.iloc[i] = np.nan
            continue
        port_cum = (1 + up['p']).prod() - 1
        mkt_cum = (1 + up['m']).prod() - 1
        out.iloc[i] = (port_cum / mkt_cum) * 100 if mkt_cum != 0 else np.nan
    return out


# =============================================================================
# SECTION B -- TURNOVER
# =============================================================================

def compute_portfolio_turnover(panel, bucket_col):
    """Strategy turnover: the fraction of portfolio VALUE that must be traded at
    each quarterly rebalance (does not exist elsewhere in the project; built here).

    Standard definition: Turnover_t = 0.5 * sum_i |w_new_i,t - w_end_i,t-1|, where
      * w_new_i,t   = firm i's freshly-formed formation-market-cap weight in the
                       INCOMING quarter's bucket (sums to 1 across that bucket).
      * w_end_i,t-1 = firm i's weight in the OUTGOING quarter's bucket, DRIFTED
                       forward by that firm's realized compounded holding-period
                       return (no interim trading), then renormalized to sum to 1.
    Averaged over all quarterly rebalances between calendar-adjacent quarters
    (transitions across a data gap, if any, are excluded and counted separately).
    Bounded in [0, 1] for a single long (or short) leg. Turnover of the Q1-Q4
    long-short is reported as the SUM of the two legs' turnover (both legs must be
    traded independently each quarter).

    Returns (turnover_avg: dict, turnover_n: dict, turnover_detail: dict of lists,
    n_skipped_gap: int).
    """
    df = panel.copy()
    df['MCAP_FORM'] = pd.to_numeric(df['MCAP_FORM'], errors='coerce')
    df['RET_SIMPLE'] = pd.to_numeric(df['RET_SIMPLE'], errors='coerce')
    df = df.dropna(subset=['MCAP_FORM', 'RET_SIMPLE', bucket_col])

    firm_q = (
        df.groupby(['FIRM', 'QUARTER'])
          .agg(BUCKET=(bucket_col, 'first'),
               MCAP_FORM=('MCAP_FORM', 'first'),
               COMPOUND_RET=('RET_SIMPLE', lambda x: float(np.prod(1.0 + x.values) - 1.0)),
               N_HOLD_MONTHS=('RET_SIMPLE', 'size'))
          .reset_index()
    )

    quarters = sorted(firm_q['QUARTER'].unique())
    turnover_detail = {b: [] for b in QUARTILE_COLS}
    n_skipped_gap = 0

    for q_prev, q_curr in zip(quarters[:-1], quarters[1:]):
        p_prev = pd.Timestamp(q_prev).to_period('Q')
        p_curr = pd.Timestamp(q_curr).to_period('Q')
        if (p_curr.ordinal - p_prev.ordinal) != 1:
            n_skipped_gap += 1
            continue
        for b in QUARTILE_COLS:
            prev = firm_q[(firm_q['QUARTER'] == q_prev) & (firm_q['BUCKET'] == b)]
            curr = firm_q[(firm_q['QUARTER'] == q_curr) & (firm_q['BUCKET'] == b)]
            if prev.empty or curr.empty:
                continue

            w_start = prev.set_index('FIRM')['MCAP_FORM']
            w_start = w_start / w_start.sum()
            growth = 1.0 + prev.set_index('FIRM')['COMPOUND_RET']
            w_end = w_start * growth
            w_end = w_end / w_end.sum()

            w_new = curr.set_index('FIRM')['MCAP_FORM']
            w_new = w_new / w_new.sum()

            all_firms = w_end.index.union(w_new.index)
            w_end_full = w_end.reindex(all_firms, fill_value=0.0)
            w_new_full = w_new.reindex(all_firms, fill_value=0.0)

            turnover = 0.5 * (w_new_full - w_end_full).abs().sum()
            turnover_detail[b].append({'quarter_curr': q_curr, 'turnover': turnover})

    turnover_avg, turnover_n = {}, {}
    for b in QUARTILE_COLS:
        vals = [x['turnover'] for x in turnover_detail[b]]
        turnover_avg[b] = float(np.mean(vals)) if vals else np.nan
        turnover_n[b] = len(vals)
    turnover_avg[LONG_SHORT_COL] = turnover_avg['Q1'] + turnover_avg['Q4']
    turnover_n[LONG_SHORT_COL] = min(turnover_n['Q1'], turnover_n['Q4'])

    return turnover_avg, turnover_n, turnover_detail, n_skipped_gap


def signal_level_turnover(df_filtered, signal_col):
    """SIGNAL-level (not $-weighted-portfolio-level) quarter-to-quarter turnover of
    a cross-sectional signal: (a) mean/median absolute quarter-to-quarter change in
    the signal value per firm, in percentage points and normalized by the signal's
    own average cross-sectional SD; (b) the fraction of firms whose quartile bucket
    (sorted within quarter via safe_quartile, Q1=highest, matching the portfolio-
    sort convention used everywhere else) differs from the prior quarter (churn),
    and its complement (stay rate)."""
    d = df_filtered[[signal_col]].copy()
    d[signal_col] = pd.to_numeric(d[signal_col], errors='coerce')
    d = d.reset_index().sort_values(['FIRM', 'DATE'])

    d['PRIOR_VAL'] = d.groupby('FIRM')[signal_col].shift(1)
    d['ABS_CHANGE'] = (d[signal_col] - d['PRIOR_VAL']).abs()

    cross_sectional_sd = d.groupby('DATE')[signal_col].std().mean()

    q_col = '_QBUCKET'
    d[q_col] = d.groupby('DATE')[signal_col].transform(safe_quartile)
    d['PRIOR_Q'] = d.groupby('FIRM')[q_col].shift(1)
    valid_q = d.dropna(subset=[q_col, 'PRIOR_Q'])
    stay_rate = (valid_q[q_col] == valid_q['PRIOR_Q']).mean()

    per_q_stay = {}
    for q in QUARTILE_COLS:
        sub = valid_q[valid_q['PRIOR_Q'] == q]
        per_q_stay[q] = (sub[q_col] == q).mean() if len(sub) > 0 else np.nan

    return {
        'signal': signal_col,
        'mean_abs_change_pp': d['ABS_CHANGE'].mean(),
        'median_abs_change_pp': d['ABS_CHANGE'].median(),
        'cross_sectional_sd_pp': cross_sectional_sd,
        'mean_abs_change_norm': (d['ABS_CHANGE'].mean() / cross_sectional_sd
                                  if cross_sectional_sd else np.nan),
        'quartile_stay_rate': stay_rate,
        'quartile_churn_rate': 1 - stay_rate,
        'per_quartile_stay_rate': per_q_stay,
        'n_firm_quarter_changes': int(d['ABS_CHANGE'].notna().sum()),
        'n_quartile_transitions': int(len(valid_q)),
    }


# =============================================================================
# SECTION C -- SoRR-SPECIFIC VERIFICATION
# =============================================================================

def verify_srr_computation(df_filtered):
    """Confirm SRR_PCT / ADJ_SRR_PCT still compute correctly after the timing
    rebuild. Two independent checks, since the timing fix (commit 1fce67f) only
    touched build_holding_panel()/portfolio formation, not Phase 1's signal
    construction, so this is expected to be unaffected -- verified, not assumed:
      (1) Recompute SRR_PCT = (#RETURNING_CUSTOMERS / #TOTAL_REVENUE) * 100 and
          ADJ_SRR_PCT = SRR_PCT - (sector x quarter mean) directly from the raw
          columns and compare to the stored values.
      (2) Cross-check SRR_PCT/100 against the INDEPENDENTLY-SOURCED raw field
          #SHARE_RET_REVENUE (present in the underlying data extract, not
          derived by this pipeline), if available.
    """
    ret = pd.to_numeric(df_filtered['#RETURNING_CUSTOMERS'], errors='coerce')
    tot = pd.to_numeric(df_filtered['#TOTAL_REVENUE'], errors='coerce')
    srr_recomputed = (ret / tot.replace(0, np.nan)) * 100
    stored_srr = pd.to_numeric(df_filtered['SRR_PCT'], errors='coerce')
    diff = (srr_recomputed - stored_srr).abs()

    sector_quarter_mean = df_filtered.groupby(
        [df_filtered.index.get_level_values('DATE'), 'SECTOR'])['SRR_PCT'].transform('mean')
    adj_recomputed = df_filtered['SRR_PCT'] - sector_quarter_mean
    stored_adj = pd.to_numeric(df_filtered['ADJ_SRR_PCT'], errors='coerce')
    diff_adj = (adj_recomputed - stored_adj).abs()

    out = {
        'srr_max_abs_diff': float(diff.max()),
        'srr_n_mismatch': int((diff > SRR_VERIFY_TOL).sum()),
        'srr_n_total': int(stored_srr.notna().sum()),
        'adj_srr_max_abs_diff': float(diff_adj.max()),
        'adj_srr_n_mismatch': int((diff_adj > SRR_VERIFY_TOL).sum()),
        'adj_srr_n_total': int(stored_adj.notna().sum()),
    }

    if '#SHARE_RET_REVENUE' in df_filtered.columns:
        share_raw = pd.to_numeric(df_filtered['#SHARE_RET_REVENUE'], errors='coerce')
        comp = pd.concat([share_raw, stored_srr / 100], axis=1).dropna()
        comp.columns = ['share_raw', 'srr_over_100']
        out['independent_source_corr'] = float(comp['share_raw'].corr(comp['srr_over_100']))
        out['independent_source_max_abs_diff'] = float(
            (comp['share_raw'] - comp['srr_over_100']).abs().max())
        out['independent_source_n'] = int(len(comp))
    else:
        out['independent_source_corr'] = np.nan
        out['independent_source_max_abs_diff'] = np.nan
        out['independent_source_n'] = 0

    return out


def rrr_sorr_correlation(df_filtered):
    """Mechanical-relationship check: RRR_PCT = retained / PRIOR-period total revenue;
    SRR_PCT = retained / CURRENT-period total revenue. Both share the same numerator
    (retained revenue), so higher RRR should be associated with higher SoRR whenever
    revenue is not shrinking sharply. Reports Pearson and Spearman correlations for
    both the raw and the industry x time adjusted signal pairs."""
    d = df_filtered[['RRR_PCT', 'SRR_PCT', 'ADJ_RRR_PCT', 'ADJ_SRR_PCT']].apply(
        pd.to_numeric, errors='coerce')

    raw = d[['RRR_PCT', 'SRR_PCT']].dropna()
    adj = d[['ADJ_RRR_PCT', 'ADJ_SRR_PCT']].dropna()

    return {
        'pearson_raw': float(raw['RRR_PCT'].corr(raw['SRR_PCT'], method='pearson')),
        'spearman_raw': float(raw['RRR_PCT'].corr(raw['SRR_PCT'], method='spearman')),
        'n_raw': int(len(raw)),
        'pearson_adj': float(adj['ADJ_RRR_PCT'].corr(adj['ADJ_SRR_PCT'], method='pearson')),
        'spearman_adj': float(adj['ADJ_RRR_PCT'].corr(adj['ADJ_SRR_PCT'], method='spearman')),
        'n_adj': int(len(adj)),
    }


# =============================================================================
# SECTION D -- ORCHESTRATION
# =============================================================================

def build_task1_metrics_table(panel, ff_factors, market_ret, signal_col, bucket_col_name):
    """Build the full Task 1 performance/risk battery for a quartile sort on
    `signal_col`, assigned into `bucket_col_name`. Returns (table, combined,
    drawdown_df, upside_capture_df, turnover_avg, turnover_n)."""

    panel = panel.copy()
    panel[signal_col] = pd.to_numeric(panel[signal_col], errors='coerce')
    panel[bucket_col_name] = panel.groupby('QUARTER')[signal_col].transform(safe_quartile)

    port = build_portfolio_returns(panel, bucket_col_name, market_ret)
    if port is None:
        raise RuntimeError(f"build_portfolio_returns returned None for {signal_col}")

    port_aligned = port.copy()
    port_aligned.index = pd.to_datetime(port_aligned.index).to_period('M').to_timestamp('M')
    combined = port_aligned.join(ff_factors[['RF']], how='inner').dropna(subset=['RF']).sort_index()
    n_months = len(combined)
    print(f"    Aligned sample: {n_months} months "
          f"({combined.index.min().strftime('%Y-%m')} to {combined.index.max().strftime('%Y-%m')})")

    mkt_ret_aligned = combined[MARKET_COL]

    ann_return, ann_vol, sharpe, sortino = {}, {}, {}, {}
    te, ir, dbeta, ubeta, mdd = {}, {}, {}, {}, {}
    hit_mkt, hit_q4 = {}, {}
    n_down_diag, n_up_diag = {}, {}

    for col in REPORT_COLS:
        raw = combined[col]

        ann_return[col] = raw.mean() * MONTHS_PER_YEAR * 100
        ann_vol[col] = raw.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR) * 100

        ex = excess_return(col, raw, combined['RF'])
        sharpe[col] = sharpe_ratio(ex)
        sortino[col] = sortino_ratio(ex)

        te[col] = tracking_error(raw, mkt_ret_aligned) * 100
        ir[col] = information_ratio(raw, mkt_ret_aligned)

        db, ndn = conditional_beta(raw, mkt_ret_aligned, 'down')
        ub, nup = conditional_beta(raw, mkt_ret_aligned, 'up')
        dbeta[col] = db
        ubeta[col] = ub
        n_down_diag[col] = ndn
        n_up_diag[col] = nup

        mdd[col] = drawdown_series(raw).min() * 100

        if col == MARKET_COL:
            hit_mkt[col] = np.nan
            hit_q4[col] = hit_rate(raw, combined['Q4']) * 100
        elif col == LONG_SHORT_COL:
            hit_mkt[col] = hit_rate(raw, mkt_ret_aligned) * 100
            hit_q4[col] = np.nan   # "long-short beats Q4" is not a coherent hit-rate comparison
        else:
            hit_mkt[col] = hit_rate(raw, mkt_ret_aligned) * 100
            hit_q4[col] = hit_rate(raw, combined['Q4']) * 100 if col != 'Q4' else 0.0

    turnover_avg, turnover_n, turnover_detail, n_skipped_gap = compute_portfolio_turnover(
        panel, bucket_col_name)
    print(f"    Turnover: {turnover_n} quarterly transitions used per bucket "
          f"({n_skipped_gap} calendar-gap transitions skipped)")

    turnover_row = {c: (turnover_avg.get(c, np.nan) * 100 if c in turnover_avg else np.nan)
                    for c in REPORT_COLS}

    n_row = {c: n_months for c in REPORT_COLS}

    table = pd.DataFrame({
        'Ann Return (%)': ann_return,
        'Ann Vol (%)': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Tracking Error (%)': te,
        'Information Ratio': ir,
        'Downside Beta': dbeta,
        'Upside Beta': ubeta,
        'Max Drawdown (%)': mdd,
        'Turnover (avg qtrly, %)': turnover_row,
        'Hit Rate vs Market (%)': hit_mkt,
        'Hit Rate vs Q4 (%)': hit_q4,
        'N (months)': n_row,
    }).T
    table = table.reindex(index=ROW_ORDER, columns=REPORT_COLS)

    print(f"    Downside/upside beta observation counts: "
          f"down={ {c: n_down_diag[c] for c in REPORT_COLS} }, "
          f"up={ {c: n_up_diag[c] for c in REPORT_COLS} }")

    drawdown_df = pd.DataFrame({c: drawdown_series(combined[c]) for c in REPORT_COLS})
    upside_capture_df = pd.DataFrame({c: rolling_upside_capture(combined[c], mkt_ret_aligned)
                                       for c in QUARTILE_COLS})

    return table, combined, drawdown_df, upside_capture_df, turnover_avg, turnover_n, port


def sanity_check_against_pipeline(panel, ff_factors, port_adj):
    """Internal-consistency check: analysis_v2.run_portfolio_analysis() is the
    pipeline's OWN entry point for the headline ADJ RRR Q1-Q4 sort (form_lag=2,
    the module default). It is not saved to any tracked output file (its
    run_factor_regressions() call result is discarded, printed only), so there is
    no pre-existing reference number for the headline sort to compare against --
    NOTE: output/factor_reg_Lag2_adj.xlsx is NOT that reference; it is written by
    the separate run_extra_lag_portfolios() robustness test, which deliberately
    forms 5 months (not 2) after quarter-end and is EXPECTED to differ.

    Instead, call run_portfolio_analysis() directly on the same panel and compare
    its returned port_rrr_adj_q4_vw DataFrame to this script's own `port_adj`
    element-by-element. Both ultimately call the same build_portfolio_returns() on
    the same inputs, so they should be bit-identical; any mismatch is a real bug."""
    print("\n  [SANITY CHECK] Comparing to analysis_v2.run_portfolio_analysis() directly "
          "(the pipeline's own headline ADJ RRR Q1-Q4 sort, form_lag=2) ...")
    pipeline_results = run_portfolio_analysis(panel.copy(), ff_factors)
    pipeline_port = pipeline_results.get('port_rrr_adj_q4_vw')
    if pipeline_port is None:
        print("    Pipeline did not return port_rrr_adj_q4_vw; cannot compare.")
        return
    shared_cols = [c for c in port_adj.columns if c in pipeline_port.columns]
    all_match = True
    for col in shared_cols:
        a = port_adj[col].dropna()
        b = pipeline_port[col].reindex(a.index).dropna()
        a = a.reindex(b.index)
        identical = np.allclose(a.values, b.values, atol=1e-12, equal_nan=True) if len(a) == len(b) else False
        all_match = all_match and identical and (len(a) == len(port_adj[col].dropna()))
        print(f"    {col}: N_mine={port_adj[col].notna().sum()}, N_pipeline={pipeline_port[col].notna().sum()}, "
              f"identical_on_overlap={identical}")
    print(f"    -> {'PASS: portfolio construction matches the pipeline exactly' if all_match else 'FAIL: investigate divergence'}")

    # Also report the extra-lag robustness file for context (NOT expected to match).
    ref_path = os.path.join(OUTPUT_DIR, 'factor_reg_Lag2_adj.xlsx')
    if os.path.exists(ref_path):
        ref = pd.read_excel(ref_path, engine='openpyxl').set_index('Unnamed: 0')
        if 'Q1-Q4_FF3' in ref.index:
            print(f"    (context only, NOT a match target: extra-lag [5mo] robustness Q1-Q4 FF3 "
                  f"alpha={ref.loc['Q1-Q4_FF3', 'alpha']*100:.4f}%/mo, t={ref.loc['Q1-Q4_FF3', 'alpha_t']:.4f})")


def run_task2_sorr(df_filtered, panel, ff_factors, market_ret):
    """Task 2: SoRR (ADJ_SRR_PCT) subsection."""
    print("\n" + "=" * 80)
    print("TASK 2: SHARE OF RETAINED REVENUE (SoRR) SUBSECTION")
    print("=" * 80)

    print("\n  --- 2.1 Verify SRR_PCT / ADJ_SRR_PCT after the timing rebuild ---")
    verify = verify_srr_computation(df_filtered)
    print(f"    SRR_PCT recompute:      max|diff|={verify['srr_max_abs_diff']:.2e}, "
          f"mismatches={verify['srr_n_mismatch']}/{verify['srr_n_total']}")
    print(f"    ADJ_SRR_PCT recompute:  max|diff|={verify['adj_srr_max_abs_diff']:.2e}, "
          f"mismatches={verify['adj_srr_n_mismatch']}/{verify['adj_srr_n_total']}")
    print(f"    Independent source (#SHARE_RET_REVENUE) vs SRR_PCT/100: "
          f"corr={verify['independent_source_corr']:.10f}, "
          f"max|diff|={verify['independent_source_max_abs_diff']:.2e}, "
          f"N={verify['independent_source_n']}")

    print("\n  --- 2.2 Portfolio sort on ADJ_SRR_PCT, same corrected timing ---")
    panel = panel.copy()
    panel[SRR_SIGNAL_COL] = pd.to_numeric(panel[SRR_SIGNAL_COL], errors='coerce')
    panel['SRR_Q'] = panel.groupby('QUARTER')[SRR_SIGNAL_COL].transform(safe_quartile)
    # Also (re)build the RRR quartile bucket on this same panel copy, needed for the
    # 2.3b $-turnover comparison below. build_task1_metrics_table() computes this on
    # its OWN internal copy of panel and does not mutate/return the caller's panel,
    # so it is not already present here -- recomputed explicitly rather than relying
    # on a cross-function side effect.
    panel[RRR_SIGNAL_COL] = pd.to_numeric(panel[RRR_SIGNAL_COL], errors='coerce')
    panel['RRR_Q_ADJ'] = panel.groupby('QUARTER')[RRR_SIGNAL_COL].transform(safe_quartile)
    port_srr = build_portfolio_returns(panel, 'SRR_Q', market_ret)
    if port_srr is None:
        raise RuntimeError("build_portfolio_returns returned None for SoRR sort")

    ff_srr = run_factor_regressions(port_srr, ff_factors, 'SoRR (ADJ_SRR_PCT) Q1-Q4 VW')

    alpha_summary = {}
    for spec in ['FF3', 'FF3+Mom', 'FF5']:
        key = f'{LONG_SHORT_COL}_{spec}'
        if key in ff_srr:
            r = ff_srr[key]
            alpha_summary[spec] = {
                'alpha_pct_month': r['alpha'] * 100,
                'alpha_pct_annual': r['alpha'] * 100 * MONTHS_PER_YEAR,
                'se_pct': r['se_alpha'] * 100,
                't_stat': r['alpha_t'],
                'p_value': r['alpha_p'],
                'r2': r['r2'],
                'n_obs': r['n_obs'],
            }
            print(f"    Q1-Q4 {spec}: alpha={r['alpha']*100:.4f}%/mo "
                  f"({r['alpha']*100*MONTHS_PER_YEAR:.4f}%/yr), SE={r['se_alpha']*100:.4f}, "
                  f"t={r['alpha_t']:.4f}, p=[{r['alpha_p']:.4f}], R2={r['r2']:.4f}, N={r['n_obs']:.0f}")

    print("\n  --- 2.3 Signal-level turnover: SoRR vs RRR ---")
    rrr_turnover = signal_level_turnover(df_filtered, RRR_SIGNAL_COL)
    srr_turnover = signal_level_turnover(df_filtered, SRR_SIGNAL_COL)
    for label, stats in [('ADJ_RRR_PCT', rrr_turnover), ('ADJ_SRR_PCT', srr_turnover)]:
        print(f"    {label}: mean|change|={stats['mean_abs_change_pp']:.4f}pp "
              f"(median={stats['median_abs_change_pp']:.4f}pp), "
              f"cross-sec SD={stats['cross_sectional_sd_pp']:.4f}pp, "
              f"normalized change={stats['mean_abs_change_norm']:.4f}, "
              f"quartile stay-rate={stats['quartile_stay_rate']*100:.2f}%, "
              f"churn-rate={stats['quartile_churn_rate']*100:.2f}% (N={stats['n_quartile_transitions']})")

    print("\n  --- 2.3b (bonus) $-weighted portfolio turnover: SoRR-sorted vs RRR-sorted quartiles ---")
    rrr_port_turnover, rrr_port_n, _, _ = compute_portfolio_turnover(panel, 'RRR_Q_ADJ')
    srr_port_turnover, srr_port_n, _, _ = compute_portfolio_turnover(panel, 'SRR_Q')
    for b in QUARTILE_COLS + [LONG_SHORT_COL]:
        print(f"    {b}: RRR-sorted turnover={rrr_port_turnover[b]*100:.4f}%  "
              f"vs  SoRR-sorted turnover={srr_port_turnover[b]*100:.4f}%")

    print("\n  --- 2.4 Mechanical relationship: RRR vs SoRR correlation ---")
    corr = rrr_sorr_correlation(df_filtered)
    print(f"    Raw:      Pearson={corr['pearson_raw']:.4f}, Spearman={corr['spearman_raw']:.4f} "
          f"(N={corr['n_raw']})")
    print(f"    Adjusted: Pearson={corr['pearson_adj']:.4f}, Spearman={corr['spearman_adj']:.4f} "
          f"(N={corr['n_adj']})")

    return {
        'verify': verify,
        'port_srr': port_srr,
        'ff_srr': ff_srr,
        'alpha_summary': alpha_summary,
        'rrr_turnover': rrr_turnover,
        'srr_turnover': srr_turnover,
        'rrr_port_turnover': rrr_port_turnover,
        'srr_port_turnover': srr_port_turnover,
        'corr': corr,
    }


def save_outputs(table1, drawdown_df, upside_capture_df, task2_results):
    """Persist all deliverables to output/ as explicitly-named files, so they can be
    git-added individually (never via a blanket `git add -A`)."""

    summary_path = os.path.join(OUTPUT_DIR, 'performance_metrics_summary.xlsx')
    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        table1.to_excel(writer, sheet_name='Metrics')
        notes_df = pd.DataFrame({'Metric': list(METRIC_NOTES.keys()),
                                  'Why included': list(METRIC_NOTES.values())})
        notes_df.to_excel(writer, sheet_name='Notes', index=False)
    print(f"\n  Saved: {summary_path}")

    drawdown_path = os.path.join(OUTPUT_DIR, 'performance_drawdown_series.xlsx')
    (drawdown_df * 100).to_excel(drawdown_path, sheet_name='Drawdown_pct', engine='openpyxl')
    print(f"  Saved: {drawdown_path}")

    upside_path = os.path.join(OUTPUT_DIR, 'performance_upside_capture_series.xlsx')
    upside_capture_df.to_excel(upside_path, sheet_name='UpsideCapture_pct', engine='openpyxl')
    print(f"  Saved: {upside_path}")

    sorr_path = os.path.join(OUTPUT_DIR, 'performance_sorr_summary.xlsx')
    with pd.ExcelWriter(sorr_path, engine='openpyxl') as writer:
        pd.DataFrame([task2_results['verify']]).to_excel(writer, sheet_name='SRR_Verification', index=False)

        alpha_rows = []
        for spec, vals in task2_results['alpha_summary'].items():
            row = {'Spec': spec}
            row.update(vals)
            alpha_rows.append(row)
        pd.DataFrame(alpha_rows).to_excel(writer, sheet_name='SoRR_LongShort_Alpha', index=False)

        turnover_rows = []
        for label, stats in [('ADJ_RRR_PCT', task2_results['rrr_turnover']),
                              ('ADJ_SRR_PCT', task2_results['srr_turnover'])]:
            row = {k: v for k, v in stats.items() if k != 'per_quartile_stay_rate'}
            for q, v in stats['per_quartile_stay_rate'].items():
                row[f'stay_rate_{q}'] = v
            turnover_rows.append(row)
        pd.DataFrame(turnover_rows).to_excel(writer, sheet_name='Signal_Turnover_Comparison', index=False)

        if task2_results['rrr_port_turnover'] is not None:
            port_to_rows = []
            for b in QUARTILE_COLS + [LONG_SHORT_COL]:
                port_to_rows.append({
                    'Bucket': b,
                    'RRR_sorted_turnover_avg': task2_results['rrr_port_turnover'][b],
                    'SoRR_sorted_turnover_avg': task2_results['srr_port_turnover'][b],
                })
            pd.DataFrame(port_to_rows).to_excel(writer, sheet_name='Portfolio_Turnover_Comparison', index=False)

        pd.DataFrame([task2_results['corr']]).to_excel(writer, sheet_name='RRR_SoRR_Correlation', index=False)
    print(f"  Saved: {sorr_path}")


def main():
    print("=" * 80)
    print("PERFORMANCE METRICS BATTERY -- RRR Financial Implications")
    print(f"Timing: form {FORM_LAG_MONTHS} months after quarter-end, hold {HOLD_MONTHS} months "
          "(build_holding_panel(), analysis_v2.py, commit 1fce67f)")
    print("Market benchmark for all relative metrics: sample's own value-weighted "
          "market portfolio (NOT the S&P 500)")
    print("Headline RRR signal: ADJ_RRR_PCT (industry x time adjusted)")
    print("=" * 80)

    # ---- Load data via analysis_v2 (single source of truth for loading/timing) ----
    print("\n[1/4] Loading Phase 1 panel, FF factors, monthly returns, holding panel ...")
    df_long, df_filtered, industry_stats = phase1_load_and_diagnose()
    ff_factors = phase1b_load_ff_factors()
    returns = phase1c_load_monthly_returns()
    panel = build_holding_panel(df_filtered, returns)
    market_ret = value_weighted_market(panel)

    # =========================================================================
    # TASK 1
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK 1: PERFORMANCE / RISK METRICS BATTERY (ADJ RRR quartiles)")
    print("=" * 80)
    print("\n[2/4] Building ADJ-RRR quartile portfolios and the full metrics battery ...")
    (table1, combined1, drawdown_df, upside_capture_df,
     turnover_avg, turnover_n, port_adj) = build_task1_metrics_table(
        panel, ff_factors, market_ret, RRR_SIGNAL_COL, 'RRR_Q_ADJ')

    sanity_check_against_pipeline(panel, ff_factors, port_adj)

    print("\n  ================= TASK 1 RESULTS TABLE =================")
    with pd.option_context('display.float_format', lambda x: f'{x:,.4f}',
                            'display.width', 160, 'display.max_columns', 10):
        print(table1.to_string())
    print("\n  ---- one-line note per metric ----")
    for row_name in ROW_ORDER:
        print(f"    {row_name}: {METRIC_NOTES[row_name]}")

    # =========================================================================
    # TASK 2
    # =========================================================================
    print("\n[3/4] Running Task 2 (SoRR subsection) ...")
    task2_results = run_task2_sorr(df_filtered, panel, ff_factors, market_ret)

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n[4/4] Saving outputs ...")
    save_outputs(table1, drawdown_df, upside_capture_df, task2_results)

    print("\n" + "=" * 80)
    print("DONE.")
    print("=" * 80)

    return {
        'table1': table1,
        'combined1': combined1,
        'drawdown_df': drawdown_df,
        'upside_capture_df': upside_capture_df,
        'turnover_avg': turnover_avg,
        'turnover_n': turnover_n,
        'task2_results': task2_results,
    }


if __name__ == '__main__':
    RESULTS = main()
