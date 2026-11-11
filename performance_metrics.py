"""
performance_metrics.py -- RRR Financial Implications
=======================================================
Task 1: Full performance/risk metrics battery (Sharpe, Sortino, tracking error,
        information ratio, strategy turnover, hit rate, downside/upside beta,
        rolling drawdown, rolling upside-capture, max drawdown) for the RRR
        quartile portfolios (Q1-Q4) and the long-short, under the corrected
        k=2-month-formation-gap / 3-month-hold timing.
        (A second task on the share of returning revenue was removed on
        2026-11-11; the paper reports that signal only as a robustness row.)

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
  * The headline RRR signal is RRR_PCT (raw, author decision 2026-09-18), the
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

CODE_DIR = os.path.dirname(os.path.abspath(__file__))   # Code/RRR-FI-IM
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

QUARTILE_COLS = ['Q1', 'Q2', 'Q3', 'Q4']
LONG_SHORT_COL = 'Q1-Q4'
MARKET_COL = 'MKT'
REPORT_COLS = QUARTILE_COLS + [LONG_SHORT_COL, MARKET_COL]

# Quarterly re-formation: four portfolio rebuilds per calendar year. Used to put
# turnover on an annual basis so it can be compared with the annual-signal test.
REFORMATIONS_PER_YEAR = 4

RRR_SIGNAL_COL = 'RRR_PCT'          # headline signal: raw is the paper's primary specification (author decision 2026-09-18)

ROW_ORDER = [
    'Ann Return (%)',
    'Ann Vol (%)',
    'Sharpe Ratio',
    'Sortino Ratio',
    'Tracking Error (%)',
    'Information Ratio',
    'Appraisal Ratio (FF3)',
    'Appraisal Ratio (FF5)',
    'Downside Beta',
    'Upside Beta',
    'Max Drawdown (%)',
    'Turnover (avg qtrly, %)',
    'Turnover (annualized, %)',
    'Hit Rate vs Market (%)',
    'Hit Rate vs Q4 (%)',
    'N (months)',
]

METRIC_NOTES = {
    'Ann Return (%)': "Raw annualized value-weighted return actually earned holding the portfolio; context for the risk-adjusted ratios below.",
    'Ann Vol (%)': "Annualized standard deviation of monthly returns; total risk, the denominator of the Sharpe ratio.",
    'Sharpe Ratio': "Return per unit of total risk, in excess of the risk-free rate; the standard risk-adjusted-return benchmark.",
    'Sortino Ratio': "Like Sharpe, but the denominator counts only downside volatility, so large positive swings are not treated as 'risk'; isolates bad-state variance.",
    'Tracking Error (%)': "Volatility of active return. For the long-only quartiles, active return is measured against the sample's own value-weighted market portfolio. For the zero-cost Q1-Q4 spread there is no long-only benchmark to deviate from, so tracking error is the spread's own annualized volatility -- the standard market-neutral convention.",
    'Information Ratio': "Active return divided by tracking error. Long-only quartiles are benchmarked on the sample market. The zero-cost Q1-Q4 spread is self-benchmarked (annualized mean divided by annualized volatility), because comparing a dollar-neutral book against a long-only market index mixes two different exposures and understates the strategy.",
    'Appraisal Ratio (FF3)': "Annualized factor-model alpha divided by annualized residual volatility from the FF3 regression; risk-adjusted skill per unit of idiosyncratic risk, the benchmark-free counterpart to the information ratio.",
    'Appraisal Ratio (FF5)': "Same as above, estimated on the Fama-French five-factor model.",
    'Downside Beta': "Beta on the sample market estimated only in months the sample market fell; how much of the portfolio's decline is systematic (market-driven) risk.",
    'Upside Beta': "Beta on the sample market estimated only in months the sample market rose; a portfolio with upside beta > downside beta has asymmetric, favorable market exposure.",
    'Max Drawdown (%)': "Largest peak-to-trough decline in cumulative value; a path-dependent tail-risk measure that volatility alone does not capture.",
    'Turnover (avg qtrly, %)': "Fraction of portfolio value traded at each quarterly re-formation (0.5 x sum of |weight changes|, new formation weights vs. prior weights drifted by realized returns); the implicit trading/implementability cost of the strategy.",
    'Turnover (annualized, %)': "One-way turnover per calendar year: the per-reformation figure above times the number of reformations per year (four for the quarterly strategy, one for the annual-signal strategy). Stated so that strategies rebalanced at different frequencies can be compared on the same basis.",
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


def self_information_ratio(port_ret):
    """Information ratio for a zero-cost (dollar-neutral) portfolio.

    A long-short spread has no long-only benchmark it is trying to track, so the
    active return IS the portfolio return and the tracking error IS its own
    volatility. Benchmarking the spread against a long-only market portfolio, as
    the generic information_ratio() below does, mixes a market-neutral book with
    a directional index and materially understates it.
    """
    r = port_ret.dropna()
    vol = r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)
    mu = r.mean() * MONTHS_PER_YEAR
    return mu / vol if vol > 0 else np.nan


def appraisal_ratio(port_excess_ret, factor_df, factor_cols):
    """Annualized alpha divided by annualized residual volatility (Treynor-Black).

    Benchmark-free measure of skill per unit of idiosyncratic risk: it asks how
    much alpha the strategy delivers relative to the diversifiable risk taken to
    get it, which is exactly what a portfolio manager sizing the position needs.
    """
    aligned = pd.concat([port_excess_ret, factor_df[factor_cols]], axis=1).dropna()
    if len(aligned) < len(factor_cols) + 5:
        return np.nan
    y = aligned.iloc[:, 0]
    X = sm.add_constant(aligned[factor_cols])
    fit = sm.OLS(y, X).fit()
    alpha_ann = fit.params['const'] * MONTHS_PER_YEAR
    resid_vol_ann = fit.resid.std(ddof=len(fit.params)) * np.sqrt(MONTHS_PER_YEAR)
    return alpha_ann / resid_vol_ann if resid_vol_ann > 0 else np.nan


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

def compute_portfolio_turnover(panel, bucket_col, quarters_between=1,
                               reformations_per_year=None):
    """Strategy turnover: the fraction of portfolio VALUE that must be traded at
    each quarterly rebalance (does not exist elsewhere in the project; built here).

    Standard definition: Turnover_t = 0.5 * sum_i |w_new_i,t - w_end_i,t-1|, where
      * w_new_i,t   = firm i's freshly-formed formation-market-cap weight in the
                       INCOMING quarter's bucket (sums to 1 across that bucket).
      * w_end_i,t-1 = firm i's weight in the OUTGOING quarter's bucket, DRIFTED
                       forward by that firm's realized compounded holding-period
                       return (no interim trading), then renormalized to sum to 1.
    Averaged over all rebalances that are exactly ``quarters_between`` calendar
    quarters apart (transitions across a data gap are excluded and counted
    separately). ``quarters_between`` is 1 for the quarterly baseline and 4 for
    the annual-signal strategy, whose formations sit a year apart; leaving it at
    1 there would silently discard every transition and return NaN turnover.

    ``reformations_per_year`` (4 quarterly, 1 annual) is carried through so the
    caller can also state turnover on an annual basis, which is the only way to
    compare strategies rebalanced at different frequencies.
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
        if (p_curr.ordinal - p_prev.ordinal) != quarters_between:
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

    if reformations_per_year is None:
        reformations_per_year = REFORMATIONS_PER_YEAR
    turnover_annual = {b: (v * reformations_per_year if v == v else np.nan)
                       for b, v in turnover_avg.items()}

    return turnover_avg, turnover_n, turnover_detail, n_skipped_gap, turnover_annual


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
    ar3, ar5 = {}, {}

    # Factor panel for the appraisal ratios, aligned to the same months.
    FF3_COLS = ['Mkt-RF', 'SMB', 'HML']
    FF5_COLS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    combined_ff = ff_factors.reindex(combined.index)
    FF3_OK = all(c in combined_ff.columns for c in FF3_COLS)
    FF5_OK = all(c in combined_ff.columns for c in FF5_COLS)
    if not FF5_OK:
        print("    NOTE: FF5 columns unavailable; Appraisal Ratio (FF5) will be blank.")
    hit_mkt, hit_q4 = {}, {}
    n_down_diag, n_up_diag = {}, {}

    for col in REPORT_COLS:
        raw = combined[col]

        ann_return[col] = raw.mean() * MONTHS_PER_YEAR * 100
        ann_vol[col] = raw.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR) * 100

        ex = excess_return(col, raw, combined['RF'])
        sharpe[col] = sharpe_ratio(ex)
        sortino[col] = sortino_ratio(ex)

        # Zero-cost spreads are self-benchmarked; long-only legs keep the
        # market-relative convention. See self_information_ratio().
        if col == LONG_SHORT_COL:
            te[col] = raw.dropna().std(ddof=1) * np.sqrt(MONTHS_PER_YEAR) * 100
            ir[col] = self_information_ratio(raw)
        else:
            te[col] = tracking_error(raw, mkt_ret_aligned) * 100
            ir[col] = information_ratio(raw, mkt_ret_aligned)

        ar3[col] = appraisal_ratio(ex, combined_ff, FF3_COLS) if FF3_OK else np.nan
        ar5[col] = appraisal_ratio(ex, combined_ff, FF5_COLS) if FF5_OK else np.nan

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

    (turnover_avg, turnover_n, turnover_detail, n_skipped_gap,
     turnover_annual) = compute_portfolio_turnover(
        panel, bucket_col_name,
        quarters_between=1, reformations_per_year=REFORMATIONS_PER_YEAR)
    print(f"    Turnover: {turnover_n} quarterly transitions used per bucket "
          f"({n_skipped_gap} calendar-gap transitions skipped)")

    turnover_row = {c: (turnover_avg.get(c, np.nan) * 100 if c in turnover_avg else np.nan)
                    for c in REPORT_COLS}
    # Annualised one-way turnover. The quarterly strategy re-forms four times a
    # year; stating it per year makes it directly comparable to the annual-signal
    # strategy, which re-forms once.
    turnover_row_ann = {c: (turnover_annual.get(c, np.nan) * 100
                            if c in turnover_annual and turnover_annual[c] == turnover_annual[c]
                            else np.nan)
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
        'Appraisal Ratio (FF3)': ar3,
        'Appraisal Ratio (FF5)': ar5,
        'Max Drawdown (%)': mdd,
        'Turnover (avg qtrly, %)': turnover_row,
        'Turnover (annualized, %)': turnover_row_ann,
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
    no pre-existing reference number for the headline sort to compare against.

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


def save_outputs(table1, drawdown_df, upside_capture_df):
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



def main():
    print("=" * 80)
    print("PERFORMANCE METRICS BATTERY -- RRR Financial Implications")
    print(f"Timing: form {FORM_LAG_MONTHS} months after quarter-end, hold {HOLD_MONTHS} months "
          "(build_holding_panel(), analysis_v2.py, commit 1fce67f)")
    print("Market benchmark for all relative metrics: sample's own value-weighted "
          "market portfolio (NOT the S&P 500)")
    print("Headline RRR signal: RRR_PCT (raw)")
    print("=" * 80)

    # ---- Load data via analysis_v2 (single source of truth for loading/timing) ----
    print("\n[1/3] Loading Phase 1 panel, FF factors, monthly returns, holding panel ...")
    # Factors first: phase 1 needs the FF risk-free rate.
    ff_factors = phase1b_load_ff_factors()
    df_long, df_filtered, industry_stats = phase1_load_and_diagnose(ff_factors)
    returns = phase1c_load_monthly_returns()
    panel = build_holding_panel(df_filtered, returns)
    market_ret = value_weighted_market(panel)

    # =========================================================================
    # TASK 1
    # =========================================================================
    print("\n" + "=" * 80)
    print("TASK 1: PERFORMANCE / RISK METRICS BATTERY (raw RRR quartiles)")
    print("=" * 80)
    print("\n[2/3] Building RRR quartile portfolios and the full metrics battery ...")
    (table1, combined1, drawdown_df, upside_capture_df,
     turnover_avg, turnover_n, port_adj) = build_task1_metrics_table(
        panel, ff_factors, market_ret, RRR_SIGNAL_COL, 'RRR_Q_MAIN')

    sanity_check_against_pipeline(panel, ff_factors, port_adj)

    print("\n  ================= TASK 1 RESULTS TABLE =================")
    with pd.option_context('display.float_format', lambda x: f'{x:,.4f}',
                            'display.width', 160, 'display.max_columns', 10):
        print(table1.to_string())
    print("\n  ---- one-line note per metric ----")
    for row_name in ROW_ORDER:
        print(f"    {row_name}: {METRIC_NOTES[row_name]}")

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print("\n[3/3] Saving outputs ...")
    save_outputs(table1, drawdown_df, upside_capture_df)

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
    }


if __name__ == '__main__':
    RESULTS = main()
