"""rrr_portfolios.py -- RRR Financial Implications
================================================

Turning quarterly signals into monthly portfolio returns:

  safe_quartile / safe_tercile   rank firms within a quarter (Q1 / T1 = highest)
  build_holding_panel            map each quarter-end signal to the months it is
                                 held (form two months after the quarter, hold three)
                                 and attach the formation-date market cap
  build_portfolio_returns        value-weighted monthly returns per bucket
  value_weighted_market          the benchmark portfolio of all sample firms
  test_signal_persistence        how often firms stay in the same bucket

Moved verbatim from analysis_v2.py on 2026-10-20; other scripts still reach
these functions through analysis_v2 (av2.build_holding_panel, ...).
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd

from config import FORM_LAG_MONTHS, HOLD_MONTHS, MCAP_TAIL_FILL_MAX_QUARTERS


# =============================================================================
# Functions
# =============================================================================
def safe_tercile(x):
    """Assign terciles: T1=highest third, T2=middle, T3=lowest third.
    Reversed labels in qcut so the highest values receive T1."""
    x = x.dropna()
    if x.nunique() < 3 or len(x) < 6:
        return pd.Series([np.nan]*len(x), index=x.index)
    try:
        # qcut assigns left-to-right (lowest→highest), so reverse labels
        return pd.qcut(x, 3, labels=['T3', 'T2', 'T1'])
    except ValueError:
        return pd.Series([np.nan]*len(x), index=x.index)


def safe_quartile(x):
    """Assign quartiles: Q1=highest quarter, Q4=lowest quarter.
    Reversed labels in qcut so the highest values receive Q1."""
    x = x.dropna()
    if x.nunique() < 4 or len(x) < 8:
        return pd.Series([np.nan]*len(x), index=x.index)
    try:
        # qcut assigns left-to-right (lowest→highest), so reverse labels
        return pd.qcut(x, 4, labels=['Q4', 'Q3', 'Q2', 'Q1'])
    except ValueError:
        return pd.Series([np.nan]*len(x), index=x.index)


def test_signal_persistence(df_filtered):
    """Test persistence of RRR and AR signals."""

    persistence_results = {}

    for signal_name, col, lag_col in [
        ('RRR', 'RRR_PCT', 'RRR_PCT_LAG1'),
        ('AR', 'ACQ_RATE_PCT', 'ACQ_RATE_PCT_LAG1'),
        ('Adj RRR', 'ADJ_RRR_PCT', 'ADJ_RRR_PCT_LAG1'),
        ('Adj AR', 'ADJ_ACQ_RATE_PCT', 'ADJ_ACQ_RATE_PCT_LAG1'),
    ]:
        # Firm-level autocorrelation
        autocorrs = []
        for firm in df_filtered.index.get_level_values('FIRM').unique():
            try:
                firm_data = df_filtered.xs(firm, level='FIRM')[[col, lag_col]].dropna()
                if len(firm_data) >= 3:
                    corr = firm_data[col].corr(firm_data[lag_col])
                    autocorrs.append(corr)
            except (KeyError, ValueError):
                continue

        if autocorrs:
            mean_ac = np.mean(autocorrs)
            median_ac = np.median(autocorrs)
            print(f"  {signal_name} AR(1): mean={mean_ac:.3f}, median={median_ac:.3f}, "
                  f"N_firms={len(autocorrs)}, pct_positive={np.mean(np.array(autocorrs)>0)*100:.1f}%")
            persistence_results[signal_name] = {
                'mean_autocorr': mean_ac,
                'median_autocorr': median_ac,
                'n_firms': len(autocorrs),
                'pct_positive': np.mean(np.array(autocorrs) > 0) * 100
            }

    # Transition matrix for raw RRR terciles
    print("\n  Transition matrix (RRR terciles, quarter-to-quarter):")
    df_temp = df_filtered.copy()
    df_temp['RRR_TERCILE'] = df_temp.groupby(level='DATE')['RRR_PCT'].transform(safe_tercile)
    df_temp['RRR_TERCILE_LAG'] = df_temp.groupby(level='FIRM')['RRR_TERCILE'].shift(1)

    trans = pd.crosstab(df_temp['RRR_TERCILE_LAG'], df_temp['RRR_TERCILE'], normalize='index')
    if not trans.empty:
        print(trans.round(3).to_string())
        persistence_results['transition_matrix_rrr'] = trans

    # Transition matrix for raw AR terciles
    print("\n  Transition matrix (AR terciles, quarter-to-quarter):")
    df_temp['AR_TERCILE'] = df_temp.groupby(level='DATE')['ACQ_RATE_PCT'].transform(safe_tercile)
    df_temp['AR_TERCILE_LAG'] = df_temp.groupby(level='FIRM')['AR_TERCILE'].shift(1)

    trans_ar = pd.crosstab(df_temp['AR_TERCILE_LAG'], df_temp['AR_TERCILE'], normalize='index')
    if not trans_ar.empty:
        print(trans_ar.round(3).to_string())
        persistence_results['transition_matrix_ar'] = trans_ar

    return persistence_results


def build_holding_panel(df_filtered, returns,
                        form_lag=FORM_LAG_MONTHS, hold=HOLD_MONTHS):
    """Map each quarter-end signal to the monthly returns it is meant to predict.

    Timing convention (see FORM_LAG_MONTHS / HOLD_MONTHS at the top of the file):
      * the signal is measured at quarter-end Q and used CONTEMPORANEOUSLY
        (no additional .shift()); the lag is provided ENTIRELY by forming the
        portfolio ``form_lag`` calendar months after Q;
      * formation month period = Q_month + form_lag;
      * holding month periods    = Q_month + form_lag + 1 ... Q_month + form_lag + hold.
    Example (form_lag=2, hold=3): Q = 3/31 -> form end of May -> hold Jun, Jul, Aug.

    Market cap is measured as of the FORMATION month: the quarter-end market cap
    is scaled by the firm's own monthly price change from the quarter-end month to
    the formation month (using the monthly price series, so splits are handled
    consistently). This replaces the stale quarter-end market cap for weighting.

    Returns one row per (firm, holding-month):
        FIRM, Date (holding-month date, as in the monthly-return file),
        QUARTER (= Q, the signal's calendar quarter-end), FORMATION_DATE,
        HOLD_IDX (1..hold), RET_SIMPLE, RETURN_LOG, MCAP_FORM (formation market cap),
        and the contemporaneous quarter-end signals/controls.
    """
    print(f"\n  Building holding panel (form {form_lag}m after quarter-end, hold {hold}m)...")

    # ---- 1. Quarter-end signals (CONTEMPORANEOUS, measured at Q) ----
    base_cols = ['RRR_PCT', 'ADJ_RRR_PCT', 'ACQ_RATE_PCT', 'ADJ_ACQ_RATE_PCT',
                 'REV_GROWTH_PCT', 'SIZE', 'BTM', 'PM_OPER_PCT', 'SECTOR',
                 'HISTORICAL_MARKET_CAP', 'PX_LAST', 'BBG_OFFSET_K']
    alt_cols = ['ADJ_SRR_PCT', 'SRR_PCT', 'ADJ_LR_RRR_PCT', 'LR_RRR_PCT',
                'ADJ_CUMRR_PCT', 'CUMRR_PCT', 'ADJ_RRR_PCT_MA4',
                'ADJ_TQ_RRR_PCT', 'TQ_RRR_PCT',
                'ADJ_TA4_RRR_PCT', 'TA4_RRR_PCT',
                'RCB4_RRR_PCT', 'MA4L_RRR_PCT']
    sig_cols = [c for c in base_cols + alt_cols if c in df_filtered.columns]

    q = df_filtered.reset_index()[['FIRM', 'DATE'] + sig_cols].copy()
    q['FIRM'] = q['FIRM'].str.upper()
    q['DATE'] = pd.to_datetime(q['DATE'])
    # Verify the quarter-end labels are uniform calendar quarters (3/6/9/12).
    bad_months = sorted(set(q['DATE'].dt.month.unique()) - {3, 6, 9, 12})
    assert not bad_months, f"Non-calendar-quarter signal dates found (months {bad_months})."
    q['QUARTER'] = q['DATE'].dt.to_period('Q').dt.to_timestamp('Q', 'end')  # calendar quarter-end
    q['Q_MONTH'] = q['DATE'].dt.to_period('M')                              # month of the quarter-end
    q['FORM_MONTH'] = q['Q_MONTH'] + form_lag                               # formation month period
    q['FORMATION_DATE'] = q['FORM_MONTH'].dt.to_timestamp('M')

    # ---- 2. Monthly price panel keyed by (FIRM, month-period) ----
    ret = returns.copy()
    ret['FIRM'] = ret['FIRM'].str.upper()
    ret['MONTH'] = ret['Date'].dt.to_period('M')
    px_df = ret[['FIRM', 'MONTH', 'PX']].dropna().drop_duplicates(['FIRM', 'MONTH'])

    # ---- 3. Formation-date market cap = quarter-end MCAP * (PX_form / PX_Q) ----
    q = q.merge(px_df.rename(columns={'MONTH': 'Q_MONTH', 'PX': 'PX_Q_MON'}),
                on=['FIRM', 'Q_MONTH'], how='left')
    q = q.merge(px_df.rename(columns={'MONTH': 'FORM_MONTH', 'PX': 'PX_FORM_MON'}),
                on=['FIRM', 'FORM_MONTH'], how='left')
    q['MCAP_Q'] = pd.to_numeric(q['HISTORICAL_MARKET_CAP'], errors='coerce')
    q['MCAP_FORM'] = q['MCAP_Q'] * (q['PX_FORM_MON'] / q['PX_Q_MON'])
    # If the formation-month price is missing, fall back to the quarter-end MCAP.
    q['MCAP_FORM'] = q['MCAP_FORM'].where(q['MCAP_FORM'].notna(), q['MCAP_Q'])

    # ---- 3b. Final quarters of re-dated firms (added 2026-10-20) ----
    # Re-dating a firm stamped k quarters late (bloomberg_dating) leaves its
    # last k quarters without Bloomberg fields. Carry its implied share count
    # (market cap / quarter-end price) forward from its last available quarter,
    # for at most k quarters (capped at MCAP_TAIL_FILL_MAX_QUARTERS), and value
    # it at the formation-month price. Firms that were not re-dated are untouched.
    if 'BBG_OFFSET_K' in q.columns:
        q = q.sort_values(['FIRM', 'QUARTER']).reset_index(drop=True)
        offset_k = pd.to_numeric(q['BBG_OFFSET_K'], errors='coerce').fillna(0).astype(int)
        quarter_no = q['QUARTER'].dt.to_period('Q').apply(lambda p: p.ordinal)
        has_mcap = q['MCAP_Q'].notna() & q['PX_Q_MON'].notna()
        last_mcap_quarter = quarter_no.where(has_mcap).groupby(q['FIRM']).transform('max')
        quarters_since = quarter_no - last_mcap_quarter
        shares_last = (q['MCAP_Q'] / q['PX_Q_MON']).where(has_mcap).groupby(q['FIRM']).ffill()
        fill_max = offset_k.clip(upper=MCAP_TAIL_FILL_MAX_QUARTERS)
        to_fill = ((offset_k > 0) & q['MCAP_FORM'].isna() & q['PX_FORM_MON'].notna()
                   & (quarters_since >= 1) & (quarters_since <= fill_max))
        q.loc[to_fill, 'MCAP_FORM'] = shares_last[to_fill] * q.loc[to_fill, 'PX_FORM_MON']
        print(f"  Market cap carried forward for {int(to_fill.sum())} firm-quarters of "
              f"{q.loc[to_fill, 'FIRM'].nunique()} re-dated firms (final quarters left empty by re-dating)")

    # ---- 4. Expand each firm-quarter to its `hold` holding months ----
    frames = []
    for h in range(1, hold + 1):
        qh = q.copy()
        qh['MONTH'] = qh['Q_MONTH'] + form_lag + h  # holding-month period
        qh['HOLD_IDX'] = h
        frames.append(qh)
    holding = pd.concat(frames, ignore_index=True)

    # ---- 5. Attach the monthly return earned in each holding month ----
    ret_small = ret[['FIRM', 'MONTH', 'Date', 'RET_SIMPLE', 'RETURN_LOG']].copy()
    panel = holding.merge(ret_small, on=['FIRM', 'MONTH'], how='inner')

    # Restrict to the sector-filtered sample (indices are already excluded there).
    valid_firms = set(s.upper() for s in df_filtered.index.get_level_values('FIRM'))
    panel = panel[panel['FIRM'].isin(valid_firms)].copy()

    # Drop the raw quarter-end 'DATE' (redundant with the normalized 'QUARTER')
    # to avoid confusion with the monthly-return 'Date' column.
    panel = panel.drop(columns=['DATE', 'Q_MONTH', 'FORM_MONTH', 'MONTH',
                                'PX_Q_MON', 'PX_FORM_MON'])
    print(f"  Holding panel: {len(panel)} firm-months, {panel['FIRM'].nunique()} firms, "
          f"{panel['QUARTER'].nunique()} signal quarters")
    return panel


def merge_signals_to_returns(df_filtered, returns, **kwargs):
    """Backward-compatible alias for build_holding_panel (import-compat)."""
    return build_holding_panel(df_filtered, returns, **kwargs)


def value_weighted_market(panel):
    """Value-weighted SIMPLE market return of all sample firms, using
    formation-date market cap as the weight (same convention as the sorts)."""
    df = panel.dropna(subset=['MCAP_FORM', 'RET_SIMPLE']).copy()
    df['MCAP_FORM'] = pd.to_numeric(df['MCAP_FORM'], errors='coerce')
    df = df.dropna(subset=['MCAP_FORM'])
    df['w'] = df['MCAP_FORM'] / df.groupby('Date')['MCAP_FORM'].transform('sum')
    df['wr'] = df['w'] * df['RET_SIMPLE']
    return df.groupby('Date')['wr'].sum().sort_index()


def echo_formation_examples(panel, n_generic=2):
    """Log the exact formation date and holding window for a few firm-quarters,
    so the timing convention is spot-checkable later without re-deriving it."""
    print("\n  [ECHO] example formation dates / 3-month holding windows "
          "(signal at quarter-end -> months actually held):")
    ex = (panel.dropna(subset=['ADJ_RRR_PCT'])
                .sort_values(['FIRM', 'QUARTER', 'HOLD_IDX']))
    # Prefer the two firm-quarters used in the written hand-trace, then fill generically.
    preferred = [('FIVE US EQUITY', '2019Q3'), ('FIVE US EQUITY', '2020Q1')]
    seen = []
    for firm, qlabel in preferred:
        g = ex[(ex['FIRM'] == firm) &
               (ex['QUARTER'].dt.to_period('Q').astype(str) == qlabel)]
        if len(g) == HOLD_MONTHS:
            seen.append((firm, g['QUARTER'].iloc[0], g))
    for (firm, qtr), g in ex.groupby(['FIRM', 'QUARTER']):
        if len(seen) >= len(preferred) + n_generic:
            break
        if len(g) == HOLD_MONTHS and not any(s[0] == firm and s[1] == qtr for s in seen):
            seen.append((firm, qtr, g))
    for firm, qtr, g in seen:
        months = ', '.join(pd.to_datetime(g['Date']).dt.strftime('%Y-%m'))
        print(f"    {firm:<16} signal Q-end {pd.Timestamp(qtr).strftime('%Y-%m-%d')} "
              f"(ADJ_RRR={g['ADJ_RRR_PCT'].iloc[0]:7.2f}) -> "
              f"form {pd.Timestamp(g['FORMATION_DATE'].iloc[0]).strftime('%Y-%m-%d')} -> "
              f"hold [{months}]  MCAP_form={g['MCAP_FORM'].iloc[0]:,.0f}M")


def build_portfolio_returns(ret, bucket_col, market_ret):
    """Value-weighted SIMPLE-return portfolios per bucket + long-short + market.

    Returns are value-weighted from SIMPLE (arithmetic) monthly returns and the
    weights use the FORMATION-date market cap (MCAP_FORM). A value-weighted mean
    of LOG returns is biased (it is not the log of the value-weighted gross
    return), so it is deliberately not used anywhere here.
    """

    df = ret.dropna(subset=[bucket_col]).copy()
    df['MCAP_FORM'] = pd.to_numeric(df['MCAP_FORM'], errors='coerce')
    df['RET_SIMPLE'] = pd.to_numeric(df['RET_SIMPLE'], errors='coerce')
    df = df.dropna(subset=['MCAP_FORM', 'RET_SIMPLE'])

    if df.empty:
        print("    No valid data for portfolio construction")
        return None

    # Value weights within each (month, bucket) from formation-date market cap,
    # then value-weight the SIMPLE returns and sum.
    df['w'] = df['MCAP_FORM'] / df.groupby(['Date', bucket_col])['MCAP_FORM'].transform('sum')
    df['w_ret'] = df['w'] * df['RET_SIMPLE']

    port = (
        df.groupby(['Date', bucket_col])['w_ret']
        .sum()
        .unstack(bucket_col)
        .sort_index()
    )

    # Long-short: HIGH minus LOW (Q1-Q4 for quartiles, T1-T3 for terciles)
    if 'Q1' in port.columns and 'Q4' in port.columns:
        port['Q1-Q4'] = port['Q1'] - port['Q4']
    elif 'T1' in port.columns and 'T3' in port.columns:
        port['T1-T3'] = port['T1'] - port['T3']

    # Add market portfolio (already a VW simple return)
    port['MKT'] = market_ret.reindex(port.index)

    # Print summary returns (simple returns; annualized as mean x 12)
    print("    Portfolio monthly SIMPLE returns (annualized, mean x 12):")
    for col in port.columns:
        ann_ret = port[col].mean() * 12
        ann_vol = port[col].std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        print(f"      {col}: ret={ann_ret*100:.2f}%, vol={ann_vol*100:.2f}%, SR={sharpe:.2f}")

    # Firms per bucket per quarter
    counts = df.groupby(['QUARTER', bucket_col])['FIRM'].nunique().unstack(bucket_col)
    print(f"    Avg firms per bucket: {counts.mean().round(1).to_dict()}")

    return port
