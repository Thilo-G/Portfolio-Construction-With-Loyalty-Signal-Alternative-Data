"""
diagnose_portfolio_overlap.py — Diagnostic checks for RRR portfolio sorts
==========================================================================
Investigates why raw RRR and industry-adjusted RRR portfolio sorts produce
near-identical results. Also checks sample market vs S&P 500 correlation
and explains N=87 in factor regressions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import statsmodels.api as sm

from analysis_v2 import (
    phase1_load_and_diagnose,
    phase1b_load_ff_factors,
    phase1c_load_monthly_returns,
    merge_signals_to_returns,
    safe_quartile,
    build_portfolio_returns,
)


def main():
    # ========================================================================
    # Load data through the same pipeline as analysis_v2
    # ========================================================================
    print("=" * 80)
    print("  DIAGNOSTIC: Raw vs Adjusted RRR Portfolio Overlap")
    print("=" * 80)

    df_long, df_filtered, industry_stats = phase1_load_and_diagnose()
    ff_factors = phase1b_load_ff_factors()
    returns = phase1c_load_monthly_returns()
    returns_merged = merge_signals_to_returns(df_filtered, returns)

    ret = returns_merged.copy()
    for col in ['RRR_PCT_LAG1', 'ACQ_RATE_PCT_LAG1', 'ADJ_RRR_PCT_LAG1',
                'ADJ_ACQ_RATE_PCT_LAG1', 'HISTORICAL_MARKET_CAP']:
        ret[col] = pd.to_numeric(ret[col], errors='coerce')

    # ========================================================================
    # CHECK 1: Rank Correlation Between Raw and Adjusted RRR
    # ========================================================================
    print("\n" + "=" * 80)
    print("  CHECK 1: Rank Correlation (Raw RRR vs Adjusted RRR)")
    print("=" * 80)

    valid = ret.dropna(subset=['RRR_PCT_LAG1', 'ADJ_RRR_PCT_LAG1'])

    # Pooled
    rho_pooled, p_pooled = spearmanr(valid['RRR_PCT_LAG1'], valid['ADJ_RRR_PCT_LAG1'])
    r_pooled, _ = pearsonr(valid['RRR_PCT_LAG1'], valid['ADJ_RRR_PCT_LAG1'])
    print(f"\n  Pooled Spearman rho: {rho_pooled:.4f}  (p={p_pooled:.2e})")
    print(f"  Pooled Pearson r:    {r_pooled:.4f}")

    # Per quarter
    print(f"\n  Per-quarter Spearman rank correlations:")
    quarter_rhos = []
    for q, grp in valid.groupby('QUARTER'):
        if len(grp) >= 10:
            rho, _ = spearmanr(grp['RRR_PCT_LAG1'], grp['ADJ_RRR_PCT_LAG1'])
            quarter_rhos.append({'Quarter': q, 'Spearman_rho': rho, 'N_firms': len(grp)})
    quarter_rho_df = pd.DataFrame(quarter_rhos)
    print(f"  Mean: {quarter_rho_df['Spearman_rho'].mean():.4f}")
    print(f"  Min:  {quarter_rho_df['Spearman_rho'].min():.4f}")
    print(f"  Max:  {quarter_rho_df['Spearman_rho'].max():.4f}")
    print(f"  SD:   {quarter_rho_df['Spearman_rho'].std():.4f}")

    # ========================================================================
    # CHECK 2: Quartile Assignment Overlap
    # ========================================================================
    print("\n" + "=" * 80)
    print("  CHECK 2: Quartile Assignment Overlap")
    print("=" * 80)

    ret['Q_RAW'] = ret.groupby('QUARTER')['RRR_PCT_LAG1'].transform(safe_quartile)
    ret['Q_ADJ'] = ret.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_quartile)

    both_valid = ret.dropna(subset=['Q_RAW', 'Q_ADJ'])
    same = (both_valid['Q_RAW'] == both_valid['Q_ADJ']).sum()
    total = len(both_valid)
    pct_same = same / total * 100

    print(f"\n  Total firm-months with both assignments: {total}")
    print(f"  Same quartile:    {same} ({pct_same:.1f}%)")
    print(f"  Different quartile: {total - same} ({100 - pct_same:.1f}%)")

    # Transition matrix
    print(f"\n  Transition matrix (Raw Q -> Adjusted Q):")
    trans = pd.crosstab(both_valid['Q_RAW'], both_valid['Q_ADJ'], margins=True)
    print(trans.to_string())

    print(f"\n  Transition matrix (normalized by row):")
    trans_norm = pd.crosstab(both_valid['Q_RAW'], both_valid['Q_ADJ'], normalize='index')
    print(trans_norm.round(3).to_string())

    # ========================================================================
    # CHECK 3: Adjustment Magnitude
    # ========================================================================
    print("\n" + "=" * 80)
    print("  CHECK 3: Adjustment Magnitude")
    print("=" * 80)

    adj_diff = valid['ADJ_RRR_PCT_LAG1'] - valid['RRR_PCT_LAG1']
    print(f"\n  ADJ_RRR - RRR (= -sector_quarter_mean):")
    print(f"  Mean:   {adj_diff.mean():.4f}")
    print(f"  Median: {adj_diff.median():.4f}")
    print(f"  SD:     {adj_diff.std():.4f}")
    print(f"  Min:    {adj_diff.min():.4f}")
    print(f"  Max:    {adj_diff.max():.4f}")
    print(f"  IQR:    {adj_diff.quantile(0.75) - adj_diff.quantile(0.25):.4f}")

    # Show sector-quarter means to see if they vary
    print(f"\n  Sector-quarter mean RRR (what gets subtracted):")
    sq_means = valid.groupby(['QUARTER', 'SECTOR'])['RRR_PCT_LAG1'].mean()
    sq_unstacked = sq_means.unstack('SECTOR')
    print(f"\n  Cross-sector SD of means per quarter (if small, adjustment is ~constant):")
    cross_sector_sd = sq_unstacked.std(axis=1)
    print(f"  Mean cross-sector SD: {cross_sector_sd.mean():.4f}")
    print(f"  Max cross-sector SD:  {cross_sector_sd.max():.4f}")

    # ========================================================================
    # CHECK 4: Sector Composition
    # ========================================================================
    print("\n" + "=" * 80)
    print("  CHECK 4: Sector Composition")
    print("=" * 80)

    firm_sectors = df_filtered.reset_index().groupby('FIRM')['SECTOR'].first()
    sector_counts = firm_sectors.value_counts()
    total_firms = sector_counts.sum()
    print(f"\n  Firm counts by sector:")
    for sector, count in sector_counts.items():
        print(f"    {sector}: {count} ({count/total_firms*100:.1f}%)")

    # Per quarter
    print(f"\n  Firms per sector per quarter (min / max across quarters):")
    qtr_sector = df_filtered.reset_index().groupby(['DATE', 'SECTOR'])['FIRM'].nunique().unstack('SECTOR')
    for sector in qtr_sector.columns:
        print(f"    {sector}: {qtr_sector[sector].min():.0f} - {qtr_sector[sector].max():.0f}")

    # ========================================================================
    # CHECK 5: Value-Weighting of Movers
    # ========================================================================
    print("\n" + "=" * 80)
    print("  CHECK 5: Market Cap of Firms That Change Quartile")
    print("=" * 80)

    both_valid_mcap = both_valid.copy()
    both_valid_mcap['MCAP'] = pd.to_numeric(both_valid_mcap['HISTORICAL_MARKET_CAP'], errors='coerce')
    both_valid_mcap['SAME_Q'] = both_valid_mcap['Q_RAW'] == both_valid_mcap['Q_ADJ']

    mcap_same = both_valid_mcap.loc[both_valid_mcap['SAME_Q'], 'MCAP'].sum()
    mcap_diff = both_valid_mcap.loc[~both_valid_mcap['SAME_Q'], 'MCAP'].sum()
    mcap_total = mcap_same + mcap_diff

    print(f"\n  Total MCAP (same quartile):      {mcap_same/1e9:.1f}B ({mcap_same/mcap_total*100:.1f}%)")
    print(f"  Total MCAP (different quartile): {mcap_diff/1e9:.1f}B ({mcap_diff/mcap_total*100:.1f}%)")

    # Median MCAP of movers vs stayers
    median_same = both_valid_mcap.loc[both_valid_mcap['SAME_Q'], 'MCAP'].median()
    median_diff = both_valid_mcap.loc[~both_valid_mcap['SAME_Q'], 'MCAP'].median()
    print(f"\n  Median MCAP (same quartile):      {median_same/1e6:.0f}M")
    print(f"  Median MCAP (different quartile): {median_diff/1e6:.0f}M")

    # ========================================================================
    # CHECK 6: Paper Table Cross-Check (FF3 regression)
    # ========================================================================
    print("\n" + "=" * 80)
    print("  CHECK 6: Paper Table Cross-Check (re-run FF3 regressions)")
    print("=" * 80)

    # Build market return
    ret_sample = ret[~ret['FIRM'].isin(['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX'])].copy()
    ret_sample['MCAP'] = ret_sample['HISTORICAL_MARKET_CAP']
    ret_sample['MCAP_TOTAL'] = ret_sample.groupby('Date')['MCAP'].transform('sum')
    ret_sample['w_mkt'] = ret_sample['MCAP'] / ret_sample['MCAP_TOTAL']
    ret_sample['w_ret_mkt'] = ret_sample['w_mkt'] * ret_sample['RETURN_LOG']
    market_ret = ret_sample.groupby('Date')['w_ret_mkt'].sum().sort_index()

    for label, col in [('RAW', 'RRR_PCT_LAG1'), ('ADJ', 'ADJ_RRR_PCT_LAG1')]:
        print(f"\n  --- {label} RRR Quartile Sort ---")
        q_col = f'RRR_Q_{label}_check'
        ret[q_col] = ret.groupby('QUARTER')[col].transform(safe_quartile)
        port = build_portfolio_returns(ret, q_col, market_ret)

        if port is not None:
            port_m = port.copy()
            port_m.index = pd.to_datetime(port_m.index).to_period('M').to_timestamp('M')
            combined = port_m.join(ff_factors, how='inner')

            if 'Mkt-RF' in combined.columns:
                for p_col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4']:
                    if p_col not in combined.columns:
                        continue
                    if 'Q1-Q4' in p_col:
                        y = combined[p_col].dropna()
                    else:
                        y = (combined[p_col] - combined['RF']).dropna() if 'RF' in combined.columns else combined[p_col].dropna()
                    X = sm.add_constant(combined[['Mkt-RF', 'SMB', 'HML']].reindex(y.index).dropna())
                    y = y.reindex(X.index)
                    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
                    print(f"    {p_col}: alpha={model.params['const']*100:.4f}%/mo  "
                          f"(SE={model.bse['const']*100:.4f})  t={model.tvalues['const']:.2f}  "
                          f"N={int(model.nobs)}")

    # ========================================================================
    # CHECK 7: Sample Market Return vs S&P 500 Correlation
    # ========================================================================
    print("\n" + "=" * 80)
    print("  CHECK 7: Sample Market Return vs S&P 500")
    print("=" * 80)

    # Get SPX returns from the returns data
    spx_ret = returns[returns['FIRM'] == 'SPX INDEX'][['Date', 'RETURN_LOG']].copy()
    spx_ret = spx_ret.rename(columns={'RETURN_LOG': 'SPX_RET'})
    spx_ret = spx_ret.set_index('Date').sort_index()

    # Sample market return
    mkt_df = market_ret.to_frame('SAMPLE_MKT').sort_index()

    # Merge
    comparison = mkt_df.join(spx_ret, how='inner')
    comparison = comparison.dropna()

    if len(comparison) > 0:
        r_pearson, p_pearson = pearsonr(comparison['SAMPLE_MKT'], comparison['SPX_RET'])
        rho_spearman, p_spearman = spearmanr(comparison['SAMPLE_MKT'], comparison['SPX_RET'])

        print(f"\n  Overlapping months: {len(comparison)}")
        print(f"  Date range: {comparison.index.min()} to {comparison.index.max()}")
        print(f"\n  Pearson r:   {r_pearson:.4f}  (p={p_pearson:.2e})")
        print(f"  Spearman rho: {rho_spearman:.4f}  (p={p_spearman:.2e})")
        print(f"  R-squared:   {r_pearson**2:.4f}")

        # Descriptive comparison
        print(f"\n  Sample MKT — Ann. ret: {comparison['SAMPLE_MKT'].mean()*12*100:.2f}%, "
              f"Ann. vol: {comparison['SAMPLE_MKT'].std()*np.sqrt(12)*100:.2f}%")
        print(f"  S&P 500    — Ann. ret: {comparison['SPX_RET'].mean()*12*100:.2f}%, "
              f"Ann. vol: {comparison['SPX_RET'].std()*np.sqrt(12)*100:.2f}%")

        # Tracking error
        te = (comparison['SAMPLE_MKT'] - comparison['SPX_RET']).std() * np.sqrt(12) * 100
        print(f"  Annualized tracking error: {te:.2f}%")
    else:
        print("  WARNING: No SPX INDEX data found in returns. Check firm names.")
        print(f"  Available firms containing 'SPX': {[f for f in returns['FIRM'].unique() if 'SPX' in f]}")

    # ========================================================================
    # CHECK 8: Why N=87? Date Range Analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("  CHECK 8: Why N=87 in Factor Regressions?")
    print("=" * 80)

    # Portfolio return dates (use the raw RRR quartile sort we already built)
    ret['RRR_Q_RAW_diag'] = ret.groupby('QUARTER')['RRR_PCT_LAG1'].transform(safe_quartile)
    port_raw = build_portfolio_returns(ret, 'RRR_Q_RAW_diag', market_ret)

    if port_raw is not None:
        port_dates = port_raw.index
        print(f"\n  Portfolio returns:")
        print(f"    Date range: {port_dates.min()} to {port_dates.max()}")
        print(f"    N months:   {len(port_dates)}")

        # FF factor dates
        print(f"\n  Fama-French factors:")
        print(f"    Date range: {ff_factors.index.min()} to {ff_factors.index.max()}")
        print(f"    N months:   {len(ff_factors)}")

        # Inner join
        port_m = port_raw.copy()
        port_m.index = pd.to_datetime(port_m.index).to_period('M').to_timestamp('M')
        combined = port_m.join(ff_factors, how='inner')
        print(f"\n  After inner join:")
        print(f"    Date range: {combined.index.min()} to {combined.index.max()}")
        print(f"    N months:   {len(combined)}")

        # Identify dropped months
        port_months = set(port_m.index)
        ff_months = set(ff_factors.index)
        combined_months = set(combined.index)

        in_port_not_ff = sorted(port_months - ff_months)
        in_ff_not_port = sorted(ff_months - port_months)
        dropped = sorted(port_months - combined_months)

        if in_port_not_ff:
            print(f"\n  Months in portfolio but NOT in FF factors ({len(in_port_not_ff)}):")
            for d in in_port_not_ff:
                print(f"    {d.strftime('%Y-%m')}")

        if dropped:
            print(f"\n  Portfolio months dropped by inner join ({len(dropped)}):")
            for d in dropped:
                print(f"    {d.strftime('%Y-%m')}")

        # Also show the quarterly signal date range
        print(f"\n  Quarterly signals (df_filtered):")
        dates = df_filtered.index.get_level_values('DATE')
        print(f"    Date range: {dates.min()} to {dates.max()}")
        print(f"    N quarters: {dates.nunique()}")
        print(f"    Lag-1 means first usable quarter is Q+1 of earliest signal")

    print("\n" + "=" * 80)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
