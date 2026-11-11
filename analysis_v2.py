"""
analysis_v2.py -- RRR Financial Implications: main analysis
===========================================================

Runs the core analysis of the paper and writes its main outputs:
  Phase 1  load and clean the data, re-date Bloomberg rows, build RRR (rrr_data.py)
  Phase 2  quartile sorts, factor regressions, CAPM check, Fama-MacBeth, risk
           statistics (rrr_portfolios.py, rrr_regressions.py)

Where the code lives (split on 2026-10-20 for readability; the functions are
unchanged):
  config.py           paths, data files, timing, sample constants, guards
  rrr_data.py         loading and the firm-quarter panel
  rrr_portfolios.py   quartile assignment, holding panel, portfolio returns
  rrr_regressions.py  factor regressions, Fama-MacBeth, risk statistics
  analysis_v2.py      this runner; every function and constant above is
                      re-exported here, so other scripts use av2.<name>

Run the whole paper pipeline with run_pipeline.py; this file is its first step.
Retired analyses (same-growth placebo, extra-lag portfolios, alternative-metric
figures) are in archive/old_versions/analysis_v2.py.before-split-20261020.
"""

# =============================================================================
# Imports (re-exported: other scripts use av2.<name>)
# =============================================================================
import datetime
import json
import os
import platform

import pandas as pd

from config import (
    CODE_DIR,
    PROJECT_DIR,
    DATA_DIR,
    SUPPORT_DIR,
    OUTPUT_DIR,
    DATA_WORKBOOK,
    SHEET_REVENUE,
    SHEET_FUNDAMENTALS,
    SHEET_RETURNS,
    SHEET_INDUSTRY,
    FF_LOCAL_CSV,
    MOM_LOCAL_CSV,
    FF3_URL,
    FF5_URL,
    MOM_URL,
    ANNUAL_REVENUE_WORKBOOK,
    SHEET_ANNUAL_REVENUE,
    REVENUE_BACKFILL_FIRMS,
    MCAP_BACKFILL_FIRMS,
    PRICE_BACKFILL_FIRMS,
    LEGACY_REVENUE_FILE,
    LEGACY_FUNDAMENTALS_FILE,
    LEGACY_RETURNS_FILE,
    LEGACY_INDUSTRY_FILE,
    VALID_SECTORS,
    EXCLUDED_FIRMS,
    ALIGN_BLOOMBERG_DATES,
    FUND_DATE_COL,
    MCAP_TAIL_FILL_MAX_QUARTERS,
    FORM_LAG_MONTHS,
    HOLD_MONTHS,
    EXPECTED_N_FIRMS_PANEL,
    EXPECTED_N_FIRMS,
    EXPECTED_HEADLINE_FF5_ALPHA_PCT,
)
from rrr_data import (
    backfill_revenue,
    backfill_market_cap,
    backfill_monthly_prices,
    download_ff_factors,
    parse_ff_local_csv,
    phase1b_load_ff_factors,
    load_fundamentals_wide,
    phase1_load_and_diagnose,
    phase1c_load_monthly_returns,
)
from rrr_portfolios import (
    safe_tercile,
    safe_quartile,
    test_signal_persistence,
    build_holding_panel,
    merge_signals_to_returns,
    value_weighted_market,
    echo_formation_examples,
    build_portfolio_returns,
)
from rrr_regressions import (
    newey_west_ols,
    run_factor_regressions,
    run_portfolio_analysis,
    run_capm_vs_sp500,
    run_fama_macbeth,
    run_risk_analysis,
)


# =============================================================================
# Phase 2
# =============================================================================
def phase2_empirical(df_filtered, returns, ff_factors):
    """Run all empirical tests: persistence, portfolio sorts, factor regressions."""

    print("\n" + "=" * 80)
    print("PHASE 2: EMPIRICAL ANALYSIS")
    print("=" * 80)

    results = {}

    # --- 2.1 Signal persistence ---
    print("\n--- 2.1 Signal persistence ---")
    results['persistence'] = test_signal_persistence(df_filtered)

    # --- 2.2-2.3 Portfolio sorts ---
    print("\n--- 2.2-2.3 Portfolio sorts ---")
    # Build the holding panel once (k=FORM_LAG_MONTHS formation gap, HOLD_MONTHS hold),
    # then reuse it for the sorts, the CAPM check, Fama-MacBeth, and risk analysis so
    # every test shares the identical signal->return timing convention.
    panel = build_holding_panel(df_filtered, returns)
    results['panel'] = panel
    results['portfolios'] = run_portfolio_analysis(panel, ff_factors)

    # --- 2.4 CAPM: sample value-weighted market vs S&P 500 ---
    print("\n--- 2.4 CAPM: sample market portfolio vs S&P 500 ---")
    results['capm_sp500'] = run_capm_vs_sp500(
        results['portfolios'].get('market_ret'), returns, ff_factors)

    # --- 2.5 Fama-MacBeth (same timing as the sorts) ---
    print("\n--- 2.5 Fama-MacBeth regressions ---")
    results['fama_macbeth'] = run_fama_macbeth(panel, ff_factors)

    # --- 2.6 Risk analysis ---
    print("\n--- 2.6 Risk analysis ---")
    results['risk'] = run_risk_analysis(panel, ff_factors)

    return results


if __name__ == '__main__':
    # Run manifest: a date-time run ID so this run's outputs are identifiable later.
    RUN_ID = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    print("=" * 80)
    print("  RRR FINANCIAL IMPLICATIONS: ANALYSIS V2")
    print("  Revenue Retention Rates & Stock Prices: High Returns, Low Risk")
    print(f"  RUN ID: {RUN_ID}")
    print(f"  Timing convention: form {FORM_LAG_MONTHS} months after quarter-end, "
          f"hold {HOLD_MONTHS} months")
    print("=" * 80)
    print()

    # Phase 1B runs FIRST since 2026-09-08: phase 1 needs the FF risk-free rate.
    ff_factors = phase1b_load_ff_factors()

    # Phase 1: Data loading and diagnostics
    df_long, df_filtered, industry_stats = phase1_load_and_diagnose(ff_factors)

    # Phase 1C: Monthly returns
    returns = phase1c_load_monthly_returns()

    # Phase 2: Empirical analysis (builds the holding panel; sorts, CAPM, FM, risk)
    empirical_results = phase2_empirical(df_filtered, returns, ff_factors)
    panel = empirical_results['panel']

    # Write the run manifest.
    manifest = {
        'run_id': RUN_ID,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'form_lag_months': FORM_LAG_MONTHS,
        'hold_months': HOLD_MONTHS,
        'n_firms': int(df_filtered.index.get_level_values('FIRM').nunique()),
        'expected_n_firms': EXPECTED_N_FIRMS,
        'python_version': platform.python_version(),
        'pandas_version': pd.__version__,
    }
    manifest_path = os.path.join(OUTPUT_DIR, f'run_manifest_{RUN_ID}.json')
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print(f"  RUN ID: {RUN_ID}")
    print(f"  Manifest: {manifest_path}")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)

    # List output files
    print("\n  Output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"    {f} ({size:,} bytes)")
