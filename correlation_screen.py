"""
correlation_screen.py -- RRR Financial Implications
=======================================================
Codifies the credit-card-panel-vs-Bloomberg revenue correlation screen
(Shah, Kumar, and Zhao 2017 methodology) that selects the project's firm
sample. Firms are kept only if their credit-card-panel quarterly revenue
correlates with Bloomberg-reported quarterly revenue (SALES_REV_TURN) at
Pearson r >= threshold (default 0.75) over the overlapping sample period.

Until now this screen has only ever been applied manually in Excel
(Data/Firms-Quarterly-Revenue-Fundamentals-2025-07-02-BBG.xlsx, sheets
'Corr_R2_Values' and 'Final_Companies') -- never in code. This script
closes that reproducibility gap: it recomputes the correlation
independently from the same two raw source files analysis_v2.py loads,
applies a parameterized threshold, and verifies the result against the
Excel workbook.

Does NOT reimplement or import analysis_v2.py's panel-building logic.
This is a standalone, from-source recomputation, by design: an
independent script that happens to agree with analysis_v2.py's inputs is
a much stronger check than a script that reuses its code and could
silently share a bug.

Usage:
    python correlation_screen.py                    # threshold=0.75, verifies against Excel
    python correlation_screen.py --threshold 0.60    # robustness variant
    python correlation_screen.py --threshold 0.85    # robustness variant
    python correlation_screen.py --skip-verification # skip the Excel comparison step
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Data"
SUPPORT_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Supporting-Documents"
OUTPUT_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REVENUE_FILE = os.path.join(DATA_DIR, "2025-0319a-TK-quarterlyrevenue-collection_Python.xlsx")
FUNDAMENTALS_FILE = os.path.join(DATA_DIR, "2025-0319a-TK-fundamentals_Python.xlsx")
WORKBOOK_FILE = os.path.join(DATA_DIR, "Firms-Quarterly-Revenue-Fundamentals-2025-07-02-BBG.xlsx")
INDUSTRY_FILE = os.path.join(SUPPORT_DIR, "Firms-Industry-2025-11-21-Python.xlsx")

DEFAULT_THRESHOLD = 0.75

# Minimum overlapping quarters required to report a correlation. Below this,
# a Pearson r is too noisy to be meaningful (e.g. N=2 is trivially +/-1).
# No firm in the current data falls below this floor (checked at runtime).
MIN_OBS_FOR_CORR = 4

# Mirrors analysis_v2.py's VALID_SECTORS constant (sector filter, independent
# of the correlation screen). Duplicated here only for the diagnostic in
# diagnose_against_current_sample() -- keep in sync with analysis_v2.py if
# that constant ever changes. This script does not import analysis_v2.py.
VALID_SECTORS = ['Consumer Discretionary', 'Communication Services', 'Consumer Staples', 'Industrials']


# =============================================================================
# STEP (a): LOAD CREDIT-CARD-PANEL AND BLOOMBERG-REPORTED REVENUE
# =============================================================================

def load_credit_card_revenue(filepath):
    """Load credit-card-panel quarterly revenue per firm-quarter.

    Source file has a 2-row header: row 1 = ticker, row 2 = variable name
    (#New_Customers / #Returning_Customers), then one data row per quarter.
    Total credit-card revenue is New + Returning customer revenue, matching
    analysis_v2.py Phase 1's #Total_Revenue definition (line ~187).

    Firm-quarters are keyed on FISCAL_QUARTER (a pandas Period, e.g. 2017Q1)
    rather than the raw date, because the fundamentals file below labels the
    same quarter with a different exact day (last trading day vs. calendar
    quarter-end -- e.g. 2017-09-29 vs. 2017-09-30). Both files were confirmed
    to share the same 32-quarter sequence in the same order; keying on
    calendar quarter is a more defensive join than relying on row position,
    since it fails loudly (a firm-quarter simply won't match) rather than
    silently if a future refresh reorders or adds a row to only one file.
    """
    raw = pd.read_excel(filepath, header=None, engine='openpyxl')
    tickers_row = raw.iloc[0]
    variables_row = raw.iloc[1]
    dates = pd.to_datetime(raw.iloc[2:, 0]).reset_index(drop=True)
    fiscal_quarters = dates.dt.to_period('Q')
    data_rows = raw.iloc[2:].reset_index(drop=True)

    firm_quarter_records = []
    for col_idx in range(1, raw.shape[1]):
        ticker = tickers_row[col_idx]
        variable_name = variables_row[col_idx]
        if pd.isna(ticker) or variable_name not in ('#New_Customers', '#Returning_Customers'):
            continue
        values = pd.to_numeric(data_rows[col_idx], errors='coerce')
        firm_quarter_records.append(pd.DataFrame({
            'FIRM': ticker.upper().strip(),
            'FISCAL_QUARTER': fiscal_quarters,
            'VARIABLE': variable_name,
            'VALUE': values.values,
        }))

    revenue_long = pd.concat(firm_quarter_records, ignore_index=True)
    revenue_wide = revenue_long.pivot_table(
        index=['FIRM', 'FISCAL_QUARTER'], columns='VARIABLE', values='VALUE', aggfunc='sum'
    ).reset_index()

    # Total credit-card revenue = new-customer + returning-customer revenue.
    # NaN propagates if either component is missing (revenue is undefined,
    # not zero, when a component wasn't observed that quarter).
    revenue_wide['TOTAL_REVENUE_CC'] = revenue_wide['#New_Customers'] + revenue_wide['#Returning_Customers']

    return revenue_wide[['FIRM', 'FISCAL_QUARTER', 'TOTAL_REVENUE_CC']]


def load_bloomberg_revenue(filepath):
    """Load Bloomberg-reported quarterly revenue (SALES_REV_TURN) per firm-quarter.

    Same 2-row header layout as the credit-card revenue file (see
    load_credit_card_revenue docstring for the FISCAL_QUARTER join-key
    rationale).
    """
    raw = pd.read_excel(filepath, header=None, engine='openpyxl')
    tickers_row = raw.iloc[0]
    variables_row = raw.iloc[1]
    dates = pd.to_datetime(raw.iloc[2:, 0]).reset_index(drop=True)
    fiscal_quarters = dates.dt.to_period('Q')
    data_rows = raw.iloc[2:].reset_index(drop=True)

    firm_quarter_records = []
    for col_idx in range(1, raw.shape[1]):
        ticker = tickers_row[col_idx]
        variable_name = variables_row[col_idx]
        if pd.isna(ticker) or variable_name != 'SALES_REV_TURN':
            continue
        values = pd.to_numeric(data_rows[col_idx], errors='coerce')
        firm_quarter_records.append(pd.DataFrame({
            'FIRM': ticker.upper().strip(),
            'FISCAL_QUARTER': fiscal_quarters,
            'SALES_REV_TURN': values.values,
        }))

    return pd.concat(firm_quarter_records, ignore_index=True)


# =============================================================================
# STEP (b): PER-FIRM PEARSON CORRELATION OVER THE OVERLAPPING SAMPLE PERIOD
# =============================================================================

def compute_correlation_screen(cc_revenue, bbg_revenue, min_obs=MIN_OBS_FOR_CORR):
    """Compute per-firm Pearson r between credit-card and Bloomberg revenue.

    "Overlapping sample period" = the firm-quarters where both series are
    non-null, following pairwise-complete-observations convention (the
    standard for this kind of two-series correlation).
    """
    merged = pd.merge(cc_revenue, bbg_revenue, on=['FIRM', 'FISCAL_QUARTER'], how='inner')

    correlation_records = []
    for firm, firm_data in merged.groupby('FIRM'):
        valid_quarters = firm_data.dropna(subset=['TOTAL_REVENUE_CC', 'SALES_REV_TURN'])
        n_obs = len(valid_quarters)

        has_enough_obs = n_obs >= min_obs
        has_variation = (
            has_enough_obs
            and valid_quarters['TOTAL_REVENUE_CC'].std() > 0
            and valid_quarters['SALES_REV_TURN'].std() > 0
        )
        if has_variation:
            pearson_r, p_value = pearsonr(valid_quarters['TOTAL_REVENUE_CC'], valid_quarters['SALES_REV_TURN'])
        else:
            pearson_r, p_value = np.nan, np.nan

        correlation_records.append({
            'FIRM': firm,
            'N_OVERLAPPING_QUARTERS': n_obs,
            'PEARSON_R': pearson_r,
            'P_VALUE': p_value,
        })

    correlation_table = pd.DataFrame(correlation_records)
    correlation_table = correlation_table.sort_values('PEARSON_R', ascending=False, na_position='last').reset_index(drop=True)
    return correlation_table


# =============================================================================
# STEP (c): APPLY PARAMETERIZED THRESHOLD
# =============================================================================

def apply_threshold(correlation_table, threshold=DEFAULT_THRESHOLD):
    """Flag firms whose Pearson r clears the screen threshold. NaN r always fails."""
    screened = correlation_table.copy()
    screened['THRESHOLD'] = threshold
    screened['PASSES_SCREEN'] = screened['PEARSON_R'] >= threshold
    screened = screened.sort_values('PEARSON_R', ascending=False, na_position='last').reset_index(drop=True)
    return screened


# =============================================================================
# STEP (d) + VERIFICATION: COMPARE AGAINST THE EXCEL WORKBOOK
# =============================================================================

def load_excel_final_companies_screen(workbook_path):
    """Load the manually-computed correlation + inclusion flag from 'Final_Companies'.

    Sheet has no header row. Column layout (0-indexed), confirmed by inspection:
      0  = Ticker
      7  = Pearson correlation, corr(observed_sales, sales_rev_turn), BQL-computed
      8  = R-squared (= col 7 squared)
      13 = inclusion flag (1/0) -- confirmed by direct comparison to equal
           (col 7 >= 0.75) exactly, for all 135 rows, zero mismatches.
    """
    raw = pd.read_excel(workbook_path, sheet_name='Final_Companies', header=None, engine='openpyxl')
    excel_screen = pd.DataFrame({
        'FIRM': raw[0].str.upper().str.strip(),
        'EXCEL_PEARSON_R': pd.to_numeric(raw[7], errors='coerce'),
        'EXCEL_FLAG': pd.to_numeric(raw[13], errors='coerce'),
    })
    return excel_screen


def verify_against_workbook(my_results, workbook_path, threshold):
    """Compare this script's threshold pass/fail decision against Final_Companies.

    Exact note: the workbook's EXCEL_FLAG column is hardcoded at a 0.75 cutoff
    inside Excel -- it does not change with `threshold`. This comparison is
    therefore an exact apples-to-apples check only when threshold=0.75; for
    other thresholds we still report the Pearson r agreement (which is
    threshold-independent) but the pass/fail agreement column is only
    meaningful at 0.75.
    """
    excel_screen = load_excel_final_companies_screen(workbook_path)
    comparison = pd.merge(my_results, excel_screen, on='FIRM', how='outer')
    comparison['MY_PASSES'] = comparison['PASSES_SCREEN'].fillna(False)
    comparison['EXCEL_PASSES'] = comparison['EXCEL_FLAG'] == 1
    comparison['R_DIFF'] = comparison['PEARSON_R'] - comparison['EXCEL_PEARSON_R']
    comparison['PASS_FAIL_AGREES'] = comparison['MY_PASSES'] == comparison['EXCEL_PASSES']
    comparison = comparison.sort_values('FIRM').reset_index(drop=True)
    return comparison


# =============================================================================
# CONTEXTUAL DIAGNOSTIC: HOW DOES THE SCREEN RELATE TO THE 124-FIRM SAMPLE?
# =============================================================================

def diagnose_against_current_sample(screened_at_075):
    """Explain how the correlation screen relates to the documented 124-firm sample.

    The 124-firm figure (CLAUDE.md: "124 firms, quarterly 2017-2024, 4
    sectors") is NOT the output of the correlation screen. It is the output
    of analysis_v2.py's *sector* filter (VALID_SECTORS) applied to the full
    135-firm candidate universe in the revenue/fundamentals files -- with NO
    correlation screen currently applied in code. This function quantifies
    the resulting gap: how many of the 124 firms currently in the pipeline
    would NOT survive an r>=0.75 correlation screen.
    """
    industry = pd.read_excel(INDUSTRY_FILE, engine='openpyxl')
    industry.columns = ['ID', 'GICS_INDUSTRY', 'GICS_SECTOR', 'GICS_SUB_INDUSTRY']
    industry['FIRM'] = industry['ID'].str.upper().str.strip()
    firm_to_sector = industry.set_index('FIRM')['GICS_SECTOR'].to_dict()

    diagnosis = screened_at_075.copy()
    diagnosis['SECTOR'] = diagnosis['FIRM'].map(firm_to_sector)
    diagnosis['IN_VALID_SECTOR'] = diagnosis['SECTOR'].isin(VALID_SECTORS)

    sector_filtered_firms = diagnosis[diagnosis['IN_VALID_SECTOR']].copy()
    n_sector_filtered = len(sector_filtered_firms)
    n_also_passes_corr = int(sector_filtered_firms['PASSES_SCREEN'].sum())
    firms_in_sample_failing_corr = sector_filtered_firms[~sector_filtered_firms['PASSES_SCREEN']].sort_values('PEARSON_R')

    return {
        'n_candidates_total': len(diagnosis),
        'n_sector_filtered': n_sector_filtered,
        'n_sector_filtered_and_passes_corr_075': n_also_passes_corr,
        'firms_in_124_sample_failing_corr_075': firms_in_sample_failing_corr[['FIRM', 'PEARSON_R', 'N_OVERLAPPING_QUARTERS']],
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Codify the credit-card-vs-Bloomberg revenue correlation screen (Shah et al. 2017).'
    )
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                         help=f'Minimum Pearson r to pass the screen (default: {DEFAULT_THRESHOLD})')
    parser.add_argument('--skip-verification', action='store_true',
                         help='Skip the comparison against the Excel workbook')
    args = parser.parse_args()

    print("=" * 80)
    print("CORRELATION SCREEN -- credit-card-panel revenue vs. Bloomberg SALES_REV_TURN")
    print("=" * 80)

    print(f"\nLoading credit-card revenue from: {REVENUE_FILE}")
    cc_revenue = load_credit_card_revenue(REVENUE_FILE)
    print(f"  {cc_revenue['FIRM'].nunique()} firms, {cc_revenue['FISCAL_QUARTER'].nunique()} quarters")

    print(f"\nLoading Bloomberg revenue from: {FUNDAMENTALS_FILE}")
    bbg_revenue = load_bloomberg_revenue(FUNDAMENTALS_FILE)
    print(f"  {bbg_revenue['FIRM'].nunique()} firms, {bbg_revenue['FISCAL_QUARTER'].nunique()} quarters")

    correlation_table = compute_correlation_screen(cc_revenue, bbg_revenue, min_obs=MIN_OBS_FOR_CORR)
    n_below_min_obs = correlation_table['PEARSON_R'].isna().sum()
    print(f"\nComputed Pearson r for {len(correlation_table)} firms "
          f"({n_below_min_obs} below the {MIN_OBS_FOR_CORR}-quarter minimum or zero-variance)")

    screened = apply_threshold(correlation_table, args.threshold)
    passing_firms = screened[screened['PASSES_SCREEN']].sort_values('FIRM').reset_index(drop=True)

    print(f"\nThreshold = {args.threshold}")
    print(f"  Firms evaluated: {len(screened)}")
    print(f"  Firms passing:   {len(passing_firms)}")

    # correlation_table (FIRM, N_OVERLAPPING_QUARTERS, PEARSON_R, P_VALUE) is
    # threshold-independent, so it always gets the same untagged filename --
    # re-running at a different --threshold must not change this file's
    # content. Threshold-dependent pass/fail decisions only ever go into the
    # threshold-tagged files below.
    full_table_path = os.path.join(OUTPUT_DIR, 'correlation_screen_all_firms.xlsx')
    correlation_table.to_excel(full_table_path, index=False, engine='openpyxl')
    print(f"\nSaved full diagnostic table (threshold-independent): {full_table_path}")

    threshold_tag = str(args.threshold).replace('.', '')
    passing_path = os.path.join(OUTPUT_DIR, f'correlation_screen_passing_firms_thr{threshold_tag}.xlsx')
    passing_firms.to_excel(passing_path, index=False, engine='openpyxl')
    print(f"Saved passing-firm list:     {passing_path}")

    screened_path = os.path.join(OUTPUT_DIR, f'correlation_screen_scored_thr{threshold_tag}.xlsx')
    screened.to_excel(screened_path, index=False, engine='openpyxl')
    print(f"Saved full scored table:     {screened_path}")

    if not args.skip_verification:
        print("\n" + "=" * 80)
        print(f"VERIFICATION vs. Excel workbook 'Final_Companies' sheet")
        print("=" * 80)
        comparison = verify_against_workbook(screened, WORKBOOK_FILE, args.threshold)
        n_agree = int(comparison['PASS_FAIL_AGREES'].sum())
        n_total = len(comparison)
        print(f"  Pass/fail agreement: {n_agree} / {n_total} firms"
              + ("" if args.threshold == 0.75 else "  (NOTE: Excel flag is fixed at 0.75; only exact at threshold=0.75)"))
        max_r_diff = comparison['R_DIFF'].abs().max()
        mean_r_diff = comparison['R_DIFF'].abs().mean()
        print(f"  Pearson r agreement: max |diff| = {max_r_diff:.4f}, mean |diff| = {mean_r_diff:.4f}")

        disagreements = comparison[~comparison['PASS_FAIL_AGREES']]
        if len(disagreements) > 0:
            print(f"\n  {len(disagreements)} firms where MY_PASSES != EXCEL_PASSES:")
            print(disagreements[['FIRM', 'PEARSON_R', 'EXCEL_PEARSON_R', 'R_DIFF', 'N_OVERLAPPING_QUARTERS', 'MY_PASSES', 'EXCEL_PASSES']].to_string(index=False))

        comparison_path = os.path.join(OUTPUT_DIR, f'correlation_screen_vs_excel_thr{threshold_tag}.xlsx')
        comparison.to_excel(comparison_path, index=False, engine='openpyxl')
        print(f"\n  Saved full comparison: {comparison_path}")

        if args.threshold == 0.75:
            print("\n" + "=" * 80)
            print("DIAGNOSTIC: correlation screen vs. the documented 124-firm sample")
            print("=" * 80)
            diagnosis = diagnose_against_current_sample(screened)
            print(f"  Candidate universe (revenue+fundamentals files): {diagnosis['n_candidates_total']} firms")
            print(f"  After sector filter only (current analysis_v2.py behavior): {diagnosis['n_sector_filtered']} firms")
            print(f"  Of those, N also passing r>=0.75: {diagnosis['n_sector_filtered_and_passes_corr_075']} firms")
            n_failing = diagnosis['n_sector_filtered'] - diagnosis['n_sector_filtered_and_passes_corr_075']
            print(f"  Of those, N FAILING r>=0.75 (in the pipeline despite failing the screen): {n_failing} firms")
            if n_failing > 0:
                print("\n  Firms currently in the sample that fail the r>=0.75 screen:")
                print(diagnosis['firms_in_124_sample_failing_corr_075'].to_string(index=False))

    print("\nDone.")
