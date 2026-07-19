"""
analysis_v2.py — RRR Financial Implications
=============================================
Revenue Retention Rates & Stock Prices: High Returns, Low Risk

This script implements the full empirical analysis:
  Phase 1: Data loading, sector filter, industry diagnostics
  Phase 2: Signal persistence, portfolio sorts, factor regressions,
           Fama-MacBeth, risk analysis
  Phase 4: Same-growth placebo, robustness tests

Preserves the original 2025-06-04b script as-is.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch runs
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis
import scipy.stats as sps
import warnings
import os
import io

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Data"
SUPPORT_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Supporting-Documents"
OUTPUT_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sector filter: only sectors with 10+ firms
VALID_SECTORS = ['Consumer Discretionary', 'Communication Services', 'Consumer Staples', 'Industrials']

# Fama-French factor data URL
FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

# =============================================================================
# PORTFOLIO TIMING CONVENTION (calendar-month based, NOT quarter-integer shifts)
# =============================================================================
# Every signal (RRR, adjusted RRR, AR, SRR, ...) is measured at a calendar
# quarter-end date Q. The portfolio is FORMED FORM_LAG_MONTHS calendar months
# after Q, so the revenue/RRR figure is public before we trade on it, and then
# HELD for HOLD_MONTHS months. Returns are therefore earned in calendar months
#   [Q + FORM_LAG_MONTHS + 1, ..., Q + FORM_LAG_MONTHS + HOLD_MONTHS].
#   Example: Q = 3/31  ->  form at end of May  ->  hold Jun, Jul, Aug.
#
# CRITICAL: the lag lives ONLY in this formation gap. Signals are used
# CONTEMPORANEOUSLY (measured at Q), never additionally .shift()-ed, so that a
# second implicit quarter-lag is never stacked on top of the formation gap.
# The identical convention is applied to the RRR sorts, the value-weighted
# market portfolio, the Fama-MacBeth regression, and the alternative-metric
# sorts, all through build_holding_panel().
FORM_LAG_MONTHS = 2      # calendar months from quarter-end Q to the formation date
HOLD_MONTHS = 3          # holding-period length in months
EXPECTED_N_FIRMS = 124   # sample-size invariant, asserted at portfolio formation

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_strings(strings):
    """Normalize firm names for matching."""
    return set(s.upper().strip() for s in strings)


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


def newey_west_ols(y, X, max_lags=None):
    """Run OLS with Newey-West standard errors."""
    X = sm.add_constant(X)
    if max_lags is None:
        max_lags = max(1, int(np.floor(4 * (len(y)/100)**(2/9))))
    model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
    return model


def download_ff_factors():
    """Download FF3, FF5, and Momentum factors from Ken French website."""
    import urllib.request
    import zipfile

    factors = {}

    for name, url in [('FF3', FF3_URL), ('FF5', FF5_URL), ('MOM', MOM_URL)]:
        try:
            print(f"  Downloading {name} factors...")
            local_zip = os.path.join(OUTPUT_DIR, f'{name}.zip')
            urllib.request.urlretrieve(url, local_zip)

            with zipfile.ZipFile(local_zip, 'r') as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    content = f.read().decode('utf-8')

            # Parse: find the monthly data section (before annual)
            lines = content.split('\n')
            data_lines = []
            header_found = False
            for line in lines:
                line = line.strip()
                if not line:
                    if header_found:
                        break
                    continue
                # Skip description lines
                if line[0].isdigit() and len(line.split(',')[0].strip()) == 6:
                    data_lines.append(line)
                    header_found = True
                elif header_found and line[0].isdigit():
                    if len(line.split(',')[0].strip()) == 6:
                        data_lines.append(line)
                    else:
                        break  # Hit annual data

            if name == 'MOM':
                header = 'Date,Mom'
            elif name == 'FF3':
                header = 'Date,Mkt-RF,SMB,HML,RF'
            else:
                header = 'Date,Mkt-RF,SMB,HML,RMW,CMA,RF'

            csv_text = header + '\n' + '\n'.join(data_lines)
            df = pd.read_csv(io.StringIO(csv_text))
            df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), format='%Y%m')
            df = df.set_index('Date').sort_index()
            # Convert from percent to decimal
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100
            factors[name] = df

            os.remove(local_zip)
            print(f"  {name}: {len(df)} months loaded")
        except Exception as e:
            print(f"  WARNING: Failed to download {name}: {e}")

    return factors


# =============================================================================
# PHASE 1: DATA LOADING & INDUSTRY DIAGNOSTICS
# =============================================================================

def phase1_load_and_diagnose():
    """Load all data, apply sector filter, compute industry diagnostics."""

    print("=" * 80)
    print("PHASE 1: DATA LOADING & INDUSTRY DIAGNOSTICS")
    print("=" * 80)

    # --- 1.1 Load revenue data ---
    print("\n--- 1.1 Loading revenue data ---")
    file_rev = os.path.join(DATA_DIR, "2025-0319a-TK-quarterlyrevenue-collection_Python.xlsx")
    df_revenue = pd.read_excel(file_rev, header=None)

    header_rows = df_revenue.iloc[:2]
    data_rows = df_revenue.iloc[2:]
    combined_headers = header_rows.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
    data_rows.columns = column_headers
    data_rows.rename(columns={data_rows.columns[0]: 'Date'}, inplace=True)
    df_revenue = data_rows.reset_index(drop=True)
    df_revenue = df_revenue.replace(r'^\s*$', pd.NA, regex=True)
    df_revenue = df_revenue.loc[:, ~df_revenue.columns.duplicated()]

    # Compute metrics per firm
    new_customer_cols = [col for col in df_revenue.columns if '#New_Customers' in col]
    returning_customer_cols = [col for col in df_revenue.columns if '#Returning_Customers' in col]

    for new_col, ret_col in zip(new_customer_cols, returning_customer_cols):
        total_col = new_col.replace('#New_Customers', '#Total_Revenue')
        rrr_col = ret_col.replace('#Returning_Customers', '#RRR')
        rg_col = ret_col.replace('#Returning_Customers', '#Revenue_Growth')
        ar_col = new_col.replace('#New_Customers', '#Acq_Rate')
        share_col = ret_col.replace('#Returning_Customers', '#Share_Ret_Revenue')

        df_revenue[total_col] = df_revenue[new_col] + df_revenue[ret_col]
        df_revenue[share_col] = df_revenue[ret_col] / df_revenue[total_col].replace(0, np.nan)
        df_revenue[rrr_col] = df_revenue[ret_col] / df_revenue[total_col].shift(1).replace(0, np.nan)
        df_revenue[rg_col] = (df_revenue[total_col] - df_revenue[total_col].shift(1)) / df_revenue[total_col].shift(1)
        df_revenue[ar_col] = df_revenue[new_col] / df_revenue[new_col].shift(1)

    print(f"  Revenue data loaded: {df_revenue.shape}")

    # --- 1.2 Load fundamentals ---
    print("\n--- 1.2 Loading fundamentals ---")
    file_fun = os.path.join(DATA_DIR, "2025-0319a-TK-fundamentals_Python.xlsx")
    df_fundamentals = pd.read_excel(file_fun, header=None)

    header_rows_fun = df_fundamentals.iloc[:2]
    data_rows_fun = df_fundamentals.iloc[2:]
    combined_headers = header_rows_fun.apply(lambda x: x.astype(str).str.strip(), axis=0)
    column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
    data_rows_fun.columns = column_headers
    df_fundamentals = data_rows_fun.reset_index(drop=True)
    df_fundamentals.replace('#N/A N/A', np.nan, inplace=True)
    df_fundamentals = df_fundamentals.loc[:, ~df_fundamentals.columns.duplicated()]

    # Fill R&D and SGA NaNs with 0
    cols_fill = [col for col in df_fundamentals.columns if "IS_RD_EXPEND" in col or "IS_SGA_EXPENSE" in col]
    df_fundamentals[cols_fill] = df_fundamentals[cols_fill].fillna(0)

    print(f"  Fundamentals loaded: {df_fundamentals.shape}")

    # --- 1.3 Merge to long format ---
    print("\n--- 1.3 Merging to long panel ---")
    excluded_columns = ['Date']
    df_revenue_to_append = df_revenue.drop(columns=excluded_columns, errors='ignore')
    df_fundamentals = df_fundamentals.rename(columns={'nan.Dates': 'Date'})
    df_combined = pd.concat([df_fundamentals, df_revenue_to_append], axis=1)
    df_combined.columns = df_combined.columns.str.upper()
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

    all_vars = [col for col in df_combined.columns if col != "DATE"]
    df_long = df_combined.melt(id_vars=["DATE"], value_vars=all_vars, var_name="Firm_Variable", value_name="Value")
    df_long[['FIRM', 'VARIABLE']] = df_long['Firm_Variable'].str.rsplit('.', n=1, expand=True)
    df_long = df_long.pivot_table(index=['FIRM', 'DATE'], columns='VARIABLE', values='Value').reset_index()
    df_long = df_long.set_index(['FIRM', 'DATE'])

    # Remove SPX INDEX, SPW INDEX, USBMMY3M INDEX from firm list
    index_firms = ['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX']
    all_firms = df_long.index.get_level_values('FIRM').unique()
    actual_firms = [f for f in all_firms if f not in index_firms]

    print(f"  Panel: {len(actual_firms)} firms, {df_long.index.get_level_values('DATE').nunique()} periods")

    # --- 1.4 Compute derived variables ---
    print("\n--- 1.4 Computing derived variables ---")

    df_long['RRR_PCT'] = pd.to_numeric(df_long.get('#RRR', np.nan), errors='coerce') * 100
    df_long['ACQ_RATE_PCT'] = pd.to_numeric(df_long.get('#ACQ_RATE', np.nan), errors='coerce') * 100
    df_long['REV_GROWTH_PCT'] = pd.to_numeric(df_long.get('#REVENUE_GROWTH', np.nan), errors='coerce') * 100
    # Share of new revenue: Rev_new / Rev_t = 1 - RRR / (1 + RG); undefined when RG = -100%
    df_long['SHARE_NEW_REV_PCT'] = np.where(
        df_long['REV_GROWTH_PCT'] == -100,
        np.nan,
        (1 - (df_long['RRR_PCT'] / 100) / (1 + df_long['REV_GROWTH_PCT'] / 100)) * 100
    )

    # Lagged values
    df_long['RRR_PCT_LAG1'] = df_long.groupby(level='FIRM')['RRR_PCT'].shift(1)
    df_long['RRR_LAG'] = df_long['RRR_PCT_LAG1'] / 100
    df_long['ACQ_RATE_PCT_LAG1'] = df_long.groupby(level='FIRM')['ACQ_RATE_PCT'].shift(1)

    # Size, BTM
    df_long['HISTORICAL_MARKET_CAP'] = pd.to_numeric(df_long.get('HISTORICAL_MARKET_CAP', np.nan), errors='coerce')
    df_long['PX_LAST'] = pd.to_numeric(df_long.get('PX_LAST', np.nan), errors='coerce')
    df_long['SIZE'] = np.log(df_long['HISTORICAL_MARKET_CAP'].replace(0, np.nan))
    df_long['BTM'] = (pd.to_numeric(df_long.get('BS_TOT_ASSET', np.nan), errors='coerce') -
                       pd.to_numeric(df_long.get('BS_TOT_LIAB2', np.nan), errors='coerce')) / df_long['HISTORICAL_MARKET_CAP'].replace(0, np.nan)
    df_long['PM_OPER_PCT'] = (pd.to_numeric(df_long.get('IS_OPER_INC', np.nan), errors='coerce') /
                               pd.to_numeric(df_long.get('SALES_REV_TURN', np.nan), errors='coerce').replace(0, np.nan)) * 100
    # Market-to-Equity (direct from Bloomberg TOTAL_EQUITY)
    df_long['MTE'] = (df_long['HISTORICAL_MARKET_CAP'] /
                      pd.to_numeric(df_long.get('TOTAL_EQUITY', np.nan), errors='coerce').replace(0, np.nan))
    # Operating ROA proxy (no net income in fundamentals file)
    df_long['ROA_OPER'] = (pd.to_numeric(df_long.get('IS_OPER_INC', np.nan), errors='coerce') /
                           pd.to_numeric(df_long.get('BS_TOT_ASSET', np.nan), errors='coerce').replace(0, np.nan)) * 100

    # Stock returns (price-only, log)
    df_long['PX_LAST_LAG1'] = df_long.groupby(level='FIRM')['PX_LAST'].shift(1)
    df_long['RETURN_LOG'] = np.log(df_long['PX_LAST'] / df_long['PX_LAST_LAG1'])
    df_long['RET_ARITH'] = np.exp(df_long['RETURN_LOG']) - 1

    # Risk-free rate
    try:
        rf_ann_pct = df_long.xs('USBMMY3M INDEX', level='FIRM')['PX_LAST']
        conversion = 4
        rf = ((1 + rf_ann_pct/100)**(1/conversion) - 1)
        dates = df_long.index.get_level_values('DATE')
        df_long['RF'] = rf.reindex(dates).values
    except KeyError:
        df_long['RF'] = 0

    df_long['EXCESS_RET'] = df_long['RET_ARITH'] - df_long['RF']

    # --- 1.5 Load industry classifications ---
    print("\n--- 1.5 Loading industry classifications ---")
    file_ind = os.path.join(SUPPORT_DIR, "Firms-Industry-2025-11-21-Python.xlsx")
    df_industry = pd.read_excel(file_ind)
    df_industry.columns = ['ID', 'GICS_INDUSTRY', 'GICS_SECTOR', 'GICS_SUB_INDUSTRY']
    df_industry['FIRM'] = df_industry['ID'].str.upper().str.strip()

    # Map firms to sectors
    firm_sector = df_industry.set_index('FIRM')['GICS_SECTOR'].to_dict()

    # Add sector to df_long
    firms_in_panel = df_long.index.get_level_values('FIRM')
    df_long['SECTOR'] = firms_in_panel.map(firm_sector)

    # --- 1.6 Sector filter ---
    print("\n--- 1.6 Applying sector filter ---")
    sector_counts = df_long.reset_index().groupby('SECTOR')['FIRM'].nunique()
    print("  Firm counts by sector (before filter):")
    print(sector_counts.sort_values(ascending=False).to_string())

    # Keep only valid sectors
    mask = df_long['SECTOR'].isin(VALID_SECTORS)
    df_filtered = df_long[mask].copy()

    n_firms_filtered = df_filtered.reset_index()['FIRM'].nunique()
    print(f"\n  After sector filter: {n_firms_filtered} firms")

    sector_counts_filtered = df_filtered.reset_index().groupby('SECTOR')['FIRM'].nunique()
    print("  Firm counts by sector (after filter):")
    print(sector_counts_filtered.sort_values(ascending=False).to_string())

    # Firms per quarter
    firms_per_q = df_filtered.reset_index().groupby('DATE')['FIRM'].nunique()
    print(f"\n  Firms per quarter: min={firms_per_q.min()}, max={firms_per_q.max()}, mean={firms_per_q.mean():.0f}")

    # --- 1.7 Industry x time adjustment ---
    print("\n--- 1.7 Computing industry x time adjusted signals ---")

    # RRR adjustment
    sector_quarter_mean_rrr = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])['RRR_PCT'].transform('mean')
    df_filtered['ADJ_RRR_PCT'] = df_filtered['RRR_PCT'] - sector_quarter_mean_rrr

    # AR adjustment
    sector_quarter_mean_ar = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])['ACQ_RATE_PCT'].transform('mean')
    df_filtered['ADJ_ACQ_RATE_PCT'] = df_filtered['ACQ_RATE_PCT'] - sector_quarter_mean_ar

    # Lagged adjusted signals
    df_filtered['ADJ_RRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_RRR_PCT'].shift(1)
    df_filtered['ADJ_RRR_LAG'] = df_filtered['ADJ_RRR_PCT_LAG1'] / 100
    df_filtered['ADJ_ACQ_RATE_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_ACQ_RATE_PCT'].shift(1)

    print("  Adjusted signals computed: ADJ_RRR_PCT, ADJ_ACQ_RATE_PCT")

    # --- 1.4b Alternative RRR variants ---
    print("\n--- 1.4b Alternative RRR variants ---")

    _ret = pd.to_numeric(df_filtered['#RETURNING_CUSTOMERS'], errors='coerce')
    _new = pd.to_numeric(df_filtered['#NEW_CUSTOMERS'], errors='coerce')
    _tot = pd.to_numeric(df_filtered['#TOTAL_REVENUE'], errors='coerce')

    # Variant A: 1-year-lag RRR — Rev_ret_t / total_revenue_{t-4}
    df_filtered['_TOT'] = _tot
    _tot_lag4 = df_filtered.groupby(level='FIRM')['_TOT'].shift(4).replace(0, np.nan)
    df_filtered.drop(columns=['_TOT'], inplace=True)
    df_filtered['LR_RRR_PCT'] = (_ret / _tot_lag4) * 100
    _sm = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])[
        'LR_RRR_PCT'].transform('mean')
    df_filtered['ADJ_LR_RRR_PCT'] = df_filtered['LR_RRR_PCT'] - _sm

    # Variant B: Share of Retained Revenue — Rev_ret_t / Rev_t  (= 1 - SoNR)
    df_filtered['SRR_PCT'] = (_ret / _tot.replace(0, np.nan)) * 100
    _sm = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])[
        'SRR_PCT'].transform('mean')
    df_filtered['ADJ_SRR_PCT'] = df_filtered['SRR_PCT'] - _sm

    # Variant C: Cumulative-denominator RRR
    # CUMRR_t = Rev_ret_t / (Rev_ret_0 + sum_{s<t} Rev_new_s)
    def _compute_cumrr_vals(grp_sorted):
        ret_rev = pd.to_numeric(grp_sorted['#RETURNING_CUSTOMERS'], errors='coerce')
        new_rev = pd.to_numeric(grp_sorted['#NEW_CUSTOMERS'], errors='coerce')
        valid = ret_rev.dropna()
        if valid.empty or valid.iloc[0] == 0:
            return np.full(len(grp_sorted), np.nan)
        ret_rev_0 = valid.iloc[0]
        cum_new_prev = new_rev.shift(1).fillna(0).cumsum()
        denom = (ret_rev_0 + cum_new_prev).replace(0, np.nan)
        return (ret_rev / denom * 100).values

    df_reset = df_filtered.reset_index()
    df_reset['CUMRR_PCT'] = np.nan
    for _firm, _grp in df_reset.groupby('FIRM'):
        _grp_s = _grp.sort_values('DATE')
        df_reset.loc[_grp_s.index, 'CUMRR_PCT'] = _compute_cumrr_vals(_grp_s)
    df_filtered = df_reset.set_index(['FIRM', 'DATE'])

    _sm = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])[
        'CUMRR_PCT'].transform('mean')
    df_filtered['ADJ_CUMRR_PCT'] = df_filtered['CUMRR_PCT'] - _sm

    # Lag-1 for portfolio sorting
    df_filtered['ADJ_LR_RRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_LR_RRR_PCT'].shift(1)
    df_filtered['ADJ_SRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_SRR_PCT'].shift(1)
    df_filtered['ADJ_CUMRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_CUMRR_PCT'].shift(1)

    print(f"  LR_RRR: {df_filtered['LR_RRR_PCT'].notna().sum()} obs; "
          f"SRR: {df_filtered['SRR_PCT'].notna().sum()}; "
          f"CUMRR: {df_filtered['CUMRR_PCT'].notna().sum()}")

    # --- 1.8 Per-industry descriptive statistics ---
    print("\n--- 1.8 Per-industry descriptive statistics ---")

    metrics = {
        'RG (%)': 'REV_GROWTH_PCT',
        'AR (%)': 'ACQ_RATE_PCT',
        'RRR (%)': 'RRR_PCT'
    }

    all_industry_stats = []

    for sector in VALID_SECTORS:
        sector_data = df_filtered[df_filtered['SECTOR'] == sector]
        n_firms = sector_data.reset_index()['FIRM'].nunique()

        print(f"\n  {sector} ({n_firms} firms):")

        for metric_name, col_name in metrics.items():
            vals = pd.to_numeric(sector_data[col_name], errors='coerce').dropna()
            if len(vals) > 0:
                stats = {
                    'Sector': sector,
                    'Metric': metric_name,
                    'N': len(vals),
                    'Mean': vals.mean(),
                    'Median': vals.median(),
                    'SD': vals.std(),
                    'P10': vals.quantile(0.10),
                    'P25': vals.quantile(0.25),
                    'P75': vals.quantile(0.75),
                    'P90': vals.quantile(0.90)
                }
                all_industry_stats.append(stats)
                print(f"    {metric_name}: N={stats['N']}, Mean={stats['Mean']:.2f}, "
                      f"Median={stats['Median']:.2f}, SD={stats['SD']:.2f}, "
                      f"P10={stats['P10']:.2f}, P25={stats['P25']:.2f}, "
                      f"P75={stats['P75']:.2f}, P90={stats['P90']:.2f}")

    industry_stats_df = pd.DataFrame(all_industry_stats)
    industry_stats_df.to_excel(os.path.join(OUTPUT_DIR, 'industry_descriptives.xlsx'), index=False)

    # --- 1.9 Pooled descriptive statistics ---
    print("\n--- 1.9 Pooled descriptive statistics ---")

    summary_vars = {
        'Assets': 'BS_TOT_ASSET',
        'Market Cap': 'HISTORICAL_MARKET_CAP',
        'Revenue': 'SALES_REV_TURN',
        'RRR (%)': 'RRR_PCT',
        'Acq Rate (%)': 'ACQ_RATE_PCT',
        'Rev Growth (%)': 'REV_GROWTH_PCT',
        'BTM': 'BTM',
        'PM (%)': 'PM_OPER_PCT',
        'MTE': 'MTE',
        'Op. ROA (%)': 'ROA_OPER',
        'Adj RRR (%)': 'ADJ_RRR_PCT',
        'Adj AR (%)': 'ADJ_ACQ_RATE_PCT',
    }

    pooled_stats = []
    for display_name, col_name in summary_vars.items():
        if col_name in df_filtered.columns:
            data = pd.to_numeric(df_filtered[col_name], errors='coerce').dropna()
            pooled_stats.append({
                'Variable': display_name,
                'N': len(data),
                'Mean': data.mean(),
                'Median': data.median(),
                'SD': data.std(),
                'P10': data.quantile(0.10),
                'P25': data.quantile(0.25),
                'P75': data.quantile(0.75),
                'P90': data.quantile(0.90)
            })

    pooled_df = pd.DataFrame(pooled_stats)
    print(pooled_df.to_string(index=False))
    pooled_df.to_excel(os.path.join(OUTPUT_DIR, 'pooled_descriptives.xlsx'), index=False)

    # --- 1.9b Portfolio characteristics by adj-RRR quartile ---
    print("\n--- 1.9b Portfolio characteristics by adj-RRR quartile ---")
    df_filtered['RRR_Q_ADJ_CHAR'] = df_filtered.groupby(
        df_filtered.index.get_level_values('DATE'))['ADJ_RRR_PCT'].transform(safe_quartile)
    df_filtered['AR_Q_ADJ_CHAR'] = df_filtered.groupby(
        df_filtered.index.get_level_values('DATE'))['ADJ_ACQ_RATE_PCT'].transform(safe_quartile)
    df_filtered['RRR_Q_RAW_CHAR'] = df_filtered.groupby(
        df_filtered.index.get_level_values('DATE'))['RRR_PCT'].transform(safe_quartile)
    df_filtered['AR_Q_RAW_CHAR'] = df_filtered.groupby(
        df_filtered.index.get_level_values('DATE'))['ACQ_RATE_PCT'].transform(safe_quartile)

    char_cols = {
        'Revenue (M)': 'SALES_REV_TURN',
        'Market Cap (M)': 'HISTORICAL_MARKET_CAP',
        'Rev Growth (%)': 'REV_GROWTH_PCT',
        'SoNR (%)': 'SHARE_NEW_REV_PCT',
        'RRR (%)': 'RRR_PCT',
        'BTM': 'BTM',
        'MTE': 'MTE',
        'Op. ROA (%)': 'ROA_OPER',
    }

    quartile_char_rows = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        qdf = df_filtered[df_filtered['RRR_Q_ADJ_CHAR'] == q].reset_index()
        avg_n = qdf.groupby('DATE')['FIRM'].nunique().mean()
        row = {'Quartile': q, 'Avg N / quarter': round(avg_n, 1)}
        for var_label, col_name in char_cols.items():
            if col_name in qdf.columns:
                vals = pd.to_numeric(qdf[col_name], errors='coerce').dropna()
                row[var_label] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
            else:
                row[var_label] = np.nan
        quartile_char_rows.append(row)

    quartile_char_df = pd.DataFrame(quartile_char_rows)
    print(quartile_char_df.to_string(index=False))
    quartile_char_df.to_excel(os.path.join(OUTPUT_DIR, 'portfolio_characteristics.xlsx'), index=False)

    # --- 1.9c Quartile persistence rate ---
    print("\n--- 1.9c Quartile persistence rate ---")
    pers_df = df_filtered[['RRR_Q_ADJ_CHAR']].copy().reset_index()
    pers_df = pers_df.sort_values(['FIRM', 'DATE'])
    pers_df['Q_PREV'] = pers_df.groupby('FIRM')['RRR_Q_ADJ_CHAR'].shift(1)
    valid = pers_df.dropna(subset=['RRR_Q_ADJ_CHAR', 'Q_PREV'])
    overall_rate = (valid['RRR_Q_ADJ_CHAR'] == valid['Q_PREV']).mean()
    print(f"  Overall quartile persistence rate: {overall_rate:.1%}")
    per_q = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = valid[valid['Q_PREV'] == q]
        per_q[q] = (sub['RRR_Q_ADJ_CHAR'] == q).mean() if len(sub) > 0 else np.nan
        print(f"  {q} stay rate: {per_q[q]:.1%}")
    pers_out = pd.DataFrame([{
        'overall_persistence': overall_rate,
        'Q1_persistence': per_q['Q1'],
        'Q2_persistence': per_q['Q2'],
        'Q3_persistence': per_q['Q3'],
        'Q4_persistence': per_q['Q4'],
    }])
    pers_out.to_excel(os.path.join(OUTPUT_DIR, 'quartile_persistence.xlsx'), index=False)

    # --- 1.9d Signal persistence comparison: Raw vs Adj, RRR vs AR, lag-1 and lag-4 ---
    print("\n--- 1.9d Signal persistence comparison ---")

    def _stay_rates(df, q_col):
        pdf = df[[q_col]].copy().reset_index().sort_values(['FIRM', 'DATE'])
        pdf['Q_PREV'] = pdf.groupby('FIRM')[q_col].shift(1)
        valid = pdf.dropna(subset=[q_col, 'Q_PREV'])
        overall = (valid[q_col] == valid['Q_PREV']).mean()
        per_q = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            sub = valid[valid['Q_PREV'] == q]
            per_q[q] = (sub[q_col] == q).mean() if len(sub) > 0 else np.nan
        return overall, per_q

    sr_raw_rrr_ov, sr_raw_rrr = _stay_rates(df_filtered, 'RRR_Q_RAW_CHAR')
    sr_adj_rrr_ov, sr_adj_rrr = _stay_rates(df_filtered, 'RRR_Q_ADJ_CHAR')
    sr_raw_ar_ov,  sr_raw_ar  = _stay_rates(df_filtered, 'AR_Q_RAW_CHAR')
    sr_adj_ar_ov,  sr_adj_ar  = _stay_rates(df_filtered, 'AR_Q_ADJ_CHAR')

    dates_idx = df_filtered.index.get_level_values('DATE')
    cs_sd = {
        'Raw RRR': df_filtered.groupby(dates_idx)['RRR_PCT'].std().mean(),
        'Adj RRR': df_filtered.groupby(dates_idx)['ADJ_RRR_PCT'].std().mean(),
        'Raw AR':  df_filtered.groupby(dates_idx)['ACQ_RATE_PCT'].std().mean(),
        'Adj AR':  df_filtered.groupby(dates_idx)['ADJ_ACQ_RATE_PCT'].std().mean(),
    }
    for k, v in cs_sd.items():
        print(f"  {k} cross-sect SD: {v:.2f}%/quarter")

    # Lag-4 columns (lag-1 already in df_filtered from §1.7)
    df_filtered['RRR_PCT_LAG4']          = df_filtered.groupby(level='FIRM')['RRR_PCT'].shift(4)
    df_filtered['ADJ_RRR_PCT_LAG4']      = df_filtered.groupby(level='FIRM')['ADJ_RRR_PCT'].shift(4)
    df_filtered['ACQ_RATE_PCT_LAG4']     = df_filtered.groupby(level='FIRM')['ACQ_RATE_PCT'].shift(4)
    df_filtered['ADJ_ACQ_RATE_PCT_LAG4'] = df_filtered.groupby(level='FIRM')['ADJ_ACQ_RATE_PCT'].shift(4)

    signal_specs = [
        ('Raw RRR', 'RRR_PCT',          'RRR_PCT_LAG1',          'RRR_PCT_LAG4'),
        ('Adj RRR', 'ADJ_RRR_PCT',      'ADJ_RRR_PCT_LAG1',      'ADJ_RRR_PCT_LAG4'),
        ('Raw AR',  'ACQ_RATE_PCT',     'ACQ_RATE_PCT_LAG1',     'ACQ_RATE_PCT_LAG4'),
        ('Adj AR',  'ADJ_ACQ_RATE_PCT', 'ADJ_ACQ_RATE_PCT_LAG1', 'ADJ_ACQ_RATE_PCT_LAG4'),
    ]
    ac = {}
    for sig_name, col, lag1_col, lag4_col in signal_specs:
        ac[sig_name] = {}
        for lag_label, lag_col in [('lag1', lag1_col), ('lag4', lag4_col)]:
            corrs = []
            for firm in df_filtered.index.get_level_values('FIRM').unique():
                try:
                    fd = df_filtered.xs(firm, level='FIRM')[[col, lag_col]].dropna()
                    if len(fd) >= 3:
                        corrs.append(fd[col].corr(fd[lag_col]))
                except (KeyError, ValueError):
                    continue
            ac[sig_name][lag_label] = {
                'mean': np.mean(corrs) if corrs else np.nan,
                'median': np.median(corrs) if corrs else np.nan,
                'pct_positive': np.mean(np.array(corrs) > 0) * 100 if corrs else np.nan,
            }
            print(f"  {sig_name} {lag_label}: mean={ac[sig_name][lag_label]['mean']:.3f}, "
                  f"pct_pos={ac[sig_name][lag_label]['pct_positive']:.1f}%")

    def _gac(sig, lag, key, ndp=3):
        v = ac.get(sig, {}).get(lag, {}).get(key, np.nan)
        return round(float(v), ndp) if not (isinstance(v, float) and np.isnan(v)) else np.nan

    cols = ['Raw RRR', 'Adj RRR', 'Raw AR', 'Adj AR']
    sr_ov   = [sr_raw_rrr_ov, sr_adj_rrr_ov, sr_raw_ar_ov, sr_adj_ar_ov]
    sr_tabs = [sr_raw_rrr,    sr_adj_rrr,    sr_raw_ar,    sr_adj_ar]

    comp_rows = [
        {'Metric': 'Cross-sect SD (%/quarter)', **{c: round(cs_sd[c], 2) for c in cols}},
        {'Metric': 'Lag-1 corr mean',   **{c: _gac(c,'lag1','mean')           for c in cols}},
        {'Metric': 'Lag-1 corr median', **{c: _gac(c,'lag1','median')         for c in cols}},
        {'Metric': 'Lag-1 % positive',  **{c: _gac(c,'lag1','pct_positive',1) for c in cols}},
        {'Metric': 'Lag-4 corr mean',   **{c: _gac(c,'lag4','mean')           for c in cols}},
        {'Metric': 'Lag-4 corr median', **{c: _gac(c,'lag4','median')         for c in cols}},
        {'Metric': 'Lag-4 % positive',  **{c: _gac(c,'lag4','pct_positive',1) for c in cols}},
        {'Metric': 'Overall stay-rate (%)', **{c: round(v*100,1) for c,v in zip(cols,sr_ov)}},
        {'Metric': 'Q1 stay-rate (%)', **{c: round(sr_tabs[i]['Q1']*100,1) for i,c in enumerate(cols)}},
        {'Metric': 'Q2 stay-rate (%)', **{c: round(sr_tabs[i]['Q2']*100,1) for i,c in enumerate(cols)}},
        {'Metric': 'Q3 stay-rate (%)', **{c: round(sr_tabs[i]['Q3']*100,1) for i,c in enumerate(cols)}},
        {'Metric': 'Q4 stay-rate (%)', **{c: round(sr_tabs[i]['Q4']*100,1) for i,c in enumerate(cols)}},
    ]
    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))
    comp_df.to_excel(os.path.join(OUTPUT_DIR, 'signal_persistence_comparison.xlsx'), index=False)

    # --- 1.10 Adjusted signal dispersion check ---
    print("\n--- 1.10 Adjusted signal dispersion check ---")
    for sig_name, sig_col in [('Adj RRR', 'ADJ_RRR_PCT'), ('Adj AR', 'ADJ_ACQ_RATE_PCT')]:
        vals = pd.to_numeric(df_filtered[sig_col], errors='coerce').dropna()
        print(f"  {sig_name}: SD={vals.std():.2f}, IQR={vals.quantile(0.75)-vals.quantile(0.25):.2f}, "
              f"P10={vals.quantile(0.10):.2f}, P90={vals.quantile(0.90):.2f}")

    return df_long, df_filtered, industry_stats_df


# =============================================================================
# PHASE 1B: DOWNLOAD FAMA-FRENCH FACTORS
# =============================================================================

def phase1b_load_ff_factors():
    """Download and prepare FF3, FF5, and Momentum factors."""

    print("\n" + "=" * 80)
    print("PHASE 1B: FAMA-FRENCH FACTOR DATA")
    print("=" * 80)

    # First try local file
    local_ff3 = os.path.join(DATA_DIR, "2025-06-27-FF-Factors_Monthly_Python.csv")

    # Try to download full factor data
    print("  Attempting to download factor data from Ken French website...")
    factors = download_ff_factors()

    if 'FF3' in factors and 'FF5' in factors and 'MOM' in factors:
        # Merge all factors
        ff_all = factors['FF3'].copy()
        ff_all = ff_all.join(factors['FF5'][['RMW', 'CMA']], how='outer')
        ff_all = ff_all.join(factors['MOM'], how='outer')
        ff_all.index = ff_all.index.to_period('M').to_timestamp('M')
        print(f"  Combined factors: {len(ff_all)} months, cols={list(ff_all.columns)}")
        return ff_all

    # Fallback: use local FF3 file
    print("  Falling back to local FF3 file...")
    ff_factors = pd.read_csv(local_ff3, skiprows=3)
    ff_factors = ff_factors.rename(columns={'Unnamed: 0': 'Date'})
    ff_factors['Date'] = pd.to_datetime(ff_factors['Date'].astype(str), format='%Y%m')
    ff_factors = ff_factors.set_index('Date').sort_index()
    for col in ff_factors.columns:
        ff_factors[col] = pd.to_numeric(ff_factors[col], errors='coerce') / 100
    ff_factors.index = ff_factors.index.to_period('M').to_timestamp('M')
    print(f"  Local FF3: {len(ff_factors)} months")
    return ff_factors


# =============================================================================
# PHASE 1C: LOAD MONTHLY RETURNS
# =============================================================================

def phase1c_load_monthly_returns():
    """Load and process monthly return data."""

    print("\n" + "=" * 80)
    print("PHASE 1C: MONTHLY RETURNS")
    print("=" * 80)

    file_ret = os.path.join(DATA_DIR, "2025-07-02a-TK-Monthly-Returns_Python.xlsx")
    monthly_raw = pd.read_excel(file_ret, header=None)

    header_rows = monthly_raw.iloc[:2]
    data_rows = monthly_raw.iloc[2:].copy()
    combined_headers = header_rows.apply(lambda x: x.astype(str).str.strip(), axis=0)
    col_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
    data_rows.columns = col_headers
    data_rows = data_rows.reset_index(drop=True)
    data_rows.rename(columns={data_rows.columns[0]: 'Date'}, inplace=True)
    data_rows.replace('#N/A N/A', np.nan, inplace=True)
    data_rows['Date'] = pd.to_datetime(data_rows['Date'], errors='coerce', dayfirst=True)

    df_prices = data_rows.copy()
    for c in df_prices.columns:
        if c != 'Date':
            df_prices[c] = pd.to_numeric(df_prices[c], errors='coerce')

    # Monthly log returns, simple returns, AND the price level.
    # The price level is retained so that market cap can later be measured as of
    # the formation date (quarter-end market cap scaled by the firm's own price
    # change), rather than using the stale quarter-end market cap.
    px_only = df_prices.drop(columns=['Date'])
    log_ret = np.log(px_only / px_only.shift(1))

    px_long = px_only.copy()
    px_long['Date'] = df_prices['Date'].values
    px_long = px_long.melt(id_vars='Date', var_name='FirmVar', value_name='PX')

    log_ret['Date'] = df_prices['Date'].values
    ret_long = log_ret.melt(id_vars='Date', var_name='FirmVar', value_name='RETURN_LOG')

    returns = ret_long.merge(px_long, on=['Date', 'FirmVar'], how='left')
    returns['FIRM'] = returns['FirmVar'].str.replace('.PX_LAST', '', regex=False).str.upper()
    returns.drop(columns=['FirmVar'], inplace=True)
    returns = returns.dropna(subset=['RETURN_LOG'])
    # Simple (arithmetic) monthly return; portfolios are value-weighted from these,
    # NOT from log returns (a value-weighted mean of log returns is biased).
    returns['RET_SIMPLE'] = np.exp(returns['RETURN_LOG']) - 1

    # NOTE: no quarter lag is baked in here. All signal->return timing (the
    # 2-month formation gap and 3-month hold) is applied once, in
    # build_holding_panel(), to avoid stacking a second implicit lag.
    print(f"  Monthly returns: {len(returns)} obs, {returns['FIRM'].nunique()} firms")
    print(f"  Date range: {returns['Date'].min()} to {returns['Date'].max()}")

    return returns


# =============================================================================
# PHASE 2: EMPIRICAL ANALYSIS
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
                 'HISTORICAL_MARKET_CAP', 'PX_LAST']
    alt_cols = ['ADJ_SRR_PCT', 'SRR_PCT', 'ADJ_LR_RRR_PCT', 'LR_RRR_PCT',
                'ADJ_CUMRR_PCT', 'CUMRR_PCT']
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


def run_portfolio_analysis(panel, ff_factors):
    """Univariate RRR quartile sorts (Q1 = highest RRR) with factor regressions.

    Consumes the holding panel from build_holding_panel(), so the signal->return
    timing (form FORM_LAG_MONTHS months after quarter-end, hold HOLD_MONTHS) is
    already correct. Sorts use the CONTEMPORANEOUS quarter-end signal (RRR_PCT /
    ADJ_RRR_PCT), never a re-lagged column, so no second lag is stacked.
    """

    results = {}
    ret = panel.copy()
    for col in ['RRR_PCT', 'ADJ_RRR_PCT', 'MCAP_FORM']:
        ret[col] = pd.to_numeric(ret[col], errors='coerce')

    # ---- Pipeline invariants, asserted at the point portfolios are formed ----
    n_firms = ret['FIRM'].nunique()
    print(f"\n  [INVARIANT] firms entering portfolio formation: {n_firms} "
          f"(expected {EXPECTED_N_FIRMS})")
    assert n_firms == EXPECTED_N_FIRMS, (
        f"Sample size changed: {n_firms} firms at formation, expected {EXPECTED_N_FIRMS}. "
        f"Investigate before trusting portfolio results.")
    echo_formation_examples(ret)

    # Market portfolio: VW (formation market cap) SIMPLE return of all sample firms.
    market_ret = value_weighted_market(ret)

    # =========================================================================
    # SECTION 1: UNIVARIATE QUARTILE SORTS (Q1 = highest value)
    # =========================================================================
    print("\n  *** SECTION 1: UNIVARIATE QUARTILE SORTS (Q1=best) ***")

    for signal_label, rrr_col in [
        ('RAW', 'RRR_PCT'),
        ('ADJ', 'ADJ_RRR_PCT'),
    ]:
        print(f"\n  === {signal_label} signal — quartile sort ===")

        # Univariate RRR quartile sort on the CONTEMPORANEOUS quarter-end signal
        print(f"\n  Univariate RRR quartile ({signal_label}):")
        ret[f'RRR_Q_{signal_label}'] = ret.groupby('QUARTER')[rrr_col].transform(safe_quartile)
        port_rrr_q = build_portfolio_returns(ret, f'RRR_Q_{signal_label}', market_ret)
        if port_rrr_q is not None:
            results[f'port_rrr_{signal_label.lower()}_q4_vw'] = port_rrr_q
            run_factor_regressions(port_rrr_q, ff_factors, f'RRR {signal_label} Q4 VW')

    results['market_ret'] = market_ret
    return results


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
    print(f"    Portfolio monthly SIMPLE returns (annualized, mean x 12):")
    for col in port.columns:
        ann_ret = port[col].mean() * 12
        ann_vol = port[col].std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        print(f"      {col}: ret={ann_ret*100:.2f}%, vol={ann_vol*100:.2f}%, SR={sharpe:.2f}")

    # Firms per bucket per quarter
    counts = df.groupby(['QUARTER', bucket_col])['FIRM'].nunique().unstack(bucket_col)
    print(f"    Avg firms per bucket: {counts.mean().round(1).to_dict()}")

    return port


def run_factor_regressions(port_rets, ff_factors, label):
    """Run FF3, FF3+Mom, FF5 regressions on portfolio returns."""

    print(f"\n    Factor regressions: {label}")

    # Align dates
    port = port_rets.copy()
    port.index = pd.to_datetime(port.index).to_period('M').to_timestamp('M')
    combined = port.join(ff_factors, how='inner')

    if combined.empty or len(combined) < 12:
        print("    Insufficient overlapping data for factor regressions")
        return {}

    results = {}

    # Columns to regress (portfolios, not factors)
    factor_cols_all = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom', 'RF']
    port_cols = [c for c in port.columns if c not in factor_cols_all and c != 'MKT']

    for p in port_cols:
        if p not in combined.columns:
            continue

        # Excess returns (long-short portfolios are already zero-cost; no RF subtraction)
        if 'RF' in combined.columns:
            if 'T1-T3' in p or 'Q1-Q4' in p or 'High-Low' in p:
                y = combined[p]  # Long-short is already zero-cost
            else:
                y = combined[p] - combined['RF']
        else:
            y = combined[p]

        y = y.dropna()
        if len(y) < 12:
            continue

        # Model specs
        specs = {}
        if all(c in combined.columns for c in ['Mkt-RF', 'SMB', 'HML']):
            specs['FF3'] = ['Mkt-RF', 'SMB', 'HML']
        if all(c in combined.columns for c in ['Mkt-RF', 'SMB', 'HML', 'Mom']):
            specs['FF3+Mom'] = ['Mkt-RF', 'SMB', 'HML', 'Mom']
        if all(c in combined.columns for c in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']):
            specs['FF5'] = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        for spec_name, factors in specs.items():
            X = combined[factors].reindex(y.index).dropna()
            y_aligned = y.reindex(X.index).dropna()
            X = X.reindex(y_aligned.index)

            if len(y_aligned) < 12:
                continue

            model = newey_west_ols(y_aligned, X)

            alpha = model.params['const']
            alpha_t = model.tvalues['const']
            alpha_p = model.pvalues['const']
            alpha_se = model.bse['const']

            key = f"{p}_{spec_name}"
            results[key] = {
                'alpha': alpha,
                'alpha_t': alpha_t,
                'alpha_p': alpha_p,
                'se_alpha': alpha_se,
                'r2': model.rsquared,
                'n_obs': model.nobs,
            }
            # Store beta, SE, t-stat, and p-value for each factor
            for f in factors:
                results[key][f'beta_{f}'] = model.params[f]
                results[key][f'se_{f}'] = model.bse[f]
                results[key][f'tstat_{f}'] = model.tvalues[f]
                results[key][f'pval_{f}'] = model.pvalues[f]

    # Print summary for long-short or all portfolios
    print(f"    {'Portfolio':<15} {'Model':<10} {'Alpha':>8} {'t-stat':>8} {'R2':>6} {'N':>5}")
    for key, vals in sorted(results.items()):
        parts = key.rsplit('_', 1)
        if len(parts) == 2:
            port_name, model_name = parts
        else:
            port_name, model_name = key, ''
        print(f"    {port_name:<15} {model_name:<10} {vals['alpha']*100:>8.3f}% {vals['alpha_t']:>8.2f} "
              f"{vals['r2']:>6.3f} {vals['n_obs']:>5.0f}")

    return results


def run_capm_vs_sp500(market_ret, returns, ff_factors):
    """CAPM OLS of the sample's value-weighted market portfolio on the S&P 500.

        (market_excess)_t = alpha + beta * (SP500_excess)_t + e_t

    Monthly SIMPLE returns; risk-free = Ken French RF. Reports alpha (monthly and
    annualized), beta, R^2, and Newey-West (HAC) t-stats. This is the real OLS
    version of the correlation / tracking-error check that previously lived in
    diagnose_portfolio_overlap.py (CHECK 7), moved here so it runs with everything
    else.
    """
    if market_ret is None or len(market_ret) == 0:
        print("  No sample market return available; skipping CAPM.")
        return {}

    spx = returns[returns['FIRM'] == 'SPX INDEX'][['Date', 'RET_SIMPLE']].copy()
    if spx.empty:
        print("  SPX INDEX not found in returns; skipping CAPM.")
        return {}
    spx = spx.rename(columns={'RET_SIMPLE': 'SPX'}).set_index('Date').sort_index()

    df = market_ret.to_frame('MKT').join(spx, how='inner').dropna()
    df.index = pd.to_datetime(df.index).to_period('M').to_timestamp('M')
    rf = ff_factors['RF'].reindex(df.index)
    df['MKT_EX'] = df['MKT'] - rf
    df['SPX_EX'] = df['SPX'] - rf
    df = df.dropna(subset=['MKT_EX', 'SPX_EX'])
    if len(df) < 12:
        print(f"  Only {len(df)} overlapping months; skipping CAPM.")
        return {}

    model = sm.OLS(df['MKT_EX'], sm.add_constant(df['SPX_EX'])).fit(
        cov_type='HAC', cov_kwds={'maxlags': 4})
    out = {
        'alpha_month': model.params['const'],
        'alpha_ann': model.params['const'] * 12,
        'alpha_se': model.bse['const'],
        'alpha_t': model.tvalues['const'],
        'alpha_p': model.pvalues['const'],
        'beta': model.params['SPX_EX'],
        'beta_se': model.bse['SPX_EX'],
        'beta_t': model.tvalues['SPX_EX'],
        'r2': model.rsquared,
        'n_obs': int(model.nobs),
    }
    print(f"  CAPM (sample VW market vs S&P 500), N={out['n_obs']} months:")
    print(f"    alpha = {out['alpha_month']*100:.4f}%/mo ({out['alpha_ann']*100:.2f}%/yr), "
          f"SE={out['alpha_se']*100:.4f}, t={out['alpha_t']:.2f}, p=[{out['alpha_p']:.3f}]")
    print(f"    beta  = {out['beta']:.4f}  (SE={out['beta_se']:.4f}, t={out['beta_t']:.2f})")
    print(f"    R^2   = {out['r2']:.4f}")
    pd.DataFrame([out]).to_excel(os.path.join(OUTPUT_DIR, 'capm_vs_sp500.xlsx'), index=False)
    return out


def run_fama_macbeth(panel, ff_factors):
    """Quarterly Fama-MacBeth on the SAME holding window as the portfolio sorts.

    For each (firm, signal quarter Q), the dependent variable is the compounded
    HOLD_MONTHS-month holding-period EXCESS return earned over exactly the months
    the sort holds (formed FORM_LAG_MONTHS months after Q). Each quarter, a
    cross-sectional OLS regresses that holding return on the CONTEMPORANEOUS
    quarter-end signal(s) and controls; the quarterly slopes are then averaged with
    Newey-West t-stats. Timing is therefore IDENTICAL to the sorts (2-month
    formation gap, 3-month hold): no additional lag is applied, and the quarterly
    structure of the original FM table is preserved. No significance stars are
    printed; exact p-values are reported in brackets.

    NOTE: this timing correction materially changes the FM result. Under the old
    0-month-gap design (signal at Q -> next calendar quarter, overlapping the
    earnings-announcement window) Adj RRR had t ~ 2.9-3.1; under the correct
    2-month publication gap the cross-sectional linear effect is t ~ 1.8-1.9
    (a monthly FM on the same panel is weaker still, t ~ 0.9-1.1). The
    value-weighted, tail-driven portfolio long-short remains significant; the
    equal-weighted linear FM does not clear t = 2.
    """

    results = {}
    df = panel.copy()

    # Monthly risk-free (Ken French, simple) aligned to each holding month.
    rf = ff_factors['RF'].copy()
    rf.index = pd.to_datetime(rf.index).to_period('M')
    df['MONTH_P'] = pd.to_datetime(df['Date']).dt.to_period('M')
    df['RF_M'] = df['MONTH_P'].map(rf)
    df['RET_SIMPLE'] = pd.to_numeric(df['RET_SIMPLE'], errors='coerce')
    for c in ['ADJ_RRR_PCT', 'RRR_PCT', 'SIZE', 'BTM', 'PM_OPER_PCT']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compound each firm's HOLD_MONTHS-month holding window into ONE excess return.
    # Controls are the quarter-end (contemporaneous) values, known by formation.
    hw = (df.sort_values(['FIRM', 'QUARTER', 'HOLD_IDX'])
            .groupby(['FIRM', 'QUARTER'])
            .agg(n=('RET_SIMPLE', 'size'),
                 gross=('RET_SIMPLE', lambda s: (1 + s).prod()),
                 gross_rf=('RF_M', lambda s: (1 + s).prod()),
                 ADJ_RRR_PCT=('ADJ_RRR_PCT', 'first'),
                 RRR_PCT=('RRR_PCT', 'first'),
                 SIZE=('SIZE', 'first'),
                 BTM=('BTM', 'first'),
                 PM_OPER_PCT=('PM_OPER_PCT', 'first'))
            .reset_index())
    # Require the full holding window so the return horizon is well defined.
    hw = hw[hw['n'] == HOLD_MONTHS].copy()
    hw['EXCESS_RET_HW'] = hw['gross'] - hw['gross_rf']

    specs = {
        '(1) Adj RRR only':       ['ADJ_RRR_PCT'],
        '(2) Adj RRR + Controls': ['ADJ_RRR_PCT', 'SIZE', 'BTM', 'PM_OPER_PCT'],
        '(3) Raw RRR only':       ['RRR_PCT'],
        '(4) Raw RRR + Controls': ['RRR_PCT', 'SIZE', 'BTM', 'PM_OPER_PCT'],
    }
    y_var = 'EXCESS_RET_HW'

    for spec_name, x_vars in specs.items():
        # Cross-sectional regression each signal quarter
        period_coefs = []
        for qtr, group in hw.groupby('QUARTER'):
            sub = group[[y_var] + x_vars].dropna()
            if len(sub) < 10:
                continue
            try:
                model = sm.OLS(sub[y_var], sm.add_constant(sub[x_vars])).fit()
                coefs = model.params.to_dict()
                coefs['QUARTER'] = qtr
                coefs['N'] = len(sub)
                period_coefs.append(coefs)
            except Exception:
                continue

        if not period_coefs:
            print(f"  {spec_name}: No valid periods")
            continue

        coef_df = pd.DataFrame(period_coefs)

        # Time-series average with Newey-West t-stats (Bartlett kernel)
        T = len(coef_df)
        avg_coefs = coef_df.drop(columns=['QUARTER', 'N']).mean()
        max_lag = max(1, int(np.floor(4 * (T/100)**(2/9))))
        se_nw = {}
        for var in avg_coefs.index:
            series = coef_df[var] - avg_coefs[var]
            gamma_sum = (series**2).mean()
            for j in range(1, max_lag + 1):
                gamma_j = (series.iloc[j:].values * series.iloc[:-j].values).mean()
                gamma_sum += 2 * (1 - j/(max_lag+1)) * gamma_j
            se_nw[var] = np.sqrt(gamma_sum / T)

        t_stats = {v: (avg_coefs[v] / se_nw[v] if se_nw[v] > 0 else np.nan)
                   for v in avg_coefs.index}
        p_values = {v: float(2 * sps.t.sf(abs(t_stats[v]), max(T - 1, 1)))
                    if np.isfinite(t_stats[v]) else np.nan for v in avg_coefs.index}

        print(f"\n  {spec_name} (T={T} quarters, avg N={coef_df['N'].mean():.0f}):")
        for var in [v for v in avg_coefs.index if v != 'const']:
            print(f"    {var:<25} coef={avg_coefs[var]:>10.4f}  "
                  f"t={t_stats[var]:>7.2f}  p=[{p_values[var]:.3f}]")

        results[spec_name] = {
            'avg_coefs': avg_coefs.to_dict(),
            't_stats': t_stats,
            'p_values': p_values,
            'T': T,
            'avg_N': coef_df['N'].mean(),
        }

    # Export
    fm_summary = []
    for spec_name, res in results.items():
        for var in res['avg_coefs']:
            if var == 'const':
                continue
            fm_summary.append({
                'Spec': spec_name,
                'Variable': var,
                'Coefficient': res['avg_coefs'][var],
                't-stat': res['t_stats'][var],
                'p-value': res['p_values'][var],
                'T': res['T'],
                'Avg N': res['avg_N'],
            })

    fm_df = pd.DataFrame(fm_summary)
    fm_df.to_excel(os.path.join(OUTPUT_DIR, 'fama_macbeth.xlsx'), index=False)
    print(f"\n  Fama-MacBeth results exported to {OUTPUT_DIR}/fama_macbeth.xlsx")

    return results


def run_risk_analysis(panel, ff_factors):
    """Analyze risk characteristics by adj-RRR tercile (T1 = highest RRR).

    Uses the holding panel (same timing as the sorts). Portfolio returns are
    value-weighted SIMPLE returns with formation-date market cap; the wealth
    index is built by compounding simple returns, not exponentiating summed logs.
    """

    print("\n  Risk analysis by RRR terciles:")

    ret = panel.copy()
    ret['ADJ_RRR_PCT'] = pd.to_numeric(ret['ADJ_RRR_PCT'], errors='coerce')
    ret['RRR_T'] = ret.groupby('QUARTER')['ADJ_RRR_PCT'].transform(safe_tercile)

    results = {}

    # Build VW (formation market cap) SIMPLE-return portfolios by RRR tercile
    df = ret.dropna(subset=['RRR_T']).copy()
    df['MCAP_FORM'] = pd.to_numeric(df['MCAP_FORM'], errors='coerce')
    df['RET_SIMPLE'] = pd.to_numeric(df['RET_SIMPLE'], errors='coerce')
    df = df.dropna(subset=['MCAP_FORM', 'RET_SIMPLE'])
    df['w'] = df['MCAP_FORM'] / df.groupby(['Date', 'RRR_T'])['MCAP_FORM'].transform('sum')
    df['w_ret'] = df['w'] * df['RET_SIMPLE']

    port = (
        df.groupby(['Date', 'RRR_T'])['w_ret']
        .sum()
        .unstack('RRR_T')
        .sort_index()
    )

    if port.empty:
        print("  No valid portfolio data for risk analysis")
        return results

    # Risk metrics (port columns are SIMPLE monthly returns)
    risk_metrics = []
    for col in port.columns:
        rets = port[col].dropna()
        cum = (1 + rets).cumprod()          # wealth index from simple returns
        cum_max = cum.cummax()
        drawdown = (cum / cum_max - 1).min()

        downside = rets[rets < 0]
        downside_dev = downside.std(ddof=0) if len(downside) > 0 else np.nan

        # Downside beta (beta in down markets)
        port_aligned = port.copy()
        port_aligned.index = pd.to_datetime(port_aligned.index).to_period('M').to_timestamp('M')
        ff_aligned = ff_factors.reindex(port_aligned.index)
        mkt = ff_aligned.get('Mkt-RF', pd.Series(dtype=float))

        down_months = mkt[mkt < 0].index
        if len(down_months) > 5 and col in port_aligned.columns:
            y_down = port_aligned.loc[down_months, col].dropna()
            x_down = mkt.loc[down_months].reindex(y_down.index).dropna()
            y_down = y_down.reindex(x_down.index)
            if len(y_down) > 5:
                X_down = sm.add_constant(x_down)
                model_down = sm.OLS(y_down, X_down).fit()
                downside_beta = model_down.params.iloc[1]
            else:
                downside_beta = np.nan
        else:
            downside_beta = np.nan

        risk_metrics.append({
            'Portfolio': col,
            'Ann Return (%)': rets.mean() * 12 * 100,
            'Ann Vol (%)': rets.std() * np.sqrt(12) * 100,
            'Sharpe': (rets.mean() * 12) / (rets.std() * np.sqrt(12)) if rets.std() > 0 else np.nan,
            'Max Drawdown (%)': drawdown * 100,
            'Downside Dev': downside_dev,
            'Sortino': rets.mean() / downside_dev if downside_dev > 0 else np.nan,
            'Downside Beta': downside_beta,
            'VaR 5% (%)': np.percentile(rets, 5) * 100,
            'Skewness': skew(rets, nan_policy='omit'),
            'Kurtosis': kurtosis(rets, nan_policy='omit', fisher=False),
            'Hit Ratio (%)': (rets > 0).mean() * 100
        })

    risk_df = pd.DataFrame(risk_metrics)
    print(risk_df.to_string(index=False))
    risk_df.to_excel(os.path.join(OUTPUT_DIR, 'risk_analysis.xlsx'), index=False)

    results['risk_metrics'] = risk_df
    return results


# =============================================================================
# PHASE 4: ADDITIONAL TESTS
# =============================================================================

def phase4_additional_tests(df_filtered, panel, returns, ff_factors):
    """Same-growth placebo and robustness tests, all on the holding panel."""

    print("\n" + "=" * 80)
    print("PHASE 4: ADDITIONAL TESTS")
    print("=" * 80)

    results = {}

    # --- 4.1 Same-growth placebo (median split) ---
    print("\n--- 4.1 Same-growth placebo (median split) ---")
    results['placebo'] = run_placebo_median_split(panel, ff_factors)

    # --- 4.2 Robustness: No-COVID (quartile) ---
    print("\n--- 4.2 Robustness: Excluding COVID (quartile) ---")
    ret_nc = panel[
        ~((panel['Date'] >= '2020-01-01') & (panel['Date'] <= '2021-06-30'))
    ].copy()

    if len(ret_nc) > 0:
        market_ret_nc = value_weighted_market(ret_nc)
        ret_nc['ADJ_RRR_PCT'] = pd.to_numeric(ret_nc['ADJ_RRR_PCT'], errors='coerce')
        ret_nc['RRR_Q_NC'] = ret_nc.groupby('QUARTER')['ADJ_RRR_PCT'].transform(safe_quartile)

        port_nc_q = build_portfolio_returns(ret_nc, 'RRR_Q_NC', market_ret_nc)
        if port_nc_q is not None:
            run_factor_regressions(port_nc_q, ff_factors, 'RRR ADJ VW NoCOVID Quartile')
            results['no_covid_quartile'] = port_nc_q

    # --- 4.3 Robustness: Equal-weighted (quartile) ---
    print("\n--- 4.3 Robustness: Equal-weighted portfolios (quartile) ---")
    ret_ew_q = panel.copy()
    ret_ew_q['ADJ_RRR_PCT'] = pd.to_numeric(ret_ew_q['ADJ_RRR_PCT'], errors='coerce')
    ret_ew_q['RRR_Q_EW'] = ret_ew_q.groupby('QUARTER')['ADJ_RRR_PCT'].transform(safe_quartile)

    df_ew_q = ret_ew_q.dropna(subset=['RRR_Q_EW']).copy()
    # Equal-weight SIMPLE returns (not log returns)
    port_ew_q = (
        df_ew_q.groupby(['Date', 'RRR_Q_EW'])['RET_SIMPLE']
        .mean()
        .unstack('RRR_Q_EW')
        .sort_index()
    )
    if 'Q1' in port_ew_q.columns and 'Q4' in port_ew_q.columns:
        port_ew_q['Q1-Q4'] = port_ew_q['Q1'] - port_ew_q['Q4']

    # EW market (quartile version)
    ew_mkt_q = df_ew_q.groupby('Date')['RET_SIMPLE'].mean().sort_index()
    port_ew_q['MKT'] = ew_mkt_q.reindex(port_ew_q.index)

    print(f"  EW quartile portfolio SIMPLE returns (annualized):")
    for col in port_ew_q.columns:
        ann = port_ew_q[col].mean() * 12 * 100
        print(f"    {col}: {ann:.2f}%")

    run_factor_regressions(port_ew_q, ff_factors, 'RRR ADJ EW Quartile')
    results['ew_quartile'] = port_ew_q

    # --- 4.4 Robustness: one-extra-quarter formation gap ---
    # Form 3 months later than the headline spec (FORM_LAG_MONTHS + 3), same
    # 3-month hold. Tests whether the RRR effect survives an even longer gap.
    print("\n--- 4.4 Robustness: one-extra-quarter formation gap ---")
    results['extra_lag'] = run_extra_lag_portfolios(df_filtered, returns, ff_factors)

    return results


def run_extra_lag_portfolios(df_filtered, returns, ff_factors):
    """Robustness: form the portfolio ONE EXTRA QUARTER later than the headline.

    Uses build_holding_panel with a formation gap of FORM_LAG_MONTHS + 3 months
    (still a 3-month hold), so the signal at quarter-end Q predicts returns roughly
    two quarters out instead of one. This tests whether the RRR effect survives an
    even longer gap and is not an artifact of the immediate post-signal quarter.
    Same value-weighting (formation market cap) and simple-return aggregation as
    the headline.
    """

    print("\n  === Extra-quarter formation gap (form 5 months after quarter-end) ===")

    panel2 = build_holding_panel(df_filtered, returns,
                                 form_lag=FORM_LAG_MONTHS + 3, hold=HOLD_MONTHS)

    if panel2.empty or len(panel2) < 100:
        print("  Insufficient data for extra-lag portfolios")
        return {}

    market_ret2 = value_weighted_market(panel2)

    panel2['ADJ_RRR_PCT'] = pd.to_numeric(panel2['ADJ_RRR_PCT'], errors='coerce')
    panel2['RRR_Q_XL'] = panel2.groupby('QUARTER')['ADJ_RRR_PCT'].transform(safe_quartile)

    port_xl = build_portfolio_returns(panel2, 'RRR_Q_XL', market_ret2)
    if port_xl is not None:
        reg_results = run_factor_regressions(port_xl, ff_factors, 'RRR ADJ VW ExtraQuarterLag')
        if reg_results:
            reg_df = pd.DataFrame(reg_results).T.reset_index()
            reg_df.rename(columns={'index': 'Unnamed: 0'}, inplace=True)
            reg_df.to_excel(os.path.join(OUTPUT_DIR, 'factor_reg_Lag2_adj.xlsx'), index=False)

    return {'port': port_xl}


def run_placebo_median_split(panel, ff_factors):
    """Same-growth placebo: top-growth-quartile firms split at median of adj RRR.

    Restrict to firms in the top revenue-growth quartile (~31 firms per quarter),
    then split at the median of adj RRR within that group, yielding a High-RRR
    and Low-RRR bucket of ~15-16 firms each. Tests whether RRR predicts returns
    beyond revenue growth level (composition, not level).
    """

    ret = panel.copy()
    ret['REV_GROWTH_PCT'] = pd.to_numeric(ret['REV_GROWTH_PCT'], errors='coerce')
    ret['ADJ_RRR_PCT'] = pd.to_numeric(ret['ADJ_RRR_PCT'], errors='coerce')

    # Identify top revenue-growth quartile per DATE (Q1 = highest growth)
    def assign_growth_quartile(x):
        x = x.dropna()
        if len(x) < 8:
            return pd.Series([np.nan] * len(x), index=x.index)
        try:
            return pd.qcut(x, 4, labels=['Q4', 'Q3', 'Q2', 'Q1'])
        except ValueError:
            return pd.Series([np.nan] * len(x), index=x.index)

    ret['GROWTH_Q'] = ret.groupby('QUARTER')['REV_GROWTH_PCT'].transform(assign_growth_quartile)
    high_growth = ret[ret['GROWTH_Q'] == 'Q1'].copy()

    if len(high_growth) < 50:
        print("  Insufficient data for placebo test")
        return {}

    # Median split of adj RRR within the top-growth group per DATE
    def median_split(x):
        x = x.dropna()
        if len(x) < 4:
            return pd.Series([np.nan] * len(x), index=x.index)
        med = x.median()
        return pd.Series(
            ['High' if v >= med else 'Low' for v in x],
            index=x.index
        )

    high_growth['RRR_MED'] = high_growth.groupby('QUARTER')['ADJ_RRR_PCT'].transform(median_split)

    df_pl = high_growth.dropna(subset=['RRR_MED']).copy()
    df_pl['MCAP_FORM'] = pd.to_numeric(df_pl['MCAP_FORM'], errors='coerce')
    df_pl['RET_SIMPLE'] = pd.to_numeric(df_pl['RET_SIMPLE'], errors='coerce')
    df_pl = df_pl.dropna(subset=['MCAP_FORM', 'RET_SIMPLE'])
    df_pl['w'] = df_pl['MCAP_FORM'] / df_pl.groupby(['Date', 'RRR_MED'])['MCAP_FORM'].transform('sum')
    df_pl['w_ret'] = df_pl['w'] * df_pl['RET_SIMPLE']  # value-weight SIMPLE returns

    port_pl = (
        df_pl.groupby(['Date', 'RRR_MED'])['w_ret']
        .sum()
        .unstack('RRR_MED')
        .sort_index()
    )

    if 'High' in port_pl.columns and 'Low' in port_pl.columns:
        port_pl['High-Low'] = port_pl['High'] - port_pl['Low']

    print(f"  Placebo (top-growth-quartile firms, median split on adj RRR):")
    for col in port_pl.columns:
        ann = port_pl[col].mean() * 12 * 100
        print(f"    {col}: {ann:.2f}% annualized")

    # Avg firms per group per quarter
    counts = df_pl.groupby(['QUARTER', 'RRR_MED'])['FIRM'].nunique().unstack('RRR_MED')
    print(f"    Avg firms/group: {counts.mean().to_dict()}")

    if not port_pl.empty:
        reg_results = run_factor_regressions(port_pl, ff_factors, 'Placebo HighGrowth Median')
        if reg_results:
            reg_df = pd.DataFrame(reg_results).T.reset_index()
            reg_df.rename(columns={'index': 'Unnamed: 0'}, inplace=True)
            reg_df.to_excel(os.path.join(OUTPUT_DIR, 'factor_reg_Placebo_HighGrowth_Median.xlsx'), index=False)

    return {'port': port_pl}


# =============================================================================
# ALTERNATIVE METRIC ANALYSIS
# =============================================================================

def run_alt_metric_analysis(df_filtered, returns, ff_factors):
    """Comparative analysis of alternative RRR variants (LR_RRR, SRR, CUMRR).

    Produces 5 PDFs in output/alt_metric/:
      summary_stats.pdf, time_trend.pdf, signal_persistence.pdf,
      portfolio_alphas_ff3.pdf, cumret_longshort.pdf
    """

    print("\n" + "=" * 80)
    print("PHASE 1D: ALTERNATIVE RRR VARIANT ANALYSIS")
    print("=" * 80)

    alt_dir = os.path.join(OUTPUT_DIR, 'alt_metric')
    os.makedirs(alt_dir, exist_ok=True)

    req = ['LR_RRR_PCT', 'ADJ_LR_RRR_PCT', 'SRR_PCT', 'ADJ_SRR_PCT',
           'CUMRR_PCT', 'ADJ_CUMRR_PCT',
           'ADJ_LR_RRR_PCT_LAG1', 'ADJ_SRR_PCT_LAG1', 'ADJ_CUMRR_PCT_LAG1']
    missing = [c for c in req if c not in df_filtered.columns]
    if missing:
        print(f"  Missing alt-metric columns: {missing}. Skipping.")
        return

    dates_idx = df_filtered.index.get_level_values('DATE')

    def _f(v, ndp=3):
        try:
            fv = float(v)
            return 'n/a' if np.isnan(fv) else f"{fv:.{ndp}f}"
        except (TypeError, ValueError):
            return 'n/a'

    def _stay_rates_alt(df, q_col):
        pdf = df[[q_col]].copy().reset_index().sort_values(['FIRM', 'DATE'])
        pdf['Q_PREV'] = pdf.groupby('FIRM')[q_col].shift(1)
        valid = pdf.dropna(subset=[q_col, 'Q_PREV'])
        overall = (valid[q_col] == valid['Q_PREV']).mean()
        per_q = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            sub = valid[valid['Q_PREV'] == q]
            per_q[q] = (sub[q_col] == q).mean() if len(sub) > 0 else np.nan
        return overall, per_q

    # =========================================================================
    # A. Summary Statistics
    # =========================================================================
    print("\n--- 1D.A Summary statistics ---")

    alt_vars = [
        ('Adj RRR', 'ADJ_RRR_PCT'),
        ('Adj AR', 'ADJ_ACQ_RATE_PCT'),
        ('LR_RRR', 'LR_RRR_PCT'),
        ('Adj LR_RRR', 'ADJ_LR_RRR_PCT'),
        ('SRR', 'SRR_PCT'),
        ('Adj SRR', 'ADJ_SRR_PCT'),
        ('CUMRR', 'CUMRR_PCT'),
        ('Adj CUMRR', 'ADJ_CUMRR_PCT'),
    ]

    stats_rows = []
    for label, col in alt_vars:
        if col in df_filtered.columns:
            vals = pd.to_numeric(df_filtered[col], errors='coerce').dropna()
            stats_rows.append([
                label,
                str(int(len(vals))),
                _f(vals.mean(), 2),
                _f(vals.median(), 2),
                _f(vals.std(), 2),
                _f(vals.quantile(0.10), 2),
                _f(vals.quantile(0.25), 2),
                _f(vals.quantile(0.75), 2),
                _f(vals.quantile(0.90), 2),
            ])

    col_hdrs_s = ['Variable', 'N', 'Mean', 'Median', 'SD', 'P10', 'P25', 'P75', 'P90']
    fig, ax = plt.subplots(figsize=(14, max(3, len(stats_rows) * 0.5 + 1.5)))
    ax.axis('off')
    tbl = ax.table(cellText=stats_rows, colLabels=col_hdrs_s, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax.set_title('Summary Statistics: Alternative RRR Variants', fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(alt_dir, 'summary_stats.pdf'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: summary_stats.pdf")

    # =========================================================================
    # B. Time Trend
    # =========================================================================
    print("\n--- 1D.B Time trend ---")

    sorted_dates = sorted(df_filtered.index.get_level_values('DATE').unique())
    date_pos = {d: i for i, d in enumerate(sorted_dates)}

    trend_vars = [
        ('Adj RRR', 'ADJ_RRR_PCT'),
        ('LR_RRR', 'LR_RRR_PCT'),
        ('SRR', 'SRR_PCT'),
        ('CUMRR', 'CUMRR_PCT'),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, col in trend_vars:
        if col in df_filtered.columns:
            series = (pd.to_numeric(df_filtered[col], errors='coerce')
                      .groupby(dates_idx).median())
            xs = [date_pos.get(d, i) for i, d in enumerate(series.index)]
            ax.plot(xs, series.values, label=label, linewidth=1.5)

    tick_step = max(1, len(sorted_dates) // 8)
    tick_pos = list(range(0, len(sorted_dates), tick_step))
    tick_lbl = [str(sorted_dates[i])[:10] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Cross-sectional Median (%)')
    ax.set_title('Quarterly Median of Alternative RRR Metrics')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(alt_dir, 'time_trend.pdf'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: time_trend.pdf")

    # =========================================================================
    # C. Signal Persistence
    # =========================================================================
    print("\n--- 1D.C Signal persistence ---")

    df_p = df_filtered.copy()
    dates_p = df_p.index.get_level_values('DATE')

    df_p['ADJ_LR_RRR_PCT_LAG4'] = df_p.groupby(level='FIRM')['ADJ_LR_RRR_PCT'].shift(4)
    df_p['ADJ_SRR_PCT_LAG4'] = df_p.groupby(level='FIRM')['ADJ_SRR_PCT'].shift(4)
    df_p['ADJ_CUMRR_PCT_LAG4'] = df_p.groupby(level='FIRM')['ADJ_CUMRR_PCT'].shift(4)

    sig_cols_p = [
        ('Adj RRR',    'ADJ_RRR_PCT',     'ADJ_RRR_PCT_LAG1',     'ADJ_RRR_PCT_LAG4'),
        ('Adj AR',     'ADJ_ACQ_RATE_PCT', 'ADJ_ACQ_RATE_PCT_LAG1', 'ADJ_ACQ_RATE_PCT_LAG4'),
        ('Adj LR_RRR', 'ADJ_LR_RRR_PCT',  'ADJ_LR_RRR_PCT_LAG1',  'ADJ_LR_RRR_PCT_LAG4'),
        ('Adj SRR',    'ADJ_SRR_PCT',      'ADJ_SRR_PCT_LAG1',      'ADJ_SRR_PCT_LAG4'),
        ('Adj CUMRR',  'ADJ_CUMRR_PCT',    'ADJ_CUMRR_PCT_LAG1',    'ADJ_CUMRR_PCT_LAG4'),
    ]

    for sig_name, col, _, _ in sig_cols_p:
        q_col = f'ALTQ_{sig_name.replace(" ", "_")}'
        df_p[q_col] = df_p.groupby(dates_p)[col].transform(safe_quartile)

    cs_sd_p, ac_p, sr_p = {}, {}, {}
    for sig_name, col, lag1_col, lag4_col in sig_cols_p:
        cs_sd_p[sig_name] = df_p.groupby(dates_p)[col].std().mean()
        ac_p[sig_name] = {}
        for lag_name, lag_col in [('lag1', lag1_col), ('lag4', lag4_col)]:
            corrs = []
            for firm in df_p.index.get_level_values('FIRM').unique():
                try:
                    fd = df_p.xs(firm, level='FIRM')[[col, lag_col]].dropna()
                    if len(fd) >= 3:
                        corrs.append(fd[col].corr(fd[lag_col]))
                except (KeyError, ValueError):
                    continue
            ac_p[sig_name][lag_name] = {
                'mean': np.mean(corrs) if corrs else np.nan,
                'median': np.median(corrs) if corrs else np.nan,
                'pct_pos': np.mean(np.array(corrs) > 0) * 100 if corrs else np.nan,
            }
        q_col = f'ALTQ_{sig_name.replace(" ", "_")}'
        ov, per_q = _stay_rates_alt(df_p, q_col)
        sr_p[sig_name] = {'overall': ov, 'Q1': per_q['Q1'], 'Q2': per_q['Q2'],
                          'Q3': per_q['Q3'], 'Q4': per_q['Q4']}

    sig_names_p = [x[0] for x in sig_cols_p]

    def _row(label, vals):
        return [label] + vals

    pers_rows = [
        _row('CS SD (%/qtr)',     [_f(cs_sd_p.get(s, np.nan), 2) for s in sig_names_p]),
        _row('Lag-1 mean',        [_f(ac_p[s]['lag1']['mean']) for s in sig_names_p]),
        _row('Lag-1 median',      [_f(ac_p[s]['lag1']['median']) for s in sig_names_p]),
        _row('Lag-1 % pos.',      [_f(ac_p[s]['lag1']['pct_pos'], 1) for s in sig_names_p]),
        _row('Lag-4 mean',        [_f(ac_p[s]['lag4']['mean']) for s in sig_names_p]),
        _row('Lag-4 median',      [_f(ac_p[s]['lag4']['median']) for s in sig_names_p]),
        _row('Lag-4 % pos.',      [_f(ac_p[s]['lag4']['pct_pos'], 1) for s in sig_names_p]),
        _row('Overall stay (%)',  [_f(sr_p[s]['overall'] * 100, 1) for s in sig_names_p]),
        _row('Q1 stay (%)',       [_f(sr_p[s]['Q1'] * 100, 1) for s in sig_names_p]),
        _row('Q2 stay (%)',       [_f(sr_p[s]['Q2'] * 100, 1) for s in sig_names_p]),
        _row('Q3 stay (%)',       [_f(sr_p[s]['Q3'] * 100, 1) for s in sig_names_p]),
        _row('Q4 stay (%)',       [_f(sr_p[s]['Q4'] * 100, 1) for s in sig_names_p]),
    ]
    col_hdrs_p = ['Metric'] + sig_names_p

    n_r_p, n_c_p = len(pers_rows), len(col_hdrs_p)
    fig, ax = plt.subplots(figsize=(max(10, n_c_p * 2.0), n_r_p * 0.45 + 1.5))
    ax.axis('off')
    tbl = ax.table(cellText=pers_rows, colLabels=col_hdrs_p, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    ax.set_title('Signal Persistence: Alternative RRR Variants', fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(alt_dir, 'signal_persistence.pdf'), bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: signal_persistence.pdf")

    # =========================================================================
    # D. Portfolio Sort + FF3 Alpha + Cumulative Returns
    # =========================================================================
    print("\n--- 1D.D Portfolio sorts + FF3 ---")

    # Build the holding panel (identical k=2 formation / 3-month hold timing as
    # the headline) so the alternative-metric sorts use the same convention and
    # weight by formation-date market cap on SIMPLE returns.
    ret_m = build_holding_panel(df_filtered, returns)
    print(f"  Holding panel: {len(ret_m)} monthly obs, {ret_m['FIRM'].nunique()} firms")

    if ret_m.empty:
        print("  No data in holding panel. Skipping portfolio analysis.")
        return

    market_ret_alt = value_weighted_market(ret_m)

    alt_signals = [
        ('Adj LR_RRR', 'ADJ_LR_RRR_PCT'),
        ('Adj SRR',    'ADJ_SRR_PCT'),
        ('Adj CUMRR',  'ADJ_CUMRR_PCT'),
    ]

    ff3_results = {}
    longshort_rets = {}
    ff3_cols = ['Mkt-RF', 'SMB', 'HML']

    for sig_label, sig_col in alt_signals:
        print(f"\n  Signal: {sig_label}")
        ret_m[sig_col] = pd.to_numeric(ret_m[sig_col], errors='coerce')
        q_col_d = f'Q_D_{sig_col}'
        ret_m[q_col_d] = ret_m.groupby('QUARTER')[sig_col].transform(safe_quartile)

        port = build_portfolio_returns(ret_m, q_col_d, market_ret_alt)
        if port is None or 'Q1-Q4' not in port.columns:
            print(f"  Insufficient data for {sig_label} portfolio sort.")
            continue

        longshort_rets[sig_label] = port['Q1-Q4']

        if not all(c in ff_factors.columns for c in ff3_cols):
            print("  FF3 factors not available. Skipping regression.")
            continue

        port_aligned = port[['Q1-Q4']].copy()
        port_aligned.index = pd.to_datetime(port_aligned.index).to_period('M').to_timestamp('M')
        combined = port_aligned.join(ff_factors[ff3_cols + (['RF'] if 'RF' in ff_factors.columns else [])], how='inner')
        if len(combined) < 12:
            print(f"  Only {len(combined)} overlapping months. Skipping FF3.")
            continue

        y = combined['Q1-Q4']
        X = combined[ff3_cols]
        model = newey_west_ols(y, X)

        ff3_results[sig_label] = {
            'alpha_pct': model.params['const'] * 100,
            'alpha_ann_pct': model.params['const'] * 100 * 12,
            'se_pct': model.bse['const'] * 100,
            't': model.tvalues['const'],
            'p': model.pvalues['const'],
            'n_obs': int(model.nobs),
        }
        print(f"  {sig_label}: alpha={model.params['const']*100:.4f}%/mo, "
              f"t={model.tvalues['const']:.3f}, p=[{model.pvalues['const']:.3f}]")

    if ff3_results:
        sig_lbl = list(ff3_results.keys())
        alpha_rows = [
            ['Alpha (%/month)'] + [_f(ff3_results[s]['alpha_pct'], 4) for s in sig_lbl],
            ['SE']              + [f"({_f(ff3_results[s]['se_pct'], 4)})" for s in sig_lbl],
            ['t-stat']          + [_f(ff3_results[s]['t'], 3) for s in sig_lbl],
            ['p-value']         + [f"[{_f(ff3_results[s]['p'], 3)}]" for s in sig_lbl],
            ['Alpha (% ann.)']  + [_f(ff3_results[s]['alpha_ann_pct'], 4) for s in sig_lbl],
            ['N (months)']      + [str(ff3_results[s]['n_obs']) for s in sig_lbl],
        ]
        col_hdrs_ff = [''] + sig_lbl

        n_r_ff = len(alpha_rows)
        n_c_ff = len(col_hdrs_ff)
        fig, ax = plt.subplots(figsize=(max(8, n_c_ff * 2.5), n_r_ff * 0.6 + 1.5))
        ax.axis('off')
        tbl = ax.table(cellText=alpha_rows, colLabels=col_hdrs_ff, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.8)
        ax.set_title('FF3 Alphas: Q1-Q4 Long-Short (Alternative RRR Variants)', fontsize=10, pad=10)
        plt.tight_layout()
        plt.savefig(os.path.join(alt_dir, 'portfolio_alphas_ff3.pdf'), bbox_inches='tight', dpi=150)
        plt.close()
        print("  Saved: portfolio_alphas_ff3.pdf")

    if longshort_rets:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        fig, ax = plt.subplots(figsize=(12, 5))
        for (sig_label, series), color in zip(longshort_rets.items(), colors):
            cum_ret = series.cumsum() * 100
            ax.plot(cum_ret.index, cum_ret.values, label=sig_label, linewidth=1.5, color=color)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Log Return (%)')
        ax.set_title('Cumulative Q1-Q4 Long-Short Returns: Alternative RRR Variants')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(alt_dir, 'cumret_longshort.pdf'), bbox_inches='tight', dpi=150)
        plt.close()
        print("  Saved: cumret_longshort.pdf")

    print("\n  All alt_metric outputs saved to:", alt_dir)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    import datetime
    import json
    import platform

    # Run manifest: a date-time run ID so this run's outputs are identifiable later.
    RUN_ID = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    print("=" * 80)
    print("  RRR FINANCIAL IMPLICATIONS — ANALYSIS V2")
    print("  Revenue Retention Rates & Stock Prices: High Returns, Low Risk")
    print(f"  RUN ID: {RUN_ID}")
    print(f"  Timing convention: form {FORM_LAG_MONTHS} months after quarter-end, "
          f"hold {HOLD_MONTHS} months")
    print("=" * 80)
    print()

    # Phase 1: Data loading and diagnostics
    df_long, df_filtered, industry_stats = phase1_load_and_diagnose()

    # Phase 1B: Fama-French factors
    ff_factors = phase1b_load_ff_factors()

    # Phase 1C: Monthly returns
    returns = phase1c_load_monthly_returns()

    # Phase 1D: Alternative metric analysis
    run_alt_metric_analysis(df_filtered, returns, ff_factors)

    # Phase 2: Empirical analysis (builds the holding panel; sorts, CAPM, FM, risk)
    empirical_results = phase2_empirical(df_filtered, returns, ff_factors)
    panel = empirical_results['panel']

    # Phase 4: Additional tests (share the same holding panel)
    additional_results = phase4_additional_tests(df_filtered, panel, returns, ff_factors)

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
