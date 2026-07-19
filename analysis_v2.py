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

    # Monthly log returns
    px_only = df_prices.drop(columns=['Date'])
    returns_px = np.log(px_only / px_only.shift(1))
    returns_px['Date'] = df_prices['Date'].values

    returns = returns_px.melt(id_vars='Date', var_name='FirmVar', value_name='RETURN_LOG')
    returns = returns.dropna(subset=['RETURN_LOG'])
    returns['FIRM'] = returns['FirmVar'].str.replace('.PX_LAST', '', regex=False).str.upper()
    returns.drop(columns=['FirmVar'], inplace=True)

    # Quarter assignment: shift back 1 quarter (use Q4 2019 RRR for Jan-Mar 2020)
    returns['QUARTER'] = (
        returns['Date']
        .dt.to_period('Q')
        .apply(lambda x: x - 1)
        .dt.to_timestamp('Q', 'end')
    )

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
    # Merge quarterly signals onto monthly returns
    returns_merged = merge_signals_to_returns(df_filtered, returns)

    # Build portfolios and run factor regressions
    results['portfolios'] = run_portfolio_analysis(returns_merged, ff_factors)

    # --- 2.5 Fama-MacBeth ---
    print("\n--- 2.5 Fama-MacBeth regressions ---")
    results['fama_macbeth'] = run_fama_macbeth(df_filtered)

    # --- 2.6 Risk analysis ---
    print("\n--- 2.6 Risk analysis ---")
    results['risk'] = run_risk_analysis(returns_merged, ff_factors)

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


def merge_signals_to_returns(df_filtered, returns):
    """Merge quarterly signals onto monthly returns data."""

    print("\n  Merging quarterly signals onto monthly returns...")

    # Prepare quarterly signal data
    df_q = df_filtered.reset_index()[[
        'FIRM', 'DATE', 'RRR_LAG', 'RRR_PCT_LAG1', 'ACQ_RATE_PCT_LAG1',
        'ADJ_RRR_PCT_LAG1', 'ADJ_ACQ_RATE_PCT_LAG1',
        'HISTORICAL_MARKET_CAP', 'SIZE', 'BTM', 'PM_OPER_PCT',
        'SECTOR', 'REV_GROWTH_PCT'
    ]].copy()
    df_q['DATE'] = pd.to_datetime(df_q['DATE'])
    df_q['QUARTER'] = df_q['DATE'].dt.to_period('Q').dt.to_timestamp('Q', 'end')
    df_q['FIRM'] = df_q['FIRM'].str.upper()

    # Merge
    ret = returns.copy()
    ret['FIRM'] = ret['FIRM'].str.upper()
    ret = ret.merge(df_q.drop(columns=['DATE']), on=['FIRM', 'QUARTER'], how='inner')

    # Filter to only firms in our sector-filtered sample
    valid_firms = df_filtered.index.get_level_values('FIRM').unique()
    ret = ret[ret['FIRM'].isin(valid_firms)]

    print(f"  Merged: {len(ret)} monthly obs, {ret['FIRM'].nunique()} firms")

    return ret


def run_portfolio_analysis(returns_merged, ff_factors):
    """Run single and double portfolio sorts with factor regressions.

    Output order:
      1. Univariate quartile sorts (Q1=highest): raw RRR, raw AR, adj RRR, adj AR
      2. Univariate tercile sorts (T1=highest): raw RRR, raw AR, adj RRR, adj AR
      3. Bivariate 3x3 double sorts: RAW then ADJ
    """

    results = {}
    ret = returns_merged.copy()

    # Ensure numeric signal columns
    for col in ['RRR_PCT_LAG1', 'ACQ_RATE_PCT_LAG1', 'ADJ_RRR_PCT_LAG1', 'ADJ_ACQ_RATE_PCT_LAG1', 'HISTORICAL_MARKET_CAP']:
        ret[col] = pd.to_numeric(ret[col], errors='coerce')

    # Market portfolio: VW return of all sample firms
    ret_sample = ret[~ret['FIRM'].isin(['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX'])].copy()
    ret_sample['MCAP'] = ret_sample['HISTORICAL_MARKET_CAP']
    ret_sample['MCAP_TOTAL'] = ret_sample.groupby('Date')['MCAP'].transform('sum')
    ret_sample['w_mkt'] = ret_sample['MCAP'] / ret_sample['MCAP_TOTAL']
    ret_sample['w_ret_mkt'] = ret_sample['w_mkt'] * ret_sample['RETURN_LOG']
    market_ret = ret_sample.groupby('Date')['w_ret_mkt'].sum().sort_index()

    # =========================================================================
    # SECTION 1: UNIVARIATE QUARTILE SORTS (Q1 = highest value)
    # =========================================================================
    print("\n  *** SECTION 1: UNIVARIATE QUARTILE SORTS (Q1=best) ***")

    for signal_label, rrr_col in [
        ('RAW', 'RRR_PCT_LAG1'),
        ('ADJ', 'ADJ_RRR_PCT_LAG1'),
    ]:
        print(f"\n  === {signal_label} signal — quartile sort ===")

        # Univariate RRR quartile sort
        print(f"\n  Univariate RRR quartile ({signal_label}):")
        ret[f'RRR_Q_{signal_label}'] = ret.groupby('QUARTER')[rrr_col].transform(safe_quartile)
        port_rrr_q = build_portfolio_returns(ret, f'RRR_Q_{signal_label}', market_ret)
        if port_rrr_q is not None:
            results[f'port_rrr_{signal_label.lower()}_q4_vw'] = port_rrr_q
            run_factor_regressions(port_rrr_q, ff_factors, f'RRR {signal_label} Q4 VW')

    return results


def build_portfolio_returns(ret, tercile_col, market_ret):
    """Build VW tercile portfolio returns + long-short + market."""

    df = ret.dropna(subset=[tercile_col]).copy()
    df['MCAP'] = pd.to_numeric(df['HISTORICAL_MARKET_CAP'], errors='coerce')

    if df.empty:
        print("    No valid data for portfolio construction")
        return None

    # Value-weighted returns per tercile
    df['MCAP_SUM'] = df.groupby(['Date', tercile_col])['MCAP'].transform('sum')
    df['w'] = df['MCAP'] / df['MCAP_SUM']
    df['w_return'] = df['w'] * df['RETURN_LOG']

    port = (
        df.groupby(['Date', tercile_col])['w_return']
        .sum()
        .unstack(tercile_col)
        .sort_index()
    )

    # Long-short: HIGH minus LOW (Q1-Q4 for quartiles, T1-T3 for terciles)
    if 'Q1' in port.columns and 'Q4' in port.columns:
        port['Q1-Q4'] = port['Q1'] - port['Q4']
    elif 'T1' in port.columns and 'T3' in port.columns:
        port['T1-T3'] = port['T1'] - port['T3']

    # Add market portfolio
    port['MKT'] = market_ret.reindex(port.index)

    # Print summary returns
    print(f"    Portfolio monthly log returns (annualized):")
    for col in port.columns:
        ann_ret = port[col].mean() * 12
        ann_vol = port[col].std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        print(f"      {col}: ret={ann_ret*100:.2f}%, vol={ann_vol*100:.2f}%, SR={sharpe:.2f}")

    # Firms per tercile per quarter
    counts = df.groupby(['QUARTER', tercile_col])['FIRM'].nunique().unstack(tercile_col)
    print(f"    Avg firms per tercile: {counts.mean().to_dict()}")

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


def run_fama_macbeth(df_filtered):
    """Run Fama-MacBeth cross-sectional regressions."""

    results = {}

    df = df_filtered.reset_index().copy()
    df['DATE'] = pd.to_datetime(df['DATE'])

    # Lead excess return (next quarter)
    df['EXCESS_RET_LEAD'] = df.groupby('FIRM')['EXCESS_RET'].shift(-1)
    df['RET_ARITH_LEAD'] = df.groupby('FIRM')['RET_ARITH'].shift(-1)

    # Control variables: lagged values
    df['SIZE_LAG'] = df.groupby('FIRM')['SIZE'].shift(1)
    df['BTM_LAG'] = df.groupby('FIRM')['BTM'].shift(1)
    df['PM_LAG'] = df.groupby('FIRM')['PM_OPER_PCT'].shift(1)

    # Momentum (use past return as proxy)
    df['RET_LAG'] = df.groupby('FIRM')['RET_ARITH'].shift(1)

    # Define specifications (RRR-only; 4 specs)
    specs = {
        '(1) Adj RRR only': ['ADJ_RRR_PCT'],
        '(2) Adj RRR + Controls': ['ADJ_RRR_PCT', 'SIZE_LAG', 'BTM_LAG', 'PM_LAG'],
        '(3) Raw RRR only': ['RRR_PCT'],
        '(4) Raw RRR + Controls': ['RRR_PCT', 'SIZE_LAG', 'BTM_LAG', 'PM_LAG'],
    }

    y_var = 'EXCESS_RET_LEAD'

    for spec_name, x_vars in specs.items():
        # Cross-sectional regression each period
        period_coefs = []

        for date, group in df.groupby('DATE'):
            sub = group[[y_var] + x_vars].dropna()
            if len(sub) < 10:
                continue

            y = sub[y_var]
            X = sm.add_constant(sub[x_vars])

            try:
                model = sm.OLS(y, X).fit()
                coefs = model.params.to_dict()
                coefs['DATE'] = date
                coefs['N'] = len(sub)
                period_coefs.append(coefs)
            except Exception:
                continue

        if not period_coefs:
            print(f"  {spec_name}: No valid periods")
            continue

        coef_df = pd.DataFrame(period_coefs)

        # Time-series average with Newey-West t-stats
        T = len(coef_df)
        avg_coefs = coef_df.drop(columns=['DATE', 'N']).mean()

        # Newey-West standard errors (Bartlett kernel)
        max_lag = max(1, int(np.floor(4 * (T/100)**(2/9))))
        se_nw = {}
        for var in avg_coefs.index:
            series = coef_df[var] - avg_coefs[var]
            gamma0 = (series**2).mean()
            gamma_sum = gamma0
            for j in range(1, max_lag + 1):
                gamma_j = (series.iloc[j:].values * series.iloc[:-j].values).mean()
                gamma_sum += 2 * (1 - j/(max_lag+1)) * gamma_j
            se_nw[var] = np.sqrt(gamma_sum / T)

        t_stats = {var: avg_coefs[var] / se_nw[var] if se_nw[var] > 0 else np.nan for var in avg_coefs.index}

        print(f"\n  {spec_name} (T={T}, avg N={coef_df['N'].mean():.0f}):")
        for var in [v for v in avg_coefs.index if v != 'const']:
            stars = ''
            t = abs(t_stats[var])
            if t > 2.576: stars = '***'
            elif t > 1.96: stars = '**'
            elif t > 1.645: stars = '*'
            print(f"    {var:<25} coef={avg_coefs[var]:>10.4f}  t={t_stats[var]:>7.2f}{stars}")

        results[spec_name] = {
            'avg_coefs': avg_coefs.to_dict(),
            't_stats': t_stats,
            'T': T,
            'avg_N': coef_df['N'].mean()
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
                'T': res['T'],
                'Avg N': res['avg_N']
            })

    fm_df = pd.DataFrame(fm_summary)
    fm_df.to_excel(os.path.join(OUTPUT_DIR, 'fama_macbeth.xlsx'), index=False)
    print(f"\n  Fama-MacBeth results exported to {OUTPUT_DIR}/fama_macbeth.xlsx")

    return results


def run_risk_analysis(returns_merged, ff_factors):
    """Analyze risk characteristics by portfolio tercile."""

    print("\n  Risk analysis by RRR and AR terciles:")

    ret = returns_merged.copy()
    ret['ADJ_RRR_PCT_LAG1'] = pd.to_numeric(ret['ADJ_RRR_PCT_LAG1'], errors='coerce')
    ret['RRR_T'] = ret.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_tercile)

    results = {}

    # Build VW portfolio returns by RRR tercile
    df = ret.dropna(subset=['RRR_T']).copy()
    df['MCAP'] = pd.to_numeric(df['HISTORICAL_MARKET_CAP'], errors='coerce')
    df['MCAP_SUM'] = df.groupby(['Date', 'RRR_T'])['MCAP'].transform('sum')
    df['w'] = df['MCAP'] / df['MCAP_SUM']
    df['w_return'] = df['w'] * df['RETURN_LOG']

    port = (
        df.groupby(['Date', 'RRR_T'])['w_return']
        .sum()
        .unstack('RRR_T')
        .sort_index()
    )

    if port.empty:
        print("  No valid portfolio data for risk analysis")
        return results

    # Risk metrics
    risk_metrics = []
    for col in port.columns:
        rets = port[col].dropna()
        cum = np.exp(rets.cumsum())
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

def phase4_additional_tests(df_filtered, returns_merged, ff_factors, returns=None):
    """Same-growth placebo and robustness tests."""

    print("\n" + "=" * 80)
    print("PHASE 4: ADDITIONAL TESTS")
    print("=" * 80)

    results = {}

    # --- 4.1 Same-growth placebo (median split) ---
    print("\n--- 4.1 Same-growth placebo (median split) ---")
    results['placebo'] = run_placebo_median_split(returns_merged, ff_factors)

    # --- 4.2 Robustness: No-COVID (quartile) ---
    print("\n--- 4.4 Robustness: Excluding COVID (quartile) ---")
    ret_no_covid_q = returns_merged[
        ~((returns_merged['Date'] >= '2020-01-01') & (returns_merged['Date'] <= '2021-06-30'))
    ].copy()

    if len(ret_no_covid_q) > 0:
        # Market portfolio (no COVID, for quartile version)
        ret_sample_q = ret_no_covid_q[~ret_no_covid_q['FIRM'].isin(['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX'])].copy()
        ret_sample_q['MCAP'] = pd.to_numeric(ret_sample_q['HISTORICAL_MARKET_CAP'], errors='coerce')
        ret_sample_q['MCAP_TOTAL'] = ret_sample_q.groupby('Date')['MCAP'].transform('sum')
        ret_sample_q['w_mkt'] = ret_sample_q['MCAP'] / ret_sample_q['MCAP_TOTAL']
        ret_sample_q['w_ret_mkt'] = ret_sample_q['w_mkt'] * ret_sample_q['RETURN_LOG']
        market_ret_nc_q = ret_sample_q.groupby('Date')['w_ret_mkt'].sum().sort_index()

        ret_no_covid_q['ADJ_RRR_PCT_LAG1'] = pd.to_numeric(ret_no_covid_q['ADJ_RRR_PCT_LAG1'], errors='coerce')
        ret_no_covid_q['RRR_Q_NC'] = ret_no_covid_q.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_quartile)

        port_nc_q = build_portfolio_returns(ret_no_covid_q, 'RRR_Q_NC', market_ret_nc_q)
        if port_nc_q is not None:
            run_factor_regressions(port_nc_q, ff_factors, 'RRR ADJ VW NoCOVID Quartile')
            results['no_covid_quartile'] = port_nc_q

    # --- 4.3 Robustness: Equal-weighted (quartile) ---
    print("\n--- 4.3 Robustness: Equal-weighted portfolios (quartile) ---")
    ret_ew_q = returns_merged.copy()
    ret_ew_q['ADJ_RRR_PCT_LAG1'] = pd.to_numeric(ret_ew_q['ADJ_RRR_PCT_LAG1'], errors='coerce')
    ret_ew_q['RRR_Q_EW'] = ret_ew_q.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_quartile)

    df_ew_q = ret_ew_q.dropna(subset=['RRR_Q_EW']).copy()
    port_ew_q = (
        df_ew_q.groupby(['Date', 'RRR_Q_EW'])['RETURN_LOG']
        .mean()
        .unstack('RRR_Q_EW')
        .sort_index()
    )
    if 'Q1' in port_ew_q.columns and 'Q4' in port_ew_q.columns:
        port_ew_q['Q1-Q4'] = port_ew_q['Q1'] - port_ew_q['Q4']

    # EW market (quartile version)
    ew_mkt_q = df_ew_q.groupby('Date')['RETURN_LOG'].mean().sort_index()
    port_ew_q['MKT'] = ew_mkt_q.reindex(port_ew_q.index)

    print(f"  EW quartile portfolio returns (annualized):")
    for col in port_ew_q.columns:
        ann = port_ew_q[col].mean() * 12 * 100
        print(f"    {col}: {ann:.2f}%")

    run_factor_regressions(port_ew_q, ff_factors, 'RRR ADJ EW Quartile')
    results['ew_quartile'] = port_ew_q

    # --- 4.4 Robustness: Lag-2 portfolios ---
    if returns is not None:
        print("\n--- 4.4 Robustness: Lag-2 portfolios (2-quarter gap) ---")
        results['lag2'] = run_lag2_portfolios(df_filtered, returns, ff_factors)

    return results


def run_lag2_portfolios(df_filtered, returns, ff_factors):
    """Robustness: quartile sorts with a 2-quarter lag (returns in Q_{t+2} from signal in Q_t).

    Shifts the QUARTER key back by 2 quarters so each monthly return in quarter Q_t
    matches the signal observed at the end of Q_{t-2}. This tests whether RRR
    predicts returns beyond the immediate post-sort quarter, ruling out contamination
    by earnings surprises concentrated in the quarter following the signal.
    """

    print("\n  === Lag-2 robustness portfolios (2-quarter gap) ===")

    # Shift QUARTER reference back one additional quarter (vs. main spec which uses -1)
    ret2 = returns.copy()
    ret2['QUARTER'] = (
        ret2['Date'].dt.to_period('Q')
        .apply(lambda x: x - 2)
        .dt.to_timestamp('Q', 'end')
    )

    ret2_merged = merge_signals_to_returns(df_filtered, ret2)

    if ret2_merged.empty or len(ret2_merged) < 100:
        print("  Insufficient data for lag-2 portfolios")
        return {}

    # VW market portfolio
    ret_sample = ret2_merged[
        ~ret2_merged['FIRM'].isin(['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX'])
    ].copy()
    ret_sample['MCAP'] = pd.to_numeric(ret_sample['HISTORICAL_MARKET_CAP'], errors='coerce')
    ret_sample['MCAP_TOTAL'] = ret_sample.groupby('Date')['MCAP'].transform('sum')
    ret_sample['w_mkt'] = ret_sample['MCAP'] / ret_sample['MCAP_TOTAL']
    ret_sample['w_ret_mkt'] = ret_sample['w_mkt'] * ret_sample['RETURN_LOG']
    market_ret_lag2 = ret_sample.groupby('Date')['w_ret_mkt'].sum().sort_index()

    # Adj RRR quartile sort
    ret2_merged['ADJ_RRR_PCT_LAG1'] = pd.to_numeric(ret2_merged['ADJ_RRR_PCT_LAG1'], errors='coerce')
    ret2_merged['RRR_Q_LAG2'] = ret2_merged.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(safe_quartile)

    port_lag2 = build_portfolio_returns(ret2_merged, 'RRR_Q_LAG2', market_ret_lag2)
    if port_lag2 is not None:
        reg_results = run_factor_regressions(port_lag2, ff_factors, 'RRR ADJ VW Lag2')
        if reg_results:
            reg_df = pd.DataFrame(reg_results).T.reset_index()
            reg_df.rename(columns={'index': 'Unnamed: 0'}, inplace=True)
            reg_df.to_excel(os.path.join(OUTPUT_DIR, 'factor_reg_Lag2_adj.xlsx'), index=False)

    return {'port': port_lag2}


def run_placebo_median_split(returns_merged, ff_factors):
    """Same-growth placebo: top-growth-quartile firms split at median of adj RRR.

    Restrict to firms in the top revenue-growth quartile (~31 firms per quarter),
    then split at the median of adj RRR within that group, yielding a High-RRR
    and Low-RRR bucket of ~15-16 firms each. Tests whether RRR predicts returns
    beyond revenue growth level (composition, not level).
    """

    ret = returns_merged.copy()
    ret['REV_GROWTH_PCT'] = pd.to_numeric(ret['REV_GROWTH_PCT'], errors='coerce')
    ret['ADJ_RRR_PCT_LAG1'] = pd.to_numeric(ret['ADJ_RRR_PCT_LAG1'], errors='coerce')

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

    high_growth['RRR_MED'] = high_growth.groupby('QUARTER')['ADJ_RRR_PCT_LAG1'].transform(median_split)

    df_pl = high_growth.dropna(subset=['RRR_MED']).copy()
    df_pl['MCAP'] = pd.to_numeric(df_pl['HISTORICAL_MARKET_CAP'], errors='coerce')
    df_pl['MCAP_SUM'] = df_pl.groupby(['Date', 'RRR_MED'])['MCAP'].transform('sum')
    df_pl['w'] = df_pl['MCAP'] / df_pl['MCAP_SUM']
    df_pl['w_return'] = df_pl['w'] * df_pl['RETURN_LOG']

    port_pl = (
        df_pl.groupby(['Date', 'RRR_MED'])['w_return']
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

    sig_lag_cols = ['ADJ_LR_RRR_PCT_LAG1', 'ADJ_SRR_PCT_LAG1', 'ADJ_CUMRR_PCT_LAG1']
    df_q = df_filtered.reset_index()[
        ['FIRM', 'DATE'] + sig_lag_cols + ['HISTORICAL_MARKET_CAP']
    ].copy()
    df_q['DATE'] = pd.to_datetime(df_q['DATE'])
    df_q['QUARTER'] = df_q['DATE'].dt.to_period('Q').dt.to_timestamp('Q', 'end')
    df_q['FIRM'] = df_q['FIRM'].str.upper()

    ret = returns.copy()
    ret['FIRM'] = ret['FIRM'].str.upper()
    ret_m = ret.merge(df_q.drop(columns=['DATE']), on=['FIRM', 'QUARTER'], how='inner')
    valid_firms = df_filtered.index.get_level_values('FIRM').unique()
    ret_m = ret_m[ret_m['FIRM'].isin(valid_firms)]
    print(f"  Merged: {len(ret_m)} monthly obs, {ret_m['FIRM'].nunique()} firms")

    if ret_m.empty:
        print("  No data after merge. Skipping portfolio analysis.")
        return

    sample_m = ret_m[~ret_m['FIRM'].isin(['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX'])].copy()
    sample_m['MCAP'] = pd.to_numeric(sample_m['HISTORICAL_MARKET_CAP'], errors='coerce')
    sample_m['MCAP_TOTAL'] = sample_m.groupby('Date')['MCAP'].transform('sum')
    sample_m['w'] = sample_m['MCAP'] / sample_m['MCAP_TOTAL']
    sample_m['w_ret'] = sample_m['w'] * sample_m['RETURN_LOG']
    market_ret_alt = sample_m.groupby('Date')['w_ret'].sum().sort_index()

    alt_signals = [
        ('Adj LR_RRR', 'ADJ_LR_RRR_PCT_LAG1'),
        ('Adj SRR',    'ADJ_SRR_PCT_LAG1'),
        ('Adj CUMRR',  'ADJ_CUMRR_PCT_LAG1'),
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
    print("=" * 80)
    print("  RRR FINANCIAL IMPLICATIONS — ANALYSIS V2")
    print("  Revenue Retention Rates & Stock Prices: High Returns, Low Risk")
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

    # Phase 2: Empirical analysis
    empirical_results = phase2_empirical(df_filtered, returns, ff_factors)

    # Phase 4: Additional tests
    returns_merged = merge_signals_to_returns(df_filtered, returns)
    additional_results = phase4_additional_tests(df_filtered, returns_merged, ff_factors, returns=returns)

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)

    # List output files
    # Factor regression results are returned in-memory only (no .xlsx per regression).
    # Data summary exports: fama_macbeth.xlsx, pooled_descriptives.xlsx,
    #                       industry_descriptives.xlsx, risk_analysis.xlsx
    print("\n  Output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    {f} ({size:,} bytes)")
