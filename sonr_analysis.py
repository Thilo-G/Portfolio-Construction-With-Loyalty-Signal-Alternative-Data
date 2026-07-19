"""
sonr_analysis.py — Exploratory: SoNR by RRR Quartile
======================================================
Hypothesis: Low-RRR firms compensate via acquisition, so they should
exhibit higher Share of New Revenue (SoNR).

Analysis:
  1. Simple group means: mean SoNR by RRR quartile
  2. Panel regressions with escalating FE (no FE / firm FE / time FE / both)
  3. Monotonicity check: is Q1 (high RRR) -> Q4 (low RRR) ordering of SoNR monotonic?

Variable definitions (from analysis_v2.py):
  #SHARE_RET_REVENUE = retained_revenue / total_revenue  (share of RETAINED revenue)
  SoNR               = 1 - #SHARE_RET_REVENUE            (share of NEW revenue)
  #RRR               = retained_revenue / prior_period_total_revenue
  Quartiles: Q1 = highest RRR (best retention), Q4 = lowest RRR (worst retention)
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, PooledOLS

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================
DATA_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Data"
SUPPORT_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Supporting-Documents"
OUTPUT_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM\output"

# Replicates the sector filter from analysis_v2.py
VALID_SECTORS = ['Consumer Discretionary', 'Communication Services', 'Consumer Staples', 'Industrials']

FILE_REVENUE = os.path.join(DATA_DIR, "2025-0319a-TK-quarterlyrevenue-collection_Python.xlsx")
FILE_FUNDAMENTALS = os.path.join(DATA_DIR, "2025-0319a-TK-fundamentals_Python.xlsx")
FILE_INDUSTRY = os.path.join(SUPPORT_DIR, "Firms-Industry-2025-11-21-Python.xlsx")

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sonr_by_quantile.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING — mirrors phase1_load_and_diagnose() from analysis_v2.py
# =============================================================================

def load_revenue_data():
    """Load and reshape the revenue Excel file into long panel format."""
    print("Loading revenue data...")
    df_revenue = pd.read_excel(FILE_REVENUE, header=None, engine='openpyxl')

    # Reconstruct two-row combined header (same logic as analysis_v2.py)
    header_rows = df_revenue.iloc[:2]
    data_rows = df_revenue.iloc[2:].copy()
    combined_headers = header_rows.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
    data_rows.columns = column_headers
    data_rows.rename(columns={data_rows.columns[0]: 'Date'}, inplace=True)
    df_revenue = data_rows.reset_index(drop=True)
    df_revenue = df_revenue.replace(r'^\s*$', pd.NA, regex=True)
    df_revenue = df_revenue.loc[:, ~df_revenue.columns.duplicated()]

    # Compute per-firm revenue metrics (RRR, AR, SoNR, RG)
    new_customer_cols = [col for col in df_revenue.columns if '#New_Customers' in col]
    returning_customer_cols = [col for col in df_revenue.columns if '#Returning_Customers' in col]

    for new_col, ret_col in zip(new_customer_cols, returning_customer_cols):
        total_col = new_col.replace('#New_Customers', '#Total_Revenue')
        rrr_col = ret_col.replace('#Returning_Customers', '#RRR')
        ar_col = new_col.replace('#New_Customers', '#Acq_Rate')
        share_ret_col = ret_col.replace('#Returning_Customers', '#Share_Ret_Revenue')

        df_revenue[total_col] = df_revenue[new_col] + df_revenue[ret_col]
        # Share of retained revenue: retained / total
        df_revenue[share_ret_col] = (
            df_revenue[ret_col] / df_revenue[total_col].replace(0, np.nan)
        )
        # RRR: retained revenue relative to prior period total
        df_revenue[rrr_col] = (
            df_revenue[ret_col] / df_revenue[total_col].shift(1).replace(0, np.nan)
        )
        # Acquisition rate
        df_revenue[ar_col] = df_revenue[new_col] / df_revenue[new_col].shift(1)

    print(f"  Revenue data: {df_revenue.shape}")
    return df_revenue


def load_fundamentals():
    """Load fundamentals file for date spine and sector merge."""
    print("Loading fundamentals...")
    df_fun = pd.read_excel(FILE_FUNDAMENTALS, header=None, engine='openpyxl')

    header_rows = df_fun.iloc[:2]
    data_rows = df_fun.iloc[2:].copy()
    combined_headers = header_rows.apply(lambda x: x.astype(str).str.strip(), axis=0)
    column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
    data_rows.columns = column_headers
    df_fun = data_rows.reset_index(drop=True)
    df_fun.replace('#N/A N/A', np.nan, inplace=True)
    df_fun = df_fun.loc[:, ~df_fun.columns.duplicated()]

    print(f"  Fundamentals: {df_fun.shape}")
    return df_fun


def build_panel(df_revenue, df_fun):
    """Merge revenue and fundamentals into a long panel with sector labels."""
    print("Building long panel...")

    # Combine wide frames, drop the duplicate Date column from revenue
    df_rev_no_date = df_revenue.drop(columns=['Date'], errors='ignore')
    df_fun = df_fun.rename(columns={'nan.Dates': 'Date'})
    df_combined = pd.concat([df_fun, df_rev_no_date], axis=1)
    df_combined.columns = df_combined.columns.str.upper()
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

    # Melt to long, then pivot to firm x date panel
    all_vars = [col for col in df_combined.columns if col != 'DATE']
    df_long = df_combined.melt(
        id_vars=['DATE'], value_vars=all_vars,
        var_name='Firm_Variable', value_name='Value'
    )
    df_long[['FIRM', 'VARIABLE']] = df_long['Firm_Variable'].str.rsplit('.', n=1, expand=True)
    df_long = (
        df_long
        .pivot_table(index=['FIRM', 'DATE'], columns='VARIABLE', values='Value')
        .reset_index()
        .set_index(['FIRM', 'DATE'])
    )

    # Remove index tickers (not real firms)
    index_firms = ['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX']
    df_long = df_long.loc[~df_long.index.get_level_values('FIRM').isin(index_firms)].copy()

    # Attach GICS sector
    df_industry = pd.read_excel(FILE_INDUSTRY, engine='openpyxl')
    df_industry.columns = ['ID', 'GICS_INDUSTRY', 'GICS_SECTOR', 'GICS_SUB_INDUSTRY']
    df_industry['FIRM'] = df_industry['ID'].str.upper().str.strip()
    firm_sector_map = df_industry.set_index('FIRM')['GICS_SECTOR'].to_dict()

    df_long['SECTOR'] = df_long.index.get_level_values('FIRM').map(firm_sector_map)

    # Apply sector filter (same as analysis_v2.py)
    df_filtered = df_long[df_long['SECTOR'].isin(VALID_SECTORS)].copy()

    n_firms = df_filtered.index.get_level_values('FIRM').nunique()
    n_periods = df_filtered.index.get_level_values('DATE').nunique()
    print(f"  Panel after sector filter: {n_firms} firms, {n_periods} quarters")

    return df_filtered


def compute_analysis_vars(df_filtered):
    """Compute RRR (%), SoNR, and industry-time adjusted RRR."""
    print("Computing analysis variables...")

    # RRR as percentage (0-100 scale)
    df_filtered['RRR_PCT'] = pd.to_numeric(df_filtered.get('#RRR', np.nan), errors='coerce') * 100

    # SoNR = 1 - Share_Retained_Revenue (both are fractions 0-1; keep SoNR as fraction)
    share_ret = pd.to_numeric(df_filtered.get('#SHARE_RET_REVENUE', np.nan), errors='coerce')
    df_filtered['SONR'] = 1.0 - share_ret  # fraction: new revenue / total revenue

    # Industry x time adjusted RRR (demeaned within sector-quarter)
    sector_quarter_mean = df_filtered.groupby(
        [df_filtered.index.get_level_values('DATE'), 'SECTOR']
    )['RRR_PCT'].transform('mean')
    df_filtered['ADJ_RRR_PCT'] = df_filtered['RRR_PCT'] - sector_quarter_mean

    n_valid = df_filtered[['RRR_PCT', 'SONR']].dropna().shape[0]
    print(f"  Firm-quarters with both RRR and SoNR: {n_valid}")

    return df_filtered


# =============================================================================
# ANALYSIS
# =============================================================================

def assign_rrr_quartiles(df):
    """
    Assign RRR quartiles cross-sectionally within each quarter.
    Q1 = highest RRR (best retention), Q4 = lowest RRR (worst retention).
    Returns df with a 'RRR_Q' column.
    """
    def _quartile_rank(x):
        # qcut with labels 1-4; then invert so Q1 = highest
        x_clean = x.dropna()
        if x_clean.nunique() < 4 or len(x_clean) < 8:
            return pd.Series(np.nan, index=x.index)
        try:
            # ascending=True → Q1=lowest raw; we flip labels so Q1=highest
            raw = pd.qcut(x_clean, 4, labels=False)  # 0=lowest, 3=highest
            flipped = 4 - raw  # 1=highest RRR, 4=lowest RRR
            return flipped.reindex(x.index)
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    df = df.copy()
    df['RRR_Q'] = df.groupby(level='DATE')['RRR_PCT'].transform(_quartile_rank)
    df['RRR_Q'] = pd.to_numeric(df['RRR_Q'], errors='coerce')

    quartile_counts = df['RRR_Q'].value_counts().sort_index()
    print(f"\n  Quartile observation counts:\n{quartile_counts.to_string()}")

    return df


def report_group_means(df):
    """
    Part (a): Mean and median SoNR by RRR quartile, with t-test vs Q1.
    """
    print("\n" + "=" * 60)
    print("PART (a): GROUP MEANS — SoNR by RRR Quartile")
    print("=" * 60)
    print("  Q1 = highest RRR (best retention)  |  Q4 = lowest RRR (worst retention)")
    print()

    work = df[['SONR', 'RRR_Q', 'RRR_PCT']].dropna()

    stats_rows = []
    for q in [1, 2, 3, 4]:
        subset = work[work['RRR_Q'] == q]['SONR']
        stats_rows.append({
            'RRR_Quartile': f'Q{int(q)}',
            'N': len(subset),
            'Mean_SoNR': subset.mean(),
            'Median_SoNR': subset.median(),
            'SD_SoNR': subset.std(),
            'Mean_RRR_PCT': work[work['RRR_Q'] == q]['RRR_PCT'].mean(),
        })

    group_means = pd.DataFrame(stats_rows)

    # t-test: each quartile vs Q1 (two-sided)
    from scipy.stats import ttest_ind
    q1_vals = work[work['RRR_Q'] == 1]['SONR'].dropna()
    t_stats, p_vals = [], []
    for q in [1, 2, 3, 4]:
        if q == 1:
            t_stats.append(np.nan)
            p_vals.append(np.nan)
        else:
            q_vals = work[work['RRR_Q'] == q]['SONR'].dropna()
            t_val, p_val = ttest_ind(q_vals, q1_vals, equal_var=False)
            t_stats.append(t_val)
            p_vals.append(p_val)

    group_means['t_vs_Q1'] = t_stats
    group_means['p_vs_Q1'] = p_vals

    print(group_means.to_string(index=False, float_format='{:.4f}'.format))

    # Q4 - Q1 spread
    q1_mean = group_means.loc[group_means['RRR_Quartile'] == 'Q1', 'Mean_SoNR'].values[0]
    q4_mean = group_means.loc[group_means['RRR_Quartile'] == 'Q4', 'Mean_SoNR'].values[0]
    print(f"\n  Q4 - Q1 spread in mean SoNR: {q4_mean - q1_mean:+.4f} "
          f"({(q4_mean - q1_mean)*100:+.2f} percentage points)")

    return group_means


def run_panel_regressions(df):
    """
    Part (b): SoNR_{i,t} = alpha + beta_Q2*D_Q2 + beta_Q3*D_Q3 + beta_Q4*D_Q4 + FE + eps
    Four specifications: no FE, firm FE, time FE, firm+time FE.
    Standard errors clustered by firm in all specs.
    """
    print("\n" + "=" * 60)
    print("PART (b): PANEL REGRESSIONS")
    print("=" * 60)
    print("  Dep. var: SoNR  |  Base: Q1 (highest RRR)")
    print("  SE: clustered by firm\n")

    work = df[['SONR', 'RRR_Q']].dropna().copy()

    # Create dummy variables Q2, Q3, Q4 (Q1 is the omitted reference)
    work['D_Q2'] = (work['RRR_Q'] == 2).astype(float)
    work['D_Q3'] = (work['RRR_Q'] == 3).astype(float)
    work['D_Q4'] = (work['RRR_Q'] == 4).astype(float)

    # Rebuild MultiIndex with datetime DATE level (linearmodels requires this)
    firms = work.index.get_level_values('FIRM')
    dates = pd.to_datetime(work.index.get_level_values('DATE'))
    work.index = pd.MultiIndex.from_arrays([firms, dates], names=['FIRM', 'DATE'])

    reg_vars = ['D_Q2', 'D_Q3', 'D_Q4']
    X = work[reg_vars]

    spec_results = []

    specs = [
        ('No FE',        dict(entity_effects=False, time_effects=False)),
        ('Firm FE',      dict(entity_effects=True,  time_effects=False)),
        ('Time FE',      dict(entity_effects=False, time_effects=True)),
        ('Firm+Time FE', dict(entity_effects=True,  time_effects=True)),
    ]

    for spec_name, fe_kwargs in specs:
        try:
            if not fe_kwargs['entity_effects'] and not fe_kwargs['time_effects']:
                # Pooled OLS via linearmodels for consistent cluster SE
                model = PooledOLS(work['SONR'], sm.add_constant(X))
            else:
                model = PanelOLS(work['SONR'], X,
                                 entity_effects=fe_kwargs['entity_effects'],
                                 time_effects=fe_kwargs['time_effects'])

            res = model.fit(cov_type='clustered', cluster_entity=True)

            # Extract coefficients
            row = {'Specification': spec_name, 'N': int(res.nobs)}
            for var in reg_vars:
                coef = res.params.get(var, np.nan)
                pval = res.pvalues.get(var, np.nan)
                stars = ('***' if pval < 0.01 else '**' if pval < 0.05
                         else '*' if pval < 0.10 else '')
                row[var] = coef
                row[f'{var}_p'] = pval
                row[f'{var}_stars'] = stars

            # R-squared
            try:
                row['R2'] = res.rsquared
            except Exception:
                row['R2'] = np.nan

            spec_results.append(row)

            # Print nicely
            print(f"  [{spec_name}]  N={row['N']:,}  R2={row.get('R2', np.nan):.4f}")
            for var in reg_vars:
                label = var.replace('D_', '')
                print(f"    {label}: coef={row[var]:+.4f}  p={row[f'{var}_p']:.3f} {row[f'{var}_stars']}")
            print()

        except Exception as exc:
            print(f"  [{spec_name}] FAILED: {exc}\n")
            spec_results.append({'Specification': spec_name, 'Error': str(exc)})

    return pd.DataFrame(spec_results)


def test_monotonicity(df, group_means):
    """
    Part (c): Is there a monotonic Q1 < Q2 < Q3 < Q4 ordering in mean SoNR?
    Also reports Spearman rank correlation between quartile rank and mean SoNR.
    """
    print("\n" + "=" * 60)
    print("PART (c): MONOTONICITY TEST")
    print("=" * 60)
    print("  Expected (hypothesis): Q1 lowest SoNR -> Q4 highest SoNR\n")

    means = group_means[['RRR_Quartile', 'Mean_SoNR']].copy()
    means['Q_num'] = means['RRR_Quartile'].str.extract(r'(\d)').astype(int)
    means = means.sort_values('Q_num')

    sonr_vals = means['Mean_SoNR'].values
    q_labels = means['RRR_Quartile'].values

    # Check each consecutive step
    print("  Consecutive steps:")
    all_increasing = True
    for i in range(len(sonr_vals) - 1):
        direction = "UP" if sonr_vals[i+1] > sonr_vals[i] else "DOWN"
        if direction == "DOWN":
            all_increasing = False
        delta = sonr_vals[i+1] - sonr_vals[i]
        print(f"    {q_labels[i]} -> {q_labels[i+1]}: {delta:+.4f}  [{direction}]")

    print(f"\n  Strictly monotonic (Q1 < Q2 < Q3 < Q4)? {'YES' if all_increasing else 'NO'}")

    # Spearman rank correlation: quartile number vs mean SoNR
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(means['Q_num'].values, sonr_vals)
    print(f"  Spearman rho (quartile rank vs mean SoNR): {rho:.4f}  (p={p_rho:.4f})")

    # Also Spearman on firm-quarter level
    work = df[['SONR', 'RRR_Q']].dropna()
    rho_micro, p_micro = spearmanr(work['RRR_Q'].values, work['SONR'].values)
    print(f"  Spearman rho (firm-quarter level): {rho_micro:.4f}  (p={p_micro:.4f})")

    monotone_result = pd.DataFrame([{
        'Strictly_Monotonic': all_increasing,
        'Spearman_Rho_Group_Means': rho,
        'p_value_group': p_rho,
        'Spearman_Rho_MicroLevel': rho_micro,
        'p_value_micro': p_micro,
    }])

    return monotone_result


def save_results(group_means, reg_results, monotone_result):
    """Save all output tables to a single Excel workbook."""
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        group_means.to_excel(writer, sheet_name='Group_Means', index=False)
        reg_results.to_excel(writer, sheet_name='Panel_Regressions', index=False)
        monotone_result.to_excel(writer, sheet_name='Monotonicity', index=False)
    print(f"\n  Results saved to: {OUTPUT_FILE}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SoNR BY RRR QUARTILE — EXPLORATORY ANALYSIS")
    print("=" * 60)

    # Load and build panel (replicates analysis_v2.py Phase 1)
    df_revenue = load_revenue_data()
    df_fun = load_fundamentals()
    df_panel = build_panel(df_revenue, df_fun)
    df_panel = compute_analysis_vars(df_panel)

    # Assign RRR quartiles (cross-sectional, within each quarter)
    df_panel = assign_rrr_quartiles(df_panel)

    # Run analyses
    group_means = report_group_means(df_panel)
    reg_results = run_panel_regressions(df_panel)
    monotone_result = test_monotonicity(df_panel, group_means)

    save_results(group_means, reg_results, monotone_result)

    print("\nDone.")
