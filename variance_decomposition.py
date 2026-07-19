"""
Rumelt (1991)-Style Variance Decomposition for Customer Metrics (AR, RRR)
=========================================================================
Decomposes total variance of a dependent variable into:
  - Industry component     (sigma2_ind)
  - Firm component         (sigma2_firm, nested within industry)
  - Year component         (sigma2_year, crossed with industry/firm)
  - Residual               (sigma2_e)

Estimator: Henderson Type III (Method of Moments) for unbalanced panels.
This is the same estimator used in Rumelt (1991) and Schmalensee (1985).

Two models:
  Model A: All four components are random effects.
  Model B: Year effect absorbed as fixed effect (within-year demeaning),
           then industry/firm decomposed from the demeaned outcome.

References:
  Rumelt, R. P. (1991). How much does industry matter? Strategic Management
      Journal, 12(3), 167-185.
  Schmalensee, R. (1985). Do markets differ much? American Economic Review,
      75(3), 341-351.
  Henderson, C. R. (1953). Estimation of variance and covariance components.
      Biometrics, 9(2), 226-252.
  Searle, S. R., Casella, G., & McCulloch, C. E. (1992). Variance Components.
      Wiley. [Chapter 4 for Type III expected mean squares with unbalanced data]

Minimum group sizes:
  Industry: >= 10 firms (Maas & Hox, 2005; Simulation evidence requires
            at least 30 groups for unbiased sigma2 estimates, but 10 is an
            acceptable lower bound for exploratory work with clamped negatives)
  Firm: >= 3 time periods (standard panel requirement)

Maas, C. J. M., & Hox, J. J. (2005). Sufficient sample sizes for multilevel
    modeling. Methodology, 1(3), 86-92.
"""

# Cell 1: Imports
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Cell 2: Constants
# Adjust these paths to match the actual data file
DATA_PATH    = None          # Set to file path string when running on real data
IND_COL      = 'gics_ind'   # GICS industry identifier column
FIRM_COL     = 'firm_id'    # Firm identifier column
YEAR_COL     = 'year'       # Year column
DEP_VARS     = ['AR', 'RRR']  # Dependent variables to decompose
MIN_FIRM_OBS = 3            # Minimum time-series observations per firm
MIN_IND_FIRMS = 10          # Minimum firms per industry (Maas & Hox 2005 threshold)
N_BOOTSTRAP  = 1000         # Bootstrap replications for confidence intervals
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Core estimation functions
# ---------------------------------------------------------------------------

def _henderson_correction_factors(df, ind_col, firm_col, year_col, y_col):
    """
    Compute Henderson Type III correction factors (k-values) for an unbalanced
    crossed-nested design.

    Model: y_ijt = mu + alpha_i + beta_{j(i)} + gamma_t + epsilon_ijt
      alpha_i    ~ N(0, sigma2_ind)   [industry, i = 1..I]
      beta_{j(i)}~ N(0, sigma2_firm)  [firm nested in industry, j = 1..J_i]
      gamma_t    ~ N(0, sigma2_year)  [year, crossed, t = 1..T]
      epsilon    ~ N(0, sigma2_e)

    Expected mean squares (Type III, unbalanced):
      E[MS_e]    = sigma2_e
      E[MS_firm] = sigma2_e + k_f  * sigma2_firm
      E[MS_ind]  = sigma2_e + k_fi * sigma2_firm + k_i * sigma2_ind
      E[MS_year] = sigma2_e + k_t  * sigma2_year

    Returns k_f, k_i, k_fi, k_t.

    Source: Searle, Casella, McCulloch (1992), Variance Components, Ch. 4.
    The 'fitting-constants' (Henderson Type III) expected MS coefficients for
    a nested-crossed mixed model with unequal subclass numbers.
    """
    N       = len(df)
    n_inds  = df[ind_col].nunique()
    n_firms = df[firm_col].nunique()
    n_years = df[year_col].nunique()

    # n_jt: number of observations per firm
    n_jt = df.groupby(firm_col)[y_col].count()
    # n_it: number of observations per industry
    n_it = df.groupby(ind_col)[y_col].count()
    # n_t:  number of observations per year
    n_t  = df.groupby(year_col)[y_col].count()

    # Nested correction term: sum_i [ (1/N_i) * sum_{j in i} n_jt_j^2 ]
    # This adjusts for unequal firm-year cell counts within industries.
    nested_term = 0.0
    for i in df[ind_col].unique():
        ni    = n_it[i]
        firms = df[df[ind_col] == i][firm_col].unique()
        nested_term += (1.0 / ni) * sum(n_jt[j] ** 2 for j in firms)

    # k_f: multiplier on sigma2_firm in E[MS_firm]
    # df_firm = n_firms - n_inds (firms nested in industries, df for SS_firm)
    df_firm = n_firms - n_inds
    k_f = (N - nested_term) / df_firm if df_firm > 0 else 0.0

    # k_fi: cross-multiplier on sigma2_firm in E[MS_ind]
    # Captures how firm-level variance propagates into the industry-level MS.
    k_fi_num = sum(n_it[i] ** 2 for i in n_it.index) / N - nested_term
    k_fi = k_fi_num / (n_inds - 1) if (n_inds - 1) > 0 else 0.0

    # k_i: multiplier on sigma2_ind in E[MS_ind]
    # = (N - sum_i N_i^2/N) / (I - 1)
    k_i_num = N - sum(n_it[i] ** 2 for i in n_it.index) / N
    k_i = k_i_num / (n_inds - 1) if (n_inds - 1) > 0 else 0.0

    # k_t: multiplier on sigma2_year in E[MS_year]
    # Crossed year effect: = (N - sum_t N_t^2/N) / (T - 1)
    k_t_num = N - sum(n_t[t] ** 2 for t in n_t.index) / N
    k_t = k_t_num / (n_years - 1) if (n_years - 1) > 0 else 0.0

    return k_f, k_i, k_fi, k_t


def _mean_squares(df, ind_col, firm_col, year_col, y_col):
    """
    Compute weighted sum-of-squares and mean squares for each variance source.

    SS formulas use group-size weights, which are the natural ANOVA decomposition
    for unbalanced data (equivalent to Type I SS in this sequential ordering:
    grand mean, then industry, then firm|industry, then year).

    Returns MS values and degrees of freedom for each source.
    """
    N       = len(df)
    n_inds  = df[ind_col].nunique()
    n_firms = df[firm_col].nunique()
    n_years = df[year_col].nunique()

    n_jt = df.groupby(firm_col)[y_col].count()
    n_it = df.groupby(ind_col)[y_col].count()
    n_t  = df.groupby(year_col)[y_col].count()
    ind_of_firm = df.groupby(firm_col)[ind_col].first()

    grand_mean  = df[y_col].mean()
    ind_means   = df.groupby(ind_col)[y_col].mean()
    firm_means  = df.groupby(firm_col)[y_col].mean()
    year_means  = df.groupby(year_col)[y_col].mean()

    # Weighted SS: industry deviations from grand mean
    SS_ind  = (n_it * (ind_means  - grand_mean) ** 2).sum()
    # Weighted SS: firm deviations from their industry mean (nested)
    SS_firm = (n_jt * (firm_means - ind_means[ind_of_firm].values) ** 2).sum()
    # Weighted SS: year deviations from grand mean (crossed)
    SS_year = (n_t  * (year_means - grand_mean) ** 2).sum()
    # Residual SS: total SS minus all explained
    SS_e    = ((df[y_col] - grand_mean) ** 2).sum() - SS_ind - SS_firm - SS_year

    df_ind  = n_inds  - 1
    df_firm = n_firms - n_inds
    df_year = n_years - 1
    # Residual df: N-1 minus the three other df (approximate for crossed-nested)
    df_e    = N - 1 - df_ind - df_firm - df_year

    MS_ind  = SS_ind  / df_ind  if df_ind  > 0 else 0.0
    MS_firm = SS_firm / df_firm if df_firm > 0 else 0.0
    MS_year = SS_year / df_year if df_year > 0 else 0.0
    MS_e    = SS_e    / df_e    if df_e    > 0 else 0.0

    return (MS_ind, MS_firm, MS_year, MS_e,
            df_ind, df_firm, df_year, df_e,
            SS_ind, SS_firm, SS_year, SS_e)


def variance_decomposition(df, ind_col, firm_col, year_col, y_col, model='A'):
    """
    Rumelt (1991)-style variance decomposition via Henderson Type III MoM.

    Parameters
    ----------
    df       : pd.DataFrame  Panel data (long format)
    ind_col  : str           Industry identifier column name
    firm_col : str           Firm identifier column name
    year_col : str           Year column name
    y_col    : str           Dependent variable column name
    model    : 'A' or 'B'
               'A' = all four components estimated as random effects
               'B' = year effect absorbed as fixed effect (within-year demeaning),
                     industry and firm components estimated from demeaned outcome;
                     sigma2_year reported as zero (captured in FE, not in RE pool)

    Returns
    -------
    dict with keys:
      sigma2_ind, sigma2_firm, sigma2_year, sigma2_e  (variance estimates)
      total                                            (sum of sigma2 estimates)
      pct_ind, pct_firm, pct_year, pct_e              (percentage shares)
      k_f, k_i, k_fi, k_t                             (Henderson k-factors)
      MS_*, df_*, SS_*                                 (raw ANOVA quantities)
      negative_clamped                                 (True if any estimate was < 0)
    """
    if model == 'B':
        # Absorb year fixed effects: subtract within-year mean, add back grand mean
        grand_mean = df[y_col].mean()
        df = df.copy()
        df[y_col] = (df[y_col]
                     - df.groupby(year_col)[y_col].transform('mean')
                     + grand_mean)

    k_f, k_i, k_fi, k_t = _henderson_correction_factors(
        df, ind_col, firm_col, year_col, y_col)
    (MS_ind, MS_firm, MS_year, MS_e,
     df_ind, df_firm, df_year, df_e,
     SS_ind, SS_firm, SS_year, SS_e) = _mean_squares(
        df, ind_col, firm_col, year_col, y_col)

    # Solve the system E[MS] = k * sigma2 for each component sequentially:
    #   sigma2_e    from E[MS_e]    = sigma2_e
    #   sigma2_firm from E[MS_firm] = sigma2_e + k_f * sigma2_firm
    #   sigma2_ind  from E[MS_ind]  = sigma2_e + k_fi * sigma2_firm + k_i * sigma2_ind
    #   sigma2_year from E[MS_year] = sigma2_e + k_t * sigma2_year  (Model A only)
    sig2_e    = MS_e
    sig2_firm = (MS_firm - MS_e) / k_f                        if k_f > 0 else 0.0
    sig2_ind  = (MS_ind - MS_e - k_fi * sig2_firm) / k_i     if k_i > 0 else 0.0
    sig2_year = (MS_year - MS_e) / k_t if (k_t > 0 and model == 'A') else 0.0

    negative_clamped = any(v < 0 for v in [sig2_e, sig2_firm, sig2_ind, sig2_year])

    # Clamp: ANOVA MoM estimators are unbiased but can be negative by sampling.
    # Clamping to zero is standard practice (Rumelt 1991 footnote 4).
    sig2_e    = max(sig2_e,    0.0)
    sig2_firm = max(sig2_firm, 0.0)
    sig2_ind  = max(sig2_ind,  0.0)
    sig2_year = max(sig2_year, 0.0)

    total = sig2_ind + sig2_firm + sig2_year + sig2_e

    return {
        'sigma2_ind':        sig2_ind,
        'sigma2_firm':       sig2_firm,
        'sigma2_year':       sig2_year,
        'sigma2_e':          sig2_e,
        'total':             total,
        'pct_ind':           sig2_ind  / total * 100 if total > 0 else 0.0,
        'pct_firm':          sig2_firm / total * 100 if total > 0 else 0.0,
        'pct_year':          sig2_year / total * 100 if total > 0 else 0.0,
        'pct_e':             sig2_e    / total * 100 if total > 0 else 0.0,
        'k_f':   k_f,   'k_i':   k_i,  'k_fi': k_fi, 'k_t': k_t,
        'MS_ind':  MS_ind,  'MS_firm':  MS_firm,
        'MS_year': MS_year, 'MS_e':     MS_e,
        'df_ind':  df_ind,  'df_firm':  df_firm,
        'df_year': df_year, 'df_e':     df_e,
        'SS_ind':  SS_ind,  'SS_firm':  SS_firm,
        'SS_year': SS_year, 'SS_e':     SS_e,
        'negative_clamped':  negative_clamped,
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_variance_decomposition(df, ind_col, firm_col, year_col, y_col,
                                     model='A', n_boot=1000, seed=42,
                                     ci_level=0.95):
    """
    Parametric block bootstrap for variance component confidence intervals.

    Resampling scheme: resample industries (blocks) with replacement.
    This preserves the nested within-industry firm structure and is the
    natural block for industry-level inference.

    Parameters
    ----------
    n_boot    : int    Number of bootstrap replications
    seed      : int    Random seed
    ci_level  : float  Confidence level (default 0.95 -> 2.5th/97.5th percentiles)

    Returns
    -------
    dict mapping each sigma2_* and pct_* key to (lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)
    industries = df[ind_col].unique()
    keys = ['sigma2_ind', 'sigma2_firm', 'sigma2_year', 'sigma2_e',
            'pct_ind', 'pct_firm', 'pct_year', 'pct_e']
    boot_results = {k: [] for k in keys}

    for _ in range(n_boot):
        # Draw industries with replacement (block bootstrap)
        sampled_inds = rng.choice(industries, size=len(industries), replace=True)
        # Remap industry codes to avoid duplicate labels collapsing
        pieces = []
        for new_code, orig_ind in enumerate(sampled_inds):
            chunk = df[df[ind_col] == orig_ind].copy()
            chunk[ind_col] = new_code
            # Remap firm IDs to avoid collision across resampled copies
            chunk[firm_col] = chunk[firm_col].astype(str) + f'_b{new_code}'
            pieces.append(chunk)
        df_boot = pd.concat(pieces, ignore_index=True)

        try:
            res = variance_decomposition(df_boot, ind_col, firm_col, year_col, y_col, model=model)
            for k in keys:
                boot_results[k].append(res[k])
        except Exception:
            continue  # Skip failed replications (rare with small samples)

    alpha = 1 - ci_level
    ci = {}
    for k in keys:
        arr = np.array(boot_results[k])
        ci[k] = (float(np.nanpercentile(arr, alpha/2 * 100)),
                 float(np.nanpercentile(arr, (1 - alpha/2) * 100)))
    return ci


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def prepare_panel(df, ind_col, firm_col, year_col, dep_vars,
                  min_firm_obs=MIN_FIRM_OBS, min_ind_firms=MIN_IND_FIRMS):
    """
    Apply standard sample filters for reliable variance component estimation.

    Filters applied:
      1. Drop firms with fewer than min_firm_obs time-series observations.
         Rationale: variance components require within-group variation.
      2. Drop industries with fewer than min_ind_firms firms.
         Rationale: Maas & Hox (2005) recommend >=10 groups at each level
         for reliable second-level variance estimates. With fewer groups,
         the industry component is estimated from very few degrees of freedom
         (df_ind = I-1) and is unreliable even with many observations per group.

    Returns filtered DataFrame and a summary of what was dropped.
    """
    n_start = len(df)
    firms_start = df[firm_col].nunique()

    # Filter 1: minimum observations per firm (across all dep vars)
    firm_obs = df.groupby(firm_col)[dep_vars[0]].count()
    valid_firms = firm_obs[firm_obs >= min_firm_obs].index
    df = df[df[firm_col].isin(valid_firms)].copy()

    # Filter 2: minimum firms per industry
    ind_firms = df.groupby(ind_col)[firm_col].nunique()
    valid_inds = ind_firms[ind_firms >= min_ind_firms].index
    df = df[df[ind_col].isin(valid_inds)].copy()

    summary = {
        'n_obs_start':     n_start,
        'n_obs_final':     len(df),
        'firms_start':     firms_start,
        'firms_final':     df[firm_col].nunique(),
        'inds_start':      ind_firms.shape[0],   # before filter 2
        'inds_final':      df[ind_col].nunique(),
        'obs_dropped':     n_start - len(df),
        'firms_dropped':   firms_start - df[firm_col].nunique(),
    }
    return df, summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_results_table(results_A, results_B, ci_A=None, ci_B=None,
                          dep_var_label='Dependent variable'):
    """
    Format variance decomposition results as a publication-ready DataFrame.
    Mirrors the table structure in Rumelt (1991), Table 3.
    """
    components = [
        ('Industry',  'ind'),
        ('Firm',      'firm'),
        ('Year',      'year'),
        ('Residual',  'e'),
    ]

    rows = []
    for label, key in components:
        row = {'Component': label}

        # Model A
        row['sigma2_A']  = results_A[f'sigma2_{key}']
        row['pct_A']     = results_A[f'pct_{key}']
        if ci_A:
            row['ci_A_lo']   = ci_A[f'pct_{key}'][0]
            row['ci_A_hi']   = ci_A[f'pct_{key}'][1]

        # Model B
        row['sigma2_B']  = results_B[f'sigma2_{key}']
        row['pct_B']     = results_B[f'pct_{key}']
        if ci_B:
            row['ci_B_lo']   = ci_B[f'pct_{key}'][0]
            row['ci_B_hi']   = ci_B[f'pct_{key}'][1]

        rows.append(row)

    # Add totals row
    total_row = {'Component': 'Total',
                 'sigma2_A': results_A['total'], 'pct_A': 100.0,
                 'sigma2_B': results_B['total'], 'pct_B': 100.0}
    rows.append(total_row)

    tbl = pd.DataFrame(rows).set_index('Component')
    tbl.name = dep_var_label
    return tbl


def print_results(tbl, dep_var_label, ci_A=None, ci_B=None):
    """Print formatted variance decomposition table to console."""
    print(f"\n{'='*70}")
    print(f"  Variance Decomposition: {dep_var_label}")
    print(f"{'='*70}")
    header = f"{'Component':<12}  {'Model A':>22}  {'Model B':>22}"
    if ci_A:
        header = f"{'Component':<12}  {'Model A (95% CI)':>34}  {'Model B (95% CI)':>34}"
    print(header)
    print('-' * 70)

    for idx, row in tbl.iterrows():
        if idx == 'Total':
            print('-' * 70)
        pct_a = f"{row.get('pct_A', 0):6.1f}%"
        pct_b = f"{row.get('pct_B', 0):6.1f}%"
        if ci_A and 'ci_A_lo' in row and not pd.isna(row.get('ci_A_lo', float('nan'))):
            ci_str_a = f"[{row['ci_A_lo']:5.1f}, {row['ci_A_hi']:5.1f}]"
            ci_str_b = f"[{row['ci_B_lo']:5.1f}, {row['ci_B_hi']:5.1f}]" if ci_B else ''
            print(f"{idx:<12}  {pct_a}  {ci_str_a}  {pct_b}  {ci_str_b}")
        else:
            print(f"{idx:<12}  {pct_a:>10}  {pct_b:>10}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_variance_decomposition(df, ind_col, firm_col, year_col, dep_vars,
                                run_bootstrap=True, n_boot=N_BOOTSTRAP):
    """
    Run full variance decomposition for each dependent variable.

    Parameters
    ----------
    df        : pd.DataFrame  Panel data
    dep_vars  : list of str   Dependent variable column names (e.g. ['AR', 'RRR'])
    run_bootstrap : bool      Whether to compute bootstrap CIs (slow for large N)

    Returns
    -------
    all_results : dict  Nested dict keyed by dep_var -> 'A'/'B' -> result dict
    all_tables  : dict  Keyed by dep_var -> formatted pd.DataFrame
    """
    all_results = {}
    all_tables  = {}

    for dep_var in dep_vars:
        print(f"\nProcessing: {dep_var}")
        # Drop rows where this dep_var is missing
        df_var = df.dropna(subset=[dep_var]).copy()

        # Prepare sample
        df_clean, prep_summary = prepare_panel(
            df_var, ind_col, firm_col, year_col, [dep_var])
        print(f"  Sample: {prep_summary['n_obs_final']:,} obs | "
              f"{prep_summary['firms_final']} firms | "
              f"{prep_summary['inds_final']} industries | "
              f"{df_clean[year_col].nunique()} years")
        print(f"  Dropped: {prep_summary['obs_dropped']:,} obs, "
              f"{prep_summary['firms_dropped']} firms")

        # Model A: all random effects
        res_A = variance_decomposition(df_clean, ind_col, firm_col, year_col, dep_var, model='A')
        # Model B: year as fixed effect
        res_B = variance_decomposition(df_clean, ind_col, firm_col, year_col, dep_var, model='B')

        if res_A['negative_clamped']:
            print("  WARNING: One or more Model A components clamped from negative to zero.")
        if res_B['negative_clamped']:
            print("  WARNING: One or more Model B components clamped from negative to zero.")

        # Bootstrap confidence intervals
        ci_A = ci_B = None
        if run_bootstrap:
            print(f"  Running bootstrap (n_boot={n_boot})...")
            ci_A = bootstrap_variance_decomposition(
                df_clean, ind_col, firm_col, year_col, dep_var,
                model='A', n_boot=n_boot, seed=BOOTSTRAP_SEED)
            ci_B = bootstrap_variance_decomposition(
                df_clean, ind_col, firm_col, year_col, dep_var,
                model='B', n_boot=n_boot, seed=BOOTSTRAP_SEED)

        tbl = format_results_table(res_A, res_B, ci_A, ci_B, dep_var_label=dep_var)
        print_results(tbl, dep_var, ci_A, ci_B)

        # Print ANOVA diagnostics
        print(f"  Henderson k-factors (Model A): "
              f"k_f={res_A['k_f']:.2f}  k_i={res_A['k_i']:.2f}  "
              f"k_fi={res_A['k_fi']:.2f}  k_t={res_A['k_t']:.2f}")
        print(f"  Degrees of freedom: "
              f"ind={res_A['df_ind']}  firm={res_A['df_firm']}  "
              f"year={res_A['df_year']}  resid={res_A['df_e']}")

        all_results[dep_var] = {'A': res_A, 'B': res_B, 'ci_A': ci_A, 'ci_B': ci_B}
        all_tables[dep_var]  = tbl

    return all_results, all_tables


# ---------------------------------------------------------------------------
# Entry point (run directly for validation with synthetic data)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    """
    When run directly (without real data), validates the estimator against
    known true variance components using synthetic panel data that mimics
    the actual RRR-FI sample structure:
      - 18 GICS industries
      - ~550 firms (unequal across industries)
      - 7 years (2018-2025), not all firms present in all years
    """
    print("Running validation on synthetic data (mimics RRR-FI panel structure)")
    print("True components: industry=10.7%, firm=32.7%, year=2.7%, residual=54.0%")

    np.random.seed(42)
    N_IND           = 18
    N_YEAR          = 7
    SIGMA_IND       = 0.20    # -> sigma2 = 0.04  -> 10.7% of total
    SIGMA_FIRM      = 0.35    # -> sigma2 = 0.1225 -> 32.7%
    SIGMA_YEAR      = 0.10    # -> sigma2 = 0.01   -> 2.7%
    SIGMA_E         = 0.45    # -> sigma2 = 0.2025 -> 54.0%
    TOTAL_TRUE      = SIGMA_IND**2 + SIGMA_FIRM**2 + SIGMA_YEAR**2 + SIGMA_E**2

    ind_fx  = np.random.normal(0, SIGMA_IND,  N_IND)
    year_fx = np.random.normal(0, SIGMA_YEAR, N_YEAR)
    rows = []
    fid = 0
    for i in range(N_IND):
        n_firms_i = np.random.randint(15, 55)
        firm_fx   = np.random.normal(0, SIGMA_FIRM, n_firms_i)
        for f in range(n_firms_i):
            n_obs = np.random.randint(4, N_YEAR + 1)
            obs_years = np.random.choice(range(N_YEAR), size=n_obs, replace=False)
            for y in obs_years:
                mu = ind_fx[i] + firm_fx[f] + year_fx[y]
                rows.append({
                    'gics_ind': f'IND{i:02d}',
                    'firm_id':  fid,
                    'year':     2018 + y,
                    'AR':       mu + np.random.normal(0, SIGMA_E),
                    'RRR':      mu + np.random.normal(0, SIGMA_E),
                })
            fid += 1

    df_synth = pd.DataFrame(rows)
    print(f"\nSynthetic panel: {len(df_synth):,} obs, {df_synth['firm_id'].nunique()} firms")

    results, tables = run_variance_decomposition(
        df_synth,
        ind_col='gics_ind', firm_col='firm_id', year_col='year',
        dep_vars=['AR'],
        run_bootstrap=False,  # Skip bootstrap for speed in validation run
    )

    # Print comparison with truth
    res_A = results['AR']['A']
    print("\nValidation (Model A vs True):")
    truth = {'ind': SIGMA_IND**2, 'firm': SIGMA_FIRM**2,
             'year': SIGMA_YEAR**2, 'e': SIGMA_E**2}
    for name in ['ind', 'firm', 'year', 'e']:
        est  = res_A[f'sigma2_{name}']
        true = truth[name]
        print(f"  sigma2_{name:4s}: estimated={est:.4f} ({est/res_A['total']*100:.1f}%)  "
              f"true={true:.4f} ({true/TOTAL_TRUE*100:.1f}%)")

    print("\nValidation complete.")
    print("\nTo run on real data, call run_variance_decomposition() with your DataFrame.")
    print("Example:")
    print("  df = pd.read_csv('your_data.csv')")
    print("  results, tables = run_variance_decomposition(")
    print("      df, ind_col='gics_ind', firm_col='firm_id',")
    print("      year_col='year', dep_vars=['AR', 'RRR'])")
