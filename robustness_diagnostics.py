"""
robustness_diagnostics.py -- RRR Financial Implications
=========================================================
Nine reviewer-style robustness checks on the corrected-timing (2-month
formation gap, 3-month hold) RRR long-short portfolio result:

  1. Concentration diagnostics (effective N, exclude-top-K names)
  2. Block-bootstrap CI on the FF3/FF5 long-short alpha + subperiod stability
  3. Consumer-Discretionary-only alpha
  4. RRR autocorrelation reconciliation (firm-level AR(1) vs transition matrix)
  5. Full-quartile monotonicity (Q1, Q2, Q3, Q4 alphas, not just the spread)
  6. Dividend-yield gap by RRR quartile (via yfinance)
  7. Net-of-cost alpha (turnover x Novy-Marx & Velikov 2016 trading costs)
  8. H2a/H2b asymmetry: downside-risk bootstrap + up/down-market interaction
  9. Agarwal et al. (2021) / Baker et al. (2023) differentiation (repeat/new
     revenue share nesting test)

Design principle: this file imports data-loading and panel-construction
functions from analysis_v2.py (build_holding_panel, build_portfolio_returns,
run_factor_regressions, value_weighted_market, safe_quartile, safe_tercile,
newey_west_ols) rather than reimplementing the verified timing convention
(FORM_LAG_MONTHS=2, HOLD_MONTHS=3). analysis_v2.py and the frozen
2025-06-04b-TK-RRR_financialimplications-main.py are NOT modified.

df_filtered (the sector-filtered, industry-adjusted firm-quarter panel) is
loaded from the already-persisted output/canonical_panel_20260719.xlsx
snapshot rather than by re-calling analysis_v2.phase1_load_and_diagnose(),
which has side effects (it overwrites several shared files in output/). This
keeps this script's I/O confined to files it creates itself, which matters
because other agents are working in analysis_v2.py's output/ directory in
parallel. The snapshot was verified (see validation run) to reproduce the
confirmed headline numbers exactly: FF3 alpha = 1.9198%/mo (t=2.9385), FF5
alpha = 2.1547%/mo (t=3.2868), on the ADJ RRR quartile sort, N=88 months.

Reproducibility: a single seeded numpy Generator (RANDOM_SEED = today's date,
20260719) is created once in main() and threaded through every bootstrap
routine, so re-running this script reproduces identical bootstrap figures.

Run:
    python robustness_diagnostics.py
"""

import os
import sys
import json
import time
import warnings
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as sps

warnings.filterwarnings('ignore')

# =============================================================================
# PATH CONSTANTS
# =============================================================================
CODE_DIR = (
    r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)"
    r"\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM"
)
OUTPUT_DIR = os.path.join(CODE_DIR, "output")
CANONICAL_PANEL_PATH = os.path.join(OUTPUT_DIR, "canonical_panel_20260719.xlsx")
DIAG_XLSX_PATH = os.path.join(OUTPUT_DIR, "robustness_diagnostics.xlsx")
DIAG_MANIFEST_PATH = os.path.join(OUTPUT_DIR, "robustness_diagnostics_manifest.json")

sys.path.insert(0, CODE_DIR)
import analysis_v2 as av2  # noqa: E402 -- import must follow sys.path setup

# =============================================================================
# ANALYSIS CONSTANTS
# =============================================================================
RANDOM_SEED = 20260719          # today's date, per project convention (seed set once, here)
N_BOOTSTRAP = 5000              # bootstrap replications for every block-bootstrap routine
PRIMARY_BLOCK_MONTHS = 6        # primary circular-block-bootstrap block length
SENSITIVITY_BLOCK_MONTHS = [3, 12]  # alternate block lengths reported alongside the primary

# Task 7: round-trip trading-cost assumption. Source: Novy-Marx, R., and
# Velikov, M. (2016), "A Taxonomy of Anomalies and Their Trading Costs,"
# Review of Financial Studies, 29(1), 104-147. Figure 1 ("Average Round Trip
# Effective Spread, 2,000 Largest Firms, by Decade") plots the Hasbrouck
# (2009) Gibbs-estimated median round-trip effective spread against market-
# capitalization rank (1 = largest of the ~2,000 largest US firms), for each
# decade 1960s-2000s. Reading the "2000s" line (the most recent, and lowest-
# cost, decade in their 1963-2013 sample): roughly 0.3-0.4% at rank ~1-100,
# rising to roughly 0.5% by rank ~400-600, and to roughly 1.0% at the
# rank-2000 boundary of their "largest firms" universe. 50bp is the chosen
# point estimate: the "2000s" line at approximately rank 400-600, a
# representative point for "large, liquid" (not just the single largest
# handful of) stocks. This is deliberately NOT the paper's headline pooled
# average ("round trip transaction costs for typical value-weighted
# strategies average in excess of 50 basis points," their abstract, mixing
# in much smaller/less liquid anomaly names) -- Figure 1's size-specific
# reading is the appropriate object for a large-cap-skewed sample. It is
# also almost certainly conservative (an upper bound) for this project's
# actual 2017-2024 sample period, since the paper's own text notes a
# "general trend towards lower costs over time" through the end of their
# sample (2013), and market structure (decimalization, HFT market-making)
# has continued to compress spreads since. 25bp and 100bp are reported
# alongside as sensitivity bounds (the observed range across the "2000s"
# line from rank ~1 to rank ~2000).
NMV_ROUNDTRIP_COST_BPS_PRIMARY = 50.0
NMV_ROUNDTRIP_COST_BPS_SENSITIVITY = [25.0, 100.0]

# Task 6: yfinance history window. Panel formation dates run from
# approximately 2017-05 to 2024-11; trailing-12-month dividend windows need
# price/dividend history from about a year before the earliest formation
# date, so the pull starts well before that with margin on both ends.
YF_HISTORY_START = "2016-01-01"
YF_HISTORY_END = "2025-02-01"
YF_REQUEST_PAUSE_SEC = 0.15  # small delay between yfinance calls

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _hdr(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _p(v, ndp=3):
    """Format a p-value in brackets, exact (no significance stars), per project convention."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "[n/a]"
    floor = 10 ** (-ndp)
    if v < floor:
        return f"[<.{'0' * (ndp - 1)}1]"
    return f"[{v:.{ndp}f}]"


def _f(v, ndp=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    return round(float(v), ndp)


# =============================================================================
# DATA LOADING (read-only reuse of analysis_v2.py)
# =============================================================================

def load_base_data():
    """Load df_filtered from the persisted canonical-panel snapshot (read-only:
    avoids re-triggering phase1_load_and_diagnose()'s side-effect writes to
    shared files in output/ while other agents work there in parallel), then
    build the holding panel via analysis_v2.build_holding_panel -- the
    verified source of truth for portfolio timing. Not a reimplementation:
    every timing-relevant step is a direct call into analysis_v2.
    """
    _hdr("LOAD BASE DATA")

    print(f"  Reading canonical panel snapshot: {CANONICAL_PANEL_PATH}")
    df_filtered = pd.read_excel(CANONICAL_PANEL_PATH, sheet_name=0, engine="openpyxl")
    df_filtered["DATE"] = pd.to_datetime(df_filtered["DATE"])
    df_filtered = df_filtered.sort_values(["FIRM", "DATE"]).set_index(["FIRM", "DATE"])
    n_firms = df_filtered.index.get_level_values("FIRM").nunique()
    print(f"  df_filtered: {df_filtered.shape[0]} firm-quarters, {n_firms} firms")
    assert n_firms == av2.EXPECTED_N_FIRMS, (
        f"Canonical panel has {n_firms} firms, expected {av2.EXPECTED_N_FIRMS}. "
        f"The snapshot may be stale relative to analysis_v2.py; investigate before trusting results."
    )

    ff_factors = av2.phase1b_load_ff_factors()
    returns = av2.phase1c_load_monthly_returns()

    panel = av2.build_holding_panel(df_filtered, returns)

    # ADJ RRR quartile assignment computed ONCE here and reused by every task
    # below, so every task shares an identical quartile assignment.
    panel["ADJ_RRR_PCT"] = pd.to_numeric(panel["ADJ_RRR_PCT"], errors="coerce")
    panel["RRR_Q_ADJ"] = panel.groupby("QUARTER")["ADJ_RRR_PCT"].transform(av2.safe_quartile)

    market_ret = av2.value_weighted_market(panel)

    return df_filtered, returns, ff_factors, panel, market_ret


def baseline_portfolio_and_regressions(panel, ff_factors, market_ret):
    """The primary ADJ-RRR value-weighted quartile sort and its FF3/FF3+Mom/FF5
    factor regressions -- the object every other task compares against.
    """
    _hdr("BASELINE: ADJ RRR quartile sort, value-weighted (reproduces confirmed headline)")
    port = av2.build_portfolio_returns(panel, "RRR_Q_ADJ", market_ret)
    reg = av2.run_factor_regressions(port, ff_factors, "BASELINE ADJ RRR Q1-Q4 VW")
    return port, reg


# =============================================================================
# GENERIC UTILITIES (shared across tasks; not from analysis_v2.py)
# =============================================================================

def circular_block_bootstrap_indices(n_obs, block_length, n_replications, rng):
    """Generate n_replications resampled index paths of length n_obs via a
    CIRCULAR BLOCK bootstrap (Politis & Romano): draws whole blocks of
    `block_length` consecutive (wrapping) months with replacement, instead of
    resampling individual months i.i.d., so short-run time-series dependence
    within a block (e.g., quarter-to-quarter carryover in a return series) is
    preserved in every resampled path. Returns an (n_replications, n_obs)
    integer index array into the ORIGINAL time-ordered series.
    """
    n_blocks_needed = int(np.ceil(n_obs / block_length))
    out = np.empty((n_replications, n_blocks_needed * block_length), dtype=int)
    block_starts = rng.integers(0, n_obs, size=(n_replications, n_blocks_needed))
    for b in range(n_replications):
        idx = np.concatenate([
            (np.arange(s, s + block_length) % n_obs) for s in block_starts[b]
        ])
        out[b] = idx
    return out[:, :n_obs]


def block_bootstrap_alpha(y, X, block_length, n_boot, rng):
    """Block-bootstrap the OLS intercept (alpha) of y ~ const + X. y and X
    must share an index, already aligned and NaN-free, sorted by date. Each
    replicate resamples ROWS (preserving the joint y/X time-series pairing in
    every block) and refits by OLS; returns the array of B alpha draws.
    """
    n = len(y)
    idx_matrix = circular_block_bootstrap_indices(n, block_length, n_boot, rng)
    X_const = sm.add_constant(X, has_constant="add")
    y_arr = np.asarray(y, dtype=float)
    X_arr = np.asarray(X_const, dtype=float)
    const_col = list(X_const.columns).index("const")
    alphas = np.empty(n_boot)
    for b in range(n_boot):
        rows = idx_matrix[b]
        beta, *_ = np.linalg.lstsq(X_arr[rows], y_arr[rows], rcond=None)
        alphas[b] = beta[const_col]
    return alphas


def hhi_effective_n(weights):
    """Herfindahl-Hirschman Index and its inverse (effective number of names)
    for a vector of portfolio weights that sum to 1. Business definition:
    effective N answers "how many EQUALLY weighted names would produce the
    same concentration as this portfolio's actual weights"; effective N well
    below the raw name count signals concentration in a handful of names.
    """
    w = np.asarray(weights, dtype=float)
    w = w[~np.isnan(w)]
    if w.sum() <= 0 or len(w) == 0:
        return np.nan, np.nan
    w = w / w.sum()
    hhi = float((w ** 2).sum())
    eff_n = 1.0 / hhi if hhi > 0 else np.nan
    return hhi, eff_n


def leg_weights_by_quarter(panel_q, bucket_col, bucket_value):
    """One row per (QUARTER, FIRM) in the given bucket, with the normalized
    formation-market-cap weight within that quarter's bucket -- the SAME
    weighting rule analysis_v2.build_portfolio_returns applies (MCAP_FORM,
    normalized within (Date, bucket), which for this design is constant
    across a cohort's 3 holding months, so weighting at the QUARTER level is
    equivalent and avoids triple-counting the 3 HOLD_IDX rows per cohort).
    """
    sub = (
        panel_q.loc[panel_q[bucket_col] == bucket_value]
        .drop_duplicates(subset=["FIRM", "QUARTER"])
        .loc[:, ["FIRM", "QUARTER", "MCAP_FORM"]]
        .copy()
    )
    sub["MCAP_FORM"] = pd.to_numeric(sub["MCAP_FORM"], errors="coerce")
    sub = sub.dropna(subset=["MCAP_FORM"])
    sub["WEIGHT"] = sub["MCAP_FORM"] / sub.groupby("QUARTER")["MCAP_FORM"].transform("sum")
    return sub


def quarter_label(ts):
    return str(pd.Period(ts, freq="Q"))


# =============================================================================
# TASK 1: CONCENTRATION DIAGNOSTICS
# =============================================================================

def task1_concentration(panel, ff_factors, market_ret, baseline_reg):
    _hdr("TASK 1: CONCENTRATION DIAGNOSTICS")

    w_q1 = leg_weights_by_quarter(panel, "RRR_Q_ADJ", "Q1")
    w_q4 = leg_weights_by_quarter(panel, "RRR_Q_ADJ", "Q4")

    def _eff_n_table(w_df):
        rows = []
        for qtr, g in w_df.groupby("QUARTER"):
            hhi, eff_n = hhi_effective_n(g["WEIGHT"].values)
            top_name = g.loc[g["WEIGHT"].idxmax(), "FIRM"] if len(g) else None
            top_w = g["WEIGHT"].max() if len(g) else np.nan
            rows.append({
                "QUARTER": quarter_label(qtr), "N_NAMES": g["FIRM"].nunique(),
                "HHI": hhi, "EFFECTIVE_N": eff_n,
                "TOP_NAME": top_name, "TOP_WEIGHT": top_w,
            })
        return pd.DataFrame(rows).sort_values("QUARTER").reset_index(drop=True)

    tab_q1 = _eff_n_table(w_q1)
    tab_q4 = _eff_n_table(w_q4)

    print("\n  Q1 (long leg) effective N across quarters:")
    print(f"    mean={tab_q1['EFFECTIVE_N'].mean():.2f}  median={tab_q1['EFFECTIVE_N'].median():.2f}  "
          f"min={tab_q1['EFFECTIVE_N'].min():.2f}  max={tab_q1['EFFECTIVE_N'].max():.2f}  "
          f"(avg N names={tab_q1['N_NAMES'].mean():.1f})")
    print("  Q4 (short leg) effective N across quarters:")
    print(f"    mean={tab_q4['EFFECTIVE_N'].mean():.2f}  median={tab_q4['EFFECTIVE_N'].median():.2f}  "
          f"min={tab_q4['EFFECTIVE_N'].min():.2f}  max={tab_q4['EFFECTIVE_N'].max():.2f}  "
          f"(avg N names={tab_q4['N_NAMES'].mean():.1f})")

    # Two representative quarters, chosen without cherry-picking: earliest
    # and latest formation quarter in the sample.
    reps = pd.concat([tab_q1.iloc[[0, -1]].assign(LEG="Q1"), tab_q4.iloc[[0, -1]].assign(LEG="Q4")])
    print("\n  Representative quarters (first and last in sample):")
    for _, r in reps.iterrows():
        print(f"    {r['LEG']} {r['QUARTER']}: N_names={r['N_NAMES']}, effective_N={r['EFFECTIVE_N']:.2f}, "
              f"top name={r['TOP_NAME']} ({r['TOP_WEIGHT']:.1%} of leg)")

    # ---- Exclude top-K names from the LONG (Q1) leg only, per quarter ----
    q1_ranked = w_q1.copy()
    q1_ranked["RANK"] = q1_ranked.groupby("QUARTER")["MCAP_FORM"].rank(method="first", ascending=False)

    exclusion_results = {}
    exclusion_results["k=0 (baseline)"] = {
        "FF3_alpha_pct": baseline_reg["Q1-Q4_FF3"]["alpha"] * 100,
        "FF3_t": baseline_reg["Q1-Q4_FF3"]["alpha_t"],
        "FF3_p": baseline_reg["Q1-Q4_FF3"]["alpha_p"],
        "FF5_alpha_pct": baseline_reg["Q1-Q4_FF5"]["alpha"] * 100,
        "FF5_t": baseline_reg["Q1-Q4_FF5"]["alpha_t"],
        "FF5_p": baseline_reg["Q1-Q4_FF5"]["alpha_p"],
        "avg_pct_weight_excluded": 0.0,
    }

    print("\n  Long-short alpha after excluding the top-K formation-cap names from the LONG leg each quarter:")
    for k in [1, 2, 3]:
        drop_keys = q1_ranked.loc[q1_ranked["RANK"] <= k, ["FIRM", "QUARTER"]].copy()
        drop_keys["DROP_FLAG"] = True
        avg_pct_weight_excluded = (
            q1_ranked.loc[q1_ranked["RANK"] <= k]
            .groupby("QUARTER")["WEIGHT"].sum()
            .reindex(q1_ranked["QUARTER"].unique()).mean()
        )

        df_exk = panel.merge(drop_keys, on=["FIRM", "QUARTER"], how="left")
        df_exk["DROP_FLAG"] = df_exk["DROP_FLAG"].fillna(False)
        df_exk["RRR_Q_ADJ_EXK"] = df_exk["RRR_Q_ADJ"].where(
            ~((df_exk["RRR_Q_ADJ"] == "Q1") & df_exk["DROP_FLAG"]), np.nan
        )

        port_exk = av2.build_portfolio_returns(df_exk, "RRR_Q_ADJ_EXK", market_ret)
        reg_exk = av2.run_factor_regressions(port_exk, ff_factors, f"Exclude top-{k} from Q1")

        row = {
            "FF3_alpha_pct": reg_exk.get("Q1-Q4_FF3", {}).get("alpha", np.nan) * 100,
            "FF3_t": reg_exk.get("Q1-Q4_FF3", {}).get("alpha_t", np.nan),
            "FF3_p": reg_exk.get("Q1-Q4_FF3", {}).get("alpha_p", np.nan),
            "FF5_alpha_pct": reg_exk.get("Q1-Q4_FF5", {}).get("alpha", np.nan) * 100,
            "FF5_t": reg_exk.get("Q1-Q4_FF5", {}).get("alpha_t", np.nan),
            "FF5_p": reg_exk.get("Q1-Q4_FF5", {}).get("alpha_p", np.nan),
            "avg_pct_weight_excluded": avg_pct_weight_excluded * 100,
        }
        exclusion_results[f"k={k}"] = row
        print(f"    Exclude top {k}: FF3 alpha={row['FF3_alpha_pct']:.4f}%/mo t={row['FF3_t']:.3f} {_p(row['FF3_p'])}  |  "
              f"FF5 alpha={row['FF5_alpha_pct']:.4f}%/mo t={row['FF5_t']:.3f} {_p(row['FF5_p'])}  "
              f"(avg {row['avg_pct_weight_excluded']:.1f}% of Q1 weight removed)")

    exclusion_df = pd.DataFrame(exclusion_results).T.reset_index().rename(columns={"index": "Specification"})

    return {
        "eff_n_q1_by_quarter": tab_q1,
        "eff_n_q4_by_quarter": tab_q4,
        "eff_n_q1_summary": {
            "mean": tab_q1["EFFECTIVE_N"].mean(), "median": tab_q1["EFFECTIVE_N"].median(),
            "min": tab_q1["EFFECTIVE_N"].min(), "max": tab_q1["EFFECTIVE_N"].max(),
        },
        "eff_n_q4_summary": {
            "mean": tab_q4["EFFECTIVE_N"].mean(), "median": tab_q4["EFFECTIVE_N"].median(),
            "min": tab_q4["EFFECTIVE_N"].min(), "max": tab_q4["EFFECTIVE_N"].max(),
        },
        "exclusion_table": exclusion_df,
    }


# =============================================================================
# TASK 2: BOOTSTRAP CI + SUBPERIOD STABILITY
# =============================================================================

def task2_bootstrap_and_subperiod(baseline_port, ff_factors, rng):
    _hdr("TASK 2: BLOCK-BOOTSTRAP CI + SUBPERIOD STABILITY")

    port = baseline_port.copy()
    port.index = pd.to_datetime(port.index).to_period("M").to_timestamp("M")
    combined = port.join(ff_factors, how="inner").dropna(subset=["Q1-Q4"])
    y = combined["Q1-Q4"]  # long-short: already zero-cost, no RF subtraction

    specs = {
        "FF3": ["Mkt-RF", "SMB", "HML"],
        "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    }

    boot_results = {}
    print(f"\n  N months in estimation sample: {len(y)}")
    for spec_name, factor_cols in specs.items():
        X = combined[factor_cols]
        point_model = av2.newey_west_ols(y, X)
        point_alpha = point_model.params["const"]
        point_t = point_model.tvalues["const"]
        print(f"\n  {spec_name} point estimate: alpha={point_alpha*100:.4f}%/mo, NW t={point_t:.3f}")

        for block_len in [PRIMARY_BLOCK_MONTHS] + SENSITIVITY_BLOCK_MONTHS:
            alphas = block_bootstrap_alpha(y, X, block_len, N_BOOTSTRAP, rng)
            boot_se = alphas.std(ddof=1)
            ci_lo, ci_hi = np.percentile(alphas, [2.5, 97.5])
            boot_t = point_alpha / boot_se if boot_se > 0 else np.nan
            tag = "PRIMARY" if block_len == PRIMARY_BLOCK_MONTHS else "sensitivity"
            print(f"    block={block_len:>2}mo [{tag:<11}]: boot mean={alphas.mean()*100:.4f}%, "
                  f"boot SE={boot_se*100:.4f}%, 95% CI=[{ci_lo*100:.4f}%, {ci_hi*100:.4f}%], "
                  f"boot t={boot_t:.3f}, excludes 0={not (ci_lo <= 0 <= ci_hi)}")
            boot_results[f"{spec_name}_block{block_len}"] = {
                "spec": spec_name, "block_length_months": block_len,
                "point_alpha_pct": point_alpha * 100, "point_NW_t": point_t,
                "boot_mean_pct": alphas.mean() * 100, "boot_se_pct": boot_se * 100,
                "ci_lo_pct": ci_lo * 100, "ci_hi_pct": ci_hi * 100,
                "boot_t": boot_t, "ci_excludes_zero": bool(not (ci_lo <= 0 <= ci_hi)),
                "n_boot": N_BOOTSTRAP, "n_obs": len(y),
            }

    # ---- Subperiod stability: chronological first half vs second half ----
    print("\n  Subperiod stability (chronological first half vs second half):")
    n = len(y)
    mid = n // 2
    subperiod_results = {}
    for label, sl in [("first_half", slice(0, mid)), ("second_half", slice(mid, n))]:
        y_sub = y.iloc[sl]
        date_lo, date_hi = y_sub.index.min(), y_sub.index.max()
        for spec_name, factor_cols in specs.items():
            X_sub = combined[factor_cols].iloc[sl]
            model = av2.newey_west_ols(y_sub, X_sub)
            alpha = model.params["const"]
            t = model.tvalues["const"]
            pval = model.pvalues["const"]
            print(f"    {label:<12} ({date_lo:%Y-%m} to {date_hi:%Y-%m}, N={len(y_sub)}) {spec_name}: "
                  f"alpha={alpha*100:.4f}%/mo, t={t:.3f}, p={_p(pval)}")
            subperiod_results[f"{label}_{spec_name}"] = {
                "period": label, "spec": spec_name,
                "date_start": str(date_lo.date()), "date_end": str(date_hi.date()),
                "n_obs": len(y_sub), "alpha_pct": alpha * 100, "t": t, "p": pval,
            }

    return {
        "bootstrap": pd.DataFrame(boot_results).T.reset_index(drop=True),
        "subperiod": pd.DataFrame(subperiod_results).T.reset_index(drop=True),
    }


# =============================================================================
# TASK 3: CONSUMER-DISCRETIONARY-ONLY ALPHA
# =============================================================================

def task3_consumer_discretionary_only(panel, ff_factors):
    _hdr("TASK 3: CONSUMER-DISCRETIONARY-ONLY ALPHA")

    total_firms = panel["FIRM"].nunique()
    cd = panel.loc[panel["SECTOR"] == "Consumer Discretionary"].copy()
    cd_firms = cd["FIRM"].nunique()
    print(f"  Consumer Discretionary firms: {cd_firms} of {total_firms} "
          f"({cd_firms/total_firms:.1%} of the sample)")

    cd["ADJ_RRR_PCT"] = pd.to_numeric(cd["ADJ_RRR_PCT"], errors="coerce")
    # Re-quartile WITHIN the CD-only cross-section each quarter (not merely a
    # filter of the all-sector quartile assignment): ADJ_RRR_PCT for CD firms
    # already equals RRR_PCT minus the CD-sector-quarter mean (the sector
    # adjustment in analysis_v2.py groups by [DATE, SECTOR]), so requartiling
    # this column within the CD subsample is the correct CD-only analogue of
    # the paper's industry-adjusted signal.
    cd["RRR_Q_CD"] = cd.groupby("QUARTER")["ADJ_RRR_PCT"].transform(av2.safe_quartile)

    market_ret_cd = av2.value_weighted_market(cd)
    port_cd = av2.build_portfolio_returns(cd, "RRR_Q_CD", market_ret_cd)
    reg_cd = av2.run_factor_regressions(port_cd, ff_factors, "Consumer Discretionary only")

    avg_n_total = cd.dropna(subset=["RRR_Q_CD"]).groupby("QUARTER")["FIRM"].nunique().mean()
    avg_n_q1 = cd.loc[cd["RRR_Q_CD"] == "Q1"].groupby("QUARTER")["FIRM"].nunique().mean()
    avg_n_q4 = cd.loc[cd["RRR_Q_CD"] == "Q4"].groupby("QUARTER")["FIRM"].nunique().mean()

    print(f"  Avg firms/quarter in CD-only sort: {avg_n_total:.1f} total, "
          f"{avg_n_q1:.1f} in Q1, {avg_n_q4:.1f} in Q4")

    out = {"n_cd_firms": cd_firms, "n_total_firms": total_firms,
           "pct_of_sample": cd_firms / total_firms,
           "avg_n_per_quarter": avg_n_total, "avg_n_q1": avg_n_q1, "avg_n_q4": avg_n_q4}
    for spec in ["FF3", "FF3+Mom", "FF5"]:
        key = f"Q1-Q4_{spec}"
        if key in reg_cd:
            r = reg_cd[key]
            print(f"    {spec}: alpha={r['alpha']*100:.4f}%/mo, t={r['alpha_t']:.3f}, "
                  f"p={_p(r['alpha_p'])}, N={r['n_obs']:.0f}")
            out[f"{spec}_alpha_pct"] = r["alpha"] * 100
            out[f"{spec}_t"] = r["alpha_t"]
            out[f"{spec}_p"] = r["alpha_p"]
            out[f"{spec}_n_obs"] = r["n_obs"]
        else:
            print(f"    {spec}: not available (insufficient overlapping data)")

    return out


# =============================================================================
# TASK 4: RRR AUTOCORRELATION RECONCILIATION
# =============================================================================

def firm_level_ar1(df_filtered, col, lag_col):
    """Per-firm correlation of a signal with its own one-quarter lag, then
    reported across firms (mean, median). This is the "firm-level AR(1)
    autocorrelation" reading: how persistent is the signal WITHIN a given
    firm's own time series, averaged across firms (as opposed to a pooled
    panel correlation, which additionally reflects cross-firm dispersion).
    """
    autocorrs = []
    for firm in df_filtered.index.get_level_values("FIRM").unique():
        fd = df_filtered.xs(firm, level="FIRM")[[col, lag_col]].dropna()
        if len(fd) >= 3:
            corr = fd[col].corr(fd[lag_col])
            if pd.notna(corr):
                autocorrs.append(corr)
    return np.array(autocorrs)


def transition_matrix_diag(df_filtered, q_col):
    pdf = df_filtered[[q_col]].copy().reset_index().sort_values(["FIRM", "DATE"])
    pdf["Q_PREV"] = pdf.groupby("FIRM")[q_col].shift(1)
    valid = pdf.dropna(subset=[q_col, "Q_PREV"])
    trans = pd.crosstab(valid["Q_PREV"], valid[q_col], normalize="index")
    diag = {q: (trans.loc[q, q] if (q in trans.index and q in trans.columns) else np.nan)
            for q in ["Q1", "Q2", "Q3", "Q4"]}
    overall = (valid[q_col] == valid["Q_PREV"]).mean()
    return trans, diag, overall, len(valid)


def task4_autocorrelation_reconciliation(df_filtered):
    _hdr("TASK 4: RRR AUTOCORRELATION RECONCILIATION")

    results = {}
    print("\n  --- Firm-level AR(1) autocorrelation (per-firm corr(RRR_t, RRR_t-1), averaged across firms) ---")
    for label, col, lag_col in [
        ("Raw RRR", "RRR_PCT", "RRR_PCT_LAG1"),
        ("Adj RRR (industry x time)", "ADJ_RRR_PCT", "ADJ_RRR_PCT_LAG1"),
    ]:
        ac = firm_level_ar1(df_filtered, col, lag_col)
        mean_ac, median_ac = np.mean(ac), np.median(ac)
        pct_pos = np.mean(ac > 0) * 100
        print(f"    {label}: mean={mean_ac:.4f}, median={median_ac:.4f}, N_firms={len(ac)}, "
              f"pct_positive={pct_pos:.1f}%")
        results[label] = {"mean": mean_ac, "median": median_ac, "n_firms": len(ac), "pct_positive": pct_pos}

    # Secondary check: pooled panel correlation (single number across ALL
    # firm-quarter pairs pooled, as opposed to averaging per-firm
    # correlations). Reported for context, not as the primary figure -- it
    # additionally reflects cross-sectional (between-firm) persistence, which
    # the firm-level average nets out.
    print("\n  --- Secondary check: pooled panel correlation (all firm-quarter pairs pooled) ---")
    for label, col, lag_col in [
        ("Raw RRR", "RRR_PCT", "RRR_PCT_LAG1"),
        ("Adj RRR (industry x time)", "ADJ_RRR_PCT", "ADJ_RRR_PCT_LAG1"),
    ]:
        pooled = df_filtered[[col, lag_col]].dropna()
        pooled_corr = pooled[col].corr(pooled[lag_col])
        print(f"    {label}: pooled corr={pooled_corr:.4f}  (N={len(pooled)} firm-quarter pairs)")
        results[f"{label} (pooled)"] = {"pooled_corr": pooled_corr, "n_pairs": len(pooled)}

    print("\n  --- Cross-sectional quartile transition-matrix diagonal (quarter t to t+1 stay probabilities) ---")
    trans_tables = {}
    for label, q_col in [
        ("Raw RRR quartile", "RRR_Q_RAW_CHAR"),
        ("Adj RRR quartile", "RRR_Q_ADJ_CHAR"),
    ]:
        trans, diag, overall, n_obs = transition_matrix_diag(df_filtered, q_col)
        print(f"    {label} (N={n_obs} firm-quarter transitions, overall stay rate={overall:.1%}):")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            print(f"      {q} -> {q} (diagonal): {diag[q]:.1%}")
        results[f"{label} transition diag"] = {**diag, "overall_stay_rate": overall, "n_transitions": n_obs}
        trans_tables[label] = trans

    return {"summary": results, "transition_matrices": trans_tables}


# =============================================================================
# TASK 5: FULL-QUARTILE MONOTONICITY
# =============================================================================

def task5_quartile_monotonicity(baseline_reg):
    _hdr("TASK 5: FULL-QUARTILE MONOTONICITY (Q1, Q2, Q3, Q4, not just the spread)")

    rows = []
    for spec in ["FF3", "FF5"]:
        alphas = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            key = f"{q}_{spec}"
            r = baseline_reg.get(key, {})
            alpha_pct = r.get("alpha", np.nan) * 100
            t = r.get("alpha_t", np.nan)
            pval = r.get("alpha_p", np.nan)
            alphas.append(alpha_pct)
            rows.append({"Spec": spec, "Portfolio": q, "Alpha_pct": alpha_pct, "t": t, "p": pval})
            print(f"    {spec} {q}: alpha={alpha_pct:.4f}%/mo, t={t:.3f}, p={_p(pval)}")
        diffs = np.diff(alphas)
        is_monotonic = bool(np.all(diffs <= 0))
        print(f"    {spec}: point estimates {'ARE' if is_monotonic else 'are NOT'} monotonically "
              f"decreasing Q1->Q4 (diffs={[round(d,4) for d in diffs]})")
        rows.append({"Spec": spec, "Portfolio": "MONOTONIC_Q1_TO_Q4", "Alpha_pct": is_monotonic, "t": np.nan, "p": np.nan})

    return pd.DataFrame(rows)


# =============================================================================
# TASK 6: DIVIDEND-YIELD GAP BY RRR QUARTILE (via yfinance)
# =============================================================================

# Bloomberg exchange-code suffix -> Yahoo Finance ticker suffix.
_YF_EXCHANGE_SUFFIX = {
    "US": "",       # United States (NYSE/NASDAQ): no suffix
    "GR": ".DE",    # Germany (Xetra)
    "HK": ".HK",    # Hong Kong
    "CN": ".TO",    # Canada (Toronto) -- Bloomberg's "CN" is Canada, not China
    "SW": ".SW",    # Switzerland (SIX Swiss Exchange)
    "BZ": ".SA",    # Brazil (B3)
    "LN": ".L",     # London Stock Exchange
}
# Explicit overrides for tickers that don't follow the mechanical mapping.
# SAVEQ = Bloomberg's post-Chapter-11 delisted-ticker suffix for Spirit
# Airlines; Yahoo Finance has no "Q"-suffix convention, so the pre-bankruptcy
# ticker is used (Yahoo retains historical price/dividend data under it).
_YF_TICKER_OVERRIDES = {"SAVEQ US EQUITY": "SAVE"}


def build_yf_ticker_map(firms):
    mapping = {}
    for f in firms:
        if f in _YF_TICKER_OVERRIDES:
            mapping[f] = _YF_TICKER_OVERRIDES[f]
            continue
        parts = f.rsplit(" ", 2)
        if len(parts) == 3 and parts[2] == "EQUITY" and parts[1] in _YF_EXCHANGE_SUFFIX:
            ticker, exch = parts[0], parts[1]
            mapping[f] = ticker + _YF_EXCHANGE_SUFFIX[exch]
        else:
            mapping[f] = None
    return mapping


def fetch_yf_history(ticker_map, start, end, pause):
    """Fetch Close price + per-share Dividends from ONE yfinance history call
    per ticker (rather than separate .dividends and .history calls), so price
    and dividends always come from the same currency/unit source -- avoids a
    currency mismatch for the non-US tickers in this sample (HK/CN/SW/GR/BZ/LN)
    that would arise from dividing a local-currency dividend by a price from a
    different (possibly USD-normalized) source.
    """
    import yfinance as yf

    hist_data, failures = {}, {}
    n = len(ticker_map)
    for i, (firm, yf_ticker) in enumerate(ticker_map.items()):
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"    ... {i+1}/{n} tickers fetched")
        if yf_ticker is None:
            failures[firm] = "no Yahoo Finance ticker mapping (unrecognized exchange suffix)"
            continue
        try:
            t = yf.Ticker(yf_ticker)
            h = t.history(start=start, end=end, auto_adjust=False)
            if h is None or h.empty:
                failures[firm] = f"yfinance returned no price history for '{yf_ticker}'"
                continue
            h = h.copy()
            h.index = pd.to_datetime(h.index).tz_localize(None)
            hist_data[firm] = h[["Close", "Dividends"]]
        except Exception as e:
            failures[firm] = f"yfinance error for '{yf_ticker}': {type(e).__name__}: {e}"
        time.sleep(pause)
    return hist_data, failures


def trailing_yield_for_date(hist_df, as_of):
    """Trailing-12-month dividend yield as of `as_of`: sum of per-share
    dividends in the 365 days ending at as_of, divided by the last available
    Close price on or before as_of. Both from the same yfinance history call.
    """
    if hist_df is None or hist_df.empty:
        return np.nan, np.nan
    prior = hist_df.loc[hist_df.index <= as_of]
    if prior.empty:
        return np.nan, np.nan
    px = prior["Close"].iloc[-1]
    if px is None or pd.isna(px) or px <= 0:
        return np.nan, np.nan
    window_start = as_of - pd.Timedelta(days=365)
    trailing_div = hist_df.loc[(hist_df.index > window_start) & (hist_df.index <= as_of), "Dividends"].sum()
    return trailing_div / px, px


def task6_dividend_yield_gap(panel):
    _hdr("TASK 6: DIVIDEND-YIELD GAP BY RRR QUARTILE (yfinance)")

    firms = sorted(panel["FIRM"].unique())
    n_total = len(firms)
    ticker_map = build_yf_ticker_map(firms)
    n_unmapped = sum(v is None for v in ticker_map.values())
    print(f"  {n_total} sample firms; {n_total - n_unmapped} have a Yahoo Finance ticker mapping, "
          f"{n_unmapped} do not (unrecognized exchange suffix).")

    print(f"  Fetching yfinance price + dividend history ({YF_HISTORY_START} to {YF_HISTORY_END})...")
    t0 = time.time()
    hist_data, failures = fetch_yf_history(ticker_map, YF_HISTORY_START, YF_HISTORY_END, YF_REQUEST_PAUSE_SEC)
    print(f"  Done in {time.time()-t0:.1f}s. Successful fetches: {len(hist_data)}/{n_total}. "
          f"Failures: {len(failures)}/{n_total}.")

    # Firm-quarters: one row per (FIRM, QUARTER, RRR_Q_ADJ) with the
    # formation date used as the "as of" date for the trailing yield, so the
    # yield is measured contemporaneously with the same signal that drives
    # the portfolio sort (consistent with the rest of the paper's
    # characteristic tables, e.g. portfolio_characteristics.xlsx).
    fq = (panel.dropna(subset=["RRR_Q_ADJ"])
          .drop_duplicates(subset=["FIRM", "QUARTER"])
          .loc[:, ["FIRM", "QUARTER", "FORMATION_DATE", "RRR_Q_ADJ"]].copy())
    fq["FORMATION_DATE"] = pd.to_datetime(fq["FORMATION_DATE"])

    yields, prices = [], []
    for _, r in fq.iterrows():
        hdf = hist_data.get(r["FIRM"])
        y, px = trailing_yield_for_date(hdf, r["FORMATION_DATE"])
        yields.append(y)
        prices.append(px)
    fq["DIV_YIELD"] = yields
    fq["PX_AT_FORMATION"] = prices

    n_firms_with_yield = fq.loc[fq["DIV_YIELD"].notna(), "FIRM"].nunique()
    print(f"\n  Firm-quarters with a usable dividend-yield observation: {fq['DIV_YIELD'].notna().sum()} "
          f"of {len(fq)}; covering {n_firms_with_yield} of {n_total} firms.")
    print(f"  ==> yield data available for {n_firms_with_yield} of {n_total} sample firms.")

    print("\n  Average trailing dividend yield by ADJ RRR quartile (pooled across firm-quarters):")
    by_q = fq.groupby("RRR_Q_ADJ")["DIV_YIELD"].agg(["mean", "median", "count"]).reindex(["Q1", "Q2", "Q3", "Q4"])
    for q, row in by_q.iterrows():
        print(f"    {q}: mean={row['mean']:.4%}, median={row['median']:.4%}, N obs={row['count']:.0f}")

    q1_mean, q4_mean = by_q.loc["Q1", "mean"], by_q.loc["Q4", "mean"]
    gap = q1_mean - q4_mean
    print(f"\n  Q1 (high RRR) minus Q4 (low RRR) dividend-yield gap: {gap:.4%} "
          f"({q1_mean:.4%} - {q4_mean:.4%})")

    # Firm-level cross-check: average each firm's OWN trailing yield across
    # its firm-quarters first, then average across firms within each
    # quartile a firm-quarter of theirs ever fell into (a firm can appear in
    # more than one quartile over time, so a firm's average yield can appear
    # in more than one quartile's firm-level average).
    firm_level = fq.groupby(["FIRM", "RRR_Q_ADJ"])["DIV_YIELD"].mean().reset_index()
    firm_level_by_q = firm_level.groupby("RRR_Q_ADJ")["DIV_YIELD"].agg(["mean", "count"]).reindex(["Q1", "Q2", "Q3", "Q4"])
    print("\n  Firm-level cross-check (average each firm's own trailing yield first, then average across firms):")
    for q, row in firm_level_by_q.iterrows():
        print(f"    {q}: mean={row['mean']:.4%}, N firms={row['count']:.0f}")

    failures_df = pd.DataFrame(
        [{"FIRM": k, "REASON": v} for k, v in failures.items()]
    ).sort_values("FIRM") if failures else pd.DataFrame(columns=["FIRM", "REASON"])

    return {
        "coverage_n_firms": n_firms_with_yield, "coverage_of_total": n_total,
        "by_quartile_pooled": by_q.reset_index(),
        "by_quartile_firm_level": firm_level_by_q.reset_index(),
        "q1_minus_q4_gap_pooled": gap,
        "firm_quarter_detail": fq,
        "failures": failures_df,
    }


# =============================================================================
# TASK 7: NET-OF-COST ALPHA
# =============================================================================

def leg_weights_wide(panel_q, bucket_col, bucket_value):
    sub = leg_weights_by_quarter(panel_q, bucket_col, bucket_value)
    wide = sub.pivot(index="FIRM", columns="QUARTER", values="WEIGHT").fillna(0.0)
    return wide


def one_way_turnover(weights_wide):
    """Standard one-way portfolio turnover between consecutive rebalances:
    half the sum of absolute weight changes (0.5 x sum|w_new - w_old|), which
    counts a full round trip (sell what's dropped, buy what's added) as one
    unit of turnover, symmetric to entries and exits. Returns a Series
    indexed by the LATER quarter of each consecutive pair.
    """
    quarters = sorted(weights_wide.columns)
    out = {}
    for prev_q, cur_q in zip(quarters[:-1], quarters[1:]):
        out[cur_q] = 0.5 * (weights_wide[cur_q] - weights_wide[prev_q]).abs().sum()
    return pd.Series(out)


def task7_net_of_cost_alpha(panel, baseline_reg):
    _hdr("TASK 7: NET-OF-COST ALPHA")

    w_q1_wide = leg_weights_wide(panel, "RRR_Q_ADJ", "Q1")
    w_q4_wide = leg_weights_wide(panel, "RRR_Q_ADJ", "Q4")
    to_q1 = one_way_turnover(w_q1_wide)
    to_q4 = one_way_turnover(w_q4_wide)

    print(f"\n  Quarterly one-way turnover, long (Q1) leg: mean={to_q1.mean():.1%}, "
          f"median={to_q1.median():.1%}, N rebalances={len(to_q1)}")
    print(f"  Quarterly one-way turnover, short (Q4) leg: mean={to_q4.mean():.1%}, "
          f"median={to_q4.median():.1%}, N rebalances={len(to_q4)}")

    mean_to_q1, mean_to_q4 = to_q1.mean(), to_q4.mean()
    combined_quarterly_turnover = mean_to_q1 + mean_to_q4
    print(f"  Combined (both legs) average quarterly turnover: {combined_quarterly_turnover:.1%}")

    print(f"\n  Trading-cost assumption: {NMV_ROUNDTRIP_COST_BPS_PRIMARY:.0f}bp round-trip "
          f"(Novy-Marx & Velikov 2016, RFS 29(1):104-147, Figure 1, 'large, liquid stocks' reading "
          f"-- see module docstring for the exact sourcing and reasoning).")

    rows = []
    for spec in ["FF3", "FF5"]:
        gross_alpha = baseline_reg[f"Q1-Q4_{spec}"]["alpha"]
        gross_t = baseline_reg[f"Q1-Q4_{spec}"]["alpha_t"]
        gross_se = baseline_reg[f"Q1-Q4_{spec}"]["se_alpha"]
        gross_p = baseline_reg[f"Q1-Q4_{spec}"]["alpha_p"]
        print(f"\n  {spec} (gross alpha = {gross_alpha*100:.4f}%/mo, t={gross_t:.3f}):")
        for cost_bps in [NMV_ROUNDTRIP_COST_BPS_PRIMARY] + NMV_ROUNDTRIP_COST_BPS_SENSITIVITY:
            cost_frac = cost_bps / 10000.0
            quarterly_cost_drag = combined_quarterly_turnover * cost_frac
            monthly_cost_drag = quarterly_cost_drag / av2.HOLD_MONTHS
            net_alpha = gross_alpha - monthly_cost_drag
            # Subtracting a CONSTANT monthly drag from the dependent variable
            # of an OLS regression that includes a constant shifts only the
            # intercept by that constant; the SE is exactly unchanged, so the
            # net t-stat below is exact (not approximate) under the
            # constant-monthly-drag assumption. p-value from the standard
            # normal (asymptotic) reference distribution, consistent with how
            # the HAC/Newey-West t-stats elsewhere in this project are
            # justified (asymptotic, not exact finite-sample t).
            net_t = net_alpha / gross_se
            net_p = float(2 * sps.norm.sf(abs(net_t)))
            tag = "PRIMARY" if cost_bps == NMV_ROUNDTRIP_COST_BPS_PRIMARY else "sensitivity"
            print(f"    cost={cost_bps:>3.0f}bp [{tag:<11}]: monthly drag={monthly_cost_drag*100:.4f}%, "
                  f"net alpha={net_alpha*100:.4f}%/mo, net t={net_t:.3f}, net p={_p(net_p)}")
            rows.append({
                "Spec": spec, "Cost_bps": cost_bps, "Tag": tag,
                "Gross_alpha_pct": gross_alpha * 100, "Gross_t": gross_t, "Gross_p": gross_p,
                "Quarterly_turnover_combined": combined_quarterly_turnover,
                "Monthly_cost_drag_pct": monthly_cost_drag * 100,
                "Net_alpha_pct": net_alpha * 100, "Net_t": net_t, "Net_p": net_p,
            })

    turnover_detail = pd.DataFrame({
        "Q1_turnover": to_q1, "Q4_turnover": to_q4,
    }).reset_index().rename(columns={"index": "QUARTER"})
    turnover_detail["QUARTER"] = turnover_detail["QUARTER"].apply(quarter_label)

    return {
        "turnover_q1_mean": mean_to_q1, "turnover_q4_mean": mean_to_q4,
        "turnover_combined_quarterly": combined_quarterly_turnover,
        "cost_bps_used": NMV_ROUNDTRIP_COST_BPS_PRIMARY,
        "net_of_cost_table": pd.DataFrame(rows),
        "turnover_detail": turnover_detail,
    }


# =============================================================================
# TASK 8: H2A / H2B ASYMMETRY
# =============================================================================

def _downside_beta(y, x):
    down = x < 0
    if down.sum() < 6:
        return np.nan
    return float(np.polyfit(x[down], y[down], 1)[0])


def _ann_vol(y):
    return float(np.std(y, ddof=1) * np.sqrt(12))


def _ann_sharpe(y):
    v = _ann_vol(y)
    return float((np.mean(y) * 12) / v) if v > 0 else np.nan


def task8_asymmetry(panel, baseline_port, ff_factors, rng):
    _hdr("TASK 8: H2A/H2B ASYMMETRY (downside-risk bootstrap + up/down-market interaction)")

    # ---------------------------------------------------------------
    # H2a: bootstrap the Q1-Q4 difference in downside beta, vol, Sharpe
    # ---------------------------------------------------------------
    print("\n  --- H2a (downside/insurance): bootstrap Q1 vs Q4 differences ---")
    port = baseline_port.copy()
    port.index = pd.to_datetime(port.index).to_period("M").to_timestamp("M")
    combined = port[["Q1", "Q4"]].join(ff_factors[["Mkt-RF"]], how="inner").dropna()
    q1_ret = combined["Q1"].values
    q4_ret = combined["Q4"].values
    mkt = combined["Mkt-RF"].values
    n = len(combined)
    print(f"    N months: {n}")

    point = {
        "downside_beta_Q1": _downside_beta(q1_ret, mkt),
        "downside_beta_Q4": _downside_beta(q4_ret, mkt),
        "ann_vol_Q1": _ann_vol(q1_ret), "ann_vol_Q4": _ann_vol(q4_ret),
        "ann_sharpe_Q1": _ann_sharpe(q1_ret), "ann_sharpe_Q4": _ann_sharpe(q4_ret),
    }
    point["diff_downside_beta"] = point["downside_beta_Q1"] - point["downside_beta_Q4"]
    point["diff_ann_vol"] = point["ann_vol_Q1"] - point["ann_vol_Q4"]
    point["diff_ann_sharpe"] = point["ann_sharpe_Q1"] - point["ann_sharpe_Q4"]
    for k, v in point.items():
        print(f"    {k}: {v:.4f}")

    idx_matrix = circular_block_bootstrap_indices(n, PRIMARY_BLOCK_MONTHS, N_BOOTSTRAP, rng)
    diffs_beta = np.empty(N_BOOTSTRAP)
    diffs_vol = np.empty(N_BOOTSTRAP)
    diffs_sharpe = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        rows = idx_matrix[b]
        q1_b, q4_b, mkt_b = q1_ret[rows], q4_ret[rows], mkt[rows]
        beta1, beta4 = _downside_beta(q1_b, mkt_b), _downside_beta(q4_b, mkt_b)
        diffs_beta[b] = beta1 - beta4
        vol1, vol4 = _ann_vol(q1_b), _ann_vol(q4_b)
        diffs_vol[b] = vol1 - vol4
        sh1, sh4 = _ann_sharpe(q1_b), _ann_sharpe(q4_b)
        diffs_sharpe[b] = sh1 - sh4

    h2a_rows = []
    for label, arr, pt in [
        ("downside_beta (Q1-Q4)", diffs_beta, point["diff_downside_beta"]),
        ("ann_vol (Q1-Q4)", diffs_vol, point["diff_ann_vol"]),
        ("ann_sharpe (Q1-Q4)", diffs_sharpe, point["diff_ann_sharpe"]),
    ]:
        arr_clean = arr[~np.isnan(arr)]
        se = np.nanstd(arr_clean, ddof=1)
        ci_lo, ci_hi = np.nanpercentile(arr_clean, [2.5, 97.5])
        excludes_zero = not (ci_lo <= 0 <= ci_hi)
        boot_t = pt / se if se > 0 else np.nan
        print(f"    {label}: point={pt:.4f}, boot SE={se:.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}], "
              f"boot t={boot_t:.3f}, distinguishable from 0={excludes_zero}")
        h2a_rows.append({
            "Metric": label, "Point_estimate": pt, "Boot_SE": se,
            "CI_lo": ci_lo, "CI_hi": ci_hi, "Boot_t": boot_t,
            "Distinguishable_from_zero_95pct": excludes_zero, "N_boot": N_BOOTSTRAP,
        })
    h2a_df = pd.DataFrame(h2a_rows)

    # ---------------------------------------------------------------
    # H2b: up-market / down-market interaction, panel regression
    # ---------------------------------------------------------------
    # RET_i,t = a0 + a1*MktRF_t + a2*RRR_i,t + a3*UP_t
    #           + a4*(RRR_i,t x MktRF_t) + a5*(RRR_i,t x MktRF_t x UP_t) + e_i,t
    # a4 is high-RRR firms' market beta differential in DOWN months (UP=0);
    # a5 is the ADDITIONAL differential specifically in UP months -- the
    # direct test of "does high-RRR firms' return sensitivity to the market
    # differ between up and down months." a5 > 0 says high-RRR firms pick up
    # MORE market beta in up months than their down-month beta would predict
    # (upside participation), which combined with H2a's downside-protection
    # finding would describe an asymmetric/convex payoff.
    print("\n  --- H2b (upside): up-market/down-market RRR-sensitivity interaction, panel regression ---")

    reg_df = panel[["FIRM", "Date", "QUARTER", "RET_SIMPLE", "ADJ_RRR_PCT"]].copy()
    reg_df["RET_SIMPLE"] = pd.to_numeric(reg_df["RET_SIMPLE"], errors="coerce")
    reg_df["ADJ_RRR_PCT"] = pd.to_numeric(reg_df["ADJ_RRR_PCT"], errors="coerce")
    reg_df["MONTH_P"] = pd.to_datetime(reg_df["Date"]).dt.to_period("M").dt.to_timestamp("M")
    mkt = ff_factors["Mkt-RF"].copy()
    mkt.index = pd.to_datetime(mkt.index).to_period("M").to_timestamp("M")
    reg_df["MKT_RF"] = reg_df["MONTH_P"].map(mkt)
    reg_df = reg_df.dropna(subset=["RET_SIMPLE", "ADJ_RRR_PCT", "MKT_RF"])
    reg_df["UP"] = (reg_df["MKT_RF"] > 0).astype(float)
    reg_df["RRR_x_MKT"] = reg_df["ADJ_RRR_PCT"] * reg_df["MKT_RF"]
    reg_df["RRR_x_MKT_x_UP"] = reg_df["RRR_x_MKT"] * reg_df["UP"]

    def _twoway_cluster_se(base_model, d):
        """Two-way (firm, month) cluster-robust covariance via the standard
        Cameron, Gelbach & Miller (2011) / Thompson (2011) combination:
            V_2way = V_cluster(firm) + V_cluster(month) - V_HC1
        computed from ONE fitted OLS (point estimates are covariance-type
        invariant; only the covariance matrix differs) via
        get_robustcov_results, rather than a single cov_type='cluster' call
        with a multi-column groups array, whose behavior in this statsmodels
        version is not something this script's author could verify -- this
        explicit three-covariance combination is unambiguous and standard.
        Guards against the (rare) non-PSD combined matrix by falling back to
        the larger of the two one-way SEs for any affected coefficient.
        """
        res_firm = base_model.get_robustcov_results(cov_type="cluster", groups=d["FIRM"].to_numpy())
        res_month = base_model.get_robustcov_results(cov_type="cluster", groups=d["MONTH_P"].astype(str).to_numpy())
        res_hc1 = base_model.get_robustcov_results(cov_type="HC1")
        v_firm = np.diag(res_firm.cov_params())
        v_month = np.diag(res_month.cov_params())
        v_hc1 = np.diag(res_hc1.cov_params())
        v_2way = v_firm + v_month - v_hc1
        fallback_used = bool(np.any(v_2way <= 0))
        se_2way = np.where(v_2way > 0, np.sqrt(np.clip(v_2way, 0, None)),
                            np.maximum(np.sqrt(v_firm), np.sqrt(v_month)))
        return se_2way, fallback_used

    def _fit_h2b(df, rrr_col, label):
        X_cols = ["MKT_RF", rrr_col, "UP", f"{rrr_col}_x_MKT", f"{rrr_col}_x_MKT_x_UP"]
        d = df.copy()
        d[f"{rrr_col}_x_MKT"] = d[rrr_col] * d["MKT_RF"]
        d[f"{rrr_col}_x_MKT_x_UP"] = d[f"{rrr_col}_x_MKT"] * d["UP"]
        y = d["RET_SIMPLE"]
        X = sm.add_constant(d[X_cols])
        base_model = sm.OLS(y, X).fit()  # point estimates: covariance-type invariant
        se_2way, fallback_used = _twoway_cluster_se(base_model, d)
        t_2way = base_model.params.values / se_2way
        p_2way = 2 * sps.t.sf(np.abs(t_2way), base_model.df_resid)
        cluster_note = "two-way (firm, month), Cameron-Gelbach-Miller (2011)"
        if fallback_used:
            cluster_note += " [non-PSD combination on >=1 coef: fell back to max(one-way SEs) there]"

        print(f"\n    {label} (N={int(base_model.nobs)}, clustered SE: {cluster_note}):")
        rows = []
        all_vars = ["const"] + X_cols
        for i, var in enumerate(all_vars):
            coef = base_model.params[var]
            se, t, pv = se_2way[i], t_2way[i], p_2way[i]
            print(f"      {var:<24} coef={coef:>10.5f}  SE={se:>9.5f}  t={t:>7.3f}  p={_p(pv)}")
            rows.append({"Variable": var, "Coef": coef, "SE": se, "t": t, "p": pv})
        return pd.DataFrame(rows), cluster_note, int(base_model.nobs), base_model.rsquared

    h2b_continuous, cluster_note_c, n_c, r2_c = _fit_h2b(reg_df, "ADJ_RRR_PCT", "Continuous ADJ_RRR_PCT specification")

    # Robustness cut: Q1-vs-Q4 dummy instead of continuous RRR, restricted to
    # firms actually assigned to Q1 or Q4 that quarter (mirrors the portfolio
    # sort's "high vs low RRR" comparison directly).
    panel_q1q4 = panel.loc[panel["RRR_Q_ADJ"].isin(["Q1", "Q4"]),
                            ["FIRM", "Date", "QUARTER", "RET_SIMPLE", "RRR_Q_ADJ"]].copy()
    panel_q1q4["RET_SIMPLE"] = pd.to_numeric(panel_q1q4["RET_SIMPLE"], errors="coerce")
    panel_q1q4["MONTH_P"] = pd.to_datetime(panel_q1q4["Date"]).dt.to_period("M").dt.to_timestamp("M")
    panel_q1q4["MKT_RF"] = panel_q1q4["MONTH_P"].map(mkt)
    panel_q1q4 = panel_q1q4.dropna(subset=["RET_SIMPLE", "MKT_RF"])
    panel_q1q4["UP"] = (panel_q1q4["MKT_RF"] > 0).astype(float)
    panel_q1q4["Q1_DUMMY"] = (panel_q1q4["RRR_Q_ADJ"] == "Q1").astype(float)

    h2b_dummy, cluster_note_d, n_d, r2_d = _fit_h2b(panel_q1q4, "Q1_DUMMY", "Q1-vs-Q4 dummy specification (robustness)")

    return {
        "h2a_bootstrap": h2a_df,
        "h2a_point_estimates": point,
        "h2b_continuous": h2b_continuous,
        "h2b_continuous_meta": {"n_obs": n_c, "r2": r2_c, "cluster": cluster_note_c},
        "h2b_dummy": h2b_dummy,
        "h2b_dummy_meta": {"n_obs": n_d, "r2": r2_d, "cluster": cluster_note_d},
    }


# =============================================================================
# TASK 9: AGARWAL ET AL. (2021) / BAKER ET AL. (2023) DIFFERENTIATION
# =============================================================================

def compound_holding_return(panel, ff_factors, extra_cols):
    """Mirrors analysis_v2.run_fama_macbeth's holding-window compounding
    exactly (same HOLD_MONTHS-month compounding of RET_SIMPLE into one
    holding-period excess return per firm-quarter, same contemporaneous
    quarter-end controls), extended to carry extra_cols through so a new
    regressor (ADJ_SRR_PCT) can be tested without editing analysis_v2.py's
    hardcoded run_fama_macbeth spec dict.
    """
    df = panel.copy()
    rf = ff_factors["RF"].copy()
    rf.index = pd.to_datetime(rf.index).to_period("M")
    df["MONTH_P"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    df["RF_M"] = df["MONTH_P"].map(rf)
    df["RET_SIMPLE"] = pd.to_numeric(df["RET_SIMPLE"], errors="coerce")
    for c in extra_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_kwargs = {
        "n": ("RET_SIMPLE", "size"),
        "gross": ("RET_SIMPLE", lambda s: (1 + s).prod()),
        "gross_rf": ("RF_M", lambda s: (1 + s).prod()),
    }
    agg_kwargs.update({c: (c, "first") for c in extra_cols})
    hw = (df.sort_values(["FIRM", "QUARTER", "HOLD_IDX"])
          .groupby(["FIRM", "QUARTER"]).agg(**agg_kwargs).reset_index())
    hw = hw.loc[hw["n"] == av2.HOLD_MONTHS].copy()
    hw["EXCESS_RET_HW"] = hw["gross"] - hw["gross_rf"]
    return hw


def fama_macbeth_spec(hw, x_vars, y_var="EXCESS_RET_HW"):
    """Quarterly cross-sectional OLS of y_var on x_vars, then a Newey-West
    (Bartlett kernel) time-series average of the quarterly slopes -- the
    identical estimation method analysis_v2.run_fama_macbeth uses, applied
    here to a spec set that includes ADJ_SRR_PCT (not one of
    run_fama_macbeth's hardcoded specs).
    """
    period_coefs = []
    for qtr, group in hw.groupby("QUARTER"):
        sub = group[[y_var] + x_vars].dropna()
        if len(sub) < 10:
            continue
        model = sm.OLS(sub[y_var], sm.add_constant(sub[x_vars])).fit()
        coefs = model.params.to_dict()
        coefs["QUARTER"] = qtr
        coefs["N"] = len(sub)
        period_coefs.append(coefs)

    if not period_coefs:
        return None

    coef_df = pd.DataFrame(period_coefs)
    T = len(coef_df)
    avg_coefs = coef_df.drop(columns=["QUARTER", "N"]).mean()
    max_lag = max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))
    se_nw = {}
    for var in avg_coefs.index:
        series = coef_df[var] - avg_coefs[var]
        gamma_sum = (series ** 2).mean()
        for j in range(1, max_lag + 1):
            gamma_j = (series.iloc[j:].values * series.iloc[:-j].values).mean()
            gamma_sum += 2 * (1 - j / (max_lag + 1)) * gamma_j
        se_nw[var] = np.sqrt(gamma_sum / T)
    t_stats = {v: (avg_coefs[v] / se_nw[v] if se_nw[v] > 0 else np.nan) for v in avg_coefs.index}
    p_values = {v: float(2 * sps.t.sf(abs(t_stats[v]), max(T - 1, 1))) if np.isfinite(t_stats[v]) else np.nan
                for v in avg_coefs.index}
    return {"avg_coefs": avg_coefs, "t_stats": t_stats, "p_values": p_values,
            "T": T, "avg_N": coef_df["N"].mean()}


def task9_repeat_new_customer_nesting(panel, ff_factors):
    _hdr("TASK 9: AGARWAL ET AL. (2021) / BAKER ET AL. (2023) DIFFERENTIATION")

    print("""
  Agarwal et al. (2021) and Baker et al. (2023) split credit-card panel
  spending into repeat-customer vs new-customer shares and use that split to
  predict returns -- close to this paper's RRR-vs-AR framework but built from
  a different data source and a different normalization.

  This project's revenue panel already carries a mechanically DISTINCT
  repeat/new revenue split: SRR_PCT ("Share of Retained Revenue" =
  Returning_Revenue_t / Total_Revenue_t, i.e. the SHARE of CURRENT revenue
  from returning customers -- a composition/mix measure) and its complement
  SHARE_NEW_REV_PCT. This is constructible and already present in the panel
  (ADJ_SRR_PCT, industry x time adjusted, same convention as ADJ_RRR_PCT).

  It is NOT the same construction as RRR: RRR_t = Returning_Revenue_t /
  Total_Revenue_{t-1} normalizes by PRIOR-period total revenue (a
  retention/renewal RATE); SRR_t = Returning_Revenue_t / Total_Revenue_t
  normalizes by CURRENT-period total revenue (a composition SHARE). SRR is
  the closer analogue to the "share of current-period spending from repeat
  customers" construction in the credit-card-panel literature.
""")

    # ---- Empirical distinctness check: how correlated are ADJ_RRR_PCT and ADJ_SRR_PCT? ----
    fq = panel.drop_duplicates(subset=["FIRM", "QUARTER"]).copy()
    fq["ADJ_RRR_PCT"] = pd.to_numeric(fq["ADJ_RRR_PCT"], errors="coerce")
    fq["ADJ_SRR_PCT"] = pd.to_numeric(fq["ADJ_SRR_PCT"], errors="coerce")
    both = fq[["ADJ_RRR_PCT", "ADJ_SRR_PCT"]].dropna()
    pooled_corr = both["ADJ_RRR_PCT"].corr(both["ADJ_SRR_PCT"])
    per_q_corr = fq.groupby("QUARTER").apply(
        lambda g: g["ADJ_RRR_PCT"].corr(g["ADJ_SRR_PCT"])
    )
    print(f"  Empirical distinctness: pooled corr(ADJ_RRR_PCT, ADJ_SRR_PCT) = {pooled_corr:.4f} "
          f"(N={len(both)} firm-quarters)")
    print(f"    Average per-quarter cross-sectional correlation: {per_q_corr.mean():.4f} "
          f"(median {per_q_corr.median():.4f}, N quarters={per_q_corr.notna().sum()})")

    coverage = fq["ADJ_SRR_PCT"].notna().sum()
    print(f"  ADJ_SRR_PCT coverage: {coverage} of {len(fq)} firm-quarters "
          f"({coverage/len(fq):.1%})")

    # ---- Nesting Fama-MacBeth: does RRR retain predictive power after controlling for SRR? ----
    hw = compound_holding_return(panel, ff_factors, ["ADJ_RRR_PCT", "ADJ_SRR_PCT", "SIZE", "BTM", "PM_OPER_PCT"])

    specs = {
        "(1) Adj RRR only": ["ADJ_RRR_PCT"],
        "(2) Adj SRR only": ["ADJ_SRR_PCT"],
        "(3) Adj RRR + Adj SRR": ["ADJ_RRR_PCT", "ADJ_SRR_PCT"],
        "(4) Adj RRR + Adj SRR + Controls": ["ADJ_RRR_PCT", "ADJ_SRR_PCT", "SIZE", "BTM", "PM_OPER_PCT"],
    }

    print("\n  Nesting Fama-MacBeth (quarterly cross-sectional OLS + Newey-West time-series average,\n"
          "  identical methodology to analysis_v2.run_fama_macbeth, extended with ADJ_SRR_PCT):")
    fm_rows = []
    for spec_name, x_vars in specs.items():
        res = fama_macbeth_spec(hw, x_vars)
        if res is None:
            print(f"    {spec_name}: no valid periods")
            continue
        print(f"\n    {spec_name} (T={res['T']} quarters, avg N={res['avg_N']:.0f}):")
        for var in [v for v in res["avg_coefs"].index if v != "const"]:
            coef, t, pv = res["avg_coefs"][var], res["t_stats"][var], res["p_values"][var]
            print(f"      {var:<16} coef={coef:>10.4f}  t={t:>7.3f}  p={_p(pv)}")
            fm_rows.append({
                "Spec": spec_name, "Variable": var, "Coefficient": coef,
                "t_stat": t, "p_value": pv, "T": res["T"], "Avg_N": res["avg_N"],
            })

    fm_df = pd.DataFrame(fm_rows)
    return {
        "pooled_corr_rrr_srr": pooled_corr, "per_quarter_corr_mean": per_q_corr.mean(),
        "srr_coverage_n": int(coverage), "srr_coverage_total": int(len(fq)),
        "nesting_fm_table": fm_df,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 80)
    print("  RRR FINANCIAL IMPLICATIONS -- ROBUSTNESS DIAGNOSTICS")
    print(f"  Random seed: {RANDOM_SEED}   Bootstrap replications: {N_BOOTSTRAP}")
    print("=" * 80)

    df_filtered, returns, ff_factors, panel, market_ret = load_base_data()
    baseline_port, baseline_reg = baseline_portfolio_and_regressions(panel, ff_factors, market_ret)

    # Sanity check against the confirmed headline numbers before trusting
    # anything downstream.
    ff3 = baseline_reg["Q1-Q4_FF3"]
    ff5 = baseline_reg["Q1-Q4_FF5"]
    print(f"\n  [CHECK] Baseline FF3 alpha={ff3['alpha']*100:.4f}%/mo t={ff3['alpha_t']:.4f} "
          f"(confirmed context: 1.920%/mo, t=2.94)")
    print(f"  [CHECK] Baseline FF5 alpha={ff5['alpha']*100:.4f}%/mo t={ff5['alpha_t']:.4f} "
          f"(confirmed context: 2.155%/mo, t=3.29)")

    results = {}
    results["task1"] = task1_concentration(panel, ff_factors, market_ret, baseline_reg)
    results["task2"] = task2_bootstrap_and_subperiod(baseline_port, ff_factors, rng)
    results["task3"] = task3_consumer_discretionary_only(panel, ff_factors)
    results["task4"] = task4_autocorrelation_reconciliation(df_filtered)
    results["task5"] = task5_quartile_monotonicity(baseline_reg)
    results["task6"] = task6_dividend_yield_gap(panel)
    results["task7"] = task7_net_of_cost_alpha(panel, baseline_reg)
    results["task8"] = task8_asymmetry(panel, baseline_port, ff_factors, rng)
    results["task9"] = task9_repeat_new_customer_nesting(panel, ff_factors)

    # =========================================================================
    # EXPORT: one Excel workbook (all tasks, one or more sheets each) + one
    # JSON manifest of headline scalar figures.
    # =========================================================================
    _hdr("EXPORTING RESULTS")

    sheets = {
        "T1_EffN_Q1": results["task1"]["eff_n_q1_by_quarter"],
        "T1_EffN_Q4": results["task1"]["eff_n_q4_by_quarter"],
        "T1_ExcludeTopK": results["task1"]["exclusion_table"],
        "T2_Bootstrap": results["task2"]["bootstrap"],
        "T2_Subperiod": results["task2"]["subperiod"],
        "T4_Transition_RawRRR": results["task4"]["transition_matrices"]["Raw RRR quartile"].reset_index(),
        "T4_Transition_AdjRRR": results["task4"]["transition_matrices"]["Adj RRR quartile"].reset_index(),
        "T5_QuartileAlphas": results["task5"],
        "T6_DivYield_Pooled": results["task6"]["by_quartile_pooled"],
        "T6_DivYield_FirmLevel": results["task6"]["by_quartile_firm_level"],
        "T6_Failures": results["task6"]["failures"],
        "T7_NetOfCost": results["task7"]["net_of_cost_table"],
        "T7_TurnoverDetail": results["task7"]["turnover_detail"],
        "T8_H2a_Bootstrap": results["task8"]["h2a_bootstrap"],
        "T8_H2b_Continuous": results["task8"]["h2b_continuous"],
        "T8_H2b_Dummy": results["task8"]["h2b_dummy"],
        "T9_NestingFM": results["task9"]["nesting_fm_table"],
    }
    with pd.ExcelWriter(DIAG_XLSX_PATH, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"  Workbook written: {DIAG_XLSX_PATH}")

    def _jsonable(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (pd.Timestamp, date)):
            return str(o)
        if isinstance(o, dict):
            return {str(k): _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple, np.ndarray)):
            return [_jsonable(v) for v in o]
        if isinstance(o, float) and np.isnan(o):
            return None
        return o

    manifest = {
        "run_timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "random_seed": RANDOM_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "baseline_ff3_alpha_pct": ff3["alpha"] * 100, "baseline_ff3_t": ff3["alpha_t"],
        "baseline_ff5_alpha_pct": ff5["alpha"] * 100, "baseline_ff5_t": ff5["alpha_t"],
        "task1_eff_n_q1_summary": results["task1"]["eff_n_q1_summary"],
        "task1_eff_n_q4_summary": results["task1"]["eff_n_q4_summary"],
        "task3_consumer_discretionary": {k: v for k, v in results["task3"].items()},
        "task4_autocorrelation": results["task4"]["summary"],
        "task6_dividend_yield": {
            "coverage_n_firms": results["task6"]["coverage_n_firms"],
            "coverage_of_total": results["task6"]["coverage_of_total"],
            "q1_minus_q4_gap_pooled": results["task6"]["q1_minus_q4_gap_pooled"],
        },
        "task7_net_of_cost": {
            "turnover_q1_mean": results["task7"]["turnover_q1_mean"],
            "turnover_q4_mean": results["task7"]["turnover_q4_mean"],
            "turnover_combined_quarterly": results["task7"]["turnover_combined_quarterly"],
            "cost_bps_used": results["task7"]["cost_bps_used"],
        },
        "task9_nesting": {
            "pooled_corr_rrr_srr": results["task9"]["pooled_corr_rrr_srr"],
            "srr_coverage_n": results["task9"]["srr_coverage_n"],
            "srr_coverage_total": results["task9"]["srr_coverage_total"],
        },
    }
    with open(DIAG_MANIFEST_PATH, "w") as fh:
        json.dump(_jsonable(manifest), fh, indent=2)
    print(f"  Manifest written: {DIAG_MANIFEST_PATH}")

    print(f"\n  Total runtime: {time.time()-t_start:.1f}s")
    print("=" * 80)
    print("  ROBUSTNESS DIAGNOSTICS COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
