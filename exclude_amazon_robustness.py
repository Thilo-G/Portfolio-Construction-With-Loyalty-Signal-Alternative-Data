"""
exclude_amazon_robustness.py -- RRR Financial Implications
=============================================================
Robustness check: exclude AMZN from the investable universe ENTIRELY (every
signal quarter of the sample, not just the quarters it happens to rank top-1
by formation-cap weight) and rerun the primary ADJ-RRR value-weighted
long-short quartile sort and its FF3/FF5 factor regressions.

Motivation: robustness_diagnostics.py's Task 1 (concentration diagnostics)
already showed that the long (Q1) leg's effective number of names collapsed
from roughly 7 in 2017 to roughly 1.9 by 2024Q3, and that a GENERIC
top-1/2/3-by-formation-weight exclusion -- dropping the top-K name(s)
post-hoc, quarter by quarter, from an ALREADY-FORMED Q1, with no
re-quantification -- survives (t stays above 2.4 throughout). This script
asks a stricter, name-specific question: what happens if AMZN specifically
is removed from the eligible universe BEFORE the quartile sort is computed,
for the ENTIRE sample period? Under that construction, in any quarter where
AMZN would have ranked into Q1, the quartile boundary is recomputed on the
remaining 123 firms and the next-best firm by ADJ_RRR_PCT is promoted into
Q1 -- the portfolio is re-FORMED from the remaining eligible firms each
quarter, not merely stripped of AMZN after the fact. This is a strictly
different (and for the firms newly promoted into Q1, more consequential)
test than the generic top-K exclusion already in robustness_diagnostics.py.

Design principle (same as robustness_diagnostics.py): import data-loading
and panel-construction functions from analysis_v2.py (build_holding_panel,
build_portfolio_returns, run_factor_regressions, value_weighted_market,
safe_quartile) and reuse the concentration-diagnostic utilities and baseline
pipeline already built in robustness_diagnostics.py (load_base_data,
baseline_portfolio_and_regressions, hhi_effective_n, leg_weights_by_quarter,
quarter_label, _p) rather than reimplementing them. Neither analysis_v2.py,
nor robustness_diagnostics.py, nor any other frozen file is modified by this
script; both are only ever imported (read-only reuse). Importing
robustness_diagnostics.py does NOT trigger its 9-task run (bootstraps,
yfinance pulls, etc.) because that work lives behind its own
`if __name__ == "__main__":` guard -- only its function/constant
definitions execute on import.

Run:
    "C:\\Users\\thkraft\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" exclude_amazon_robustness.py
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# PATH CONSTANTS
# =============================================================================
CODE_DIR = (
    r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)"
    r"\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM"
)
OUTPUT_DIR = os.path.join(CODE_DIR, "output")
RESULT_XLSX_PATH = os.path.join(OUTPUT_DIR, "exclude_amazon_robustness.xlsx")
RESULT_MANIFEST_PATH = os.path.join(OUTPUT_DIR, "exclude_amazon_robustness_manifest.json")

sys.path.insert(0, CODE_DIR)
import analysis_v2 as av2            # noqa: E402 -- build_holding_panel, build_portfolio_returns, run_factor_regressions, value_weighted_market, safe_quartile, EXPECTED_N_FIRMS
import robustness_diagnostics as rd  # noqa: E402 -- load_base_data, baseline_portfolio_and_regressions, hhi_effective_n, leg_weights_by_quarter, quarter_label, _p

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# ANALYSIS CONSTANTS
# =============================================================================
AMZN_SEARCH_TOKEN = "AMZN"  # substring used to auto-detect Amazon's Bloomberg-style FIRM identifier

# Headline baseline this run is checked against: robustness_diagnostics.py's
# BASELINE ADJ RRR Q1-Q4 VW regression, as persisted in
# output/robustness_diagnostics_manifest.json (run_timestamp 2026-07-19T10:27:19).
CONFIRMED_BASELINE_FF3_ALPHA_PCT = 1.9198
CONFIRMED_BASELINE_FF3_T = 2.94
CONFIRMED_BASELINE_FF5_ALPHA_PCT = 2.1547
CONFIRMED_BASELINE_FF5_T = 3.29

# Two-tailed critical values used only to CHARACTERIZE the result in plain
# language ("clears the 5%/1% threshold"). No significance stars are ever
# printed or exported; exact t-stats and p-values are reported throughout.
T_CRIT_5PCT_TWOTAIL = 1.96
T_CRIT_1PCT_TWOTAIL = 2.58


def _hdr(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def _jsonable(o):
    """Recursively convert numpy/pandas scalar types to native Python types
    so the manifest dict can be serialized with the stdlib json module."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def find_amzn_firm_id(firm_ids):
    """Auto-detect Amazon's exact Bloomberg-style FIRM identifier (e.g.
    'AMZN US EQUITY') from the panel's own FIRM universe, rather than
    hardcoding a guessed string. Fails loudly if zero or more than one
    candidate matches, so a silent typo/mismatch can never pass quietly.
    """
    candidates = sorted(set(f for f in firm_ids if AMZN_SEARCH_TOKEN in str(f).upper()))
    if len(candidates) == 0:
        raise ValueError(f"No FIRM identifier containing '{AMZN_SEARCH_TOKEN}' found in the panel.")
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous AMZN identifier -- multiple candidates found: {candidates}")
    return candidates[0]


def eff_n_table_for_leg(panel_q, bucket_col, bucket_value):
    """Per-quarter HHI / effective-N / top-name table for one leg of a sort.
    Thin wrapper around robustness_diagnostics.leg_weights_by_quarter() +
    hhi_effective_n() (both reused, not reimplemented) so the same
    concentration diagnostic can be recomputed on any bucket column -- the
    original RRR_Q_ADJ (AMZN included) and the AMZN-excluded RRR_Q_ADJ_EXAMZN
    both need it.
    """
    weights = rd.leg_weights_by_quarter(panel_q, bucket_col, bucket_value)
    rows = []
    for qtr, g in weights.groupby("QUARTER"):
        hhi, eff_n = rd.hhi_effective_n(g["WEIGHT"].values)
        top_row = g.loc[g["WEIGHT"].idxmax()]
        rows.append({
            "QUARTER": rd.quarter_label(qtr),
            "N_NAMES": g["FIRM"].nunique(),
            "HHI": hhi,
            "EFFECTIVE_N": eff_n,
            "TOP_NAME": top_row["FIRM"],
            "TOP_WEIGHT": top_row["WEIGHT"],
        })
    return pd.DataFrame(rows).sort_values("QUARTER").reset_index(drop=True), weights


def reg_dict_to_df(reg_dict, specification_label):
    """Flatten a run_factor_regressions() result dict into a tidy DataFrame:
    one row per (portfolio bucket, factor-model spec), for Excel export."""
    rows = []
    for key, vals in sorted(reg_dict.items()):
        parts = key.rsplit("_", 1)
        portfolio, model = parts if len(parts) == 2 else (key, "")
        rows.append({
            "Specification": specification_label,
            "Portfolio": portfolio,
            "Model": model,
            "Alpha_pct_per_month": vals["alpha"] * 100,
            "t_stat": vals["alpha_t"],
            "p_value": vals["alpha_p"],
            "SE_alpha_pct": vals["se_alpha"] * 100,
            "R2": vals["r2"],
            "N_obs": vals["n_obs"],
        })
    return pd.DataFrame(rows)


def characterize_significance(t_stat):
    """Plain-language significance characterization from a two-tailed t-stat.
    No stars are ever produced; this only chooses the sentence fragment used
    in the console narrative, and the exact t/p is always printed alongside."""
    if not np.isfinite(t_stat):
        return "undefined (non-finite t-stat)"
    if abs(t_stat) > T_CRIT_1PCT_TWOTAIL:
        return "clears the 1% two-tailed threshold"
    if abs(t_stat) > T_CRIT_5PCT_TWOTAIL:
        return "clears the 5% (but not the 1%) two-tailed threshold"
    return "does NOT clear the conventional 5% two-tailed threshold"


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    print("=" * 80)
    print("  RRR FINANCIAL IMPLICATIONS -- EXCLUDE-AMAZON ROBUSTNESS CHECK")
    print("  (AMZN removed from the investable universe for the ENTIRE sample,")
    print("   BEFORE the quartile sort -- distinct from the generic top-K-by-")
    print("   weight post-hoc exclusion already in robustness_diagnostics.py)")
    print("=" * 80)

    # =========================================================================
    # STEP 1: LOAD BASE DATA + REPRODUCE THE BASELINE (reuse robustness_diagnostics.py)
    # =========================================================================
    df_filtered, returns, ff_factors, panel, market_ret = rd.load_base_data()
    baseline_port, baseline_reg = rd.baseline_portfolio_and_regressions(panel, ff_factors, market_ret)

    baseline_ff3 = baseline_reg["Q1-Q4_FF3"]
    baseline_ff5 = baseline_reg["Q1-Q4_FF5"]
    n_firms_orig = panel["FIRM"].nunique()
    print(f"\n  [CHECK] Baseline ({n_firms_orig} firms) FF3 alpha={baseline_ff3['alpha']*100:.4f}%/mo "
          f"t={baseline_ff3['alpha_t']:.4f}  "
          f"(task context: {CONFIRMED_BASELINE_FF3_ALPHA_PCT}%/mo, t={CONFIRMED_BASELINE_FF3_T})")
    print(f"  [CHECK] Baseline ({n_firms_orig} firms) FF5 alpha={baseline_ff5['alpha']*100:.4f}%/mo "
          f"t={baseline_ff5['alpha_t']:.4f}  "
          f"(task context: {CONFIRMED_BASELINE_FF5_ALPHA_PCT}%/mo, t={CONFIRMED_BASELINE_FF5_T})")

    # =========================================================================
    # STEP 2: HOW MANY QUARTERS WAS AMZN IN Q1 UNDER THE ORIGINAL CONSTRUCTION?
    # =========================================================================
    _hdr("STEP 2: AMZN's Q1 (long-leg) membership under the ORIGINAL sort (AMZN included)")

    amzn_id = find_amzn_firm_id(panel["FIRM"].unique())
    print(f"  Detected AMZN identifier: '{amzn_id}'")

    total_sample_quarters = panel["QUARTER"].nunique()
    amzn_quarterly = (panel.loc[panel["FIRM"] == amzn_id]
                       .drop_duplicates(subset=["QUARTER"])
                       .sort_values("QUARTER")
                       .copy())
    n_quarters_amzn_present = len(amzn_quarterly)
    amzn_q1 = amzn_quarterly.loc[amzn_quarterly["RRR_Q_ADJ"] == "Q1"].copy()
    n_quarters_amzn_q1 = len(amzn_q1)

    print(f"  Total signal quarters in sample: {total_sample_quarters}")
    print(f"  AMZN present in {n_quarters_amzn_present} of those quarters; "
          f"ranked into Q1 (the long leg) in {n_quarters_amzn_q1} of those "
          f"({n_quarters_amzn_q1 / n_quarters_amzn_present:.1%} of AMZN's quarters, "
          f"{n_quarters_amzn_q1 / total_sample_quarters:.1%} of all sample quarters).")

    # AMZN's own bucket every quarter it is present (context: was it always
    # Q1, or did it migrate into Q1 partway through the sample?).
    print("\n  AMZN's ADJ-RRR quartile bucket, by quarter (original 124-firm sort):")
    for _, row in amzn_quarterly.iterrows():
        flag = "  <-- Q1 (long leg)" if row["RRR_Q_ADJ"] == "Q1" else ""
        print(f"    {rd.quarter_label(row['QUARTER'])}: bucket={row['RRR_Q_ADJ']!s:<4} "
              f"ADJ_RRR_PCT={row['ADJ_RRR_PCT']:7.2f}{flag}")

    # Cross-check against the previously reported concentration fact (AMZN
    # reaching ~70.8% of the Q1 leg's formation-cap weight in the last sample
    # quarter): recompute each quarter's Q1-leg weights with the SAME
    # weighting rule as the sort (formation MCAP, normalized within
    # (QUARTER, Q1)) and pull out AMZN's own share whenever it is a member.
    eff_n_baseline_df, w_q1_baseline = eff_n_table_for_leg(panel, "RRR_Q_ADJ", "Q1")
    amzn_weights = w_q1_baseline.loc[w_q1_baseline["FIRM"] == amzn_id].sort_values("QUARTER").copy()
    amzn_weights["QUARTER_LABEL"] = amzn_weights["QUARTER"].apply(rd.quarter_label)
    print("\n  AMZN's share of the Q1 leg's formation-cap weight, by quarter (when in Q1):")
    for _, row in amzn_weights.iterrows():
        print(f"    {row['QUARTER_LABEL']}: {row['WEIGHT']:.1%} of Q1 leg weight")
    if len(amzn_weights) > 0:
        last_q_weight = amzn_weights.iloc[-1]
        print(f"  Last sample quarter AMZN is in Q1 ({last_q_weight['QUARTER_LABEL']}): "
              f"AMZN = {last_q_weight['WEIGHT']:.1%} of Q1 leg weight "
              f"(context claim: 70.8%).")

    # =========================================================================
    # STEP 3: REBUILD THE UNIVERSE WITH AMZN REMOVED ENTIRELY, THEN RE-SORT
    # =========================================================================
    _hdr("STEP 3: Exclude AMZN from the investable universe (entire sample), rebuild panel, RE-SORT")

    df_filtered_ex = df_filtered.loc[df_filtered.index.get_level_values("FIRM") != amzn_id].copy()
    n_firms_ex = df_filtered_ex.index.get_level_values("FIRM").nunique()
    print(f"  Firms in investable universe: {n_firms_orig} (original) -> {n_firms_ex} (AMZN excluded)")
    assert n_firms_ex == n_firms_orig - 1, (
        f"Expected exactly one firm (AMZN) removed from the universe; "
        f"got {n_firms_orig} -> {n_firms_ex}."
    )

    # build_holding_panel() has no cross-sectional dependency on which firms
    # are present (it is a per-firm merge/reshape), so removing AMZN from
    # df_filtered before this call removes it from every downstream object:
    # formation-date market cap, the holding panel, and (next) the quartile
    # sort itself. NOTE: the industry x time adjustment used to build
    # ADJ_RRR_PCT (sector-quarter mean RRR, subtracted in
    # analysis_v2.phase1_load_and_diagnose) is inherited unchanged from the
    # canonical panel snapshot and therefore still reflects AMZN's raw RRR in
    # its Consumer-Discretionary sector-quarter mean. Recomputing that
    # adjustment without AMZN is a separate methodological question from "is
    # AMZN in the investable/sortable universe," which is what this check
    # targets; it is flagged here for transparency, not smoothed over.
    panel_ex = av2.build_holding_panel(df_filtered_ex, returns)
    assert amzn_id not in set(panel_ex["FIRM"].unique()), "AMZN still present in the AMZN-excluded panel."

    # Re-quartile ADJ_RRR_PCT on the REMAINING firms' cross-section, quarter
    # by quarter. This is the step that distinguishes this check from the
    # GENERIC top-K exclusion in robustness_diagnostics.py Task 1: there, the
    # top-K names are nulled out of an ALREADY-FORMED Q1 (post-hoc removal --
    # the eliminated names' slots simply shrink Q1 rather than being
    # backfilled). Here, safe_quartile() is re-run on the reduced universe
    # each quarter. Verified directly (see STEP 3 "promoted" diagnostic
    # below): this does NOT mechanically promote a replacement firm into Q1
    # every single time AMZN is removed -- pd.qcut's quantile boundaries are
    # recomputed on the actual value distribution of the remaining firms, not
    # a fixed headcount rule, so in some quarters the previously-marginal Q2
    # firm IS promoted into Q1 (backfilling AMZN's slot 1-for-1), and in
    # others Q1's headcount simply shrinks by one with no replacement. Both
    # outcomes are legitimate re-sorts; build_portfolio_returns() weights
    # whatever firms actually land in Q1 that quarter either way.
    panel_ex["ADJ_RRR_PCT"] = pd.to_numeric(panel_ex["ADJ_RRR_PCT"], errors="coerce")
    panel_ex["RRR_Q_ADJ_EXAMZN"] = panel_ex.groupby("QUARTER")["ADJ_RRR_PCT"].transform(av2.safe_quartile)

    market_ret_ex = av2.value_weighted_market(panel_ex)
    port_ex = av2.build_portfolio_returns(panel_ex, "RRR_Q_ADJ_EXAMZN", market_ret_ex)
    reg_ex = av2.run_factor_regressions(port_ex, ff_factors, "ADJ RRR Q1-Q4 VW, AMZN excluded from universe")

    assert "Q1-Q4_FF3" in reg_ex and "Q1-Q4_FF5" in reg_ex, (
        "AMZN-excluded regression did not produce Q1-Q4 FF3/FF5 results -- "
        "insufficient overlapping data after re-sorting. Investigate before trusting anything downstream."
    )
    ex_ff3 = reg_ex["Q1-Q4_FF3"]
    ex_ff5 = reg_ex["Q1-Q4_FF5"]

    # How many firm-quarter promotions did the re-sort actually cause? (Every
    # quarter AMZN was in Q1 originally, exactly one non-AMZN firm is
    # promoted from its original bucket into the new Q1, by construction of
    # safe_quartile on one fewer name -- this reports WHICH firms.)
    orig_bucket_by_fq = (panel.drop_duplicates(subset=["FIRM", "QUARTER"])
                          .set_index(["FIRM", "QUARTER"])["RRR_Q_ADJ"])
    new_q1_by_fq = (panel_ex.loc[panel_ex["RRR_Q_ADJ_EXAMZN"] == "Q1"]
                     .drop_duplicates(subset=["FIRM", "QUARTER"])
                     .loc[:, ["FIRM", "QUARTER"]])
    new_q1_by_fq["ORIG_BUCKET"] = new_q1_by_fq.set_index(["FIRM", "QUARTER"]).index.map(orig_bucket_by_fq).values
    promoted = new_q1_by_fq.loc[new_q1_by_fq["ORIG_BUCKET"] != "Q1"].copy()
    promoted["QUARTER_LABEL"] = promoted["QUARTER"].apply(rd.quarter_label)
    print(f"\n  Firm-quarters newly promoted into Q1 by the re-sort (were NOT Q1 originally): {len(promoted)}")
    if len(promoted) > 0:
        print(promoted[["QUARTER_LABEL", "FIRM", "ORIG_BUCKET"]].sort_values("QUARTER_LABEL").to_string(index=False))

    # =========================================================================
    # STEP 4: HEADLINE COMPARISON -- BASELINE vs AMZN-EXCLUDED
    # =========================================================================
    _hdr("STEP 4: HEADLINE COMPARISON -- baseline vs AMZN excluded throughout")

    comparison_rows = [
        {
            "Specification": f"Baseline ({n_firms_orig} firms, AMZN included)",
            "FF3_alpha_pct": baseline_ff3["alpha"] * 100, "FF3_t": baseline_ff3["alpha_t"],
            "FF3_p": baseline_ff3["alpha_p"], "FF3_n_obs": baseline_ff3["n_obs"],
            "FF5_alpha_pct": baseline_ff5["alpha"] * 100, "FF5_t": baseline_ff5["alpha_t"],
            "FF5_p": baseline_ff5["alpha_p"], "FF5_n_obs": baseline_ff5["n_obs"],
        },
        {
            "Specification": f"AMZN excluded from universe ({n_firms_ex} firms, entire sample, re-sorted)",
            "FF3_alpha_pct": ex_ff3["alpha"] * 100, "FF3_t": ex_ff3["alpha_t"],
            "FF3_p": ex_ff3["alpha_p"], "FF3_n_obs": ex_ff3["n_obs"],
            "FF5_alpha_pct": ex_ff5["alpha"] * 100, "FF5_t": ex_ff5["alpha_t"],
            "FF5_p": ex_ff5["alpha_p"], "FF5_n_obs": ex_ff5["n_obs"],
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["FF3_alpha_delta_pct"] = comparison_df["FF3_alpha_pct"] - comparison_df.loc[0, "FF3_alpha_pct"]
    comparison_df["FF3_t_delta"] = comparison_df["FF3_t"] - comparison_df.loc[0, "FF3_t"]
    comparison_df["FF5_alpha_delta_pct"] = comparison_df["FF5_alpha_pct"] - comparison_df.loc[0, "FF5_alpha_pct"]
    comparison_df["FF5_t_delta"] = comparison_df["FF5_t"] - comparison_df.loc[0, "FF5_t"]

    print(comparison_df.to_string(index=False))

    print(f"\n  FF3 long-short alpha: {baseline_ff3['alpha']*100:.4f}%/mo (t={baseline_ff3['alpha_t']:.4f}, "
          f"{rd._p(baseline_ff3['alpha_p'])}, N={baseline_ff3['n_obs']:.0f}) "
          f"-> {ex_ff3['alpha']*100:.4f}%/mo (t={ex_ff3['alpha_t']:.4f}, "
          f"{rd._p(ex_ff3['alpha_p'])}, N={ex_ff3['n_obs']:.0f})")
    print(f"  FF5 long-short alpha: {baseline_ff5['alpha']*100:.4f}%/mo (t={baseline_ff5['alpha_t']:.4f}, "
          f"{rd._p(baseline_ff5['alpha_p'])}, N={baseline_ff5['n_obs']:.0f}) "
          f"-> {ex_ff5['alpha']*100:.4f}%/mo (t={ex_ff5['alpha_t']:.4f}, "
          f"{rd._p(ex_ff5['alpha_p'])}, N={ex_ff5['n_obs']:.0f})")
    print(f"\n  FF3 significance, AMZN excluded: {characterize_significance(ex_ff3['alpha_t'])}")
    print(f"  FF5 significance, AMZN excluded: {characterize_significance(ex_ff5['alpha_t'])}")

    # =========================================================================
    # STEP 5: DOES CONCENTRATION MOVE TO A DIFFERENT SINGLE NAME?
    # =========================================================================
    _hdr("STEP 5: Concentration of the NEW Q1 leg (post-AMZN-exclusion) -- did it just move to another name?")

    eff_n_ex_df, w_q1_ex = eff_n_table_for_leg(panel_ex, "RRR_Q_ADJ_EXAMZN", "Q1")

    print("  Q1 (long leg) effective N across quarters:")
    print(f"    Baseline (AMZN included):  mean={eff_n_baseline_df['EFFECTIVE_N'].mean():.2f}  "
          f"median={eff_n_baseline_df['EFFECTIVE_N'].median():.2f}  "
          f"min={eff_n_baseline_df['EFFECTIVE_N'].min():.2f}  max={eff_n_baseline_df['EFFECTIVE_N'].max():.2f}")
    print(f"    AMZN excluded:              mean={eff_n_ex_df['EFFECTIVE_N'].mean():.2f}  "
          f"median={eff_n_ex_df['EFFECTIVE_N'].median():.2f}  "
          f"min={eff_n_ex_df['EFFECTIVE_N'].min():.2f}  max={eff_n_ex_df['EFFECTIVE_N'].max():.2f}")

    print("\n  Last 4 sample quarters, AMZN excluded -- new top name and its share of the Q1 leg:")
    print(eff_n_ex_df.tail(4).to_string(index=False))

    new_top_name_counts = eff_n_ex_df["TOP_NAME"].value_counts()
    print("\n  Frequency each name is the TOP (highest-weight) name of the new Q1 leg, AMZN excluded, all quarters:")
    print(new_top_name_counts.to_string())
    single_new_top = new_top_name_counts.index[0]
    single_new_top_share_of_quarters = new_top_name_counts.iloc[0] / len(eff_n_ex_df)
    max_top_weight_ex = eff_n_ex_df["TOP_WEIGHT"].max()
    max_top_weight_ex_row = eff_n_ex_df.loc[eff_n_ex_df["TOP_WEIGHT"].idxmax()]
    print(f"\n  Most frequent new top name: '{single_new_top}' "
          f"(top of Q1 in {new_top_name_counts.iloc[0]} of {len(eff_n_ex_df)} quarters, "
          f"{single_new_top_share_of_quarters:.1%}).")
    print(f"  Highest single-quarter top-name weight anywhere in the AMZN-excluded series: "
          f"{max_top_weight_ex:.1%} ('{max_top_weight_ex_row['TOP_NAME']}', {max_top_weight_ex_row['QUARTER']}).")

    # =========================================================================
    # STEP 6: EXPORT
    # =========================================================================
    _hdr("STEP 6: EXPORTING RESULTS")

    amzn_all_quarters_export = amzn_quarterly[["QUARTER", "RRR_Q_ADJ", "ADJ_RRR_PCT"]].copy()
    amzn_all_quarters_export["QUARTER"] = amzn_all_quarters_export["QUARTER"].apply(rd.quarter_label)
    amzn_all_quarters_export = amzn_all_quarters_export.merge(
        amzn_weights[["QUARTER_LABEL", "WEIGHT"]].rename(
            columns={"QUARTER_LABEL": "QUARTER", "WEIGHT": "Q1_LEG_WEIGHT"}),
        on="QUARTER", how="left",
    )

    detail_baseline_df = reg_dict_to_df(baseline_reg, f"Baseline ({n_firms_orig} firms, AMZN included)")
    detail_ex_df = reg_dict_to_df(reg_ex, f"AMZN excluded ({n_firms_ex} firms)")

    sheets = {
        "Summary_Comparison": comparison_df,
        "AMZN_Quarterly_Detail": amzn_all_quarters_export,
        "Promoted_Into_Q1": promoted[["QUARTER_LABEL", "FIRM", "ORIG_BUCKET"]] if len(promoted) > 0
                             else pd.DataFrame(columns=["QUARTER_LABEL", "FIRM", "ORIG_BUCKET"]),
        "EffN_Q1_Baseline": eff_n_baseline_df,
        "EffN_Q1_AMZN_Excluded": eff_n_ex_df,
        "NewTopName_Frequency": new_top_name_counts.rename_axis("FIRM").reset_index(name="N_QUARTERS_AS_TOP"),
        "FactorReg_Baseline_Detail": detail_baseline_df,
        "FactorReg_AMZN_Excluded_Detail": detail_ex_df,
    }
    with pd.ExcelWriter(RESULT_XLSX_PATH, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    print(f"  Workbook written: {RESULT_XLSX_PATH}")

    manifest = {
        "run_timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "amzn_firm_id": amzn_id,
        "n_firms_baseline": n_firms_orig,
        "n_firms_amzn_excluded": n_firms_ex,
        "total_sample_quarters": total_sample_quarters,
        "n_quarters_amzn_present": n_quarters_amzn_present,
        "n_quarters_amzn_in_q1_original": n_quarters_amzn_q1,
        "pct_of_amzn_quarters_in_q1": n_quarters_amzn_q1 / n_quarters_amzn_present,
        "pct_of_all_quarters_amzn_in_q1": n_quarters_amzn_q1 / total_sample_quarters,
        "amzn_q1_weight_max_pct": (amzn_weights["WEIGHT"].max() * 100) if len(amzn_weights) else None,
        "amzn_q1_weight_last_quarter_pct": (amzn_weights.iloc[-1]["WEIGHT"] * 100) if len(amzn_weights) else None,
        "amzn_q1_weight_last_quarter_label": (amzn_weights.iloc[-1]["QUARTER_LABEL"]) if len(amzn_weights) else None,
        "n_firm_quarters_promoted_into_q1": len(promoted),
        "baseline": {
            "FF3_alpha_pct": baseline_ff3["alpha"] * 100, "FF3_t": baseline_ff3["alpha_t"],
            "FF3_p": baseline_ff3["alpha_p"], "FF3_n_obs": baseline_ff3["n_obs"],
            "FF5_alpha_pct": baseline_ff5["alpha"] * 100, "FF5_t": baseline_ff5["alpha_t"],
            "FF5_p": baseline_ff5["alpha_p"], "FF5_n_obs": baseline_ff5["n_obs"],
        },
        "amzn_excluded": {
            "FF3_alpha_pct": ex_ff3["alpha"] * 100, "FF3_t": ex_ff3["alpha_t"],
            "FF3_p": ex_ff3["alpha_p"], "FF3_n_obs": ex_ff3["n_obs"],
            "FF5_alpha_pct": ex_ff5["alpha"] * 100, "FF5_t": ex_ff5["alpha_t"],
            "FF5_p": ex_ff5["alpha_p"], "FF5_n_obs": ex_ff5["n_obs"],
        },
        "delta": {
            "FF3_alpha_pct": ex_ff3["alpha"] * 100 - baseline_ff3["alpha"] * 100,
            "FF3_t": ex_ff3["alpha_t"] - baseline_ff3["alpha_t"],
            "FF5_alpha_pct": ex_ff5["alpha"] * 100 - baseline_ff5["alpha"] * 100,
            "FF5_t": ex_ff5["alpha_t"] - baseline_ff5["alpha_t"],
        },
        "eff_n_q1_baseline_summary": {
            "mean": eff_n_baseline_df["EFFECTIVE_N"].mean(), "median": eff_n_baseline_df["EFFECTIVE_N"].median(),
            "min": eff_n_baseline_df["EFFECTIVE_N"].min(), "max": eff_n_baseline_df["EFFECTIVE_N"].max(),
        },
        "eff_n_q1_amzn_excluded_summary": {
            "mean": eff_n_ex_df["EFFECTIVE_N"].mean(), "median": eff_n_ex_df["EFFECTIVE_N"].median(),
            "min": eff_n_ex_df["EFFECTIVE_N"].min(), "max": eff_n_ex_df["EFFECTIVE_N"].max(),
        },
        "new_top_name_most_frequent": {
            "name": str(single_new_top), "n_quarters": int(new_top_name_counts.iloc[0]),
            "share_of_quarters": single_new_top_share_of_quarters,
        },
        "new_top_name_max_single_quarter_weight_pct": max_top_weight_ex * 100,
        "new_top_name_max_single_quarter_weight_name": str(max_top_weight_ex_row["TOP_NAME"]),
        "new_top_name_max_single_quarter_weight_quarter": str(max_top_weight_ex_row["QUARTER"]),
    }
    with open(RESULT_MANIFEST_PATH, "w") as fh:
        json.dump(_jsonable(manifest), fh, indent=2)
    print(f"  Manifest written: {RESULT_MANIFEST_PATH}")

    # =========================================================================
    # FINAL SUMMARY (the exact (a)/(b)/(c) the task asked for, gathered in one place)
    # =========================================================================
    _hdr("FINAL SUMMARY")
    print(f"  (a) Quarters AMZN was in Q1 (long leg) under the ORIGINAL construction: "
          f"{n_quarters_amzn_q1} of {n_quarters_amzn_present} quarters AMZN is present "
          f"({total_sample_quarters} total sample quarters).")
    print(f"  (b) FF3 alpha: baseline {baseline_ff3['alpha']*100:.4f}%/mo (t={baseline_ff3['alpha_t']:.4f}) "
          f"vs AMZN-excluded {ex_ff3['alpha']*100:.4f}%/mo (t={ex_ff3['alpha_t']:.4f}), "
          f"delta={ex_ff3['alpha']*100 - baseline_ff3['alpha']*100:+.4f}pp / "
          f"{ex_ff3['alpha_t'] - baseline_ff3['alpha_t']:+.4f}t")
    print(f"      FF5 alpha: baseline {baseline_ff5['alpha']*100:.4f}%/mo (t={baseline_ff5['alpha_t']:.4f}) "
          f"vs AMZN-excluded {ex_ff5['alpha']*100:.4f}%/mo (t={ex_ff5['alpha_t']:.4f}), "
          f"delta={ex_ff5['alpha']*100 - baseline_ff5['alpha']*100:+.4f}pp / "
          f"{ex_ff5['alpha_t'] - baseline_ff5['alpha_t']:+.4f}t")
    print(f"  (c) Q1 effective-N: baseline mean={eff_n_baseline_df['EFFECTIVE_N'].mean():.2f} -> "
          f"AMZN-excluded mean={eff_n_ex_df['EFFECTIVE_N'].mean():.2f}. "
          f"Most frequent new top name: '{single_new_top}' "
          f"({new_top_name_counts.iloc[0]}/{len(eff_n_ex_df)} quarters, "
          f"max single-quarter weight {max_top_weight_ex:.1%}).")

    print(f"\n  Total runtime: {time.time() - t_start:.1f}s")
    print("=" * 80)
    print("  EXCLUDE-AMAZON ROBUSTNESS CHECK COMPLETE")
    print("=" * 80)

    return {
        "comparison_df": comparison_df,
        "manifest": manifest,
    }


if __name__ == "__main__":
    main()
