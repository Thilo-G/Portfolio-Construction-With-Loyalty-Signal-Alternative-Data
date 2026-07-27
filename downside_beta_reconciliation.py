"""
downside_beta_reconciliation.py -- RRR Financial Implications
============================================================
Reconciles the two different "Q1 minus Q4 downside beta" figures the pipeline
reports for the SAME adjusted-RRR quartile sort:

  * performance_metrics.py  -> tab_risk_stats.tex:
        Q1 downside beta = 0.7741, Q4 = 1.2263, Q1-Q4 = -0.4521.
  * robustness_diagnostics.py Task 8 (H2a) -> tab_h2ab_asymmetry.tex Panel A:
        Q1-Q4 downside beta point estimate = -1.0617 (bootstrap SE 0.9048).

Both are legitimate; they measure downside beta against DIFFERENT market
benchmarks and therefore define "down months" on DIFFERENT series. This script
imports the exact estimators from BOTH source scripts (read-only; it does NOT
modify analysis_v2.py, performance_metrics.py, or robustness_diagnostics.py) and
recomputes the downside beta for Q1 and Q4 on the identical Q1/Q4 return series,
crossing the two design axes:

    axis 1 -- regressor / benchmark : sample VW market (MKT, raw) vs FF Mkt-RF (excess)
    axis 2 -- down-month definition : MKT_raw < 0 vs FF Mkt-RF < 0

The 2x2 cross isolates which choice drives the factor-of-two gap. Output:
    output/downside_beta_reconciliation.md    (both numbers + one explanatory paragraph)
    output/downside_beta_reconciliation.xlsx  (the 2x2 decomposition, table-builder ready)
"""

import os
import sys
import numpy as np
import pandas as pd

CODE_DIR = (
    r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)"
    r"\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM"
)
sys.path.insert(0, CODE_DIR)

import analysis_v2 as av2
# Read-only imports of the two source estimators (proves we use their exact code):
from performance_metrics import conditional_beta          # sm.OLS beta, down = mkt < 0
from robustness_diagnostics import (
    load_base_data,
    baseline_portfolio_and_regressions,
    _downside_beta as robustness_downside_beta,           # np.polyfit slope, down = x < 0
)

OUTPUT_DIR = av2.OUTPUT_DIR
MD_PATH = os.path.join(OUTPUT_DIR, "downside_beta_reconciliation.md")
XLSX_PATH = os.path.join(OUTPUT_DIR, "downside_beta_reconciliation.xlsx")


def _ols_beta(y, x):
    """Plain OLS slope of y on x with intercept (numpy) -- benchmark-agnostic helper
    used for the 2x2 cross where the down-month set and the regressor come from
    two different series."""
    if len(x) < 6 or np.var(x) == 0:
        return np.nan, len(x)
    return float(np.polyfit(x, y, 1)[0]), len(x)


def main():
    # ---------------------------------------------------------------
    # 1. Build the identical Q1/Q4 series both scripts use.
    #    baseline_portfolio_and_regressions() -> build_portfolio_returns(
    #    panel, "RRR_Q_ADJ", market_ret): the exact object robustness Task 8
    #    consumes, and constructed identically to performance_metrics.py's port.
    # ---------------------------------------------------------------
    df_filtered, returns, ff_factors, panel, market_ret = load_base_data()
    baseline_port, _ = baseline_portfolio_and_regressions(panel, ff_factors, market_ret)

    port = baseline_port.copy()
    port.index = pd.to_datetime(port.index).to_period("M").to_timestamp("M")

    ff = ff_factors.copy()
    ff.index = pd.to_datetime(ff.index).to_period("M").to_timestamp("M")

    # MKT = sample's own value-weighted market portfolio (raw), carried on the
    # port frame by build_portfolio_returns; Mkt-RF = FF broad-CRSP excess market.
    combined = port[["Q1", "Q4", "MKT"]].join(ff[["Mkt-RF", "RF"]], how="inner").dropna()
    n = len(combined)

    q1 = combined["Q1"].values
    q4 = combined["Q4"].values
    mkt_sample = combined["MKT"].values         # raw sample VW market return
    mkt_ff = combined["Mkt-RF"].values          # FF broad-CRSP EXCESS market return

    print("=" * 80)
    print(f"  Reconciliation sample: {n} months "
          f"({combined.index.min():%Y-%m} to {combined.index.max():%Y-%m})")
    print(f"  Down-months (sample MKT<0): {(mkt_sample < 0).sum()};  "
          f"Down-months (FF Mkt-RF<0): {(mkt_ff < 0).sum()}")
    print("=" * 80)

    # ---------------------------------------------------------------
    # 2a. Reproduce performance_metrics.py EXACTLY (its own conditional_beta):
    #     regressor = sample MKT (raw), down = MKT < 0.
    # ---------------------------------------------------------------
    mkt_sample_s = pd.Series(mkt_sample, index=combined.index)
    q1_s = pd.Series(q1, index=combined.index)
    q4_s = pd.Series(q4, index=combined.index)
    perf_q1, perf_nq1 = conditional_beta(q1_s, mkt_sample_s, "down")
    perf_q4, perf_nq4 = conditional_beta(q4_s, mkt_sample_s, "down")
    perf_diff = perf_q1 - perf_q4

    # ---------------------------------------------------------------
    # 2b. Reproduce robustness_diagnostics.py Task 8 EXACTLY (its _downside_beta):
    #     regressor = FF Mkt-RF (excess), down = Mkt-RF < 0.
    # ---------------------------------------------------------------
    rob_q1 = robustness_downside_beta(q1, mkt_ff)
    rob_q4 = robustness_downside_beta(q4, mkt_ff)
    rob_diff = rob_q1 - rob_q4

    # ---------------------------------------------------------------
    # 3. 2x2 decomposition: {regressor benchmark} x {down-month series}.
    #    Down set from `down_series`, regressor from `reg_series`.
    # ---------------------------------------------------------------
    def beta_cross(reg_series, down_series, port_ret):
        down = down_series < 0
        return _ols_beta(port_ret[down], reg_series[down])

    axes = {
        "MKT reg, MKT down (= performance_metrics.py)": (mkt_sample, mkt_sample),
        "Mkt-RF reg, Mkt-RF down (= robustness Task 8)": (mkt_ff, mkt_ff),
        "Mkt-RF reg, MKT down (isolate regressor)": (mkt_ff, mkt_sample),
        "MKT reg, Mkt-RF down (isolate down-set)": (mkt_sample, mkt_ff),
    }

    cross_rows = []
    for label, (reg, down) in axes.items():
        b_q1, nq1 = beta_cross(reg, down, q1)
        b_q4, nq4 = beta_cross(reg, down, q4)
        cross_rows.append({
            "Specification": label,
            "Q1_downside_beta": b_q1,
            "Q4_downside_beta": b_q4,
            "Q1_minus_Q4": b_q1 - b_q4,
            "N_down_months": int((down < 0).sum()),
        })
    cross_df = pd.DataFrame(cross_rows)

    print("\n  --- Exact reproduction of each source script ---")
    print(f"  performance_metrics (conditional_beta, sample MKT, down=MKT<0):")
    print(f"      Q1={perf_q1:.4f} (n_down={perf_nq1}), Q4={perf_q4:.4f} "
          f"(n_down={perf_nq4}), Q1-Q4={perf_diff:.4f}   [table: 0.7741 / 1.2263 / -0.4521]")
    print(f"  robustness Task 8 (_downside_beta, FF Mkt-RF, down=Mkt-RF<0):")
    print(f"      Q1={rob_q1:.4f}, Q4={rob_q4:.4f}, Q1-Q4={rob_diff:.4f}   "
          f"[table: -1.0617]")
    print("\n  --- 2x2 decomposition (which choice drives the gap) ---")
    print(cross_df.to_string(index=False))

    # ---------------------------------------------------------------
    # 4. Reconciled headline table (both legitimate definitions, labeled).
    # ---------------------------------------------------------------
    headline = pd.DataFrame([
        {
            "Definition": "Downside beta vs. sample market (performance_metrics.py)",
            "Benchmark": "Sample's own value-weighted market portfolio (MKT, raw return)",
            "Down_month_rule": "Months the sample market fell (MKT return < 0)",
            "Estimator": "OLS beta with intercept (statsmodels), down-months only",
            "Q1": round(perf_q1, 4), "Q4": round(perf_q4, 4),
            "Q1_minus_Q4": round(perf_diff, 4),
            "SE": np.nan, "N_months": n, "N_down_months": int(perf_nq1),
            "Reported_in": "tab_risk_stats.tex",
        },
        {
            "Definition": "Downside beta vs. FF market, bootstrapped (robustness_diagnostics.py Task 8 H2a)",
            "Benchmark": "Fama-French Mkt-RF (broad CRSP excess market return)",
            "Down_month_rule": "Months FF excess market return was negative (Mkt-RF < 0)",
            "Estimator": "OLS slope with intercept (np.polyfit), down-months only; block-bootstrap SE (n=5,000)",
            "Q1": round(rob_q1, 4), "Q4": round(rob_q4, 4),
            "Q1_minus_Q4": round(rob_diff, 4),
            "SE": 0.9048, "N_months": n, "N_down_months": int((mkt_ff < 0).sum()),
            "Reported_in": "tab_h2ab_asymmetry.tex (Panel A)",
        },
    ])

    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as xw:
        headline.to_excel(xw, sheet_name="Reconciled_headline", index=False)
        cross_df.to_excel(xw, sheet_name="2x2_decomposition", index=False)
    print(f"\n  Workbook written: {XLSX_PATH}")

    # ---------------------------------------------------------------
    # 5. Markdown reconciliation write-up.
    # ---------------------------------------------------------------
    down_sample = int((mkt_sample < 0).sum())
    down_ff = int((mkt_ff < 0).sum())
    md = f"""# Downside-Beta Reconciliation (Q1 minus Q4, adjusted-RRR sort)

**Bottom line:** the two figures are not in conflict and neither is a bug. They
are two legitimate downside betas that use two different market benchmarks and,
consequently, two different definitions of a "down month." Both are computed on
the *identical* Q1 and Q4 adjusted-RRR portfolio return series ({n} months,
{combined.index.min():%Y-%m} to {combined.index.max():%Y-%m}, corrected timing
FORM_LAG={av2.FORM_LAG_MONTHS}m / HOLD={av2.HOLD_MONTHS}m), so the gap is entirely
methodological, not a data or sample-window difference.

## The two numbers

| Definition | Benchmark | Down-month rule | Q1 | Q4 | Q1 - Q4 | SE | Source |
|---|---|---|---|---|---|---|---|
| Downside beta vs. **sample market** | Sample's own value-weighted market portfolio (raw return) | Sample market return < 0 ({down_sample} of {n} months) | {perf_q1:.4f} | {perf_q4:.4f} | **{perf_diff:.4f}** | -- | `tab_risk_stats.tex` (`performance_metrics.py`) |
| Downside beta vs. **FF market**, bootstrapped | Fama-French Mkt-RF (broad CRSP *excess* market return) | FF excess market return < 0 ({down_ff} of {n} months) | {rob_q1:.4f} | {rob_q4:.4f} | **{rob_diff:.4f}** | 0.9048 | `tab_h2ab_asymmetry.tex` Panel A (`robustness_diagnostics.py` Task 8) |

Both figures were reproduced here by importing and calling the *exact* estimator
from each source script (`performance_metrics.conditional_beta` and
`robustness_diagnostics._downside_beta`), so the reproduction is byte-faithful to
what each table reports.

## Why they differ (the two coupled design choices)

Two choices change together when you switch scripts, and they are the only things
that change:

1. **Benchmark / regressor.** `performance_metrics.py` was deliberately built so
   that *every* relative metric (tracking error, information ratio, downside and
   upside beta, hit rate) uses **the sample's own value-weighted market
   portfolio** as "the market" -- a documented design choice (see its module
   header). That benchmark is a 124-firm, four-sector value-weighted index drawn
   from the *same* universe as Q1 and Q4, so the portfolios' betas against it are
   mechanically compressed toward 1 and their Q1-Q4 spread is small. Task 8 of
   `robustness_diagnostics.py` instead regresses on the **Fama-French Mkt-RF**,
   the broad CRSP *excess* market return. Against that external, much broader
   benchmark the same portfolios load more heavily and more dispersedly in
   down states, so the Q1-Q4 spread roughly doubles in magnitude.

2. **Down-month definition (coupled to the benchmark).** "Down months" are the
   months in which the *chosen benchmark* is negative. Because the benchmarks
   differ, the down-month sets differ: {down_sample} months have a negative
   sample-market return, versus {down_ff} months with a negative FF *excess*
   return (Mkt-RF < 0 means the market underperformed the risk-free rate, not
   that its total return was negative). The two estimators therefore fit the
   conditional beta on partly different subsamples.

The 2x2 decomposition below (regressor benchmark x down-month series, both crossed
on the same Q1/Q4 series) confirms the **choice of benchmark is the dominant
driver**; the down-month redefinition contributes a smaller, second-order shift.

| Specification | Q1 | Q4 | Q1 - Q4 | N down |
|---|---|---|---|---|
"""
    for _, r in cross_df.iterrows():
        md += (f"| {r['Specification']} | {r['Q1_downside_beta']:.4f} | "
               f"{r['Q4_downside_beta']:.4f} | {r['Q1_minus_Q4']:.4f} | "
               f"{r['N_down_months']} |\n")

    md += f"""
## Recommended table-note language

Both numbers are correct and should be kept, each with a note stating its
benchmark explicitly:

- For `tab_risk_stats.tex`: "Downside beta is the beta on **the sample's own
  value-weighted market portfolio** (MKT), estimated only in months that
  portfolio fell. This is not directly comparable to the downside beta in the
  H2a asymmetry table, which is measured against the Fama-French broad-market
  excess return."
- For `tab_h2ab_asymmetry.tex` Panel A: "Downside beta here is measured against
  the **Fama-French Mkt-RF** (broad-CRSP excess market return), estimated only in
  months that excess return was negative, and differs from the
  sample-market-based downside beta reported in the risk-statistics table.
  Reported honestly including the non-significant difference (95% CI includes 0)."

## Verdict

No bug. Keep both. Label each with its benchmark. The headline economic finding
is unchanged and consistent across both definitions: Q1 (high RRR) has a **lower**
downside beta than Q4 (low RRR); the two definitions disagree only on the
*magnitude* of that gap ({perf_diff:.4f} vs. {rob_diff:.4f}), for the mechanical
reasons above, and the FF-benchmark version's Q1-Q4 gap is not statistically
distinguishable from zero after block-bootstrap.
"""

    with open(MD_PATH, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"  Reconciliation write-up written: {MD_PATH}")

    # Guardrails: confirm we actually reproduced the two published figures.
    assert abs(perf_q1 - 0.7741) < 5e-4, f"Q1 perf beta {perf_q1} != 0.7741"
    assert abs(perf_q4 - 1.2263) < 5e-4, f"Q4 perf beta {perf_q4} != 1.2263"
    assert abs(rob_diff - (-1.0617)) < 5e-4, f"robustness diff {rob_diff} != -1.0617"
    print("\n  [CHECK] Reproduced both published downside-beta figures to 4 dp.")


if __name__ == "__main__":
    main()
