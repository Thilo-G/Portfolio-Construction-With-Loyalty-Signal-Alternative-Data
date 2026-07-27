"""
fix_cd_only_export.py -- RRR Financial Implications
===================================================
Export-gap fix for robustness_diagnostics.py Task 3 (Consumer-Discretionary-only
alpha). That task computes the CD-only long-short FF3/FF3+Mom/FF5 alphas and puts
them in the JSON manifest, but never writes them to the exported Excel workbook
(robustness_diagnostics.xlsx has no T3 sheet), so the numbers exist only as
console printouts / a manifest entry and get manually transcribed elsewhere.

This wrapper imports Task 3's exact logic from robustness_diagnostics.py
(load_base_data + task3_consumer_discretionary_only; it does NOT reimplement the
sort or the factor regressions and does NOT modify any frozen script) and
persists the result to its own workbook.

Expected (context): FF3 = 2.4802%/mo, t = 3.359; FF5 = 2.6084%/mo, t = 3.740; N = 88.

Output: output/cd_only_alpha.xlsx
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
from robustness_diagnostics import (
    load_base_data,
    task3_consumer_discretionary_only,   # exact Task 3 logic, imported read-only
)

OUTPUT_DIR = av2.OUTPUT_DIR
XLSX_PATH = os.path.join(OUTPUT_DIR, "cd_only_alpha.xlsx")

SPEC_ORDER = ["FF3", "FF3+Mom", "FF5"]


def main():
    df_filtered, returns, ff_factors, panel, market_ret = load_base_data()

    # Re-run Task 3 exactly as robustness_diagnostics.main() does.
    out = task3_consumer_discretionary_only(panel, ff_factors)

    # ---- Tidy per-specification alpha table (one row per factor model) ----
    alpha_rows = []
    for spec in SPEC_ORDER:
        if f"{spec}_alpha_pct" not in out:
            continue
        alpha_rows.append({
            "Sample": "Consumer Discretionary only",
            "Portfolio": "Q1-Q4 (adjusted-RRR, re-quartiled within CD)",
            "Model": spec,
            "Alpha_pct_per_month": out[f"{spec}_alpha_pct"],
            "t_stat": out[f"{spec}_t"],
            "p_value": out[f"{spec}_p"],
            "N_months": out[f"{spec}_n_obs"],
        })
    alpha_df = pd.DataFrame(alpha_rows)

    # ---- Sample-composition companion table ----
    meta_df = pd.DataFrame([{
        "n_cd_firms": out["n_cd_firms"],
        "n_total_firms": out["n_total_firms"],
        "pct_of_sample": out["pct_of_sample"],
        "avg_firms_per_quarter": out["avg_n_per_quarter"],
        "avg_firms_Q1": out["avg_n_q1"],
        "avg_firms_Q4": out["avg_n_q4"],
    }])

    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as xw:
        alpha_df.to_excel(xw, sheet_name="CD_only_alpha", index=False)
        meta_df.to_excel(xw, sheet_name="CD_only_sample", index=False)

    print("\n" + "=" * 80)
    print("  CONSUMER-DISCRETIONARY-ONLY ALPHA -- now persisted to workbook")
    print("=" * 80)
    print(alpha_df.to_string(index=False))
    print(f"\n  Workbook written: {XLSX_PATH}")

    # Guardrail: confirm the persisted numbers match the known console figures.
    ff3 = alpha_df.loc[alpha_df["Model"] == "FF3"].iloc[0]
    ff5 = alpha_df.loc[alpha_df["Model"] == "FF5"].iloc[0]
    assert abs(ff3["Alpha_pct_per_month"] - 2.4802) < 5e-3, ff3["Alpha_pct_per_month"]
    assert abs(ff3["t_stat"] - 3.359) < 5e-3, ff3["t_stat"]
    assert abs(ff5["Alpha_pct_per_month"] - 2.6084) < 5e-3, ff5["Alpha_pct_per_month"]
    assert abs(ff5["t_stat"] - 3.740) < 5e-3, ff5["t_stat"]
    print("  [CHECK] Persisted FF3/FF5 CD-only alphas match the console figures.")


if __name__ == "__main__":
    main()
