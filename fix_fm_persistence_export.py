"""
fix_fm_persistence_export.py -- RRR Financial Implications
=========================================================
Export-gap fix for identification_battery.py Task 4 (_persistence_regression).
The Fama-MacBeth version of the Sloan persistence test (the `fm` dict inside
_persistence_regression: quarter-by-quarter cross-sectional OLS of a future
outcome on the retention and acquisition intensities, then a Newey-West average
of the per-quarter retention-minus-acquisition difference) is computed and
printed to the console, but its `fm` dict is never appended to the exported
T4_sloan_persistence sheet, so the headline FM difference (diff = 1.7845,
t = 5.92, p < .001, T = 30) exists only as an unlogged console printout that gets
manually transcribed elsewhere.

This wrapper imports Task 4's exact computation from identification_battery.py
(load_base_data + _sloan_panel + _persistence_regression; it does NOT reimplement
the persistence regression and does NOT modify any frozen script) and persists
the Fama-MacBeth retention, acquisition, and difference statistics for every DV x
winsorization cell to its own workbook.

Output: output/fm_persistence.xlsx
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
from identification_battery import (
    load_base_data,
    _sloan_panel,
    _persistence_regression,   # its `fm` dict is the object the T4 sheet drops
)

OUTPUT_DIR = av2.OUTPUT_DIR
XLSX_PATH = os.path.join(OUTPUT_DIR, "fm_persistence.xlsx")

# Same DV specs as identification_battery.task4_sloan_persistence
SPECS = [
    ("RG_LEAD",        "Future revenue growth RG(t+1) [%]",             "RET_INTENS", "ACQ_INTENS"),
    ("OPINC_ROA_LEAD", "Future operating ROA OpInc(t+1)/Assets(t) [%]", "RET_INTENS", "ACQ_INTENS"),
]


def main():
    df_filtered, returns, ff, panel, market_ret = load_base_data()
    dat = _sloan_panel(df_filtered)

    rows = []
    print("\n" + "=" * 80)
    print("  FAMA-MACBETH PERSISTENCE (retention vs acquisition) -- now persisted")
    print("=" * 80)
    for dv, dv_label, xr, xa in SPECS:
        for winsor in [False, True]:
            wl = "1/99-winsorized" if winsor else "raw"
            r = _persistence_regression(dat, dv, xr, xa, winsor)
            fm = r["fm"]
            if fm is None:
                print(f"  [{dv_label} | {wl}] FM: insufficient cross-sections")
                continue
            ret, acq, diff = fm[xr], fm[xa], fm["DIFF"]
            T = fm["_T"]
            print(f"  [{dv_label[:38]:<38} | {wl:<15}] "
                  f"FM diff(ret-acq)={diff['coef']:.4f} (t={diff['t']:.2f}, "
                  f"p=[{diff['p']:.3f}], T={T})")
            rows.append({
                "DV": dv, "DV_label": dv_label, "Winsor": wl, "T_quarters": T,
                "fm_retention_coef": ret["coef"], "fm_retention_se": ret["se"],
                "fm_retention_t": ret["t"], "fm_retention_p": ret["p"],
                "fm_acquisition_coef": acq["coef"], "fm_acquisition_se": acq["se"],
                "fm_acquisition_t": acq["t"], "fm_acquisition_p": acq["p"],
                "fm_diff_coef": diff["coef"], "fm_diff_se": diff["se"],
                "fm_diff_t": diff["t"], "fm_diff_p": diff["p"],
            })

    fm_df = pd.DataFrame(rows)
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as xw:
        fm_df.to_excel(xw, sheet_name="T4_FM_persistence", index=False)
    print(f"\n  Workbook written: {XLSX_PATH}")

    # Guardrail: the headline FM difference (winsorized future operating ROA)
    # must reproduce diff=1.7845, t=5.92, T=30.
    focal = fm_df[(fm_df["DV"] == "OPINC_ROA_LEAD") & (fm_df["Winsor"] == "1/99-winsorized")]
    assert len(focal) == 1, "focal FM persistence row not found"
    f = focal.iloc[0]
    print(f"\n  [FOCAL] Winsorized future operating ROA: FM diff={f['fm_diff_coef']:.4f}, "
          f"t={f['fm_diff_t']:.2f}, p=[{f['fm_diff_p']:.3f}], T={int(f['T_quarters'])}")
    assert abs(f["fm_diff_coef"] - 1.7845) < 5e-3, f["fm_diff_coef"]
    assert abs(f["fm_diff_t"] - 5.92) < 5e-2, f["fm_diff_t"]
    assert int(f["T_quarters"]) == 30, f["T_quarters"]
    print("  [CHECK] Reproduced the headline FM persistence diff (1.7845, t=5.92, T=30).")


if __name__ == "__main__":
    main()
