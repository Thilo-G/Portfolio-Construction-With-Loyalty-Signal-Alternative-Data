"""
firm_fe_persistence.py -- RRR Financial Implications
====================================================
Adds a FIRM-FIXED-EFFECTS specification to the Sloan-style persistence test in
identification_battery.py Task 4 (_persistence_regression), which currently uses
QUARTER fixed effects + firm-clustered SEs but NO firm fixed effects.

Motivation: without firm FE, the finding "retention-driven revenue predicts
future operating profitability (and future revenue growth) more than
acquisition-driven revenue" could reflect a stable, permanent cross-firm
difference (large established firms both retain better AND are persistently more
profitable) rather than a true WITHIN-firm effect. Firm FE strip out every
time-invariant firm characteristic and identify the retention-vs-acquisition gap
from within-firm variation only.

This script does NOT modify identification_battery.py or analysis_v2.py. It
imports their data loaders, the Sloan panel builder, the winsorizer, the two-way
cluster helper, and the Fama-MacBeth averager, then estimates, for BOTH DVs
(future operating ROA OpInc(t+1)/Assets(t) and future revenue growth RG(t+1)):

    (1) Quarter FE only  (the existing Task 4 spec, reproduced via the imported
        _persistence_regression for an exact apples-to-apples baseline)
    (2) Firm FE only     (within-firm variation)
    (3) Firm + Quarter FE (two-way; within-firm, time-demeaned)

with firm-clustered and two-way (firm, quarter) clustered SEs and a formal test
that the retention coefficient exceeds the acquisition coefficient. Same
1/99-winsorization as the existing test.

Output: output/firm_fe_persistence.xlsx
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

CODE_DIR = (
    r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)"
    r"\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM"
)
sys.path.insert(0, CODE_DIR)

import analysis_v2 as av2
from identification_battery import (
    load_base_data,
    _sloan_panel,
    _persistence_regression,     # existing quarter-FE baseline (reused, not reimplemented)
    winsorize_series,
    ols_two_way_cluster,
    FM_MIN_CROSS_N,
)

OUTPUT_DIR = av2.OUTPUT_DIR
XLSX_PATH = os.path.join(OUTPUT_DIR, "firm_fe_persistence.xlsx")

# DV specs identical to identification_battery.task4_sloan_persistence
SPECS = [
    ("RG_LEAD",         "Future revenue growth RG(t+1) [%]",              "RET_INTENS", "ACQ_INTENS"),
    ("OPINC_ROA_LEAD",  "Future operating ROA OpInc(t+1)/Assets(t) [%]",  "RET_INTENS", "ACQ_INTENS"),
]


def _fe_persistence(dat, dv, x_ret, x_acq, winsor, firm_fe, quarter_fe):
    """Pooled OLS of dv on the retention (x_ret) and acquisition (x_acq)
    intensities with optional firm and/or quarter dummy fixed effects, firm-
    clustered SEs (m1) and two-way (firm, quarter) clustered SEs (m2), plus a
    formal (x_ret - x_acq) contrast under each covariance. Mirrors
    identification_battery._persistence_regression exactly except for WHICH
    dummy fixed effects are absorbed. Same 1/99 winsorization convention."""
    cols = [dv, x_ret, x_acq, "FIRM", "QP", "QUARTER"]
    s = dat[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if winsor:
        for c in [dv, x_ret, x_acq]:
            s[c], _, _ = winsorize_series(s[c])

    blocks = [s[[x_ret, x_acq]].reset_index(drop=True)]
    if firm_fe:
        fd = pd.get_dummies(s["FIRM"].astype(str), prefix="F", drop_first=True).astype(float)
        blocks.append(fd.reset_index(drop=True))
    if quarter_fe:
        qd = pd.get_dummies(s["QP"].astype(str), prefix="Q", drop_first=True).astype(float)
        blocks.append(qd.reset_index(drop=True))
    X = pd.concat(blocks, axis=1)
    y = s[dv].reset_index(drop=True)
    Xc = sm.add_constant(X)
    fc = pd.factorize(s["FIRM"])[0]
    qc = pd.factorize(s["QP"].astype(str))[0]

    m1 = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": fc})
    m2, two_way_label = ols_two_way_cluster(y, X, fc, qc)

    names = list(Xc.columns)
    R = np.zeros(len(names)); R[names.index(x_ret)] = 1.0; R[names.index(x_acq)] = -1.0
    d1 = m1.t_test(R); d2 = m2.t_test(R)

    return {
        "dv": dv, "winsor": winsor, "firm_fe": firm_fe, "quarter_fe": quarter_fe,
        "N": int(m1.nobs), "n_firms": int(s["FIRM"].nunique()),
        "b_ret": float(m1.params[x_ret]), "t_ret": float(m1.tvalues[x_ret]),
        "p_ret": float(m1.pvalues[x_ret]),
        "b_acq": float(m1.params[x_acq]), "t_acq": float(m1.tvalues[x_acq]),
        "p_acq": float(m1.pvalues[x_acq]),
        "diff_firm": float(d1.effect[0]), "diff_firm_t": float(d1.tvalue),
        "diff_firm_p": float(d1.pvalue),
        "diff_2w": float(d2.effect[0]), "diff_2w_t": float(d2.tvalue),
        "diff_2w_p": float(d2.pvalue), "two_way_label": two_way_label,
    }


def _fe_label(firm_fe, quarter_fe):
    if firm_fe and quarter_fe:
        return "Firm + Quarter FE"
    if firm_fe:
        return "Firm FE only"
    return "Quarter FE only"


def main():
    df_filtered, returns, ff, panel, market_ret = load_base_data()
    dat = _sloan_panel(df_filtered)

    rows = []
    print("\n" + "=" * 80)
    print("FIRM-FE ROBUSTNESS OF THE SLOAN PERSISTENCE TEST (retention vs acquisition)")
    print("=" * 80)

    for dv, dv_label, xr, xa in SPECS:
        print(f"\nDV: {dv_label}")
        for winsor in [False, True]:
            wl = "1/99-winsorized" if winsor else "raw"

            # (1) Quarter FE only -- reproduce the EXISTING Task 4 spec exactly.
            base = _persistence_regression(dat, dv, xr, xa, winsor)
            specs_out = [
                ("Quarter FE only", False, True, base["b_ret"], base["t_ret"], base["p_ret"],
                 base["b_acq"], base["t_acq"], base["p_acq"],
                 base["diff_firm"], base["diff_firm_t"], base["diff_firm_p"],
                 base["diff_2w"], base["diff_2w_t"], base["diff_2w_p"],
                 base["two_way_label"], base["N"], None),
            ]

            # (2) Firm FE only; (3) Firm + Quarter FE.
            for firm_fe, quarter_fe in [(True, False), (True, True)]:
                r = _fe_persistence(dat, dv, xr, xa, winsor, firm_fe, quarter_fe)
                specs_out.append((
                    _fe_label(firm_fe, quarter_fe), firm_fe, quarter_fe,
                    r["b_ret"], r["t_ret"], r["p_ret"], r["b_acq"], r["t_acq"], r["p_acq"],
                    r["diff_firm"], r["diff_firm_t"], r["diff_firm_p"],
                    r["diff_2w"], r["diff_2w_t"], r["diff_2w_p"], r["two_way_label"],
                    r["N"], r["n_firms"],
                ))

            for (fe_label, firm_fe, quarter_fe, b_ret, t_ret, p_ret, b_acq, t_acq, p_acq,
                 dfirm, dfirm_t, dfirm_p, d2w, d2w_t, d2w_p, twlab, N, nfirms) in specs_out:
                survives = (b_ret > b_acq) and (dfirm_p < 0.05)
                verdict = "PASS (ret>acq, diff sig.)" if survives else \
                          ("sign-only (ret>acq, diff n.s.)" if b_ret > b_acq else "FAIL (ret<=acq)")
                print(f"  [{wl:<15} | {fe_label:<18}] N={N} "
                      f"ret b={b_ret:.4f} (t={t_ret:.2f}); acq b={b_acq:.4f} (t={t_acq:.2f}); "
                      f"diff={dfirm:.4f} (t={dfirm_t:.2f}, p=[{dfirm_p:.3f}]) -> {verdict}")
                rows.append({
                    "DV": dv, "DV_label": dv_label, "Winsor": wl, "FE": fe_label,
                    "firm_fe": firm_fe, "quarter_fe": quarter_fe, "N": N, "n_firms": nfirms,
                    "b_retention": b_ret, "t_retention": t_ret, "p_retention": p_ret,
                    "b_acquisition": b_acq, "t_acquisition": t_acq, "p_acquisition": p_acq,
                    "diff_ret_minus_acq_firmcl": dfirm, "diff_t_firmcl": dfirm_t,
                    "diff_p_firmcl": dfirm_p,
                    "diff_twoway": d2w, "diff_t_twoway": d2w_t, "diff_p_twoway": d2w_p,
                    "two_way_label": twlab, "survives_firmFE_gap": verdict,
                })

    out = pd.DataFrame(rows)
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="firm_fe_persistence", index=False)
    print(f"\n  Workbook written: {XLSX_PATH}")

    # Focal summary: does the gap survive firm FE for the profitability leg?
    print("\n  --- Focal read: retention-vs-acquisition gap under firm FE ---")
    for dv, dv_label, xr, xa in SPECS:
        sub = out[(out["DV"] == dv) & (out["Winsor"] == "1/99-winsorized")]
        for _, r in sub.iterrows():
            print(f"    {dv_label[:38]:<38} [{r['FE']:<18}] "
                  f"diff={r['diff_ret_minus_acq_firmcl']:.4f} "
                  f"(t={r['diff_t_firmcl']:.2f}, p=[{r['diff_p_firmcl']:.3f}])")


if __name__ == "__main__":
    main()
