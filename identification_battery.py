"""
identification_battery.py -- RRR Financial Implications
=========================================================
Revenue Retention Rates & Stock Prices: High Returns, Low Risk

Identification / confound battery for the corrected-timing (k=2-month formation
gap, 3-month hold, contemporaneous adjusted-RRR signal, value-weighted on
formation-date market cap) long-short RRR portfolio. Six independent tests:

  Task 1  QMJ / BAB factor regressions on the long-short (the decisive test of
          whether "high return, low risk" is a genuine alpha or a repackaged
          quality / low-beta tilt): FF3+QMJ, FF5+QMJ, FF5+BAB, FF5+QMJ+BAB.
  Task 2  Does lagged RRR predict a firm's OWN future 36-month market beta?
          (Is the low-beta pattern a documented consequence of the risk channel?)
  Task 3  RRR x Size and RRR x BTM 2x3 conditional (dependent) double sorts.
  Task 4  Sloan-style persistence: retention-driven vs acquisition-driven revenue
          components predicting future revenue growth and future operating income,
          with a FORMAL test that the retention coefficient exceeds the
          acquisition coefficient.
  Task 5  RRR-vs-AR Fama-MacBeth horse race with AR winsorized at 1st/99th pct.
  Task 6  Fama-MacBeth (holding-window design) adding a revenue-growth control to
          the RRR + size + BTM + profit-margin specification.

Design principle: this file DOES NOT reimplement any timing or panel logic. It
imports the verified source of truth from analysis_v2.py (build_holding_panel,
build_portfolio_returns, run_factor_regressions, value_weighted_market,
safe_quartile, safe_tercile, newey_west_ols, phase1b_load_ff_factors,
phase1c_load_monthly_returns) and the committed AQR loader (load_aqr_factors.py).
df_filtered is read from the persisted canonical-panel snapshot (read-only) rather
than re-running phase1_load_and_diagnose(), to avoid side-effect writes to shared
files in output/ while other work proceeds in parallel.

No significance stars anywhere; exact p-values are reported in brackets.

Run:
    C:\\Users\\thkraft\\AppData\\Local\\Programs\\Python\\Python311\\python.exe identification_battery.py
"""

# =============================================================================
# Cell 1: imports
# =============================================================================
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as sps

warnings.filterwarnings("ignore")

CODE_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM"
sys.path.insert(0, CODE_DIR)

import analysis_v2 as av2  # noqa: E402 -- import must follow sys.path setup
from load_aqr_factors import (  # noqa: E402
    load_us_monthly_factor,
    QMJ_FILE_PATH, QMJ_SHEET_NAME,
    BAB_FILE_PATH, BAB_SHEET_NAME,
)

# =============================================================================
# Cell 2: constants
# =============================================================================
OUTPUT_DIR = av2.OUTPUT_DIR
CANONICAL_PANEL_PATH = os.path.join(OUTPUT_DIR, "canonical_panel_20260719.xlsx")
RESULTS_XLSX = os.path.join(OUTPUT_DIR, "identification_battery.xlsx")

FF3_FACTORS = ["Mkt-RF", "SMB", "HML"]
FF5_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

LS_COL = "Q1-Q4"                 # long-short column produced by build_portfolio_returns
BETA_WINDOW_MONTHS = 36          # rolling window for firm-level market beta (Task 2)
BETA_MIN_OBS = 24                # minimum monthly obs required inside a beta window
WINSOR_LOWER, WINSOR_UPPER = 1.0, 99.0   # winsorization percentiles for AR (Task 5)
FM_MIN_CROSS_N = 10              # minimum cross-sectional firms per quarter in a FM regression


# =============================================================================
# Cell 3: generic helpers (inference), all matched to analysis_v2 conventions
# =============================================================================
def winsorize_series(s, lower_pct=WINSOR_LOWER, upper_pct=WINSOR_UPPER):
    """Clip a numeric series at the given lower/upper percentiles (pooled)."""
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = np.nanpercentile(s.dropna(), [lower_pct, upper_pct])
    return s.clip(lower=lo, upper=hi), lo, hi


def nw_maxlag(T):
    """Newey-West lag truncation, identical rule to analysis_v2 (floor(4*(T/100)^(2/9)))."""
    return max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))


def nw_mean_tstat(x, maxlag):
    """Newey-West (Bartlett) mean, SE, t, p for a single time series at a GIVEN lag
    truncation. Used to make the forward-beta Fama-MacBeth overlap-aware (the 36m
    windows sampled quarterly overlap by ~11 quarters, so the default short lag
    understates the SE)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    T = len(x)
    mu = x.mean()
    e = x - mu
    s = (e ** 2).mean()
    for j in range(1, min(maxlag, T - 1) + 1):
        gj = (e[j:] * e[:-j]).mean()
        s += 2 * (1 - j / (maxlag + 1)) * gj
    se = np.sqrt(s / T)
    t = mu / se if se > 0 else np.nan
    p = float(2 * sps.t.sf(abs(t), max(T - 1, 1))) if np.isfinite(t) else np.nan
    return float(mu), float(se), float(t), p, int(T)


def fm_average(coef_df, coef_names):
    """Time-series average of per-period cross-sectional coefficients with
    Newey-West (Bartlett) standard errors. Byte-for-byte the same estimator as
    analysis_v2.run_fama_macbeth (same kernel, same lag rule, same p-value df)."""
    T = len(coef_df)
    avg = coef_df[coef_names].mean()
    max_lag = nw_maxlag(T)
    out = {}
    for var in coef_names:
        series = coef_df[var] - avg[var]
        gamma_sum = (series ** 2).mean()
        for j in range(1, max_lag + 1):
            gamma_j = (series.iloc[j:].values * series.iloc[:-j].values).mean()
            gamma_sum += 2 * (1 - j / (max_lag + 1)) * gamma_j
        se = np.sqrt(gamma_sum / T)
        t = avg[var] / se if se > 0 else np.nan
        p = float(2 * sps.t.sf(abs(t), max(T - 1, 1))) if np.isfinite(t) else np.nan
        out[var] = {"coef": float(avg[var]), "se": float(se), "t": float(t),
                    "p": p, "T": int(T)}
    return out


def run_cross_sectional_fm(hw, specs, y_var="EXCESS_RET_HW", min_n=FM_MIN_CROSS_N):
    """Quarterly cross-sectional OLS on a firm-quarter holding-window table, then a
    Newey-West time-series average of the slopes. Mirrors analysis_v2.run_fama_macbeth.
    Returns {spec_name: {var: {coef, se, t, p, T}}, ..., '_T': int, '_avgN': float}."""
    results = {}
    for spec_name, x_vars in specs.items():
        period_coefs = []
        for qtr, grp in hw.groupby("QUARTER"):
            sub = grp[[y_var] + x_vars].dropna()
            if len(sub) < min_n:
                continue
            try:
                m = sm.OLS(sub[y_var], sm.add_constant(sub[x_vars])).fit()
                d = m.params.to_dict()
                d["QUARTER"] = qtr
                d["N"] = len(sub)
                period_coefs.append(d)
            except Exception:
                continue
        if not period_coefs:
            results[spec_name] = None
            continue
        coef_df = pd.DataFrame(period_coefs)
        coef_names = [c for c in coef_df.columns if c not in ("QUARTER", "N")]
        stats = fm_average(coef_df, coef_names)
        stats["_T"] = len(coef_df)
        stats["_avgN"] = float(coef_df["N"].mean())
        results[spec_name] = stats
    return results


def ls_factor_alpha(y_series, factor_panel, factor_lists):
    """Newey-West (HAC) factor regressions of a zero-cost long-short return series.
    The long-short is already dollar-neutral, so RF is NOT subtracted (identical
    convention to analysis_v2.run_factor_regressions for Q1-Q4). Returns a dict
    keyed by spec name with alpha, t, p, betas (+ t) for every factor, R2, N."""
    y = y_series.copy()
    y.index = pd.to_datetime(y.index).to_period("M").to_timestamp("M")
    out = {}
    for spec_name, factors in factor_lists.items():
        X = factor_panel[factors].dropna()
        yy = y.reindex(X.index).dropna()
        X = X.reindex(yy.index)
        if len(yy) < 12:
            out[spec_name] = None
            continue
        m = av2.newey_west_ols(yy, X)
        rec = {
            "alpha": float(m.params["const"]),
            "alpha_t": float(m.tvalues["const"]),
            "alpha_p": float(m.pvalues["const"]),
            "alpha_se": float(m.bse["const"]),
            "r2": float(m.rsquared),
            "n": int(m.nobs),
        }
        for f in factors:
            rec[f"beta_{f}"] = float(m.params[f])
            rec[f"t_{f}"] = float(m.tvalues[f])
            rec[f"p_{f}"] = float(m.pvalues[f])
        out[spec_name] = rec
    return out


def ols_two_way_cluster(y, X, firm_codes, quarter_codes):
    """OLS with two-way (firm and quarter) clustered covariance where supported,
    else fall back to one-way firm clustering. Returns (model, cluster_label)."""
    X = sm.add_constant(X)
    try:
        groups = np.column_stack([np.asarray(firm_codes), np.asarray(quarter_codes)])
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        return m, "two-way (firm, quarter)"
    except Exception:
        m = sm.OLS(y, X).fit(cov_type="cluster",
                             cov_kwds={"groups": np.asarray(firm_codes)})
        return m, "one-way (firm)"


# =============================================================================
# Cell 4: base data loader (read-only; reuses analysis_v2 source of truth)
# =============================================================================
def load_base_data():
    """Read df_filtered from the canonical snapshot, then build the verified holding
    panel and the value-weighted market via analysis_v2. No side-effect writes."""
    print("=" * 80)
    print("LOAD BASE DATA (read-only canonical snapshot + analysis_v2 panel builders)")
    print("=" * 80)

    df_filtered = pd.read_excel(CANONICAL_PANEL_PATH, sheet_name=0, engine="openpyxl")
    df_filtered["DATE"] = pd.to_datetime(df_filtered["DATE"])
    df_filtered = df_filtered.sort_values(["FIRM", "DATE"]).set_index(["FIRM", "DATE"])
    n_firms = df_filtered.index.get_level_values("FIRM").nunique()
    print(f"  df_filtered: {df_filtered.shape[0]} firm-quarters, {n_firms} firms")
    assert n_firms == av2.EXPECTED_N_FIRMS, (
        f"Canonical panel has {n_firms} firms, expected {av2.EXPECTED_N_FIRMS}."
    )

    ff = av2.phase1b_load_ff_factors()
    returns = av2.phase1c_load_monthly_returns()
    panel = av2.build_holding_panel(df_filtered, returns)

    panel["ADJ_RRR_PCT"] = pd.to_numeric(panel["ADJ_RRR_PCT"], errors="coerce")
    panel["RRR_Q_ADJ"] = panel.groupby("QUARTER")["ADJ_RRR_PCT"].transform(av2.safe_quartile)
    market_ret = av2.value_weighted_market(panel)

    return df_filtered, returns, ff, panel, market_ret


def build_factor_panel(ff):
    """Monthly factor panel indexed by month-end: FF5 factors (+RF) merged with
    AQR US QMJ and BAB long/short factor returns (already decimal, already zero-cost)."""
    qmj = load_us_monthly_factor(QMJ_FILE_PATH, QMJ_SHEET_NAME, "QMJ")
    bab = load_us_monthly_factor(BAB_FILE_PATH, BAB_SHEET_NAME, "BAB")
    aqr = qmj.merge(bab, on="date", how="inner")
    # Same DatetimeIndex -> month-period -> month-timestamp transform used for the
    # FF panel and the long-short return index, so all three align exactly.
    aqr.index = pd.DatetimeIndex(aqr["date"]).to_period("M").to_timestamp("M")
    aqr = aqr[["QMJ", "BAB"]]

    ffm = ff.copy()
    ffm.index = pd.to_datetime(ffm.index).to_period("M").to_timestamp("M")
    fp = ffm.join(aqr, how="inner")
    return fp


def build_holding_window_table(panel, extra_first_cols=None):
    """Collapse the monthly holding panel to ONE compounded 3-month holding-window
    EXCESS return per (firm, signal-quarter), carrying the contemporaneous
    quarter-end signals/controls. This reproduces the aggregation inside
    analysis_v2.run_fama_macbeth and is the DV for every Fama-MacBeth spec here."""
    ff = FF_FACTORS_GLOBAL
    rf = ff["RF"].copy()
    rf.index = pd.to_datetime(rf.index).to_period("M")

    df = panel.copy()
    df["MONTH_P"] = pd.to_datetime(df["Date"]).dt.to_period("M")
    df["RF_M"] = df["MONTH_P"].map(rf)
    df["RET_SIMPLE"] = pd.to_numeric(df["RET_SIMPLE"], errors="coerce")

    first_cols = ["ADJ_RRR_PCT", "RRR_PCT", "ACQ_RATE_PCT", "ADJ_ACQ_RATE_PCT",
                  "SIZE", "BTM", "PM_OPER_PCT", "REV_GROWTH_PCT", "SECTOR"]
    if extra_first_cols:
        first_cols += [c for c in extra_first_cols if c not in first_cols]
    for c in first_cols:
        if c != "SECTOR" and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = {"n": ("RET_SIMPLE", "size"),
           "gross": ("RET_SIMPLE", lambda s: (1 + s).prod()),
           "gross_rf": ("RF_M", lambda s: (1 + s).prod())}
    for c in first_cols:
        if c in df.columns:
            agg[c] = (c, "first")

    hw = (df.sort_values(["FIRM", "QUARTER", "HOLD_IDX"])
            .groupby(["FIRM", "QUARTER"]).agg(**agg).reset_index())
    hw = hw[hw["n"] == av2.HOLD_MONTHS].copy()          # require full 3-month window
    hw["EXCESS_RET_HW"] = hw["gross"] - hw["gross_rf"]
    return hw


# Populated in main() so build_holding_window_table can reach the FF frame.
FF_FACTORS_GLOBAL = None


# =============================================================================
# Cell 5: TASK 1 -- QMJ / BAB factor regressions on the long-short
# =============================================================================
def task1_qmj_bab(panel, ff, market_ret):
    print("\n" + "=" * 80)
    print("TASK 1  QMJ / BAB factor regressions on the ADJ-RRR long-short (Q1-Q4, VW)")
    print("=" * 80)

    port = av2.build_portfolio_returns(panel, "RRR_Q_ADJ", market_ret)
    ls = port[LS_COL].dropna()
    fp = build_factor_panel(ff)

    factor_lists = {
        "FF3 (ref)":     FF3_FACTORS,
        "FF5 (ref)":     FF5_FACTORS,
        "FF3+QMJ":       FF3_FACTORS + ["QMJ"],
        "FF5+QMJ":       FF5_FACTORS + ["QMJ"],
        "FF5+BAB":       FF5_FACTORS + ["BAB"],
        "FF5+QMJ+BAB":   FF5_FACTORS + ["QMJ", "BAB"],
    }
    res = ls_factor_alpha(ls, fp, factor_lists)

    rows = []
    print(f"\n  {'Spec':<14}{'Alpha%/mo':>10}{'t':>8}{'p':>9}"
          f"{'bQMJ':>8}{'tQMJ':>7}{'bBAB':>8}{'tBAB':>7}{'R2':>7}{'N':>5}")
    for spec, r in res.items():
        if r is None:
            continue
        bqmj = r.get("beta_QMJ", np.nan); tqmj = r.get("t_QMJ", np.nan)
        bbab = r.get("beta_BAB", np.nan); tbab = r.get("t_BAB", np.nan)
        print(f"  {spec:<14}{r['alpha']*100:>10.4f}{r['alpha_t']:>8.2f}"
              f"{r['alpha_p']:>9.3f}"
              f"{bqmj:>8.3f}{tqmj:>7.2f}{bbab:>8.3f}{tbab:>7.2f}"
              f"{r['r2']:>7.3f}{r['n']:>5d}")
        rows.append({
            "Spec": spec, "Alpha_pct_mo": r["alpha"] * 100, "Alpha_t": r["alpha_t"],
            "Alpha_p": r["alpha_p"], "Alpha_se_pct": r["alpha_se"] * 100,
            "beta_QMJ": bqmj, "t_QMJ": tqmj, "beta_BAB": bbab, "t_BAB": tbab,
            "R2": r["r2"], "N": r["n"],
        })

    print("\n  Decision rule: if the FF5+QMJ+BAB alpha t-stat < 2.0 the long-short is")
    print("  absorbed by the quality / low-beta factors; if it stays >= 2.0 it survives.")
    full = res["FF5+QMJ+BAB"]
    verdict = "ABSORBED (t<2.0)" if abs(full["alpha_t"]) < 2.0 else "SURVIVES (t>=2.0)"
    print(f"  --> FF5+QMJ+BAB alpha = {full['alpha']*100:.4f}%/mo, "
          f"t = {full['alpha_t']:.2f}  ==>  {verdict}")
    return pd.DataFrame(rows)


# =============================================================================
# Cell 6: TASK 2 -- does lagged RRR predict a firm's own future 36m market beta?
# =============================================================================
def _forward_beta_table(returns, df_filtered, ff, market_ret):
    """For every (firm, quarter-end Q), estimate the firm's CAPM market beta over the
    36 months strictly AFTER Q, against (a) the FF market and (b) the sample VW
    market. Merge with the contemporaneous quarter-end RRR / controls."""
    valid_firms = set(df_filtered.index.get_level_values("FIRM").str.upper())

    r = returns.copy()
    r["FIRM"] = r["FIRM"].str.upper()
    r = r[r["FIRM"].isin(valid_firms)].copy()
    r["MP"] = pd.to_datetime(r["Date"]).dt.to_period("M")
    r["RET_SIMPLE"] = pd.to_numeric(r["RET_SIMPLE"], errors="coerce")

    rf = ff["RF"].copy(); rf.index = pd.to_datetime(rf.index).to_period("M")
    mkt_ff = ff["Mkt-RF"].copy(); mkt_ff.index = pd.to_datetime(mkt_ff.index).to_period("M")
    smp = market_ret.copy(); smp.index = pd.to_datetime(smp.index).to_period("M")

    r["RF_M"] = r["MP"].map(rf)
    r["FIRM_EXC"] = r["RET_SIMPLE"] - r["RF_M"]
    mkt_smp_exc = (smp - rf.reindex(smp.index)).dropna()

    firm_exc = r.pivot_table(index="MP", columns="FIRM", values="FIRM_EXC")
    mkt_ff_exc = mkt_ff.reindex(firm_exc.index)
    mkt_smp_exc = mkt_smp_exc.reindex(firm_exc.index)

    def _beta(y, x):
        d = pd.concat([y, x], axis=1).dropna()
        if len(d) < BETA_MIN_OBS:
            return np.nan, len(d)
        yy = d.iloc[:, 0].values; xx = d.iloc[:, 1].values
        vx = np.var(xx)
        if vx <= 0:
            return np.nan, len(d)
        return float(np.cov(yy, xx, ddof=1)[0, 1] / np.var(xx, ddof=1)), len(d)

    dff = df_filtered.reset_index()
    dff["FIRM"] = dff["FIRM"].str.upper()
    dff["QP"] = pd.to_datetime(dff["DATE"]).dt.to_period("M")
    for c in ["RRR_PCT", "ADJ_RRR_PCT", "SIZE", "BTM"]:
        dff[c] = pd.to_numeric(dff[c], errors="coerce")

    rows = []
    for _, rec in dff.iterrows():
        firm, q = rec["FIRM"], rec["QP"]
        if firm not in firm_exc.columns:
            continue
        win = pd.period_range(q + 1, q + BETA_WINDOW_MONTHS, freq="M")
        win = win[win.isin(firm_exc.index)]
        if len(win) < BETA_MIN_OBS:
            continue
        y = firm_exc.loc[win, firm]
        b_ff, n_ff = _beta(y, mkt_ff_exc.loc[win])
        b_smp, n_smp = _beta(y, mkt_smp_exc.loc[win])
        rows.append({
            "FIRM": firm, "QUARTER": rec["DATE"], "QP": q,
            "BETA_FF_FWD": b_ff, "BETA_SMP_FWD": b_smp, "N_WIN": n_ff,
            "RRR_PCT": rec["RRR_PCT"], "ADJ_RRR_PCT": rec["ADJ_RRR_PCT"],
            "SIZE": rec["SIZE"], "BTM": rec["BTM"],
        })
    return pd.DataFrame(rows)


def task2_rrr_future_beta(returns, df_filtered, ff, market_ret):
    print("\n" + "=" * 80)
    print("TASK 2  Does lagged RRR predict a firm's own future 36-month market beta?")
    print("=" * 80)
    bt = _forward_beta_table(returns, df_filtered, ff, market_ret)
    print(f"  Forward-beta firm-quarter observations: {len(bt)} "
          f"({bt['FIRM'].nunique()} firms, {bt['QUARTER'].nunique()} quarters); "
          f"window={BETA_WINDOW_MONTHS}m, min obs={BETA_MIN_OBS}")
    print(f"  Mean forward beta (FF market): {bt['BETA_FF_FWD'].mean():.4f}; "
          f"(sample VW market): {bt['BETA_SMP_FWD'].mean():.4f}")

    rows = []
    # Pooled OLS WITH quarter fixed effects (isolates the cross-sectional relation,
    # the panel analogue of the Fama-MacBeth below) and two-way (firm, quarter)
    # clustered SEs. Without quarter FE the estimate is contaminated by the aggregate
    # beta cycle (all betas comove over the market cycle, unrelated to RRR).
    for beta_col, mkt_label in [("BETA_FF_FWD", "FF market"),
                                ("BETA_SMP_FWD", "sample VW market")]:
        for sig in ["ADJ_RRR_PCT", "RRR_PCT"]:
            for controls, clab in [([], "no controls"),
                                   (["SIZE", "BTM"], "+SIZE +BTM")]:
                sub = (bt[[beta_col, sig, "FIRM", "QP"] + controls]
                       .replace([np.inf, -np.inf], np.nan).dropna())
                y = sub[beta_col].reset_index(drop=True)
                qd = pd.get_dummies(sub["QP"].astype(str), prefix="Q",
                                    drop_first=True).astype(float).reset_index(drop=True)
                X = pd.concat([sub[[sig] + controls].reset_index(drop=True), qd], axis=1)
                fc = pd.factorize(sub["FIRM"])[0]
                qc = pd.factorize(sub["QP"].astype(str))[0]
                m, clabused = ols_two_way_cluster(y, X, fc, qc)
                coef = m.params[sig]; tval = m.tvalues[sig]; pval = m.pvalues[sig]
                print(f"\n  Beta on {mkt_label:<16} | signal={sig:<12} | {clab:<11} "
                      f"| quarter FE | N={int(m.nobs)} | cluster={clabused}")
                print(f"    {sig}: coef={coef:.6f}  t={tval:.2f}  p=[{pval:.3f}]")
                rows.append({
                    "Beta_def": mkt_label, "Signal": sig, "Controls": clab + " + quarter FE",
                    "coef": float(coef), "t": float(tval), "p": float(pval),
                    "N": int(m.nobs), "cluster": clabused,
                })

    # Fama-MacBeth robustness on the ADJ signal / FF-market beta. Reported at TWO
    # lag truncations: the default short lag (which IGNORES the window overlap and is
    # therefore anti-conservative) and an overlap-aware lag equal to the window
    # length in quarters (36m/3 = 12), which is the honest correction for the fact
    # that adjacent quarterly forward betas share ~33 of their 36 months.
    overlap_lag = BETA_WINDOW_MONTHS // 3       # = 12 quarters of overlap
    fm_rows = []
    for qtr, grp in bt.dropna(subset=["BETA_FF_FWD", "ADJ_RRR_PCT"]).groupby("QUARTER"):
        if len(grp) < FM_MIN_CROSS_N:
            continue
        mm = sm.OLS(grp["BETA_FF_FWD"], sm.add_constant(grp[["ADJ_RRR_PCT"]])).fit()
        fm_rows.append({"slope": mm.params["ADJ_RRR_PCT"], "QUARTER": qtr, "N": len(grp)})
    if fm_rows:
        cdf = pd.DataFrame(fm_rows)
        slopes = cdf["slope"].values
        mu_d, _, t_d, p_d, T = nw_mean_tstat(slopes, nw_maxlag(len(slopes)))
        mu_o, _, t_o, p_o, _ = nw_mean_tstat(slopes, overlap_lag)
        print(f"\n  [FM robustness] Beta_FF_FWD ~ ADJ_RRR_PCT, cross-sectional by quarter (T={T}):")
        print(f"    default NW lag={nw_maxlag(T)} : coef={mu_d:.6f}  t={t_d:.2f}  p=[{p_d:.3f}]"
              f"   (anti-conservative: ignores window overlap)")
        print(f"    overlap-aware lag={overlap_lag}: coef={mu_o:.6f}  t={t_o:.2f}  p=[{p_o:.3f}]"
              f"   (honest correction for ~{overlap_lag}-quarter forward-window overlap)")
        rows.append({"Beta_def": "FF market (FM, default NW lag)", "Signal": "ADJ_RRR_PCT",
                     "Controls": f"NW lag={nw_maxlag(T)} (overlap-ignoring)", "coef": mu_d,
                     "t": t_d, "p": p_d, "N": int(cdf["N"].sum()), "cluster": "NW time-series"})
        rows.append({"Beta_def": "FF market (FM, overlap-aware lag)", "Signal": "ADJ_RRR_PCT",
                     "Controls": f"NW lag={overlap_lag} (overlap-consistent)", "coef": mu_o,
                     "t": t_o, "p": p_o, "N": int(cdf["N"].sum()), "cluster": "NW time-series"})
    return pd.DataFrame(rows)


# =============================================================================
# Cell 7: TASK 3 -- RRR x Size and RRR x BTM 2x3 conditional double sorts
# =============================================================================
def _conditional_double_sort(panel, cond_col, market_ret, ff):
    """2x3 dependent sort: within each QUARTER split firms into 2 halves on cond_col
    (below/above the quarter median), then sort ADJ_RRR into terciles WITHIN each
    half. Value-weighted (MCAP_FORM) tercile portfolios via build_portfolio_returns;
    the RRR long-short (T1-T3) is formed inside each half and then averaged across
    halves to give a cond-neutral RRR long-short. Returns FF3/FF5 alphas."""
    p = panel.copy()
    p[cond_col] = pd.to_numeric(p[cond_col], errors="coerce")
    p["ADJ_RRR_PCT"] = pd.to_numeric(p["ADJ_RRR_PCT"], errors="coerce")
    p = p.dropna(subset=[cond_col, "ADJ_RRR_PCT"])

    med = p.groupby("QUARTER")[cond_col].transform("median")
    p["HALF"] = np.where(p[cond_col] <= med, "LOW", "HIGH")

    factor_lists = {"FF3": FF3_FACTORS, "FF5": FF5_FACTORS}
    fp = ff.copy(); fp.index = pd.to_datetime(fp.index).to_period("M").to_timestamp("M")

    ls_by_half = {}
    out = {}
    for half in ["LOW", "HIGH"]:
        sub = p[p["HALF"] == half].copy()
        sub["RRR_T_COND"] = sub.groupby("QUARTER")["ADJ_RRR_PCT"].transform(av2.safe_tercile)
        port = av2.build_portfolio_returns(sub, "RRR_T_COND", market_ret)
        if port is None or "T1-T3" not in port.columns:
            continue
        ls = port["T1-T3"].dropna()
        ls_by_half[half] = ls
        out[half] = ls_factor_alpha(ls, fp, factor_lists)

    if "LOW" in ls_by_half and "HIGH" in ls_by_half:
        avg = pd.concat([ls_by_half["LOW"], ls_by_half["HIGH"]], axis=1).mean(axis=1).dropna()
        out["AVG"] = ls_factor_alpha(avg, fp, factor_lists)
    return out


def task3_double_sorts(panel, market_ret, ff):
    print("\n" + "=" * 80)
    print("TASK 3  RRR x Size and RRR x BTM 2x3 conditional (dependent) double sorts")
    print("=" * 80)
    rows = []
    for cond_col, cond_label in [("SIZE", "Size"), ("BTM", "BTM")]:
        print(f"\n  --- Conditioning on {cond_label} ({cond_col}); "
              f"RRR tercile long-short (T1-T3, high-minus-low RRR) within each half ---")
        res = _conditional_double_sort(panel, cond_col, market_ret, ff)
        for half_key, half_label in [("LOW", f"Low-{cond_label} half"),
                                     ("HIGH", f"High-{cond_label} half"),
                                     ("AVG", f"{cond_label}-neutral (avg of halves)")]:
            if half_key not in res:
                continue
            for spec in ["FF3", "FF5"]:
                r = res[half_key].get(spec)
                if r is None:
                    continue
                print(f"    {half_label:<34} {spec}: "
                      f"alpha={r['alpha']*100:>8.4f}%/mo  t={r['alpha_t']:>6.2f}  "
                      f"p=[{r['alpha_p']:.3f}]  N={r['n']}")
                rows.append({
                    "Conditioning": cond_label, "Cell": half_label, "Spec": spec,
                    "Alpha_pct_mo": r["alpha"] * 100, "t": r["alpha_t"],
                    "p": r["alpha_p"], "N": r["n"],
                })
    return pd.DataFrame(rows)


# =============================================================================
# Cell 8: TASK 4 -- Sloan-style persistence of revenue components
# =============================================================================
def _sloan_panel(df_filtered):
    """Build the firm-quarter Sloan panel with retention/acquisition revenue
    components (scaled) and the leads of future revenue growth and operating income.
    Adjacency is enforced by merging on QP (period) shifted by +/-1 quarter, so no
    non-consecutive quarters are ever paired."""
    d = df_filtered.reset_index().copy()
    d["FIRM"] = d["FIRM"].str.upper()
    d["QP"] = pd.to_datetime(d["DATE"]).dt.to_period("Q")
    for c in ["#RETURNING_CUSTOMERS", "#NEW_CUSTOMERS", "#TOTAL_REVENUE",
              "IS_OPER_INC", "BS_TOT_ASSET", "REV_GROWTH_PCT", "RRR_PCT"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # t-1 base variables (previous quarter): merge previous-quarter values onto row t
    prev = d[["FIRM", "QP", "#TOTAL_REVENUE", "BS_TOT_ASSET"]].copy()
    prev["QP"] = prev["QP"] + 1
    prev = prev.rename(columns={"#TOTAL_REVENUE": "TOTREV_LAG", "BS_TOT_ASSET": "ASSET_LAG"})
    d = d.merge(prev, on=["FIRM", "QP"], how="left")

    # t+1 leads (next quarter): merge next-quarter DV values onto row t
    nxt = d[["FIRM", "QP", "REV_GROWTH_PCT", "IS_OPER_INC"]].copy()
    nxt["QP"] = nxt["QP"] - 1
    nxt = nxt.rename(columns={"REV_GROWTH_PCT": "RG_LEAD", "IS_OPER_INC": "OPINC_LEAD"})
    d = d.merge(nxt, on=["FIRM", "QP"], how="left")

    # Revenue-scaled components (decompose current gross growth): sum ~ 1 + RG_t.
    # Numerator and denominator are BOTH the ALTD revenue proxy, so these are
    # dimensionless (O(1)) and mutually comparable.
    d["RET_INTENS"] = d["#RETURNING_CUSTOMERS"] / d["TOTREV_LAG"]     # == RRR_t (fraction)
    d["ACQ_INTENS"] = d["#NEW_CUSTOMERS"] / d["TOTREV_LAG"]           # new revenue / lagged total

    # Future operating profitability (GAAP), scaled by CURRENT total assets so the
    # DV is itself dimensionless (operating ROA, %). CRITICAL: the ALTD revenue
    # proxy (#RETURNING/#NEW_CUSTOMERS, median ~1.4e7) is on a DIFFERENT measurement
    # scale than the GAAP fundamentals (assets in $M), so the revenue components are
    # NOT scaled by GAAP assets (RET/ASSET has median ~6229, a meaningless ratio).
    # Both DVs are therefore regressed on the SAME dimensionless ALTD revenue-
    # composition intensities; only the future outcome differs.
    d["OPINC_ROA_LEAD"] = d["OPINC_LEAD"] / d["BS_TOT_ASSET"] * 100.0

    d["QUARTER"] = pd.to_datetime(d["DATE"])
    return d


def _persistence_regression(dat, dv, x_ret, x_acq, winsor):
    """Pooled OLS of dv on the retention component (x_ret), acquisition component
    (x_acq) and quarter fixed effects, with firm-clustered SEs, plus a two-way
    (firm+quarter) clustered version, and a formal test of (x_ret - x_acq).
    A Fama-MacBeth version is added as a second robustness estimator."""
    cols = [dv, x_ret, x_acq, "FIRM", "QP", "QUARTER"]
    # Drop inf (from zero/near-zero scaling denominators) before dropna, since
    # dropna alone does not remove +/-inf and statsmodels rejects inf in exog.
    s = dat[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if winsor:
        for c in [dv, x_ret, x_acq]:
            s[c], _, _ = winsorize_series(s[c])

    qd = pd.get_dummies(s["QP"].astype(str), prefix="Q", drop_first=True).astype(float)
    X = pd.concat([s[[x_ret, x_acq]].reset_index(drop=True),
                   qd.reset_index(drop=True)], axis=1)
    y = s[dv].reset_index(drop=True)
    Xc = sm.add_constant(X)
    fc = pd.factorize(s["FIRM"])[0]
    qc = pd.factorize(s["QP"].astype(str))[0]

    m1 = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": fc})
    m2, two_way_label = ols_two_way_cluster(y, X, fc, qc)

    # difference test x_ret - x_acq under each covariance
    names = list(Xc.columns)
    R = np.zeros(len(names)); R[names.index(x_ret)] = 1.0; R[names.index(x_acq)] = -1.0
    d1 = m1.t_test(R); d2 = m2.t_test(R)

    # Fama-MacBeth cross-sectional
    fm = []
    for qtr, grp in s.groupby("QP"):
        if len(grp) < FM_MIN_CROSS_N:
            continue
        mm = sm.OLS(grp[dv], sm.add_constant(grp[[x_ret, x_acq]])).fit()
        rec = mm.params.to_dict(); rec["DIFF"] = rec[x_ret] - rec[x_acq]
        rec["QUARTER"] = qtr; rec["N"] = len(grp)
        fm.append(rec)
    fm_stat = None
    if fm:
        cdf = pd.DataFrame(fm)
        fm_stat = fm_average(cdf, [x_ret, x_acq, "DIFF"])
        fm_stat["_T"] = len(cdf)

    return {
        "dv": dv, "x_ret": x_ret, "x_acq": x_acq, "winsor": winsor, "N": int(m1.nobs),
        "b_ret": float(m1.params[x_ret]), "t_ret": float(m1.tvalues[x_ret]),
        "p_ret": float(m1.pvalues[x_ret]),
        "b_acq": float(m1.params[x_acq]), "t_acq": float(m1.tvalues[x_acq]),
        "p_acq": float(m1.pvalues[x_acq]),
        "diff_firm": float(d1.effect[0]), "diff_firm_t": float(d1.tvalue),
        "diff_firm_p": float(d1.pvalue),
        "diff_2w": float(d2.effect[0]), "diff_2w_t": float(d2.tvalue),
        "diff_2w_p": float(d2.pvalue), "two_way_label": two_way_label,
        "fm": fm_stat,
    }


def task4_sloan_persistence(df_filtered):
    print("\n" + "=" * 80)
    print("TASK 4  Sloan-style persistence: retention vs acquisition revenue components")
    print("=" * 80)
    dat = _sloan_panel(df_filtered)

    # sanity: revenue-scaled components should sum to ~ 1 + RG_t
    chk = dat.dropna(subset=["RET_INTENS", "ACQ_INTENS", "REV_GROWTH_PCT"])
    approx = (chk["RET_INTENS"] + chk["ACQ_INTENS"]) - (1 + chk["REV_GROWTH_PCT"] / 100)
    print(f"  [sanity] max|RET_INTENS+ACQ_INTENS - (1+RG)| = {approx.abs().max():.2e} "
          f"(should be ~0; confirms the revenue decomposition)")

    specs = [
        ("Future revenue growth RG(t+1) [%] on revenue-composition intensities",
         "RG_LEAD", "RET_INTENS", "ACQ_INTENS"),
        ("Future operating ROA OpInc(t+1)/Assets(t) [%] on revenue-composition intensities",
         "OPINC_ROA_LEAD", "RET_INTENS", "ACQ_INTENS"),
    ]
    rows = []
    for label, dv, xr, xa in specs:
        print(f"\n  DV: {label}")
        for winsor in [False, True]:
            r = _persistence_regression(dat, dv, xr, xa, winsor)
            wl = "1/99-winsorized" if winsor else "raw"
            print(f"    [{wl:<15}] N={r['N']}  "
                  f"retention b={r['b_ret']:.4f} (t={r['t_ret']:.2f}, p=[{r['p_ret']:.3f}]);  "
                  f"acquisition b={r['b_acq']:.4f} (t={r['t_acq']:.2f}, p=[{r['p_acq']:.3f}])")
            print(f"                       diff(ret-acq) [firm-clustered] = "
                  f"{r['diff_firm']:.4f} (t={r['diff_firm_t']:.2f}, p=[{r['diff_firm_p']:.3f}]);  "
                  f"[{r['two_way_label']}] = {r['diff_2w']:.4f} "
                  f"(t={r['diff_2w_t']:.2f}, p=[{r['diff_2w_p']:.3f}])")
            if r["fm"] is not None:
                fmd = r["fm"]["DIFF"]
                print(f"                       diff(ret-acq) [Fama-MacBeth, T={r['fm']['_T']}] = "
                      f"{fmd['coef']:.4f} (t={fmd['t']:.2f}, p=[{fmd['p']:.3f}])")
            crit = "PASS" if (r["b_ret"] > r["b_acq"] and r["diff_firm_p"] < 0.05) else \
                   ("sign-only (ret>acq, diff n.s.)" if r["b_ret"] > r["b_acq"] else "FAIL (ret<=acq)")
            print(f"                       pre-registered criterion (b_ret>b_acq & diff sig.): {crit}")
            rows.append({
                "DV": dv, "Winsor": wl, "N": r["N"],
                "b_retention": r["b_ret"], "t_retention": r["t_ret"], "p_retention": r["p_ret"],
                "b_acquisition": r["b_acq"], "t_acquisition": r["t_acq"], "p_acquisition": r["p_acq"],
                "diff_ret_minus_acq_firmcl": r["diff_firm"], "diff_t_firmcl": r["diff_firm_t"],
                "diff_p_firmcl": r["diff_firm_p"],
                "diff_twoway": r["diff_2w"], "diff_t_twoway": r["diff_2w_t"],
                "diff_p_twoway": r["diff_2w_p"], "criterion": crit,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Cell 9: TASK 5 -- RRR vs AR Fama-MacBeth with AR winsorized (1st/99th)
# =============================================================================
def task5_ar_winsorized_fm(panel):
    print("\n" + "=" * 80)
    print("TASK 5  RRR-vs-AR Fama-MacBeth horse race with AR winsorized at 1st/99th pct")
    print("=" * 80)
    hw = build_holding_window_table(panel)

    # Winsorize RAW AR pooled across firm-quarters, then RE-DERIVE the industry x time
    # adjustment on the winsorized raw AR (denominator unchanged: new-rev / prior-new-rev).
    hw["ACQ_RATE_PCT_W"], lo, hi = winsorize_series(hw["ACQ_RATE_PCT"])
    print(f"  AR winsorization bounds (1st/99th pct of raw AR%): "
          f"[{lo:.2f}%, {hi:.2f}%]  (raw AR SD={hw['ACQ_RATE_PCT'].std():.1f}%, "
          f"winsorized SD={hw['ACQ_RATE_PCT_W'].std():.1f}%)")
    hw["ADJ_ACQ_RATE_PCT_W"] = hw["ACQ_RATE_PCT_W"] - hw.groupby(
        ["QUARTER", "SECTOR"])["ACQ_RATE_PCT_W"].transform("mean")

    specs = {
        "(A) Adj RRR only":              ["ADJ_RRR_PCT"],
        "(B) Adj AR raw only":           ["ADJ_ACQ_RATE_PCT"],
        "(C) Adj AR winsor only":        ["ADJ_ACQ_RATE_PCT_W"],
        "(D) Adj RRR + Adj AR raw":      ["ADJ_RRR_PCT", "ADJ_ACQ_RATE_PCT"],
        "(E) Adj RRR + Adj AR winsor":   ["ADJ_RRR_PCT", "ADJ_ACQ_RATE_PCT_W"],
        "(F) E + SIZE + BTM + PM":       ["ADJ_RRR_PCT", "ADJ_ACQ_RATE_PCT_W",
                                          "SIZE", "BTM", "PM_OPER_PCT"],
    }
    res = run_cross_sectional_fm(hw, specs)

    rows = []
    for spec, st in res.items():
        if st is None:
            print(f"  {spec}: no valid periods")
            continue
        print(f"\n  {spec}  (T={st['_T']} quarters, avg N={st['_avgN']:.0f})")
        for var in [v for v in st if not v.startswith("_")]:
            if var == "const":
                continue
            s = st[var]
            print(f"    {var:<22} coef={s['coef']:>10.4f}  t={s['t']:>6.2f}  p=[{s['p']:.3f}]")
            rows.append({"Spec": spec, "Variable": var, "coef": s["coef"],
                         "t": s["t"], "p": s["p"], "T": st["_T"], "avgN": st["_avgN"]})
    return pd.DataFrame(rows)


# =============================================================================
# Cell 10: TASK 6 -- Fama-MacBeth adding a revenue-growth control
# =============================================================================
def task6_fm_revgrowth(panel):
    print("\n" + "=" * 80)
    print("TASK 6  Fama-MacBeth (holding-window) adding revenue growth to RRR+size+BTM+PM")
    print("=" * 80)
    hw = build_holding_window_table(panel)

    specs = {
        "(1) Adj RRR only":                  ["ADJ_RRR_PCT"],
        "(2) Adj RRR + SIZE+BTM+PM":         ["ADJ_RRR_PCT", "SIZE", "BTM", "PM_OPER_PCT"],
        "(2+RG) Adj RRR + SIZE+BTM+PM + RG": ["ADJ_RRR_PCT", "SIZE", "BTM",
                                              "PM_OPER_PCT", "REV_GROWTH_PCT"],
    }
    res = run_cross_sectional_fm(hw, specs)

    # collinearity diagnostic: ADJ_RRR vs REV_GROWTH (they share the growth identity)
    dd = hw[["ADJ_RRR_PCT", "REV_GROWTH_PCT"]].dropna()
    corr = dd["ADJ_RRR_PCT"].corr(dd["REV_GROWTH_PCT"])
    print(f"  [diagnostic] corr(ADJ_RRR_PCT, REV_GROWTH_PCT) = {corr:.3f} "
          f"(RRR is a component of the revenue-growth identity; expect collinearity)")

    rows = []
    for spec, st in res.items():
        if st is None:
            print(f"  {spec}: no valid periods")
            continue
        print(f"\n  {spec}  (T={st['_T']} quarters, avg N={st['_avgN']:.0f})")
        for var in [v for v in st if not v.startswith("_")]:
            if var == "const":
                continue
            s = st[var]
            print(f"    {var:<22} coef={s['coef']:>10.4f}  t={s['t']:>6.2f}  p=[{s['p']:.3f}]")
            rows.append({"Spec": spec, "Variable": var, "coef": s["coef"],
                         "t": s["t"], "p": s["p"], "T": st["_T"], "avgN": st["_avgN"]})
    r_focal = res["(2+RG) Adj RRR + SIZE+BTM+PM + RG"]["ADJ_RRR_PCT"]
    print(f"\n  --> Focal: with revenue growth added, ADJ RRR coef={r_focal['coef']:.4f}, "
          f"t={r_focal['t']:.2f}, p=[{r_focal['p']:.3f}]")
    return pd.DataFrame(rows)


# =============================================================================
# Cell 11: main
# =============================================================================
def main():
    global FF_FACTORS_GLOBAL
    print("=" * 80)
    print("  IDENTIFICATION / CONFOUND BATTERY  (corrected timing: k=2m gap, 3m hold)")
    print(f"  Timing source of truth: analysis_v2.build_holding_panel "
          f"(FORM_LAG={av2.FORM_LAG_MONTHS}m, HOLD={av2.HOLD_MONTHS}m)")
    print("=" * 80)

    df_filtered, returns, ff, panel, market_ret = load_base_data()
    FF_FACTORS_GLOBAL = ff

    t1 = task1_qmj_bab(panel, ff, market_ret)
    t2 = task2_rrr_future_beta(returns, df_filtered, ff, market_ret)
    t3 = task3_double_sorts(panel, market_ret, ff)
    t4 = task4_sloan_persistence(df_filtered)
    t5 = task5_ar_winsorized_fm(panel)
    t6 = task6_fm_revgrowth(panel)

    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as xw:
        t1.to_excel(xw, sheet_name="T1_QMJ_BAB", index=False)
        t2.to_excel(xw, sheet_name="T2_RRR_future_beta", index=False)
        t3.to_excel(xw, sheet_name="T3_double_sorts", index=False)
        t4.to_excel(xw, sheet_name="T4_sloan_persistence", index=False)
        t5.to_excel(xw, sheet_name="T5_AR_winsor_FM", index=False)
        t6.to_excel(xw, sheet_name="T6_FM_revgrowth", index=False)
    print("\n" + "=" * 80)
    print(f"  BATTERY COMPLETE. Results written to: {RESULTS_XLSX}")
    print("=" * 80)


if __name__ == "__main__":
    main()
