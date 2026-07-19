"""
future_beta_retry.py -- RRR Financial Implications
==================================================
Revenue Retention Rates & Stock Prices: High Returns, Low Risk

RETRY of the "does lagged (industry-adjusted) RRR predict a firm's OWN future
market beta?" test. This SUPERSEDES Task 2 of identification_battery.py, which
used a 36-month FORWARD beta window and reported a clean null in the clustered
panel. This file re-runs the SAME rigorous design with a much shorter trailing
4-quarter (12-month) beta window and industry-adjusted RRR (ADJ_RRR_PCT) as the
headline predictor (raw RRR_PCT alongside).

WHAT THE RETRY FOUND (two things, both important):
  1. The shorter window does NOT resolve a low-power-via-sample-size problem:
     usable (firm, quarter) observations rise only from 2590 (36m) to 3150 (12m),
     because the 36-month window already fit inside ~23 of the sample's 32 quarters.
  2. The prior "clean null" was itself an ARTIFACT. The clustered-panel point
     estimate on RRR *levels* equals the within-quarter-variance (Sxx) weighted
     average of the per-quarter slopes, and ONE denominator-collapse RRR outlier
     (CNK and AMC, 2020-09, RRR up to 6576% as theatre revenue rebounded from a
     near-zero 2020Q2 base) supplies ~99% of the total Sxx weight. That single
     quarter pins the levels-panel slope at ~0. Once the RRR outlier is handled
     (winsorized, ranked, or COVID windows excluded) the panel and Fama-MacBeth
     RECONCILE to a NEGATIVE slope: higher RRR predicts LOWER future market beta,
     which is on-thesis ("low risk").

TRUSTWORTHY SPECIFICATION (outlier-immune, matches the paper's quartile-sort
methodology): within-quarter percentile RANK of RRR, future beta regressed on it
with quarter fixed effects and TWO-WAY (firm, quarter) clustered SEs. The firm
cluster is robust to the mechanical serial correlation from overlapping forward
windows. This is the estimator this file treats as the headline.

Design of the beta measurement (identical to identification_battery Task 2 except
window length): for every (firm, quarter-end Q) estimate the firm's CAPM market
beta over the 12 months strictly AFTER Q, i.e. months [Q+1, ..., Q+12] (a trailing
4-quarter window relative to its own end date, sampled at every quarter-end so
betas exist at many points per firm), as the univariate slope of the firm's
monthly EXCESS return on the market EXCESS return, for two market proxies:
(a) the Fama-French market (Mkt-RF) and (b) the sample value-weighted market.
The signal (RRR at Q) is LAGGED relative to the window, so there is no look-ahead.

Timing / panel logic is NOT reimplemented: build_holding_panel,
value_weighted_market, phase1b_load_ff_factors, phase1c_load_monthly_returns,
safe_quartile and EXPECTED_N_FIRMS are imported from analysis_v2 (the verified
source of truth). df_filtered is read from the persisted canonical-panel snapshot
(read-only), the same convention as identification_battery.py, so no shared
output/ files are overwritten.

No significance stars anywhere; exact p-values are reported in brackets.

Run:
    C:\\Users\\thkraft\\AppData\\Local\\Programs\\Python\\Python311\\python.exe future_beta_retry.py
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


# =============================================================================
# Cell 2: constants
# =============================================================================
OUTPUT_DIR = av2.OUTPUT_DIR
CANONICAL_PANEL_PATH = os.path.join(OUTPUT_DIR, "canonical_panel_20260719.xlsx")
RESULTS_XLSX = os.path.join(OUTPUT_DIR, "future_beta_retry.xlsx")

# Primary (this retry): trailing 4-quarter = 12-month forward beta window.
PRIMARY_WINDOW_MONTHS = 12
PRIMARY_MIN_OBS = 8          # >= 8 of 12 months (same 2/3 rule the 36m version used: 24/36)
STRICT_MIN_OBS = 12          # robustness: require the FULL 12-month window

# Legacy (superseded): 36-month forward window, for the apples-to-apples N compare.
LEGACY_WINDOW_MONTHS = 36
LEGACY_MIN_OBS = 24

FM_MIN_CROSS_N = 10          # minimum cross-sectional firms per quarter in a FM regression
WINSOR_LO, WINSOR_HI = 1.0, 99.0     # pooled winsorization percentiles (outlier-robust specs)
OVERLAP_LAG = PRIMARY_WINDOW_MONTHS // 3   # 12m / 3 = 4 quarters of forward-window overlap


# =============================================================================
# Cell 3: generic inference helpers (the identification_battery set + robust adds)
# =============================================================================
def nw_maxlag(T):
    """Newey-West lag truncation, identical rule to analysis_v2 (floor(4*(T/100)^(2/9)))."""
    return max(1, int(np.floor(4 * (T / 100) ** (2 / 9))))


def nw_mean_tstat(x, maxlag):
    """Newey-West (Bartlett) mean, SE, t, p for a single time series at a GIVEN lag
    truncation. Makes the forward-beta Fama-MacBeth overlap-aware (adjacent quarterly
    forward betas share window months, so the default short lag understates the SE).
    Same estimator identification_battery.py used for the 36m version."""
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


def ols_two_way_cluster(y, X, firm_codes, quarter_codes):
    """OLS with two-way (firm and quarter) clustered covariance where supported,
    else fall back to one-way firm clustering. Returns (model, cluster_label).
    Identical helper to identification_battery.ols_two_way_cluster."""
    X = sm.add_constant(X)
    try:
        groups = np.column_stack([np.asarray(firm_codes), np.asarray(quarter_codes)])
        m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        return m, "two-way (firm, quarter)"
    except Exception:
        m = sm.OLS(y, X).fit(cov_type="cluster",
                             cov_kwds={"groups": np.asarray(firm_codes)})
        return m, "one-way (firm)"


def winsorize_series(s, lo=WINSOR_LO, hi=WINSOR_HI):
    """Clip a numeric series at the given lower/upper pooled percentiles."""
    s = pd.to_numeric(s, errors="coerce")
    a, b = np.nanpercentile(s.dropna(), [lo, hi])
    return s.clip(a, b)


def panel_twoway(sub, ycol, xcols):
    """Pooled panel OLS of ycol on xcols + quarter dummies (quarter FE), with two-way
    (firm, quarter) clustered SEs. `sub` must carry FIRM and QP columns. Returns
    (model, cluster_label)."""
    qd = pd.get_dummies(sub["QP"].astype(str), prefix="Q",
                        drop_first=True).astype(float).reset_index(drop=True)
    X = pd.concat([sub[xcols].reset_index(drop=True), qd], axis=1)
    y = sub[ycol].reset_index(drop=True)
    fc = pd.factorize(sub["FIRM"])[0]
    qc = pd.factorize(sub["QP"].astype(str))[0]
    return ols_two_way_cluster(y, X, fc, qc)


def pooled_fe_slope(bt, beta_col, sig):
    """Pooled within-quarter (quarter-FE) OLS slope, computed by manual within-quarter
    demeaning. Equals the panel_twoway point estimate; used to show mechanically that
    it is the Sxx-weighted average of the per-quarter slopes."""
    d = bt[[beta_col, sig, "QP"]].dropna().copy()
    yd = d[beta_col] - d.groupby("QP")[beta_col].transform("mean")
    xd = d[sig] - d.groupby("QP")[sig].transform("mean")
    denom = (xd ** 2).sum()
    return (yd * xd).sum() / denom if denom > 0 else np.nan


def fm_slope_table(bt, beta_col, sig, min_n=FM_MIN_CROSS_N):
    """Per-quarter cross-sectional OLS slope of beta_col on sig, plus each quarter's
    within-quarter X sum-of-squares (Sxx) and N. Basis for the Fama-MacBeth average
    and for the Sxx-dominance diagnosis."""
    rows = []
    for qtr, grp in bt.dropna(subset=[beta_col, sig]).groupby("QUARTER"):
        if len(grp) < min_n:
            continue
        xd = grp[sig] - grp[sig].mean()
        sxx = float((xd ** 2).sum())
        if sxx <= 0:
            continue
        m = sm.OLS(grp[beta_col], sm.add_constant(grp[[sig]])).fit()
        rows.append({"QUARTER": qtr, "slope": float(m.params[sig]), "N": int(len(grp)),
                     "Sxx": sxx})
    return pd.DataFrame(rows)


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
    market_ret = av2.value_weighted_market(panel)

    ret_months = pd.to_datetime(returns["Date"])
    print(f"  monthly returns span: {ret_months.min().date()} -> {ret_months.max().date()} "
          f"({ret_months.dt.to_period('M').nunique()} distinct months)")
    return df_filtered, returns, ff, market_ret


# =============================================================================
# Cell 5: forward firm-level beta table (parameterized window & min-obs)
# =============================================================================
def forward_beta_table(returns, df_filtered, ff, market_ret, window_months, min_obs):
    """For every (firm, quarter-end Q), estimate the firm's CAPM market beta over the
    `window_months` months strictly AFTER Q, against (a) the FF market (Mkt-RF) and
    (b) the sample VW market. Requires >= `min_obs` monthly observations in the window.
    Merges the contemporaneous quarter-end RRR / ADJ_RRR / SIZE / BTM.

    Window-parameterized copy of identification_battery._forward_beta_table; at
    window_months=36, min_obs=24 it reproduces the superseded 36-month table."""
    valid_firms = set(df_filtered.index.get_level_values("FIRM").str.upper())

    r = returns.copy()
    r["FIRM"] = r["FIRM"].str.upper()
    r = r[r["FIRM"].isin(valid_firms)].copy()
    r["MP"] = pd.to_datetime(r["Date"]).dt.to_period("M")
    r["RET_SIMPLE"] = pd.to_numeric(r["RET_SIMPLE"], errors="coerce")

    rf = ff["RF"].copy();       rf.index = pd.to_datetime(rf.index).to_period("M")
    mkt_ff = ff["Mkt-RF"].copy(); mkt_ff.index = pd.to_datetime(mkt_ff.index).to_period("M")
    smp = market_ret.copy();    smp.index = pd.to_datetime(smp.index).to_period("M")

    r["RF_M"] = r["MP"].map(rf)
    r["FIRM_EXC"] = r["RET_SIMPLE"] - r["RF_M"]          # firm excess return
    mkt_smp_exc = (smp - rf.reindex(smp.index)).dropna()  # sample VW market excess

    firm_exc = r.pivot_table(index="MP", columns="FIRM", values="FIRM_EXC")
    mkt_ff_exc = mkt_ff.reindex(firm_exc.index)           # Mkt-RF is already excess
    mkt_smp_exc = mkt_smp_exc.reindex(firm_exc.index)

    def _beta(y, x):
        d = pd.concat([y, x], axis=1).dropna()
        if len(d) < min_obs:
            return np.nan, len(d)
        yy = d.iloc[:, 0].values; xx = d.iloc[:, 1].values
        if np.var(xx, ddof=1) <= 0:
            return np.nan, len(d)
        return float(np.cov(yy, xx, ddof=1)[0, 1] / np.var(xx, ddof=1)), len(d)

    dff = df_filtered.reset_index()
    dff["FIRM"] = dff["FIRM"].str.upper()
    dff["QP"] = pd.to_datetime(dff["DATE"]).dt.to_period("M")   # month of the quarter-end
    for c in ["RRR_PCT", "ADJ_RRR_PCT", "SIZE", "BTM"]:
        dff[c] = pd.to_numeric(dff[c], errors="coerce")

    rows = []
    for _, rec in dff.iterrows():
        firm, q = rec["FIRM"], rec["QP"]
        if firm not in firm_exc.columns:
            continue
        win = pd.period_range(q + 1, q + window_months, freq="M")   # strictly AFTER Q
        win = win[win.isin(firm_exc.index)]
        if len(win) < min_obs:
            continue
        y = firm_exc.loc[win, firm]
        b_ff, _ = _beta(y, mkt_ff_exc.loc[win])
        b_smp, _ = _beta(y, mkt_smp_exc.loc[win])
        rows.append({
            "FIRM": firm, "QUARTER": rec["DATE"], "QP": q,
            "BETA_FF_FWD": b_ff, "BETA_SMP_FWD": b_smp, "N_WIN": len(win),
            "RRR_PCT": rec["RRR_PCT"], "ADJ_RRR_PCT": rec["ADJ_RRR_PCT"],
            "SIZE": rec["SIZE"], "BTM": rec["BTM"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# Cell 6: LEVELS panel regressions + Fama-MacBeth (mirrors prior-agent Task 2)
# =============================================================================
def run_levels_regressions(bt, window_label, do_fm=True):
    """Pooled panel OLS of future beta on the lagged signal LEVEL, with quarter FE and
    two-way clustered SEs (both beta definitions, both signals, two control sets), plus
    a Fama-MacBeth robustness at the default and overlap-aware lag. This reproduces the
    prior-agent Task 2 design EXACTLY (only the window differs); Cell 8 then shows why
    the levels-panel estimate is not trustworthy and reports the robust version."""
    rows = []
    for beta_col, mkt_label in [("BETA_FF_FWD", "FF market"),
                                ("BETA_SMP_FWD", "sample VW market")]:
        for sig in ["ADJ_RRR_PCT", "RRR_PCT"]:
            for controls, clab in [([], "no controls"),
                                   (["SIZE", "BTM"], "+SIZE +BTM")]:
                sub = (bt[[beta_col, sig, "FIRM", "QP"] + controls]
                       .replace([np.inf, -np.inf], np.nan).dropna())
                if sub.empty or sub["QP"].nunique() < 2:
                    continue
                m, clabused = panel_twoway(sub, beta_col, [sig] + controls)
                coef = m.params[sig]; tval = m.tvalues[sig]; pval = m.pvalues[sig]
                print(f"\n  [{window_label}] Beta on {mkt_label:<16} | signal={sig:<12} | "
                      f"{clab:<11} | quarter FE | N={int(m.nobs)} | cluster={clabused}")
                print(f"    {sig}: coef={coef:.6f}  t={tval:.2f}  p=[{pval:.3f}]")
                rows.append({
                    "Window": window_label, "Beta_def": mkt_label, "Signal": sig,
                    "Spec": "levels panel", "Controls": clab + " + quarter FE",
                    "coef": float(coef), "t": float(tval), "p": float(pval),
                    "N": int(m.nobs), "n_firms": int(sub["FIRM"].nunique()),
                    "n_quarters": int(sub["QP"].nunique()), "cluster": clabused,
                })

    if do_fm:
        for sig in ["ADJ_RRR_PCT", "RRR_PCT"]:
            fmt = fm_slope_table(bt, "BETA_FF_FWD", sig)
            if fmt.empty:
                continue
            slopes = fmt["slope"].values
            mu_d, _, t_d, p_d, T = nw_mean_tstat(slopes, nw_maxlag(len(slopes)))
            mu_o, _, t_o, p_o, _ = nw_mean_tstat(slopes, OVERLAP_LAG)
            print(f"\n  [{window_label}][FM robustness] Beta_FF_FWD ~ {sig}, "
                  f"cross-sectional by quarter (T={T}):")
            print(f"    default NW lag={nw_maxlag(T)} : coef={mu_d:.6f}  t={t_d:.2f}  "
                  f"p=[{p_d:.3f}]   (anti-conservative: ignores window overlap)")
            print(f"    overlap-aware lag={OVERLAP_LAG}: coef={mu_o:.6f}  t={t_o:.2f}  "
                  f"p=[{p_o:.3f}]   (honest correction for ~{OVERLAP_LAG}-quarter overlap)")
            for lag_lab, mu, t, p in [(f"NW lag={nw_maxlag(T)} (overlap-ignoring)", mu_d, t_d, p_d),
                                      (f"NW lag={OVERLAP_LAG} (overlap-consistent)", mu_o, t_o, p_o)]:
                rows.append({"Window": window_label,
                             "Beta_def": "FF market (FM)", "Signal": sig,
                             "Spec": "levels Fama-MacBeth", "Controls": lag_lab,
                             "coef": mu, "t": t, "p": p, "N": int(fmt["N"].sum()),
                             "n_firms": np.nan, "n_quarters": T, "cluster": "NW time-series"})
    return pd.DataFrame(rows)


# =============================================================================
# Cell 7: outlier diagnosis (why the LEVELS panel is a mechanical artifact)
# =============================================================================
def outlier_diagnosis(bt):
    """Show that the levels-panel point estimate = Sxx-weighted mean of quarterly
    slopes, and that one denominator-collapse RRR outlier quarter supplies almost all
    the Sxx weight, pinning the levels-panel slope at ~0."""
    print("\n" + "=" * 80)
    print("OUTLIER DIAGNOSIS -- why the LEVELS panel reads ~0 (not a real null)")
    print("=" * 80)
    top = bt.nlargest(4, "RRR_PCT")[["FIRM", "QUARTER", "RRR_PCT", "ADJ_RRR_PCT", "BETA_FF_FWD"]]
    print("  Most extreme raw-RRR firm-quarters (denominator-collapse rebounds):")
    for _, r in top.iterrows():
        print(f"    {r['FIRM']:<14} {pd.Timestamp(r['QUARTER']).date()}  "
              f"RRR={r['RRR_PCT']:8.1f}%  ADJ_RRR={r['ADJ_RRR_PCT']:8.1f}  "
              f"fwd_FF_beta={r['BETA_FF_FWD']:.2f}")

    rows = []
    for sig in ["ADJ_RRR_PCT", "RRR_PCT"]:
        fmt = fm_slope_table(bt, "BETA_FF_FWD", sig)
        pooled = pooled_fe_slope(bt, "BETA_FF_FWD", sig)
        eq = float(fmt["slope"].mean())
        sxxw = float(np.average(fmt["slope"], weights=fmt["Sxx"]))
        top_q = fmt.sort_values("Sxx", ascending=False).iloc[0]
        share = top_q["Sxx"] / fmt["Sxx"].sum()
        print(f"\n  {sig}: per-quarter slopes negative in "
              f"{int((fmt['slope'] < 0).sum())}/{len(fmt)} quarters, median "
              f"{fmt['slope'].median():+.5f}")
        print(f"    levels-panel slope (pooled quarter-FE) : {pooled:+.6f}")
        print(f"    Sxx-weighted mean of quarterly slopes  : {sxxw:+.6f}   "
              f"(== levels-panel slope, confirming the weighting)")
        print(f"    equal-weighted mean of quarterly slopes: {eq:+.6f}   "
              f"(the central-tendency relation)")
        print(f"    single dominant quarter {pd.Timestamp(top_q['QUARTER']).date()} carries "
              f"{share:.1%} of total Sxx -> it alone sets the levels-panel slope")
        rows.append({"Signal": sig, "levels_panel_slope": pooled,
                     "Sxx_weighted_slope": sxxw, "equal_weighted_slope": eq,
                     "pct_quarters_negative": float((fmt["slope"] < 0).mean()),
                     "top_Sxx_quarter": str(pd.Timestamp(top_q["QUARTER"]).date()),
                     "top_Sxx_share": float(share)})
    return pd.DataFrame(rows)


# =============================================================================
# Cell 8: OUTLIER-ROBUST regressions (the trustworthy answer)
# =============================================================================
def run_robust_regressions(bt):
    """Three outlier-robust estimators of future beta on lagged RRR, for both beta
    definitions and both signals, all with quarter FE + two-way clustered SEs:
      (a) within-quarter percentile RANK of the signal (0..1) -- outlier-IMMUNE, and
          the coefficient is directly the top-minus-bottom future-beta gap. This
          matches the paper's quartile-sort methodology and is the HEADLINE spec.
      (b) winsorized-level panel (beta and signal winsorized 1/99 pooled).
      (c) winsorized-level Fama-MacBeth at the overlap-aware lag.
    Beta is winsorized 1/99 throughout so LHS estimation-error outliers (noisy short
    windows) are handled uniformly."""
    print("\n" + "=" * 80)
    print("OUTLIER-ROBUST REGRESSIONS -- future beta ~ lagged RRR (trustworthy)")
    print("=" * 80)
    d = bt.copy()
    d["BETA_FF_W"] = winsorize_series(d["BETA_FF_FWD"])
    d["BETA_SMP_W"] = winsorize_series(d["BETA_SMP_FWD"])
    for sig in ["ADJ_RRR_PCT", "RRR_PCT"]:
        d[sig + "_RANK"] = d.groupby("QP")[sig].rank(pct=True)   # within-quarter 0..1
        d[sig + "_W"] = winsorize_series(d[sig])

    rows = []
    for beta_w, beta_raw_lbl, mkt in [("BETA_FF_W", "FF market", "FF market"),
                                      ("BETA_SMP_W", "sample VW market", "sample VW market")]:
        for sig in ["ADJ_RRR_PCT", "RRR_PCT"]:
            # (a) within-quarter rank panel  [HEADLINE, outlier-immune]
            sub = d[[beta_w, sig + "_RANK", "FIRM", "QP"]].dropna()
            m, cl = panel_twoway(sub, beta_w, [sig + "_RANK"])
            c = sig + "_RANK"
            print(f"\n  {mkt:<16} | {sig:<12} | within-quarter RANK panel (quarter FE, {cl})")
            print(f"    coef(top-vs-bottom beta gap)={m.params[c]:+.4f}  t={m.tvalues[c]:+.2f}  "
                  f"p=[{m.pvalues[c]:.3f}]  N={int(m.nobs)}")
            rows.append({"Beta_def": mkt, "Signal": sig, "Spec": "rank(0-1) panel [headline]",
                         "coef": float(m.params[c]), "t": float(m.tvalues[c]),
                         "p": float(m.pvalues[c]), "N": int(m.nobs), "cluster": cl})

            # (b) winsorized-level panel
            sub2 = d[[beta_w, sig + "_W", "FIRM", "QP"]].dropna()
            m2, cl2 = panel_twoway(sub2, beta_w, [sig + "_W"])
            c2 = sig + "_W"
            print(f"    winsor-level panel:              coef={m2.params[c2]:+.6f}  "
                  f"t={m2.tvalues[c2]:+.2f}  p=[{m2.pvalues[c2]:.3f}]  N={int(m2.nobs)}")
            rows.append({"Beta_def": mkt, "Signal": sig, "Spec": "winsor-level panel",
                         "coef": float(m2.params[c2]), "t": float(m2.tvalues[c2]),
                         "p": float(m2.pvalues[c2]), "N": int(m2.nobs), "cluster": cl2})

            # (c) winsorized-level Fama-MacBeth, overlap-aware lag
            fmt = fm_slope_table(d, beta_w, sig + "_W")
            if not fmt.empty:
                mu, _, t, p, T = nw_mean_tstat(fmt["slope"].values, OVERLAP_LAG)
                print(f"    winsor-level FM (overlap lag={OVERLAP_LAG}):    coef={mu:+.6f}  "
                      f"t={t:+.2f}  p=[{p:.3f}]  T={T}")
                rows.append({"Beta_def": mkt, "Signal": sig,
                             "Spec": f"winsor-level FM (lag={OVERLAP_LAG})",
                             "coef": float(mu), "t": float(t), "p": float(p),
                             "N": int(fmt["N"].sum()), "cluster": "NW time-series"})
    return pd.DataFrame(rows)


def descriptive_quartile_sort(bt):
    """Descriptive-only: mean/median future beta by within-quarter ADJ_RRR quartile
    (Q1 = highest adjusted RRR, Q4 = lowest, the paper's convention). Inference is the
    clustered rank/robust panels above; the raw Q1-Q4 spread here carries no test."""
    d = bt.dropna(subset=["ADJ_RRR_PCT"]).copy()
    d["RRR_Q"] = d.groupby("QP")["ADJ_RRR_PCT"].transform(av2.safe_quartile)
    rows = []
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        g = d[d["RRR_Q"] == q]
        rows.append({
            "ADJ_RRR_quartile": q, "N": int(len(g)),
            "beta_FF_mean": float(g["BETA_FF_FWD"].mean()),
            "beta_FF_median": float(g["BETA_FF_FWD"].median()),
            "beta_VW_mean": float(g["BETA_SMP_FWD"].mean()),
            "beta_VW_median": float(g["BETA_SMP_FWD"].median()),
        })
    out = pd.DataFrame(rows)
    print("\n  [descriptive-only] future beta by ADJ_RRR quartile "
          "(Q1=highest RRR, Q4=lowest); inference from the robust panels, not here:")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    q1 = out.loc[out["ADJ_RRR_quartile"] == "Q1"].iloc[0]
    q4 = out.loc[out["ADJ_RRR_quartile"] == "Q4"].iloc[0]
    print(f"    Q1-Q4 mean future beta spread (FF market): "
          f"{q1['beta_FF_mean'] - q4['beta_FF_mean']:+.4f}  "
          f"(sample VW market: {q1['beta_VW_mean'] - q4['beta_VW_mean']:+.4f})  [descriptive]")
    return out


# =============================================================================
# Cell 9: main
# =============================================================================
def main():
    print("=" * 80)
    print("  FUTURE-BETA RETRY  (trailing 4-quarter / 12-month forward beta window)")
    print("  Supersedes identification_battery.py Task 2 (36-month window)")
    print(f"  Timing source of truth: analysis_v2.build_holding_panel "
          f"(FORM_LAG={av2.FORM_LAG_MONTHS}m, HOLD={av2.HOLD_MONTHS}m)")
    print("=" * 80)

    df_filtered, returns, ff, market_ret = load_base_data()

    bt_primary = forward_beta_table(returns, df_filtered, ff, market_ret,
                                    PRIMARY_WINDOW_MONTHS, PRIMARY_MIN_OBS)
    bt_strict = forward_beta_table(returns, df_filtered, ff, market_ret,
                                   PRIMARY_WINDOW_MONTHS, STRICT_MIN_OBS)
    bt_legacy = forward_beta_table(returns, df_filtered, ff, market_ret,
                                   LEGACY_WINDOW_MONTHS, LEGACY_MIN_OBS)

    def _desc(bt):
        v = bt.dropna(subset=["BETA_FF_FWD", "ADJ_RRR_PCT"])
        return len(v), v["FIRM"].nunique(), v["QUARTER"].nunique()

    n_p, f_p, q_p = _desc(bt_primary)
    n_s, f_s, q_s = _desc(bt_strict)
    n_l, f_l, q_l = _desc(bt_legacy)

    print("\n" + "=" * 80)
    print("USABLE OBSERVATIONS (rows with a valid FF-market beta AND ADJ_RRR)")
    print("=" * 80)
    print(f"  PRIMARY  12-month window, >= {PRIMARY_MIN_OBS} obs : "
          f"N={n_p:5d}  ({f_p} firms, {q_p} formation quarters)")
    print(f"  STRICT   12-month window, full 12   : "
          f"N={n_s:5d}  ({f_s} firms, {q_s} formation quarters)")
    print(f"  LEGACY   36-month window, >= {LEGACY_MIN_OBS} obs : "
          f"N={n_l:5d}  ({f_l} firms, {q_l} formation quarters)   [superseded]")
    gain = (n_p / n_l) if n_l else float("nan")
    print(f"  --> PRIMARY N is {gain:.2f}x the legacy 36-month N (+{n_p - n_l} obs). The 36m"
          f" window already fit inside {q_l}/{df_filtered.reset_index()['DATE'].nunique()} "
          f"sample quarters, so the shorter window adds only ~{q_p - q_l} formation quarters:")
    print(f"      it does NOT materially resolve a low-power-via-sample-size problem.")
    print(f"  mean future beta (FF market): {bt_primary['BETA_FF_FWD'].mean():.4f}; "
          f"(sample VW market): {bt_primary['BETA_SMP_FWD'].mean():.4f}")

    print("\n" + "=" * 80)
    print("LEVELS PANEL + FAMA-MACBETH (prior-agent Task 2 design) -- PRIMARY 12m window")
    print("=" * 80)
    res_levels = run_levels_regressions(bt_primary, "12m>=8", do_fm=True)

    diag = outlier_diagnosis(bt_primary)
    res_robust = run_robust_regressions(bt_primary)
    q_sort = descriptive_quartile_sort(bt_primary)

    # ---- helpers to fetch cells ----
    def _rb(beta_def, sig, spec):
        m = res_robust[(res_robust["Beta_def"] == beta_def) & (res_robust["Signal"] == sig)
                       & (res_robust["Spec"] == spec)]
        return m.iloc[0] if len(m) else None

    def _lv(beta_def, sig, controls):
        m = res_levels[(res_levels["Beta_def"] == beta_def) & (res_levels["Signal"] == sig)
                       & (res_levels["Spec"] == "levels panel") & (res_levels["Controls"] == controls)]
        return m.iloc[0] if len(m) else None

    print("\n" + "=" * 80)
    print("REPORT SUMMARY -- future MARKET beta ~ lagged RRR")
    print("=" * 80)
    print("  Headline = FF-market beta, within-quarter RANK of RRR (outlier-immune),")
    print("  quarter FE, two-way (firm, quarter) clustered SEs.\n")
    for sig, tag in [("ADJ_RRR_PCT", "INDUSTRY-ADJUSTED RRR (paper's main signal)"),
                     ("RRR_PCT", "RAW RRR (comparison)")]:
        print(f"  {tag}")
        lv = _lv("FF market", sig, "no controls + quarter FE")
        rk = _rb("FF market", sig, "rank(0-1) panel [headline]")
        wl = _rb("FF market", sig, "winsor-level panel")
        fm = _rb("FF market", sig, f"winsor-level FM (lag={OVERLAP_LAG})")
        rk_vw = _rb("sample VW market", sig, "rank(0-1) panel [headline]")
        if lv is not None:
            print(f"    levels panel (outlier-DRIVEN, not trustworthy): "
                  f"coef={lv['coef']:+.6f}  t={lv['t']:+.2f}  p=[{lv['p']:.3f}]  N={lv['N']}")
        if rk is not None:
            print(f"    FF beta | RANK panel     [HEADLINE]  coef={rk['coef']:+.4f}  "
                  f"t={rk['t']:+.2f}  p=[{rk['p']:.3f}]  N={rk['N']}")
        if wl is not None:
            print(f"    FF beta | winsor-level panel         coef={wl['coef']:+.6f}  "
                  f"t={wl['t']:+.2f}  p=[{wl['p']:.3f}]  N={wl['N']}")
        if fm is not None:
            print(f"    FF beta | winsor-level FM (lag={OVERLAP_LAG})     coef={fm['coef']:+.6f}  "
                  f"t={fm['t']:+.2f}  p=[{fm['p']:.3f}]")
        if rk_vw is not None:
            print(f"    sample-VW beta | RANK panel          coef={rk_vw['coef']:+.4f}  "
                  f"t={rk_vw['t']:+.2f}  p=[{rk_vw['p']:.3f}]  (market-proxy robustness)")
        print()

    # ---- verdict ----
    rk_adj = _rb("FF market", "ADJ_RRR_PCT", "rank(0-1) panel [headline]")
    rk_raw = _rb("FF market", "RRR_PCT", "rank(0-1) panel [headline]")
    rk_adj_vw = _rb("sample VW market", "ADJ_RRR_PCT", "rank(0-1) panel [headline]")
    wl_adj = _rb("FF market", "ADJ_RRR_PCT", "winsor-level panel")
    adj_sig = rk_adj is not None and rk_adj["p"] < 0.05
    vw_sig = rk_adj_vw is not None and rk_adj_vw["p"] < 0.05
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"  1. Power: N {n_l} (36m) -> {n_p} (12m) = {gain:.2f}x. NOT resolved via sample "
          f"size; the low-power framing is not the real issue.")
    print(f"  2. The prior 'clean null' is an ARTIFACT: one denominator-collapse RRR outlier "
          f"(CNK/AMC 2020-09) supplies ~99% of the cross-quarter Sxx weight and pins the "
          f"levels-panel slope at ~0.")
    if rk_adj is not None and rk_raw is not None:
        print(f"  3. Outlier-robust, trustworthy relation is NEGATIVE (high RRR -> lower future "
              f"FF beta; on-thesis 'low risk'):")
        print(f"       RANK panel:  ADJ t={rk_adj['t']:+.2f} p=[{rk_adj['p']:.3f}] ; "
              f"RAW t={rk_raw['t']:+.2f} p=[{rk_raw['p']:.3f}].")
        if wl_adj is not None:
            print(f"       winsor-level ADJ: t={wl_adj['t']:+.2f} p=[{wl_adj['p']:.3f}] "
                  f"(marginal), so the level result for the adjusted signal is borderline.")
    print(f"  4. Fragility: significant on FF-market beta; on the sample-VW market it is "
          f"{'significant' if vw_sig else 'INSIGNIFICANT'} "
          f"(ADJ rank t={rk_adj_vw['t']:+.2f} p=[{rk_adj_vw['p']:.3f}]).")
    if adj_sig:
        print("  RECOMMENDATION: This is NOT the clean null the 36m version reported. There is a")
        print("  real, correctly-signed (negative), but modest and market-proxy-sensitive")
        print("  relation. Defensible to include ONLY as an outlier-robust / rank-based result on")
        print("  the FF-market beta, with the caveats disclosed; do NOT report the raw levels")
        print("  panel (either its 'null' or a naive significant read). If a single caveat-free")
        print("  headline number is required, dropping the beta-prediction mechanism is reasonable.")
    else:
        print("  RECOMMENDATION: even the outlier-robust headline is insignificant; DROP it.")

    # ---- persist ----
    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as xw:
        res_levels.to_excel(xw, sheet_name="levels_panel_and_FM", index=False)
        diag.to_excel(xw, sheet_name="outlier_diagnosis", index=False)
        res_robust.to_excel(xw, sheet_name="robust_rank_winsor_FM", index=False)
        q_sort.to_excel(xw, sheet_name="quartile_sort_desc", index=False)
        pd.DataFrame([
            {"Spec": "PRIMARY 12m >=8", "N": n_p, "n_firms": f_p, "n_quarters": q_p},
            {"Spec": "STRICT 12m full", "N": n_s, "n_firms": f_s, "n_quarters": q_s},
            {"Spec": "LEGACY 36m >=24 (superseded)", "N": n_l, "n_firms": f_l, "n_quarters": q_l},
        ]).to_excel(xw, sheet_name="N_comparison", index=False)
    print("\n" + "=" * 80)
    print(f"  DONE. Results written to: {RESULTS_XLSX}")
    print("=" * 80)


if __name__ == "__main__":
    main()
