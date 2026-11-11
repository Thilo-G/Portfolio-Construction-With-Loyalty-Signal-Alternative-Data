"""rrr_regressions.py -- RRR Financial Implications
=================================================

Estimating risk-adjusted performance:

  newey_west_ols          OLS with Newey-West standard errors (Bartlett kernel,
                          floor(4 (T/100)^(2/9)) lags)
  run_factor_regressions  FF3, Carhart and FF5 regressions of each portfolio
  run_portfolio_analysis  the headline quartile sorts (raw and industry-adjusted)
  run_capm_vs_sp500       the benchmark portfolio regressed on the S&P 500
  run_fama_macbeth        monthly Fama-MacBeth regressions, equal- and value-weighted
  run_risk_analysis       volatility, Sharpe, drawdown and related risk statistics

Moved verbatim from analysis_v2.py on 2026-10-20; other scripts still reach
these functions through analysis_v2 (av2.run_factor_regressions, ...).
"""

# =============================================================================
# Imports
# =============================================================================
import os

import numpy as np
import pandas as pd
import scipy.stats as sps
import statsmodels.api as sm
from scipy.stats import kurtosis, skew

from config import EXPECTED_N_FIRMS, OUTPUT_DIR
from rrr_portfolios import (build_portfolio_returns, echo_formation_examples, safe_quartile,
                            safe_tercile, value_weighted_market)


# =============================================================================
# Functions
# =============================================================================
def newey_west_ols(y, X, max_lags=None):
    """Run OLS with Newey-West standard errors."""
    X = sm.add_constant(X)
    if max_lags is None:
        max_lags = max(1, int(np.floor(4 * (len(y)/100)**(2/9))))
    model = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
    return model


def run_factor_regressions(port_rets, ff_factors, label):
    """Run FF3, FF3+Mom, FF5 regressions on portfolio returns."""

    print(f"\n    Factor regressions: {label}")

    # Align dates
    port = port_rets.copy()
    port.index = pd.to_datetime(port.index).to_period('M').to_timestamp('M')
    combined = port.join(ff_factors, how='inner')

    if combined.empty or len(combined) < 12:
        print("    Insufficient overlapping data for factor regressions")
        return {}

    results = {}

    # Columns to regress (portfolios, not factors)
    factor_cols_all = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom', 'RF']
    port_cols = [c for c in port.columns if c not in factor_cols_all and c != 'MKT']

    for p in port_cols:
        if p not in combined.columns:
            continue

        # Excess returns (long-short portfolios are already zero-cost; no RF subtraction)
        if 'RF' in combined.columns:
            if 'T1-T3' in p or 'Q1-Q4' in p or 'High-Low' in p:
                y = combined[p]  # Long-short is already zero-cost
            else:
                y = combined[p] - combined['RF']
        else:
            y = combined[p]

        y = y.dropna()
        if len(y) < 12:
            continue

        # Model specs
        specs = {}
        if all(c in combined.columns for c in ['Mkt-RF', 'SMB', 'HML']):
            specs['FF3'] = ['Mkt-RF', 'SMB', 'HML']
        if all(c in combined.columns for c in ['Mkt-RF', 'SMB', 'HML', 'Mom']):
            specs['FF3+Mom'] = ['Mkt-RF', 'SMB', 'HML', 'Mom']
        if all(c in combined.columns for c in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']):
            specs['FF5'] = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        for spec_name, factors in specs.items():
            X = combined[factors].reindex(y.index).dropna()
            y_aligned = y.reindex(X.index).dropna()
            X = X.reindex(y_aligned.index)

            if len(y_aligned) < 12:
                continue

            model = newey_west_ols(y_aligned, X)

            alpha = model.params['const']
            alpha_t = model.tvalues['const']
            alpha_p = model.pvalues['const']
            alpha_se = model.bse['const']

            key = f"{p}_{spec_name}"
            results[key] = {
                'alpha': alpha,
                'alpha_t': alpha_t,
                'alpha_p': alpha_p,
                'se_alpha': alpha_se,
                'r2': model.rsquared,
                'n_obs': model.nobs,
            }
            # Store beta, SE, t-stat, and p-value for each factor
            for f in factors:
                results[key][f'beta_{f}'] = model.params[f]
                results[key][f'se_{f}'] = model.bse[f]
                results[key][f'tstat_{f}'] = model.tvalues[f]
                results[key][f'pval_{f}'] = model.pvalues[f]

    # Print summary for long-short or all portfolios
    print(f"    {'Portfolio':<15} {'Model':<10} {'Alpha':>8} {'t-stat':>8} {'R2':>6} {'N':>5}")
    for key, vals in sorted(results.items()):
        parts = key.rsplit('_', 1)
        if len(parts) == 2:
            port_name, model_name = parts
        else:
            port_name, model_name = key, ''
        print(f"    {port_name:<15} {model_name:<10} {vals['alpha']*100:>8.3f}% {vals['alpha_t']:>8.2f} "
              f"{vals['r2']:>6.3f} {vals['n_obs']:>5.0f}")

    return results


def run_portfolio_analysis(panel, ff_factors):
    """Univariate RRR quartile sorts (Q1 = highest RRR) with factor regressions.

    Consumes the holding panel from build_holding_panel(), so the signal->return
    timing (form FORM_LAG_MONTHS months after quarter-end, hold HOLD_MONTHS) is
    already correct. Sorts use the CONTEMPORANEOUS quarter-end signal (RRR_PCT /
    ADJ_RRR_PCT), never a re-lagged column, so no second lag is stacked.
    """

    results = {}
    ret = panel.copy()
    for col in ['RRR_PCT', 'ADJ_RRR_PCT', 'MCAP_FORM']:
        ret[col] = pd.to_numeric(ret[col], errors='coerce')

    # ---- Pipeline invariants, asserted at the point portfolios are formed ----
    n_firms = ret['FIRM'].nunique()
    print(f"\n  [INVARIANT] firms entering portfolio formation: {n_firms} "
          f"(expected {EXPECTED_N_FIRMS})")

    # A firm with no formation market cap gets zero weight in a value-weighted
    # portfolio, so it is in the panel but not in the book. Report both numbers:
    # the invariant above guards the pipeline, this one describes the sample.
    _weightable = ret.loc[ret['MCAP_FORM'].notna(), 'FIRM'].nunique()
    print(f"  [SAMPLE]    firms with a formation market cap (value-weightable): "
          f"{_weightable} of {n_firms}")
    if _weightable < n_firms:
        _no_mcap = sorted(set(ret['FIRM']) - set(ret.loc[ret['MCAP_FORM'].notna(), 'FIRM']))
        print(f"              no market cap ({len(_no_mcap)}): {_no_mcap}")
    assert n_firms == EXPECTED_N_FIRMS, (
        f"Sample size changed: {n_firms} firms at formation, expected {EXPECTED_N_FIRMS}. "
        f"Investigate before trusting portfolio results.")
    echo_formation_examples(ret)

    # Market portfolio: VW (formation market cap) SIMPLE return of all sample firms.
    market_ret = value_weighted_market(ret)

    # =========================================================================
    # SECTION 1: UNIVARIATE QUARTILE SORTS (Q1 = highest value)
    # =========================================================================
    print("\n  *** SECTION 1: UNIVARIATE QUARTILE SORTS (Q1=best) ***")

    for signal_label, rrr_col in [
        ('RAW', 'RRR_PCT'),
        ('ADJ', 'ADJ_RRR_PCT'),
    ]:
        print(f"\n  === {signal_label} signal: quartile sort ===")

        # Univariate RRR quartile sort on the CONTEMPORANEOUS quarter-end signal
        print(f"\n  Univariate RRR quartile ({signal_label}):")
        ret[f'RRR_Q_{signal_label}'] = ret.groupby('QUARTER')[rrr_col].transform(safe_quartile)
        port_rrr_q = build_portfolio_returns(ret, f'RRR_Q_{signal_label}', market_ret)
        if port_rrr_q is not None:
            results[f'port_rrr_{signal_label.lower()}_q4_vw'] = port_rrr_q
            run_factor_regressions(port_rrr_q, ff_factors, f'RRR {signal_label} Q4 VW')

    results['market_ret'] = market_ret
    return results


def run_capm_vs_sp500(market_ret, returns, ff_factors):
    """CAPM OLS of the sample's value-weighted market portfolio on the S&P 500.

        (market_excess)_t = alpha + beta * (SP500_excess)_t + e_t

    Monthly SIMPLE returns; risk-free = Ken French RF. Reports alpha (monthly and
    annualized), beta, R^2, and Newey-West (HAC) t-stats. This is the real OLS
    version of the correlation / tracking-error check that previously lived in
    diagnose_portfolio_overlap.py (CHECK 7), moved here so it runs with everything
    else.
    """
    if market_ret is None or len(market_ret) == 0:
        print("  No sample market return available; skipping CAPM.")
        return {}

    spx = returns[returns['FIRM'] == 'SPX INDEX'][['Date', 'RET_SIMPLE']].copy()
    if spx.empty:
        print("  SPX INDEX not found in returns; skipping CAPM.")
        return {}
    spx = spx.rename(columns={'RET_SIMPLE': 'SPX'}).set_index('Date').sort_index()

    df = market_ret.to_frame('MKT').join(spx, how='inner').dropna()
    df.index = pd.to_datetime(df.index).to_period('M').to_timestamp('M')
    rf = ff_factors['RF'].reindex(df.index)
    df['MKT_EX'] = df['MKT'] - rf
    df['SPX_EX'] = df['SPX'] - rf
    df = df.dropna(subset=['MKT_EX', 'SPX_EX'])
    if len(df) < 12:
        print(f"  Only {len(df)} overlapping months; skipping CAPM.")
        return {}

    model = sm.OLS(df['MKT_EX'], sm.add_constant(df['SPX_EX'])).fit(
        cov_type='HAC', cov_kwds={'maxlags': 4})
    out = {
        'alpha_month': model.params['const'],
        'alpha_ann': model.params['const'] * 12,
        'alpha_se': model.bse['const'],
        'alpha_t': model.tvalues['const'],
        'alpha_p': model.pvalues['const'],
        'beta': model.params['SPX_EX'],
        'beta_se': model.bse['SPX_EX'],
        'beta_t': model.tvalues['SPX_EX'],
        'r2': model.rsquared,
        'n_obs': int(model.nobs),
    }
    print(f"  CAPM (sample VW market vs S&P 500), N={out['n_obs']} months:")
    print(f"    alpha = {out['alpha_month']*100:.4f}%/mo ({out['alpha_ann']*100:.2f}%/yr), "
          f"SE={out['alpha_se']*100:.4f}, t={out['alpha_t']:.2f}, p=[{out['alpha_p']:.3f}]")
    print(f"    beta  = {out['beta']:.4f}  (SE={out['beta_se']:.4f}, t={out['beta_t']:.2f})")
    print(f"    R^2   = {out['r2']:.4f}")
    pd.DataFrame([out]).to_excel(os.path.join(OUTPUT_DIR, 'capm_vs_sp500.xlsx'), index=False)
    return out


def run_fama_macbeth(panel, ff_factors):
    """MONTHLY Fama-MacBeth on the same timing as the portfolio sorts.

    Author decision 2026-09-27: monthly cross-sections, raw signal only. Each
    holding month, a cross-sectional OLS regresses the firm's monthly EXCESS
    return (in percent) on the raw RRR and controls measured at the signal
    quarter-end that the portfolio sorts use for that month (formed
    FORM_LAG_MONTHS months after the quarter-end, held HOLD_MONTHS months). The
    monthly slopes are then averaged with Newey-West t-stats. This matches the
    monthly frequency of the portfolio tests (about 107 cross-sections instead
    of 36 quarterly ones). The quarterly version, with the 3-month holding
    return compounded into one observation and the industry-adjusted signal in
    two further columns, is superseded. No significance stars are printed;
    exact p-values are reported in brackets.

    HISTORICAL NOTE (quarterly version): the timing correction materially changed the FM result. Under the old
    0-month-gap design (signal at Q -> next calendar quarter, overlapping the
    earnings-announcement window) Adj RRR had t ~ 2.9-3.1; under the correct
    2-month publication gap the cross-sectional linear effect is a clean null
    (2026-09 vintage: t = 0.97, p = .334, Adj RRR alone). The value-weighted,
    tail-driven portfolio long-short remains significant (2.14-2.22%/mo); the
    equal-weighted linear FM does not. See fmb_reconciliation.py for why: the
    effect is concentrated in Q4 rather than linear in the signal level (Table
    5's quartile alphas are +0.51/-0.41/-0.84/-1.71%/mo), RRR is unbounded and
    heavy-tailed (SD 103.02 > mean 81.19) so a levels regression is exposed to
    high-leverage outliers a quartile sort is not, and the signal's own
    mechanical denominator noise (see TA4_RRR_PCT above) is measurement error
    in the regressor, which attenuates an OLS slope toward zero in a way a
    rank-based sort is far less sensitive to. A quartile-dummy FM specification
    (fmb_reconciliation.py) is the direct test of the first explanation.
    """

    results = {}
    df = panel.copy()

    # Monthly risk-free (Ken French, simple) aligned to each holding month.
    rf = ff_factors['RF'].copy()
    rf.index = pd.to_datetime(rf.index).to_period('M')
    df['MONTH_P'] = pd.to_datetime(df['Date']).dt.to_period('M')
    df['RF_M'] = df['MONTH_P'].map(rf)
    df['RET_SIMPLE'] = pd.to_numeric(df['RET_SIMPLE'], errors='coerce')
    for c in ['ADJ_RRR_PCT', 'RRR_PCT', 'SIZE', 'BTM', 'PM_OPER_PCT', 'MCAP_FORM']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # One observation per firm and holding month. Each row of the holding panel
    # already carries the quarter-end signal and controls that the sorts use for
    # that month, so no further lag is applied. The dependent variable is the
    # monthly excess return in percent.
    mdf = df.drop_duplicates(['FIRM', 'MONTH_P']).copy()
    mdf['EXCESS_RET_M_PCT'] = (mdf['RET_SIMPLE'] - mdf['RF_M']) * 100.0

    # (spec name): (regressors, weighted by formation market cap?)
    # Columns (3) and (4), added 2026-10-20, repeat (1) and (2) with each firm
    # weighted by its formation-date market cap (WLS). The sorts are
    # value-weighted while (1) and (2) weight every firm equally, so the pair
    # tests whether weighting explains the gap between the two kinds of evidence.
    specs = {
        '(1) RRR only':                       (['RRR_PCT'], False),
        '(2) RRR + Controls':                 (['RRR_PCT', 'SIZE', 'BTM', 'PM_OPER_PCT'], False),
        '(3) RRR only, value-weighted':       (['RRR_PCT'], True),
        '(4) RRR + Controls, value-weighted': (['RRR_PCT', 'SIZE', 'BTM', 'PM_OPER_PCT'], True),
    }
    y_var = 'EXCESS_RET_M_PCT'

    for spec_name, (x_vars, weighted) in specs.items():
        # Cross-sectional regression each holding month
        period_coefs = []
        needed = [y_var] + x_vars + (['MCAP_FORM'] if weighted else [])
        for mon, group in mdf.groupby('MONTH_P'):
            sub = group[needed].dropna()
            if weighted:
                sub = sub[sub['MCAP_FORM'] > 0]
            if len(sub) < 10:
                continue
            try:
                X = sm.add_constant(sub[x_vars])
                if weighted:
                    model = sm.WLS(sub[y_var], X, weights=sub['MCAP_FORM']).fit()
                else:
                    model = sm.OLS(sub[y_var], X).fit()
                coefs = model.params.to_dict()
                coefs['PERIOD'] = mon
                coefs['N'] = len(sub)
                period_coefs.append(coefs)
            except Exception:
                continue

        if not period_coefs:
            print(f"  {spec_name}: No valid periods")
            continue

        coef_df = pd.DataFrame(period_coefs)

        # Time-series average with Newey-West t-stats (Bartlett kernel)
        T = len(coef_df)
        avg_coefs = coef_df.drop(columns=['PERIOD', 'N']).mean()
        max_lag = max(1, int(np.floor(4 * (T/100)**(2/9))))
        se_nw = {}
        for var in avg_coefs.index:
            series = coef_df[var] - avg_coefs[var]
            gamma_sum = (series**2).mean()
            for j in range(1, max_lag + 1):
                gamma_j = (series.iloc[j:].values * series.iloc[:-j].values).mean()
                gamma_sum += 2 * (1 - j/(max_lag+1)) * gamma_j
            se_nw[var] = np.sqrt(gamma_sum / T)

        t_stats = {v: (avg_coefs[v] / se_nw[v] if se_nw[v] > 0 else np.nan)
                   for v in avg_coefs.index}
        p_values = {v: float(2 * sps.t.sf(abs(t_stats[v]), max(T - 1, 1)))
                    if np.isfinite(t_stats[v]) else np.nan for v in avg_coefs.index}

        print(f"\n  {spec_name} (T={T} months, avg N={coef_df['N'].mean():.0f}):")
        for var in [v for v in avg_coefs.index if v != 'const']:
            print(f"    {var:<25} coef={avg_coefs[var]:>10.4f}  "
                  f"t={t_stats[var]:>7.2f}  p=[{p_values[var]:.3f}]")

        results[spec_name] = {
            'avg_coefs': avg_coefs.to_dict(),
            't_stats': t_stats,
            'p_values': p_values,
            'T': T,
            'avg_N': coef_df['N'].mean(),
        }

    # Export
    fm_summary = []
    for spec_name, res in results.items():
        for var in res['avg_coefs']:
            if var == 'const':
                continue
            fm_summary.append({
                'Spec': spec_name,
                'Variable': var,
                'Coefficient': res['avg_coefs'][var],
                't-stat': res['t_stats'][var],
                'p-value': res['p_values'][var],
                'T': res['T'],
                'Avg N': res['avg_N'],
            })

    fm_df = pd.DataFrame(fm_summary)
    fm_df.to_excel(os.path.join(OUTPUT_DIR, 'fama_macbeth.xlsx'), index=False)
    print(f"\n  Fama-MacBeth results exported to {OUTPUT_DIR}/fama_macbeth.xlsx")

    return results


def run_risk_analysis(panel, ff_factors):
    """Analyze risk characteristics by adj-RRR tercile (T1 = highest RRR).

    Uses the holding panel (same timing as the sorts). Portfolio returns are
    value-weighted SIMPLE returns with formation-date market cap; the wealth
    index is built by compounding simple returns, not exponentiating summed logs.
    """

    print("\n  Risk analysis by RRR terciles:")

    ret = panel.copy()
    ret['RRR_PCT'] = pd.to_numeric(ret['RRR_PCT'], errors='coerce')
    # Raw signal: raw is the paper's primary specification (author decision 2026-09-18).
    ret['RRR_T'] = ret.groupby('QUARTER')['RRR_PCT'].transform(safe_tercile)

    results = {}

    # Build VW (formation market cap) SIMPLE-return portfolios by RRR tercile
    df = ret.dropna(subset=['RRR_T']).copy()
    df['MCAP_FORM'] = pd.to_numeric(df['MCAP_FORM'], errors='coerce')
    df['RET_SIMPLE'] = pd.to_numeric(df['RET_SIMPLE'], errors='coerce')
    df = df.dropna(subset=['MCAP_FORM', 'RET_SIMPLE'])
    df['w'] = df['MCAP_FORM'] / df.groupby(['Date', 'RRR_T'])['MCAP_FORM'].transform('sum')
    df['w_ret'] = df['w'] * df['RET_SIMPLE']

    port = (
        df.groupby(['Date', 'RRR_T'])['w_ret']
        .sum()
        .unstack('RRR_T')
        .sort_index()
    )

    if port.empty:
        print("  No valid portfolio data for risk analysis")
        return results

    # Risk metrics (port columns are SIMPLE monthly returns)
    risk_metrics = []
    for col in port.columns:
        rets = port[col].dropna()
        cum = (1 + rets).cumprod()          # wealth index from simple returns
        cum_max = cum.cummax()
        drawdown = (cum / cum_max - 1).min()

        downside = rets[rets < 0]
        downside_dev = downside.std(ddof=0) if len(downside) > 0 else np.nan

        # Downside beta (beta in down markets)
        port_aligned = port.copy()
        port_aligned.index = pd.to_datetime(port_aligned.index).to_period('M').to_timestamp('M')
        ff_aligned = ff_factors.reindex(port_aligned.index)
        mkt = ff_aligned.get('Mkt-RF', pd.Series(dtype=float))

        down_months = mkt[mkt < 0].index
        if len(down_months) > 5 and col in port_aligned.columns:
            y_down = port_aligned.loc[down_months, col].dropna()
            x_down = mkt.loc[down_months].reindex(y_down.index).dropna()
            y_down = y_down.reindex(x_down.index)
            if len(y_down) > 5:
                X_down = sm.add_constant(x_down)
                model_down = sm.OLS(y_down, X_down).fit()
                downside_beta = model_down.params.iloc[1]
            else:
                downside_beta = np.nan
        else:
            downside_beta = np.nan

        risk_metrics.append({
            'Portfolio': col,
            'Ann Return (%)': rets.mean() * 12 * 100,
            'Ann Vol (%)': rets.std() * np.sqrt(12) * 100,
            'Sharpe': (rets.mean() * 12) / (rets.std() * np.sqrt(12)) if rets.std() > 0 else np.nan,
            'Max Drawdown (%)': drawdown * 100,
            'Downside Dev': downside_dev,
            'Sortino': rets.mean() / downside_dev if downside_dev > 0 else np.nan,
            'Downside Beta': downside_beta,
            'VaR 5% (%)': np.percentile(rets, 5) * 100,
            'Skewness': skew(rets, nan_policy='omit'),
            'Kurtosis': kurtosis(rets, nan_policy='omit', fisher=False),
            'Hit Ratio (%)': (rets > 0).mean() * 100
        })

    risk_df = pd.DataFrame(risk_metrics)
    print(risk_df.to_string(index=False))
    risk_df.to_excel(os.path.join(OUTPUT_DIR, 'risk_analysis.xlsx'), index=False)

    results['risk_metrics'] = risk_df
    return results
