"""ta4_signal_analysis.py -- full treatment of the TA4 (trailing-average-
denominator) signal, given the same analysis as the headline RRR signal.

IN THE PAPER as of 2026-09-16 (author decision): this variant is reported in
Section~6 as supporting evidence for H2's asymmetry claim, NOT as a competing
headline signal. Its long-short alpha is 1.0442%/mo (p=.067) against the raw
baseline's 2.2511%, on a sample nine months shorter, so it is weaker evidence
for H1 than the baseline and the paper says so plainly rather than presenting
it as a clean replication. What it does reproduce is the Q1/Q4 asymmetry.
Its results reach the paper through output/ta4_signal_analysis.xlsx, which
asymmetry_across_variants.py reads for the robustness table.

WHY TA4. The baseline signal divides returning-customer revenue by ONE prior
quarter's total revenue:

    RRR_t = Rev_ret_t / Rev_total_{t-1}

That same Rev_ret_{t-1} is also the baseline signal's OWN numerator one period
earlier (RRR_{t-1} = Rev_ret_{t-1} / Rev_total_{t-2}), and it feeds Rev_total_{t-1}
via Rev_total = Rev_ret + Rev_new. So one unusually strong or weak quarter for a
firm's returning-customer revenue mechanically pushes RRR_{t-1} UP and RRR_t DOWN,
one quarter apart, independent of any real change in customer behavior. This is
the leading candidate explanation for the signal's negative lag-1 autocorrelation,
and the industry-time adjustment cannot remove it because the effect is
firm-specific, not sector-common -- confirmed by the adjusted lag-1
autocorrelation being MORE negative than the raw one (-0.2200 vs -0.2090: the
adjustment isn't touching it).

TA4_RRR replaces the single-quarter denominator with the trailing 4-quarter
average of total revenue, keeping the numerator fully contemporaneous (unlike
TQ_RRR/LR_RRR, which lag the numerator too):

    TA4_RRR_t = Rev_ret_t / mean(Rev_total_{t-1}, ..., Rev_total_{t-4})

The shared term now carries roughly 1/4 of the denominator's weight instead of
all of it, attenuating the mechanical link about fourfold, and the construct
becomes "returning-customer revenue against the firm's normal recent revenue
base" rather than "against one possibly anomalous quarter."

THIS SCRIPT reruns, on TA4_RRR alongside the baseline for direct comparison:
  - quartile characteristics
  - long-short + individual-quartile alphas, FF3/Carhart/FF5, full loadings
  - Fama-MacBeth
  - full risk battery (Sharpe, Sortino, TE, IR, appraisal ratio, up/down beta,
    max drawdown, hit rate)
  - turnover, per re-formation and annualized
  - quartile stay-rates and transition persistence
  - lag-1 / lag-4 firm-level autocorrelation
  - concentration: effective N per leg
  - holdings overlap vs the baseline sort (Jaccard)
  - cumulative-return figure

Baseline reference points (2026-09 vintage): lag-1 autocorr -0.2200, lag-4
+0.4770, cross-sectional SD 41.58pp, Q4 stay-rate 61.2%, annualized turnover
338.12%, FF3 alpha 2.1428%/mo, FF5 alpha 2.2206%/mo.

Outputs
    output/ta4_signal_analysis.xlsx (many sheets)
    Paper_LaTeX/figures/cumret_rrr_ta4.pdf (diagnostic only, not yet \\input anywhere)
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402
import performance_metrics as pm          # noqa: E402
import robustness_diagnostics as rd       # noqa: E402

RUN_ID = datetime.now().strftime('%Y%m%d-%H%M%S')

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
OUTPUT_DIR = av2.OUTPUT_DIR


# ---------------------------------------------------------------------------
# Shared helpers (mirrors analysis_v2._stay_rates, kept local and pure)
# ---------------------------------------------------------------------------

def stay_rates(df, q_col):
    pdf = df[[q_col]].copy().reset_index().sort_values(['FIRM', 'DATE'])
    pdf['Q_PREV'] = pdf.groupby('FIRM')[q_col].shift(1)
    valid = pdf.dropna(subset=[q_col, 'Q_PREV'])
    overall = (valid[q_col] == valid['Q_PREV']).mean()
    per_q = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = valid[valid['Q_PREV'] == q]
        per_q[q] = (sub[q_col] == q).mean() if len(sub) > 0 else np.nan
    return overall, per_q


def flatten_fmb(results):
    """run_fama_macbeth() returns {spec: {'avg_coefs': {...}, 't_stats': {...},
    'p_values': {...}, 'T':.., 'avg_N':..}}. Flatten to one row per (spec, var),
    matching the fm_summary shape the function itself builds before exporting.
    """
    rows = []
    for spec_name, res in results.items():
        for var in res['avg_coefs']:
            if var == 'const':
                continue
            rows.append({
                'Spec': spec_name, 'Variable': var,
                'Coefficient': res['avg_coefs'][var],
                't-stat': res['t_stats'][var],
                'p-value': res['p_values'][var],
                'T': res['T'], 'Avg N': res['avg_N'],
            })
    return pd.DataFrame(rows)


def firm_level_autocorr(df, col, lag_col):
    corrs = []
    for firm in df.index.get_level_values('FIRM').unique():
        try:
            fd = df.xs(firm, level='FIRM')[[col, lag_col]].dropna()
            if len(fd) >= 3:
                corrs.append(fd[col].corr(fd[lag_col]))
        except (KeyError, ValueError):
            continue
    corrs = [c for c in corrs if c == c]
    return {
        'mean': float(np.mean(corrs)) if corrs else np.nan,
        'median': float(np.median(corrs)) if corrs else np.nan,
        'pct_positive': float(np.mean(np.array(corrs) > 0) * 100) if corrs else np.nan,
        'n_firms': len(corrs),
    }


def build_sort(panel, signal_col, market):
    """Quartile sort on signal_col, portfolio returns, factor regressions."""
    p = panel.copy()
    p[signal_col] = pd.to_numeric(p[signal_col], errors='coerce')
    bucket = f'{signal_col}_Q'
    p[bucket] = p.groupby('QUARTER')[signal_col].transform(av2.safe_quartile)
    port = av2.build_portfolio_returns(p, bucket, market)
    port['MKT'] = market.reindex(port.index)
    return p, bucket, port


def risk_battery(port, ff):
    """Sharpe/Sortino/TE/IR/appraisal/up-down beta/max DD/hit rate per column."""
    rf = ff['RF'] if 'RF' in ff.columns else None
    mkt = port['MKT'] if 'MKT' in port.columns else None
    rows = {}
    for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4']:
        if col not in port.columns:
            continue
        raw = pd.to_numeric(port[col], errors='coerce').dropna()
        if rf is not None:
            ex = pm.excess_return(col, raw, rf)
        else:
            ex = raw
        ann_ret = raw.mean() * pm.MONTHS_PER_YEAR
        ann_vol = raw.std(ddof=1) * np.sqrt(pm.MONTHS_PER_YEAR)
        row = {
            'Ann. Return': ann_ret,
            'Ann. Volatility': ann_vol,
            'Sharpe': pm.sharpe_ratio(ex),
            'Sortino': pm.sortino_ratio(ex),
            'Max Drawdown': pm.drawdown_series(raw).min() if len(raw) else np.nan,
        }
        if mkt is not None:
            row['Tracking Error'] = pm.tracking_error(raw, mkt)
            row['Hit Rate vs MKT'] = pm.hit_rate(raw, mkt)
            db, ndb = pm.conditional_beta(raw, mkt, 'down')
            ub, nub = pm.conditional_beta(raw, mkt, 'up')
            row['Downside Beta'] = db
            row['Upside Beta'] = ub
        if col == 'Q1-Q4':
            row['Information Ratio'] = pm.self_information_ratio(raw)
        elif mkt is not None:
            row['Information Ratio'] = pm.information_ratio(raw, mkt)
        rows[col] = row
    return pd.DataFrame(rows).T


def appraisal_ratios(port, ff, reg):
    """Appraisal ratio per column, FF3 and FF5."""
    rf = ff['RF'] if 'RF' in ff.columns else None
    out = {}
    for spec, cols in [('FF3', ['Mkt-RF', 'SMB', 'HML']),
                       ('FF5', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])]:
        for col in ['Q1', 'Q2', 'Q3', 'Q4', 'Q1-Q4']:
            if col not in port.columns:
                continue
            raw = pd.to_numeric(port[col], errors='coerce').dropna()
            ex = pm.excess_return(col, raw, rf) if rf is not None else raw
            ar = pm.appraisal_ratio(ex, ff, cols)
            out[f'{col}_{spec}'] = ar
    return out


def holdings_jaccard(panel_a, bucket_a, panel_b, bucket_b):
    """Per-quarter Jaccard overlap of Q1 and Q4 names between two sorts."""
    rows = []
    for leg in ['Q1', 'Q4']:
        a = (panel_a.loc[panel_a[bucket_a] == leg]
             .drop_duplicates(['FIRM', 'QUARTER'])[['FIRM', 'QUARTER']])
        b = (panel_b.loc[panel_b[bucket_b] == leg]
             .drop_duplicates(['FIRM', 'QUARTER'])[['FIRM', 'QUARTER']])
        for q in sorted(set(a['QUARTER']) | set(b['QUARTER'])):
            na = set(a.loc[a['QUARTER'] == q, 'FIRM'])
            nb = set(b.loc[b['QUARTER'] == q, 'FIRM'])
            if not na and not nb:
                continue
            inter = na & nb
            union = na | nb
            rows.append({
                'LEG': leg, 'QUARTER': str(pd.Period(q, freq='Q')),
                'N_BASELINE': len(na), 'N_TA4': len(nb), 'N_SHARED': len(inter),
                'JACCARD': len(inter) / len(union) if union else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    print('=' * 78)
    print('  TA4 SIGNAL -- FULL TREATMENT, COMPARED AGAINST THE BASELINE')
    print(f'  RUN ID: {RUN_ID}')
    print('=' * 78)

    ff = av2.phase1b_load_ff_factors()
    df_long, df_filtered, _ = av2.phase1_load_and_diagnose(ff)
    returns = av2.phase1c_load_monthly_returns()

    panel = av2.build_holding_panel(df_filtered, returns)
    market = av2.value_weighted_market(panel)

    # RAW-VS-RAW as of 2026-09-16: the first run of this script sorted the
    # baseline on ADJ_RRR_PCT and TA4 on ADJ_TA4_RRR_PCT, which stacks the
    # denominator change together with the same industry-time adjustment the
    # baseline gets. That confounds which of the two caused TA4's alpha loss.
    # This version sorts both RAW, so denominator construction is the only
    # thing that differs between the two columns being compared.
    print('\n  Building sorts (baseline RRR_PCT, TA4 TA4_RRR_PCT -- RAW, no industry-time adjustment)...')
    panel_b, bucket_b, port_b = build_sort(panel, 'RRR_PCT', market)
    panel_t, bucket_t, port_t = build_sort(panel, 'TA4_RRR_PCT', market)

    reg_b = av2.run_factor_regressions(port_b, ff, label='Baseline_RAW_RRR')
    reg_t = av2.run_factor_regressions(port_t, ff, label='TA4_RAW_RRR')
    reg_b_flat = {k: {kk: vv for kk, vv in v.items() if np.isscalar(vv)}
                  for k, v in reg_b.items() if isinstance(v, dict)}
    reg_t_flat = {k: {kk: vv for kk, vv in v.items() if np.isscalar(vv)}
                  for k, v in reg_t.items() if isinstance(v, dict)}

    print('\n  --- HEADLINE ALPHA COMPARISON ---')
    for label, r in [('Baseline', reg_b_flat), ('TA4', reg_t_flat)]:
        for spec in ['FF3', 'FF5']:
            k = f'Q1-Q4_{spec}'
            if k in r:
                a = r[k]
                print(f'    {label:10s} {spec}: {a["alpha"]*100:7.4f}%/mo  '
                      f'SE={a["se_alpha"]*100:.4f}  p={a["alpha_p"]:.4f}  '
                      f'N={int(a["n_obs"])}')

    # --- Fama-MacBeth on TA4 -------------------------------------------------
    # run_fama_macbeth() unconditionally writes output/fama_macbeth.xlsx, the
    # shared file tab_fmb.tex was sourced from. Back it up and restore it
    # afterward so this diagnostic run cannot leave that canonical artifact
    # overwritten with the TA4-relabeled version.
    print('\n  Fama-MacBeth (baseline vs TA4)...')
    _fmb_canonical = os.path.join(OUTPUT_DIR, 'fama_macbeth.xlsx')
    _fmb_backup = None
    if os.path.exists(_fmb_canonical):
        _fmb_backup = open(_fmb_canonical, 'rb').read()

    # Only overwrite RRR_PCT (run_fama_macbeth specs (3)/(4) key off it). ADJ_RRR_PCT
    # is deliberately left untouched -- there is no adjusted TA4 column in this
    # raw-only round, so specs (1)/(2) below still reflect the baseline's own
    # adjusted signal and are not a TA4 comparison; only specs (3)/(4) are.
    panel_fmb_t = panel.copy()
    panel_fmb_t['RRR_PCT'] = panel_fmb_t['TA4_RRR_PCT']  # reuse existing FMB code path
    fmb_b_raw = av2.run_fama_macbeth(panel, ff)
    fmb_t_raw = av2.run_fama_macbeth(panel_fmb_t, ff)
    fmb_b = flatten_fmb(fmb_b_raw)
    fmb_t = flatten_fmb(fmb_t_raw)

    if _fmb_backup is not None:
        with open(_fmb_canonical, 'wb') as fh:
            fh.write(_fmb_backup)
        print(f'    restored canonical {_fmb_canonical} (this script wrote over it twice above)')

    # --- Risk battery + appraisal ratio --------------------------------------
    print('\n  Risk battery...')
    risk_b = risk_battery(port_b, ff)
    risk_t = risk_battery(port_t, ff)
    apr_b = appraisal_ratios(port_b, ff, reg_b_flat)
    apr_t = appraisal_ratios(port_t, ff, reg_t_flat)

    # --- Turnover -------------------------------------------------------------
    print('  Turnover...')
    to_b, _, _, _, to_b_ann = pm.compute_portfolio_turnover(
        panel_b, bucket_b, quarters_between=1,
        reformations_per_year=pm.REFORMATIONS_PER_YEAR)
    to_t, _, _, _, to_t_ann = pm.compute_portfolio_turnover(
        panel_t, bucket_t, quarters_between=1,
        reformations_per_year=pm.REFORMATIONS_PER_YEAR)
    print(f'    Baseline Q1-Q4 annualized turnover: {to_b_ann.get("Q1-Q4", np.nan)*100:.2f}%')
    print(f'    TA4      Q1-Q4 annualized turnover: {to_t_ann.get("Q1-Q4", np.nan)*100:.2f}%')

    # --- Persistence: stay-rates + autocorrelation ----------------------------
    print('  Persistence (stay-rates, autocorrelation)...')
    df_p = df_filtered.copy()
    df_p['RRR_Q_RAW_CHAR'] = df_p.groupby(
        df_p.index.get_level_values('DATE'))['RRR_PCT'].transform(av2.safe_quartile)
    df_p['TA4_Q_RAW_CHAR'] = df_p.groupby(
        df_p.index.get_level_values('DATE'))['TA4_RRR_PCT'].transform(av2.safe_quartile)
    sr_b_ov, sr_b = stay_rates(df_p, 'RRR_Q_RAW_CHAR')
    sr_t_ov, sr_t = stay_rates(df_p, 'TA4_Q_RAW_CHAR')

    # TA4_RRR_PCT_LAG1/LAG4 don't exist in df_filtered (only the ADJ_ lag columns
    # were added in analysis_v2.py); compute both locally, matching the pattern
    # already used here for the lag-4 column.
    df_p['TA4_RRR_PCT_LAG1'] = df_p.groupby(level='FIRM')['TA4_RRR_PCT'].shift(1)
    df_p['TA4_RRR_PCT_LAG4'] = df_p.groupby(level='FIRM')['TA4_RRR_PCT'].shift(4)
    ac_b_1 = firm_level_autocorr(df_p, 'RRR_PCT', 'RRR_PCT_LAG1')
    ac_b_4 = firm_level_autocorr(df_p, 'RRR_PCT', 'RRR_PCT_LAG4')
    ac_t_1 = firm_level_autocorr(df_p, 'TA4_RRR_PCT', 'TA4_RRR_PCT_LAG1')
    ac_t_4 = firm_level_autocorr(df_p, 'TA4_RRR_PCT', 'TA4_RRR_PCT_LAG4')

    cs_sd_b = df_p.groupby(df_p.index.get_level_values('DATE'))['RRR_PCT'].std().mean()
    cs_sd_t = df_p.groupby(df_p.index.get_level_values('DATE'))['TA4_RRR_PCT'].std().mean()

    print(f'\n    {"":22s}{"Baseline":>12s}{"TA4":>12s}')
    print(f'    {"Lag-1 autocorr (mean)":22s}{ac_b_1["mean"]:>12.4f}{ac_t_1["mean"]:>12.4f}')
    print(f'    {"Lag-4 autocorr (mean)":22s}{ac_b_4["mean"]:>12.4f}{ac_t_4["mean"]:>12.4f}')
    print(f'    {"Cross-sect SD (pp)":22s}{cs_sd_b:>12.2f}{cs_sd_t:>12.2f}')
    print(f'    {"Q1 stay-rate (%)":22s}{sr_b["Q1"]*100:>12.1f}{sr_t["Q1"]*100:>12.1f}')
    print(f'    {"Q4 stay-rate (%)":22s}{sr_b["Q4"]*100:>12.1f}{sr_t["Q4"]*100:>12.1f}')

    # --- Concentration: effective N -------------------------------------------
    print('\n  Concentration...')
    def eff_n_summary(panel_x, bucket_x):
        out = {}
        for leg in ['Q1', 'Q4']:
            w = rd.leg_weights_by_quarter(panel_x, bucket_x, leg)
            ens = []
            for q, g in w.groupby('QUARTER'):
                _hhi, en = rd.hhi_effective_n(g['WEIGHT'].values)
                ens.append(en)
            out[leg] = {'mean': float(np.mean(ens)) if ens else np.nan,
                        'median': float(np.median(ens)) if ens else np.nan,
                        'min': float(np.min(ens)) if ens else np.nan}
        return out
    effn_b = eff_n_summary(panel_b, bucket_b)
    effn_t = eff_n_summary(panel_t, bucket_t)
    print(f'    Baseline Q1 effective N: mean {effn_b["Q1"]["mean"]:.2f}, '
          f'median {effn_b["Q1"]["median"]:.2f}')
    print(f'    TA4      Q1 effective N: mean {effn_t["Q1"]["mean"]:.2f}, '
          f'median {effn_t["Q1"]["median"]:.2f}')

    # --- Holdings overlap -----------------------------------------------------
    print('\n  Holdings overlap (Jaccard) vs baseline...')
    overlap = holdings_jaccard(panel_b, bucket_b, panel_t, bucket_t)
    print(f'    Mean Jaccard, Q1: {overlap.loc[overlap.LEG=="Q1","JACCARD"].mean():.3f}  '
          f'Q4: {overlap.loc[overlap.LEG=="Q4","JACCARD"].mean():.3f}')

    # --- Quartile characteristics ----------------------------------------------
    print('\n  Quartile characteristics...')
    char_cols = {
        'Revenue (M)': 'HISTORICAL_MARKET_CAP',  # placeholder swapped below if present
    }
    def quartile_chars(panel_x, bucket_x):
        rows = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            sub = panel_x[panel_x[bucket_x] == q]
            row = {'Quartile': q, 'Avg N': sub.groupby('QUARTER')['FIRM'].nunique().mean()}
            for label, col in [('MCap (M)', 'MCAP_FORM'), ('BTM', 'BTM'),
                               ('SoNR (%)', 'ACQ_RATE_PCT')]:
                if col in sub.columns:
                    v = pd.to_numeric(sub[col], errors='coerce').dropna()
                    row[label] = v.mean() if len(v) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)
    chars_b = quartile_chars(panel_b, bucket_b)
    chars_t = quartile_chars(panel_t, bucket_t)

    # --- Figure -----------------------------------------------------------------
    print('\n  Figure...')

    # --- Write everything --------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx = os.path.join(OUTPUT_DIR, 'ta4_signal_analysis.xlsx')
    with pd.ExcelWriter(xlsx, engine='openpyxl') as xw:
        pd.DataFrame(reg_b_flat).T.to_excel(xw, sheet_name='Baseline_FactorReg')
        pd.DataFrame(reg_t_flat).T.to_excel(xw, sheet_name='TA4_FactorReg')
        port_b.to_excel(xw, sheet_name='Baseline_Returns')
        port_t.to_excel(xw, sheet_name='TA4_Returns')
        risk_b.to_excel(xw, sheet_name='Baseline_RiskBattery')
        risk_t.to_excel(xw, sheet_name='TA4_RiskBattery')
        pd.DataFrame([apr_b]).T.rename(columns={0: 'Appraisal_Ratio'}).to_excel(
            xw, sheet_name='Baseline_Appraisal')
        pd.DataFrame([apr_t]).T.rename(columns={0: 'Appraisal_Ratio'}).to_excel(
            xw, sheet_name='TA4_Appraisal')
        pd.DataFrame({'Baseline_per_reform': to_b, 'Baseline_annualized': to_b_ann,
                      'TA4_per_reform': to_t, 'TA4_annualized': to_t_ann}).to_excel(
            xw, sheet_name='Turnover')
        pd.DataFrame({
            'Metric': ['Lag-1 mean', 'Lag-1 median', 'Lag-1 %pos',
                      'Lag-4 mean', 'Lag-4 median', 'Lag-4 %pos', 'Cross-sect SD (pp)',
                      'Overall stay-rate', 'Q1 stay-rate', 'Q2 stay-rate',
                      'Q3 stay-rate', 'Q4 stay-rate'],
            'Baseline': [ac_b_1['mean'], ac_b_1['median'], ac_b_1['pct_positive'],
                        ac_b_4['mean'], ac_b_4['median'], ac_b_4['pct_positive'], cs_sd_b,
                        sr_b_ov, sr_b['Q1'], sr_b['Q2'], sr_b['Q3'], sr_b['Q4']],
            'TA4': [ac_t_1['mean'], ac_t_1['median'], ac_t_1['pct_positive'],
                    ac_t_4['mean'], ac_t_4['median'], ac_t_4['pct_positive'], cs_sd_t,
                    sr_t_ov, sr_t['Q1'], sr_t['Q2'], sr_t['Q3'], sr_t['Q4']],
        }).to_excel(xw, sheet_name='Persistence', index=False)
        pd.DataFrame(effn_b).T.to_excel(xw, sheet_name='Baseline_Concentration')
        pd.DataFrame(effn_t).T.to_excel(xw, sheet_name='TA4_Concentration')
        overlap.to_excel(xw, sheet_name='Holdings_Overlap', index=False)
        chars_b.to_excel(xw, sheet_name='Baseline_Characteristics', index=False)
        chars_t.to_excel(xw, sheet_name='TA4_Characteristics', index=False)
        fmb_b.to_excel(xw, sheet_name='Baseline_FMB', index=False)
        fmb_t.to_excel(xw, sheet_name='TA4_FMB', index=False)

    print(f'\n  wrote {xlsx}')


    print()
    print('=' * 78)
    print('  DONE')
    print('=' * 78)


if __name__ == '__main__':
    main()
