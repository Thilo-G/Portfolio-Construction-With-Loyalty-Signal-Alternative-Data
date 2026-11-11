"""signal_sort_robustness.py -- alternative sorting signals, one script.

Three of the nine robustness checks the author listed (research_revision_notes.md
lines 36-47) sort the SAME portfolios on a DIFFERENT signal, with timing,
weighting, and industry-time adjustment held identical to the baseline. Rather
than three near-identical scripts, they are parametrised here:

  SoRR  (N042) -- Share of Retained Revenue, Rev_ret_t / Rev_t. Already built in
        analysis_v2.py as SRR_PCT / ADJ_SRR_PCT. Note this is a SHARE of current
        revenue, whereas RRR divides by PRIOR-period revenue: SoRR is bounded
        above by 100% and cannot exceed it, RRR is unbounded. The two therefore
        rank firms differently whenever revenue is growing or shrinking.

  Two-quarter RRR (N045) -- Rev_ret_t / Rev_{t-2}. Sits between the baseline
        (t-1) and LR_RRR (t-4, which collapses to 0.26%/mo, p=.62). Locates
        where the signal's horizon decays.

  Prior-period revenue growth (N047) -- sorts on REV_GROWTH_PCT instead of
        retention. This is the paper's direct answer to "is the RRR premium
        just a growth premium?": if growth alone produced the spread, this
        sort would reproduce it.

Each is reported raw and industry-time adjusted, matching the paper's
raw/adjusted convention, with the adjusted sort as headline.

Outputs
    output/signal_sort_robustness.xlsx
    output/signal_sort_manifest.json   (read by asymmetry_across_variants.py)
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402
import performance_metrics as pm          # noqa: E402

RUN_ID = datetime.now().strftime('%Y%m%d-%H%M%S')

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
OUTPUT_DIR = av2.OUTPUT_DIR

# key, adjusted column, raw column, table label, short label for the xlsx sheet
SIGNALS = [
    ('sorr',      'ADJ_SRR_PCT',    'SRR_PCT',
     'Share of returning revenue (denominator $=$ current revenue)', 'SoRR'),
    ('tq_rrr',    'ADJ_TQ_RRR_PCT', 'TQ_RRR_PCT',
     'Two-quarter \\RRR{} (denominator $t-2$)',          'TwoQtr'),
    # Rev_ret_t / Rev_total_{t-4}: a POINT denominator four quarters back, not a
    # trailing mean. Distinct from TA4, which averages the four prior quarters.
    # Earlier drafts labelled this "4-quarter-trailing", which invited exactly
    # that confusion; the label is corrected here.
    ('lr_rrr',    'ADJ_LR_RRR_PCT', 'LR_RRR_PCT',
     'Four-quarter \\RRR{} (denominator $t-4$)',         'FourQtr'),
    # RCB3 (author-proposed 2026-09-18) was removed from the paper on
    # 2026-09-27 ("out, wrong signal") and replaced by the moving average below.
    # Mean of RRR over the four quarters before the measurement quarter,
    # t-1 to t-4 (author decision 2026-09-27). Raw only.
    ('ma4lag',    None,             'MA4L_RRR_PCT',
     'Four-quarter moving average of \\RRR{} (quarters $t-4$ to $t-1$)', 'MA4L'),
    ('revgrowth', None,             'REV_GROWTH_PCT',
     'Prior-period revenue growth',                      'RevGrowth'),
]


def firm_lag1_autocorr(panel, signal_col):
    """Mean across firms of the within-firm lag-1 autocorrelation of the signal.

    Reported because two of the denominator variants were built to reduce the
    mechanical dependence between consecutive quarters, and the honest way to
    show whether they did is to report what the dependence became.
    """
    d = panel[['FIRM', 'QUARTER', signal_col]].dropna().copy()
    d[signal_col] = pd.to_numeric(d[signal_col], errors='coerce')
    d = d.drop_duplicates(['FIRM', 'QUARTER']).sort_values(['FIRM', 'QUARTER'])
    out = []
    for _firm, g in d.groupby('FIRM'):
        v = g[signal_col].dropna()
        if len(v) >= 4:
            c = v.corr(v.shift(1))
            if c == c:
                out.append(c)
    return float(np.mean(out)) if out else float('nan')


def sort_on(panel, signal_col, bucket_name):
    """Assign quartiles on signal_col within each formation quarter.

    Q1 = highest signal value, matching av2.safe_quartile's convention and the
    baseline RRR sort, so Q1-Q4 is directionally comparable across every sort.
    """
    p = panel.copy()
    p[signal_col] = pd.to_numeric(p[signal_col], errors='coerce')
    p[bucket_name] = p.groupby('QUARTER')[signal_col].transform(av2.safe_quartile)
    return p


def run_one(panel, ff, signal_col, label, market):
    """Full sort, portfolio build, factor regression and turnover for one signal."""
    bucket = 'SORT_Q'
    p = sort_on(panel, signal_col, bucket)

    n_q = p.loc[p[bucket].notna(), 'QUARTER'].nunique()
    n_firms = p.loc[p[bucket].notna(), 'FIRM'].nunique()
    if n_q == 0:
        print(f'    {label}: no valid formation quarters, skipped')
        return None

    port = av2.build_portfolio_returns(p, bucket, market)
    port['MKT'] = market.reindex(port.index)

    reg = av2.run_factor_regressions(port, ff, label=label)
    reg_flat = {k: {kk: vv for kk, vv in v.items() if np.isscalar(vv)}
                for k, v in reg.items() if isinstance(v, dict)}

    avg, n_trans, _detail, n_skip, avg_ann = pm.compute_portfolio_turnover(
        p, bucket, quarters_between=1,
        reformations_per_year=pm.REFORMATIONS_PER_YEAR)

    # Quartile stay-rate: share of firms in the same quartile at the next
    # reformation. Read straight off the bucket column so it matches this sort.
    q = p[['FIRM', 'QUARTER', bucket]].dropna().drop_duplicates(['FIRM', 'QUARTER'])
    q = q.sort_values(['FIRM', 'QUARTER'])
    q['PRIOR'] = q.groupby('FIRM')[bucket].shift(1)
    qv = q.dropna(subset=['PRIOR'])
    stay_rate = float((qv[bucket] == qv['PRIOR']).mean()) if len(qv) else float('nan')
    stay_by_q = {k: float(v) for k, v in
                 qv.assign(same=qv[bucket] == qv['PRIOR'])
                   .groupby(bucket, observed=True)['same'].mean().items()}
    ac1 = firm_lag1_autocorr(p, signal_col)

    r3 = reg_flat.get('Q1-Q4_FF3', {})
    r5 = reg_flat.get('Q1-Q4_FF5', {})
    nan = float('nan')
    print(f'    {label}: FF3 {r3.get("alpha", nan) * 100:7.4f}%/mo '
          f'[p={r3.get("alpha_p", nan):.3f}]  '
          f'FF5 {r5.get("alpha", nan) * 100:7.4f}%/mo '
          f'[p={r5.get("alpha_p", nan):.3f}]  '
          f'N={int(r3.get("n_obs", 0))}  firms={n_firms}  quarters={n_q}')

    return {
        'label': label,
        'panel': p,
        'bucket': bucket,
        'port': port,
        'reg': reg_flat,
        'turnover_ann': {k: v * 100 for k, v in avg_ann.items()},
        'turnover_per': {k: v * 100 for k, v in avg.items()},
        'n_months': len(port),
        'n_firms': n_firms,
        'n_quarters': n_q,
        'stay_rate': stay_rate,
        'stay_by_quartile': stay_by_q,
        'lag1_autocorr': ac1,
    }


def build_all():
    """Load the data and run every alternative sort."""
    ff = av2.phase1b_load_ff_factors()
    df_long, df_filtered, _ = av2.phase1_load_and_diagnose(ff)
    returns = av2.phase1c_load_monthly_returns()

    panel = av2.build_holding_panel(df_filtered, returns)
    market = av2.value_weighted_market(panel)

    print('\n  Sorts:')
    # Raw baseline: raw is the paper's primary specification (2026-09-18).
    # The industry-time adjusted signal is run alongside as one robustness row.
    baseline = run_one(panel, ff, 'RRR_PCT', 'Baseline_RRR', market)

    results = {}
    results['rrr_adj'] = run_one(panel, ff, 'ADJ_RRR_PCT', 'rrr_adj', market)

    for key, adj_col, raw_col, label, _short in SIGNALS:
        if adj_col is not None:
            if adj_col not in panel.columns:
                print(f'    {label}: column {adj_col} missing from panel, skipped')
            else:
                results[key + '_adj'] = run_one(panel, ff, adj_col, f'{key}_adj', market)
        if raw_col not in panel.columns:
            print(f'    {label}: column {raw_col} missing from panel, skipped')
        else:
            results[key + '_raw'] = run_one(panel, ff, raw_col, f'{key}_raw', market)

    return panel, market, ff, baseline, results


def main():
    print('=' * 78)
    print('  ALTERNATIVE SORTING-SIGNAL ROBUSTNESS')
    print(f'  RUN ID: {RUN_ID}')
    print('=' * 78)

    panel, market, ff, baseline, results = build_all()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx = os.path.join(OUTPUT_DIR, 'signal_sort_robustness.xlsx')
    with pd.ExcelWriter(xlsx, engine='openpyxl') as xw:
        if baseline:
            baseline['port'].to_excel(xw, sheet_name='Baseline_Returns')
            pd.DataFrame(baseline['reg']).T.to_excel(xw, sheet_name='Baseline_Reg')
        for key, res in results.items():
            if res is None:
                continue
            short = key[:24]
            res['port'].to_excel(xw, sheet_name=f'{short}_ret'[:31])
            pd.DataFrame(res['reg']).T.to_excel(xw, sheet_name=f'{short}_reg'[:31])
    print(f'  wrote {xlsx}')

    def _summary(v):
        if v is None:
            return None
        out = {
            'n_months': v['n_months'],
            'n_firms': v['n_firms'],
            'n_quarters': v['n_quarters'],
            'turnover_ann_ls': v['turnover_ann'].get('Q1-Q4'),
            'stay_rate': v.get('stay_rate'),
            'stay_by_quartile': v.get('stay_by_quartile'),
            'lag1_autocorr': v.get('lag1_autocorr'),
        }
        # Per-leg alphas as well as the long-short: the combined robustness
        # table in the paper reports Q1 and Q4 for every check, because the
        # paper's second finding is about which leg carries the spread.
        for leg in ('Q1', 'Q4', 'Q1-Q4'):
            for mdl in ('FF3', 'FF5'):
                r = v['reg'].get(f'{leg}_{mdl}', {})
                tag = leg.replace('-', '') + '_' + mdl.lower()
                out[tag + '_alpha_pct_mo'] = r.get('alpha')
                out[tag + '_se'] = r.get('se_alpha')
                out[tag + '_p'] = r.get('alpha_p')
                out[tag + '_n'] = r.get('n_obs')
        return out

    manifest = {
        'run_id': RUN_ID,
        'script': 'signal_sort_robustness.py',
        'form_lag_months': av2.FORM_LAG_MONTHS,
        'hold_months': av2.HOLD_MONTHS,
        'signals': {k: _summary(v) for k, v in
                    list(results.items()) + [('baseline', baseline)]},
    }
    with open(os.path.join(OUTPUT_DIR, 'signal_sort_manifest.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, default=float)

    print()
    print('=' * 78)
    print('  DONE')
    print('=' * 78)


if __name__ == '__main__':
    main()
