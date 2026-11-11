"""estimation_variants_quartiles.py -- per-quartile alphas and turnover for the
robustness checks that hold the SIGNAL fixed and vary the estimation or sample.

WHY THIS EXISTS. The paper's combined robustness table reports Q1 alpha, Q4
alpha and the long-short for every check, because the paper's second finding is
about which leg carries the spread. The signal-definition checks already report
all three (signal_sort_robustness.py). The estimation and sample checks did not:
they were only ever run for the long-short. This script fills that gap, so the
combined table has one column set across every row instead of blanks in half of
it.

SIGNAL BASIS: raw RRR_PCT throughout. Raw is the paper's primary specification
(author decision 2026-09-18), so every variant here perturbs the raw baseline.

Variants:
  1. First-half subperiod
  2. Second-half subperiod
  3. Excluding COVID-19 (2020Q1-2021Q2), dropped from formation and estimation
  4. Equal-weighted in place of value-weighted
  5. Five-month formation lag in place of two

The Consumer-Discretionary-only and trading-cost checks are deliberately absent:
the author removed both from the paper on 2026-09-18.

Outputs
    output/estimation_variants_quartiles.xlsx
    output/estimation_variants_manifest.json
"""

import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402
import performance_metrics as pm          # noqa: E402

RUN_ID = datetime.now().strftime('%Y%m%d-%H%M%S')
OUTPUT_DIR = av2.OUTPUT_DIR

COVID_START = '2020-01-01'
COVID_END = '2021-06-30'
SIGNAL_COL = 'RRR_PCT'


def sort_raw(panel, bucket='RRR_Q_MAIN'):
    """Raw-RRR quartiles within each formation quarter. Q1 = highest."""
    p = panel.copy()
    p[SIGNAL_COL] = pd.to_numeric(p[SIGNAL_COL], errors='coerce')
    p[bucket] = p.groupby('QUARTER')[SIGNAL_COL].transform(av2.safe_quartile)
    return p, bucket


def equal_weighted_portfolio(panel, bucket):
    """Equal-weighted simple returns per bucket, plus long-short and market.

    Mirrors analysis_v2's equal-weighted block: a plain mean of SIMPLE monthly
    returns, never of log returns.
    """
    d = panel.dropna(subset=[bucket]).copy()
    d['RET_SIMPLE'] = pd.to_numeric(d['RET_SIMPLE'], errors='coerce')
    port = (d.groupby(['Date', bucket], observed=True)['RET_SIMPLE']
              .mean().unstack(bucket))
    if 'Q1' in port.columns and 'Q4' in port.columns:
        port['Q1-Q4'] = port['Q1'] - port['Q4']
    port['MKT'] = d.groupby('Date')['RET_SIMPLE'].mean().reindex(port.index)
    return port


def extract(reg, turnover_ann, n_firms, n_quarters, note):
    """Flatten one variant's regressions into the manifest's row shape."""
    out = {'note': note, 'n_firms': n_firms, 'n_quarters': n_quarters,
           'turnover_ann_ls': turnover_ann}
    for leg in ('Q1', 'Q4', 'Q1-Q4'):
        for mdl in ('FF3', 'FF5'):
            r = reg.get(f'{leg}_{mdl}', {}) or {}
            tag = leg.replace('-', '') + '_' + mdl.lower()
            out[tag + '_alpha_pct_mo'] = r.get('alpha')
            out[tag + '_se'] = r.get('se_alpha')
            out[tag + '_p'] = r.get('alpha_p')
            out[tag + '_n'] = r.get('n_obs')
    return out


def _flat(reg):
    return {k: {kk: vv for kk, vv in v.items() if np.isscalar(vv)}
            for k, v in reg.items() if isinstance(v, dict)}


def _turnover(panel, bucket):
    _avg, _n, _d, _skip, avg_ann = pm.compute_portfolio_turnover(
        panel, bucket, quarters_between=1,
        reformations_per_year=pm.REFORMATIONS_PER_YEAR)
    return float(avg_ann.get('Q1-Q4', np.nan)) * 100


def _report(label, row):
    def g(k):
        v = row.get(k)
        return float('nan') if v is None else v
    print(f"    {label}: "
          f"LS {g('Q1Q4_ff5_alpha_pct_mo') * 100:7.4f}%/mo [p={g('Q1Q4_ff5_p'):.3f}]  "
          f"Q1 {g('Q1_ff5_alpha_pct_mo') * 100:7.4f} [p={g('Q1_ff5_p'):.3f}]  "
          f"Q4 {g('Q4_ff5_alpha_pct_mo') * 100:7.4f} [p={g('Q4_ff5_p'):.3f}]  "
          f"N={int(g('Q1Q4_ff5_n') or 0)}")


def main():
    print('=' * 78)
    print('  ESTIMATION / SAMPLE VARIANTS: per-quartile alphas and turnover')
    print(f'  RUN ID: {RUN_ID}   signal: {SIGNAL_COL} (raw)')
    print('=' * 78)

    ff = av2.phase1b_load_ff_factors()
    df_long, df_filtered, _ = av2.phase1_load_and_diagnose(ff)
    returns = av2.phase1c_load_monthly_returns()

    panel = av2.build_holding_panel(df_filtered, returns)
    panel, bucket = sort_raw(panel)
    market = av2.value_weighted_market(panel)

    rows = {}
    ports = {}

    print('\n  Variants:')

    # -- baseline, for reference in the same shape ---------------------------
    base_port = av2.build_portfolio_returns(panel, bucket, market)
    base_reg = _flat(av2.run_factor_regressions(base_port, ff, 'Baseline raw RRR'))
    rows['baseline'] = extract(
        base_reg, _turnover(panel, bucket),
        int(panel.loc[panel[bucket].notna(), 'FIRM'].nunique()),
        int(panel.loc[panel[bucket].notna(), 'QUARTER'].nunique()),
        'Raw RRR, value-weighted, two-month formation lag')
    ports['baseline'] = base_port
    _report('baseline', rows['baseline'])

    # -- 1 & 2: sample halves ------------------------------------------------
    # Split the RETURN series at its midpoint, matching how the paper's
    # subperiod rows were always defined: same portfolio, two estimation
    # windows, not two separately re-sorted samples.
    mid = len(base_port) // 2
    for key, label, sl in [
            ('first_half', 'First-half subperiod', slice(None, mid)),
            ('second_half', 'Second-half subperiod', slice(mid, None))]:
        sub_port = base_port.iloc[sl]
        reg = _flat(av2.run_factor_regressions(sub_port, ff, label))
        span = f'{sub_port.index.min():%Y-%m} to {sub_port.index.max():%Y-%m}'
        rows[key] = extract(reg, rows['baseline']['turnover_ann_ls'],
                            rows['baseline']['n_firms'], None,
                            f'{label} ({span}); portfolio unchanged, estimation window split')
        ports[key] = sub_port
        _report(key, rows[key])

    # -- 3: excluding COVID --------------------------------------------------
    # Dropped from formation AND estimation entirely, not absorbed by a dummy.
    nc = panel[~((panel['Date'] >= COVID_START) & (panel['Date'] <= COVID_END))].copy()
    nc, nc_bucket = sort_raw(nc)
    nc_market = av2.value_weighted_market(nc)
    nc_port = av2.build_portfolio_returns(nc, nc_bucket, nc_market)
    nc_reg = _flat(av2.run_factor_regressions(nc_port, ff, 'Excluding COVID-19'))
    rows['no_covid'] = extract(
        nc_reg, _turnover(nc, nc_bucket),
        int(nc.loc[nc[nc_bucket].notna(), 'FIRM'].nunique()),
        int(nc.loc[nc[nc_bucket].notna(), 'QUARTER'].nunique()),
        'Excluding 2020Q1-2021Q2 from formation and estimation')
    ports['no_covid'] = nc_port
    _report('no_covid', rows['no_covid'])

    # -- 4: equal-weighted ---------------------------------------------------
    ew_port = equal_weighted_portfolio(panel, bucket)
    ew_reg = _flat(av2.run_factor_regressions(ew_port, ff, 'Equal-weighted'))
    rows['equal_weighted'] = extract(
        ew_reg, _turnover(panel, bucket),
        rows['baseline']['n_firms'], rows['baseline']['n_quarters'],
        'Equal-weighted in place of value-weighted; same sort, same timing')
    ports['equal_weighted'] = ew_port
    _report('equal_weighted', rows['equal_weighted'])

    # -- 5: five-month formation lag ----------------------------------------
    # Holds the SIGNAL fixed and moves the trade date from two months after
    # quarter-end to five. This is a decay test, and is a different question
    # from the two-quarter signal, which moves the denominator instead.
    panel5 = av2.build_holding_panel(df_filtered, returns,
                                     form_lag=av2.FORM_LAG_MONTHS + 3,
                                     hold=av2.HOLD_MONTHS)
    panel5, b5 = sort_raw(panel5)
    market5 = av2.value_weighted_market(panel5)
    port5 = av2.build_portfolio_returns(panel5, b5, market5)
    reg5 = _flat(av2.run_factor_regressions(port5, ff, 'Five-month formation lag'))
    rows['lag5'] = extract(
        reg5, _turnover(panel5, b5),
        int(panel5.loc[panel5[b5].notna(), 'FIRM'].nunique()),
        int(panel5.loc[panel5[b5].notna(), 'QUARTER'].nunique()),
        'Formation five months after quarter-end instead of two; signal unchanged')
    ports['lag5'] = port5
    _report('lag5', rows['lag5'])

    # -- export --------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx = os.path.join(OUTPUT_DIR, 'estimation_variants_quartiles.xlsx')
    with pd.ExcelWriter(xlsx, engine='openpyxl') as xw:
        pd.DataFrame(rows).T.to_excel(xw, sheet_name='Summary')
        for name, p in ports.items():
            p.to_excel(xw, sheet_name=name[:31])
    print(f'\n  wrote {xlsx}')

    manifest = {'run_id': RUN_ID,
                'script': 'estimation_variants_quartiles.py',
                'signal_col': SIGNAL_COL,
                'form_lag_months': av2.FORM_LAG_MONTHS,
                'hold_months': av2.HOLD_MONTHS,
                'variants': rows}
    with open(os.path.join(OUTPUT_DIR, 'estimation_variants_manifest.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, default=float)
    print('  wrote estimation_variants_manifest.json')

    print()
    print('=' * 78)
    print('  DONE')
    print('=' * 78)


if __name__ == '__main__':
    main()
