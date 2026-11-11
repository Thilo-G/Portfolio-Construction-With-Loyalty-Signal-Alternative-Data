"""annual_rrr_robustness.py -- Revenue Retention Rates formed ANNUALLY.

Robustness test requested 2026-09-08. The paper's headline signal is a
quarterly revenue retention rate. This script asks whether the same effect
survives when the signal is built once a year instead, which is the frequency
at which a discretionary manager would realistically refresh a customer-quality
view, and which lets turnover be compared like for like against the baseline.

Design, deliberately parallel to the quarterly baseline:
  * signal:      RRR_annual_t = Rev_returning,t / (Rev_new,t-1 + Rev_returning,t-1)
                 then demeaned within GICS sector x year (the paper's ADJ signal)
  * sort:        quartiles, Q1 = highest RRR (analysis_v2.safe_quartile)
  * timing:      signal at 31 Dec of year Y, formed 2 calendar months later,
                 held 12 months  -> returns earned Mar Y+1 .. Feb Y+2
  * weighting:   value-weighted on formation-date market cap
  * evaluation:  FF3 / FF5 alphas with Newey-West standard errors, plus
                 turnover per re-formation and annualised

Every piece of machinery (build_holding_panel, build_portfolio_returns,
run_factor_regressions, safe_quartile, value_weighted_market) is imported from
analysis_v2 rather than reimplemented, so the annual test and the quarterly
baseline cannot silently drift apart.

CAVEAT, reported in the output and in the table notes: the annual revenue file
covers 2017-2025, giving 9 annual observations, hence 8 usable RRR values and
roughly 7 complete plus 1 truncated formation years. The month count is
comparable to the baseline, but the number of INDEPENDENT portfolio formations
is very small and the standard errors should be read with that in mind. This is
a robustness check, not a standalone test.

Inputs
    Data/20260409-Firms-Revenue-2025-all-Python.xlsx, sheet Year_Revenue_Value
        (1,753 tickers -- the analysis sample is a strict subset and is filtered
        out of it)
Outputs
    output/annual_rrr_robustness.xlsx   (read by asymmetry_across_variants.py)
    output/annual_rrr_manifest.json
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

# Annual signal: measured at 31 Dec, traded 2 months later, held a full year.
ANNUAL_FORM_LAG_MONTHS = 2
ANNUAL_HOLD_MONTHS = 12
ANNUAL_REFORMATIONS_PER_YEAR = 1
ANNUAL_QUARTERS_BETWEEN = 4      # consecutive formations sit four quarters apart


# =============================================================================
# STEP 1 -- LOAD AND FILTER THE ANNUAL REVENUE PANEL
# =============================================================================

def load_annual_revenue(sample_firms):
    """Load Year_Revenue_Value and restrict it to the analysis sample.

    The workbook holds 1,753 tickers, most of them private companies with
    placeholder Bloomberg identifiers and no listed equity. Only firms that
    survive the quarterly pipeline's own screens can be used, so the caller
    passes that list in rather than this script re-deriving it.
    """
    print('=' * 78)
    print('STEP 1: ANNUAL REVENUE PANEL')
    print('=' * 78)

    raw = pd.read_excel(av2.ANNUAL_REVENUE_WORKBOOK,
                        sheet_name=av2.SHEET_ANNUAL_REVENUE, header=None)

    header = raw.iloc[:2]
    data = raw.iloc[2:].copy()
    combined = header.apply(lambda x: x.astype(str).str.strip(), axis=0)
    cols = combined.apply(
        lambda x: '.'.join(v for v in x if v and v.lower() != 'nan'), axis=0)
    data.columns = cols
    data = data.rename(columns={data.columns[0]: 'Date'})
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data[data['Date'].notna()].reset_index(drop=True)
    data = data.loc[:, ~data.columns.duplicated()]

    print(f'  Raw sheet: {raw.shape[0]} rows x {raw.shape[1]} cols')
    print(f'  Annual dates: {data["Date"].min():%Y-%m-%d} to '
          f'{data["Date"].max():%Y-%m-%d} ({len(data)} observations)')

    new_cols = [c for c in data.columns if '#New_Customers' in c]
    universe = {c.rsplit('.', 1)[0].upper().strip() for c in new_cols}
    print(f'  Universe in file: {len(universe)} tickers')

    sample_firms = {f.upper().strip() for f in sample_firms}
    matched = sorted(sample_firms & universe)
    missing = sorted(sample_firms - universe)
    print(f'  Analysis sample: {len(sample_firms)} firms')
    print(f'  Matched in annual file: {len(matched)} '
          f'({100.0 * len(matched) / max(len(sample_firms), 1):.1f}%)')
    if missing:
        print(f'  !! NOT FOUND in the annual file ({len(missing)}): {missing}')

    lookup = {c.rsplit('.', 1)[0].upper().strip(): c for c in data.columns
              if '#New_Customers' in c}
    lookup_r = {c.rsplit('.', 1)[0].upper().strip(): c for c in data.columns
                if '#Returning_Customers' in c}

    records = []
    for firm in matched:
        if firm not in lookup or firm not in lookup_r:
            continue
        sub = pd.DataFrame({
            'DATE': data['Date'],
            'NEW': pd.to_numeric(data[lookup[firm]], errors='coerce'),
            'RET': pd.to_numeric(data[lookup_r[firm]], errors='coerce'),
        })
        sub['TOTAL'] = sub['NEW'] + sub['RET']
        # Same formula as analysis_v2.py:210, one year apart instead of one quarter.
        sub['RRR_PCT'] = sub['RET'] / sub['TOTAL'].shift(1).replace(0, np.nan) * 100
        sub['ACQ_RATE_PCT'] = sub['NEW'] / sub['NEW'].shift(1).replace(0, np.nan) * 100
        sub['REV_GROWTH_PCT'] = ((sub['TOTAL'] - sub['TOTAL'].shift(1))
                                 / sub['TOTAL'].shift(1).replace(0, np.nan) * 100)
        sub['FIRM'] = firm
        records.append(sub)

    if not records:
        raise ValueError('No sample firm matched the annual revenue file.')

    panel = pd.concat(records, ignore_index=True).dropna(subset=['RRR_PCT'])
    print(f'  Annual RRR observations: {len(panel)} '
          f'({panel["FIRM"].nunique()} firms, {panel["DATE"].nunique()} year-ends)')
    print(f'  Formation years: {sorted(panel["DATE"].dt.year.unique())}')
    return panel, matched, missing


def attach_sector_and_adjust(panel, df_filtered):
    """Attach GICS sector and demean the signal within sector x year."""
    print()
    print('  Industry x year adjustment...')
    sec = (df_filtered.reset_index()[['FIRM', 'SECTOR']]
           .dropna().drop_duplicates('FIRM'))
    sec['FIRM'] = sec['FIRM'].str.upper().str.strip()
    panel = panel.merge(sec, on='FIRM', how='left')

    n_nosec = int(panel['SECTOR'].isna().sum())
    if n_nosec:
        print(f'    dropping {n_nosec} obs without a sector')
        panel = panel[panel['SECTOR'].notna()]

    panel['ADJ_RRR_PCT'] = panel['RRR_PCT'] - panel.groupby(
        [panel['DATE'], 'SECTOR'])['RRR_PCT'].transform('mean')
    panel['ADJ_ACQ_RATE_PCT'] = panel['ACQ_RATE_PCT'] - panel.groupby(
        [panel['DATE'], 'SECTOR'])['ACQ_RATE_PCT'].transform('mean')
    print(f'    ADJ_RRR_PCT: mean {panel["ADJ_RRR_PCT"].mean():.4f}, '
          f'SD {panel["ADJ_RRR_PCT"].std():.4f}')
    return panel


def merge_quarterly_controls(annual_panel, df_filtered):
    """Carry market cap and price over from the quarterly panel's Q4 row.

    build_holding_panel needs HISTORICAL_MARKET_CAP and PX_LAST at the signal
    date to derive the formation-date market cap used for value weighting. The
    annual revenue file carries neither.

    Matched by (FIRM, calendar YEAR), not exact date. The quarterly panel's Q4
    row is dated at the last TRADING day of December (e.g. 2022-12-30, since
    2022-12-31 was a Saturday; 2023-12-29, since 2023-12-31 was a Sunday), while
    the annual revenue file's year-end is always literally Dec 31. An exact-date
    merge silently drops every firm in whichever years December 31 is not a
    trading day -- it found ZERO matches for 2022 and 2023 here, which is what
    caused those two formations to contribute no portfolio months at all,
    despite the signal and the return data both being fully present for them.
    """
    q = df_filtered.reset_index()[
        ['FIRM', 'DATE', 'HISTORICAL_MARKET_CAP', 'PX_LAST',
         'SIZE', 'BTM', 'PM_OPER_PCT']].copy()
    q['FIRM'] = q['FIRM'].str.upper().str.strip()
    q['DATE'] = pd.to_datetime(q['DATE'])
    q = q[q['DATE'].dt.quarter == 4].copy()   # keep only the Q4 (year-end) row
    q['YEAR'] = q['DATE'].dt.year
    q = q.drop(columns=['DATE'])

    ap = annual_panel.copy()
    ap['YEAR'] = ap['DATE'].dt.year
    merged = ap.merge(q, on=['FIRM', 'YEAR'], how='left').drop(columns=['YEAR'])
    n_nomcap = int(merged['HISTORICAL_MARKET_CAP'].isna().sum())
    print(f'    obs without market cap at the year-end: {n_nomcap} of {len(merged)}')
    return merged


# =============================================================================
# STEP 2 -- PORTFOLIOS
# =============================================================================

def build_annual_portfolios(panel, returns):
    """Quartile sort on the annual adjusted RRR, held for twelve months."""
    print()
    print('=' * 78)
    print('STEP 2: ANNUAL PORTFOLIO CONSTRUCTION')
    print('=' * 78)

    df = panel.copy()
    df['YEAR'] = df['DATE'].dt.year
    # Raw annual signal: raw is the paper's primary specification (author decision 2026-09-18).
    df['RRR_Q'] = df.groupby('YEAR')['RRR_PCT'].transform(av2.safe_quartile)

    counts = df.groupby(['YEAR', 'RRR_Q'], observed=True).size().unstack(fill_value=0)
    print('  Firms per quartile per formation year:')
    print(counts.to_string())

    idx = df.set_index(['FIRM', 'DATE'])
    holding = av2.build_holding_panel(
        idx, returns,
        form_lag=ANNUAL_FORM_LAG_MONTHS, hold=ANNUAL_HOLD_MONTHS)

    holding['RRR_Q'] = holding.groupby('QUARTER')['RRR_PCT'].transform(av2.safe_quartile)
    market = av2.value_weighted_market(holding)
    port = av2.build_portfolio_returns(holding, 'RRR_Q', market)
    port['MKT'] = market.reindex(port.index)

    n_form = holding['QUARTER'].nunique()
    print(f'  Portfolio months: {len(port)} '
          f'({port.index.min():%Y-%m} to {port.index.max():%Y-%m})')
    print(f'  Independent formation events: {n_form}')
    if n_form < 10:
        print(f'  !! Only {n_form} formations. Alphas below are reported with')
        print('  !! Newey-West errors, but inference rests on very few independent')
        print('  !! portfolio rebuilds. Treat as a robustness check, not a test.')
    return port, holding, counts, n_form


def compute_annual_turnover(holding):
    """Turnover per re-formation and annualised, for the annual strategy."""
    print()
    print('  Turnover...')
    avg, n_trans, detail, n_skip, avg_ann = pm.compute_portfolio_turnover(
        holding, 'RRR_Q',
        quarters_between=ANNUAL_QUARTERS_BETWEEN,
        reformations_per_year=ANNUAL_REFORMATIONS_PER_YEAR)
    out = pd.DataFrame({
        'Turnover (per reformation, %)': {k: v * 100 for k, v in avg.items()},
        'Turnover (annualized, %)': {k: v * 100 for k, v in avg_ann.items()},
    }).T
    print(f'    {n_trans} transitions used, {n_skip} skipped')
    print(out.round(2).to_string())
    return out, n_trans, n_skip


def compare_signals(annual_panel, df_filtered):
    """Correlate the annual signal with the quarterly one at the same year-ends."""
    print()
    print('  Annual vs quarterly signal at 31 Dec...')
    q = df_filtered.reset_index()[['FIRM', 'DATE', 'ADJ_RRR_PCT']].copy()
    q['FIRM'] = q['FIRM'].str.upper().str.strip()
    q['DATE'] = pd.to_datetime(q['DATE'])
    q = q.rename(columns={'ADJ_RRR_PCT': 'ADJ_RRR_QUARTERLY'})
    merged = annual_panel[['FIRM', 'DATE', 'ADJ_RRR_PCT']].merge(q, on=['FIRM', 'DATE'], how='inner')
    merged = merged.dropna()
    if merged.empty:
        return pd.DataFrame()
    pear = merged['ADJ_RRR_PCT'].corr(merged['ADJ_RRR_QUARTERLY'])
    spear = merged['ADJ_RRR_PCT'].corr(merged['ADJ_RRR_QUARTERLY'], method='spearman')
    print(f'    n={len(merged)}, Pearson={pear:.4f}, Spearman={spear:.4f}')
    return pd.DataFrame([{'N': len(merged), 'Pearson': pear, 'Spearman': spear}])


# =============================================================================
# STEP 4 -- OUTPUTS
# =============================================================================

def main():
    print('=' * 78)
    print('  ANNUAL RRR ROBUSTNESS TEST')
    print(f'  RUN ID: {RUN_ID}')
    print(f'  Timing: signal at 31 Dec, form +{ANNUAL_FORM_LAG_MONTHS}m, '
          f'hold {ANNUAL_HOLD_MONTHS}m')
    print('=' * 78)

    ff = av2.phase1b_load_ff_factors()
    df_long, df_filtered, _ = av2.phase1_load_and_diagnose(ff)
    returns = av2.phase1c_load_monthly_returns()

    # The analysis sample: firms that actually make it into a portfolio, i.e.
    # they carry both a signal and a formation market cap.
    r = df_filtered.reset_index()
    sample_firms = sorted(set(r.loc[r['RRR_PCT'].notna(), 'FIRM'])
                          & set(r.loc[r['HISTORICAL_MARKET_CAP'].notna(), 'FIRM']))
    print(f'\n  Quarterly analysis sample: {len(sample_firms)} firms')

    panel, matched, missing = load_annual_revenue(sample_firms)
    panel = attach_sector_and_adjust(panel, df_filtered)
    panel = merge_quarterly_controls(panel, df_filtered)

    port, holding, quart_counts, n_form = build_annual_portfolios(panel, returns)

    print()
    print('=' * 78)
    print('STEP 3: FACTOR MODELS (Newey-West)')
    print('=' * 78)
    reg = av2.run_factor_regressions(port, ff, label='AnnualRRR')

    turnover, n_trans, n_skip = compute_annual_turnover(holding)
    sig_cmp = compare_signals(panel, df_filtered)

    n_months = len(port)
    n_firms = panel['FIRM'].nunique()
    reg_flat = {}
    for k, v in reg.items():
        if isinstance(v, dict):
            reg_flat[k] = {kk: vv for kk, vv in v.items() if np.isscalar(vv)}

    # ---- persist ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    xlsx = os.path.join(OUTPUT_DIR, 'annual_rrr_robustness.xlsx')
    with pd.ExcelWriter(xlsx, engine='openpyxl') as xw:
        port.to_excel(xw, sheet_name='Portfolio_Returns')
        pd.DataFrame(reg_flat).T.to_excel(xw, sheet_name='Factor_Regressions')
        turnover.to_excel(xw, sheet_name='Turnover')
        quart_counts.to_excel(xw, sheet_name='Firms_Per_Quartile')
        if not sig_cmp.empty:
            sig_cmp.to_excel(xw, sheet_name='Signal_Comparison', index=False)
        pd.DataFrame({'matched_firms': matched}).to_excel(
            xw, sheet_name='Sample_Firms', index=False)
        if missing:
            pd.DataFrame({'not_in_annual_file': missing}).to_excel(
                xw, sheet_name='Missing_Firms', index=False)
    print(f'\n  wrote {xlsx}')

    manifest = {
        'run_id': RUN_ID,
        'script': 'annual_rrr_robustness.py',
        'signal': 'ADJ annual RRR (sector x year demeaned)',
        'form_lag_months': ANNUAL_FORM_LAG_MONTHS,
        'hold_months': ANNUAL_HOLD_MONTHS,
        'reformations_per_year': ANNUAL_REFORMATIONS_PER_YEAR,
        'n_formation_events': int(n_form),
        'n_months': int(n_months),
        'n_firms': int(n_firms),
        'n_sample_firms_quarterly': len(sample_firms),
        'n_matched_in_annual_file': len(matched),
        'firms_not_in_annual_file': missing,
        # n_trans is a per-bucket dict, not a scalar.
        'turnover_transitions_used': {k: int(v) for k, v in n_trans.items()},
        'turnover_transitions_skipped': int(n_skip),
        'caveat': (f'Only {n_form} independent portfolio formations; the annual '
                   'revenue panel spans 2017-2025. Robustness check, not a '
                   'standalone test.'),
    }
    mpath = os.path.join(OUTPUT_DIR, 'annual_rrr_manifest.json')
    with open(mpath, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)
    print(f'  wrote {mpath}')

    print()
    print('=' * 78)
    print('  DONE')
    print('=' * 78)
    for spec in ['FF3', 'FF5']:
        k = f'Q1-Q4_{spec}'
        if k in reg_flat:
            r = reg_flat[k]
            print(f'  Q1-Q4 {spec}: alpha={r.get("alpha") * 100:.4f}%/mo, '
                  f't={r.get("alpha_t"):.4f}, p={r.get("alpha_p"):.4f}, '
                  f'N={int(r.get("n_obs"))}')


if __name__ == '__main__':
    main()
