"""sample_funnel.py -- firm counts at each sample-construction step.

Author request 2026-09-27 ("how many firms are out here?"): the Data section
reports how many firms each step removes. Every count below is computed from
the same loaders the analysis uses (analysis_v2.phase1_load_and_diagnose and
build_holding_panel), so the text cannot drift from the pipeline.

Scope. The steps upstream of the delivered revenue panel (the card vendor's
full coverage list and the r >= 0.75 revenue-correlation screen) were run in
Excel with Bloomberg BQL formulas whose values are not stored in the
repository, so they cannot be recounted here. The funnel therefore starts
from the screened revenue panel.

Outputs
    output/sample_funnel.xlsx   (sheets: Funnel, Dropped_by_sector)
"""

import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402


def main():
    ff = av2.phase1b_load_ff_factors()
    df_long, df_filtered, _ = av2.phase1_load_and_diagnose(ff)
    returns = av2.phase1c_load_monthly_returns()
    panel = av2.build_holding_panel(df_filtered, returns)

    long_r = df_long.reset_index()
    filt_r = df_filtered.reset_index()

    # The merged panel also carries firms that appear only in the fundamentals
    # file. The screened revenue panel is the set of firms with card revenue.
    has_rev = pd.to_numeric(long_r['#RETURNING_CUSTOMERS'], errors='coerce').notna()
    rev_firms = set(long_r.loc[has_rev, 'FIRM'])
    long_rev = long_r[long_r['FIRM'].isin(rev_firms)]

    n_screened = len(rev_firms)
    by_sector = long_rev.groupby('SECTOR', dropna=False)['FIRM'].nunique()
    in_sectors = long_rev.loc[long_rev['SECTOR'].isin(av2.VALID_SECTORS), 'FIRM'].nunique()
    # df_filtered already excludes av2.EXCLUDED_FIRMS (GameStop, 2026-10-20).
    kept = filt_r.loc[filt_r['FIRM'].isin(rev_firms), 'FIRM'].nunique()
    dropped = (by_sector[~by_sector.index.isin(av2.VALID_SECTORS)]
               .rename('firms_dropped').to_frame())
    n_no_sector = long_rev.loc[long_rev['SECTOR'].isna(), 'FIRM'].nunique()

    rrr = pd.to_numeric(filt_r['RRR_PCT'], errors='coerce')
    n_with_signal = filt_r.loc[rrr.notna(), 'FIRM'].nunique()
    n_formation = panel['FIRM'].nunique()

    funnel = pd.DataFrame([
        ('Firms in the screened revenue panel (passed r >= 0.75)', n_screened),
        ('Firms in the four GICS sectors with at least 10 firms', in_sectors),
        ('Firms after excluding GameStop (meme-stock trading in 2021)', kept),
        ('Firms with at least one computable RRR', n_with_signal),
        ('Firms entering portfolio formation', n_formation),
    ], columns=['Step', 'Firms'])
    funnel['Removed at this step'] = (funnel['Firms'].shift(1) - funnel['Firms']).fillna(0).astype(int)

    print('\nSAMPLE FUNNEL')
    print(funnel.to_string(index=False))
    print('\nFirms dropped by the sector filter, by sector:')
    print(dropped.to_string())
    if n_no_sector:
        print(f'  plus {n_no_sector} firm(s) with no sector assignment')

    out = os.path.join(av2.OUTPUT_DIR, 'sample_funnel.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        funnel.to_excel(xw, sheet_name='Funnel', index=False)
        dropped.to_excel(xw, sheet_name='Dropped_by_sector')
    print(f'\n  wrote {out}')


if __name__ == '__main__':
    main()
