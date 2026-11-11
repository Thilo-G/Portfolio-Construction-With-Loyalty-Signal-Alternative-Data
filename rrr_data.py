"""rrr_data.py -- RRR Financial Implications
==========================================

Loading the data and building the firm-quarter panel (Phase 1):

  phase1b_load_ff_factors       Fama-French factors and the risk-free rate
  phase1c_load_monthly_returns  monthly prices and returns (cached per run)
  load_fundamentals_wide        the quarterly Bloomberg sheet, one row per quarter
  phase1_load_and_diagnose      card revenue + Bloomberg fields -> the panel:
                                re-dates mis-stamped Bloomberg rows
                                (bloomberg_dating.py), builds RRR and its variants,
                                applies the sector filter and the firm exclusions
  backfill_*                    fill 17 firms from the previous data vintage

Moved verbatim from analysis_v2.py on 2026-10-20; other scripts still reach
these functions through analysis_v2 (av2.phase1_load_and_diagnose, ...).
"""

# =============================================================================
# Imports
# =============================================================================
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

import bloomberg_dating
from config import (
    ALIGN_BLOOMBERG_DATES,
    DATA_WORKBOOK,
    EXCLUDED_FIRMS,
    FF3_URL,
    FF5_URL,
    FF_LOCAL_CSV,
    FUND_DATE_COL,
    LEGACY_FUNDAMENTALS_FILE,
    LEGACY_INDUSTRY_FILE,
    LEGACY_RETURNS_FILE,
    LEGACY_REVENUE_FILE,
    MCAP_BACKFILL_FIRMS,
    MOM_LOCAL_CSV,
    MOM_URL,
    OUTPUT_DIR,
    PRICE_BACKFILL_FIRMS,
    REVENUE_BACKFILL_FIRMS,
    SHEET_FUNDAMENTALS,
    SHEET_INDUSTRY,
    SHEET_RETURNS,
    SHEET_REVENUE,
    VALID_SECTORS,
)
from rrr_portfolios import safe_quartile


# =============================================================================
# Functions
# =============================================================================
def _load_legacy_wide(path, ticker_filter=None):
    """Load a legacy 2-header-row wide workbook: row 0 = ticker, row 1 = field.
    This is the layout every input file used before the 2026-09-08 consolidation
    into DATA_WORKBOOK. Returns a DataFrame indexed by Date with columns
    '<TICKER>.<field>', tickers uppercased/stripped for matching.
    """
    raw = pd.read_excel(path, header=None)
    header_rows = raw.iloc[:2]
    data_rows = raw.iloc[2:].copy()
    combined = header_rows.apply(lambda x: x.astype(str).str.strip(), axis=0)
    cols = combined.apply(lambda x: '.'.join(v for v in x if v and v.lower() != 'nan'), axis=0)
    data_rows.columns = cols
    data_rows = data_rows.rename(columns={data_rows.columns[0]: 'Date'})
    data_rows['Date'] = pd.to_datetime(data_rows['Date'], errors='coerce')
    data_rows = data_rows[data_rows['Date'].notna()].set_index('Date')
    renamed = {}
    for c in data_rows.columns:
        if '.' in c:
            tick, field = c.split('.', 1)
            renamed[c] = f"{tick.upper().strip()}.{field}"
    data_rows = data_rows.rename(columns=renamed)
    if ticker_filter:
        keep = [c for c in data_rows.columns if c.split('.', 1)[0] in ticker_filter]
        data_rows = data_rows[keep]
    return data_rows


def backfill_revenue(df_revenue):
    """Add #New_Customers / #Returning_Customers for the 11 Group-A firms,
    entirely absent from the new revenue sheet, from the last vintage that had
    them (through 2024-Q4 only -- see LEGACY_REVENUE_FILE above).
    """
    legacy = _load_legacy_wide(LEGACY_REVENUE_FILE, REVENUE_BACKFILL_FIRMS)
    legacy = legacy[[c for c in legacy.columns
                     if c.split('.', 1)[1] in ('#New_Customers', '#Returning_Customers')]]
    for c in legacy.columns:
        legacy[c] = pd.to_numeric(legacy[c], errors='coerce')
    dates = pd.to_datetime(df_revenue['Date'])
    added = 0
    for col in legacy.columns:
        if col in df_revenue.columns:
            continue
        df_revenue[col] = legacy[col].reindex(dates).values
        added += 1
    n_firms = len({c.split('.', 1)[0] for c in legacy.columns})
    print(f"  Backfilled revenue for {n_firms} firms ({added} columns) from "
          f"{os.path.basename(LEGACY_REVENUE_FILE)}, through {legacy.index.max().date()}")
    return df_revenue


def backfill_market_cap(df_fundamentals):
    """Fill HISTORICAL_MARKET_CAP for the 6 Group-B firms, whose column
    exists but is entirely empty in the new fundamentals sheet.
    """
    legacy = _load_legacy_wide(LEGACY_FUNDAMENTALS_FILE, MCAP_BACKFILL_FIRMS)
    legacy = legacy[[c for c in legacy.columns if c.split('.', 1)[1] == 'HISTORICAL_MARKET_CAP']]
    date_col = 'nan.Dates' if 'nan.Dates' in df_fundamentals.columns else df_fundamentals.columns[0]
    dates = pd.to_datetime(df_fundamentals[date_col])
    filled = 0
    for firm in MCAP_BACKFILL_FIRMS:
        target_cols = [c for c in df_fundamentals.columns
                       if c.split('.', 1)[0].upper().strip() == firm
                       and c.rsplit('.', 1)[1] == 'HISTORICAL_MARKET_CAP']
        legacy_col = f"{firm}.HISTORICAL_MARKET_CAP"
        if not target_cols or legacy_col not in legacy.columns:
            continue
        vals = pd.to_numeric(legacy[legacy_col], errors='coerce').reindex(dates).values
        df_fundamentals[target_cols[0]] = vals
        filled += int(pd.notna(vals).sum())
    print(f"  Backfilled market cap for {len(MCAP_BACKFILL_FIRMS)} firms "
          f"({filled} firm-quarter values) from {os.path.basename(LEGACY_FUNDAMENTALS_FILE)}")
    return df_fundamentals


def backfill_monthly_prices(df_prices):
    """Add PX_LAST for all 17 backfill firms, entirely absent from the new
    monthly-returns sheet, matched by CALENDAR MONTH (not exact date) against
    the legacy file -- the two vintages' month-end trading days need not
    coincide exactly, same reasoning as the Fama-French join fix above.
    """
    legacy = _load_legacy_wide(LEGACY_RETURNS_FILE, PRICE_BACKFILL_FIRMS)
    legacy = legacy[[c for c in legacy.columns if c.rsplit('.', 1)[1] == 'PX_LAST']]
    for c in legacy.columns:
        legacy[c] = pd.to_numeric(legacy[c], errors='coerce')
    legacy_by_month = legacy.copy()
    legacy_by_month.index = legacy_by_month.index.to_period('M')

    months = pd.to_datetime(df_prices['Date']).dt.to_period('M')
    added, filled = 0, 0
    for col in legacy_by_month.columns:
        if col in df_prices.columns:
            continue  # should not happen -- these firms have no PX_LAST at all
        vals = legacy_by_month[col].reindex(months).values
        df_prices[col] = vals
        added += 1
        filled += int(pd.notna(vals).sum())
    n_firms = len({c.split('.', 1)[0] for c in legacy_by_month.columns})
    print(f"  Backfilled monthly PX_LAST for {n_firms} firms ({added} columns, "
          f"{filled} firm-months) from {os.path.basename(LEGACY_RETURNS_FILE)}")
    return df_prices


def download_ff_factors():
    """Download FF3, FF5, and Momentum factors from Ken French website."""

    factors = {}

    for name, url in [('FF3', FF3_URL), ('FF5', FF5_URL), ('MOM', MOM_URL)]:
        try:
            print(f"  Downloading {name} factors...")
            local_zip = os.path.join(OUTPUT_DIR, f'{name}.zip')
            urllib.request.urlretrieve(url, local_zip)

            with zipfile.ZipFile(local_zip, 'r') as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    content = f.read().decode('utf-8')

            # Parse: find the monthly data section (before annual)
            lines = content.split('\n')
            data_lines = []
            header_found = False
            for line in lines:
                line = line.strip()
                if not line:
                    if header_found:
                        break
                    continue
                # Skip description lines
                if line[0].isdigit() and len(line.split(',')[0].strip()) == 6:
                    data_lines.append(line)
                    header_found = True
                elif header_found and line[0].isdigit():
                    if len(line.split(',')[0].strip()) == 6:
                        data_lines.append(line)
                    else:
                        break  # Hit annual data

            if name == 'MOM':
                header = 'Date,Mom'
            elif name == 'FF3':
                header = 'Date,Mkt-RF,SMB,HML,RF'
            else:
                header = 'Date,Mkt-RF,SMB,HML,RMW,CMA,RF'

            csv_text = header + '\n' + '\n'.join(data_lines)
            df = pd.read_csv(io.StringIO(csv_text))
            df['Date'] = pd.to_datetime(df['Date'].astype(str).str.strip(), format='%Y%m')
            df = df.set_index('Date').sort_index()
            # Convert from percent to decimal
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100
            factors[name] = df

            os.remove(local_zip)
            print(f"  {name}: {len(df)} months loaded")
        except Exception as e:
            print(f"  WARNING: Failed to download {name}: {e}")

    return factors


def parse_ff_local_csv(path, header_token='Mkt-RF'):
    """Parse a Ken French monthly factor CSV robustly.

    The 2026-09-08 vintage differs from the 2025 one in three ways that broke
    the old `skiprows=3` call: the preamble is 4 lines, the file carries FF5
    (Mkt-RF, SMB, HML, RMW, CMA, RF) rather than FF3, and a trailing
    "Annual Factors" block plus a copyright line follow the monthly rows.
    Rather than hardcode offsets again, locate the header line and stop at the
    first blank / non-YYYYMM line after it. header_token selects the header
    row: 'Mkt-RF' for the factor file, 'Mom' for the momentum file.
    """
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()

    hdr = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(',') and header_token in line:
            hdr = i
            break
    if hdr is None:
        raise ValueError(f"No Fama-French header row (',{header_token},...') found in {path}")

    end = len(lines)
    for i in range(hdr + 1, len(lines)):
        token = lines[i].split(',')[0].strip()
        if not token.isdigit() or len(token) != 6:
            end = i
            break

    block = chr(10).join(lines[hdr:end])
    ff = pd.read_csv(io.StringIO(block))
    ff = ff.rename(columns={ff.columns[0]: 'Date'})
    ff.columns = [str(c).strip() for c in ff.columns]
    ff['Date'] = pd.to_datetime(ff['Date'].astype(str).str.strip(), format='%Y%m')
    ff = ff.set_index('Date').sort_index()
    for col in ff.columns:
        ff[col] = pd.to_numeric(ff[col], errors='coerce') / 100.0
    ff.index = ff.index.to_period('M').to_timestamp('M')
    return ff


def phase1b_load_ff_factors():
    """Load FF factors from local files; momentum from its own pinned file.

    Changed 2026-09-08. The local file is now the primary source because it is
    the vintage the author actually pulled and is what the replication package
    ships. Changed 2026-09-28: momentum is read from MOM_LOCAL_CSV; the live
    download is used only if that file is missing, with a loud warning because
    the Carhart numbers then depend on the library's current release. If
    momentum cannot be obtained at all, the Carhart (FF3+Mom) specification is
    dropped LOUDLY rather than silently.
    """

    print()
    print("=" * 80)
    print("PHASE 1B: FAMA-FRENCH FACTOR DATA")
    print("=" * 80)

    if not os.path.exists(FF_LOCAL_CSV):
        raise FileNotFoundError(f"Local Fama-French file not found: {FF_LOCAL_CSV}")

    ff_all = parse_ff_local_csv(FF_LOCAL_CSV)
    print(f"  Local FF file: {os.path.basename(FF_LOCAL_CSV)}")
    print(f"    {len(ff_all)} months, {ff_all.index.min():%Y-%m} to {ff_all.index.max():%Y-%m}")
    print(f"    columns: {list(ff_all.columns)}")

    required = ['Mkt-RF', 'SMB', 'HML', 'RF']
    missing = [c for c in required if c not in ff_all.columns]
    if missing:
        raise ValueError(f"Local FF file is missing required columns: {missing}")

    # Momentum: pinned local file first; live download only as a fallback.
    if 'Mom' not in ff_all.columns and os.path.exists(MOM_LOCAL_CSV):
        mom = parse_ff_local_csv(MOM_LOCAL_CSV, header_token='Mom')[['Mom']]
        ff_all = ff_all.join(mom, how='left')
        n_mom = int(ff_all['Mom'].notna().sum())
        print(f"  Momentum file: {os.path.basename(MOM_LOCAL_CSV)} ({n_mom} months merged)")
    if 'Mom' not in ff_all.columns:
        print(f"  !! Pinned momentum file not found: {MOM_LOCAL_CSV}")
        print("  !! Falling back to the LIVE Ken French download; Carhart numbers may")
        print("  !! differ from the paper's if the library has released new data.")
        try:
            factors = download_ff_factors()
        except Exception as exc:
            factors = {}
            print(f"  !! Momentum download raised: {type(exc).__name__}: {exc}")
        if 'MOM' in factors:
            mom = factors['MOM'].copy()
            mom.index = mom.index.to_period('M').to_timestamp('M')
            ff_all = ff_all.join(mom, how='left')
            n_mom = int(ff_all['Mom'].notna().sum())
            print(f"    Momentum merged: {n_mom} non-missing months")
        else:
            print("  !! WARNING: momentum factor unavailable.")
            print("  !! The Carhart (FF3+Mom) specification will be SKIPPED in all")
            print("  !! factor regressions. This is a visible degradation, not a silent one.")

    print(f"  Final factor set: {len(ff_all)} months, cols={list(ff_all.columns)}")
    return ff_all


def load_fundamentals_wide():
    """Read the quarterly Bloomberg fundamentals sheet into one wide frame.

    One row per quarter; columns '<ticker>.<MNEMONIC>' plus the date column
    FUND_DATE_COL. The 6 Group-B firms get their market cap from the legacy file
    (backfill_market_cap), and R&D and SG&A gaps are set to 0. No re-dating
    happens here: phase1_load_and_diagnose() calls bloomberg_dating afterwards.
    """
    # 2026-09-08: the fundamentals sheet gained a THIRD header row. Layout is
    # now row 0 = ticker, row 1 = human-readable label ("Revenue"), row 2 =
    # Bloomberg mnemonic ("SALES_REV_TURN"). The old code joined rows 0 and 1,
    # which would name every column "<TICKER>.Revenue" and silently collapse
    # all 13 fields into one. Join rows 0 and 2 instead; data starts at row 3.
    df_fundamentals = pd.read_excel(DATA_WORKBOOK, sheet_name=SHEET_FUNDAMENTALS, header=None)

    header_rows_fun = df_fundamentals.iloc[[0, 2]]
    data_rows_fun = df_fundamentals.iloc[3:]
    combined_headers = header_rows_fun.apply(lambda x: x.astype(str).str.strip(), axis=0)
    column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
    data_rows_fun.columns = column_headers
    df_fundamentals = data_rows_fun.reset_index(drop=True)
    df_fundamentals.replace('#N/A N/A', np.nan, inplace=True)
    df_fundamentals = df_fundamentals.loc[:, ~df_fundamentals.columns.duplicated()]
    df_fundamentals = backfill_market_cap(df_fundamentals)

    # Fill R&D and SGA NaNs with 0
    cols_fill = [col for col in df_fundamentals.columns if "IS_RD_EXPEND" in col or "IS_SGA_EXPENSE" in col]
    df_fundamentals[cols_fill] = df_fundamentals[cols_fill].fillna(0)
    return df_fundamentals


def phase1_load_and_diagnose(ff_factors=None):
    """Load all data, apply sector filter, compute industry diagnostics.

    ff_factors: monthly Fama-French frame (from phase1b_load_ff_factors()).
    Required since 2026-09-08: the risk-free rate is taken from the FF RF
    column, because USBMMY3M INDEX no longer lives in the fundamentals sheet.
    """

    print("=" * 80)
    print("PHASE 1: DATA LOADING & INDUSTRY DIAGNOSTICS")
    print("=" * 80)

    # --- 1.1 Load revenue data ---
    print("\n--- 1.1 Loading revenue data ---")
    # 2026-09-08: revenue is now a SHEET in the consolidated workbook.
    # Header layout is unchanged (row 0 = ticker, row 1 = field), but the sheet
    # carries a trailing stub row (a data-pull timestamp with no values) that
    # must be dropped before any .shift(1) is taken.
    df_revenue = pd.read_excel(DATA_WORKBOOK, sheet_name=SHEET_REVENUE, header=None)

    header_rows = df_revenue.iloc[:2]
    data_rows = df_revenue.iloc[2:]
    # Drop rows whose data columns are entirely empty (the pull-date stub).
    _payload = data_rows.iloc[:, 1:]
    _keep = _payload.notna().any(axis=1)
    if (~_keep).any():
        _dropped = data_rows.loc[~_keep].iloc[:, 0].tolist()
        print(f"  Dropping {int((~_keep).sum())} empty revenue row(s): {_dropped}")
    data_rows = data_rows.loc[_keep]
    combined_headers = header_rows.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    column_headers = combined_headers.apply(lambda x: '.'.join(x.dropna()), axis=0)
    data_rows.columns = column_headers
    data_rows.rename(columns={data_rows.columns[0]: 'Date'}, inplace=True)
    df_revenue = data_rows.reset_index(drop=True)
    df_revenue = df_revenue.replace(r'^\s*$', pd.NA, regex=True)
    df_revenue = df_revenue.loc[:, ~df_revenue.columns.duplicated()]
    df_revenue = backfill_revenue(df_revenue)

    # Compute metrics per firm
    new_customer_cols = [col for col in df_revenue.columns if '#New_Customers' in col]
    returning_customer_cols = [col for col in df_revenue.columns if '#Returning_Customers' in col]

    for new_col, ret_col in zip(new_customer_cols, returning_customer_cols):
        total_col = new_col.replace('#New_Customers', '#Total_Revenue')
        rrr_col = ret_col.replace('#Returning_Customers', '#RRR')
        rg_col = ret_col.replace('#Returning_Customers', '#Revenue_Growth')
        ar_col = new_col.replace('#New_Customers', '#Acq_Rate')
        share_col = ret_col.replace('#Returning_Customers', '#Share_Ret_Revenue')

        df_revenue[total_col] = df_revenue[new_col] + df_revenue[ret_col]
        df_revenue[share_col] = df_revenue[ret_col] / df_revenue[total_col].replace(0, np.nan)
        df_revenue[rrr_col] = df_revenue[ret_col] / df_revenue[total_col].shift(1).replace(0, np.nan)
        df_revenue[rg_col] = (df_revenue[total_col] - df_revenue[total_col].shift(1)) / df_revenue[total_col].shift(1)
        df_revenue[ar_col] = df_revenue[new_col] / df_revenue[new_col].shift(1)

    # The revenue sheet defines the sample candidates: a firm without
    # credit-card revenue has no RRR and cannot enter any portfolio, whatever
    # else the workbook contains. Keep this list for the GICS guard below.
    revenue_firms = sorted({
        c.rsplit('.', 1)[0].upper().strip()
        for c in df_revenue.columns
        if '#New_Customers' in c or '#Returning_Customers' in c
    })
    print(f"  Revenue data loaded: {df_revenue.shape}; {len(revenue_firms)} firms with revenue")

    # --- 1.2 Load fundamentals ---
    print("\n--- 1.2 Loading fundamentals ---")
    df_fundamentals = load_fundamentals_wide()
    print(f"  Fundamentals loaded: {df_fundamentals.shape}")

    # --- 1.2b Bloomberg quarter stamping (added 2026-10-20) ---
    # Detect, on the data just read, which firms' Bloomberg rows are stamped off
    # calendar time, compare with the stored offsets, and re-date them. Stops the
    # run if the offsets disagree with the stored file (see bloomberg_dating.py).
    print("\n--- 1.2b Checking Bloomberg quarter stamping ---")
    bbg_offsets_applied = {}
    if ALIGN_BLOOMBERG_DATES:
        df_fundamentals, bbg_offsets_applied = bloomberg_dating.check_and_apply(
            df_fundamentals, phase1c_load_monthly_returns(), FUND_DATE_COL)
    else:
        print("  ALIGN_BLOOMBERG_DATES is False: Bloomberg fields used as stamped (pre-fix numbers).")

    # --- 1.3 Merge to long format ---
    print("\n--- 1.3 Merging to long panel ---")
    excluded_columns = ['Date']
    df_revenue_to_append = df_revenue.drop(columns=excluded_columns, errors='ignore')
    df_fundamentals = df_fundamentals.rename(columns={'nan.Dates': 'Date'})
    df_combined = pd.concat([df_fundamentals, df_revenue_to_append], axis=1)
    df_combined.columns = df_combined.columns.str.upper()
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

    all_vars = [col for col in df_combined.columns if col != "DATE"]
    df_long = df_combined.melt(id_vars=["DATE"], value_vars=all_vars, var_name="Firm_Variable", value_name="Value")
    df_long[['FIRM', 'VARIABLE']] = df_long['Firm_Variable'].str.rsplit('.', n=1, expand=True)
    df_long = df_long.pivot_table(index=['FIRM', 'DATE'], columns='VARIABLE', values='Value').reset_index()
    df_long = df_long.set_index(['FIRM', 'DATE'])
    # Quarters by which each firm's Bloomberg rows were re-dated in step 1.2b
    # (0 = untouched). build_holding_panel() uses it to carry a re-dated firm's
    # market cap into the final quarters that the shift left empty.
    df_long['BBG_OFFSET_K'] = [bbg_offsets_applied.get(f, 0)
                               for f in df_long.index.get_level_values('FIRM')]

    # Remove SPX INDEX, SPW INDEX, USBMMY3M INDEX from firm list
    index_firms = ['SPX INDEX', 'SPW INDEX', 'USBMMY3M INDEX']
    all_firms = df_long.index.get_level_values('FIRM').unique()
    actual_firms = [f for f in all_firms if f not in index_firms]

    print(f"  Panel: {len(actual_firms)} firms, {df_long.index.get_level_values('DATE').nunique()} periods")

    # --- 1.4 Compute derived variables ---
    print("\n--- 1.4 Computing derived variables ---")

    df_long['RRR_PCT'] = pd.to_numeric(df_long.get('#RRR', np.nan), errors='coerce') * 100
    df_long['ACQ_RATE_PCT'] = pd.to_numeric(df_long.get('#ACQ_RATE', np.nan), errors='coerce') * 100
    df_long['REV_GROWTH_PCT'] = pd.to_numeric(df_long.get('#REVENUE_GROWTH', np.nan), errors='coerce') * 100
    # Share of new revenue: Rev_new / Rev_t = 1 - RRR / (1 + RG); undefined when RG = -100%
    df_long['SHARE_NEW_REV_PCT'] = np.where(
        df_long['REV_GROWTH_PCT'] == -100,
        np.nan,
        (1 - (df_long['RRR_PCT'] / 100) / (1 + df_long['REV_GROWTH_PCT'] / 100)) * 100
    )

    # Lagged values
    df_long['RRR_PCT_LAG1'] = df_long.groupby(level='FIRM')['RRR_PCT'].shift(1)
    df_long['RRR_LAG'] = df_long['RRR_PCT_LAG1'] / 100
    df_long['ACQ_RATE_PCT_LAG1'] = df_long.groupby(level='FIRM')['ACQ_RATE_PCT'].shift(1)

    # Size, BTM
    df_long['HISTORICAL_MARKET_CAP'] = pd.to_numeric(df_long.get('HISTORICAL_MARKET_CAP', np.nan), errors='coerce')
    df_long['PX_LAST'] = pd.to_numeric(df_long.get('PX_LAST', np.nan), errors='coerce')
    df_long['SIZE'] = np.log(df_long['HISTORICAL_MARKET_CAP'].replace(0, np.nan))
    df_long['BTM'] = (pd.to_numeric(df_long.get('BS_TOT_ASSET', np.nan), errors='coerce') -
                       pd.to_numeric(df_long.get('BS_TOT_LIAB2', np.nan), errors='coerce')) / df_long['HISTORICAL_MARKET_CAP'].replace(0, np.nan)
    df_long['PM_OPER_PCT'] = (pd.to_numeric(df_long.get('IS_OPER_INC', np.nan), errors='coerce') /
                               pd.to_numeric(df_long.get('SALES_REV_TURN', np.nan), errors='coerce').replace(0, np.nan)) * 100
    # Market-to-Equity (direct from Bloomberg TOTAL_EQUITY)
    df_long['MTE'] = (df_long['HISTORICAL_MARKET_CAP'] /
                      pd.to_numeric(df_long.get('TOTAL_EQUITY', np.nan), errors='coerce').replace(0, np.nan))
    # Operating ROA proxy (no net income in fundamentals file)
    df_long['ROA_OPER'] = (pd.to_numeric(df_long.get('IS_OPER_INC', np.nan), errors='coerce') /
                           pd.to_numeric(df_long.get('BS_TOT_ASSET', np.nan), errors='coerce').replace(0, np.nan)) * 100

    # Stock returns (price-only, log)
    df_long['PX_LAST_LAG1'] = df_long.groupby(level='FIRM')['PX_LAST'].shift(1)
    df_long['RETURN_LOG'] = np.log(df_long['PX_LAST'] / df_long['PX_LAST_LAG1'])
    df_long['RET_ARITH'] = np.exp(df_long['RETURN_LOG']) - 1

    # Risk-free rate.
    # 2026-09-08: sourced from the Fama-French RF column instead of the
    # USBMMY3M INDEX row, which no longer exists in the fundamentals sheet.
    # The old code swallowed that KeyError and set RF = 0, which would have
    # silently inflated every Sharpe and Sortino ratio in the paper. There is
    # no fallback now: a missing RF is a hard failure.
    #
    # FF RF is a MONTHLY simple rate. The panel here is quarterly, so compound
    # the three monthly rates inside each quarter rather than annualising.
    if ff_factors is None or 'RF' not in getattr(ff_factors, 'columns', []):
        raise ValueError(
            "phase1_load_and_diagnose() requires ff_factors with an 'RF' column. "
            "Call phase1b_load_ff_factors() first and pass the result in.")

    _rf_m = pd.to_numeric(ff_factors['RF'], errors='coerce').dropna()
    _rf_q = (1 + _rf_m).groupby(_rf_m.index.to_period('Q')).prod() - 1
    _dates = pd.to_datetime(df_long.index.get_level_values('DATE'))
    df_long['RF'] = _rf_q.reindex(_dates.to_period('Q')).values

    _n_missing = int(df_long['RF'].isna().sum())
    if _n_missing:
        _bad = sorted(set(_dates[df_long['RF'].isna()].date))
        print(f"  RF missing for {_n_missing} obs across quarters: {_bad}")
    _rf_ok = df_long['RF'].dropna()
    if _rf_ok.empty or not (_rf_ok >= 0).all():
        raise ValueError("Risk-free rate from Fama-French is empty or negative; refusing to continue.")
    print(f"  RF from Fama-French: {_rf_q.index.min()} to {_rf_q.index.max()}, "
          f"quarterly mean {_rf_ok.mean() * 100:.4f}%")

    df_long['EXCESS_RET'] = df_long['RET_ARITH'] - df_long['RF']

    # --- 1.5 Load industry classifications ---
    print("\n--- 1.5 Loading industry classifications ---")
    # 2026-09-08: GICS mapping now ships as a sheet in the data workbook.
    df_industry = pd.read_excel(DATA_WORKBOOK, sheet_name=SHEET_INDUSTRY)
    df_industry.columns = ['ID', 'GICS_INDUSTRY', 'GICS_SECTOR', 'GICS_SUB_INDUSTRY']
    df_industry['FIRM'] = df_industry['ID'].astype(str).str.upper().str.strip()

    # Map firms to sectors
    firm_sector = df_industry.set_index('FIRM')['GICS_SECTOR'].to_dict()

    # Backfill GICS sector for the 11 Group-A firms (see REVENUE_BACKFILL_FIRMS
    # above), absent from the new Industry_Values sheet because they are absent
    # from the new revenue sheet's ticker list that sheet was built from.
    _legacy_ind = pd.read_excel(LEGACY_INDUSTRY_FILE)
    _legacy_ind.columns = ['ID', 'GICS_INDUSTRY', 'GICS_SECTOR', 'GICS_SUB_INDUSTRY']
    _legacy_ind['ID'] = _legacy_ind['ID'].astype(str).str.upper().str.strip()
    _legacy_sector = _legacy_ind.set_index('ID')['GICS_SECTOR'].to_dict()
    _n_sector_backfilled = 0
    for _f in REVENUE_BACKFILL_FIRMS:
        if _f not in firm_sector and _f in _legacy_sector:
            firm_sector[_f] = _legacy_sector[_f]
            _n_sector_backfilled += 1
    if _n_sector_backfilled:
        print(f"  Backfilled GICS sector for {_n_sector_backfilled} firms from "
              f"{os.path.basename(LEGACY_INDUSTRY_FILE)}")

    # Add sector to df_long
    firms_in_panel = df_long.index.get_level_values('FIRM')
    df_long['SECTOR'] = firms_in_panel.map(firm_sector)

    # GUARD: the Industry_Values sheet contains hand-added rows (ASO US Equity,
    # ATZ CN Equity) that exist in no Bloomberg pull. If a future re-pull
    # overwrites the sheet, those firms would silently drop out of the sample
    # via a NaN sector. Fail loudly instead of losing them quietly.
    # Only revenue-bearing firms matter here. The fundamentals sheet carries a
    # further 16 tickers with no credit-card revenue; those have no RRR and are
    # dropped downstream regardless of their GICS status.
    _unmapped = sorted({f for f in revenue_firms if f not in firm_sector})
    if _unmapped:
        raise ValueError(
            f"{len(_unmapped)} revenue-bearing firm(s) have no GICS mapping in "
            f"'{SHEET_INDUSTRY}': {_unmapped}. Manual rows (ASO US Equity, "
            f"ATZ CN Equity) may have been overwritten by a Bloomberg re-pull; "
            f"restore from the .manual-edits-backup copy.")
    print(f"  GICS mapping complete: all {len(revenue_firms)} revenue firms matched")

    # --- 1.6 Sector filter ---
    print("\n--- 1.6 Applying sector filter ---")
    sector_counts = df_long.reset_index().groupby('SECTOR')['FIRM'].nunique()
    print("  Firm counts by sector (before filter):")
    print(sector_counts.sort_values(ascending=False).to_string())

    # Keep only valid sectors
    mask = df_long['SECTOR'].isin(VALID_SECTORS)
    df_filtered = df_long[mask].copy()

    n_firms_filtered = df_filtered.reset_index()['FIRM'].nunique()
    print(f"\n  After sector filter: {n_firms_filtered} firms")

    # Firm exclusions decided by the author (EXCLUDED_FIRMS at the top of the file).
    _firm_level = df_filtered.index.get_level_values('FIRM')
    for _firm, _reason in EXCLUDED_FIRMS.items():
        print(f"  Excluding {_firm}: {_reason} "
              f"({int((_firm_level == _firm).sum())} firm-quarters)")
    df_filtered = df_filtered[~_firm_level.isin(list(EXCLUDED_FIRMS))].copy()
    n_firms_filtered = df_filtered.reset_index()['FIRM'].nunique()
    print(f"  After firm exclusions: {n_firms_filtered} firms")

    sector_counts_filtered = df_filtered.reset_index().groupby('SECTOR')['FIRM'].nunique()
    print("  Firm counts by sector (after filter):")
    print(sector_counts_filtered.sort_values(ascending=False).to_string())

    # Firms per quarter
    firms_per_q = df_filtered.reset_index().groupby('DATE')['FIRM'].nunique()
    print(f"\n  Firms per quarter: min={firms_per_q.min()}, max={firms_per_q.max()}, mean={firms_per_q.mean():.0f}")

    # --- 1.7 Industry x time adjustment ---
    print("\n--- 1.7 Computing industry x time adjusted signals ---")

    # RRR adjustment
    sector_quarter_mean_rrr = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])['RRR_PCT'].transform('mean')
    df_filtered['ADJ_RRR_PCT'] = df_filtered['RRR_PCT'] - sector_quarter_mean_rrr

    # AR adjustment
    sector_quarter_mean_ar = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])['ACQ_RATE_PCT'].transform('mean')
    df_filtered['ADJ_ACQ_RATE_PCT'] = df_filtered['ACQ_RATE_PCT'] - sector_quarter_mean_ar

    # Lagged adjusted signals
    df_filtered['ADJ_RRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_RRR_PCT'].shift(1)
    df_filtered['ADJ_RRR_LAG'] = df_filtered['ADJ_RRR_PCT_LAG1'] / 100
    df_filtered['ADJ_ACQ_RATE_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_ACQ_RATE_PCT'].shift(1)

    print("  Adjusted signals computed: ADJ_RRR_PCT, ADJ_ACQ_RATE_PCT")

    # --- 1.4b Alternative RRR variants ---
    print("\n--- 1.4b Alternative RRR variants ---")

    _ret = pd.to_numeric(df_filtered['#RETURNING_CUSTOMERS'], errors='coerce')
    _new = pd.to_numeric(df_filtered['#NEW_CUSTOMERS'], errors='coerce')
    _tot = pd.to_numeric(df_filtered['#TOTAL_REVENUE'], errors='coerce')

    # Variant A: 1-year-lag RRR, Rev_ret_t / total_revenue_{t-4}
    df_filtered['_TOT'] = _tot
    _tot_lag4 = df_filtered.groupby(level='FIRM')['_TOT'].shift(4).replace(0, np.nan)
    df_filtered.drop(columns=['_TOT'], inplace=True)
    df_filtered['LR_RRR_PCT'] = (_ret / _tot_lag4) * 100
    _sm = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])[
        'LR_RRR_PCT'].transform('mean')
    df_filtered['ADJ_LR_RRR_PCT'] = df_filtered['LR_RRR_PCT'] - _sm

    # Variant A2: 2-quarter-lag RRR - Rev_ret_t / total_revenue_{t-2}
    # Sits between the baseline (t-1 denominator) and LR_RRR (t-4). Isolates
    # where the signal's horizon breaks down: the baseline is significant and
    # LR_RRR collapses, so the two-quarter variant locates the decay.
    df_filtered['_TOT'] = _tot
    _tot_lag2 = df_filtered.groupby(level='FIRM')['_TOT'].shift(2).replace(0, np.nan)
    df_filtered.drop(columns=['_TOT'], inplace=True)
    df_filtered['TQ_RRR_PCT'] = (_ret / _tot_lag2) * 100
    _sm = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])[
        'TQ_RRR_PCT'].transform('mean')
    df_filtered['ADJ_TQ_RRR_PCT'] = df_filtered['TQ_RRR_PCT'] - _sm

    # Variant A3: trailing-average-denominator RRR (TA4)
    # Rev_ret_t / mean(total_revenue_{t-1..t-4}). The baseline signal divides by
    # ONE prior quarter's total revenue, and that same quarter's returning-customer
    # revenue is also the baseline signal's OWN numerator one period earlier
    # (RRR_{t-1} = Rev_ret_{t-1} / Rev_total_{t-2}), so a single unusually strong
    # or weak quarter for returning-customer revenue mechanically pushes RRR_{t-1}
    # up and RRR_t down, one quarter apart, independent of any real change in
    # customer behavior. This is the leading candidate explanation for the
    # signal's negative lag-1 autocorrelation (see tab_signal_persistence.tex),
    # which the industry-time adjustment cannot remove because the effect is
    # firm-specific, not sector-common (confirmed: adjusted lag-1 autocorrelation
    # is MORE negative than raw, -0.2200 vs -0.2090 -- the adjustment isn't
    # touching it). Averaging the denominator over 4 quarters means the shared
    # term carries roughly 1/4 of the denominator's weight instead of all of it,
    # attenuating the mechanical link about fourfold, while keeping the numerator
    # fully contemporaneous (unlike TQ_RRR/LR_RRR, which lag the NUMERATOR too).
    df_filtered['_TOT'] = _tot
    _tot_ta4 = (df_filtered.groupby(level='FIRM')['_TOT']
                .transform(lambda s: s.shift(1).rolling(4, min_periods=4).mean())
                .replace(0, np.nan))
    df_filtered.drop(columns=['_TOT'], inplace=True)
    df_filtered['TA4_RRR_PCT'] = (_ret / _tot_ta4) * 100
    _sm = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])[
        'TA4_RRR_PCT'].transform('mean')
    df_filtered['ADJ_TA4_RRR_PCT'] = df_filtered['TA4_RRR_PCT'] - _sm

    # Variant A4: rolling-customer-base RRR (RCB4), author-proposed 2026-09-16
    # Rev_ret_t / (Rev_total_{t-4} + Rev_new_{t-1} + Rev_new_{t-2} + Rev_new_{t-3}).
    # Expanding Rev_total_{t-4} = Rev_ret_{t-4} + Rev_new_{t-4}, the denominator is
    # Rev_ret_{t-4} + Rev_new_{t-4} + Rev_new_{t-3} + Rev_new_{t-2} + Rev_new_{t-1}:
    # a rolling reconstruction of the customer-revenue base, anchored on the total
    # base four quarters back and rolled forward using only the NEW-customer
    # inflows realized since. Structurally different from TA4 in the one respect
    # that matters most for the mechanical-autocorrelation finding: it contains
    # NO Rev_ret_{t-1} term at all (TA4 dilutes that shared term to 1/4 weight;
    # this removes it entirely). What it does introduce, by design, is a
    # different serial dependence: D_t and D_{t-1} share Rev_new_{t-2} and
    # Rev_new_{t-3} (a rolling-window overlap), which is expected and is not the
    # numerator/denominator artifact -- reported alongside the result, not
    # assumed away.
    #
    # RAW ONLY: no industry-time adjustment, and no ADJ_RCB4_RRR_PCT column.
    # TA4's headline comparison stacked a denominator change together with the
    # same sector-quarter demeaning the baseline gets, which confounds which of
    # the two caused TA4's alpha loss. This variant, and the TA4 rerun beside
    # it, are compared raw-vs-raw against RRR_PCT so denominator construction is
    # the only thing that differs.
    df_filtered['_NEW'] = _new
    df_filtered['_TOT'] = _tot
    _new_lag1 = df_filtered.groupby(level='FIRM')['_NEW'].shift(1)
    _new_lag2 = df_filtered.groupby(level='FIRM')['_NEW'].shift(2)
    _new_lag3 = df_filtered.groupby(level='FIRM')['_NEW'].shift(3)
    _tot_lag4_rcb = df_filtered.groupby(level='FIRM')['_TOT'].shift(4)
    df_filtered.drop(columns=['_NEW', '_TOT'], inplace=True)
    _rcb4_denom = (_tot_lag4_rcb + _new_lag1 + _new_lag2 + _new_lag3).replace(0, np.nan)
    df_filtered['RCB4_RRR_PCT'] = (_ret / _rcb4_denom) * 100

    # (A three-quarter rolling base, RCB3, was computed here from 2026-09-18 to
    # 2026-11-11; the author retired it from the paper on 2026-09-27.)

    # Variant A6: four-quarter moving average of the headline signal over the
    # four quarters BEFORE the measurement quarter, author decision 2026-09-27:
    # MA4L_t = mean(RRR_{t-1}, RRR_{t-2}, RRR_{t-3}, RRR_{t-4}), full window
    # required. The idea is to reward firms whose
    # retention has been durably high, and averaging four readings dilutes any
    # single-quarter spike. It excludes RRR_t itself, so it is one quarter older
    # than the headline signal at every formation date. Raw only.
    df_filtered['MA4L_RRR_PCT'] = (
        df_filtered.groupby(level='FIRM')['RRR_PCT']
          .transform(lambda s: s.shift(1).rolling(4, min_periods=4).mean()))

    # Variant B: Share of Retained Revenue, Rev_ret_t / Rev_t  (= 1 - SoNR)
    df_filtered['SRR_PCT'] = (_ret / _tot.replace(0, np.nan)) * 100
    _sm = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])[
        'SRR_PCT'].transform('mean')
    df_filtered['ADJ_SRR_PCT'] = df_filtered['SRR_PCT'] - _sm

    # Variant C: Cumulative-denominator RRR
    # CUMRR_t = Rev_ret_t / (Rev_ret_0 + sum_{s<t} Rev_new_s)
    def _compute_cumrr_vals(grp_sorted):
        ret_rev = pd.to_numeric(grp_sorted['#RETURNING_CUSTOMERS'], errors='coerce')
        new_rev = pd.to_numeric(grp_sorted['#NEW_CUSTOMERS'], errors='coerce')
        valid = ret_rev.dropna()
        if valid.empty or valid.iloc[0] == 0:
            return np.full(len(grp_sorted), np.nan)
        ret_rev_0 = valid.iloc[0]
        cum_new_prev = new_rev.shift(1).fillna(0).cumsum()
        denom = (ret_rev_0 + cum_new_prev).replace(0, np.nan)
        return (ret_rev / denom * 100).values

    df_reset = df_filtered.reset_index()
    df_reset['CUMRR_PCT'] = np.nan
    for _firm, _grp in df_reset.groupby('FIRM'):
        _grp_s = _grp.sort_values('DATE')
        df_reset.loc[_grp_s.index, 'CUMRR_PCT'] = _compute_cumrr_vals(_grp_s)
    df_filtered = df_reset.set_index(['FIRM', 'DATE'])

    _sm = df_filtered.groupby([df_filtered.index.get_level_values('DATE'), 'SECTOR'])[
        'CUMRR_PCT'].transform('mean')
    df_filtered['ADJ_CUMRR_PCT'] = df_filtered['CUMRR_PCT'] - _sm

    # Lag-1 for portfolio sorting
    df_filtered['ADJ_LR_RRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_LR_RRR_PCT'].shift(1)
    df_filtered['ADJ_TQ_RRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_TQ_RRR_PCT'].shift(1)
    df_filtered['ADJ_TA4_RRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_TA4_RRR_PCT'].shift(1)
    df_filtered['RCB4_RRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['RCB4_RRR_PCT'].shift(1)
    df_filtered['ADJ_SRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_SRR_PCT'].shift(1)
    df_filtered['ADJ_CUMRR_PCT_LAG1'] = df_filtered.groupby(level='FIRM')['ADJ_CUMRR_PCT'].shift(1)

    print(f"  LR_RRR: {df_filtered['LR_RRR_PCT'].notna().sum()} obs; "
          f"TQ_RRR: {df_filtered['TQ_RRR_PCT'].notna().sum()}; "
          f"TA4_RRR: {df_filtered['TA4_RRR_PCT'].notna().sum()}; "
          f"RCB4_RRR: {df_filtered['RCB4_RRR_PCT'].notna().sum()} (raw only); "
          f"SRR: {df_filtered['SRR_PCT'].notna().sum()}; "
          f"CUMRR: {df_filtered['CUMRR_PCT'].notna().sum()}")

    # --- 1.8 Per-industry descriptive statistics ---
    print("\n--- 1.8 Per-industry descriptive statistics ---")

    metrics = {
        'RG (%)': 'REV_GROWTH_PCT',
        'AR (%)': 'ACQ_RATE_PCT',
        'RRR (%)': 'RRR_PCT'
    }

    all_industry_stats = []

    for sector in VALID_SECTORS:
        sector_data = df_filtered[df_filtered['SECTOR'] == sector]
        n_firms = sector_data.reset_index()['FIRM'].nunique()

        print(f"\n  {sector} ({n_firms} firms):")

        for metric_name, col_name in metrics.items():
            vals = pd.to_numeric(sector_data[col_name], errors='coerce').dropna()
            if len(vals) > 0:
                stats = {
                    'Sector': sector,
                    'Metric': metric_name,
                    'N': len(vals),
                    'Mean': vals.mean(),
                    'Median': vals.median(),
                    'SD': vals.std(),
                    'P10': vals.quantile(0.10),
                    'P25': vals.quantile(0.25),
                    'P75': vals.quantile(0.75),
                    'P90': vals.quantile(0.90)
                }
                all_industry_stats.append(stats)
                print(f"    {metric_name}: N={stats['N']}, Mean={stats['Mean']:.2f}, "
                      f"Median={stats['Median']:.2f}, SD={stats['SD']:.2f}, "
                      f"P10={stats['P10']:.2f}, P25={stats['P25']:.2f}, "
                      f"P75={stats['P75']:.2f}, P90={stats['P90']:.2f}")

    industry_stats_df = pd.DataFrame(all_industry_stats)
    industry_stats_df.to_excel(os.path.join(OUTPUT_DIR, 'industry_descriptives.xlsx'), index=False)

    # --- 1.9 Pooled descriptive statistics ---
    print("\n--- 1.9 Pooled descriptive statistics ---")

    summary_vars = {
        'Assets': 'BS_TOT_ASSET',
        'Market Cap': 'HISTORICAL_MARKET_CAP',
        'Revenue': 'SALES_REV_TURN',
        'RRR (%)': 'RRR_PCT',
        'Acq Rate (%)': 'ACQ_RATE_PCT',
        'Rev Growth (%)': 'REV_GROWTH_PCT',
        'BTM': 'BTM',
        'PM (%)': 'PM_OPER_PCT',
        'MTE': 'MTE',
        'Op. ROA (%)': 'ROA_OPER',
        'Adj RRR (%)': 'ADJ_RRR_PCT',
        'Adj AR (%)': 'ADJ_ACQ_RATE_PCT',
    }

    pooled_stats = []
    for display_name, col_name in summary_vars.items():
        if col_name in df_filtered.columns:
            data = pd.to_numeric(df_filtered[col_name], errors='coerce').dropna()
            pooled_stats.append({
                'Variable': display_name,
                'N': len(data),
                'Mean': data.mean(),
                'Median': data.median(),
                'SD': data.std(),
                'P10': data.quantile(0.10),
                'P25': data.quantile(0.25),
                'P75': data.quantile(0.75),
                'P90': data.quantile(0.90)
            })

    pooled_df = pd.DataFrame(pooled_stats)
    print(pooled_df.to_string(index=False))
    pooled_df.to_excel(os.path.join(OUTPUT_DIR, 'pooled_descriptives.xlsx'), index=False)

    # --- 1.9b Portfolio characteristics by RRR quartile ---
    # Sorted on the RAW signal: raw is the paper's primary specification (author decision 2026-09-18).
    print("\n--- 1.9b Portfolio characteristics by RRR quartile (raw signal) ---")
    df_filtered['RRR_Q_ADJ_CHAR'] = df_filtered.groupby(
        df_filtered.index.get_level_values('DATE'))['ADJ_RRR_PCT'].transform(safe_quartile)
    df_filtered['AR_Q_ADJ_CHAR'] = df_filtered.groupby(
        df_filtered.index.get_level_values('DATE'))['ADJ_ACQ_RATE_PCT'].transform(safe_quartile)
    df_filtered['RRR_Q_RAW_CHAR'] = df_filtered.groupby(
        df_filtered.index.get_level_values('DATE'))['RRR_PCT'].transform(safe_quartile)
    df_filtered['AR_Q_RAW_CHAR'] = df_filtered.groupby(
        df_filtered.index.get_level_values('DATE'))['ACQ_RATE_PCT'].transform(safe_quartile)

    char_cols = {
        'Revenue (M)': 'SALES_REV_TURN',
        'Market Cap (M)': 'HISTORICAL_MARKET_CAP',
        'Rev Growth (%)': 'REV_GROWTH_PCT',
        'SoNR (%)': 'SHARE_NEW_REV_PCT',
        'RRR (%)': 'RRR_PCT',
        'BTM': 'BTM',
        'MTE': 'MTE',
        'Op. ROA (%)': 'ROA_OPER',
    }

    quartile_char_rows = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        qdf = df_filtered[df_filtered['RRR_Q_RAW_CHAR'] == q].reset_index()
        avg_n = qdf.groupby('DATE')['FIRM'].nunique().mean()
        row = {'Quartile': q, 'Avg N / quarter': round(avg_n, 1)}
        for var_label, col_name in char_cols.items():
            if col_name in qdf.columns:
                vals = pd.to_numeric(qdf[col_name], errors='coerce').dropna()
                row[var_label] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
            else:
                row[var_label] = np.nan
        quartile_char_rows.append(row)

    quartile_char_df = pd.DataFrame(quartile_char_rows)
    print(quartile_char_df.to_string(index=False))
    quartile_char_df.to_excel(os.path.join(OUTPUT_DIR, 'portfolio_characteristics.xlsx'), index=False)

    # --- 1.9c Quartile persistence rate ---
    print("\n--- 1.9c Quartile persistence rate ---")
    pers_df = df_filtered[['RRR_Q_RAW_CHAR']].copy().reset_index()
    pers_df = pers_df.sort_values(['FIRM', 'DATE'])
    pers_df['Q_PREV'] = pers_df.groupby('FIRM')['RRR_Q_RAW_CHAR'].shift(1)
    valid = pers_df.dropna(subset=['RRR_Q_RAW_CHAR', 'Q_PREV'])
    overall_rate = (valid['RRR_Q_RAW_CHAR'] == valid['Q_PREV']).mean()
    print(f"  Overall quartile persistence rate: {overall_rate:.1%}")
    per_q = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = valid[valid['Q_PREV'] == q]
        per_q[q] = (sub['RRR_Q_RAW_CHAR'] == q).mean() if len(sub) > 0 else np.nan
        print(f"  {q} stay rate: {per_q[q]:.1%}")
    pers_out = pd.DataFrame([{
        'overall_persistence': overall_rate,
        'Q1_persistence': per_q['Q1'],
        'Q2_persistence': per_q['Q2'],
        'Q3_persistence': per_q['Q3'],
        'Q4_persistence': per_q['Q4'],
    }])
    pers_out.to_excel(os.path.join(OUTPUT_DIR, 'quartile_persistence.xlsx'), index=False)

    # --- 1.9d Lag-4 columns (lag-1 already in df_filtered from 1.7) ---
    # The signal-persistence comparison that used to follow here (raw vs adjusted
    # RRR and AR, output/signal_persistence_comparison.xlsx) fed an Appendix table
    # the author removed on 2026-11-10; only the lagged columns remain.
    df_filtered['RRR_PCT_LAG4']          = df_filtered.groupby(level='FIRM')['RRR_PCT'].shift(4)
    df_filtered['ADJ_RRR_PCT_LAG4']      = df_filtered.groupby(level='FIRM')['ADJ_RRR_PCT'].shift(4)
    df_filtered['ACQ_RATE_PCT_LAG4']     = df_filtered.groupby(level='FIRM')['ACQ_RATE_PCT'].shift(4)
    df_filtered['ADJ_ACQ_RATE_PCT_LAG4'] = df_filtered.groupby(level='FIRM')['ADJ_ACQ_RATE_PCT'].shift(4)

    # --- 1.10 Adjusted signal dispersion check ---
    print("\n--- 1.10 Adjusted signal dispersion check ---")
    for sig_name, sig_col in [('Adj RRR', 'ADJ_RRR_PCT'), ('Adj AR', 'ADJ_ACQ_RATE_PCT')]:
        vals = pd.to_numeric(df_filtered[sig_col], errors='coerce').dropna()
        print(f"  {sig_name}: SD={vals.std():.2f}, IQR={vals.quantile(0.75)-vals.quantile(0.25):.2f}, "
              f"P10={vals.quantile(0.10):.2f}, P90={vals.quantile(0.90):.2f}")

    return df_long, df_filtered, industry_stats_df


_MONTHLY_RETURNS_CACHE = {}   # the monthly sheet is read once per run (see below)


def phase1c_load_monthly_returns():
    """Load and process monthly return data.

    Since 2026-10-20 phase1_load_and_diagnose() needs the monthly prices for the
    Bloomberg stamping check, so this loader can run twice per script. The
    result is cached on first use and a copy is returned each time.
    """
    if 'returns' in _MONTHLY_RETURNS_CACHE:
        return _MONTHLY_RETURNS_CACHE['returns'].copy()

    print("\n" + "=" * 80)
    print("PHASE 1C: MONTHLY RETURNS")
    print("=" * 80)

    # 2026-09-08: monthly returns are now a SHEET with a SIX-row header block:
    #   row 0 "Start Date", row 1 "End Date", row 2 blank,
    #   row 3 ticker, row 4 label ("Last Price"), row 5 mnemonic ("PX_LAST").
    # Data starts at row 6. The old code assumed a 2-row header and would have
    # read the ticker block itself as price data.
    monthly_raw = pd.read_excel(DATA_WORKBOOK, sheet_name=SHEET_RETURNS, header=None)

    TICKER_ROW, MNEMONIC_ROW, FIRST_DATA_ROW = 3, 5, 6
    header_rows = monthly_raw.iloc[[TICKER_ROW, MNEMONIC_ROW]]
    data_rows = monthly_raw.iloc[FIRST_DATA_ROW:].copy()
    combined_headers = header_rows.apply(lambda x: x.astype(str).str.strip(), axis=0)
    col_headers = combined_headers.apply(
        lambda x: '.'.join(v for v in x if v and v.lower() != 'nan'), axis=0)
    data_rows.columns = col_headers
    data_rows = data_rows.reset_index(drop=True)
    data_rows.rename(columns={data_rows.columns[0]: 'Date'}, inplace=True)
    data_rows.replace('#N/A N/A', np.nan, inplace=True)
    data_rows['Date'] = pd.to_datetime(data_rows['Date'], errors='coerce', dayfirst=True)
    # Drop any trailing rows without a parseable date (data-pull stubs).
    data_rows = data_rows[data_rows['Date'].notna()].reset_index(drop=True)

    # The 2026-09-04 sheet ships DLTH US Equity twice. The revenue and
    # fundamentals loaders already de-duplicate; this one did not, and a
    # duplicate label makes df[col] return a DataFrame, which blows up
    # pd.to_numeric downstream. Drop repeats, but only after confirming they
    # are genuinely identical -- a duplicate label hiding DIFFERENT data would
    # be a data error we must not paper over.
    _dupes = data_rows.columns[data_rows.columns.duplicated()].unique().tolist()
    if _dupes:
        for _name in _dupes:
            _block = data_rows.loc[:, data_rows.columns == _name]
            _first = _block.iloc[:, 0]
            for _j in range(1, _block.shape[1]):
                if not _first.astype(str).equals(_block.iloc[:, _j].astype(str)):
                    raise ValueError(
                        f"Column '{_name}' appears {_block.shape[1]} times in "
                        f"'{SHEET_RETURNS}' with DIFFERENT values. Resolve this in "
                        f"the source workbook; refusing to guess which copy is right.")
        print(f"  Dropping {len(_dupes)} exact duplicate column(s): {_dupes}")
        data_rows = data_rows.loc[:, ~data_rows.columns.duplicated()]

    df_prices = data_rows.copy()
    for c in df_prices.columns:
        if c != 'Date':
            df_prices[c] = pd.to_numeric(df_prices[c], errors='coerce')

    df_prices = backfill_monthly_prices(df_prices)

    # Monthly log returns, simple returns, AND the price level.
    # The price level is retained so that market cap can later be measured as of
    # the formation date (quarter-end market cap scaled by the firm's own price
    # change), rather than using the stale quarter-end market cap.
    px_only = df_prices.drop(columns=['Date'])
    log_ret = np.log(px_only / px_only.shift(1))

    px_long = px_only.copy()
    px_long['Date'] = df_prices['Date'].values
    px_long = px_long.melt(id_vars='Date', var_name='FirmVar', value_name='PX')

    log_ret['Date'] = df_prices['Date'].values
    ret_long = log_ret.melt(id_vars='Date', var_name='FirmVar', value_name='RETURN_LOG')

    returns = ret_long.merge(px_long, on=['Date', 'FirmVar'], how='left')
    returns['FIRM'] = returns['FirmVar'].str.replace('.PX_LAST', '', regex=False).str.upper()
    returns.drop(columns=['FirmVar'], inplace=True)
    returns = returns.dropna(subset=['RETURN_LOG'])
    # Simple (arithmetic) monthly return; portfolios are value-weighted from these,
    # NOT from log returns (a value-weighted mean of log returns is biased).
    returns['RET_SIMPLE'] = np.exp(returns['RETURN_LOG']) - 1

    # NOTE: no quarter lag is baked in here. All signal->return timing (the
    # 2-month formation gap and 3-month hold) is applied once, in
    # build_holding_panel(), to avoid stacking a second implicit lag.
    print(f"  Monthly returns: {len(returns)} obs, {returns['FIRM'].nunique()} firms")
    print(f"  Date range: {returns['Date'].min()} to {returns['Date'].max()}")

    _MONTHLY_RETURNS_CACHE['returns'] = returns
    return returns.copy()
