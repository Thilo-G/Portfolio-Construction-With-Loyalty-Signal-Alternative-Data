"""bloomberg_dating.py -- RRR Financial Implications
===================================================

Re-dates the quarterly Bloomberg fields of firms whose rows in the data
workbook are not stamped in calendar time.

The problem (found 2026-10-19)
------------------------------
In the quarterly fundamentals sheet, the Bloomberg fields (market cap, sales,
balance sheet, income statement) of about 51 sample firms sit up to three rows
away from the calendar quarter they belong to. Most are retailers whose fiscal
year ends in late January. Example: Macy's revenue for February-April 2020 sits
in the December 2020 row. The card-panel revenue, and therefore RRR, is dated
correctly. Only the Bloomberg-sourced fields are affected: the value weights
(market cap) and the firm characteristics (size, book-to-market, profit margin).

How an offset is found
----------------------
Market data only. For each firm, the quarterly log change in
HISTORICAL_MARKET_CAP (fundamentals sheet) is correlated with the quarterly log
change in the firm's quarter-end share price (monthly price sheet, which is
dated correctly), for offsets k = -3, ..., +3: the price change in quarter t is
paired with the market-cap change in row t + k. A positive k means the
Bloomberg row is stamped k quarters late; a negative k means early.
A firm is re-dated only when its best offset is clearly identified: the best
correlation is at least MIN_BEST_R and beats the k = 0 correlation by at least
MIN_GAIN_OVER_ZERO. Otherwise k = 0.

How the fix is applied
----------------------
Every Bloomberg column of a re-dated firm is shifted by -k rows, so the value
that belongs to calendar quarter t moves into row t. The last k rows of a
late-stamped firm are left empty, because the workbook holds no later data.

How it is used
--------------
1. detect_bloomberg_offsets.py runs detect_offsets() once on the raw data and
   writes config/bloomberg_stamping_offsets.csv.
2. analysis_v2.phase1_load_and_diagnose() calls check_and_apply() on every load.
   It detects the offsets again on the data it has just read, compares them
   with the stored file, and then
     - every offset is 0: the data were fixed upstream, nothing is shifted, and
       a notice says the stored file can be retired;
     - the detected offsets equal the stored ones: they are applied;
     - anything else: the run stops and lists the firms that disagree.
   This is the independent check the author asked for on every new data load
   (2026-10-20).

This is a stop-gap. The clean fix is a Bloomberg re-pull aligned to calendar
quarters, after which step 2 reports every offset as 0.

No significance stars; nothing here estimates a regression.
"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Constants
# =============================================================================
HERE = Path(__file__).resolve().parent
OFFSETS_CSV = HERE / 'config' / 'bloomberg_stamping_offsets.csv'

MCAP_FIELD = 'HISTORICAL_MARKET_CAP'
OFFSETS_TESTED = range(-3, 4)    # offsets k tested, in quarters
MIN_BEST_R = 0.6                 # best correlation needed before a firm is re-dated
MIN_GAIN_OVER_ZERO = 0.3         # how far the best offset must beat k = 0
MIN_PAIRED_QUARTERS = 12         # quarters needed before a correlation is computed


# =============================================================================
# Helpers
# =============================================================================
def firm_of(column):
    """Firm key of a wide-sheet column: 'M US Equity.SALES_REV_TURN' -> 'M US EQUITY'."""
    return column.rsplit('.', 1)[0].upper().strip()


def field_of(column):
    """Bloomberg mnemonic of a wide-sheet column: '... .SALES_REV_TURN' -> 'SALES_REV_TURN'."""
    return column.rsplit('.', 1)[1].strip() if '.' in column else ''


def quarter_end_prices(returns):
    """Share price in the last month of each calendar quarter, indexed by (FIRM, quarter).

    `returns` is the long monthly frame of analysis_v2.phase1c_load_monthly_returns()
    with columns FIRM, Date and PX."""
    r = returns[['FIRM', 'Date', 'PX']].copy()
    r['FIRM'] = r['FIRM'].str.upper().str.strip()
    r['MONTH'] = pd.to_datetime(r['Date']).dt.to_period('M')
    r = r[r['MONTH'].dt.month.isin([3, 6, 9, 12])].drop_duplicates(['FIRM', 'MONTH'])
    r['QP'] = r['MONTH'].dt.asfreq('Q')
    return r.set_index(['FIRM', 'QP'])['PX']


def market_caps_long(df_fundamentals, date_col):
    """Quarterly market cap per firm, long format (FIRM, QP, MCAP), from the wide sheet."""
    quarters = pd.to_datetime(df_fundamentals[date_col]).dt.to_period('Q')
    frames = [pd.DataFrame({'FIRM': firm_of(col),
                            'QP': quarters.values,
                            'MCAP': pd.to_numeric(df_fundamentals[col], errors='coerce').values})
              for col in df_fundamentals.columns if field_of(col) == MCAP_FIELD]
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Detection
# =============================================================================
def detect_offsets(mcap_long, px_quarter_end):
    """One row per firm: the offset k (0 unless clearly identified) and the
    correlation at every offset tested. Firms without a monthly price series
    keep k = 0, since nothing can be compared."""
    firms_with_px = set(px_quarter_end.index.get_level_values(0))
    rows = []
    for firm, g in mcap_long.groupby('FIRM'):
        rec = {'FIRM': firm, 'k': 0, 'r_best': np.nan, 'r_k0': np.nan}
        if firm in firms_with_px:
            mcap = g.set_index('QP').sort_index()['MCAP']
            quarters = pd.period_range(mcap.index.min(), mcap.index.max(), freq='Q')
            d_mcap = np.log(mcap.where(mcap > 0).reindex(quarters)).diff()
            d_px = np.log(px_quarter_end.loc[firm].reindex(quarters)).diff()
            corr = {}
            for k in OFFSETS_TESTED:
                pair = pd.concat([d_px, d_mcap.shift(-k)], axis=1, keys=['px', 'mcap']).dropna()
                corr[k] = pair['px'].corr(pair['mcap']) if len(pair) >= MIN_PAIRED_QUARTERS else np.nan
                rec[f'r_k{k}'] = corr[k]
            corr = pd.Series(corr).dropna()
            if not corr.empty:
                best = int(corr.idxmax())
                rec['r_best'] = corr[best]
                rec['r_k0'] = corr.get(0, np.nan)
                # A NaN k = 0 correlation makes the comparison False, so the firm keeps k = 0.
                if best != 0 and corr[best] >= MIN_BEST_R and corr[best] - corr.get(0, np.nan) >= MIN_GAIN_OVER_ZERO:
                    rec['k'] = best
        rows.append(rec)
    return pd.DataFrame(rows).set_index('FIRM')


# =============================================================================
# Re-dating
# =============================================================================
def apply_offsets(df_fundamentals, offsets, date_col):
    """Shift every Bloomberg column of each firm in `offsets` ({FIRM: k}) by -k rows.

    Rows must be consecutive calendar quarters in ascending order; the function
    stops otherwise, because a row shift would then not be a quarter shift."""
    quarters = pd.to_datetime(df_fundamentals[date_col]).dt.to_period('Q')
    steps = np.diff([q.ordinal for q in quarters])
    if not (steps == 1).all():
        raise ValueError('Fundamentals rows are not consecutive calendar quarters; '
                         'cannot re-date Bloomberg fields by shifting rows.')
    out = df_fundamentals.copy()
    for firm, k in offsets.items():
        if k == 0:
            continue
        cols = [c for c in out.columns if c != date_col and firm_of(c) == firm]
        out[cols] = out[cols].shift(-int(k))
    return out


def check_and_apply(df_fundamentals, returns, date_col, offsets_csv=OFFSETS_CSV):
    """Detect the offsets on the data just loaded, compare them with the stored
    file, and re-date the fundamentals. Returns (fundamentals, {FIRM: k applied})."""
    detected = detect_offsets(market_caps_long(df_fundamentals, date_col),
                              quarter_end_prices(returns))['k'].astype(int)

    if (detected == 0).all():
        print('  Bloomberg stamping: every firm is aligned with calendar quarters; nothing re-dated.')
        if Path(offsets_csv).exists():
            print(f'  The data look fixed upstream: {Path(offsets_csv).name} can be retired.')
        return df_fundamentals, {}

    if not Path(offsets_csv).exists():
        raise FileNotFoundError(
            f'{Path(offsets_csv).name} not found, but the data contain mis-stamped Bloomberg rows. '
            'Run detect_bloomberg_offsets.py first and review the offsets it writes.')

    stored = pd.read_csv(offsets_csv).set_index('FIRM')['k'].astype(int)
    stored = stored.reindex(detected.index).fillna(0).astype(int)
    disagree = detected[detected != stored].index
    if len(disagree):
        table = pd.DataFrame({'detected k': detected[disagree], 'stored k': stored[disagree]})
        raise RuntimeError(
            'Bloomberg stamping offsets in the loaded data differ from '
            f'{Path(offsets_csv).name} for {len(disagree)} firm(s):\n{table.to_string()}\n'
            'Check the new data: were some firms fixed upstream, or are new firms mis-stamped? '
            'Re-run detect_bloomberg_offsets.py only after reviewing these firms.')

    applied = detected[detected != 0]
    out = apply_offsets(df_fundamentals, applied.to_dict(), date_col)
    per_k = applied.value_counts().sort_index().to_dict()
    print(f'  Bloomberg stamping: re-dated {len(applied)} firms; firms per offset k: {per_k}')
    return out, applied.to_dict()
