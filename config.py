"""config.py -- RRR Financial Implications
=========================================

Every path, data file, and fixed setting of the analysis, in one place. The
analysis modules import from here; other scripts reach the same names through
analysis_v2 (av2.OUTPUT_DIR, av2.FORM_LAG_MONTHS, ...).

Paths are built from this file's location (Code/RRR-FI-IM), so the project
folder can move without edits.
"""

# =============================================================================
# Imports
# =============================================================================
import os

# =============================================================================
# Folders
# =============================================================================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))           # Code/RRR-FI-IM
PROJECT_DIR = os.path.dirname(os.path.dirname(CODE_DIR))       # RRR_FinancialImplication
DATA_DIR = os.path.join(PROJECT_DIR, 'Data')
SUPPORT_DIR = os.path.join(PROJECT_DIR, 'Supporting-Documents')
OUTPUT_DIR = os.path.join(CODE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Input data (2026-09-08 vintage)
# =============================================================================
# One workbook with four sheets. The header depth differs per sheet, so each
# loader in rrr_data.py states the rows it uses.
DATA_WORKBOOK = os.path.join(DATA_DIR, '20260904-Firms-Quarterly-Revenue-Fundamentals-Python.xlsx')
SHEET_REVENUE = 'Quarterly_Revenue_Values'          # card revenue: header rows 0 (ticker), 1 (field)
SHEET_FUNDAMENTALS = 'Quarterly_Fundamentals_Value'   # Bloomberg: rows 0 (ticker), 1 (label), 2 (mnemonic)
SHEET_RETURNS = 'Monthly_Returns_Values'            # monthly prices: header rows 0-5; ticker = 3, mnemonic = 5
SHEET_INDUSTRY = 'Industry_Values'                  # GICS classification: one header row

# Fama-French five factors and RF (Ken French library, pulled 2026-09-08), and the
# momentum factor pinned separately (release of 2026-09-24, CRSP 202608). The live
# download in rrr_data.download_ff_factors() is only a fallback if a file is missing.
FF_LOCAL_CSV = os.path.join(DATA_DIR, '20260908-FF-Factors-Monthly_Python.csv')
MOM_LOCAL_CSV = os.path.join(DATA_DIR, '20260928-FF-Momentum-Monthly_Python.csv')
FF3_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'
FF5_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip'
MOM_URL = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip'

# Annual revenue panel (all firms; filter to the sample before use).
ANNUAL_REVENUE_WORKBOOK = os.path.join(DATA_DIR, '20260409-Firms-Revenue-2025-all-Python.xlsx')
SHEET_ANNUAL_REVENUE = 'Year_Revenue_Value'         # header rows 0 (ticker), 1 (field)

# =============================================================================
# Backfill from the previous data vintage (added 2026-09-08)
# =============================================================================
# 17 firms of the earlier sample miss fields in the 2026-09-04 pull and are
# filled from the last vintage that had them (through 2024Q4 only, so they drop
# out of 2025+ automatically). See output/DATA_COVERAGE_20260908.md.
REVENUE_BACKFILL_FIRMS = [      # card revenue and GICS sector missing
    'ACI US EQUITY', 'AKA US EQUITY', 'FL US EQUITY', 'JWN US EQUITY',
    'NTCO3 BZ EQUITY', 'PBPB US EQUITY', 'PRTS US EQUITY', 'PZZA US EQUITY',
    'RGS US EQUITY', 'SAVEQ US EQUITY', 'SN US EQUITY',
]
MCAP_BACKFILL_FIRMS = [         # HISTORICAL_MARKET_CAP missing
    'DSHK US EQUITY', 'FAT US EQUITY', 'QVCGA US EQUITY',
    'SCVL US EQUITY', 'SKLZ US EQUITY', 'VSCO US EQUITY',
]
PRICE_BACKFILL_FIRMS = REVENUE_BACKFILL_FIRMS + MCAP_BACKFILL_FIRMS   # all 17 lost monthly prices

LEGACY_REVENUE_FILE = os.path.join(DATA_DIR, 'Archive', '2025-0319a-TK-quarterlyrevenue-collection_Python.xlsx')
LEGACY_FUNDAMENTALS_FILE = os.path.join(DATA_DIR, '2025-0319a-TK-fundamentals_Python.xlsx')
LEGACY_RETURNS_FILE = os.path.join(DATA_DIR, 'Archive', '2025-07-02a-TK-Monthly-Returns_Python.xlsx')
LEGACY_INDUSTRY_FILE = os.path.join(SUPPORT_DIR, 'Firms-Industry-2025-11-21-Python.xlsx')

# =============================================================================
# Sample
# =============================================================================
# GICS sectors with at least 10 firms after the revenue-correlation screen.
VALID_SECTORS = ['Consumer Discretionary', 'Communication Services', 'Consumer Staples', 'Industrials']

# Firms removed by author decision (2026-10-19), applied right after the sector
# filter so every script inherits it. GameStop's 2021 price moves came from
# coordinated retail trading (a "meme stock"), not from fundamentals.
EXCLUDED_FIRMS = {'GME US EQUITY': 'meme-stock trading in 2021, not driven by fundamentals'}

# =============================================================================
# Bloomberg quarter stamping (author request 2026-10-19; see bloomberg_dating.py)
# =============================================================================
# About 51 firms have their Bloomberg quarterly fields stamped up to three
# quarters off calendar time. rrr_data.phase1_load_and_diagnose() checks and
# re-dates them on every load. Set to False only to reproduce pre-fix numbers.
ALIGN_BLOOMBERG_DATES = True
FUND_DATE_COL = 'nan.Dates'          # date column of the wide fundamentals sheet
# Re-dating leaves the last k quarters of a firm stamped k quarters late without
# Bloomberg fields; its market cap is then carried forward from its last
# available quarter (implied shares x monthly price), for at most this many quarters.
MCAP_TAIL_FILL_MAX_QUARTERS = 3

# =============================================================================
# Portfolio timing
# =============================================================================
# Every signal is measured at a calendar quarter-end Q. The portfolio is formed
# FORM_LAG_MONTHS calendar months after Q, when the quarter's results are public,
# and held for HOLD_MONTHS months: Q = March 31 -> form end of May -> hold June,
# July, August. The lag lives only in this formation gap; signals are never
# shifted a second time (rrr_portfolios.build_holding_panel applies the timing).
FORM_LAG_MONTHS = 2
HOLD_MONTHS = 3

# =============================================================================
# Guards: values the pipeline must reproduce (update only on purpose)
# =============================================================================
# Firms in the quarterly panel after the sector filter and exclusions
# (the canonical panel snapshot). 2026-10-20: 137 -> 136 (GameStop excluded).
EXPECTED_N_FIRMS_PANEL = 136
# Firms that also have monthly returns and so enter the portfolio sorts
# (asserted in rrr_regressions.run_portfolio_analysis). 2026-10-20: 134 -> 133.
# Of these, 121 carry a formation market cap; the others get no value weight.
EXPECTED_N_FIRMS = 133
# Headline value-weighted Q1-Q4 five-factor alpha, percent per month, checked by
# signal_ic_diagnostics.py, asymmetry_across_variants.py and numbers_tex.py.
# 2026-10-20: 2.2511 -> 2.1721 after re-dating Bloomberg fields and excluding GameStop.
EXPECTED_HEADLINE_FF5_ALPHA_PCT = 2.1721
