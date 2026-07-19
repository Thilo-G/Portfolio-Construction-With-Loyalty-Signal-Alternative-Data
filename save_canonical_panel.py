"""
save_canonical_panel.py -- RRR Financial Implications
=======================================================
Persists the firm-quarter analysis panel produced by analysis_v2.py's
Phase 1 (phase1_load_and_diagnose) to a dated file on disk.

Problem this solves: the panel (RRR, AR, and related variables, after
the sector filter and industry x time adjustment) is currently rebuilt
at runtime every time analysis_v2.py runs and is never saved. That
makes it impossible to inspect the exact firm-quarter sample used for a
given draft of the paper, or to diff the panel across code changes.

This script does not reimplement any loading/filtering logic. It
imports analysis_v2 and calls its existing phase1_load_and_diagnose()
entry point, then writes the returned df_filtered (the sector-filtered,
industry-adjusted panel -- the one actually used for the paper's main
analysis) to Code/RRR-FI-IM/output/canonical_panel_<YYYYMMDD>.<ext>.

Output format: parquet if a parquet engine (pyarrow or fastparquet) is
installed, otherwise xlsx. As of 2026-07, neither pyarrow nor
fastparquet is installed in the project interpreter
(C:\\Users\\thkraft\\AppData\\Local\\Programs\\Python\\Python311\\python.exe),
so this currently writes xlsx.

Usage:
    python save_canonical_panel.py
"""

import os
import sys
import importlib.util
from datetime import date

import pandas as pd

CODE_DIR = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Code\RRR-FI-IM"

sys.path.insert(0, CODE_DIR)
import analysis_v2  # noqa: E402 -- import must follow sys.path setup


def _parquet_engine_available():
    """Check for an installed parquet engine without relying on exception control flow."""
    return (
        importlib.util.find_spec('pyarrow') is not None
        or importlib.util.find_spec('fastparquet') is not None
    )


def save_canonical_panel():
    """Run analysis_v2 Phase 1 once and persist df_filtered to a dated output file."""
    df_long, df_filtered, industry_stats_df = analysis_v2.phase1_load_and_diagnose()

    # FIRM and DATE move from MultiIndex to plain columns for on-disk storage.
    panel_to_save = df_filtered.reset_index()

    today_str = date.today().strftime('%Y%m%d')

    if _parquet_engine_available():
        output_path = os.path.join(analysis_v2.OUTPUT_DIR, f"canonical_panel_{today_str}.parquet")
        panel_to_save.to_parquet(output_path, index=False)
        saved_format = "parquet"
    else:
        output_path = os.path.join(analysis_v2.OUTPUT_DIR, f"canonical_panel_{today_str}.xlsx")
        panel_to_save.to_excel(output_path, index=False, engine='openpyxl')
        saved_format = "xlsx (no parquet engine installed: pyarrow/fastparquet both missing)"

    n_firms = panel_to_save['FIRM'].nunique()
    n_quarters = panel_to_save['DATE'].nunique()

    print("\n" + "=" * 80)
    print("CANONICAL PANEL SAVED")
    print("=" * 80)
    print(f"Path:            {output_path}")
    print(f"Format:          {saved_format}")
    print(f"Shape:           {panel_to_save.shape[0]:,} rows x {panel_to_save.shape[1]} columns")
    print(f"Unique firms:    {n_firms}")
    print(f"Unique quarters: {n_quarters}")
    print(f"Date range:      {panel_to_save['DATE'].min()} to {panel_to_save['DATE'].max()}")

    return panel_to_save, output_path


if __name__ == '__main__':
    save_canonical_panel()
