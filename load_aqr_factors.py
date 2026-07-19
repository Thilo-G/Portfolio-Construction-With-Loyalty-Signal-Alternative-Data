"""Load AQR Quality Minus Junk (QMJ) and Betting Against Beta (BAB) US factor series.

Standalone data-acquisition script. Loads the raw AQR workbooks in Data\\, extracts
the US monthly long/short factor return series, restricts to the 2017-01 through
2024-12 research sample window, and validates completeness and value ranges.

This script does NOT feed into analysis_v2.py or the frozen
2025-06-04b-TK-RRR_financialimplications-main.py. Integration into the main
pipeline happens separately.

Source (confirmed by browsing, not guessed):
  QMJ landing page : https://www.aqr.com/Insights/Datasets/Quality-Minus-Junk-Factors-Monthly
  QMJ file         : https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Monthly.xlsx
  BAB landing page : https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly
  BAB file         : https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx

QMJ underlying research: Asness, Frazzini & Pedersen (2019), "Quality Minus Junk,"
  Review of Accounting Studies (AQR's workbook cites the 2014 SSRN working paper
  version of the same portfolio construction).
BAB underlying research: Frazzini & Pedersen (2014), "Betting Against Beta,"
  Journal of Financial Economics, 111, 1-25.

Downloaded: 2026-07-19.
"""
import pandas as pd
from pathlib import Path

# --- File path constants -----------------------------------------------------
DATA_DIR = Path(
    r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication\Data"
)
QMJ_FILE_PATH = DATA_DIR / "AQR_QMJ_Factors_Monthly_2026-07-19.xlsx"
BAB_FILE_PATH = DATA_DIR / "AQR_BAB_Factors_Monthly_2026-07-19.xlsx"

# --- Workbook layout constants (confirmed by inspection with openpyxl) -------
# Both AQR workbooks share the same layout: ~17 rows of free-text description,
# a group-label row, then a header row with DATE + ISO3 country/region columns.
QMJ_SHEET_NAME = "QMJ Factors"
BAB_SHEET_NAME = "BAB Factors"
HEADER_ROW_INDEX = 18  # 0-based pandas row index -> Excel row 19, the real header row
DATE_COLUMN = "DATE"
COUNTRY_COLUMN = "USA"
DATE_FORMAT = "%m/%d/%Y"  # AQR stores DATE as month-end strings, e.g. "12/31/2024"

# --- Research sample window ---------------------------------------------------
WINDOW_START = pd.Timestamp("2017-01-01")
WINDOW_END = pd.Timestamp("2024-12-31")

# --- Sanity bounds for monthly factor returns (decimal scale, not %) ---------
# Long/short equity factor returns rarely exceed +/-20% in a single month.
# Anything outside this band is flagged for manual review, not silently accepted.
MONTHLY_RETURN_SANE_MIN = -0.20
MONTHLY_RETURN_SANE_MAX = 0.20


def load_us_monthly_factor(file_path: Path, sheet_name: str, factor_label: str) -> pd.DataFrame:
    """Load one AQR monthly factor workbook and extract the US long/short series.

    Returns a DataFrame with columns ['date', factor_label], sorted ascending by date,
    covering the full history available in the file (not yet windowed).
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Re-download from AQR (see module docstring for source URLs)."
        )

    factor_raw = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=HEADER_ROW_INDEX,
        usecols=[DATE_COLUMN, COUNTRY_COLUMN],
        dtype={DATE_COLUMN: str, COUNTRY_COLUMN: "float64"},
        engine="openpyxl",
    )
    factor_raw.columns = factor_raw.columns.str.strip()

    factor_clean = factor_raw.copy()
    factor_clean["date"] = pd.to_datetime(factor_clean[DATE_COLUMN], format=DATE_FORMAT)
    factor_clean = factor_clean.rename(columns={COUNTRY_COLUMN: factor_label})
    factor_clean = factor_clean[["date", factor_label]].copy()
    factor_clean = factor_clean.sort_values("date").reset_index(drop=True)

    duplicate_month_count = factor_clean["date"].dt.to_period("M").duplicated().sum()
    if duplicate_month_count > 0:
        raise ValueError(
            f"{factor_label}: found {duplicate_month_count} duplicate month(s) in "
            f"{file_path.name}; investigate the source file before proceeding."
        )

    return factor_clean


def restrict_to_window(factor_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Restrict a date-sorted factor DataFrame to [start, end], inclusive."""
    window_mask = (factor_df["date"] >= start) & (factor_df["date"] <= end)
    factor_window = factor_df.loc[window_mask].copy()
    factor_window = factor_window.sort_values("date").reset_index(drop=True)
    return factor_window


def check_no_missing_months(
    factor_window: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, factor_label: str
) -> list:
    """Confirm every calendar month in [start, end] has exactly one observation."""
    expected_months = pd.period_range(start=start, end=end, freq="M")
    actual_months = factor_window["date"].dt.to_period("M")
    missing_months = sorted(set(expected_months) - set(actual_months))

    if missing_months:
        print(f"  WARNING [{factor_label}]: {len(missing_months)} missing month(s): {missing_months}")
    else:
        print(f"  OK [{factor_label}]: all {len(expected_months)} months present, no gaps.")

    return missing_months


def check_value_range(
    factor_window: pd.DataFrame, factor_label: str, min_sane: float, max_sane: float
) -> pd.DataFrame:
    """Flag monthly returns outside a sane range for manual review; does not drop them."""
    out_of_range_mask = (factor_window[factor_label] < min_sane) | (factor_window[factor_label] > max_sane)
    out_of_range_rows = factor_window.loc[out_of_range_mask].copy()

    if len(out_of_range_rows) > 0:
        print(
            f"  WARNING [{factor_label}]: {len(out_of_range_rows)} month(s) outside "
            f"[{min_sane:.0%}, {max_sane:.0%}], review before use:"
        )
        for _, flagged_row in out_of_range_rows.iterrows():
            month_label = flagged_row["date"].strftime("%Y-%m")
            flagged_value = flagged_row[factor_label]
            print(f"    {month_label}: {flagged_value:.4f} ({flagged_value:.2%})")
    else:
        print(f"  OK [{factor_label}]: all values within [{min_sane:.0%}, {max_sane:.0%}].")

    return out_of_range_rows


def print_summary_statistics(factor_window: pd.DataFrame, factor_label: str) -> None:
    """Print mean, SD, min, max of the monthly factor return series (decimal and % display)."""
    mean_return = factor_window[factor_label].mean()
    sd_return = factor_window[factor_label].std()
    min_return = factor_window[factor_label].min()
    max_return = factor_window[factor_label].max()

    print(f"  {factor_label} summary over {len(factor_window)} months:")
    print(f"    mean = {mean_return:.4f}  ({mean_return:.2%})")
    print(f"    sd   = {sd_return:.4f}  ({sd_return:.2%})")
    print(f"    min  = {min_return:.4f}  ({min_return:.2%})")
    print(f"    max  = {max_return:.4f}  ({max_return:.2%})")


if __name__ == "__main__":
    qmj_full_history = load_us_monthly_factor(QMJ_FILE_PATH, QMJ_SHEET_NAME, "QMJ_USA")
    bab_full_history = load_us_monthly_factor(BAB_FILE_PATH, BAB_SHEET_NAME, "BAB_USA")

    print(f"QMJ file covers {qmj_full_history['date'].min():%Y-%m} to {qmj_full_history['date'].max():%Y-%m}")
    print(f"BAB file covers {bab_full_history['date'].min():%Y-%m} to {bab_full_history['date'].max():%Y-%m}")

    qmj_covers_window = qmj_full_history["date"].min() <= WINDOW_START and qmj_full_history["date"].max() >= WINDOW_END
    bab_covers_window = bab_full_history["date"].min() <= WINDOW_START and bab_full_history["date"].max() >= WINDOW_END
    both_files_cover_window = qmj_covers_window and bab_covers_window
    coverage_label = "within" if both_files_cover_window else "NOT fully within"
    print(f"Research window {WINDOW_START:%Y-%m} to {WINDOW_END:%Y-%m} is {coverage_label} both files' coverage.")
    print()

    qmj_window = restrict_to_window(qmj_full_history, WINDOW_START, WINDOW_END)
    bab_window = restrict_to_window(bab_full_history, WINDOW_START, WINDOW_END)

    print("Missing-month check (2017-01 through 2024-12):")
    check_no_missing_months(qmj_window, WINDOW_START, WINDOW_END, "QMJ_USA")
    check_no_missing_months(bab_window, WINDOW_START, WINDOW_END, "BAB_USA")
    print()

    print(f"Value-range check (sane band [{MONTHLY_RETURN_SANE_MIN:.0%}, {MONTHLY_RETURN_SANE_MAX:.0%}]):")
    check_value_range(qmj_window, "QMJ_USA", MONTHLY_RETURN_SANE_MIN, MONTHLY_RETURN_SANE_MAX)
    check_value_range(bab_window, "BAB_USA", MONTHLY_RETURN_SANE_MIN, MONTHLY_RETURN_SANE_MAX)
    print()

    print("Summary statistics, 2017-01 through 2024-12:")
    print_summary_statistics(qmj_window, "QMJ_USA")
    print_summary_statistics(bab_window, "BAB_USA")
    print()

    combined_window = pd.merge(qmj_window, bab_window, on="date", how="inner")
    print(f"Combined US factor frame: {combined_window.shape[0]} rows, {combined_window.shape[1]} columns")
    print(combined_window.head())
    print("...")
    print(combined_window.tail())
