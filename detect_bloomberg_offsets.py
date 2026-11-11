"""detect_bloomberg_offsets.py -- RRR Financial Implications
============================================================

Detects, on the raw data workbook, by how many quarters each firm's Bloomberg
quarterly fields are stamped away from calendar time, and stores the result in
config/bloomberg_stamping_offsets.csv. The method is described in
bloomberg_dating.py.

When to run it: once per data vintage, and only after reviewing the firms the
loader reports. analysis_v2.phase1_load_and_diagnose() re-detects the offsets on
every load and stops if they differ from the stored file, so a new data pull
cannot be shifted with stale offsets by accident.

Output columns: FIRM, k (quarters late; negative = early), r_best (correlation
at the chosen offset), r_k0 (correlation without a shift), and the correlation
at every offset tested (r_k-3 ... r_k3).

Run (from Code/RRR-FI-IM):
    ..\\venv\\Scripts\\python.exe detect_bloomberg_offsets.py
"""

# =============================================================================
# Imports
# =============================================================================
import analysis_v2 as av2
import bloomberg_dating as bd


# =============================================================================
# Main
# =============================================================================
def main():
    returns = av2.phase1c_load_monthly_returns()
    fundamentals = av2.load_fundamentals_wide()
    offsets = bd.detect_offsets(bd.market_caps_long(fundamentals, av2.FUND_DATE_COL),
                                bd.quarter_end_prices(returns))

    bd.OFFSETS_CSV.parent.mkdir(exist_ok=True)
    offsets.reset_index().to_csv(bd.OFFSETS_CSV, index=False)

    shifted = offsets[offsets['k'] != 0]
    print(f"\n  Firms checked: {len(offsets)}; firms with an offset: {len(shifted)}")
    print(f"  Firms per offset k: {shifted['k'].value_counts().sort_index().to_dict()}")
    print(f"  Wrote {bd.OFFSETS_CSV}")


if __name__ == '__main__':
    main()
