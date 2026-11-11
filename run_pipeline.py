"""run_pipeline.py -- RRR Financial Implications
===============================================

Runs every script that produces a number, table or figure in the paper, in
dependency order, with the Python interpreter that runs this file. Each step's
console output goes to _runlogs/<script>_<YYYYMMDD>.log. The pipeline stops at
the first script that fails, so no later table is built on stale inputs.

Order and why
-------------
 1. analysis_v2.py               loads the data (incl. the Bloomberg stamping check,
                                 bloomberg_dating.py), runs the main sorts, the CAPM
                                 check, Fama-MacBeth and the risk statistics
 2. save_canonical_panel.py      saves the firm-quarter panel the robustness scripts read
 3. performance_metrics.py       risk and performance battery by quartile
 4. robustness_diagnostics.py    concentration and dividend diagnostics
 5. signal_sort_robustness.py    alternative signal constructions and the growth sort
 6. estimation_variants_quartiles.py   sample halves, no COVID, equal weights, five-month lag
 7. annual_rrr_robustness.py     annual signal with annual re-formation
 8. ta4_signal_analysis.py       trailing-mean denominator (TA4)
 9. rcb4_signal_analysis.py      rolling customer base (RCB4)
10. concentration_trim_robustness.py   largest contributors and largest names removed from both legs
11. spanning_regression.py       RRR and revenue-growth spanning regressions
12. h2_asymmetry_test.py         direct test of the Q1/Q4 asymmetry
13. asymmetry_across_variants.py the robustness summary table (reads steps 5-10)
14. double_sort_size.py          size double sort (Appendix)
15. signal_ic_diagnostics.py     monthly consistency statistics
16. robustness_figures.py        robustness graph and histogram of monthly returns
17. sample_funnel.py             firm counts at each sample step
18. firm_table.py                Appendix list of sample firms
19. industry_rrr_dispersion.py   figure: RRR by sector over time
20. signal_example_table.py      methodology example tables (hypothetical numbers)
21. generate_tables_figures.py   remaining tables and figures
22. numbers_tex.py               Paper_LaTeX/numbers.tex: every result number cited in the prose

Run (from Code/RRR-FI-IM):
    ..\\venv\\Scripts\\python.exe run_pipeline.py            # whole pipeline
    ..\\venv\\Scripts\\python.exe run_pipeline.py --from 13  # resume at step 13
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================
HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / '_runlogs'

PIPELINE = [
    'analysis_v2.py',
    'save_canonical_panel.py',
    'performance_metrics.py',
    'robustness_diagnostics.py',
    'signal_sort_robustness.py',
    'estimation_variants_quartiles.py',
    'annual_rrr_robustness.py',
    'ta4_signal_analysis.py',
    'rcb4_signal_analysis.py',
    'concentration_trim_robustness.py',
    'spanning_regression.py',
    'h2_asymmetry_test.py',
    'asymmetry_across_variants.py',
    'double_sort_size.py',
    'signal_ic_diagnostics.py',
    'robustness_figures.py',
    'sample_funnel.py',
    'firm_table.py',
    'industry_rrr_dispersion.py',
    'signal_example_table.py',
    'generate_tables_figures.py',
    'numbers_tex.py',
]


# =============================================================================
# Main
# =============================================================================
def run_step(number, script, stamp):
    """Run one script, write its output to a log file, and return its exit code."""
    log_path = LOG_DIR / f'{Path(script).stem}_{stamp}.log'
    started = time.time()
    with open(log_path, 'w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, script], cwd=HERE, stdout=log,
                                stderr=subprocess.STDOUT, env=None)
    minutes = (time.time() - started) / 60
    status = 'ok' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f'  {number:2d}. {script:36s} {status:18s} {minutes:5.1f} min   log: {log_path.name}')
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run the RRR-FI paper pipeline.')
    parser.add_argument('--from', dest='start', type=int, default=1,
                        help='step number to start at (default 1)')
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.date.today().strftime('%Y%m%d')
    print(f'RRR-FI pipeline, {len(PIPELINE)} steps, starting at step {args.start}')
    for number, script in enumerate(PIPELINE, start=1):
        if number < args.start:
            continue
        if run_step(number, script, stamp) != 0:
            print(f'\nStopped at step {number}. Read the log, fix the cause, then resume with --from {number}.')
            sys.exit(1)
    print('\nPipeline complete.')


if __name__ == '__main__':
    main()
