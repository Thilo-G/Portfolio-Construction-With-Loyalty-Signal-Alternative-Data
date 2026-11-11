# Revenue Retention and Stock Returns: research code

Code for the paper "Do Firms That Retain More Revenue Than Their Peers Beat the
Market?" (Thilo Kraft). The paper measures each firm's **Revenue Retention Rate
(RRR)**, the revenue it earns in a quarter from returning customers divided by
its total revenue in the previous quarter, from credit card transaction data,
and asks whether portfolios sorted on RRR earn different risk-adjusted returns.

The data are licensed (card transactions) or proprietary (Bloomberg) and are
**not** part of this repository. The code runs only with the data workbook in
`../../Data/`.

## How to run

```
cd Code/RRR-FI-IM
..\venv\Scripts\python.exe run_pipeline.py            # all 22 steps, about 12 minutes
..\venv\Scripts\python.exe run_pipeline.py --from 13  # resume at step 13
```

Each step writes its console output to `_runlogs/<script>_<date>.log`, and the
pipeline stops at the first error. The paper is then compiled from
`../../Paper_LaTeX/main.tex`.

## The pipeline, in order

| Step | Script | What it produces |
|---|---|---|
| 1 | `analysis_v2.py` | the panel, headline sorts, CAPM check, Fama-MacBeth, risk statistics |
| 2 | `save_canonical_panel.py` | `output/canonical_panel_<date>.xlsx`, read by the robustness scripts |
| 3 | `performance_metrics.py` | return and risk statistics by quartile |
| 4 | `robustness_diagnostics.py` | concentration and dividend diagnostics |
| 5 | `signal_sort_robustness.py` | alternative signal constructions and the revenue-growth sort |
| 6 | `estimation_variants_quartiles.py` | sample halves, no COVID-19, equal weights, five-month lag |
| 7 | `annual_rrr_robustness.py` | annual signal, annual re-formation |
| 8-9 | `ta4_signal_analysis.py`, `rcb4_signal_analysis.py` | two smoothed denominators |
| 10 | `concentration_trim_robustness.py` | largest contributors and largest names removed from both legs (robustness row and Appendix) |
| 11 | `spanning_regression.py` | RRR and revenue-growth spanning regressions |
| 12 | `h2_asymmetry_test.py` | direct test of the Q1/Q4 asymmetry |
| 13 | `asymmetry_across_variants.py` | the robustness table (Appendix) and `output/robustness_summary.xlsx` |
| 14 | `double_sort_size.py` | size double sort (Appendix) |
| 15 | `signal_ic_diagnostics.py` | monthly consistency statistics |
| 16 | `robustness_figures.py` | robustness graph and histogram of monthly returns |
| 17 | `sample_funnel.py` | firm counts at each sample step |
| 18 | `firm_table.py` | list of sample firms (Appendix) |
| 19 | `industry_rrr_dispersion.py` | figure: RRR by sector over time |
| 20 | `signal_example_table.py` | worked example of the signal and its alternatives (hypothetical numbers) |
| 21 | `generate_tables_figures.py` | the remaining tables and the cumulative-return figure |
| 22 | `numbers_tex.py` | `Paper_LaTeX/numbers.tex`: every number cited in the text, after checking that every source file comes from the current run |

## Where the code lives

**Core modules** (imported by every script through `analysis_v2`, e.g. `av2.build_holding_panel`):

- `config.py`: paths, data files, timing, sample constants, and the values the pipeline must reproduce.
- `rrr_data.py`: loads the workbook and builds the firm-quarter panel, including RRR and its variants.
- `rrr_portfolios.py`: quartile assignment, the holding panel (form two months after quarter-end, hold three), value-weighted portfolio returns.
- `rrr_regressions.py`: Newey-West factor regressions (FF3, Carhart, FF5), CAPM check, Fama-MacBeth, risk statistics.
- `bloomberg_dating.py`: detects and fixes Bloomberg rows dated off calendar time (see below).

**Helpers:** `run_pipeline.py` (runs everything), `numbers_tex.py` (number macros for the text),
`detect_bloomberg_offsets.py` (see below).

**Not in the pipeline:** `correlation_screen.py`, the revenue-correlation screen that defines the
sample; its outputs (`output/correlation_screen_*.xlsx`) are kept.

**`archive/`** (kept locally, not part of the repository): retired analyses, the original 2025
script, and earlier versions of edited files.

## Conventions worth knowing

- **Timing.** RRR is measured at calendar quarter-end. Portfolios are formed two months later
  (after the 10-Q deadline) and held three months. The lag is applied once, in
  `rrr_portfolios.build_holding_panel`; signals are never shifted again.
- **Weights.** Portfolios are value-weighted on market capitalization at the formation date.
- **Signal.** The raw RRR is the headline; the industry-adjusted RRR is one robustness check.
- **Numbers in the paper.** No result is typed into the LaTeX text. `numbers_tex.py` writes one
  macro per number (`\valAlphaLS`, ...) from the saved outputs, and the tables are written by scripts.
- **Reporting.** No significance stars; exact p-values, AMA number format (`.034`, `[< .001]`).
- **Guards.** `config.py` holds the expected sample sizes and the headline alpha. Several scripts
  stop if their rebuild of the baseline differs, so a silent data change cannot pass unnoticed.

## The Bloomberg dating check

In the Bloomberg download, the quarterly fields of about 51 firms (mostly retailers whose fiscal
year ends in January) sit up to three quarters away from the calendar quarter they describe.
`bloomberg_dating.py` finds each firm's shift from market data alone: the quarter-to-quarter change
in its Bloomberg market cap is matched with the change in its monthly share price, which is dated
correctly. The card revenue, and therefore RRR, is not affected.

On every run, `rrr_data.phase1_load_and_diagnose` detects the shifts again and compares them with
`config/bloomberg_stamping_offsets.csv`:

- every shift 0: the data were fixed upstream, nothing is shifted;
- shifts equal to the stored file: they are applied;
- anything else: the run stops and lists the firms that differ.

After a new data pull, read that message, then rerun `detect_bloomberg_offsets.py` to store the new
shifts. A clean re-pull aligned to calendar quarters makes the whole step a no-op.

## Environment

Python 3.11 in `Code/venv`. `requirements.txt` pins the direct dependencies; `requirements-lock.txt`
lists the full environment. Keep pandas at 2.2.3 (pandas 3.x breaks statsmodels 0.14.4).
