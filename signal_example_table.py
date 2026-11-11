"""signal_example_table.py -- worked example of the baseline RRR and its alternatives.

Author request 2026-09-27, extended 2026-11-11: a hypothetical firm whose new
AND returning customers spend more in one quarter only (quarter 3).

Table 3 (tab_signal_example) shows the firm's revenue and the baseline RRR over
six quarters. Because the baseline divides returning-customer revenue by the
PRIOR quarter's total revenue, the spike raises RRR in quarter 3 and lowers it
mechanically in quarter 4, although customer behavior in quarter 4 is back to
normal.

Table 4 (tab_signal_overview) extends the same firm to eight quarters and
reports every alternative measure of Methodology 5.5 next to the baseline, so
the reader sees where each one registers the spike and how long it carries it.

Revenue in the quarters before quarter 1 equals that of quarter 1; the
four-quarter measures need up to five earlier quarters.

The numbers are illustrative and deterministic. They are generated here so that
the tables, like every other table in the paper, come from a script, and
output/signal_example.json carries the values numbers_tex.py cites in the text.

Outputs
    Paper_LaTeX/tables/tab_signal_example.tex
    Paper_LaTeX/tables/tab_signal_overview.tex
    output/signal_example.json
"""

# =============================================================================
# Imports
# =============================================================================
import json
import os

# =============================================================================
# Constants
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
TABLE_DIR = os.path.join(_PROJECT, 'Paper_LaTeX', 'tables')
OUTPUT_DIR = os.path.join(_HERE, 'output')

N_PRE = 5                                   # quarters before quarter 1, identical to quarter 1
QUARTERS_EXAMPLE = [1, 2, 3, 4, 5, 6]       # Table 3
QUARTERS_OVERVIEW = [1, 2, 3, 4, 5, 6, 7, 8]  # Table 4
SPIKE_QUARTER = 3
NEW_BASE, RET_BASE = 20, 80                 # new- and returning-customer revenue in a normal quarter
NEW_SPIKE, RET_SPIKE = 40, 120              # both spike in SPIKE_QUARTER only

# (key, column head, formula in the second header row); notation defined in the note
SIGNALS = [
    ('rrr', '\\RRR{}', '$R_t/T_{t-1}$'),
    ('tq', 'Two-quarter', '$R_t/T_{t-2}$'),
    ('fq', 'Four-quarter', '$R_t/T_{t-4}$'),
    ('rcb4', 'RCB4', '$R_t/B_t$'),
    ('sorr', 'Share', '$R_t/T_t$'),
    ('ta4', 'TA4', '$R_t/\\bar{T}_t$'),
    ('ma4', 'MA4', '$\\overline{\\RRR{}}_t$'),
]


# =============================================================================
# The firm and its signals
# =============================================================================
def idx(q):
    """Position of quarter q in the extended series (quarter 1 follows N_PRE pre-quarters)."""
    return N_PRE + q - 1


def revenue_series():
    n = N_PRE + max(QUARTERS_OVERVIEW)
    new = [NEW_BASE] * n
    ret = [RET_BASE] * n
    new[idx(SPIKE_QUARTER)], ret[idx(SPIKE_QUARTER)] = NEW_SPIKE, RET_SPIKE
    tot = [a + b for a, b in zip(new, ret)]
    return new, ret, tot


def signal_values(new, ret, tot):
    """Every measure of Methodology 5.5 for each quarter of Table 4, in percent."""
    n = len(tot)
    rrr = [None] + [100.0 * ret[i] / tot[i - 1] for i in range(1, n)]
    out = {key: {} for key, _, _ in SIGNALS}
    for q in QUARTERS_OVERVIEW:
        i = idx(q)
        out['rrr'][q] = rrr[i]
        out['tq'][q] = 100.0 * ret[i] / tot[i - 2]
        out['fq'][q] = 100.0 * ret[i] / tot[i - 4]
        out['rcb4'][q] = 100.0 * ret[i] / (tot[i - 4] + new[i - 3] + new[i - 2] + new[i - 1])
        out['sorr'][q] = 100.0 * ret[i] / tot[i]
        out['ta4'][q] = 100.0 * ret[i] / (sum(tot[i - 4:i]) / 4.0)
        out['ma4'][q] = sum(rrr[i - 4:i]) / 4.0
    return out


# =============================================================================
# Tables
# =============================================================================
def write_example_table(new, ret, tot, sig):
    L = ['\\begin{threeparttable}', '\\footnotesize',
         '\\begin{tabular}{crrrr}', '\\toprule',
         'Quarter & New-customer revenue & Returning-customer revenue & '
         'Total revenue & \\RRR{} (\\%) \\\\',
         '\\midrule']
    for q in QUARTERS_EXAMPLE:
        i = idx(q)
        L.append(f'{q} & {new[i]} & {ret[i]} & {tot[i]} & {sig["rrr"][q]:.1f} \\\\')
    L += ['\\bottomrule', '\\end{tabular}',
          '\\begin{tablenotes}', '\\small',
          '\\item \\textit{Notes:} Hypothetical firm whose new and returning customers both '
          f'spend more in quarter {SPIKE_QUARTER} only; revenue before quarter 1 equals that of quarter 1.',
          '\\end{tablenotes}', '\\end{threeparttable}']
    path = os.path.join(TABLE_DIR, 'tab_signal_example.tex')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'  wrote {path}')


def write_overview_table(new, ret, tot, sig):
    n_sig = len(SIGNALS)
    L = ['\\begin{threeparttable}', '\\scriptsize',
         '\\setlength{\\tabcolsep}{4pt}',
         '\\begin{tabular}{c' + 'r' * (3 + n_sig) + '}', '\\toprule',
         ' & \\multicolumn{3}{c}{Revenue} & '
         f'\\multicolumn{{{n_sig}}}{{c}}{{Retention measure (\\%)}} \\\\',
         f'\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-{4 + n_sig}}}',
         'Quarter & New & Returning & Total & ' + ' & '.join(h for _, h, _ in SIGNALS) + ' \\\\',
         ' & $N_t$ & $R_t$ & $T_t$ & ' + ' & '.join(f for _, _, f in SIGNALS) + ' \\\\',
         '\\midrule']
    for q in QUARTERS_OVERVIEW:
        i = idx(q)
        cells = [str(q), str(new[i]), str(ret[i]), str(tot[i])]
        cells += [f'{sig[key][q]:.1f}' for key, _, _ in SIGNALS]
        L.append(' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}',
          '\\begin{tablenotes}', '\\small',
          '\\item \\textit{Notes:} $N_t$, $R_t$, and $T_t$ are new-customer, returning-customer, '
          'and total revenue in quarter $t$; $B_t = T_{t-4} + N_{t-3} + N_{t-2} + N_{t-1}$ is the '
          'rolling customer base, $\\bar{T}_t$ the mean of $T_{t-4}$ through $T_{t-1}$, and '
          '$\\overline{\\RRR{}}_t$ the mean of \\RRR{} over quarters $t-4$ through $t-1$. Share is the '
          'share of returning revenue and MA4 the four-quarter moving average. The firm is that of '
          f'Table~\\ref{{tab:signal_example}}: new and returning customers both spend more in quarter '
          f'{SPIKE_QUARTER} only, and revenue before quarter 1 equals that of quarter 1.',
          '\\end{tablenotes}', '\\end{threeparttable}']
    path = os.path.join(TABLE_DIR, 'tab_signal_overview.tex')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'  wrote {path}')


def write_json(new, ret, tot, sig):
    payload = {
        'quarters': QUARTERS_OVERVIEW,
        'spike_quarter': SPIKE_QUARTER,
        'new': [new[idx(q)] for q in QUARTERS_OVERVIEW],
        'returning': [ret[idx(q)] for q in QUARTERS_OVERVIEW],
        'total': [tot[idx(q)] for q in QUARTERS_OVERVIEW],
        'signals': {key: [round(sig[key][q], 1) for q in QUARTERS_OVERVIEW] for key, _, _ in SIGNALS},
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'signal_example.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)
    print(f'  wrote {path}')


# =============================================================================
# Main
# =============================================================================
def main():
    new, ret, tot = revenue_series()
    sig = signal_values(new, ret, tot)

    print('  quarter  new  ret  tot | ' + ' '.join(f'{key:>6}' for key, _, _ in SIGNALS))
    for q in QUARTERS_OVERVIEW:
        i = idx(q)
        print(f'  {q:>7} {new[i]:>4} {ret[i]:>4} {tot[i]:>4} | '
              + ' '.join(f'{sig[key][q]:>6.1f}' for key, _, _ in SIGNALS))

    # Guards: the baseline must read 100*RET_BASE/T_base before the spike, 100*RET_SPIKE/T_base
    # in the spike quarter and 100*RET_BASE/T_spike in the quarter after.
    t_base, t_spike = NEW_BASE + RET_BASE, NEW_SPIKE + RET_SPIKE
    assert abs(sig['rrr'][SPIKE_QUARTER - 1] - 100.0 * RET_BASE / t_base) < 1e-9
    assert abs(sig['rrr'][SPIKE_QUARTER] - 100.0 * RET_SPIKE / t_base) < 1e-9
    assert abs(sig['rrr'][SPIKE_QUARTER + 1] - 100.0 * RET_BASE / t_spike) < 1e-9
    # Every measure is back at its pre-spike value by the last quarter except the
    # moving average, which still carries the post-spike drop.
    for key, _, _ in SIGNALS:
        if key != 'ma4':
            assert abs(sig[key][QUARTERS_OVERVIEW[-1]] - sig[key][1]) < 1e-9, key

    os.makedirs(TABLE_DIR, exist_ok=True)
    write_example_table(new, ret, tot, sig)
    write_overview_table(new, ret, tot, sig)
    write_json(new, ret, tot, sig)


if __name__ == '__main__':
    main()
