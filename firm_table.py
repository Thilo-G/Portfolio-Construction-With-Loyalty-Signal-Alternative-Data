"""firm_table.py -- list of all sample firms for the appendix.

Author decision 2026-09-29: the appendix lists the names of all firms used in
the analysis (the 137 firms of the quarterly panel), with no further
description. An industry-group table built the same day was dropped at the
author's request.

Names come from the Final_Companies and ALTD Coverage sheets of the Bloomberg
workbook (cached values); NAME_OVERRIDES covers the one firm neither lists and
the two names Bloomberg writes in capitals. The list shows each name without
legal suffixes ("Inc", "Corp", "Holdings", ...), alphabetically, in three
columns.

Outputs
    Paper_LaTeX/tables/tab_firm_list.tex
    output/sample_firms.xlsx   (ticker, Bloomberg name, listed name, GICS, in sorts)
"""

import os
import re
import sys
import warnings

import openpyxl
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis_v2 as av2                 # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
TABLE_DIR = os.path.join(_PROJECT, 'Paper_LaTeX', 'tables')
BBG_WORKBOOK = os.path.join(av2.DATA_DIR, '20260904-Firms-Quarterly-Revenue-Fundamentals-BBG.xlsx')

# Spirit Airlines' post-bankruptcy ticker is in neither Bloomberg sheet.
NAME_OVERRIDES = {'SAVEQ US EQUITY': 'Spirit Airlines Inc'}
# Listed names Bloomberg writes in capitals.
LISTED_NAME_OVERRIDES = {'BOSS GR EQUITY': 'Hugo Boss', 'RBLX US EQUITY': 'Roblox'}

_SUFFIX = re.compile(r'(\s+|,\s*)(Inc|Corp|Corporation|Co|Cos|Holdings?|Group|SA|AG|SE|PLC|Ltd|'
                     r'International|Holding Co)\.?$')


def listed_name(name):
    """Firm name without 'The' and legal suffixes: 'TJX Cos Inc/The' -> 'TJX'."""
    s = name[:-4] if name.endswith('/The') else name
    prev = None
    while prev != s:
        prev = s
        s = _SUFFIX.sub('', s).rstrip(' &,')
    return s


def latex(s):
    return s.replace('&', '\\&').replace('%', '\\%')


def load_names():
    wb = openpyxl.load_workbook(BBG_WORKBOOK, read_only=True, data_only=True)
    names = {}
    for sheet, first_row in [('Final_Companies', 2), ('ALTD Coverage', 7)]:
        for row in wb[sheet].iter_rows(min_row=first_row, values_only=True):
            if row[0] and row[1]:
                names.setdefault(str(row[0]).upper(), str(row[1]).strip())
    names.update(NAME_OVERRIDES)
    return names


def build():
    ff = av2.phase1b_load_ff_factors()
    _long, df_filtered, _ = av2.phase1_load_and_diagnose(ff)
    returns = av2.phase1c_load_monthly_returns()
    panel = av2.build_holding_panel(df_filtered, returns)

    dfr = df_filtered.reset_index()
    in_sorts = set(panel['FIRM'].unique())
    # Current industry sheet first; firms it lacks (e.g. the backfilled ones)
    # take their GICS codes from the legacy file, as analysis_v2 does.
    ind = pd.concat([pd.read_excel(av2.DATA_WORKBOOK, sheet_name=av2.SHEET_INDUSTRY),
                     pd.read_excel(av2.LEGACY_INDUSTRY_FILE)])
    ind = (ind.assign(KEY=ind['ID'].astype(str).str.upper())
              .dropna(subset=['gics_sub_industry_name'])
              .drop_duplicates('KEY').set_index('KEY'))
    names = load_names()

    rows = []
    for firm in sorted(dfr['FIRM'].unique()):
        rows.append({'FIRM': firm, 'TICKER': firm.replace(' EQUITY', '').upper(),
                     'BLOOMBERG_NAME': names[firm],
                     'LISTED_NAME': LISTED_NAME_OVERRIDES.get(firm, listed_name(names[firm])),
                     'GICS_SECTOR': ind.loc[firm, 'gics_sector_name'],
                     'GICS_SUB_INDUSTRY': ind.loc[firm, 'gics_sub_industry_name'],
                     'IN_SORTS': firm in in_sorts})
    firms = pd.DataFrame(rows)
    assert len(firms) == dfr['FIRM'].nunique() and firms['LISTED_NAME'].str.len().gt(0).all()
    assert firms['LISTED_NAME'].is_unique
    return firms


def write_firm_list(firms, n_cols=3):
    """Alphabetical list in balanced columns (multicol), single-spaced, one name
    per line; it breaks across pages if it has to, unlike a tabular."""
    listed = sorted(firms['LISTED_NAME'], key=str.lower)
    L = ['{\\footnotesize\\setstretch{1}', f'\\begin{{multicols}}{{{n_cols}}}',
         '\\raggedright\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{0pt}']
    L += [latex(n) + '\\par' for n in listed]
    L += ['\\end{multicols}', '}']
    path = os.path.join(TABLE_DIR, 'tab_firm_list.tex')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'  wrote {path} ({len(listed)} firms in {n_cols} columns)')


def main():
    firms = build()
    write_firm_list(firms)
    out = os.path.join(av2.OUTPUT_DIR, 'sample_firms.xlsx')
    firms.to_excel(out, index=False)
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
