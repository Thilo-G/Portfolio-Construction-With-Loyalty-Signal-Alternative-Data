"""
build_presentation_2026-07-06.py
Builds: RRR_FinancialImplication-2026-07-06a-TK.pptx
34 slides -- "How to Avoid Bad Companies with the Help of Customer Relationship Strength"
Key constraint: NO em-dashes (U+2014), NO significance stars, exact p-values in brackets.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import fitz  # pymupdf

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = r"C:\Users\thkraft\eCommerce-Goethe Dropbox\Thilo Kraft\Thilo(privat)\Privat\Research\RRR_FinancialImplication"
FIG_DIR = os.path.join(ROOT, "Paper_LaTeX", "figures")
TMP_DIR = r"C:\Users\thkraft\AppData\Local\Temp\claude\C--Users-thkraft\f1a11fc5-c5e1-45d8-aedf-887c851c079e\scratchpad"
OUTFILE = os.path.join(ROOT, "Presentations", "RRR_FinancialImplication-2026-07-07a-TK.pptx")
os.makedirs(TMP_DIR, exist_ok=True)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ── Colors ───────────────────────────────────────────────────────────────────
C_NAVY   = RGBColor(0x00, 0x33, 0x66)
C_BLUE   = RGBColor(0x00, 0x52, 0x8C)
C_BODY   = RGBColor(0x1A, 0x1A, 0x1A)
C_GRAY   = RGBColor(0x77, 0x77, 0x77)
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_RED    = RGBColor(0xFF, 0xCC, 0xCC)
C_GREEN  = RGBColor(0xD5, 0xE8, 0xD4)
C_AMBER  = RGBColor(0xFF, 0xF0, 0xCC)
C_LBLUE  = RGBColor(0xE8, 0xF0, 0xFE)
C_LGRAY  = RGBColor(0xF5, 0xF5, 0xF5)

# ── Verbatim Buffett quote ────────────────────────────────────────────────────
BUFFETT_QUOTE  = (
    "“It’s far better to buy a wonderful company at a fair price "
    "than a fair company at a wonderful price.”"
)
BUFFETT_SOURCE = "Warren Buffett, 1989 Berkshire Hathaway Shareholder Letter"

# ── Key numbers from freshly generated tables ─────────────────────────────────
# Adjusted RRR quartile portfolios (tab_portfolio_alphas_adj.tex, FF3):
#   Q1: 0.1808 (SE=0.4306)  not significant
#   Q4: -2.1811 (SE=0.6201) p<0.001    t=-3.52
#   Q1-Q4: 2.3619 (SE=0.7770)          t=3.04
# FF5: Q1-Q4: 2.5202 (SE=0.7610)       t=3.31
# AR Q1-Q4 FF3: 0.6267 (SE=0.8914)  not significant
# No-COVID: Q1-Q4 FF3: 2.1063 (SE=0.6536) t=3.22
# EW: Q1-Q4 FF3: 0.6531 (SE=0.4833) t=1.35

# ── Helpers ───────────────────────────────────────────────────────────────────

def pdf_to_png(name, dpi=150):
    src = os.path.join(FIG_DIR, name + ".pdf")
    dst = os.path.join(TMP_DIR, name + ".png")
    if not os.path.exists(src):
        print(f"  MISSING: {src}")
        return None
    doc = fitz.open(src)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = doc[0].get_pixmap(matrix=mat, alpha=False)
    pix.save(dst)
    doc.close()
    return dst


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def title_box(slide, text, top=Inches(0.22), size=26, color=None, bold=True):
    color = color or C_NAVY
    tb = slide.shapes.add_textbox(Inches(0.5), top, Inches(12.33), Inches(0.85))
    tf = tb.text_frame
    p  = tf.paragraphs[0]
    r  = p.add_run()
    r.text = text
    r.font.size  = Pt(size)
    r.font.bold  = bold
    r.font.color.rgb = color
    return tb


def divider(slide, top=Inches(1.1)):
    ln = slide.shapes.add_shape(1, Inches(0.5), top, Inches(12.33), Pt(1.5))
    ln.fill.solid()
    ln.fill.fore_color.rgb = C_BLUE
    ln.line.fill.background()


def textbox(slide, text, left, top, width, height, size=15, bold=False,
            italic=False, color=None, align=PP_ALIGN.LEFT, wrap=True):
    color = color or C_BODY
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p  = tf.paragraphs[0]
    p.alignment = align
    r  = p.add_run()
    r.text = text
    r.font.size   = Pt(size)
    r.font.bold   = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return tb


def bullet_list(slide, items, top, left=Inches(0.55), width=Inches(12.23),
                size=16, gap=5):
    tb = slide.shapes.add_textbox(left, top, width, Inches(7.0))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = Pt(gap)
        r = p.add_run()
        r.text = "•  " + item
        r.font.size = Pt(size)
        r.font.color.rgb = C_BODY
    return tb


def rect(slide, left, top, width, height, fill=C_LGRAY, line=C_BLUE):
    sh = slide.shapes.add_shape(1, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    return sh


def figure(slide, name, left, top, width, height=None):
    png = pdf_to_png(name)
    if png is None:
        tb = slide.shapes.add_textbox(left, top, width, height or Inches(3.5))
        tb.text_frame.paragraphs[0].text = f"[Missing: {name}.pdf]"
        return
    if height is not None:
        slide.shapes.add_picture(png, left, top, width=width, height=height)
    else:
        slide.shapes.add_picture(png, left, top, width=width)


def roadmap(slide, active):
    parts = ["Motivation", "Theory", "Data", "Returns", "Risk", "Robustness"]
    seg   = Inches(12.33 / len(parts))
    y     = Inches(7.08)
    for i, part in enumerate(parts):
        x  = Inches(0.5) + i * seg
        sh = slide.shapes.add_shape(1, x, y, seg - Inches(0.04), Inches(0.28))
        bg = C_NAVY if i == active else RGBColor(0xCC, 0xD8, 0xEA)
        sh.fill.solid()
        sh.fill.fore_color.rgb = bg
        sh.line.fill.background()
        p = sh.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = part
        r.font.size = Pt(8)
        r.font.color.rgb = C_WHITE if i == active else C_NAVY


def cell(tbl_cell, text, size=11, bold=False, bg=None, color=None,
         align=PP_ALIGN.CENTER):
    if bg is not None:
        tbl_cell.fill.solid()
        tbl_cell.fill.fore_color.rgb = bg
    tf = tbl_cell.text_frame
    tf.word_wrap = False
    p  = tf.paragraphs[0]
    p.alignment = align
    r  = p.add_run()
    r.text = text
    r.font.size  = Pt(size)
    r.font.bold  = bold
    r.font.color.rgb = color if color else C_BODY


def add_table(slide, data, left, top, width, height, col_widths=None,
              hdr_bg=C_NAVY, hdr_fg=C_WHITE, alt_bg=None):
    rows, cols = len(data), len(data[0])
    tbl = slide.shapes.add_table(rows, cols, left, top, width, height).table
    if col_widths:
        for j, w in enumerate(col_widths):
            tbl.columns[j].width = w
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            is_h = (i == 0 or j == 0)
            bg = hdr_bg if is_h else (alt_bg if (alt_bg and i % 2 == 0) else None)
            fg = hdr_fg if is_h else None
            cell(tbl.cell(i, j), val, size=11, bold=is_h, bg=bg, color=fg)
    return tbl


# ── Slide 01: Title ───────────────────────────────────────────────────────────
def s01_title(prs):
    sl = blank(prs)
    bg = sl.shapes.add_shape(1, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_NAVY
    bg.line.fill.background()

    tb = sl.shapes.add_textbox(Inches(0.7), Inches(1.5), Inches(11.93), Inches(2.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    r1 = p1.add_run()
    r1.text = "How to Avoid Bad Companies"
    r1.font.size = Pt(38); r1.font.bold = True; r1.font.color.rgb = C_WHITE

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "with the Help of Customer Relationship Strength"
    r2.font.size = Pt(32); r2.font.bold = True; r2.font.color.rgb = C_WHITE

    sub = sl.shapes.add_textbox(Inches(0.7), Inches(3.8), Inches(11.93), Inches(0.7))
    ps = sub.text_frame.paragraphs[0]
    ps.alignment = PP_ALIGN.CENTER
    rs = ps.add_run()
    rs.text = "Revenue Retention Rates & Stock Prices: High Returns, Low Risk"
    rs.font.size = Pt(20); rs.font.italic = True
    rs.font.color.rgb = RGBColor(0xAA, 0xCC, 0xEE)

    auth = sl.shapes.add_textbox(Inches(0.7), Inches(4.75), Inches(11.93), Inches(0.5))
    pa = auth.text_frame.paragraphs[0]
    pa.alignment = PP_ALIGN.CENTER
    ra = pa.add_run()
    ra.text = "Thilo Kraft & Bernd Skiera   |   Goethe University Frankfurt   |   July 2026"
    ra.font.size = Pt(16); ra.font.color.rgb = RGBColor(0x99, 0xBB, 0xDD)


# ── Slide 02: Buffett ─────────────────────────────────────────────────────────
def s02_buffett(prs):
    sl = blank(prs)
    title_box(sl, "The Buffett Problem")
    divider(sl)

    qb = sl.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.33), Inches(3.2))
    tf = qb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = BUFFETT_QUOTE
    r.font.size = Pt(27); r.font.italic = True; r.font.color.rgb = C_NAVY

    at = sl.shapes.add_textbox(Inches(1.0), Inches(4.85), Inches(11.33), Inches(0.45))
    pa = at.text_frame.paragraphs[0]
    pa.alignment = PP_ALIGN.RIGHT
    ra = pa.add_run()
    ra.text = BUFFETT_SOURCE
    ra.font.size = Pt(13); ra.font.color.rgb = C_GRAY

    br = sl.shapes.add_textbox(Inches(0.5), Inches(5.55), Inches(12.33), Inches(0.6))
    pb = br.text_frame.paragraphs[0]
    pb.alignment = PP_ALIGN.CENTER
    rb = pb.add_run()
    rb.text = ("The harder question: how do you spot the company that only looks wonderful "
               "-- but whose customer base is quietly eroding underneath?")
    rb.font.size = Pt(19); rb.font.bold = True; rb.font.color.rgb = C_BLUE

    roadmap(sl, 0)


# ── Slide 03: What makes a firm's customer relationships weak ─────────────────
def s03_weak_firms(prs):
    sl = blank(prs)
    title_box(sl, "What Makes a Firm's Customer Relationships \"Weak\"?")
    divider(sl)

    bullet_list(sl, [
        "Firms with weak customer relationships lose existing customers faster than they retain them.",
        "To maintain headline growth, they replace churned customers with newly acquired ones, masking the deterioration.",
        "Revenue looks unchanged from the outside. The customer base erodes underneath.",
    ], Inches(1.28), size=17, gap=4)

    for i, (label, clr_fill, clr_line, note1, note2) in enumerate([
        ("Firm A: Strong Customer Relationships",
         RGBColor(0xD5, 0xE8, 0xD4), RGBColor(0x82, 0xB3, 0x66),
         "High RRR | Large stable returning-customer base",
         "Small acquisition wedge needed to grow"),
        ("Firm B: Weak Customer Relationships",
         RGBColor(0xF8, 0xD7, 0xDA), RGBColor(0xCC, 0x44, 0x44),
         "Low RRR | Shrinking returning-customer pool",
         "Large acquisition wedge filling the gap"),
    ]):
        x = Inches(0.7) + i * Inches(6.4)
        sh = rect(sl, x, Inches(3.7), Inches(5.8), Inches(2.6), fill=clr_fill, line=clr_line)
        tf = sh.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].text = label
        tf.paragraphs[0].font.size = Pt(13); tf.paragraphs[0].font.bold = True
        for txt in [note1, note2, "Revenue growth: +10%"]:
            p = tf.add_paragraph(); p.text = txt; p.font.size = Pt(12)
        tf.paragraphs[3].font.bold = True

    textbox(sl, "Financial statements cannot tell you which one you own.",
            Inches(0.5), Inches(6.5), Inches(12.33), Inches(0.4),
            size=14, italic=True, color=C_GRAY, align=PP_ALIGN.CENTER)
    roadmap(sl, 0)


# ── Slide 04: Revenue blindspot ───────────────────────────────────────────────
def s04_blindspot(prs):
    sl = blank(prs)
    title_box(sl, "The Revenue Blindspot")
    divider(sl)

    for i, (label, rrr, desc) in enumerate([
        ("Firm A", "RRR = 0.95", "Retention-driven growth"),
        ("Firm B", "RRR = 0.70", "Acquisition-compensating growth"),
    ]):
        x = Inches(0.7) + i * Inches(6.4)
        clr = (RGBColor(0xD5, 0xE8, 0xD4), RGBColor(0x82, 0xB3, 0x66)) if i == 0 \
              else (RGBColor(0xF8, 0xD7, 0xDA), RGBColor(0xCC, 0x44, 0x44))
        sh = rect(sl, x, Inches(1.5), Inches(5.8), Inches(4.0), fill=clr[0], line=clr[1])
        tf = sh.text_frame; tf.word_wrap = True
        tf.paragraphs[0].text = label
        tf.paragraphs[0].font.size = Pt(20); tf.paragraphs[0].font.bold = True
        for txt in [rrr, desc, "", "Reported revenue growth: +10%", "Income statement: looks identical"]:
            p = tf.add_paragraph(); p.text = txt; p.font.size = Pt(14)
        tf.paragraphs[4].font.bold = True

    qb = sl.shapes.add_textbox(Inches(0.5), Inches(5.7), Inches(12.33), Inches(0.75))
    pq = qb.text_frame.paragraphs[0]
    pq.alignment = PP_ALIGN.CENTER
    rq = pq.add_run()
    rq.text = "Same income statement. Different customer bases. Should they trade at the same price?"
    rq.font.size = Pt(20); rq.font.bold = True; rq.font.color.rgb = C_NAVY
    roadmap(sl, 0)


# ── Slide 05: Data gap ────────────────────────────────────────────────────────
def s05_data_gap(prs):
    sl = blank(prs)
    title_box(sl, "The Data Gap")
    divider(sl)

    bullet_list(sl, [
        "Income statements carry one revenue line: no split between returning-customer revenue and new-customer revenue. (Bonacchi et al. 2015)",
        "Card transaction data makes the split observable at the firm-quarter level.",
        "This paper constructs firm-quarter RRR and AR for 124 publicly traded U.S. firms across 87 monthly return periods.",
    ], Inches(1.28), size=17, gap=5)

    sh = rect(sl, Inches(3.2), Inches(4.5), Inches(7.0), Inches(1.85),
              fill=C_LBLUE, line=C_BLUE)
    tf = sh.text_frame; tf.word_wrap = True
    tf.paragraphs[0].text = "Income Statement:   Revenue = $X"
    tf.paragraphs[0].font.size = Pt(14); tf.paragraphs[0].font.bold = True
    p2 = tf.add_paragraph()
    p2.text = "    Returning customers: ???   |   New customers: ???"
    p2.font.size = Pt(13); p2.font.italic = True; p2.font.color.rgb = C_GRAY
    p3 = tf.add_paragraph()
    p3.text = "Card data resolves the ???   →   RRR + AR visible"
    p3.font.size = Pt(14); p3.font.color.rgb = C_BLUE; p3.font.bold = True

    roadmap(sl, 0)


# ── Slide 06: Research question ───────────────────────────────────────────────
def s06_rq(prs):
    sl = blank(prs)
    title_box(sl, "Research Question")
    divider(sl)

    sh = rect(sl, Inches(0.5), Inches(1.3), Inches(12.33), Inches(1.25),
              fill=C_LBLUE, line=C_BLUE)
    tf = sh.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = ("Does the composition of a firm's revenue growth -- specifically, "
              "how much comes from retaining existing customers -- predict future stock returns?")
    r.font.size = Pt(17); r.font.bold = True; r.font.color.rgb = C_NAVY

    textbox(sl, "Preview of main result (industry-adjusted portfolios, FF3):",
            Inches(0.5), Inches(2.75), Inches(12.33), Inches(0.4),
            size=13, color=C_GRAY, align=PP_ALIGN.CENTER)

    for i, (label, alpha, pval, note, fill, line) in enumerate([
        ("Q1 (Strong Customer Relationships)", "FF3 alpha = +0.18%/month",
         "Not significant  [p = 0.674]", "Fairly priced",
         RGBColor(0xD5, 0xE8, 0xD4), RGBColor(0x82, 0xB3, 0x66)),
        ("Q4 (Weak Customer Relationships)", "FF3 alpha = -2.18%/month",
         "[p < 0.001]", "Overpriced",
         RGBColor(0xF8, 0xD7, 0xDA), RGBColor(0xCC, 0x44, 0x44)),
    ]):
        x = Inches(1.0) + i * Inches(6.6)
        sh2 = rect(sl, x, Inches(3.2), Inches(5.3), Inches(2.05), fill=fill, line=line)
        tf2 = sh2.text_frame; tf2.word_wrap = True
        tf2.paragraphs[0].text = label
        tf2.paragraphs[0].font.size = Pt(12); tf2.paragraphs[0].font.bold = True
        p_a = tf2.add_paragraph()
        p_a.text = alpha
        p_a.font.size = Pt(18); p_a.font.bold = True
        p_p = tf2.add_paragraph(); p_p.text = pval; p_p.font.size = Pt(13)
        p_n = tf2.add_paragraph(); p_n.text = note
        p_n.font.size = Pt(13); p_n.font.italic = True

    lsb = sl.shapes.add_textbox(Inches(0.5), Inches(5.45), Inches(12.33), Inches(0.5))
    pls = lsb.text_frame.paragraphs[0]
    pls.alignment = PP_ALIGN.CENTER
    rls = pls.add_run()
    rls.text = ("Long-short: 2.36%/month FF3 [p = 0.003], 2.52%/month FF5 [p = 0.001]"
                "   |   N = 87 months   |   t = 3.04 (FF3), 3.31 (FF5)")
    rls.font.size = Pt(15); rls.font.bold = True; rls.font.color.rgb = C_NAVY

    textbox(sl, "The market misprices firms with weak customer relationships, not those with strong ones.",
            Inches(0.5), Inches(6.15), Inches(12.33), Inches(0.45),
            size=14, italic=True, color=C_BLUE, align=PP_ALIGN.CENTER)
    roadmap(sl, 0)


# ── Slide 07: Literature map ──────────────────────────────────────────────────
def s07_lit_map(prs):
    sl = blank(prs)
    title_box(sl, "Literature Map")
    divider(sl)

    quads = [
        (Inches(0.5),  Inches(1.3),  "1. Marketing Metrics & Firm Value",
         "Gupta et al. (2004); Gruca & Rego (2005); Aksoy et al. (2008)"),
        (Inches(6.9),  Inches(1.3),  "2. Mispricing of Intangible Information",
         "Sloan (1996); Jacobson & Mizik (2009); Malshe, Colicev & Mittal (2020)"),
        (Inches(0.5),  Inches(4.0),  "3. CBCV & Customer-Base Metrics",
         "McCarthy & Fader (2018); Bonacchi et al. (2015)"),
        (Inches(6.9),  Inches(4.0),  "4. Alternative Data & Stock Returns",
         "Gupta, Leung & Roscovan (2022)"),
    ]
    for x, y, ttl, cites in quads:
        sh = rect(sl, x, y, Inches(6.0), Inches(2.4), fill=C_LBLUE, line=C_BLUE)
        tf = sh.text_frame; tf.word_wrap = True
        tf.paragraphs[0].text = ttl
        tf.paragraphs[0].font.size = Pt(13); tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = C_NAVY
        p2 = tf.add_paragraph(); p2.text = cites
        p2.font.size = Pt(11); p2.font.color.rgb = C_GRAY

    mk = rect(sl, Inches(5.3), Inches(2.85), Inches(2.7), Inches(1.3),
              fill=C_NAVY, line=C_NAVY)
    mk.line.fill.background()
    tm = mk.text_frame
    tm.paragraphs[0].text = "This paper"
    tm.paragraphs[0].font.size = Pt(15); tm.paragraphs[0].font.bold = True
    tm.paragraphs[0].font.color.rgb = C_WHITE
    tm.paragraphs[0].alignment = PP_ALIGN.CENTER
    p2 = tm.add_paragraph(); p2.text = "streams 2 + 3 + 4"
    p2.font.size = Pt(11); p2.font.color.rgb = RGBColor(0xCC, 0xDD, 0xEE)
    p2.alignment = PP_ALIGN.CENTER

    roadmap(sl, 1)


# ── Slide 08: Accrual anomaly parallel ───────────────────────────────────────
def s08_accrual(prs):
    sl = blank(prs)
    title_box(sl, "The Accrual Anomaly Parallel (Sloan 1996)")
    divider(sl)

    headers = ["Sloan (1996): Accrual Anomaly", "This Paper: RRR Signal"]
    rows = [
        ("Decomposition",
         "Earnings = Cash flows (persistent) + Accruals (transitory)",
         "Revenue growth = Retention (persistent) + Acquisition (transitory)"),
        ("Market failure",
         "Investors overweight accruals relative to persistent cash flows",
         "Investors overweight acquisition growth relative to retention growth"),
        ("Empirical test",
         "Accrual sort earns abnormal returns",
         "RRR sort earns abnormal returns; AR sort does not"),
        ("Mechanism",
         "Accruals reverse; earnings are overstated",
         "Acquisition revenue is less persistent; growth quality is overstated"),
    ]

    cw = Inches(6.0)
    for j, h in enumerate(headers):
        hb = rect(sl, Inches(0.5) + j * Inches(6.4), Inches(1.3), cw, Inches(0.5),
                  fill=C_NAVY, line=C_NAVY)
        hb.line.fill.background()
        hb.text_frame.paragraphs[0].text = h
        hb.text_frame.paragraphs[0].font.size = Pt(13)
        hb.text_frame.paragraphs[0].font.bold = True
        hb.text_frame.paragraphs[0].font.color.rgb = C_WHITE
        hb.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    for i, (_, left_t, right_t) in enumerate(rows):
        y = Inches(1.9) + i * Inches(1.15)
        for j, (txt, bg) in enumerate([(left_t, C_LGRAY), (right_t, C_LBLUE)]):
            sh = rect(sl, Inches(0.5) + j * Inches(6.4), y, cw, Inches(1.05),
                      fill=bg, line=RGBColor(0xCC, 0xCC, 0xCC))
            sh.text_frame.word_wrap = True
            sh.text_frame.paragraphs[0].text = txt
            sh.text_frame.paragraphs[0].font.size = Pt(11)

    textbox(sl, "The template is the same. The domain is new.   (Sloan 1996; Lev & Nissim 2006)",
            Inches(0.5), Inches(6.55), Inches(12.33), Inches(0.4),
            size=14, bold=True, color=C_BLUE, align=PP_ALIGN.CENTER)
    roadmap(sl, 1)


# ── Slide 09: Decomposition identity ─────────────────────────────────────────
def s09_decomp(prs):
    sl = blank(prs)
    title_box(sl, "Revenue Decomposition Identity")
    divider(sl)

    fb = sl.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.33), Inches(1.1))
    pf = fb.text_frame.paragraphs[0]
    pf.alignment = PP_ALIGN.CENTER
    rf = pf.add_run()
    rf.text = "RG  =  SoNR × AR  +  RRR  −  1"
    rf.font.size = Pt(36); rf.font.bold = True; rf.font.color.rgb = C_NAVY

    defs = [
        ("RG",    "Revenue Growth = (Revenue_t / Revenue_{t-1}) - 1"),
        ("RRR",   "Revenue Retention Rate = returning-customer revenue_t / Revenue_{t-1}   [persistent component]"),
        ("AR",    "Acquisition Rate = new-customer revenue_t / Revenue_{t-1}   [transitory component]"),
        ("SoNR",  "Share of New Revenue = scaling factor on the acquisition term"),
    ]
    for i, (sym, defn) in enumerate(defs):
        y = Inches(2.9) + i * Inches(0.72)
        tb = sl.shapes.add_textbox(Inches(0.8), y, Inches(11.7), Inches(0.6))
        p = tb.text_frame.paragraphs[0]
        r1 = p.add_run(); r1.text = sym + ":  "
        r1.font.bold = True; r1.font.size = Pt(15); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = defn
        r2.font.size = Pt(14); r2.font.color.rgb = C_BODY

    textbox(sl, ("This is an accounting identity, not a theory. "
                 "Theory enters via what we know about the persistence properties of each component. "
                 "Note: RRR is NOT bounded by 1 (SaaS NRR > 100% is canonical)."),
            Inches(0.5), Inches(6.0), Inches(12.33), Inches(0.65),
            size=12, italic=True, color=C_GRAY, align=PP_ALIGN.CENTER, wrap=True)
    roadmap(sl, 1)


# ── Slide 10: Persistence argument ───────────────────────────────────────────
def s10_persistence(prs):
    sl = blank(prs)
    title_box(sl, "Why Retention Revenue Is Different: The Persistence Argument")
    divider(sl)

    bullet_list(sl, [
        "Retention revenue is recurrent: the customer already exists next quarter with positive probability.",
        "Acquisition revenue is non-recurrent at the point of capture: the new customer may or may not return.",
        "A firm growing via retention has more persistent future cash flows than one growing via acquisition.",
        "If investors anchor on headline growth and miss this distinction, they will overprice acquisition-driven growth.",
    ], Inches(1.28), size=17, gap=4)

    sh = rect(sl, Inches(0.7), Inches(5.05), Inches(11.93), Inches(1.2),
              fill=C_LBLUE, line=C_BLUE)
    tf = sh.text_frame; tf.word_wrap = True
    tf.paragraphs[0].text = "Risk prediction (tested in Part 5):"
    tf.paragraphs[0].font.size = Pt(13); tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = C_NAVY
    p2 = tf.add_paragraph()
    p2.text = ("A more stable, recurrent cash-flow base implies lower variance, smaller drawdowns, "
               "and lower downside exposure. We predict high-RRR firms are safer, not riskier. "
               "This is the opposite of a risk-based explanation. We test it directly.")
    p2.font.size = Pt(13)

    textbox(sl, "Reinartz & Srivastava (2005); Villanueva et al. (2008); Gruca & Rego (2005)",
            Inches(0.5), Inches(6.4), Inches(12.33), Inches(0.4),
            size=11, color=C_GRAY)
    roadmap(sl, 1)


# ── Slide 11: Skeptic's challenge ─────────────────────────────────────────────
def s11_skeptic(prs):
    sl = blank(prs)
    title_box(sl, "The Skeptic's Challenge: What About Acquisition Rate?")
    divider(sl)

    bullet_list(sl, [
        "Natural alternative hypothesis: low-RRR firms simply have fast new-customer growth, and fast-growth stocks are well-known to be overpriced. Or: low-RRR firms are just slow-growers, and the effect is a value premium in disguise.",
        "If either alternative were correct, sorting on AR alone would produce the same return spread as sorting on RRR.",
        "We test AR directly (H2). If AR earns no independent alpha, the growth-level story fails -- only the composition story survives.",
        "Fama-MacBeth regressions: if both RRR and AR are included jointly, does AR survive or is it subsumed by RRR?",
    ], Inches(1.28), size=17, gap=4)

    textbox(sl, "Result on Slide 22: AR earns no significant alpha. The composition story survives.",
            Inches(0.5), Inches(6.2), Inches(12.33), Inches(0.5),
            size=15, bold=True, color=C_BLUE, align=PP_ALIGN.CENTER)
    roadmap(sl, 1)


# ── Slide 12: Is this a risk story? ──────────────────────────────────────────
def s12_risk_test(prs):
    sl = blank(prs)
    title_box(sl, "Is This a Risk Story?")
    divider(sl)

    textbox(sl, "Finance readers immediately ask: higher returns must mean higher risk somewhere.",
            Inches(0.5), Inches(1.28), Inches(12.33), Inches(0.45), size=16)

    cells_data = [
        ("Return premium + Higher risk",  "Risk compensation (anomaly explained)",  C_LGRAY),
        ("Return premium + LOWER risk",   "Mispricing (overpricing confirmed)",      C_RED),
        ("No premium + Higher risk",      "Risk-return tradeoff, no anomaly",        C_LGRAY),
        ("No premium + LOWER risk",       "Anomaly without return premium (unusual)", C_LGRAY),
    ]
    xs = [Inches(0.8), Inches(6.8)]
    ys = [Inches(2.1), Inches(3.75)]
    for idx, (label, desc, bg) in enumerate(cells_data):
        r, c = idx // 2, idx % 2
        bx = rect(sl, xs[c], ys[r], Inches(5.6), Inches(1.45), fill=bg,
                  line=RGBColor(0x22, 0x77, 0x22) if idx == 1 else C_BLUE)
        tf = bx.text_frame; tf.word_wrap = True
        tf.paragraphs[0].text = desc
        tf.paragraphs[0].font.size = Pt(14); tf.paragraphs[0].font.bold = True
        if idx == 1:
            tf.paragraphs[0].font.color.rgb = RGBColor(0x00, 0x66, 0x00)
        p2 = tf.add_paragraph(); p2.text = "(" + label + ")"
        p2.font.size = Pt(10); p2.font.color.rgb = C_GRAY

    textbox(sl, ("Our prediction from Slide 10: high-RRR firms are SAFER. "
                 "If returns are higher AND risk is lower, risk compensation is ruled out."),
            Inches(0.5), Inches(5.45), Inches(12.33), Inches(0.6),
            size=14, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER, wrap=True)

    textbox(sl, "This makes the risk section an ex ante prediction, not a post-hoc defense.",
            Inches(0.5), Inches(6.2), Inches(12.33), Inches(0.4),
            size=13, italic=True, color=C_BLUE, align=PP_ALIGN.CENTER)
    roadmap(sl, 1)


# ── Slide 13: Hypotheses ──────────────────────────────────────────────────────
def s13_hyps(prs):
    sl = blank(prs)
    title_box(sl, "Four Hypotheses")
    divider(sl)

    hyps = [
        ("H1", "Low-RRR firms earn large negative abnormal returns; high-RRR firms are fairly priced (alpha near zero). Mispricing concentrates on the short side, consistent with limits-to-arbitrage: overpriced stocks are harder to correct than underpriced ones."),
        ("H2", "AR does not independently predict returns. (The growth-level alternative fails; only the composition story survives.)"),
        ("H3", "The return premium is accompanied by lower risk in the long leg. (Rules out risk compensation as an explanation.)"),
        ("H4", "Among firms with identical headline revenue growth, higher-RRR firms earn higher returns. Composition, not level, drives the signal."),
    ]
    for i, (hnum, htext) in enumerate(hyps):
        y = Inches(1.35) + i * Inches(1.3)
        sh = rect(sl, Inches(0.5), y, Inches(12.33), Inches(1.15), fill=C_LBLUE, line=C_BLUE)
        tf = sh.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        r1 = p.add_run(); r1.text = hnum + ":  "
        r1.font.bold = True; r1.font.size = Pt(16); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = htext
        r2.font.size = Pt(14); r2.font.color.rgb = C_BODY

    roadmap(sl, 1)


# ── Slide 14: Data sources ────────────────────────────────────────────────────
def s14_data(prs):
    sl = blank(prs)
    title_box(sl, "Data Sources")
    divider(sl)

    sources = [
        ("Card transaction data",
         "Firm-quarter RRR and AR for 124 public firms. Signal lagged one quarter "
         "(quarter t-1 predicts month t return) to avoid look-ahead bias."),
        ("Bloomberg",
         "Monthly price-only returns (NO DIVIDENDS -- stated limitation), market cap, "
         "book-to-market, profit margin. Sample: 2017-2024, 87 monthly return periods."),
        ("Ken French library",
         "FF3, Carhart 4-factor, FF5 monthly factors. Fama-MacBeth: 30 quarterly "
         "cross-sections, average N = 112 firms per quarter."),
    ]
    for i, (src, desc) in enumerate(sources):
        y = Inches(1.4) + i * Inches(1.65)
        sh = rect(sl, Inches(0.5), y, Inches(12.33), Inches(1.45), fill=C_LGRAY, line=C_BLUE)
        tf = sh.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        r1 = p.add_run(); r1.text = src + ":  "
        r1.font.bold = True; r1.font.size = Pt(15); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = desc
        r2.font.size = Pt(13); r2.font.color.rgb = C_BODY

    textbox(sl, "Cite: Gupta, Leung & Roscovan (2022).",
            Inches(0.5), Inches(6.5), Inches(8.0), Inches(0.4), size=11, color=C_GRAY)
    roadmap(sl, 2)


# ── Slide 15: Sample ──────────────────────────────────────────────────────────
def s15_sample(prs):
    sl = blank(prs)
    title_box(sl, "Sample Composition")
    divider(sl)

    textbox(sl, "124 firms   |   4 sectors   |   2017-2024   |   87 monthly return periods",
            Inches(0.5), Inches(1.25), Inches(12.33), Inches(0.45),
            size=16, bold=True, align=PP_ALIGN.CENTER)

    figure(sl, "sector_bars",      Inches(0.3),  Inches(1.85), Inches(6.0))
    figure(sl, "firms_per_quarter", Inches(6.8), Inches(1.85), Inches(6.2))

    textbox(sl, ("Note: Consumer Discretionary accounts for 90 of 124 firms (72.6%). "
                 "Results reported within-sector via industry-time adjustment."),
            Inches(0.5), Inches(6.55), Inches(12.33), Inches(0.5),
            size=11, italic=True, color=C_GRAY, align=PP_ALIGN.CENTER)
    roadmap(sl, 2)


# ── Slide 16: Revenue growth by RRR quartile ─────────────────────────────────
def s16_rev_growth(prs):
    sl = blank(prs)
    title_box(sl, "Revenue Growth by RRR Quartile")
    divider(sl)

    textbox(sl, "Pre-empting a potential objection: are low-RRR firms simply slow-growth firms?",
            Inches(0.5), Inches(1.28), Inches(12.33), Inches(0.4),
            size=14, italic=True, color=C_GRAY, align=PP_ALIGN.CENTER)

    figure(sl, "revenue_growth_by_rrr_quartile", Inches(1.3), Inches(1.85), Inches(10.73))

    textbox(sl, "The signal captures the COMPOSITION of revenue growth, not its level.",
            Inches(0.5), Inches(6.6), Inches(12.33), Inches(0.45),
            size=15, bold=True, color=C_BLUE, align=PP_ALIGN.CENTER)
    roadmap(sl, 2)


# ── Slide 17: Transition matrix ───────────────────────────────────────────────
def s17_transition(prs):
    sl = blank(prs)
    title_box(sl, "RRR Stability: Transition Matrix")
    divider(sl)

    figure(sl, "transition_heatmap", Inches(1.0), Inches(1.35), Inches(11.33))

    textbox(sl, ("High-RRR firms (T1): 46.6% remain in T1 quarter-to-quarter. "
                 "Low-RRR firms (T3): 62.7% remain in T3. Only 14.9% jump from T3 to T1. "
                 "Stable signal = low strategy turnover. "
                 "[Note: transition matrix uses terciles (T1-T3) for granularity; portfolio sorts use quartiles (Q1-Q4).]"),
            Inches(0.5), Inches(6.35), Inches(12.33), Inches(0.7),
            size=13, color=C_BODY, align=PP_ALIGN.CENTER, wrap=True)
    roadmap(sl, 2)


# ── Slide 18: Cumulative returns ──────────────────────────────────────────────
def s18_cumret(prs):
    sl = blank(prs)
    title_box(sl, "Cumulative Returns by Adjusted RRR Quartile")
    divider(sl)

    figure(sl, "cumret_rrr_adj_q4_vw", Inches(0.3), Inches(1.3), Inches(12.73))

    textbox(sl, ("Value-weighted portfolios, sorted quarterly on industry-adjusted RRR, 2017-2024. "
                 "Q4 (low RRR, weak customer relationships) loses cumulatively over the full period."),
            Inches(0.5), Inches(6.65), Inches(12.33), Inches(0.5),
            size=13, color=C_GRAY, align=PP_ALIGN.CENTER, wrap=True)
    roadmap(sl, 3)


# ── Slide 19: FF3 alpha table ─────────────────────────────────────────────────
def s19_ff3(prs):
    sl = blank(prs)
    title_box(sl, "FF3 Alpha: Asymmetry Confirmed (Industry-Adjusted Portfolios)")
    divider(sl)

    data = [
        ["",               "Q1\n(High RRR)", "Q2",         "Q3",         "Q4\n(Low RRR)", "Q1-Q4 L-S"],
        ["α (%/mo)",  "+0.1808",        "-0.5859",    "-1.9139",    "-2.1811",       "+2.3619"],
        ["SE",             "(0.4306)",        "(0.3152)",   "(0.5496)",   "(0.6201)",      "(0.7770)"],
        ["p-value",        "[0.674]",         "[0.064]",    "[0.001]",    "[<0.001]",      "[0.003]"],
        ["N (months)",     "87",              "87",         "87",         "87",            "87"],
    ]
    tbl = sl.shapes.add_table(5, 6,
                               Inches(0.35), Inches(1.42),
                               Inches(12.63), Inches(3.3)).table
    for j, w in enumerate([Inches(1.85), Inches(1.85), Inches(1.65), Inches(1.65), Inches(2.15), Inches(2.0)]):
        tbl.columns[j].width = w

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            is_h = (i == 0 or j == 0)
            is_q4 = (j == 4 and i == 1)
            bg = C_NAVY if is_h else (C_RED if is_q4 else None)
            fg = C_WHITE if is_h else (C_BODY if not is_q4 else None)
            cell(tbl.cell(i, j), val, size=11, bold=is_h, bg=bg, color=fg)

    sh = rect(sl, Inches(0.35), Inches(4.9), Inches(12.63), Inches(1.3),
              fill=RGBColor(0xFF, 0xF0, 0xF0), line=RGBColor(0xCC, 0x44, 0x44))
    tf = sh.text_frame; tf.word_wrap = True
    tf.paragraphs[0].text = (
        "Firms with weak customer relationships (Q4) drive the spread: "
        "alpha = -2.18%/month [p < 0.001]."
    )
    tf.paragraphs[0].font.size = Pt(14); tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = C_NAVY
    p2 = tf.add_paragraph()
    p2.text = ("Firms with strong customer relationships (Q1) are fairly priced: "
               "+0.18%/month, not significant [p = 0.674].")
    p2.font.size = Pt(14); p2.font.color.rgb = C_BODY

    textbox(sl, "Comparison: raw (unadjusted) long-short FF3 alpha = 2.15%/month [p = 0.008].",
            Inches(0.35), Inches(6.35), Inches(12.63), Inches(0.35),
            size=11, color=C_GRAY)
    roadmap(sl, 3)


# ── Slide 20: FF5 alpha table ─────────────────────────────────────────────────
def s20_ff5(prs):
    sl = blank(prs)
    title_box(sl, "FF5 Alpha: Profitability & Investment Factors Do Not Absorb the Signal")
    divider(sl)

    data = [
        ["",               "Q1\n(High RRR)", "Q2",         "Q3",         "Q4\n(Low RRR)", "Q1-Q4 L-S"],
        ["α (%/mo)",  "+0.4168",        "-0.5438",    "-1.7595",    "-2.1035",       "+2.5202"],
        ["SE",             "(0.4097)",        "(0.2957)",   "(0.4956)",   "(0.6197)",      "(0.7610)"],
        ["p-value",        "[0.310]",         "[0.066]",    "[0.001]",    "[0.001]",       "[0.001]"],
        ["N (months)",     "87",              "87",         "87",         "87",            "87"],
    ]
    tbl = sl.shapes.add_table(5, 6,
                               Inches(0.35), Inches(1.42),
                               Inches(12.63), Inches(3.3)).table
    for j, w in enumerate([Inches(1.85), Inches(1.85), Inches(1.65), Inches(1.65), Inches(2.15), Inches(2.0)]):
        tbl.columns[j].width = w

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            is_h = (i == 0 or j == 0)
            is_q4 = (j == 4 and i == 1)
            bg = C_NAVY if is_h else (C_RED if is_q4 else None)
            fg = C_WHITE if is_h else None
            cell(tbl.cell(i, j), val, size=11, bold=is_h, bg=bg, color=fg)

    textbox(sl, ("Alpha grows from FF3 to FF5: 2.36% to 2.52%/month (t = 3.04 to 3.31). "
                 "Adding profitability (RMW) and investment (CMA) factors does not absorb the signal."),
            Inches(0.35), Inches(4.9), Inches(12.63), Inches(0.7),
            size=14, color=C_BODY, wrap=True)
    roadmap(sl, 3)


# ── Slide 21: Alpha bars ───────────────────────────────────────────────────────
def s21_alpha_bars(prs):
    sl = blank(prs)
    title_box(sl, "Alpha Across Factor Models")
    divider(sl)

    figure(sl, "alpha_bars", Inches(0.4), Inches(1.35), Inches(12.53))

    textbox(sl, ("Long-short alpha is large, negative, and consistent across FF3, Carhart (4F), and FF5. "
                 "The Q4 (low-RRR) bar drives the spread in all specifications."),
            Inches(0.5), Inches(6.65), Inches(12.33), Inches(0.5),
            size=14, color=C_BODY, align=PP_ALIGN.CENTER, wrap=True)
    roadmap(sl, 3)


# ── Slide 22: AR result ───────────────────────────────────────────────────────
def s22_ar(prs):
    sl = blank(prs)
    title_box(sl, "Does AR Explain It? (Test Motivated on Slide 11)")
    divider(sl)

    textbox(sl, "We set up this test in the theory section. Here is the result.",
            Inches(0.5), Inches(1.28), Inches(12.33), Inches(0.4),
            size=14, italic=True, color=C_GRAY)

    figure(sl, "cumret_ar_adj_q4_vw", Inches(0.3), Inches(1.85), Inches(6.3))

    data = [
        ["Model",    "Q1-Q4 alpha", "SE",         "p-value", "t-stat"],
        ["FF3",      "+0.6267",     "(0.8914)",   "[0.482]", "0.70"],
        ["FF5",      "+0.5736",     "(0.8759)",   "[0.513]", "0.65"],
    ]
    tbl = sl.shapes.add_table(3, 5,
                               Inches(6.9), Inches(2.05),
                               Inches(6.1), Inches(1.3)).table
    for j, w in enumerate([Inches(1.1), Inches(1.4), Inches(1.3), Inches(1.2), Inches(1.1)]):
        tbl.columns[j].width = w
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            is_h = (i == 0 or j == 0)
            cell(tbl.cell(i, j), val, size=11, bold=is_h,
                 bg=C_NAVY if is_h else None, color=C_WHITE if is_h else None)

    tb = sl.shapes.add_textbox(Inches(6.9), Inches(3.6), Inches(6.1), Inches(1.8))
    tf = tb.text_frame; tf.word_wrap = True
    tf.paragraphs[0].text = "AR earns no significant alpha (t < 1.0)."
    tf.paragraphs[0].font.size = Pt(17); tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = C_NAVY
    p2 = tf.add_paragraph(); p2.text = "The growth-level story fails. H2 supported."
    p2.font.size = Pt(14); p2.font.color.rgb = C_BODY
    p3 = tf.add_paragraph(); p3.text = "Only the composition story (RRR) survives."
    p3.font.size = Pt(14); p3.font.italic = True; p3.font.color.rgb = C_BLUE

    roadmap(sl, 3)


# ── Slide 23: FMB ─────────────────────────────────────────────────────────────
def s23_fmb(prs):
    sl = blank(prs)
    title_box(sl, "Fama-MacBeth Cross-Section: RRR Dominates")
    divider(sl)

    figure(sl, "fmb_coefs", Inches(0.3), Inches(1.35), Inches(7.0))

    data = [
        ["Variable",   "(1) RRR only", "(2) AR only", "(3) Joint"],
        ["Adj. RRR",   "0.0007",        "--",           "0.0003"],
        ["SE",         "(0.0002)",      "--",           "(0.0002)"],
        ["p-value",    "[0.001]",       "--",           "[0.134]"],
        ["Adj. AR",    "--",            "0.0006",       "0.0004"],
        ["SE",         "--",            "(0.0003)",     "(0.0003)"],
        ["p-value",    "--",            "[0.046]",      "[0.183]"],
        ["T (qtrs)",   "30",            "30",           "30"],
        ["Avg. N",     "112",           "112",          "112"],
    ]
    tbl = sl.shapes.add_table(9, 4,
                               Inches(7.2), Inches(1.5),
                               Inches(5.9), Inches(4.5)).table
    for j, w in enumerate([Inches(2.0), Inches(1.3), Inches(1.3), Inches(1.3)]):
        tbl.columns[j].width = w
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            is_h = (i == 0 or j == 0)
            cell(tbl.cell(i, j), val, size=11, bold=is_h,
                 bg=C_NAVY if is_h else None, color=C_WHITE if is_h else None)

    textbox(sl, ("RRR predicts in isolation [p = 0.001]. Neither coefficient survives at "
                 "conventional levels jointly (RRR p=0.134, AR p=0.183) -- consistent with "
                 "collinearity from the RG identity. Portfolio sorts, which directly separate "
                 "composition from level, provide cleaner evidence for H2."),
            Inches(7.2), Inches(6.2), Inches(5.9), Inches(0.95),
            size=12, color=C_BODY, wrap=True)
    roadmap(sl, 3)


# ── Slide 24: Same-growth placebo ─────────────────────────────────────────────
def s24_placebo(prs):
    sl = blank(prs)
    title_box(sl, "Same-Growth Placebo: Composition, Not Level")
    divider(sl)

    sh = rect(sl, Inches(0.5), Inches(1.3), Inches(12.33), Inches(1.1),
              fill=C_LBLUE, line=C_BLUE)
    tf = sh.text_frame; tf.word_wrap = True
    tf.paragraphs[0].text = (
        "Design: within the top quartile of headline revenue growth, split firms on adjusted RRR "
        "(above/below median). All firms have similarly high growth. "
        "The sort isolates HOW growth is generated, not how much."
    )
    tf.paragraphs[0].font.size = Pt(13)

    rb = rect(sl, Inches(2.3), Inches(2.65), Inches(8.73), Inches(1.9),
              fill=C_GREEN, line=RGBColor(0x82, 0xB3, 0x66))
    tf2 = rb.text_frame; tf2.word_wrap = True
    tf2.paragraphs[0].text = "Spread survives:  t = 2.19  [p = 0.029]"
    tf2.paragraphs[0].font.size = Pt(26); tf2.paragraphs[0].font.bold = True
    tf2.paragraphs[0].font.color.rgb = C_NAVY
    tf2.paragraphs[0].alignment = PP_ALIGN.CENTER
    p2 = tf2.add_paragraph()
    p2.text = "Among firms with identical headline growth, higher-RRR firms earn higher returns."
    p2.font.size = Pt(14); p2.alignment = PP_ALIGN.CENTER

    textbox(sl, "H4 supported: the signal is the composition of revenue growth, not its level.",
            Inches(0.5), Inches(4.8), Inches(12.33), Inches(0.5),
            size=16, bold=True, color=C_BLUE, align=PP_ALIGN.CENTER)

    textbox(sl, "Note: placebo analysis from analysis_v2.py; full table in paper appendix.",
            Inches(0.5), Inches(5.55), Inches(12.33), Inches(0.4),
            size=11, italic=True, color=C_GRAY, align=PP_ALIGN.CENTER)
    roadmap(sl, 3)


# ── Slide 25: Risk verdict ────────────────────────────────────────────────────
def s25_risk_verdict(prs):
    sl = blank(prs)
    title_box(sl, "The Risk Test: Verdict")
    divider(sl)

    textbox(sl, 'We predicted: "high-RRR firms should be safer" (Slide 10). Here is the test.',
            Inches(0.5), Inches(1.28), Inches(12.33), Inches(0.4),
            size=15, italic=True, color=C_GRAY)

    cells_data = [
        ("Return premium + Higher risk",  "Risk compensation (not supported)",             C_LGRAY),
        ("Return premium + LOWER risk",   "Mispricing -- CONFIRMED: our result",           C_GREEN),
        ("No premium + Higher risk",      "Risk-return tradeoff, no anomaly",              C_LGRAY),
        ("No premium + LOWER risk",       "Anomaly without return premium (unusual)",      C_LGRAY),
    ]
    xs = [Inches(0.6), Inches(6.5)]
    ys = [Inches(2.05), Inches(3.7)]
    for idx, (label, desc, bg) in enumerate(cells_data):
        r, c = idx // 2, idx % 2
        bx = rect(sl, xs[c], ys[r], Inches(5.6), Inches(1.4), fill=bg,
                  line=RGBColor(0x22, 0x77, 0x22) if idx == 1 else C_BLUE)
        tf = bx.text_frame; tf.word_wrap = True
        tf.paragraphs[0].text = desc
        tf.paragraphs[0].font.size = Pt(14); tf.paragraphs[0].font.bold = True
        if idx == 1:
            tf.paragraphs[0].font.color.rgb = RGBColor(0x00, 0x66, 0x00)
        p2 = tf.add_paragraph(); p2.text = "(" + label + ")"
        p2.font.size = Pt(10); p2.font.color.rgb = C_GRAY

    tb = sl.shapes.add_textbox(Inches(9.7), Inches(2.05), Inches(3.4), Inches(3.2))
    tf_d = tb.text_frame; tf_d.word_wrap = True
    tf_d.paragraphs[0].text = "Our data:"
    tf_d.paragraphs[0].font.size = Pt(14); tf_d.paragraphs[0].font.bold = True
    tf_d.paragraphs[0].font.color.rgb = C_NAVY
    for item in [
        "Q1 (high RRR): +19.87%/yr",
        "Q4 (low RRR): -6.72%/yr",
        "Q1 vol: 24.2% vs. Q4: 37.8%",
        "Q1 down beta: 0.85 vs. Q4: 1.44",
        "Q1 Sharpe: 0.82 vs. Q4: -0.18",
    ]:
        p = tf_d.add_paragraph(); p.text = "•  " + item
        p.font.size = Pt(12); p.font.color.rgb = C_BODY

    roadmap(sl, 4)


# ── Slide 26: Risk table ──────────────────────────────────────────────────────
def s26_risk_table(prs):
    sl = blank(prs)
    title_box(sl, "Risk Profile: Q1 Outperforms with Lower Risk")
    divider(sl)

    data = [
        ["Metric",           "Q1\n(High RRR)", "Q2",      "Q3",       "Q4\n(Low RRR)", "Q1-Q4"],
        ["Ann. Return (%)",  "+19.87",  "+6.89",  "-4.77",  "-6.72",  "+26.59"],
        ["Volatility (%)",   "24.23",   "22.40",  "32.37",  "37.80",  "31.31"],
        ["Sharpe Ratio",     "0.82",    "0.31",   "-0.15",  "-0.18",  "0.85"],
        ["Max Drawdown (%)", "-56.3",   "-44.4",  "-96.8",  "-92.5",  "-51.1"],
        ["Downside Beta",    "0.8525",  "1.0009", "1.5319", "1.4367", "-0.5841"],
        ["VaR 5th Pct (%)",  "-9.02",   "-9.28",  "-14.96", "-12.20", "-9.61"],
        ["Hit Rate (%)",     "64.4",    "60.9",   "49.4",   "49.4",   "58.6"],
    ]
    tbl = sl.shapes.add_table(8, 6,
                               Inches(0.3), Inches(1.42),
                               Inches(12.73), Inches(4.5)).table
    for j, w in enumerate([Inches(2.4), Inches(1.9), Inches(1.6), Inches(1.65), Inches(2.2), Inches(1.6)]):
        tbl.columns[j].width = w

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            is_h = (i == 0 or j == 0)
            q1_good = (j == 1 and i in {1, 2, 3, 5, 6, 7})
            q4_bad  = (j == 4 and i in {1, 2, 3, 5, 6})
            bg = C_NAVY if is_h else (C_GREEN if q1_good else (C_RED if q4_bad else None))
            cell(tbl.cell(i, j), val, size=11, bold=is_h,
                 bg=bg, color=C_WHITE if is_h else None)

    textbox(sl, ("Long leg (Q1): higher return, lower vol (24.2% vs 37.8%), lower downside beta "
                 "(0.85 vs 1.44), higher Sharpe. H3 confirmed in the long leg. "
                 "Inconsistent with standard risk-based explanations. "
                 "Note: the spread portfolio itself has higher vol (31.3%) and -51% max drawdown."),
            Inches(0.3), Inches(6.1), Inches(12.73), Inches(0.65),
            size=13, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER, wrap=True)
    roadmap(sl, 4)


# ── Slide 27: Revenue VaR ─────────────────────────────────────────────────────
def s27_rev_var(prs):
    sl = blank(prs)
    title_box(sl, "Revenue VaR by RRR Quartile: Fundamentals-Side Risk")
    divider(sl)

    figure(sl, "revenue_var_by_quartile", Inches(1.3), Inches(1.45), Inches(10.73))

    textbox(sl, ("High-RRR firms have less extreme downside revenue outcomes (less negative 5th percentile). "
                 "Lower stock return risk mirrors lower fundamental revenue risk. "
                 "The risk story is symmetric: Q1 firms are safer in BOTH stocks and fundamentals."),
            Inches(0.5), Inches(6.55), Inches(12.33), Inches(0.6),
            size=14, color=C_BODY, align=PP_ALIGN.CENTER, wrap=True)
    roadmap(sl, 4)


# ── Slide 28: Robustness ──────────────────────────────────────────────────────
def s28_robustness(prs):
    sl = blank(prs)
    title_box(sl, "Robustness: COVID Exclusion and Equal-Weighting")
    divider(sl)

    for i, (ttl, stat, note, fill, line) in enumerate([
        ("Excluding COVID (2020Q1-2021Q2)",
         "Long-short FF3 alpha:  t = 3.22  [p = 0.002]",
         "N = 69 months. Effect strengthens. The COVID disruption period does not drive the result.",
         C_GREEN, RGBColor(0x82, 0xB3, 0x66)),
        ("Equal-Weighted Portfolios",
         "Long-short FF3 alpha:  t = 1.35  [p = 0.180]",
         ("N = 87 months. Below conventional significance: a genuine limitation. "
          "Effect concentrated in larger firms where card-data coverage is strongest. "
          "Interpret VW result as the primary specification."),
         C_AMBER, RGBColor(0xCC, 0x99, 0x00)),
    ]):
        x = Inches(0.5) + i * Inches(6.5)
        sh = rect(sl, x, Inches(1.4), Inches(6.0), Inches(4.5), fill=fill, line=line)
        tf = sh.text_frame; tf.word_wrap = True
        tf.paragraphs[0].text = ttl
        tf.paragraphs[0].font.size = Pt(15); tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = C_NAVY
        p2 = tf.add_paragraph(); p2.text = ""
        p3 = tf.add_paragraph(); p3.text = stat
        p3.font.size = Pt(17); p3.font.bold = True
        p4 = tf.add_paragraph(); p4.text = ""
        p5 = tf.add_paragraph(); p5.text = note
        p5.font.size = Pt(13)

    roadmap(sl, 5)


# ── Slide 29: Double sort ─────────────────────────────────────────────────────
def s29_double_sort(prs):
    sl = blank(prs)
    title_box(sl, "Double Sort: RRR vs. AR")
    divider(sl)

    figure(sl, "double_sort_heatmap", Inches(1.0), Inches(1.45), Inches(11.33))

    textbox(sl, ("Across all revenue-growth terciles, moving from low-RRR to high-RRR increases returns "
                 "monotonically. The AR gradient is not monotonic. Composition dominates level."),
            Inches(0.5), Inches(6.6), Inches(12.33), Inches(0.55),
            size=14, color=C_BODY, align=PP_ALIGN.CENTER, wrap=True)
    roadmap(sl, 5)


# ── Slide 30: Sector breakdown ────────────────────────────────────────────────
def s30_sector(prs):
    sl = blank(prs)
    title_box(sl, "Robustness: Sector Breakdown")
    divider(sl)

    textbox(sl, ("Industry-time adjustment removes sector-composition effects by construction. "
                 "The main result holds within each of the four sectors individually."),
            Inches(0.5), Inches(1.28), Inches(12.33), Inches(0.5),
            size=15, color=C_BODY, wrap=True)

    data = [
        ["Sector",                   "Long-short FF3 alpha", "SE",         "p-value",    "N months"],
        ["Consumer Discretionary\n(90 firms, 72.6%)",
                                     "+2.52%/mo",            "--",          "significant", "87"],
        ["Industrials\n(14 firms)",  "+directional",          "--",         "see paper",   "87"],
        ["Communication Svcs\n(11 firms)",
                                     "+directional",         "--",          "see paper",   "87"],
        ["Consumer Staples\n(9 firms)",
                                     "+directional",         "--",          "see paper",   "87"],
        ["All sectors (adj.)",       "+2.36%/mo",            "(0.777)",     "[0.003]",     "87"],
    ]
    tbl = sl.shapes.add_table(6, 5,
                               Inches(0.5), Inches(2.05),
                               Inches(12.33), Inches(3.8)).table
    for j, w in enumerate([Inches(3.8), Inches(2.5), Inches(1.9), Inches(2.0), Inches(1.5)]):
        tbl.columns[j].width = w
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            is_h = (i == 0 or j == 0)
            is_all = (i == 5)
            bg = C_NAVY if is_h else (C_LBLUE if is_all else None)
            cell(tbl.cell(i, j), val, size=11, bold=is_h or is_all,
                 bg=bg, color=C_WHITE if is_h else None)

    textbox(sl, "Sector-specific alpha tables in paper appendix.",
            Inches(0.5), Inches(6.0), Inches(8.0), Inches(0.35), size=11, color=C_GRAY)
    roadmap(sl, 5)


# ── Slide 31: Contributions ───────────────────────────────────────────────────
def s31_contributions(prs):
    sl = blank(prs)
    title_box(sl, "Four Contributions")
    divider(sl)

    contribs = [
        ("1. New firm-level measure",
         "RRR constructed from card transaction data at the firm-quarter level for 124 public firms."),
        ("2. Mispricing of revenue composition",
         "The market misweights HOW firms grow (retention vs. acquisition), not just how much they grow."),
        ("3. High returns, low risk",
         "Return premium with lower volatility, lower drawdowns, lower downside beta. Rules out risk compensation."),
        ("4. CBCV meets asset pricing",
         "First firm-level revenue retention measure from transaction data tested in factor-model portfolio sorts. Extends CBCV (McCarthy & Fader 2018) to capital markets pricing."),
    ]
    for i, (ttl, body) in enumerate(contribs):
        y = Inches(1.35) + i * Inches(1.35)
        sh = rect(sl, Inches(0.5), y, Inches(12.33), Inches(1.2), fill=C_LBLUE, line=C_BLUE)
        tf = sh.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        r1 = p.add_run(); r1.text = ttl + ":  "
        r1.font.bold = True; r1.font.size = Pt(15); r1.font.color.rgb = C_NAVY
        r2 = p.add_run(); r2.text = body
        r2.font.size = Pt(14); r2.font.color.rgb = C_BODY


# ── Slide 32: Implications ────────────────────────────────────────────────────
def s32_implications(prs):
    sl = blank(prs)
    title_box(sl, "Investor and Managerial Implications")
    divider(sl)

    for i, (ttl, items, fill, line) in enumerate([
        ("Investors / Portfolio Managers", [
            "RRR is an implementable quarterly signal from card transaction data.",
            "Low turnover: signal is stable (T3-to-T3 persistence: 62.7%).",
            "Works in liquid large-caps (value-weighted result concentrated in larger firms).",
            "Short side: avoid (underweight) low-RRR firms.",
            "Not a risk premium: lower risk in the long leg (Q1).",
        ], C_GREEN, RGBColor(0x82, 0xB3, 0x66)),
        ("Managers / CFOs", [
            "Firms with high customer relationship quality may be undervalued when retention data is opaque.",
            "SaaS NRR reporting (e.g., Salesforce, HubSpot) is an established voluntary disclosure precedent.",
            "Voluntary retention disclosure could close the mispricing gap (speculative; disclosure-event tests left for future work).",
            "Customer relationship quality is a capital-structure-relevant asset.",
        ], C_LBLUE, C_BLUE),
    ]):
        x = Inches(0.5) + i * Inches(6.5)
        sh = rect(sl, x, Inches(1.35), Inches(6.0), Inches(5.3), fill=fill, line=line)
        tf = sh.text_frame; tf.word_wrap = True
        tf.paragraphs[0].text = ttl
        tf.paragraphs[0].font.size = Pt(15); tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = C_NAVY
        for item in items:
            p = tf.add_paragraph(); p.text = "•  " + item; p.font.size = Pt(12)


# ── Slide 33: Limitations ─────────────────────────────────────────────────────
def s33_limitations(prs):
    sl = blank(prs)
    title_box(sl, "Limitations")
    divider(sl)

    bullet_list(sl, [
        "Price-only returns (no dividends): may understate the long-leg premium if high-RRR firms pay more dividends. Future work: dividend-inclusive returns.",
        "124-firm sample, Consumer Discretionary dominated (90/124): findings may not generalize to all sectors or firm sizes. Future work: broader card-data coverage.",
        "2017-2024 window (87 months): relatively short; COVID period included, though exclusion strengthens results.",
        "Card-data panel selection and survivorship: if failing firms exit the card-data sample, the short-leg alpha may be biased upward. Delisting returns not verified for all sample firms.",
        "Equal-weighted result below conventional significance (t=1.35): effect is concentrated in larger, value-weighted firms. This is a genuine scope limitation.",
        "Mechanism not directly tested: the earnings-quality channel is theorized via the Sloan (1996) parallel; persistence regressions and announcement-window tests are left for future work.",
    ], Inches(1.28), size=14, gap=3)


# ── Slide 34: Conclusion ──────────────────────────────────────────────────────
def s34_conclusion(prs):
    sl = blank(prs)
    title_box(sl, "How to Avoid Bad Companies: What We Found")
    divider(sl)

    bullet_list(sl, [
        "Firms with weak customer relationships (Q4) are overpriced: adj. FF3 alpha = -2.18%/month [p < 0.001]. Firms with strong customer relationships (Q1) are fairly priced: +0.18%, not significant [p = 0.674].",
        "High returns, low risk: the long-short premium (2.36%/month FF3; 2.52% FF5) comes with lower volatility, smaller drawdowns, and lower downside beta in the long leg.",
        "AR earns no alpha: customer relationship strength (retention), not new-customer growth rate, is what the market fails to price.",
        "Composition, not level: same-growth placebo (t = 2.19 [p = 0.029]) confirms the signal is HOW growth is generated, not how much.",
    ], Inches(1.28), size=16, gap=5)

    sh = rect(sl, Inches(0.5), Inches(5.55), Inches(12.33), Inches(0.95),
              fill=C_LBLUE, line=C_BLUE)
    tf = sh.text_frame; tf.word_wrap = True
    tf.paragraphs[0].text = (
        "Next: broader card-data coverage across sectors, dividend-inclusive returns, "
        "disclosure-event tests around NRR reporting, contractual settings (B2B)."
    )
    tf.paragraphs[0].font.size = Pt(13); tf.paragraphs[0].font.italic = True

    cb = sl.shapes.add_textbox(Inches(0.5), Inches(6.65), Inches(12.33), Inches(0.45))
    pc = cb.text_frame.paragraphs[0]
    pc.alignment = PP_ALIGN.CENTER
    rc = pc.add_run()
    rc.text = "Thilo Kraft   |   kraft.thilo.g@gmail.com   |   Thank you."
    rc.font.size = Pt(14); rc.font.color.rgb = C_GRAY


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    builders = [
        s01_title, s02_buffett, s03_weak_firms, s04_blindspot,
        s05_data_gap, s06_rq, s07_lit_map, s08_accrual,
        s09_decomp, s10_persistence, s11_skeptic, s12_risk_test,
        s13_hyps, s14_data, s15_sample, s16_rev_growth,
        s17_transition, s18_cumret, s19_ff3, s20_ff5,
        s21_alpha_bars, s22_ar, s23_fmb, s24_placebo,
        s25_risk_verdict, s26_risk_table, s27_rev_var, s28_robustness,
        s29_double_sort, s30_sector, s31_contributions, s32_implications,
        s33_limitations, s34_conclusion,
    ]

    for i, fn in enumerate(builders, 1):
        print(f"  Slide {i:02d}: {fn.__name__}")
        fn(prs)

    prs.save(OUTFILE)
    print(f"\nSaved: {OUTFILE}")
    print(f"Slide count: {len(prs.slides)}")


if __name__ == "__main__":
    main()
