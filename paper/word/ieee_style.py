"""Apply IEEE conference layout to the pandoc DOCX of the paper.

Adds the title and author block, two-column US-letter layout, Times New Roman, Roman-numbered
headings, IEEE captions (Fig. n. / TABLE I), numbered equations, booktabs-style tables,
full-width placement for Table III and Fig. 3, and IEEE reference formatting.
"""

import argparse
import logging
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Pt, RGBColor

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

FONT = "Times New Roman"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
COL_W = Inches(3.5)
FULL_W = Inches(7.25)
COL_TWIPS = 5040  # 3.5 in
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


# ----------------------------------------------------------------------------- helpers
def set_run_font(rpr_parent, size_pt: float | None = None) -> None:
    """Force Times New Roman (all scripts) on a style or run element."""
    rpr = rpr_parent.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), FONT)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        fonts.attrib.pop(qn(attr), None)


def sect_pr(cols: int, last: bool = False) -> OxmlElement:
    """Continuous section properties for a US-letter page with ``cols`` columns."""
    s = OxmlElement("w:sectPr")
    t = OxmlElement("w:type")
    t.set(qn("w:val"), "continuous")
    s.append(t)
    pg = OxmlElement("w:pgSz")
    pg.set(qn("w:w"), "12240")
    pg.set(qn("w:h"), "15840")
    s.append(pg)
    mar = OxmlElement("w:pgMar")
    for k, v in {"top": 1080, "right": 900, "bottom": 1440, "left": 900, "header": 720,
                 "footer": 720, "gutter": 0}.items():
        mar.set(qn(f"w:{k}"), str(v))
    s.append(mar)
    c = OxmlElement("w:cols")
    c.set(qn("w:num"), str(cols))
    c.set(qn("w:space"), "360")
    s.append(c)
    return s


def break_paragraph(cols: int) -> OxmlElement:
    """Tiny empty paragraph that ends a section with ``cols`` columns."""
    p = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    sp = OxmlElement("w:spacing")
    for k in ("before", "after"):
        sp.set(qn(f"w:{k}"), "0")
    sp.set(qn("w:line"), "20")
    sp.set(qn("w:lineRule"), "exact")
    ppr.append(sp)
    ppr.append(sect_pr(cols))
    p.append(ppr)
    return p


def math_size(el, size: float) -> None:
    """Set the point size of every OMML run under ``el``."""
    for mr in el.iter(f"{{{M_NS}}}r"):
        rpr = mr.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            mr.insert(0, rpr)
        for tag in ("w:sz", "w:szCs"):
            e = rpr.find(qn(tag))
            if e is None:
                e = OxmlElement(tag)
                rpr.append(e)
            e.set(qn("w:val"), str(int(size * 2)))


def fmt(p, size: float, align=None, indent: float | None = None, before: float = 0,
        after: float = 0, bold: bool | None = None, italic: bool | None = None,
        small_caps: bool = False) -> None:
    """Paragraph and run formatting in one call."""
    pf = p.paragraph_format
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = 1.0
    if align is not None:
        pf.alignment = align
    if indent is not None:
        pf.first_line_indent = Inches(indent)
    for r in p.runs:
        r.font.size = Pt(size)
        r.font.name = FONT
        set_run_font(r._r)
        if bold is not None:
            r.font.bold = bold
        if italic is not None:
            r.font.italic = italic
        if small_caps:
            r.font.small_caps = True
        r.font.color.rgb = RGBColor(0, 0, 0)
    math_size(p._p, size)


def insert_paragraph_before(par, text: str = ""):
    """New paragraph (python-docx object) inserted before ``par``."""
    new = par.insert_paragraph_before(text)
    new.style = par.part.document.styles["Normal"]
    return new


def element_paragraph(doc, el):
    """Wrap a raw w:p element as a python-docx Paragraph."""
    from docx.text.paragraph import Paragraph
    return Paragraph(el, doc._body)


# ----------------------------------------------------------------------------- steps
def base_styles(doc) -> None:
    """Times New Roman 10 pt everywhere, black text, single spacing."""
    defaults = doc.styles.element.find(qn("w:docDefaults"))
    rpr_default = defaults.find(qn("w:rPrDefault"))
    rpr = rpr_default.find(qn("w:rPr"))
    if rpr is None:
        rpr = OxmlElement("w:rPr")
        rpr_default.append(rpr)
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), FONT)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        fonts.attrib.pop(qn(attr), None)
    for st in doc.styles:
        if st.type != 1:  # paragraph styles only
            continue
        st.font.name = FONT
        set_run_font(st.element)
        st.font.color.rgb = RGBColor(0, 0, 0)
        if st.name in ("Normal", "Body Text", "First Paragraph", "Compact"):
            st.font.size = Pt(10)
            pf = st.paragraph_format
            pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(0), 1.0
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if st.name in ("Body Text", "First Paragraph"):
            st.paragraph_format.first_line_indent = Inches(0.2)


def title_block(doc, title: str, authors: list[list[str]]) -> None:
    """Title and three-column author block in a one-column first section."""
    first = doc.paragraphs[0]
    tp = insert_paragraph_before(first, title)
    fmt(tp, 24, WD_ALIGN_PARAGRAPH.CENTER, 0, after=12)
    table = doc.add_table(rows=1, cols=len(authors))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, lines in zip(table.rows[0].cells, authors, strict=True):
        cell.width = FULL_W // len(authors)
        cell.paragraphs[0].text = lines[0]
        fmt(cell.paragraphs[0], 11, WD_ALIGN_PARAGRAPH.CENTER, 0)
        for line in lines[1:]:
            # IEEE template: department and organization lines are italic
            ital = re.fullmatch(r"\\textit\{(.*)\}", line)
            p = cell.add_paragraph(ital.group(1) if ital else line)
            fmt(p, 10, WD_ALIGN_PARAGRAPH.CENTER, 0, italic=True if ital else None)
    for cell in table.rows[0].cells:  # ordinal suffix (1st, 2nd, 3rd) as superscript
        p = cell.paragraphs[0]
        m = re.match(r"(\d)(st|nd|rd) (.*)", p.text)
        if m:
            p.text = ""
            p.add_run(m.group(1))
            p.add_run(m.group(2)).font.superscript = True
            p.add_run(" " + m.group(3))
            fmt(p, 11, WD_ALIGN_PARAGRAPH.CENTER, 0)
    tp._p.addnext(table._tbl)
    spacer = OxmlElement("w:p")
    table._tbl.addnext(spacer)
    sp = element_paragraph(doc, spacer)
    sp.paragraph_format.space_after = Pt(14)
    spacer.get_or_add_pPr().append(sect_pr(1))


def abstract_keywords(doc) -> None:
    """IEEE abstract and index terms: bold 9 pt, run-in italic label."""
    for p in doc.paragraphs[:6]:
        if p.text.startswith(("Abstract", "Index Terms")):
            fmt(p, 9, WD_ALIGN_PARAGRAPH.JUSTIFY, 0.2, bold=True,
                after=4 if p.text.startswith("Abstract") else 8)


def headings(doc) -> dict[str, str]:
    """Number headings as I., II. (small caps, centred) and A., B. (italic)."""
    h1 = h2 = 0
    numbers = {}
    for p in doc.paragraphs:
        name = p.style.name
        if name == "Heading 1":
            text = p.text.strip()
            if text == "References":
                label = text
            else:
                h1, h2 = h1 + 1, 0
                label = f"{ROMAN[h1 - 1]}. {text}"
                numbers[text] = ROMAN[h1 - 1]
            p.text = label
            fmt(p, 10, WD_ALIGN_PARAGRAPH.CENTER, 0, before=8, after=4, bold=False,
                italic=False, small_caps=True)
        elif name == "Heading 2":
            h2 += 1
            p.text = f"{chr(64 + h2)}. {p.text.strip()}"
            fmt(p, 10, WD_ALIGN_PARAGRAPH.LEFT, 0, before=6, after=3, bold=False, italic=True)
        if name.startswith("Heading"):
            p.paragraph_format.keep_with_next = True
    return numbers


def captions_and_images(doc, wide_figs: set[int]) -> None:
    """IEEE captions; images sized to the column (or page) width."""
    fig = 0
    tab = 0
    for p in list(doc.paragraphs):
        name = p.style.name
        if name == "Image Caption":
            fig += 1
            text = p.text.strip()
            p.text = ""
            p.add_run(f"Fig. {fig}. ")
            p.add_run(text)
            fmt(p, 8, WD_ALIGN_PARAGRAPH.JUSTIFY, 0, before=2, after=8)
        elif name == "Table Caption":
            tab += 1
            text = p.text.strip()
            p.text = text
            fmt(p, 8, WD_ALIGN_PARAGRAPH.CENTER, 0, after=4, small_caps=True)
            lab = insert_paragraph_before(p, f"TABLE {ROMAN[tab - 1]}")
            fmt(lab, 8, WD_ALIGN_PARAGRAPH.CENTER, 0, before=6)
            lab.paragraph_format.keep_with_next = True
            p.paragraph_format.keep_with_next = True
        elif name == "Captioned Figure":
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.keep_with_next = True
    for i, shape in enumerate(doc.inline_shapes, 1):
        target = FULL_W if i in wide_figs else COL_W
        ratio = shape.height / shape.width
        shape.width = Emu(int(target))
        shape.height = Emu(int(target * ratio))


def merge_math_runs(omath) -> None:
    """Join adjacent OMML runs with identical formatting (e.g. l-o-g-i-t -> logit)."""
    from lxml import etree
    for parent in [omath, *omath.iter()]:
        kids = list(parent)
        prev = None
        for el in kids:
            if el.tag != f"{{{M_NS}}}r":
                prev = None
                continue
            fmt_key = b"".join(etree.tostring(c) for c in el if c.tag != f"{{{M_NS}}}t")
            t = el.find(f"{{{M_NS}}}t")
            if prev is not None and prev[1] == fmt_key and t is not None:
                pt = prev[0].find(f"{{{M_NS}}}t")
                pt.text = (pt.text or "") + (t.text or "")
                pt.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                parent.remove(el)
            else:
                prev = (el, fmt_key)


def equations(doc) -> int:
    """Display equations: centred, with (n) at the right margin."""
    n = 0
    for p in doc.paragraphs:
        omp = p._p.find(f".//{{{M_NS}}}oMathPara")
        if omp is None:
            continue
        n += 1
        maths = omp.findall(f"{{{M_NS}}}oMath")
        for m in maths:
            merge_math_runs(m)
        parent = omp.getparent()
        idx = list(parent).index(omp)
        parent.remove(omp)
        new = []
        tab1 = OxmlElement("w:r")
        tab1.append(OxmlElement("w:tab"))
        new.append(tab1)
        new.extend(maths)
        tab2 = OxmlElement("w:r")
        tab2.append(OxmlElement("w:tab"))
        new.append(tab2)
        num = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.text = f"({n})"
        num.append(t)
        new.append(num)
        for k, el in enumerate(new):
            parent.insert(idx + k, el)
        pf = p.paragraph_format
        pf.first_line_indent = Inches(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.space_before = pf.space_after = Pt(3)
        pf.tab_stops.clear_all()
        from docx.enum.text import WD_TAB_ALIGNMENT
        pf.tab_stops.add_tab_stop(Emu(int(COL_W / 2)), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Emu(int(COL_W)), WD_TAB_ALIGNMENT.RIGHT)
        nxt = p._p.getnext()
        if nxt is not None and nxt.tag == qn("w:p"):
            element_paragraph(doc, nxt).paragraph_format.first_line_indent = Inches(0)
    return n


def border(tag: str, size: int) -> OxmlElement:
    b = OxmlElement(f"w:{tag}")
    b.set(qn("w:val"), "single" if size else "nil")
    if size:
        b.set(qn("w:sz"), str(size))
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "000000")
    return b


def column_widths(t, total: int) -> None:
    """Fixed layout with column widths proportional to their longest cell text."""
    rows = t._tbl.findall(qn("w:tr"))
    ncol = len(t._tbl.find(qn("w:tblGrid")).findall(qn("w:gridCol")))
    longest = [4] * ncol
    for tr in rows:
        col = 0
        for tc in tr.findall(qn("w:tc")):
            span_el = tc.find(f"{qn('w:tcPr')}/{qn('w:gridSpan')}")
            span = int(span_el.get(qn("w:val"))) if span_el is not None else 1
            text = "".join(x.text or "" for x in tc.iter(qn("w:t")))
            if span == 1 and col < ncol:
                longest[col] = max(longest[col], min(len(text), 40))
            col += span
    need = [n * 88 + 100 for n in longest]  # ~88 twips per 8-pt character + cell margins
    if sum(need) <= total:
        spare = total - sum(need)
        widths = [int(n + spare * n / sum(need)) for n in need]
    else:
        widths = [int(total * n / sum(need)) for n in need]
    widths[-1] += total - sum(widths)
    grid = t._tbl.find(qn("w:tblGrid"))
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths, strict=True):
        gc.set(qn("w:w"), str(w))
    for tr in rows:
        col = 0
        for tc in tr.findall(qn("w:tc")):
            tcpr = tc.get_or_add_tcPr()
            span_el = tcpr.find(qn("w:gridSpan"))
            span = int(span_el.get(qn("w:val"))) if span_el is not None else 1
            tcw = tcpr.find(qn("w:tcW"))
            if tcw is None:
                tcw = OxmlElement("w:tcW")
                tcpr.insert(0, tcw)
            tcw.set(qn("w:w"), str(sum(widths[col:col + span])))
            tcw.set(qn("w:type"), "dxa")
            col += span
    tpr = t._tbl.tblPr
    layout = tpr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tpr.append(layout)
    layout.set(qn("w:type"), "fixed")
    w = tpr.find(qn("w:tblW"))
    w.set(qn("w:type"), "dxa")
    w.set(qn("w:w"), str(total))


def keep_together(t) -> None:
    """Rows never split, and every row stays with the next one."""
    rows = t.rows
    for ri, row in enumerate(rows):
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if ri < len(rows) - 1:
            for cell in row.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True


def tables(doc, midrule_rows: dict[int, list[int]]) -> None:
    """Booktabs-style rules, 8 pt text, centred; full-width tables sized to the page."""
    for ti, t in enumerate(doc.tables[1:], 1):  # table 0 is the author block
        tbl = t._tbl
        tpr = tbl.tblPr
        for old in tpr.findall(qn("w:tblBorders")) + tpr.findall(qn("w:tblStyle")):
            tpr.remove(old)
        borders = OxmlElement("w:tblBorders")
        for tag, size in (("top", 8), ("left", 0), ("bottom", 8), ("right", 0),
                          ("insideH", 0), ("insideV", 0)):
            borders.append(border(tag, size))
        tpr.append(borders)
        w = tpr.find(qn("w:tblW"))
        if w is None:
            w = OxmlElement("w:tblW")
            tpr.append(w)
        w.set(qn("w:type"), "pct")
        w.set(qn("w:w"), "5000")
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        rules = [0] + midrule_rows.get(ti, [])
        for ri, row in enumerate(t.rows):
            for cell in row.cells:
                tcpr = cell._tc.get_or_add_tcPr()
                if ri in rules:
                    tb = OxmlElement("w:tcBorders")
                    tb.append(border("bottom", 4))
                    tcpr.append(tb)
                for p in cell.paragraphs:
                    fmt(p, 8, None, 0)
                mar = OxmlElement("w:tcMar")
                for side, v in (("left", 40), ("right", 40), ("top", 10), ("bottom", 10)):
                    e = OxmlElement(f"w:{side}")
                    e.set(qn("w:w"), str(v))
                    e.set(qn("w:type"), "dxa")
                    mar.append(e)
                tcpr.append(mar)
        column_widths(t, 10440 if ti == 3 else COL_TWIPS)
        keep_together(t)


def wide_blocks(doc, wide_table: int, wide_fig: int) -> None:
    """Put the full-width table and figure into one-column continuous sections."""
    caps = [p for p in doc.paragraphs if p.style.name == "Table Caption"]
    # block for the wide table: "TABLE n" label paragraph .. table element
    cap = caps[wide_table - 1]
    start = cap._p.getprevious()  # the "TABLE n" label
    end = doc.tables[wide_table]._tbl
    start.addprevious(break_paragraph(2))
    end.addnext(break_paragraph(1))
    figs = [p for p in doc.paragraphs if p.style.name == "Captioned Figure"]
    fig_p = figs[wide_fig - 1]._p
    cap_p = fig_p.getnext()
    fig_p.addprevious(break_paragraph(2))
    cap_p.addnext(break_paragraph(1))


def references(doc) -> None:
    """8 pt references with a hanging [n] label."""
    in_refs = False
    for p in doc.paragraphs:
        if p.style.name == "Heading 1" and p.text.strip() == "References":
            in_refs = True
            continue
        if not in_refs or not p.text.strip():
            continue
        for r in p.runs:
            if "\u00a0" in r.text:
                r.text = r.text.replace("\u00a0", "\t", 1)
                break
        fmt(p, 8, WD_ALIGN_PARAGRAPH.JUSTIFY, None, after=1)
        pf = p.paragraph_format
        pf.left_indent = Inches(0.3)
        pf.first_line_indent = Inches(-0.3)
        pf.tab_stops.add_tab_stop(Inches(0.3))


SCHEMA_ORDER = {
    "tblPr": ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
              "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders",
              "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"],
    "tcPr": ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap",
             "tcMar", "textDirection", "tcFitText", "vAlign", "hideMark"],
    "tcMar": ["top", "start", "left", "bottom", "end", "right"],
    "tcBorders": ["top", "start", "left", "bottom", "end", "right", "insideH", "insideV",
                  "tl2br", "tr2bl"],
    "tblBorders": ["top", "start", "left", "bottom", "end", "right", "insideH", "insideV"],
}


def schema_order(doc) -> None:
    """Sort table and cell property children into the order the OOXML schema requires."""
    for tag, order in SCHEMA_ORDER.items():
        for el in doc.element.body.iter(qn(f"w:{tag}")):
            kids, seen = [], set()
            for k in reversed(list(el)):  # merged cells were visited twice: keep one copy
                if k.tag in seen:
                    el.remove(k)
                    continue
                seen.add(k.tag)
                kids.insert(0, k)
            rank = {qn(f"w:{n}"): i for i, n in enumerate(order)}
            kids.sort(key=lambda k: rank.get(k.tag, len(order)))
            for k in kids:
                el.remove(k)
            for k in kids:
                el.append(k)


def final_section(doc) -> None:
    """Document-level section: two columns, continuous."""
    body = doc.element.body
    old = body.find(qn("w:sectPr"))
    if old is not None:
        body.remove(old)
    body.append(sect_pr(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("tex", type=Path, help="original main.tex (title and authors)")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    tex = args.tex.read_text()
    title = re.search(r"\\title\{(.*?)\}\n", tex).group(1)
    authors = []
    for name, aff in re.findall(r"\\IEEEauthorblockN\{(.*?)\}\s*\\IEEEauthorblockA\{(.*?)\}"
                                r"(?=\s*(?:\\and|\}))", tex, flags=re.S):
        name = re.sub(r"\\textsuperscript\{(\w+)\}", r"\1", name)
        authors.append([name] + [s.strip() for s in aff.split("\\\\")])
    doc = Document(str(args.raw))
    base_styles(doc)
    title_block(doc, title, authors)  # author table becomes doc.tables[0]
    abstract_keywords(doc)
    numbers = headings(doc)
    captions_and_images(doc, wide_figs={3})
    n_eq = equations(doc)
    tables(doc, midrule_rows={5: [4]})  # Table V: rule between S5 and S4
    wide_blocks(doc, wide_table=3, wide_fig=3)
    references(doc)
    final_section(doc)
    schema_order(doc)
    link = doc.styles["Hyperlink"]
    link.font.color.rgb = RGBColor(0, 0, 0)
    link.font.underline = False
    cp = doc.core_properties
    cp.title = title
    cp.author = "; ".join(re.sub(r"^\d(st|nd|rd) ", "", a[0]) for a in authors)
    cp.last_modified_by = cp.author
    cp.comments = ""
    kw = re.search(r"\\begin\{IEEEkeywords\}\s*(.*?)\s*\\end\{IEEEkeywords\}", tex, re.S)
    cp.keywords = "; ".join(k.strip() for k in kw.group(1).split(","))
    doc.save(str(args.out))
    log.info(f"saved {args.out}: {len(numbers)} sections, {n_eq} equations, "
             f"{len(doc.inline_shapes)} images, {len(doc.tables) - 1} tables")


if __name__ == "__main__":
    main()
