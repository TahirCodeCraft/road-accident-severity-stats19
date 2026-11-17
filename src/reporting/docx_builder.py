"""Render a parsed Markdown report (see :mod:`src.reporting.md_parser`) to a styled DOCX.

Typical use::

    from src.reporting.docx_builder import ReportStyle, build_docx, convert_to_pdf

    stats = build_docx(Path("reports/report.md"), Path("reports/Report.docx"))
    convert_to_pdf(Path("reports/Report.docx"), Path("reports"))

Layout: A4, 2.2 cm margins, Calibri 11 pt (LibreOffice substitutes metric-compatible Carlito),
1.15 line spacing, justified body, a title block on page 1, a small grey running header and a
footer with a live ``PAGE`` field.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document as new_document
from docx.document import Document as DocxDocument
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Emu, Pt, RGBColor
from docx.styles.style import BaseStyle
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from docx.text.run import Run

from src.reporting.md_parser import (
    BlockQuote,
    Document,
    FrontMatter,
    Heading,
    Image,
    InlineRun,
    ListBlock,
    ListItem,
    PageBreak,
    Paragraph,
    Reference,
    Table,
    parse_inline,
    parse_markdown,
    plain_text,
)

logger = logging.getLogger(__name__)

ALIGNMENTS = {
    "left": WD_ALIGN_PARAGRAPH.LEFT,
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
}
BULLET_GLYPHS = ("•", "–")
EN_DASH = "–"


# --------------------------------------------------------------------------------------------
# Configuration and results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportStyle:
    """Visual settings of the rendered report (all sizes in pt unless suffixed ``_cm``)."""

    page_width_cm: float = 21.0
    page_height_cm: float = 29.7
    margin_cm: float = 2.2
    header_distance_cm: float = 1.1
    body_font: str = "Calibri"
    code_font: str = "Consolas"
    body_size: float = 11.0
    line_spacing: float = 1.15
    para_space_after: float = 6.0
    small_size: float = 9.0
    header_size: float = 8.0
    title_size: float = 18.0
    subtitle_size: float = 13.0
    heading_sizes: tuple[float, float, float] = (14.0, 12.0, 11.0)
    heading_space_before: tuple[float, float, float] = (16.0, 11.0, 8.0)
    heading_space_after: tuple[float, float, float] = (6.0, 4.0, 3.0)
    accent_rgb: str = "1F3864"
    muted_rgb: str = "595959"
    rule_rgb: str = "BFBFBF"
    table_header_fill: str = "E7E6E6"
    table_border_rgb: str = "BFBFBF"
    quote_fill: str = "F2F5FA"
    quote_bar_rgb: str = "2E5597"
    default_image_width_cm: float = 16.0
    max_image_height_cm: float = 21.0
    list_indent_cm: float = 0.63
    reference_indent_cm: float = 0.9
    header_text: str = "Road Accident Severity Prediction — DSTT Project"
    footer_text: str = "Tahir Ahmed Khan – 25i-8025 – Page "

    @property
    def text_width_cm(self) -> float:
        """Usable text width between the side margins."""
        return self.page_width_cm - 2 * self.margin_cm


@dataclass
class BuildStats:
    """Counts of rendered elements, returned by :func:`build_docx`."""

    headings: int = 0
    paragraphs: int = 0
    lists: int = 0
    tables: int = 0
    figures: int = 0
    quotes: int = 0
    references: int = 0
    page_breaks: int = 0
    figure_paths: list[Path] = field(default_factory=list)


# --------------------------------------------------------------------------------------------
# Low-level OOXML helpers
# --------------------------------------------------------------------------------------------


def set_run_font(run: Run, name: str) -> None:
    """Set a run's font for all scripts (ASCII, high-ANSI, East-Asian, complex).

    Args:
        run: Target run.
        name: Font family.
    """
    run.font.name = name
    run.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), name)


def set_style_font(style: BaseStyle, name: str, size: float, rgb: str | None = None) -> None:
    """Force a style's font (dropping theme-font and theme-colour links that override it).

    Args:
        style: Paragraph or character style.
        name: Font family.
        size: Size in points.
        rgb: Optional hex colour such as ``"1F3864"``.
    """
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rfonts.attrib.pop(qn(attr), None)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), name)
    style.font.size = Pt(size)
    if rgb is not None:
        for color in rpr.findall(qn("w:color")):
            rpr.remove(color)
        style.font.color.rgb = RGBColor.from_string(rgb)


def make_element(tag: str, **attrs: str) -> OxmlElement:
    """Create an OOXML element with ``w:``-namespaced attributes.

    Args:
        tag: Qualified tag, e.g. ``"w:shd"``.
        **attrs: Attribute names without prefix (``val="clear"``).

    Returns:
        The new element.
    """
    element = OxmlElement(tag)
    for key, value in attrs.items():
        element.set(qn(f"w:{key}"), value)
    return element


def shade_cell(cell: object, fill: str) -> None:
    """Apply a solid background fill to a table cell.

    Args:
        cell: python-docx ``_Cell``.
        fill: Hex colour.
    """
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_pr.append(make_element("w:shd", val="clear", color="auto", fill=fill))


def set_paragraph_border(paragraph: DocxParagraph, side: str, size: int, rgb: str,
                         space: int = 1) -> None:
    """Add a border line to one side of a paragraph.

    Args:
        paragraph: Target paragraph.
        side: ``"top"``, ``"bottom"``, ``"left"`` or ``"right"``.
        size: Width in eighths of a point.
        rgb: Hex colour.
        space: Gap between text and border in points.
    """
    p_pr = paragraph._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is None:
        borders = make_element("w:pBdr")
        p_pr.append(borders)
    borders.append(make_element(f"w:{side}", val="single", sz=str(size), space=str(space),
                                color=rgb))


def shade_paragraph(paragraph: DocxParagraph, fill: str) -> None:
    """Apply a background fill to a paragraph.

    Args:
        paragraph: Target paragraph.
        fill: Hex colour.
    """
    paragraph._p.get_or_add_pPr().append(
        make_element("w:shd", val="clear", color="auto", fill=fill)
    )


def append_field(paragraph: DocxParagraph, instruction: str,
                 placeholder: str = "1") -> list[Run]:
    """Append a complex field (e.g. ``PAGE``) that Word and LibreOffice update on render.

    Args:
        paragraph: Paragraph receiving the field.
        instruction: Field code, e.g. ``"PAGE"``.
        placeholder: Cached result shown before the field is updated.

    Returns:
        All runs making up the field (format every one: LibreOffice takes the field's
        formatting from the ``begin`` run, Word from the result run).
    """
    begin = paragraph.add_run()
    begin._r.append(make_element("w:fldChar", fldCharType="begin"))
    instr_run = paragraph.add_run()
    instr = make_element("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} \\* MERGEFORMAT "
    instr_run._r.append(instr)
    separate = paragraph.add_run()
    separate._r.append(make_element("w:fldChar", fldCharType="separate"))
    result = paragraph.add_run(placeholder)
    end = paragraph.add_run()
    end._r.append(make_element("w:fldChar", fldCharType="end"))
    return [begin, instr_run, separate, result, end]


def set_table_borders(table: DocxTable, rgb: str, size: int = 4) -> None:
    """Draw a uniform thin grid on a table.

    Args:
        table: Target table.
        rgb: Hex border colour.
        size: Width in eighths of a point.
    """
    tbl_pr = table._tbl.tblPr
    for old in tbl_pr.findall(qn("w:tblBorders")):
        tbl_pr.remove(old)
    borders = make_element("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        borders.append(make_element(f"w:{side}", val="single", sz=str(size), space="0",
                                    color=rgb))
    tbl_pr.append(borders)


def set_table_cell_margins(table: DocxTable, top_bottom_cm: float, left_right_cm: float) -> None:
    """Set default internal cell padding for a table.

    Args:
        table: Target table.
        top_bottom_cm: Vertical padding.
        left_right_cm: Horizontal padding.
    """
    margins = make_element("w:tblCellMar")
    for side, value in (("top", top_bottom_cm), ("bottom", top_bottom_cm),
                        ("left", left_right_cm), ("right", left_right_cm)):
        margins.append(make_element(f"w:{side}", w=str(int(Cm(value).twips)), type="dxa"))
    table._tbl.tblPr.append(margins)


def mark_header_row(row: object) -> None:
    """Repeat a table row at the top of each page the table spans.

    Args:
        row: python-docx ``_Row``.
    """
    row._tr.get_or_add_trPr().append(make_element("w:tblHeader", val="true"))


def estimate_column_widths(header: list[str], rows: list[list[str]], text_width_cm: float,
                           cm_per_char: float = 0.18, padding_cm: float = 0.45,
                           min_cm: float = 1.3) -> list[float]:
    """Estimate column widths from content length, shrinking to fit the text width.

    Args:
        header: Header cells.
        rows: Body rows.
        text_width_cm: Available width.
        cm_per_char: Approximate glyph width at 9 pt.
        padding_cm: Horizontal cell padding.
        min_cm: Minimum column width.

    Returns:
        Width of each column in cm.
    """
    widths: list[float] = []
    for col, title in enumerate(header):
        lengths = [len(plain_text(row[col])) for row in rows] + [int(len(plain_text(title)) * 1.1)]
        natural = min(max(lengths), 60) * cm_per_char + padding_cm
        widths.append(max(natural, min_cm))
    total = sum(widths)
    if total <= text_width_cm:
        return widths
    return [w * text_width_cm / total for w in widths]


# --------------------------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------------------------


class DocxReportBuilder:
    """Stateful renderer: holds the document, style, assets directory and caption counters."""

    def __init__(self, style: ReportStyle, assets_dir: Path) -> None:
        """Create an empty, fully styled document.

        Args:
            style: Visual settings.
            assets_dir: Directory that image paths are resolved against.
        """
        self.style = style
        self.assets_dir = assets_dir
        self.doc: DocxDocument = new_document()
        self.stats = BuildStats()
        self._configure_page()
        self._configure_styles()
        self._configure_header_footer()

    # ---- document setup ---------------------------------------------------------------------

    def _configure_page(self) -> None:
        """Apply A4 page size and margins to the (single) section."""
        section = self.doc.sections[0]
        section.start_type = WD_SECTION.NEW_PAGE
        section.page_width = Cm(self.style.page_width_cm)
        section.page_height = Cm(self.style.page_height_cm)
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Cm(self.style.margin_cm))
        section.header_distance = Cm(self.style.header_distance_cm)
        section.footer_distance = Cm(self.style.header_distance_cm)

    def _configure_styles(self) -> None:
        """Configure Normal, Heading 1-3 and Caption styles."""
        s = self.style
        normal = self.doc.styles["Normal"]
        set_style_font(normal, s.body_font, s.body_size)
        fmt = normal.paragraph_format
        fmt.line_spacing = s.line_spacing
        fmt.space_before = Pt(0)
        fmt.space_after = Pt(s.para_space_after)
        fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        fmt.widow_control = True
        for level in (1, 2, 3):
            self._configure_heading_style(level)
        caption = self.doc.styles["Caption"]
        set_style_font(caption, s.body_font, s.small_size, s.muted_rgb)
        caption.font.bold = False
        caption.font.italic = True
        caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.line_spacing = 1.0

    def _configure_heading_style(self, level: int) -> None:
        """Style one heading level (font, size, colour, spacing, keep-with-next).

        Args:
            level: Heading level 1-3.
        """
        s = self.style
        heading = self.doc.styles[f"Heading {level}"]
        set_style_font(heading, s.body_font, s.heading_sizes[level - 1], s.accent_rgb)
        heading.font.bold = True
        heading.font.italic = False
        fmt = heading.paragraph_format
        fmt.space_before = Pt(s.heading_space_before[level - 1])
        fmt.space_after = Pt(s.heading_space_after[level - 1])
        fmt.line_spacing = 1.0
        fmt.keep_with_next = True
        fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT

    def _configure_header_footer(self) -> None:
        """Add the running header text and the footer with a PAGE field."""
        s = self.style
        section = self.doc.sections[0]
        header = section.header.paragraphs[0]
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        self._small_run(header, s.header_text)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._small_run(footer, s.footer_text)
        for run in append_field(footer, "PAGE"):
            self._format_small(run)

    def _format_small(self, run: Run) -> Run:
        """Format a run as small grey header/footer text."""
        set_run_font(run, self.style.body_font)
        run.font.size = Pt(self.style.header_size)
        run.font.color.rgb = RGBColor.from_string(self.style.muted_rgb)
        return run

    def _small_run(self, paragraph: DocxParagraph, text: str) -> Run:
        """Append small grey text to a header/footer paragraph."""
        return self._format_small(paragraph.add_run(text))

    # ---- inline text --------------------------------------------------------------------------

    def add_inline(self, paragraph: DocxParagraph, text: str, size: float | None = None,
                   italic: bool = False, bold: bool = False) -> None:
        """Append text with inline ``**bold**``/``*italic*``/```code``` markup as runs.

        Args:
            paragraph: Target paragraph.
            text: Source text with inline markup.
            size: Optional font size override (pt).
            italic: Force italic on every run.
            bold: Force bold on every run.
        """
        for piece in parse_inline(text):
            self._add_inline_run(paragraph, piece, size, italic, bold)

    def _add_inline_run(self, paragraph: DocxParagraph, piece: InlineRun, size: float | None,
                        italic: bool, bold: bool) -> None:
        """Append one formatted inline run."""
        run = paragraph.add_run(piece.text)
        run.bold = piece.bold or bold or None
        run.italic = piece.italic or italic or None
        if size is not None:
            run.font.size = Pt(size)
        if piece.code:
            set_run_font(run, self.style.code_font)
            run.font.size = Pt((size or self.style.body_size) - 1)

    # ---- block renderers --------------------------------------------------------------------

    def add_title_block(self, fm: FrontMatter) -> None:
        """Render the page-1 title block (title, subtitle, authors, affiliation, course, date).

        Args:
            fm: Front matter.
        """
        s = self.style
        lines: list[tuple[str, float, bool, str, float]] = [
            (fm.title, s.title_size, True, s.accent_rgb, 4),
            (fm.subtitle, s.subtitle_size, False, s.muted_rgb, 10),
            (", ".join(fm.authors), s.body_size, True, "000000", 3),
            (fm.affiliation, s.body_size - 1, False, s.muted_rgb, 1),
            (" · ".join(x for x in (fm.course, fm.date) if x), s.body_size - 1, False,
             s.muted_rgb, 10),
        ]
        last: DocxParagraph | None = None
        for text, size, bold, rgb, after in lines:
            if text:
                last = self._title_line(text, size, bold, rgb, after)
        if last is not None:
            set_paragraph_border(last, "bottom", 6, s.rule_rgb, space=8)
            last.paragraph_format.space_after = Pt(14)

    def _title_line(self, text: str, size: float, bold: bool, rgb: str,
                    space_after: float) -> DocxParagraph:
        """Add one centred line of the title block."""
        paragraph = self.doc.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fmt.space_after = Pt(space_after)
        fmt.line_spacing = 1.0
        fmt.keep_with_next = True
        self.add_inline(paragraph, text, size=size, bold=bold)
        for run in paragraph.runs:
            run.font.color.rgb = RGBColor.from_string(rgb)
        return paragraph

    def add_heading(self, block: Heading) -> None:
        """Render a heading with the matching built-in Heading style."""
        paragraph = self.doc.add_paragraph(style=f"Heading {block.level}")
        self.add_inline(paragraph, block.text)
        self.stats.headings += 1

    def add_paragraph(self, block: Paragraph) -> None:
        """Render a justified body paragraph."""
        self.add_inline(self.doc.add_paragraph(), block.text)
        self.stats.paragraphs += 1

    def add_list(self, block: ListBlock) -> None:
        """Render list items with hanging indents and generated bullets/numbers."""
        counters = [0, 0]
        for index, item in enumerate(block.items):
            if item.level == 0:
                counters[1] = 0
            counters[item.level] += 1
            is_last = index == len(block.items) - 1
            self._add_list_item(item, self._list_marker(item, counters[item.level]), is_last)
        self.stats.lists += 1

    @staticmethod
    def _list_marker(item: ListItem, number: int) -> str:
        """Return the bullet glyph or number label for an item."""
        if not item.ordered:
            return BULLET_GLYPHS[item.level]
        return f"{number}." if item.level == 0 else f"{chr(ord('a') + (number - 1) % 26)})"

    def _add_list_item(self, item: ListItem, marker: str, is_last: bool) -> None:
        """Add one list paragraph: marker, tab, then the item text."""
        indent = Cm(self.style.list_indent_cm * (item.level + 1))
        paragraph = self.doc.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.left_indent = indent
        fmt.first_line_indent = -Cm(self.style.list_indent_cm)
        fmt.tab_stops.add_tab_stop(indent, WD_TAB_ALIGNMENT.LEFT)
        fmt.space_after = Pt(self.style.para_space_after if is_last else 2)
        paragraph.add_run(f"{marker}\t")
        self.add_inline(paragraph, item.text)

    def add_table(self, block: Table) -> None:
        """Render a captioned, light-grid table with a shaded bold header row."""
        self.stats.tables += 1
        if block.caption:
            self._caption(f"Table {self.stats.tables} {EN_DASH} {block.caption}", above=True)
        table = self.doc.add_table(rows=1 + len(block.rows), cols=len(block.header))
        table.style = self.doc.styles["Table Grid"]
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        set_table_borders(table, self.style.table_border_rgb)
        set_table_cell_margins(table, 0.04, 0.12)
        self._fill_table(table, block)
        self._apply_column_widths(table, block)
        self._spacer()

    def _fill_table(self, table: DocxTable, block: Table) -> None:
        """Write header and body cells (9 pt, header bold + shaded, per-column alignment)."""
        mark_header_row(table.rows[0])
        for r, cells in enumerate([block.header, *block.rows]):
            for c, text in enumerate(cells):
                cell = table.cell(r, c)
                paragraph = cell.paragraphs[0]
                fmt = paragraph.paragraph_format
                fmt.alignment = ALIGNMENTS[block.aligns[c]]
                fmt.space_after = Pt(0)
                fmt.line_spacing = 1.0
                fmt.keep_with_next = r < len(block.rows)
                self.add_inline(paragraph, text, size=self.style.small_size, bold=r == 0)
                if r == 0:
                    shade_cell(cell, self.style.table_header_fill)

    def _apply_column_widths(self, table: DocxTable, block: Table) -> None:
        """Size columns from content so LibreOffice (which ignores autofit) lays out well."""
        widths = estimate_column_widths(block.header, block.rows, self.style.text_width_cm)
        for c, width in enumerate(widths):
            table.columns[c].width = Cm(width)
            for cell in table.columns[c].cells:
                cell.width = Cm(width)

    def _spacer(self) -> None:
        """Add a small empty paragraph after a table."""
        fmt = self.doc.add_paragraph().paragraph_format
        fmt.space_after = Pt(0)
        fmt.line_spacing = Pt(8)

    def _caption(self, text: str, above: bool) -> None:
        """Add an italic 9 pt caption; captions above a table stay with it.

        Args:
            text: Full caption including the "Table N –"/"Figure N –" prefix.
            above: True for table captions (placed before the object).
        """
        paragraph = self.doc.add_paragraph(style="Caption")
        fmt = paragraph.paragraph_format
        fmt.space_before = Pt(8 if above else 2)
        fmt.space_after = Pt(4 if above else 10)
        fmt.keep_with_next = above
        self.add_inline(paragraph, text)

    def add_image(self, block: Image) -> None:
        """Render a centred figure with a numbered caption below.

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        path = (self.assets_dir / block.path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path} (from '{block.path}', "
                                    f"resolved against {self.assets_dir})")
        width_cm = min(block.width_cm or self.style.default_image_width_cm,
                       self.style.text_width_cm)
        paragraph = self.doc.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fmt.keep_with_next = True
        fmt.space_before = Pt(6)
        fmt.space_after = Pt(2)
        fmt.line_spacing = 1.0
        shape = paragraph.add_run().add_picture(str(path), width=Cm(width_cm))
        self._limit_height(shape)
        self.stats.figures += 1
        self.stats.figure_paths.append(path)
        self._caption(f"Figure {self.stats.figures} {EN_DASH} {block.caption}", above=False)

    def _limit_height(self, shape: object) -> None:
        """Scale a picture down proportionally if it is taller than the allowed height."""
        max_height = Cm(self.style.max_image_height_cm)
        if shape.height > max_height:
            ratio = max_height / shape.height
            shape.width = Emu(int(shape.width * ratio))
            shape.height = Emu(int(max_height))

    def add_quote(self, block: BlockQuote) -> None:
        """Render a key-findings box: indented, italic, shaded, with an accent bar."""
        s = self.style
        paragraph = self.doc.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.left_indent = Cm(0.6)
        fmt.right_indent = Cm(0.6)
        fmt.space_before = Pt(6)
        fmt.space_after = Pt(10)
        set_paragraph_border(paragraph, "left", 24, s.quote_bar_rgb, space=8)
        shade_paragraph(paragraph, s.quote_fill)
        self.add_inline(paragraph, block.text, italic=True)
        self.stats.quotes += 1

    def add_reference(self, block: Reference) -> None:
        """Render one bibliography entry with a hanging indent at 9 pt."""
        indent = Cm(self.style.reference_indent_cm)
        paragraph = self.doc.add_paragraph()
        fmt = paragraph.paragraph_format
        fmt.left_indent = indent
        fmt.first_line_indent = -indent
        fmt.tab_stops.add_tab_stop(indent, WD_TAB_ALIGNMENT.LEFT)
        fmt.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt.line_spacing = 1.0
        fmt.space_after = Pt(3)
        self.add_inline(paragraph, f"[{block.label}]\t{block.text}", size=self.style.small_size)
        self.stats.references += 1

    def add_page_break(self, _block: PageBreak) -> None:
        """Insert a hard page break."""
        self.doc.add_page_break()
        self.stats.page_breaks += 1

    # ---- driver -----------------------------------------------------------------------------

    def render(self, document: Document) -> BuildStats:
        """Render a parsed document into ``self.doc``.

        Args:
            document: Parsed Markdown.

        Returns:
            Element counts.
        """
        if document.front_matter is not None:
            self.add_title_block(document.front_matter)
            self._set_core_properties(document.front_matter)
        handlers = {
            Heading: self.add_heading,
            Paragraph: self.add_paragraph,
            ListBlock: self.add_list,
            Table: self.add_table,
            Image: self.add_image,
            BlockQuote: self.add_quote,
            Reference: self.add_reference,
            PageBreak: self.add_page_break,
        }
        for block in document.blocks:
            handlers[type(block)](block)
        return self.stats

    def _set_core_properties(self, fm: FrontMatter) -> None:
        """Copy title/authors into the DOCX metadata."""
        self.doc.core_properties.title = fm.title
        self.doc.core_properties.subject = fm.subtitle
        self.doc.core_properties.author = "; ".join(fm.authors)

    def save(self, path: Path) -> None:
        """Write the DOCX, creating parent directories.

        Args:
            path: Output ``.docx`` path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def build_docx(source: Path, output: Path, assets_dir: Path | None = None,
               style: ReportStyle | None = None) -> BuildStats:
    """Convert a Markdown report into a DOCX file.

    Args:
        source: Markdown file.
        output: Destination ``.docx``.
        assets_dir: Base directory for image paths (defaults to the source's directory).
        style: Visual settings (defaults to :class:`ReportStyle`).

    Returns:
        Counts of rendered elements.

    Raises:
        FileNotFoundError: If the source or any referenced image is missing.
    """
    if not source.is_file():
        raise FileNotFoundError(f"Markdown source not found: {source}")
    document = parse_markdown(source.read_text(encoding="utf-8"))
    builder = DocxReportBuilder(style or ReportStyle(), assets_dir or source.parent)
    stats = builder.render(document)
    builder.save(output)
    logger.info(f"Wrote {output} ({stats.headings} headings, {stats.tables} tables, "
                f"{stats.figures} figures, {stats.references} references)")
    return stats


def convert_to_pdf(docx_path: Path, outdir: Path, soffice: str = "soffice",
                   timeout_s: int = 300) -> Path:
    """Convert a DOCX to PDF with headless LibreOffice.

    Args:
        docx_path: Input ``.docx``.
        outdir: Directory receiving the PDF (same stem as the DOCX).
        soffice: LibreOffice executable name or path.
        timeout_s: Conversion timeout in seconds.

    Returns:
        Path of the written PDF.

    Raises:
        FileNotFoundError: If LibreOffice is not installed.
        RuntimeError: If the conversion fails or produces no PDF.
    """
    executable = shutil.which(soffice)
    if executable is None:
        raise FileNotFoundError(f"LibreOffice executable '{soffice}' not found on PATH")
    pdf_path = outdir / f"{docx_path.stem}.pdf"
    pdf_path.unlink(missing_ok=True)
    cmd = [executable, "--headless", "--convert-to", "pdf", "--outdir", str(outdir),
           str(docx_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    if result.returncode != 0 or not pdf_path.is_file():
        raise RuntimeError(
            f"PDF conversion failed (exit {result.returncode}); is another LibreOffice instance "
            f"running? stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}"
        )
    logger.info(f"Wrote {pdf_path}")
    return pdf_path
