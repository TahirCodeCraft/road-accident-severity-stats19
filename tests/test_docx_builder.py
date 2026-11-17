"""Tests for the Markdown -> DOCX report renderer (src.reporting)."""

from __future__ import annotations

import shutil
import struct
import zlib
from pathlib import Path

import pytest
from docx import Document as load_docx
from docx.document import Document as DocxDocument

from src.reporting.docx_builder import BuildStats, build_docx, estimate_column_widths
from src.reporting.md_parser import (
    Heading,
    Image,
    ListBlock,
    MarkdownSyntaxError,
    Reference,
    Table,
    parse_inline,
    parse_markdown,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_report.md"


def write_tiny_png(path: Path, size: int = 8) -> None:
    """Write a valid ``size`` x ``size`` grey RGB PNG without any imaging library.

    Args:
        path: Destination file.
        size: Edge length in pixels.
    """
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\x80\x80\x80" * size for _ in range(size))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png + chunk(b"IEND", b""))


@pytest.fixture
def report_dir(tmp_path: Path) -> Path:
    """Copy the sample report into a tmp dir and create the figure it references."""
    shutil.copy(FIXTURE, tmp_path / "report.md")
    write_tiny_png(tmp_path / "figures" / "fig_sample.png")
    return tmp_path


@pytest.fixture
def built(report_dir: Path) -> tuple[DocxDocument, BuildStats]:
    """Build the sample report and reload the DOCX."""
    output = report_dir / "out" / "report.docx"
    stats = build_docx(report_dir / "report.md", output)
    return load_docx(str(output)), stats


def captions(doc: DocxDocument, prefix: str) -> list[str]:
    """Return texts of Caption-style paragraphs starting with ``prefix``."""
    return [p.text for p in doc.paragraphs
            if p.style.name == "Caption" and p.text.startswith(prefix)]


# ---- parser -----------------------------------------------------------------------------------


def test_parse_front_matter() -> None:
    fm = parse_markdown(FIXTURE.read_text(encoding="utf-8")).front_matter
    assert fm is not None
    assert fm.title.startswith("Predicting Road Accident Severity")
    assert len(fm.authors) == 3
    assert fm.course and fm.date and fm.affiliation and fm.subtitle


def test_parse_block_types() -> None:
    blocks = parse_markdown(FIXTURE.read_text(encoding="utf-8")).blocks
    tables = [b for b in blocks if isinstance(b, Table)]
    assert [len(t.rows) for t in tables] == [3, 4]
    assert tables[0].aligns == ["left", "center", "right", "right", "left"]
    images = [b for b in blocks if isinstance(b, Image)]
    assert [i.width_cm for i in images] == [12.0, None]
    assert len([b for b in blocks if isinstance(b, Reference)]) == 5
    nested = [item.level for b in blocks if isinstance(b, ListBlock) for item in b.items]
    assert 1 in nested and 0 in nested


def test_parse_inline_markup() -> None:
    runs = parse_inline("a **b** *c* `d` [1], [3]–[5] p * q")
    styled = [(r.text, r.bold, r.italic, r.code) for r in runs if r.bold or r.italic or r.code]
    assert styled == [("b", True, False, False), ("c", False, True, False),
                      ("d", False, False, True)]
    assert "".join(r.text for r in runs).endswith("[1], [3]–[5] p * q")


def test_heading_text_is_literal() -> None:
    doc = parse_markdown("## 3.2 Feature engineering\n")
    assert doc.blocks == [Heading(level=2, text="3.2 Feature engineering")]


def test_table_caption_without_table_fails() -> None:
    with pytest.raises(MarkdownSyntaxError):
        parse_markdown("Table: orphan caption\n\nJust text.\n")


def test_column_widths_fit_text_width() -> None:
    widths = estimate_column_widths(["a", "b"], [["x" * 200, "y" * 200]], text_width_cm=16.6)
    assert sum(widths) == pytest.approx(16.6)


# ---- builder ----------------------------------------------------------------------------------


def test_headings_rendered(built: tuple[DocxDocument, BuildStats]) -> None:
    doc, stats = built
    by_style = {lvl: [p.text for p in doc.paragraphs if p.style.name == f"Heading {lvl}"]
                for lvl in (1, 2, 3)}
    assert "1 Introduction" in by_style[1]
    assert "2.2 Feature engineering" in by_style[2]
    assert "2.2.1 Selection" in by_style[3]
    assert stats.headings == sum(len(v) for v in by_style.values())


def test_title_block_on_first_page(built: tuple[DocxDocument, BuildStats]) -> None:
    doc, _ = built
    title = doc.paragraphs[0]
    assert title.text.startswith("Predicting Road Accident Severity")
    assert all(run.bold for run in title.runs)
    assert title.runs[0].font.size.pt == 18


def test_tables_and_captions(built: tuple[DocxDocument, BuildStats]) -> None:
    doc, stats = built
    assert len(doc.tables) == stats.tables == 2
    assert captions(doc, "Table") == [
        "Table 1 – Temporal split of the STATS19 collisions used in this study",
        "Table 2 – Test-set metrics (2025) for the selected configuration of each learner",
    ]
    header = doc.tables[0].rows[0].cells[0]
    assert header.paragraphs[0].runs[0].bold
    assert "w:shd" in header._tc.xml
    assert doc.tables[1].cell(0, 1).paragraphs[0].runs[0].font.size.pt == 9


def test_figures_and_captions(built: tuple[DocxDocument, BuildStats]) -> None:
    doc, stats = built
    assert stats.figures == len(doc.inline_shapes) == 2
    figure_captions = captions(doc, "Figure")
    assert [c.split(" – ")[0] for c in figure_captions] == ["Figure 1", "Figure 2"]
    assert all(run.italic is not False for p in doc.paragraphs if p.style.name == "Caption"
               for run in p.runs)


def test_footer_has_page_field(built: tuple[DocxDocument, BuildStats]) -> None:
    doc, _ = built
    section = doc.sections[0]
    footer_xml = section.footer._element.xml
    assert "w:fldChar" in footer_xml and " PAGE " in footer_xml
    assert section.footer.paragraphs[0].text.startswith("Tahir Ahmed Khan – 25i-8025 – Page")
    assert "DSTT Project" in section.header.paragraphs[0].text
    assert round(section.page_width.cm, 1) == 21.0
    assert round(section.left_margin.cm, 1) == 2.2


def test_references_and_quote(built: tuple[DocxDocument, BuildStats]) -> None:
    doc, stats = built
    refs = [p for p in doc.paragraphs if p.text.startswith("[")]
    assert stats.references == len(refs) == 5
    assert refs[0].paragraph_format.first_line_indent < 0
    assert stats.quotes == 1


def test_missing_image_fails_loudly(tmp_path: Path) -> None:
    source = tmp_path / "r.md"
    source.write_text("![Missing](figures/nope.png)\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="nope.png"):
        build_docx(source, tmp_path / "r.docx")
