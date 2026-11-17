"""Report rendering: Markdown subset -> styled DOCX -> PDF (via LibreOffice)."""

from src.reporting.docx_builder import BuildStats, ReportStyle, build_docx, convert_to_pdf
from src.reporting.md_parser import parse_markdown

__all__ = ["BuildStats", "ReportStyle", "build_docx", "convert_to_pdf", "parse_markdown"]
