"""Line-based parser for the small Markdown subset used by the project report.

The parser turns a Markdown string into a flat list of typed blocks (dataclasses) that the
DOCX builder renders. Supported constructs:

* front matter between ``---`` lines (title, subtitle, authors, affiliation, course, date);
* ``#``/``##``/``###`` headings (numbering is taken literally from the text);
* paragraphs (consecutive lines are joined with a space);
* ``- `` bullet and ``1. `` numbered lists with one nesting level (two-space indent);
* GitHub pipe tables, optionally preceded by a ``Table: caption`` line;
* images ``![caption](path){width=12cm}``;
* ``\\pagebreak`` lines;
* ``> `` block quotes;
* ``[n] ...`` reference entries inside a section whose heading mentions "References";
* ``<!-- ... -->`` HTML comments (skipped).

Inline markup (``**bold**``, ``*italic*``, ```code```) is kept verbatim in block text and is
split into runs by :func:`parse_inline`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
BULLET_RE = re.compile(r"^( *)- +(.*)$")
NUMBERED_RE = re.compile(r"^( *)\d+[.)] +(.*)$")
IMAGE_RE = re.compile(
    r"^!\[(?P<caption>[^\]]*)\]\((?P<path>[^)\s]+)\)"
    r"(?:\{\s*width\s*=\s*(?P<width>\d+(?:\.\d+)?)\s*cm\s*\})?\s*$"
)
TABLE_CAPTION_RE = re.compile(r"^Table:\s*(.+?)\s*$")
TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")
REFERENCE_RE = re.compile(r"^\[(?P<label>\d+)\]\s+(?P<text>.+)$")
PAGEBREAK_TOKENS = frozenset({r"\pagebreak", r"\newpage"})
REFERENCE_HEADING_RE = re.compile(r"\b(references|bibliography)\b", re.IGNORECASE)
INLINE_RE = re.compile(r"(\*\*(?=\S)(?:.+?)(?<=\S)\*\*|`[^`]+`|\*(?=[^\s*])(?:[^*]+?)(?<=\S)\*)")
LIST_INDENT = 2


@dataclass
class FrontMatter:
    """Title-block metadata from the front matter."""

    title: str = ""
    subtitle: str = ""
    authors: list[str] = field(default_factory=list)
    affiliation: str = ""
    course: str = ""
    date: str = ""


@dataclass
class Heading:
    """Section heading (level 1-3)."""

    level: int
    text: str


@dataclass
class Paragraph:
    """Body paragraph with inline markup."""

    text: str


@dataclass
class ListItem:
    """One list item; ``level`` is 0 (top) or 1 (nested)."""

    text: str
    level: int
    ordered: bool


@dataclass
class ListBlock:
    """Contiguous run of bullet and/or numbered list items."""

    items: list[ListItem]


@dataclass
class Table:
    """Pipe table with optional caption and per-column alignment ("left"/"center"/"right")."""

    header: list[str]
    rows: list[list[str]]
    aligns: list[str]
    caption: str | None = None


@dataclass
class Image:
    """Figure reference; ``path`` is relative to the assets directory."""

    caption: str
    path: str
    width_cm: float | None = None


@dataclass
class PageBreak:
    """Explicit page break."""


@dataclass
class BlockQuote:
    """Indented call-out paragraph (key-findings box)."""

    text: str


@dataclass
class Reference:
    """Bibliography entry ``[label] text``."""

    label: str
    text: str


Block = Heading | Paragraph | ListBlock | Table | Image | PageBreak | BlockQuote | Reference


@dataclass
class Document:
    """Parsed report: front matter plus ordered blocks."""

    front_matter: FrontMatter | None
    blocks: list[Block]


@dataclass
class InlineRun:
    """A run of text with uniform inline formatting."""

    text: str
    bold: bool = False
    italic: bool = False
    code: bool = False


class MarkdownSyntaxError(ValueError):
    """Raised when the source violates the supported Markdown subset."""


# --------------------------------------------------------------------------------------------
# Inline markup
# --------------------------------------------------------------------------------------------


def parse_inline(text: str) -> list[InlineRun]:
    """Split text into runs for ``**bold**``, ``*italic*`` and ```code``` markup.

    Citation brackets such as ``[1]`` or ``[3]–[5]`` are plain text and pass through untouched.

    Args:
        text: Source text of a paragraph, list item, cell, etc.

    Returns:
        Ordered list of runs; concatenating their text gives the visible text.
    """
    runs: list[InlineRun] = []
    for token in INLINE_RE.split(text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            runs.append(InlineRun(token[2:-2], bold=True))
        elif token.startswith("`") and token.endswith("`") and len(token) > 2:
            runs.append(InlineRun(token[1:-1], code=True))
        elif token.startswith("*") and token.endswith("*") and len(token) > 2:
            runs.append(InlineRun(token[1:-1], italic=True))
        else:
            runs.append(InlineRun(token))
    return runs


def plain_text(text: str) -> str:
    """Return text with inline markup removed.

    Args:
        text: Text possibly containing inline markup.

    Returns:
        Visible text only.
    """
    return "".join(run.text for run in parse_inline(text))


# --------------------------------------------------------------------------------------------
# Front matter
# --------------------------------------------------------------------------------------------


def _strip_quotes(value: str) -> str:
    """Remove one level of matching surrounding quotes."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _inline_list(value: str) -> list[str]:
    """Parse ``[a, b, c]`` into a list of stripped strings."""
    return [_strip_quotes(item) for item in value.strip()[1:-1].split(",") if item.strip()]


def parse_front_matter(lines: list[str]) -> FrontMatter:
    """Parse the YAML-ish key/value lines of the front matter.

    Args:
        lines: Lines between the opening and closing ``---`` markers.

    Returns:
        Populated front matter.

    Raises:
        MarkdownSyntaxError: On an unknown key or malformed line.
    """
    fm = FrontMatter()
    known = set(FrontMatter.__dataclass_fields__)
    current_list: str | None = None
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        item = re.match(r"^\s+-\s+(.+)$", raw)
        if item and current_list:
            getattr(fm, current_list).append(_strip_quotes(item.group(1)))
            continue
        match = re.match(r"^(\w+)\s*:\s*(.*)$", raw)
        if not match or match.group(1) not in known:
            raise MarkdownSyntaxError(f"Bad front-matter line: {raw!r}")
        key, value = match.group(1), match.group(2).strip()
        current_list = None
        if key == "authors":
            if value.startswith("["):
                fm.authors = _inline_list(value)
            elif value:
                fm.authors = [_strip_quotes(value)]
            else:
                current_list = "authors"
        else:
            setattr(fm, key, _strip_quotes(value))
    return fm


def split_front_matter(lines: list[str]) -> tuple[FrontMatter | None, list[str]]:
    """Detach a leading ``---`` front-matter block from the body lines.

    Args:
        lines: All source lines.

    Returns:
        ``(front_matter_or_None, remaining_lines)``.

    Raises:
        MarkdownSyntaxError: If the block is opened but never closed.
    """
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
    if start >= len(lines) or lines[start].strip() != "---":
        return None, lines
    for end in range(start + 1, len(lines)):
        if lines[end].strip() == "---":
            return parse_front_matter(lines[start + 1 : end]), lines[end + 1 :]
    raise MarkdownSyntaxError("Front matter opened with '---' but never closed")


# --------------------------------------------------------------------------------------------
# Block-level helpers
# --------------------------------------------------------------------------------------------


def strip_comments(lines: list[str]) -> list[str]:
    """Drop ``<!-- ... -->`` comments (single- or multi-line) that occupy whole lines.

    Args:
        lines: Source lines.

    Returns:
        Lines with comment-only lines removed (replaced by blanks to keep block separation).
    """
    out: list[str] = []
    in_comment = False
    for line in lines:
        stripped = line.strip()
        if in_comment or stripped.startswith("<!--"):
            in_comment = "-->" not in stripped
            out.append("")
        else:
            out.append(line)
    return out


def is_table_row(line: str) -> bool:
    """Return True for a pipe-table row (starts with ``|``)."""
    return line.lstrip().startswith("|")


def split_row(line: str) -> list[str]:
    """Split a pipe-table row into stripped cell strings (``\\|`` escapes a pipe).

    Args:
        line: Table row text.

    Returns:
        Cell contents.
    """
    body = line.strip()
    body = body.removeprefix("|")
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    cells = re.split(r"(?<!\\)\|", body)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def parse_alignment(cell: str) -> str:
    """Map a separator cell (``:---``, ``---:``, ``:---:``) to an alignment name."""
    left, right = cell.startswith(":"), cell.endswith(":")
    if left and right:
        return "center"
    return "right" if right else "left"


def is_list_line(line: str) -> bool:
    """Return True for a bullet or numbered list item line."""
    return bool(BULLET_RE.match(line) or NUMBERED_RE.match(line))


def starts_block(line: str) -> bool:
    """Return True if the line begins a non-paragraph block (ends a running paragraph)."""
    stripped = line.strip()
    return bool(
        HEADING_RE.match(stripped)
        or is_list_line(line)
        or is_table_row(line)
        or IMAGE_RE.match(stripped)
        or TABLE_CAPTION_RE.match(stripped)
        or stripped.startswith(">")
        or stripped in PAGEBREAK_TOKENS
    )


# --------------------------------------------------------------------------------------------
# Block parsers: each takes (lines, index) and returns (block, next_index)
# --------------------------------------------------------------------------------------------


def parse_table(lines: list[str], i: int, caption: str | None) -> tuple[Table, int]:
    """Parse a pipe table starting at ``lines[i]``.

    Args:
        lines: Source lines.
        i: Index of the header row.
        caption: Caption text from a preceding ``Table:`` line, if any.

    Returns:
        The table block and the index after its last row.

    Raises:
        MarkdownSyntaxError: If the separator row is missing or malformed.
    """
    header = split_row(lines[i])
    if i + 1 >= len(lines) or not is_table_row(lines[i + 1]):
        raise MarkdownSyntaxError(f"Table at line {i + 1} lacks a '|---|' separator row")
    separator = split_row(lines[i + 1])
    valid = all(TABLE_SEPARATOR_CELL_RE.match(cell) for cell in separator)
    if len(separator) != len(header) or not valid:
        raise MarkdownSyntaxError(f"Malformed table separator row: {lines[i + 1]!r}")
    rows: list[list[str]] = []
    j = i + 2
    while j < len(lines) and is_table_row(lines[j]):
        cells = split_row(lines[j])
        rows.append((cells + [""] * len(header))[: len(header)])
        j += 1
    aligns = [parse_alignment(cell) for cell in separator]
    return Table(header=header, rows=rows, aligns=aligns, caption=caption), j


def parse_table_caption(lines: list[str], i: int) -> tuple[Table, int]:
    """Parse ``Table: caption`` followed (after optional blank lines) by a pipe table.

    Args:
        lines: Source lines.
        i: Index of the caption line.

    Returns:
        The captioned table and the index after it.

    Raises:
        MarkdownSyntaxError: If no table follows the caption.
    """
    match = TABLE_CAPTION_RE.match(lines[i].strip())
    assert match is not None
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    if j >= len(lines) or not is_table_row(lines[j]):
        raise MarkdownSyntaxError(f"'Table:' caption at line {i + 1} is not followed by a table")
    return parse_table(lines, j, caption=match.group(1))


def list_item_from(line: str) -> ListItem | None:
    """Build a list item from a marker line, or None if the line is not one."""
    for regex, ordered in ((BULLET_RE, False), (NUMBERED_RE, True)):
        match = regex.match(line)
        if match:
            level = 1 if len(match.group(1)) >= LIST_INDENT else 0
            return ListItem(text=match.group(2).strip(), level=level, ordered=ordered)
    return None


def parse_list(lines: list[str], i: int) -> tuple[ListBlock, int]:
    """Parse consecutive list items (indented non-marker lines continue the previous item).

    Args:
        lines: Source lines.
        i: Index of the first item.

    Returns:
        The list block and the index after it.
    """
    items: list[ListItem] = []
    j = i
    while j < len(lines) and lines[j].strip():
        item = list_item_from(lines[j])
        if item is not None:
            items.append(item)
        elif lines[j].startswith(" ") and items and not starts_block(lines[j]):
            items[-1].text = f"{items[-1].text} {lines[j].strip()}"
        else:
            break
        j += 1
    return ListBlock(items=items), j


def parse_quote(lines: list[str], i: int) -> tuple[BlockQuote, int]:
    """Parse consecutive ``>`` lines into one block quote.

    Args:
        lines: Source lines.
        i: Index of the first quote line.

    Returns:
        The quote block and the index after it.
    """
    parts: list[str] = []
    j = i
    while j < len(lines) and lines[j].strip().startswith(">"):
        parts.append(lines[j].strip()[1:].strip())
        j += 1
    return BlockQuote(text=" ".join(p for p in parts if p)), j


def parse_paragraph(lines: list[str], i: int) -> tuple[Paragraph, int]:
    """Join consecutive plain lines into one paragraph.

    Args:
        lines: Source lines.
        i: Index of the first line.

    Returns:
        The paragraph block and the index after it.
    """
    parts = [lines[i].strip()]
    j = i + 1
    while j < len(lines) and lines[j].strip() and not starts_block(lines[j]):
        parts.append(lines[j].strip())
        j += 1
    return Paragraph(text=" ".join(parts)), j


def parse_references(lines: list[str], i: int) -> tuple[list[Reference], int]:
    """Parse ``[n] text`` entries; non-marker lines continue the previous entry.

    Args:
        lines: Source lines.
        i: Index of the first reference line.

    Returns:
        Reference blocks and the index after them.
    """
    refs: list[Reference] = []
    j = i
    while j < len(lines):
        stripped = lines[j].strip()
        match = REFERENCE_RE.match(stripped)
        if match:
            refs.append(Reference(label=match.group("label"), text=match.group("text")))
        elif stripped and refs and not starts_block(lines[j]):
            refs[-1].text = f"{refs[-1].text} {stripped}"
        elif stripped:
            break
        j += 1
    return refs, j


def parse_single_line(stripped: str) -> Block | None:
    """Parse one-line blocks (heading, image, page break); None if the line is none of them."""
    if stripped in PAGEBREAK_TOKENS:
        return PageBreak()
    heading = HEADING_RE.match(stripped)
    if heading:
        return Heading(level=len(heading.group(1)), text=heading.group(2))
    image = IMAGE_RE.match(stripped)
    if image:
        width = float(image.group("width")) if image.group("width") else None
        return Image(caption=image.group("caption").strip(), path=image.group("path"),
                     width_cm=width)
    return None


def parse_blocks(lines: list[str]) -> list[Block]:
    """Parse body lines into blocks.

    Args:
        lines: Body lines (front matter already removed).

    Returns:
        Ordered blocks.
    """
    blocks: list[Block] = []
    in_references = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        single = parse_single_line(stripped)
        if single is not None:
            if isinstance(single, Heading):
                in_references = bool(REFERENCE_HEADING_RE.search(single.text))
            blocks.append(single)
            i += 1
            continue
        if in_references and REFERENCE_RE.match(stripped):
            refs, i = parse_references(lines, i)
            blocks.extend(refs)
            continue
        block: Block
        if TABLE_CAPTION_RE.match(stripped):
            block, i = parse_table_caption(lines, i)
        elif is_table_row(lines[i]):
            block, i = parse_table(lines, i, caption=None)
        elif is_list_line(lines[i]):
            block, i = parse_list(lines, i)
        elif stripped.startswith(">"):
            block, i = parse_quote(lines, i)
        else:
            block, i = parse_paragraph(lines, i)
        blocks.append(block)
    return blocks


def parse_markdown(source: str) -> Document:
    """Parse a Markdown report into front matter and blocks.

    Args:
        source: Full Markdown text.

    Returns:
        Parsed document.
    """
    lines = source.replace("\r\n", "\n").replace("\t", "    ").split("\n")
    front_matter, body = split_front_matter(lines)
    blocks = parse_blocks(strip_comments(body))
    logger.debug(f"Parsed {len(blocks)} blocks")
    return Document(front_matter=front_matter, blocks=blocks)
