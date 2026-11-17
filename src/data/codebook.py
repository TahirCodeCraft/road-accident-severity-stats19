"""Parse the DfT data guide (code -> label lookup) and derive per-field missing codes.

Missing-code rule (plan section 1): ``-1`` is always missing; additionally any code whose label
contains "unknown", "self reported", "not known", "missing", "undefined" or "data missing or out
of range" is missing. A few labels contain "unknown" but describe a real, informative category
(e.g. "Darkness - lighting unknown"); these are listed in :data:`MISSING_CODE_EXCEPTIONS` and kept.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.paths import DATA_GUIDE_PATH

logger = logging.getLogger(__name__)

CODE_LIST_SHEET = "2024_code_list"
GUIDE_COLUMNS: tuple[str, ...] = ("table", "field", "code", "label", "note")
ALWAYS_MISSING_CODE = -1
MISSING_LABEL_PATTERN = re.compile(
    r"unknown|self reported|not known|missing|undefined|data missing or out of range",
    flags=re.IGNORECASE,
)
# (table, field) -> codes whose label matches the pattern but carry a real category.
MISSING_CODE_EXCEPTIONS: dict[tuple[str, str], frozenset[int]] = {
    ("collision", "light_conditions"): frozenset({7}),  # Darkness - lighting unknown
    ("vehicle", "vehicle_type"): frozenset({97, 98}),  # Motorcycle unknown cc / goods unknown wt
    ("casualty", "casualty_type"): frozenset({97, 98}),  # same, for casualty vehicle type
}

Code = int | str


@dataclass
class Codebook:
    """Parsed data guide.

    Attributes:
        labels: ``{table: {field: {code: label}}}``.
        missing_codes: ``{table: {field: frozenset of integer codes treated as missing}}``. Only
            fields present in the guide are listed; use :meth:`missing_for` which always adds -1.
    """

    labels: dict[str, dict[str, dict[Code, str]]] = field(default_factory=dict)
    missing_codes: dict[str, dict[str, frozenset[int]]] = field(default_factory=dict)

    def field_labels(self, table: str, name: str) -> dict[Code, str]:
        """Return ``{code: label}`` for one field (empty if absent from the guide)."""
        return self.labels.get(table, {}).get(name, {})

    def missing_for(self, table: str, name: str) -> frozenset[int]:
        """Return the integer missing codes of a field, always including ``-1``."""
        derived = self.missing_codes.get(table, {}).get(name, frozenset())
        return derived | {ALWAYS_MISSING_CODE}


def normalise_code(raw: object) -> Code | None:
    """Convert a guide code cell to ``int`` when it is integral, else a stripped string.

    Args:
        raw: Cell value (int, float, str or NaN).

    Returns:
        Integer code, string code, or ``None`` for empty cells.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else str(raw)
    text = str(raw).strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text or None


def read_code_list(path: Path = DATA_GUIDE_PATH) -> pd.DataFrame:
    """Read the code-list sheet of the data guide.

    Args:
        path: Path to ``data-guide-2025.xlsx``.

    Returns:
        DataFrame with columns ``table, field, code, label, note`` (rows without a code or label
        are dropped; codes normalised by :func:`normalise_code`).
    """
    frame = pd.read_excel(path, sheet_name=CODE_LIST_SHEET)
    frame.columns = list(GUIDE_COLUMNS)
    frame["code"] = frame["code"].map(normalise_code)
    frame = frame.dropna(subset=["code", "label"])
    frame["table"] = frame["table"].astype(str).str.strip()
    frame["field"] = frame["field"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    return frame.reset_index(drop=True)


def build_labels(code_list: pd.DataFrame) -> dict[str, dict[str, dict[Code, str]]]:
    """Group a code list into ``{table: {field: {code: label}}}``.

    Args:
        code_list: Output of :func:`read_code_list` (or an equivalent synthetic frame).

    Returns:
        Nested mapping of labels.
    """
    labels: dict[str, dict[str, dict[Code, str]]] = {}
    for row in code_list.itertuples(index=False):
        labels.setdefault(row.table, {}).setdefault(row.field, {})[row.code] = row.label
    return labels


def is_missing_label(label: str) -> bool:
    """Return True if a label denotes an unknown / missing value."""
    return bool(MISSING_LABEL_PATTERN.search(label))


def derive_missing_codes(
    labels: dict[str, dict[str, dict[Code, str]]],
) -> dict[str, dict[str, frozenset[int]]]:
    """Derive per-field integer missing codes from labels.

    Args:
        labels: ``{table: {field: {code: label}}}``.

    Returns:
        ``{table: {field: frozenset(codes)}}`` including ``-1`` for every field.
    """
    result: dict[str, dict[str, frozenset[int]]] = {}
    for table, fields in labels.items():
        for name, mapping in fields.items():
            exceptions = MISSING_CODE_EXCEPTIONS.get((table, name), frozenset())
            codes = {
                code
                for code, label in mapping.items()
                if isinstance(code, int) and is_missing_label(label) and code not in exceptions
            }
            result.setdefault(table, {})[name] = frozenset(codes | {ALWAYS_MISSING_CODE})
    return result


def load_codebook(path: Path = DATA_GUIDE_PATH) -> Codebook:
    """Parse the data guide into a :class:`Codebook`.

    Args:
        path: Path to the data guide workbook.

    Returns:
        The parsed codebook.
    """
    code_list = read_code_list(path)
    labels = build_labels(code_list)
    missing = derive_missing_codes(labels)
    n_fields = sum(len(f) for f in labels.values())
    logger.info(f"Parsed data guide: {len(code_list)} codes across {n_fields} fields")
    return Codebook(labels=labels, missing_codes=missing)


def missing_code_summary(codebook: Codebook, columns: dict[str, list[str]]) -> pd.DataFrame:
    """Tabulate the non-trivial missing codes (beyond -1) for the given columns.

    Args:
        codebook: Parsed codebook.
        columns: ``{table: [columns]}`` to report on.

    Returns:
        DataFrame with ``table, field, code, label`` for every mapped code other than -1.
    """
    rows = []
    for table, names in columns.items():
        for name in names:
            labels = codebook.field_labels(table, name)
            for code in sorted(codebook.missing_for(table, name) - {ALWAYS_MISSING_CODE}):
                rows.append(
                    {"table": table, "field": name, "code": code, "label": labels.get(code, "")}
                )
    return pd.DataFrame(rows, columns=["table", "field", "code", "label"])


def log_missing_codes(codebook: Codebook, columns: dict[str, list[str]]) -> None:
    """Log which codes (beyond -1) are mapped to missing, and the kept exceptions.

    Args:
        codebook: Parsed codebook.
        columns: ``{table: [columns]}`` to report on.
    """
    summary = missing_code_summary(codebook, columns)
    for row in summary.itertuples(index=False):
        logger.info(f"missing code {row.table}.{row.field}={row.code} ({row.label!r}) -> NaN")
    for (table, name), codes in MISSING_CODE_EXCEPTIONS.items():
        labels = codebook.field_labels(table, name)
        kept = {c: labels.get(c, "") for c in sorted(codes)}
        logger.info(f"kept as real category despite 'unknown' label: {table}.{name} {kept}")
