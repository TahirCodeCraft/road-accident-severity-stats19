"""Make a pandoc-friendly copy of the IEEE paper source (wording unchanged).

Cross-references, citations and the reference list are taken from LaTeX's own build output
(main.aux, main.bbl), so the Word file uses exactly the numbers printed in the PDF.
"""

import argparse
import json
import re
from pathlib import Path


def read_aux(aux: str) -> tuple[dict[str, str], dict[str, int]]:
    """Return label -> printed number, and citation key -> reference number."""
    labels = {}
    for key, val in re.findall(r"\\newlabel\{([^}]+)\}\{\{(.*?)\}\{", aux):
        labels[key] = re.sub(r"\\mbox\s*\{", "", val).replace("{", "").replace("}", "").strip()
    cites = {k: int(n) for k, n in re.findall(r"\\bibcite\{([^}]+)\}\{(\d+)\}", aux)}
    return labels, cites


def compress(nums: list[int]) -> str:
    """Format citation numbers like the LaTeX cite package: [1]–[3], [5], [7]."""
    nums = sorted(set(nums))
    runs: list[list[int]] = []
    for n in nums:
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    parts = []
    for r in runs:
        if len(r) >= 3:
            parts.append(f"[{r[0]}]–[{r[-1]}]")
        else:
            parts.extend(f"[{n}]" for n in r)
    return ", ".join(parts)


def bib_items(bbl: str) -> list[str]:
    """Reference entries from IEEEtran's .bbl, in printed order."""
    body = bbl.split("\\BIBdecl", 1)[1].split("\\end{thebibliography}")[0]
    items = []
    for chunk in re.split(r"\\bibitem\{[^}]*\}", body)[1:]:
        chunk = re.sub(r"\\BIBentry(ALT|STD)interwordspacing", "", chunk)
        chunk = chunk.replace("\\newblock", "")
        items.append(re.sub(r"\s+", " ", chunk).strip())
    return items


SYMBOLS = {r"\Delta": "Δ", r"\approx": "≈", r"\ge": "≥", r"\sigma": "σ", r"\rho": "ρ",
           r"\times": "×", r"\mid": "|"}


def plain_math(expr: str) -> str:
    """Typeset a short inline formula as text: italic variables, upright functions, Unicode."""
    s = re.sub(r"\\mathrm\{([^}]*)\}", lambda m: "\x00" + m.group(1) + "\x01", expr)
    for fn in ("ln", "log"):
        s = s.replace("\\" + fn, " \x00" + fn + "\x01 ")
    s = re.sub(r"\\hat\s*\{?(\w)\}?", "\\1\u0302", s)
    s = s.replace(r"\,", " ").replace("{,}", ",")
    for k, v in SYMBOLS.items():
        s = s.replace(k, v)
    s = re.sub(r"_\{([^}]*)\}|_(\w)", lambda m: "\x02" + (m.group(1) or m.group(2)) + "\x03", s)
    s = re.sub(r"\^\{([^}]*)\}|\^(\S)", lambda m: "\x04" + (m.group(1) or m.group(2)) + "\x05", s)
    out, upright = [], False
    for ch in s:  # every Latin letter outside \mathrm/\ln/\log is a variable -> italic
        if ch == "\x00":
            upright = True
        elif ch == "\x01":
            upright = False
        out.append(r"\textit{" + ch + "}" if ch.isascii() and ch.isalpha() and not upright
                   else ch)
    s = "".join(out).replace("\x00", "").replace("\x01", "")
    s = s.replace("\x02", r"\textsubscript{").replace("\x03", "}")
    s = s.replace("\x04", r"\textsuperscript{").replace("\x05", "}")
    s = s.replace("-", "−")
    s = re.sub(r"(?<=[\w)}])\s*([−+])\s*(?=[\w(\\])", r" \1 ", s)  # binary operators
    s = re.sub(r"\s*([=≥≈|])\s*", r" \1 ", s)
    s = re.sub(r"  +", " ", s).strip()
    s = re.sub(r"(^|[=≥≈(] ?)− ", r"\1−", s)  # unary minus stays tight
    return s


def plain_regions(tex: str) -> str:
    """Inline formulas become formatted text; numbered display equations stay as equations."""
    return re.sub(r"(?<!\\)\$([^$]+)\$", lambda k: plain_math(k.group(1)), tex)


def prep(tex: str, labels: dict[str, str], cites: dict[str, int], refs: list[str]) -> str:
    """Rewrite IEEE-specific constructs for pandoc."""
    tex = plain_regions(tex)
    tex = re.sub(r"\\big\|(.*?)\\big\|", r"\\left|\1\\right|", tex)  # ECE bars as a delimiter
    tex = re.sub(r"\\title\{.*?\\maketitle", "", tex, flags=re.S)
    tex = re.sub(r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}",
                 r"\\textbf{\\textit{Abstract}}---\1", tex, flags=re.S)
    tex = re.sub(r"\\begin\{IEEEkeywords\}\s*(.*?)\s*\\end\{IEEEkeywords\}",
                 r"\\textbf{\\textit{Index Terms}}---\1", tex, flags=re.S)
    tex = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                 r"\\includegraphics{fig_design.png}", tex, flags=re.S)
    tex = re.sub(r"figures/(fig_\w+)\.pdf", r"\1.png", tex)
    tex = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", tex)
    tex = re.sub(r"\\begin\{(table|figure)\*\}", r"\\begin{\1}", tex)
    tex = re.sub(r"\\end\{(table|figure)\*\}", r"\\end{\1}", tex)
    tex = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m.group(1)], tex)
    tex = re.sub(r"\\cite\{([^}]+)\}",
                 lambda m: compress([cites[k.strip()] for k in m.group(1).split(",")]), tex)
    tex = re.sub(r"\\label\{[^}]+\}", "", tex)
    ref_block = "\\section*{References}\n\n" + "\n\n".join(
        f"[{i}]~{r}" for i, r in enumerate(refs, 1))
    tex = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}",
                 lambda _: ref_block, tex)
    return tex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("aux", type=Path)
    ap.add_argument("bbl", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    labels, cites = read_aux(args.aux.read_text())
    refs = bib_items(args.bbl.read_text())
    assert len(refs) == len(cites), (len(refs), len(cites))
    args.out.write_text(prep(args.src.read_text(), labels, cites, refs))
    meta = {"labels": labels, "n_refs": len(refs)}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
