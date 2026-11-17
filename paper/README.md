# Paper source

| File | Content |
|---|---|
| `main.tex` | IEEE conference paper (IEEEtran class) |
| `refs.bib` | bibliography (28 entries) |
| `figures/` | Figs. 2–5 as PDF; Fig. 1 is drawn in TikZ inside `main.tex` |
| `figures/figure_numbers.json` | the numbers plotted in the figures |
| `main.pdf` | compiled paper (7 pages) |
| `main.docx` | Word version of the same paper, generated from `main.tex` |
| `word/` | scripts that produce `main.docx` |

Figures are generated from the saved experiment outputs with
`python scripts/14_paper_figures.py`.

## Build

With TeX Live (IEEEtran is included):

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Overleaf, upload this folder and compile `main.tex` with pdfLaTeX.

## Word version

`main.docx` has the same text, tables, figures, equation and reference numbering as
`main.pdf`, in a two-column IEEE layout. It is generated, not edited by hand: section, table,
figure, equation and citation numbers are read from the LaTeX build (`main.aux`, `main.bbl`), so
they match the PDF. Line breaks and float positions differ slightly from the PDF.

Rebuild after a LaTeX build (needs `pip install pypandoc_binary python-docx`, and PNG copies of
the figures, e.g. `pdftoppm -png -r 300 -singlefile figures/fig_prauc.pdf png/fig_prauc`; Fig. 1
is the TikZ picture compiled on its own as `png/fig_design.png`):

```bash
cd paper
python word/prep_tex.py main.tex main.aux main.bbl main_pd.tex
pandoc main_pd.tex -f latex -t docx -o main_raw.docx --resource-path png
python word/ieee_style.py main_raw.docx main.tex main.docx
```
