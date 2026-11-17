# Paper source

| File | Content |
|---|---|
| `main.tex` | IEEE conference paper (IEEEtran class) |
| `refs.bib` | bibliography (28 entries) |
| `figures/` | Figs. 2–5 as PDF; Fig. 1 is drawn in TikZ inside `main.tex` |
| `figures/figure_numbers.json` | the numbers plotted in the figures |
| `main.pdf` | compiled paper (7 pages) |

Figures are generated from the saved experiment outputs with
`python scripts/14_paper_figures.py`.

## Build

With TeX Live (IEEEtran is included):

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Overleaf, upload this folder and compile `main.tex` with pdfLaTeX.
