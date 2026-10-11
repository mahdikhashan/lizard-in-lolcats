# Thesis

This folder holds the master thesis about Lizard on Llama-3.2-1B.

| File | Contents |
|---|---|
| [`thesis.pdf`](thesis.pdf) | The first full draft: 51 pages of main text, front matter and appendices A–G |
| [`outline.md`](outline.md) | The base outline with the sources of each section |
| [`src/`](src/) | The LaTeX source, the JKU template files, the plot script, the data of the MMLU appendix and the lint tool |

## Layout

The thesis uses the [JKU LaTeX thesis template](https://github.com/michaelroland/jku-templates-report-latex) by Michael Roland (commit `080904c`). The template sets the title page, the fonts, the colors, the headings and the bibliography style.

- `src/jkureport.sty`, `src/ACM-Reference-Format.*`, `src/fonts/` and `src/logos/` are copies of the template files without changes.
- The template files are under the Mozilla Public License 2.0 ([`src/JKU-TEMPLATE-LICENSE.txt`](src/JKU-TEMPLATE-LICENSE.txt)).
- The fonts are Merriweather, Public Sans and Inconsolata. Each font has its license file in `src/fonts/`.
- The thesis uses the gray JKU color scheme of the template. The template option `TNF` gives the blue color scheme of the Faculty of Engineering and Natural Sciences.
- The plots use the colors and the sans font of the template.

## Build

The build needs TeX Live with XeLaTeX, biber and `latexmk`. On Debian or Ubuntu, install these packages: `texlive-xetex`, `texlive-latex-extra`, `texlive-fonts-extra`, `texlive-bibtex-extra`, `texlive-lang-german`, `texlive-plain-generic` and `biber`. The plots need Python 3 with matplotlib and numpy.

```bash
cd docs/thesis/src
make figures PLOTPY=python3        # draws the new plots into figures/
python3 tools/make_mmlu_table.py   # writes appendix E from data/mmlu_per_subject.csv
make                               # builds thesis.pdf and copies it to docs/thesis/
make lint                          # checks the prose with the ASD-STE100 linter
```

## Sources

- The text uses only the project notes in `docs/` and the pull requests of `mahdikhashan/lolcats` and `mahdikhashan/jku-thesis`.
- The new plots take their numbers from the notes. Each data block in `src/plots/make_plots.py` names its source document.
- The existing plots come from `docs/experiments/` and `docs/figures/`. The build reads them from there.
- `src/data/mmlu_per_subject.csv` has the MMLU accuracy of each subject. The values of the teacher and of `window_rope` come from the answer files of the full MMLU run. The values of the first Lizard model come from [document 7](../07-results.md).
- ARC-Challenge (36.1) and HellaSwag (60.7) of `window_rope` (`acc_norm`) come from the evaluation of 2026-10-11. Pull request #93 adds them to the notes.

## Status

The draft is not final. The author must check every statement, number and reference. WinoGrande of `window_rope` is still missing in Table 9.6.

The title page has placeholders for the supervisor, the study program and the institute. The German abstract (Kurzfassung) is a placeholder. Each page shows the word DRAFT. To remove it, remove the line `\setwatermark` from `src/thesis.tex`.
