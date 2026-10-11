# Thesis

This folder holds the master thesis about Lizard on Llama-3.2-1B.

| File | Contents |
|---|---|
| [`thesis.pdf`](thesis.pdf) | The first full draft: 51 pages of main text, front matter and appendices A–G |
| [`outline.md`](outline.md) | The base outline with the sources of each section |
| [`src/`](src/) | The LaTeX source, the plot script, the data of the MMLU appendix and the lint tool |

## Build

The build needs TeX Live (`pdflatex`, `bibtex`, `latexmk`). The plots need Python 3 with matplotlib and numpy.

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
- ARC-Challenge of `window_rope` (36.1, `acc_norm`) comes from the evaluation of 2026-10-11. The notes do not have it yet.

## Status

The draft is not final. The author must check every statement, number and reference. HellaSwag and WinoGrande of `window_rope` are still missing in Table 9.6.
