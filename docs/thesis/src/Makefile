# Build the thesis PDF with the JKU template. Needs TeX Live with XeLaTeX, biber, biblatex and latexmk.
# On Debian/Ubuntu: texlive-xetex texlive-latex-extra texlive-fonts-extra texlive-bibtex-extra
# texlive-lang-german texlive-plain-generic biber.
# The plots need Python 3 with matplotlib and numpy: make figures PLOTPY=/path/to/python
PLOTPY ?= python3

all: ../thesis.pdf

figures:
	$(PLOTPY) plots/make_plots.py

thesis.pdf: $(wildcard *.tex *.sty frontmatter/*.tex chapters/*.tex appendices/*.tex figures/*) references.bib
	latexmk -pdfxe -interaction=nonstopmode -halt-on-error thesis.tex

../thesis.pdf: thesis.pdf
	cp thesis.pdf ../thesis.pdf

lint:
	python3 tools/ste_check.py

clean:
	latexmk -C

.PHONY: all figures lint clean
