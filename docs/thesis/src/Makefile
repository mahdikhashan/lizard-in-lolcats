# Build the thesis PDF. Needs TeX Live (pdflatex, bibtex, latexmk).
# The plots need Python 3 with matplotlib and numpy: make figures PLOTPY=/path/to/python
PLOTPY ?= python3

all: ../thesis.pdf

figures:
	$(PLOTPY) plots/make_plots.py

thesis.pdf: $(wildcard *.tex frontmatter/*.tex chapters/*.tex appendices/*.tex figures/*) references.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error thesis.tex

../thesis.pdf: thesis.pdf
	cp thesis.pdf ../thesis.pdf

lint:
	python3 tools/ste_check.py

clean:
	latexmk -C

.PHONY: all figures lint clean
