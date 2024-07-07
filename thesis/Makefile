# makefile for tex documents

FIG_DIR = ./figs

# detect the OS: if the content of OS is 'Darwin' the OS is MacOS
OS = $(shell uname)

ifndef MAIN
MAIN = thesis
endif

#ifndef PREFIX
#PREFIX = $(DATE)_
#endif
LATEX=pdflatex --shell-escape $(SYNCTEX)

$(MAIN).pdf: *.tex pdf
	$(LATEX) $(MAIN).tex
	bibtex $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

# toOl: build only if $(MAIN).pdf is not already present
all: pdf $(MAIN).pdf

# automatic generation of pdf files from sources

DIA_FILES = $(shell find $(FIG_DIR) -iname '*.dia')
PDF_DIA_FILES = $(patsubst %.dia,%.pdf, $(DIA_FILES))

FIG_FILES = $(shell find $(FIG_DIR) -iname '*.fig')
PDF_FIG_FILES = $(patsubst %.fig,%.pdf, $(FIG_FILES))

GIF_FILES = $(shell find $(FIG_DIR) -iname '*.gif')
PDF_GIF_FILES = $(patsubst %.gif,%.pdf, $(GIF_FILES))

SVG_FILES = $(shell find $(FIG_DIR) -iname '*.svg')
PDF_SVG_FILES = $(patsubst %.svg,%.pdf, $(SVG_FILES))

.PHONY: pdf

PDF_TO_BUILD = $(PDF_DIA_FILES) $(PDF_FIG_FILES) $(PDF_GIF_FILES) $(PDF_SVG_FILES)

clean:
	rm -rf *.aux *.dvi *.log *.idx *.ind *.toc *.ilg *.out *.bbl *.blg *.nav *.snm *.lof *.lot *~ *.vrb *.backup *.atfi ./figs/*.bak

cleandist: clean
	rm -rf *.ps *.pdf

cleanall: cleandist
	rm -rf  $(PDF_TO_BUILD)

pdf: $(PDF_TO_BUILD)

%.pdf: %.svg
ifeq ("$(OS)", "Darwin")
	rsvg-convert -f pdf $< -o $@
else
	inkscape --without-gui --export-area-drawing --export-pdf=$@ $<
endif

%.pdf : %.dia
	dia $< -t pdf-builtin -e $@

%.pdf : %.fig
	fig2dev -L pdf $< $@

%.pdf : %.gif
	convert $< $@
