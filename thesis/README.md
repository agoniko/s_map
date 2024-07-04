# LaTeX Thesis Template

`thesis_templatex` is a template for theses to be written in LaTeX.
The name of the repository reflects its purpose: `thesis temp-latex` :-D

Now ships with the possibility to select between different frontpages (currently supporting frontpages for the University of Pavia - Dept. of Industrial, Computer and Biomedical Engineerint, and the LM AI4ST).

The frontpage can be selected by renewing the following command in `thesis.tex`:

```
\renewcommand{\frontpagefile}{frontpages/frontpage-ai4st.inc}
```
Put the frontpage that is suitable for your thesis.
The available frontpages are in the `frontpages` sub-directory.

# Notes on installation

MacOS user **MUST** install the rsvg-convert software in order to convert svg to pdf.

How to install rsvg-convert run on a terminal:
```
brew install librsvg
```

# Building the pdf

If you use the terminal and have all the tools installed, you can run

```
make all
```

to generate the pdf file.

It automatically converts the figures into supported formats, and handles the correct generation of the bibliography.
It uses external tools to convert the following file formats:

* `.gif` with [ImageMagick](https://imagemagick.org/index.php) `convert`
* `.dia` with [Dia](http://dia-installer.de/download/linux.html.en)
* `.fig` with [XFig](https://www.xfig.org/)
* `.svg` with [Inkscape](https://inkscape.org/)

# Credits

* Tullio Facchinetti
* Guido Benetti
* Gianluca Roveda
