\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tthesis}[2021/06/21 Example LaTeX class]

\LoadClass[a4paper,english,twoside,openright,11pt]{book}

% makes PDF searchable
\usepackage{cmap}

\usepackage[english]{babel}

%\usepackage[latin9]{inputenc}
\usepackage[latin1]{inputenc}
\usepackage{lmodern} % Latin Modern font
\usepackage[T1]{fontenc}

\usepackage{csquotes}

\usepackage{textcomp} % needed for fontenc
%\usepackage{styles/extref}
\usepackage{xurl}
\usepackage[final]{listings}
	%\lstset{numbers=left,breaklines}
\usepackage{acronym}
\usepackage{makeidx}
	\makeindex
\usepackage[binding=0.8cm]{styles/layaureo} % better handling of A4 format
\usepackage{tocbibind} 
\usepackage[labelfont=bf]{caption}[2004/07/16] % format of captions
\usepackage{subfig}

% tables
\usepackage{booktabs}
\usepackage{tabulary}

% citations at the beginning of each chapter
\usepackage{epigraph}
	\setlength{\epigraphwidth}{0.59\textwidth}
	\renewcommand{\epigraphsize}{\small}
	\renewcommand{\epigraphflush}{flushleft}

\usepackage{styles/reallyblank} % insert empty page at the beginning of a chapter with empty headings

% the following is for thesis in Italian language; adapt it for English ones
\usepackage{sistyle}
\AddToSIstyle{English}{
   \SIdecimalsign{.}
   \SIthousandsep{\,}
   \SIproductsign{\cdot}
   \SIunitsep{\,}
   \SIunitspace{\,}
   \SIunitdot{\cdot}
   \SIobeyboldfalse
   \SIgroupfourfalse}
\SIstyle{English}

% used to use the footnote inside a figure's caption
\usepackage{afterpage}

\usepackage{gensymb}
\usepackage{wasysym}
\usepackage{amsfonts}
\usepackage{graphicx}

\usepackage{verbatim}
\usepackage{courier}
\usepackage{float}

\lstset{numbersep=5pt,
numberstyle=\tiny,
numbers=none,
frame=lines,
basicstyle=\linespread{1.0}\ttfamily\footnotesize
}

\usepackage{cleveref}

\newcommand{\printtitle}{\@title}
\newcommand{\printauthor}{\@author}
\newcommand{\regnumber}{N. Matricola}

% Set linespread to 1 in the verbatim environment
\makeatletter
\def\verbatim@font{\linespread{1}\normalfont\ttfamily}
\makeatother

% Set default linespread for the whole document
\renewcommand{\baselinestretch}{1.3}

%-------------------------------------------------------------------
% Frontpage
%-------------------------------------------------------------------

\makeatletter
\newcommand{\graduation}[1]{\def\@graduation{#1}}
\newcommand{\printgraduation}{\@ifundefined{@graduation}{}{\@graduation}}
\newcommand{\academicyear}[1]{\def\@academicyear{#1}}
\newcommand{\printacademicyear}{\@ifundefined{@academicyear}{}{\@academicyear}}
\newcommand{\supervisor}[1]{\def\@supervisor{#1}}
\newcommand{\printsupervisor}{\@ifundefined{@supervisor}{}{\@supervisor}}
\makeatother

\newcommand{\frontpagefile}{frontpages/frontpage-unipv.inc}

\newcommand{\printfrontpage}{%
\input{\frontpagefile}%
\thispagestyle{empty}%
\mbox{}%
\newpage%
}

