%---------------------------------------------------------------------------
%
% Bachelor in AI Thesis Style File for LaTeX
% Author: Rafael Penaloza
% Version: 2024-03-10
%
% Based on:
% ÖREBRO UNIVERSITY THESIS STYLE FILE FOR LATEX
% By: Martin Magnusson (martin.magnusson@oru.se)
%---------------------------------------------------------------------------
% USAGE:
%
% \documentclass[options]{bai-thesis}
%
% For the final version you also need to include a number of new commands, which
% are discussed below.
%
% For each option below, we mention whether or not the option should be used for
% the final submitted version.
%
%---------------------------------------------------------------------------
% OPTIONS:
%
% [mas] - FINAL: YES (if Master's thesis)
%   Make masters thesis (one of doc, mas, or lic is needed).
%
% [bac] - FINAL: YES (if Bachelor thesis)
%   Make bachelor thesis (one of doc, mas, or lic is needed).
%
% [rep] - FINAL: YES (if report)
%   Make a report (not a thesis).
%
% [notfinished] - FINAL: NO
%   Print "draft" marks on output, to mark that the document is not
%   finished. Works correctly only together with nopageheaders.
%   With pageheaders it puts draft only on title pages.
%
% [osf] - FINAL: NO
%   Use Sabon "old style figures". This is recommended by the university
%   library, but looks bad in math mode. Not applicable if [times] is used.
%
% [decoratedheadings] - FINAL: NO
%   Use decorated chapter and section headings
%
% [nopageheaders] - FINAL: NO
%   Do NOT use page headers. Instead: number at bottom of the page
%
% [notitle] - FINAL: NO
%   Skip title page.
%
% [bibliography] - FINAL: NO
%   Changes bibname to Bibliography instead of References.
%
% [dyslexic] - 
%   Use dyslexic-friendly font
%---------------------------------------------------------------------------
% COMMANDS:
%
% \oruseries{n} - FINAL: YES
%   Default: none (get from print editors)
%
% \isbn{yournumber} - FINAL: YES
%   Default: none (get from print editors, must be set for doc)
%
% \issn{yournumber} - FINAL: YES
%   Default: none (get from print editors)
%
% \printer{yourprinter} - FINAL: YES
%   Default: none (get from print editors)
%
% \publisher{yourpublisher} - FINAL: YES (default is usually correct)
%   Default: University of Pavia
%
% \publisherweb{yourpublisherwebpage} - FINAL: YES (default is usually correct)
%   Default: www.bai.unipv.it
%
% \editor{youreditor} - FINAL: NO (unless told otherwise)
%   Default: none
%
% \editoremail{youreditoremail} - FINAL: NO (unless told otherwise)
%   Default: none (only shown if editor is set)
%
% \supervisor{yoursupervisor} - FINAL: NO (unless told otherwise)
%   Default: none (can be used more than once)
%
% \examinor{yourexaminor} - FINAL: NO (unless told otherwise)
%   Default: none (can be used more than once)
%
% \subject{subject area}
%
%---------------------------------------------------------------------------


%======================================================================
% Identification
%======================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{bai-thesis}[2024/03/10 v0.1 Last revised by Rafael Penaloza]

%======================================================================
% Options:
%======================================================================

% Whether or not to print ``draft'' watermarks (calling this option ``draft''
% screws up figures, possibly because of hyperref):
\newif\ifnotfinished
\DeclareOption{notfinished}{\notfinishedtrue}

% Whether to use Sabon OSF
\newif\ifosf
\DeclareOption{osf}{\osftrue}

% Whether to use decorated headings
\newif\ifdecoratedheadings
\DeclareOption{decoratedheadings}{\decoratedheadingstrue}

% Whether to print title page
\newif\ifnotitle
\DeclareOption{notitle}{\notitletrue}

% Whether to print official logos and stuff
\newif\ifplaintitle
\DeclareOption{plaintitle}{\plaintitletrue}

% Whether to use page headers or not
\newif\ifnopageheaders
\DeclareOption{nopageheaders}{\nopageheaderstrue}

% Type of thesis
\newif\ifmas
    \DeclareOption{mas}{\mastrue}
\newif\ifbac
    \DeclareOption{bac}{\bactrue}
\newif\ifrep
    \DeclareOption{rep}{\reptrue}

% Use this option if the thesis is primarily for digital publishing.
\newif\ifscreen
    \DeclareOption{screen}{\screentrue}

% Whether to use References or Bibliography
\newif\ifbibliography
    \DeclareOption{bibliography}{\bibliographytrue}
    
\newif\ifdyslexic
    \DeclareOption{dyslexic}{\dyslexictrue}

\ProcessOptions

% Make sure mas, bac, or rep are set
\ifmas
\else
  \ifbac
  \else
    \ifrep
    \else
      \ClassError{oru-thesis}{Type not set}{Use one of mas, bac, or rep}
    \fi
  \fi
\fi

%======================================================================
% Package loading
%======================================================================
%\LoadClass[openany]{book}
\LoadClass[]{report}
%======================================================================
% Commands
%======================================================================
\newcommand{\theissn}{}
\newcommand{\issn}[1]{\renewcommand{\theissn}{#1}}

\newcommand{\theisbn}{}
\newcommand{\isbn}[1]{\renewcommand{\theisbn}{#1}}

\newcommand{\theoruseries}{}
\newcommand{\oruseries}[1]{\renewcommand{\theoruseries}{#1}}

\newcommand{\thepublisher}{University of Pavia}
\newcommand{\publisher}[1]{\renewcommand{\thepublisher}{#1}}

\newcommand{\thepublisherweb}{www.unipv.it}
\newcommand{\publisherweb}[1]{\renewcommand{\thepublisherweb}{#1}}

\newcommand{\theprinter}{}
\newcommand{\printer}[1]{\renewcommand{\theprinter}{#1}}

\newcommand{\theeditor}{}
\newif\ifhasEditor
\newcommand{\editor}[1]
{
  \expandafter\def\expandafter\theeditor\expandafter{#1}
  \hasEditortrue
}

\newcommand{\theeditoremail}{}
\newif\ifhasEditorEmail
\newcommand{\editoremail}[1]
{
  \expandafter\def\expandafter\theeditoremail\expandafter{#1}
  \hasEditorEmailtrue
}

\newcommand{\thesupervisors}{}
\newif\ifhasSupervisors
\newif\ifhasMultipleSupervisors
\newcommand{\supervisor}[1]
{
  \ifhasSupervisors
    \expandafter\def\expandafter\thesupervisors\expandafter{\thesupervisors \newline{} #1}
    \hasMultipleSupervisorstrue
  \else
    \expandafter\def\expandafter\thesupervisors\expandafter{#1}
  \fi
  \hasSupervisorstrue
}

\newcommand{\thesubject}{}
\newif\ifhasSubject
\newcommand{\subject}[1]
{
  \expandafter\def\expandafter\thesubject\expandafter{#1}
  \hasSubjecttrue
}

\newcommand{\theAY}{}
\newif\ifhasAY
\newcommand{\AY}[1]
{
  \expandafter\def\expandafter\theAY\expandafter{#1}
  \hasAYtrue
}

%======================================================================
% Layout:
%======================================================================
\usepackage[
  a4paper,
  right  = 35mm,
  left   = 35mm,
  top    = 33mm,
  bottom = 20mm,
  marginparwidth={35mm},
  marginparsep={5mm},
  includemp,
  includefoot,
  reversemp,
  nomarginpar,
]{geometry}
  
\newlength{\marginwidth}
\setlength{\marginwidth}{\marginparwidth}
\addtolength{\marginwidth}{\marginparsep}

\newlength{\bodywidth}
\setlength{\bodywidth}{\textwidth}
\addtolength{\bodywidth}{\marginwidth}

% Hairline line thickness for frames (keylines and such)
\linethickness{0.25pt}

% == For empty page =================================================

\let\origdoublepage\cleardoublepage
\newcommand{\clearemptydoublepage}{%
  \clearpage
  {\pagestyle{empty}\origdoublepage}%
}
\let\cleardoublepage\clearemptydoublepage

% == Fonts used: =====================================================
% Original proprietary fonts replaced to closest open-source version available on Overleaf

\usepackage[T1]{fontenc}
\usepackage{textcomp}
    \ifosf
      % Palladio with OSF:
      \usepackage[osf]{mathpazo}
      \renewcommand{\rmdefault}{pplj}
    \else
      % Palladio without OSF:
      \usepackage[sc]{mathpazo}
      \renewcommand{\rmdefault}{pplx}
    \fi
      \ifosf
        \usepackage[euler-digits]{eulervm}%[small]
      \else
        \usepackage[]{eulervm}%[small]
      \fi
      \providecommand{\vect}[1]{\ensuremath{\vec{#1}}}
      \providecommand{\matr}[1]{\ensuremath{\mathbold{#1}}}
    % Trade Gothic is replaced to FiraSans
    \usepackage[]{FiraSans}
    \def\firasans{\fontfamily{FiraSans-\fira@figurealign OsF}}
%   \def\firasans{\fontfamily{FiraSans-\fira@figurealign LF}}
%   \def\firasans{\fontfamily{FiraSans-T\fira@figurestyle}}
%   \def\firasans{\fontfamily{FiraSans-\fira@figurestyle}}
	% FiraSans weights: t,l,el,ul,mb,k,sb,eb,h
	\renewcommand{\sffamily}{\firasans\fontseries{sb}\selectfont}

% No extra large spaces after end of sentence.
\frenchspacing

\raggedbottom

%% Rules to get rid of widows and orphans, but not very effective.
%% Use manual \looseness adjustments instead, if \flushbottom is in
%% effect. (Does PDFTeX have a problem with this?)
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

%======================================================================
% Headers:
%======================================================================
  \usepackage{fancyhdr}
  \pagestyle{fancy}
  \renewcommand{\headrulewidth}{0pt}%{0.65pt}%
  \ifnopageheaders
    \fancyhf{}
    \ifnotfinished
      \lfoot{\textit{Draft}}
    \else
      \lfoot{}
    \fi
    \cfoot{\thepage}
    \ifnotfinished
      \rfoot{\textit{\@date}}
    \else
      \rfoot{}
    \fi
  \else
    \renewcommand{\headrulewidth}{0pt}
    \fancyhead[LE,RO]{\rmfamily \small \thepage} %{\sffamily\itshape\thepage}
    \fancyhead[LO]{\rmfamily\itshape\small\rightmark} %{\sffamily\itshape\rightmark}
    \fancyhead[RE]{\rmfamily\itshape\small\leftmark} %{\sffamily\itshape\leftmark}
    \fancyfoot[C]{} % {}
    \fancypagestyle{plain}{%
    \fancyhf{} % clear all header and footer fields
    \fancyfoot[C]{\rmfamily\small\thepage} % except the center %{\sffamily\itshape\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}}
  \fi
  \addtolength{\headwidth}{\oddsidemargin}

%======================================================================
% Title pages:
%======================================================================
\renewcommand{\maketitle}{
  \begin{titlepage}%
    \pagestyle{empty}
    \enlargethispage{20mm}
    %
    % Centering hack:
    \begin{list}{}{%
      \setlength{\topsep}{0pt}%
      \setlength{\leftmargin}{-\oddsidemargin}%
      \setlength{\rightmargin}{0cm}%
      \setlength{\listparindent}{\parindent}%
      \setlength{\itemindent}{\parindent}%
      \setlength{\parsep}{\parskip}}%
      \item[]%
      \centering%
      %
      \ifplaintitle
        % Skip extra front matter
      \else
      \large\fontfamily{pplj}\selectfont
      \vspace*{0.1\textheight}
      \noindent
      \includegraphics[width=4.0cm]{logo/unimi.png} \hfill
      \includegraphics[width=4.0cm]{logo/unimib.jpg} \hfill
      \includegraphics[width=4.0cm]{logo/unipv.pdf}
      \\
      \vspace*{4ex}
        \ifmas
          \textit{Master Thesis}\\
        \else
          	\ifbac
              \textit{Bachelor Thesis}\\
            \else
              \ifrep
                \textit{Report}\\
              \else
                \textit{}\\
              \fi
            \fi
        \fi
      \vspace*{2.5em}
      \Large\fontfamily{pplj}\selectfont \@title\\
      \vspace*{6em}
      \large\fontfamily{pplj}\selectfont \textsc{\@author}\\
      \ifhasSubject
      \textit{\thesubject} \\
      \fi
      \ifhasAY
        \vspace*{3em}
        Academic Year \theAY
      \fi
            
      \vspace*{\fill}
      \footnotesize\fontfamily{pplj}\selectfont
      \textit{Bachelor in Artificial Intelligence}\\

      \textsc{Pavia} \the\year
      \ifscreen
      \else
          \cleardoublepage
          \vspace*{0.09\textheight}%
          \Large\firasans\selectfont\@title\\
          \cleardoublepage
      \fi
      \clearpage
      \vspace*{0.09\textheight}%
      \Large\fontfamily{pplj}\selectfont

      \textit{Bachelor in Artificial Intelligence}\\

      \vspace*{3ex}%
      \includegraphics[width=1.7cm]{logo/unimi.png} \qquad
      \includegraphics[width=1.7cm]{logo/unimib.jpg} \qquad 
      \includegraphics[width=1.7cm]{logo/unipv.pdf}
      \\
      
      \fi % end \ifplaintitle
      \vspace*{7ex}%
      \Large\firasans\selectfont\@author\\
      \vspace*{3ex}%
      \huge\firasans\fontseries{sb}\selectfont\@title\\
      \ifnotfinished
        \vspace*{7ex}
        \large\sffamily
        \@date \\  DRAFT
      \else \ifplaintitle \else

        \vfill
        \newcommand{\pind}{0.20\textwidth}
        \ifhasSupervisors
          \normalsize
          \ifhasMultipleSupervisors
            \parbox[t]{\pind}{Supervisors:}
          \else
            \parbox[t]{\pind}{Supervisor:}
          \fi
          \parbox[t]{0.50\textwidth}{\thesupervisors}
          \newline
        \fi


        % \clearpage
        \vspace*{\fill}
        \large\fontfamily{pplj}\selectfont%
        \copyright\ \@author, \the\year

        % \normalsize
        % \vspace*{3ex}
        % \textit{Title: } \@title

      \fi
      \fi
    \end{list} % End of centering hack
  \end{titlepage}%
}

%======================================================================
% Headings:
%======================================================================
    \ifdecoratedheadings
        \input{decoratedheadings}
    \else

    %======================================================================
    % Chapter headings:
    %======================================================================
    
    % Numbered chapters:
    \def\@makechapterhead#1{
      {
        % \cleardoublepage%
        \vspace*{0.09\textheight}%
        \noindent\raggedright
        \huge \bfseries\sffamily \@chapapp{} \thechapter \\
        \Huge \bfseries\sffamily #1
        \vskip 0.09\textheight
        \ifnopageheaders
        \else
          \thispagestyle{plain}
        \fi
      }
    }
    
    % Un-numbered chapters:
    \def\@makeschapterhead#1{
      {
        % \cleardoublepage%
        \vspace*{0.07\textheight}%
        \noindent\raggedright
        \phantom{\huge \bfseries\sffamily \@chapapp{} \thechapter} \\
        \Huge\bfseries\sffamily #1
        \vskip 0.09\textheight
        \ifnopageheaders
        \else
          \thispagestyle{plain}
        \fi
      }
    }
    
    %======================================================================
    % Section headings:
    %======================================================================
    \renewcommand\section{%
        \@startsection {section}{1}%
        {\z@}%
        {-6.5ex \@plus -1ex \@minus -.2ex}%
        {2.3ex \@plus.2ex}%
        {
          \normalfont\Large\bfseries\sffamily%
          \raggedright  %  Always use ragged right!!
          \vspace{-3mm}
        }%
    }
    
    %======================================================================
    % Subsection headings:
    %======================================================================
    \renewcommand\subsection{\@startsection{subsection}{2}%
      {\z@}%
      {-3.25ex\@plus -1ex \@minus -.2ex}%
      {1.5ex \@plus .2ex}%
      {
       \reset@font\large\bfseries\sffamily
       \raggedright
       %Hyphenated headings are ugly:
       \hyphenpenalty=5000
       \tolerance=1000
    }}
    
    %======================================================================
    % Sub-subsection headings:
    %======================================================================
    \renewcommand\subsubsection{\@startsection{subsubsection}{3}
      {\z@}%
      {-3.25ex\@plus -1ex \@minus -.2ex}%
      {1.5ex \@plus .2ex}%
      {\reset@font\normalsize\bfseries\sffamily%
    }}
    
    %======================================================================
    % Paragraph headings:
    %======================================================================
    \renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                    {3.25ex \@plus1ex \@minus.2ex}%
                    {-1em}%
                    {\reset@font\normalsize\bfseries\sffamily%
    }}
    
    %======================================================================
    % Subparagraph headings:
    %======================================================================
    \renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                    {3.25ex \@plus1ex \@minus .2ex}%
                    {-1em}%
                    {\reset@font\normalsize\bfseries\sffamily%
    }}


    \fi % end \ifdecoratedheadings

\ifnotitle
    \renewcommand{\maketitle}{}
\fi

%======================================================================
% Different font in captions:
%======================================================================
    \newcommand{\captionfonts}{\small}
    \def\@makecaption#1#2{%
    \vskip\abovecaptionskip
    \sbox\@tempboxa{\captionfonts \textbf{#1:} #2}%
    \ifdim \wd\@tempboxa >\hsize
        {\captionfonts \textbf{#1:} #2\par}
    \else
        \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
    \fi
    \vskip\belowcaptionskip}

%======================================================================
% Change name of bibliography to references by default:
%======================================================================
\ifbibliography
\else
    \providecommand*{\bibname}{}
    \renewcommand*{\bibname}{References}
\fi
\let\@internalcite\cite
\def\cite{\def\citename##1{##1}\@internalcite}
\def\shortcite{\def\citename##1{}\@internalcite}

%======================================================================
% End of file 'oru-thesis.cls'.
%======================================================================