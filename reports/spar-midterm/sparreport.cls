% sparreport.cls
% LaTeX class for SPAR (Supervised Program for Alignment Research) reports
% Compatible with arXiv (TeX Live 2023+, pdflatex)

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sparreport}[2025/01/01 SPAR Report Template]

% ============================================================================
% CLASS OPTIONS
% ============================================================================
% Report type options
\newif\ifspar@midterm
\newif\ifspar@final
\newif\ifspar@arxiv

\DeclareOption{midterm}{%
  \spar@midtermtrue
  \spar@finalfalse
  \spar@arxivfalse
}
\DeclareOption{final}{%
  \spar@midtermfalse
  \spar@finaltrue
  \spar@arxivfalse
}
\DeclareOption{arxiv}{%
  \spar@midtermfalse
  \spar@finalfalse
  \spar@arxivtrue
}

% Pass all other options to article class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

% Default to midterm if no option specified
\ExecuteOptions{midterm}
\ProcessOptions\relax

% Load base class
\LoadClass[11pt]{article}

% ============================================================================
% REQUIRED PACKAGES
% ============================================================================
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}

% arXiv compatibility: pin array package version for TeX Live 2025
\RequirePackage{array}[=2016-10-06]

\RequirePackage{amsmath,amssymb,amsthm}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{microtype}

% Geometry: arxiv mode uses tighter layout, SPAR modes use 1in margins
\RequirePackage{geometry}

% Hyperref should be loaded near the end
\RequirePackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}

% For footer
\RequirePackage{fancyhdr}

% Cleveref for better cross-references (load after hyperref)
\RequirePackage{cleveref}

% ============================================================================
% PROFESSIONAL TYPOGRAPHY SETTINGS
% ============================================================================
% Prevent widows and orphans (single lines at top/bottom of page)
\widowpenalty=10000
\clubpenalty=10000

% Align bottom margin across pages
\flushbottom

% Allow slightly more flexible word spacing for better line breaks
\sloppy

% ============================================================================
% FLOAT PLACEMENT PARAMETERS
% ============================================================================
% Better default placement for figures and tables
\renewcommand{\topfraction}{0.85}        % Max fraction of page for floats at top
\renewcommand{\bottomfraction}{0.4}      % Max fraction of page for floats at bottom
\renewcommand{\textfraction}{0.1}        % Min fraction of page for text
\renewcommand{\floatpagefraction}{0.7}   % Min fraction of float page that must be floats

% ============================================================================
% PAGE GEOMETRY
% ============================================================================
\ifspar@arxiv
  % arXiv mode: use compact layout similar to arxiv.sty
  \geometry{
    textheight=9in,
    textwidth=6.5in,
    top=1in,
    headheight=14pt,
    headsep=25pt,
    footskip=30pt
  }
\else
  % SPAR modes: use comfortable 1in margins with extra footer space
  \geometry{
    margin=1in,
    footskip=0.5in
  }
\fi

% ============================================================================
% CAPTION SPACING
% ============================================================================
% Custom spacing for captions (more space above than below)
\newlength{\@abovecaptionskip}\setlength{\@abovecaptionskip}{7\p@}
\newlength{\@belowcaptionskip}\setlength{\@belowcaptionskip}{\z@}

\setlength{\abovecaptionskip}{\@abovecaptionskip}
\setlength{\belowcaptionskip}{\@belowcaptionskip}

% Swap above/below caption spacing for tables (captions go above tables)
\renewenvironment{table}
  {\setlength{\abovecaptionskip}{\@belowcaptionskip}%
   \setlength{\belowcaptionskip}{\@abovecaptionskip}%
   \@float{table}}
  {\end@float}

% ============================================================================
% PARAGRAPH FORMATTING
% ============================================================================
\ifspar@arxiv
  % arXiv mode: no indentation, space between paragraphs
  \setlength{\parindent}{\z@}
  \setlength{\parskip}{5.5\p@}
\else
  % SPAR modes: traditional indentation (use LaTeX defaults)
  % \parindent and \parskip already set by article class
\fi

% ============================================================================
% SECTION SPACING
% ============================================================================
\ifspar@arxiv
  % arXiv mode: more compact section spacing
  \renewcommand{\section}{%
    \@startsection{section}{1}{\z@}%
                  {-2.0ex \@plus -0.5ex \@minus -0.2ex}%
                  { 1.5ex \@plus  0.3ex \@minus  0.2ex}%
                  {\large\bfseries\raggedright}%
  }
  \renewcommand{\subsection}{%
    \@startsection{subsection}{2}{\z@}%
                  {-1.8ex \@plus -0.5ex \@minus -0.2ex}%
                  { 0.8ex \@plus  0.2ex}%
                  {\normalsize\bfseries\raggedright}%
  }
  \renewcommand{\subsubsection}{%
    \@startsection{subsubsection}{3}{\z@}%
                  {-1.5ex \@plus -0.5ex \@minus -0.2ex}%
                  { 0.5ex \@plus  0.2ex}%
                  {\normalsize\bfseries\raggedright}%
  }
\fi
% SPAR modes use default section spacing

% ============================================================================
% ABSTRACT STYLING
% ============================================================================
\renewenvironment{abstract}
{%
  \begin{center}%
    {\large\bfseries\abstractname}%
  \end{center}%
  \quotation
}
{%
  \endquotation
}

% ============================================================================
% METADATA STORAGE
% ============================================================================
\def\spar@round{}
\def\spar@keywords{}

% Report type prefix for title
\def\spar@reportprefix{}
\ifspar@midterm
  \def\spar@reportprefix{Midterm Report: }
\else
  \ifspar@arxiv
    \def\spar@reportprefix{}
  \else
    \def\spar@reportprefix{Final Report: }
  \fi
\fi

% ============================================================================
% USER COMMANDS
% ============================================================================

% Macro for SPAR round (hidden in arxiv mode)
\newcommand{\sparround}[1]{%
  \ifspar@arxiv\else
    \def\spar@round{#1}%
  \fi
}

% Keywords command (visible in all modes)
\def\keywordname{{\bfseries \emph{Keywords}}}%
\newcommand{\keywords}[1]{%
  \par\addvspace\medskipamount{%
    \rightskip=0pt plus1cm
    \def\and{\ifhmode\unskip\nobreak\fi\ $\cdot$
    }\noindent\keywordname\enspace\ignorespaces#1\par
  }%
}

% ============================================================================
% TITLE FORMATTING
% ============================================================================

% Store the original title
\let\spar@originaltitle\@title

% Override \maketitle to add SPAR formatting
\renewcommand{\maketitle}{%
  \begin{center}
    {\LARGE \textbf{\spar@reportprefix\@title}}
    \vspace{2em}

    % Authors in horizontal layout with wrapping support
    % Each author is in a top-aligned centered tabular
    {\renewcommand{\and}{\end{tabular}\hspace{4em}\begin{tabular}[t]{c}}%
     \begin{tabular}[t]{c}
     \@author
     \end{tabular}}

    \vspace{1em}

    % SPAR affiliation and round info (hidden in arxiv mode)
    \ifspar@arxiv\else
      \vspace{0.5em}

      {\small \href{https://sparai.org}{\color{black}\underline{Supervised Program for Alignment Research}} --- \spar@round}
    \fi
  \end{center}
}

% ============================================================================
% HEADER/FOOTER SETUP
% ============================================================================
\ifspar@arxiv
  % No special headers/footers for arxiv mode
  \pagestyle{plain}
\else
  % SPAR branding in footer for midterm/final modes
  \pagestyle{fancy}
  \fancyhf{} % Clear all headers/footers
  \fancyfoot[C]{\thepage}
  \fancyfoot[R]{\small SPAR \spar@round}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
\fi

% ============================================================================
% END OF CLASS
% ============================================================================
\endinput
