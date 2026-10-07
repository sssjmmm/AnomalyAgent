This is the README for libertine, version 5.3.0 (2024-12-04).

This package provides support for use of the Linux Libertine and Linux
Biolinum families of fonts with LaTeX. Most features are usable with
LaTeX and dvips, pdfLaTeX, xeLaTeX and luaLaTeX, but some features are
only usable with xeLaTeX or luaLaTeX. This package compatibly replaces
several earlier packages (libertine-type1, biolinum-type1, libertine)
and should provide partial compatibility with the obsolete libertineotf
and libertine-legacy packages.

To install this package on a TDS-compliant TeX system download
the file "tex-archive"/install/fonts/libertine.tds.zip, where the
preferred URL for "tex-archive" is http://mirror.ctan.org. Unzip the
archive at the root of an appropriate texmf tree, likely a personal
or local tree. If necessary, update the file-name database (e.g.,
texhash texmf). Update the font-map files by enabling the Map file
libertine.map.

For most purposes, simply add

\usepackage{libertine}

to the preamble of your document. This will activate Libertine as the
main (serifed) text font, Biolinum as the sans font and LibertineMono
as the monospaced font. It is recommended that the font encoding be set
to T1 or LY1, but the default OT1 encoding is also supported. Available
shapes in all series (except tt) include:

it              italic
sc              small caps
scit            italic small caps

Slanted variants are not supported; the designed italic variants will be
automatically substituted. The exception is the bold series of Biolinum,
for which a designed italic is not currently available. An artificially
slanted variant has been generated and treated as if it were italic.

To activate Libertine (without Biolinum), use the libertine (or rm)
option. Similarly, to activate Biolinum (without Libertine), use the
biolinum (or sf or ss) option. To use Biolinum as the main text font
(as well as the sans font), use the option sfdefault. To suppress
activating LibertineMono, use mono=false (or tt=false). To activate
single font families, use one or more of

\usepackage{libertineRoman}
\usepackage{libertineMono}
\usepackage{biolinum}

For more advanced usage, please see the libertine.pdf document.

For compatible mathematics, it is recommended to use 

\usepackage[libertine]{newtxmath}

with [pdf]LaTeX and

  \usepackage{unicode-math}
  \setmathfont[Scale=MatchUppercase]{libertinusmath-regular.otf}

with xeLaTeX or luaLaTeX.

The original opentype fonts were created by Philipp H. Poll (gillian at
linuxlibertine.org) and are licensed under the terms of the GNU General
Public License (Version 2, with font exception) and under the terms of
the Open Font License. For details look into the doc directory of the
distribution or at

http://www.linuxlibertine.org/

The Glyph and KeyCap support was adapted from the original libertine
package by Michael Niedermair.

Three of the Libertine fonts were modified by Michael Sharpe (msharpe at
ucsd.edu) using fontforge to correct minor problems, including adding
three missing ligatures (fl, ffl, ffi) to the bold-italic font.

The type 1 fonts were created using cfftot1 or fontforge. The support
files were created using autoinst. The support files are licensed under
the terms of the LaTeX Project Public License. 

Thanks to Herbert Voss, Patrick Gundlach, Silke Hofstra, Marc Penninga,
Michael Sharpe, Denis Bitouzé and Khaled Hosny for their assistance. The
maintainer of this package is Bob Tennent (rdtennent at gmail dot com)
