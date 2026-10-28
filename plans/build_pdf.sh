#!/usr/bin/env bash
# Build a single PDF from the plan markdown files so they can be read/reviewed.
#
# Usage:
#   ./build_pdf.sh                       # -> plans.pdf from 01/02/03 (default)
#   ./build_pdf.sh a.md b.md             # specific files, in order
#   OUT=review.pdf ./build_pdf.sh *.md   # custom output name
#
# Toolchain: TeX Live's `markdown` package + lualatex. No pandoc / pip / network
# required. lualatex runs markdown.sty's Lua converter in-process (no shell-escape
# and no cwd-relative helper-script lookup, unlike pdflatex) and reads the .md as
# UTF-8; the non-ASCII glyphs used in the plans are mapped via \newunicodechar.
#
# One-time deps (Ubuntu/Debian):
#   sudo apt-get install -y texlive-luatex texlive-latex-recommended \
#        texlive-latex-extra texlive-fonts-recommended texlive-plain-generic lmodern
set -euo pipefail

cd "$(dirname "$0")"
OUT="${OUT:-plans.pdf}"

# Files: args if given, else the three plan docs in order.
if [ "$#" -gt 0 ]; then
  FILES=("$@")
else
  FILES=(01_dp_plus_split.md 02_pipeline_plus_split.md 03_configurations_and_gpus.md)
fi

for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "error: no such file: $f" >&2; exit 1; }
done

command -v lualatex >/dev/null 2>&1 || {
  echo "error: lualatex not found. Install TeX Live, e.g.:" >&2
  echo "  sudo apt-get install -y texlive-luatex texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-plain-generic lmodern" >&2
  exit 1
}

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
TEX="$BUILD/plans.tex"

# --- preamble ---------------------------------------------------------------
cat > "$TEX" <<'PREAMBLE'
\documentclass[11pt]{article}
% lualatex reads source as UTF-8 natively, so no inputenc. T1 fontenc + lmodern
% still give proper hyphenation and a complete text font.
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amssymb}
\usepackage{newunicodechar}
\usepackage[margin=1in]{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\usepackage[hashEnumerators,pipeTables,fencedCode,startNumber]{markdown}

% Non-ASCII glyphs used in the plans -> explicit LaTeX so nothing silently drops.
\newunicodechar{·}{\textperiodcentered}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{–}{\textendash}
\newunicodechar{—}{\textemdash}
\newunicodechar{…}{\ldots}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{↔}{\ensuremath{\leftrightarrow}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{∪}{\ensuremath{\cup}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{‖}{\ensuremath{\|}}
\newunicodechar{≈}{\ensuremath{\approx}}

% Inline-code renderer. A plain \texttt (NOT a \url-based verbatim command): a
% url/verbatim context sets catcodes to "other", which suppresses the
% \newunicodechar substitutions and would drop math glyphs (>=, in, cup) that
% appear inside code spans. \DeclareRobustCommand + \texorpdfstring also keep it
% safe in "moving arguments" -- a code span inside a heading flows into the PDF
% bookmark/section title, where a fragile command would abort.
\DeclareRobustCommand{\ttbreak}[1]{\texorpdfstring{\texttt{#1}}{#1}}
\markdownSetup{renderers={codeSpan={\ttbreak{#1}}}}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5\baselineskip}
\sloppy

\begin{document}
PREAMBLE

# --- one \markdownInput per file, page break between -----------------------
first=1
for f in "${FILES[@]}"; do
  abs="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
  [ "$first" -eq 1 ] || printf '\\clearpage\n' >> "$TEX"
  printf '\\markdownInput{%s}\n' "$abs" >> "$TEX"
  first=0
done
printf '\\end{document}\n' >> "$TEX"

# --- compile (twice: hyperref/markdown caches) ------------------------------
# LuaTeX has an in-process Lua interpreter, so markdown.sty converts each .md
# without shelling out -- there's no `texlua ./<jobname>.markdown.lua` helper
# resolved against the cwd (that cwd-relative helper is exactly why the old
# pdflatex path silently produced an empty PDF, "Script file ...markdown.lua
# not found", when compiled from outside the build dir). We still run *inside*
# $BUILD so markdown.sty's `_markdown_*` cache and the .aux/.log/.out land in
# the temp dir rather than littering the source folder. \markdownInput paths
# are absolute, so cwd doesn't affect which .md files are read.
for i in 1 2; do
  ( cd "$BUILD" && lualatex -interaction=nonstopmode -halt-on-error \
      "$(basename "$TEX")" ) > "$BUILD/log.$i.txt" 2>&1 || {
      echo "error: lualatex failed; last lines of log:" >&2
      tail -n 40 "$BUILD/log.$i.txt" >&2
      exit 1
    }
done

cp "$BUILD/plans.pdf" "$OUT"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)) from: ${FILES[*]}"

# Warn (don't fail) on any overfull boxes so wrapping regressions are visible.
if grep -q "Overfull \\\\hbox" "$BUILD/log.2.txt"; then
  echo "note: some overfull hboxes remain:" >&2
  grep "Overfull \\\\hbox" "$BUILD/log.2.txt" | head -n 5 >&2
fi
