#!/usr/bin/env bash
# Build the defense dissertation PDF from main.tex.
# Prefers latexmk; falls back to two pdflatex passes (for the ToC/refs).
set -euo pipefail
cd "$(dirname "$0")"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
else
  echo "ERROR: need latexmk or pdflatex (e.g. apt-get install texlive-latex-recommended texlive-latex-extra)"
  exit 1
fi

echo "Built: $(pwd)/main.pdf"
