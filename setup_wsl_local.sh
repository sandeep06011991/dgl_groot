#!/usr/bin/env bash
# setup_wsl_local.sh — install the LOCAL (WSL) tooling for CPU-only data
# processing of results + compiling the defense_latex PDF.
#
# This is the control/orchestration box (per CLAUDE.md): data processing of local
# result files (pandas/matplotlib) and LaTeX compilation are allowed here; builds,
# training and GPU work still go to RunPod.
#
# Ubuntu 24.04 ships an "externally managed" (PEP 668) system Python, so we install
# the Python stack from apt (clean, no --break-system-packages). A venv alternative
# is at the bottom if you prefer newer pip versions.
#
#   Usage:  bash setup_wsl_local.sh            # required bits
#           WITH_OPTIONAL=1 bash setup_wsl_local.sh   # + rclone, build-essential, xetex
set -euo pipefail

echo "== apt update =="
sudo apt-get update -y

echo "== Python data stack (pandas/numpy/matplotlib) + pip =="
sudo apt-get install -y \
  python3-pip \
  python3-pandas \
  python3-numpy \
  python3-matplotlib \
  python3-tabulate

echo "== LaTeX packages for defense_latex (tikz, xcolor[table], siunitx, booktabs) =="
# latexmk/pdflatex/bibtex/tlmgr are already present; these add the macro packages
# the chapter uses. Already-installed ones are a no-op.
sudo apt-get install -y \
  texlive-latex-extra \
  texlive-pictures \
  texlive-science \
  texlive-fonts-recommended

if [ "${WITH_OPTIONAL:-0}" = "1" ]; then
  echo "== optional: rclone (B2 from WSL), build-essential, xelatex =="
  sudo apt-get install -y build-essential texlive-xetex
  command -v rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | sudo bash
fi

echo
echo "== verify =="
python3 -c "import pandas, numpy, matplotlib; print('python:', 'pandas', pandas.__version__, '| numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)"
for c in latexmk pdflatex bibtex; do command -v "$c" >/dev/null && echo "latex:  $c $(command -v $c)"; done
echo
echo "quick checks you can now run locally:"
echo "  python3 defense_latex/gen_results_table.py --self-test"
echo "  cd defense_latex && latexmk -pdf main.tex"

# ---------------------------------------------------------------------------
# Alternative (venv) instead of the apt Python packages above:
#   sudo apt-get install -y python3-venv
#   python3 -m venv ~/.venvs/groot-local && source ~/.venvs/groot-local/bin/activate
#   pip install pandas numpy matplotlib tabulate
#   # then run tools with that venv active
# ---------------------------------------------------------------------------
