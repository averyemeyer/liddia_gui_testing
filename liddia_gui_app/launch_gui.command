#!/usr/bin/env bash
set -euo pipefail

# Launch the modular LIDDIA GUI from this repository checkout.
# macOS users can double-click this file after making it executable:
#   chmod +x liddia_gui_app/launch_gui.command

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export LIDDIA_GUI_PREFERRED_PORT="${LIDDIA_GUI_PREFERRED_PORT:-7961}"

# Optional local environment activation. Edit these for your machine if needed.
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-liddia-mac}"

if [[ -n "${CONDA_ENV:-}" && -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
fi

cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
export LIDDIA_RUN_PYTHON="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

echo "Repository: $REPO_DIR"
echo "Python: $LIDDIA_RUN_PYTHON"
if [[ -n "${GRADIO_SERVER_PORT:-}" ]]; then
  echo "URL: http://127.0.0.1:${GRADIO_SERVER_PORT}/"
else
  echo "Preferred URL: http://127.0.0.1:${LIDDIA_GUI_PREFERRED_PORT}/"
fi

"$PYTHON_BIN" -m liddia_gui.app
