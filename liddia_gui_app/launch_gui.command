#!/usr/bin/env bash
set -euo pipefail

# Launch the modular LIDDIA GUI from this repository checkout.
# macOS users can double-click this file after making it executable:
#   chmod +x liddia_gui_app/launch_gui.command

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7961}"

# Prefer the active environment. If none is active, try the development env path.
if [[ -z "${CONDA_PREFIX:-}" && -x "$HOME/anaconda3/envs/liddia-mac/bin/python" ]]; then
  PYTHON_BIN="$HOME/anaconda3/envs/liddia-mac/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

echo "Repository: $REPO_DIR"
echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "URL: http://127.0.0.1:${GRADIO_SERVER_PORT}/"

"$PYTHON_BIN" -m liddia_gui.app
