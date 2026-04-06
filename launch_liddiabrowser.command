#!/bin/bash

# ---- EDIT THESE FOR YOUR MACHINE ----
REPO_DIR="/Users/yourname/path/to/LIDDIA"
CONDA_BIN="/Users/yourname/anaconda3/bin/activate"
CONDA_ENV="liddia"
# -------------------------------------

cd "$REPO_DIR" || exit 1
source "$CONDA_BIN" "$CONDA_ENV"
python liddia_gradio_appv2_rewrite.py
