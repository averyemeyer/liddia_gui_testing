#!/bin/bash

# ---- EDIT THESE FOR YOUR MACHINE ----
REPO_DIR="/Users/meyer.1938/Desktop/LIDDIA"
CONDA_BIN="/Users/meyer.1938/anaconda3/bin/activate"
CONDA_ENV="liddia-mac"
# -------------------------------------

cd "$REPO_DIR" || exit 1
source "$CONDA_BIN" "$CONDA_ENV"

# Local stability/workaround settings (do not affect source code).
export LOKY_MAX_CPU_COUNT=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JOBLIB_TEMP_FOLDER="$REPO_DIR/log/.joblib_tmp"
mkdir -p "$JOBLIB_TEMP_FOLDER"

python liddia_gradio_appv2_rewrite.py
