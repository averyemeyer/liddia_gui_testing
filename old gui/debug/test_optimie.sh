#!/bin/bash
set -euo pipefail
cd /Users/meyer.1938/Desktop/LIDDIA
source /Users/meyer.1938/anaconda3/bin/activate liddia-mac

python - <<'PY'
import pickle
from pathlib import Path
from liddia.action import graph_ga_optimizer

log_root = Path("log")
run_dirs = [p for p in log_root.glob("*") if p.is_dir()]
if not run_dirs:
    raise SystemExit("No run directories found in ./log")

run_dir = sorted(run_dirs, key=lambda p: p.stat().st_mtime)[-1]
mem_files = sorted(run_dir.glob("*_memory.pkl"), key=lambda p: p.stat().st_mtime)
if not mem_files:
    raise SystemExit(f"No *_memory.pkl in {run_dir}")

mem = pickle.load(open(mem_files[-1], "rb"))
pool_ids = [k for k, v in mem.stream.items() if isinstance(v, dict) and v.get("data") is not None]
if not pool_ids:
    raise SystemExit("No pool dataframes found in memory stream")

pool_id = pool_ids[-1]
df = mem.stream[pool_id]["data"]

out, meta = graph_ga_optimizer(
    input_df=df,
    property="QED",
    n_outputs=20,
    output_dir=str(run_dir / "log_optimize"),
    env_dir=".env",
    target_pdb="EGFR.pdb",
)

print("Run dir:", run_dir)
print("Pool:", pool_id)
print("Rows out:", len(out))
print("Meta:", meta)
print(out.head())
PY

