# LIDDIA GUI v2

A local Gradio interface for launching, monitoring, recovering, and reviewing LIDDIA runs.

## Start Here

From the repository root, create the environment once:

```bash
conda env create -f liddia_gui_app/environment.yml
conda activate liddia-gui
```

Then launch the GUI.

macOS/Linux:

```bash
./liddia_gui_app/launch_gui.command
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\liddia_gui_app\launch_gui.ps1
```

Open the local URL printed in the terminal. The GUI starts at port `7961` when available and automatically tries another port if needed.

To use an existing conda environment on macOS/Linux:

```bash
CONDA_ENV=my-liddia-env ./liddia_gui_app/launch_gui.command
```

To launch manually:

```bash
cd liddia_gui_app
python -m liddia_gui.app
```

## Workflow

1. Open Monitor and choose a target, iteration budget, and model.
2. Enter an Anthropic API key and select `Run LIDDIA`.
3. Follow progress, elapsed time, actions, and logs.
4. Open Results to inspect pools, metrics, requirements, and exports.
5. Use Trends for iteration-level metric changes.
6. Use 3D Viewer for ligand poses, Vina scores, and receptor surfaces.

Runs continue independently of the browser tab. Reopen the GUI and select `Recover latest run` to reconnect. Results and Trends maintain a separate review selection, so a previous run can be inspected while another run is active.

## Run Data

The GUI reads standard LIDDIA artifacts rather than maintaining a separate results format:

- `log/<run_id>/<target>.json`
- `log/<run_id>/<target>_memory.pkl`
- structure and docking files produced by LIDDIA

Small GUI recovery files are stored alongside these outputs:

- `log/.run.lock`
- `log/<run_id>/run_state.json`
- detached stdout and stderr logs

Exports are written to the selected run folder.

## Configuration

Useful optional environment variables:

| Variable | Purpose |
| --- | --- |
| `CONDA_ENV` | Conda environment activated by the macOS/Linux launcher |
| `CONDA_SH` | Path to `conda.sh` when conda is installed outside `~/anaconda3` |
| `GRADIO_SERVER_PORT` | Require a specific GUI port |
| `LIDDIA_GUI_PREFERRED_PORT` | First port to try |
| `LIDDIA_RUN_PYTHON` | Python executable used for LIDDIA subprocesses |

The API key entered in the GUI is passed to the run subprocess as `ANTHROPIC_API_KEY`; it is not written into the repository.

## Troubleshooting

**Port already in use:** allow the launcher to choose another port, or set one explicitly:

```bash
GRADIO_SERVER_PORT=7962 ./liddia_gui_app/launch_gui.command
```

**Missing `fire`, RDKit, or another package:** the GUI and LIDDIA subprocess are using different Python environments. Relaunch from the environment used to install `environment.yml`.

**Missing `MolKit` or `protein.pdbqt`:** receptor preparation is not available in the current environment. Confirm the AutoDockTools dependency installed successfully.

**Run interrupted in the browser:** reopen the GUI and use `Recover latest run`. The browser is not the source of truth for process state.

## Tests

From the repository root:

```bash
python -m pytest liddia_gui_app/tests
```

Implementation boundaries and the upstream integration notes are documented in [ARCHITECTURE.md](ARCHITECTURE.md).
