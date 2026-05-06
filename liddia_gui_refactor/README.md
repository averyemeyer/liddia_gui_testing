# LIDDIA GUI v2

Modular Gradio interface for launching LIDDIA runs, monitoring progress, reviewing molecule pools, inspecting 3D structures, and exporting reports.

This GUI is a refactor of the older flat Gradio script. The current goal is to keep GUI concerns in `liddia_gui_refactor/liddia_gui/` while relying on normal LIDDIA run artifacts under `log/<run_id>/`.

## Current Status

- Monitor tab launches and recovers LIDDIA runs.
- Results tab loads completed or previous runs from disk.
- Results and Trends use a separate review state, so a live Monitor refresh should not overwrite a run being reviewed.
- 3D Viewer supports ligand/pose files and receptor surface inspection.
- Help page includes workflow placeholders and metric hover tooltips.
- Run reports export a bundle with text, JSON, and final-pool metric CSV.

## Setup

Use a Python environment that has LIDDIA's runtime dependencies installed. At minimum, the GUI/runtime currently expects packages such as Gradio, Fire, pandas, RDKit, Plotly, and the LIDDIA docking stack used by the main repository.

Example with conda:

```bash
conda activate liddia-mac
```

If you are setting this up on a new machine, create or update an environment from the repository's environment file when available, then install any missing LIDDIA-specific dependencies.

## Launch

Recommended shell launchers:

macOS/Linux:

```bash
cd /path/to/LIDDIA
chmod +x liddia_gui_refactor/launch_gui.command
./liddia_gui_refactor/launch_gui.command
```

Windows PowerShell:

```powershell
cd C:\path\to\LIDDIA
powershell -ExecutionPolicy Bypass -File .\liddia_gui_refactor\launch_gui.ps1
```

Manual launch from the repository root:

```bash
cd /path/to/LIDDIA/liddia_gui_refactor
python -m liddia_gui.app
```

Open the local URL printed by Gradio, usually:

```text
http://127.0.0.1:7960/
```

To force a port:

```bash
GRADIO_SERVER_PORT=7961 python -m liddia_gui.app
```

On Windows PowerShell:

```powershell
$env:GRADIO_SERVER_PORT = "7961"
python -m liddia_gui.app
```

macOS example from the original development machine:

```bash
cd /Users/meyer.1938/Desktop/LIDDIA/liddia_gui_refactor
GRADIO_SERVER_PORT=7961 /Users/meyer.1938/anaconda3/envs/liddia-mac/bin/python -m liddia_gui.app
```

The app picks the first available port from `7960` upward unless `GRADIO_SERVER_PORT` is set. Use the same Python environment for the GUI that you use to run LIDDIA itself.

## Typical Workflow

1. Open the Monitor tab.
2. Select target, budget, model, and enter an Anthropic API key.
3. Click `Run LIDDIA`.
4. Watch status, elapsed time, action timeline, logs, and recovery metadata.
5. Use Results to load a previous run or click `Review active run`.
6. Use Trends to inspect median metric movement across iterations.
7. Use 3D Viewer to upload ligand/pose/receptor files for visual inspection.

## Run Persistence

Browser tabs are not the source of truth. The GUI writes and reads disk state:

- `log/.run.lock`: active process lock and recovery metadata.
- `log/<run_id>/run_state.json`: per-run status, pid, target, model, run JSON path, and log paths.
- `log/<run_id>/<target>.json`: LIDDIA run snapshot.
- `log/.run_<timestamp>.stdout.log` and `.stderr.log`: detached subprocess logs.

If the browser is closed, the run process should continue. Reopen the app and use `Load latest / recover` in Monitor.

## Results And Live Runs

The Monitor tab tracks the active run. Results and Trends track the review run.

This means you can monitor a live run while browsing a previous run. Use `Review active run` when you want Results and Trends to point back to the active run.

Molecule-level pool browsing depends on LIDDIA's normal `*_memory.pkl` artifact. At present, that artifact is expected after the run writes it; the GUI does not modify `run.py` to stream memory during an active run.

## Troubleshooting

### Port already in use

Set another port:

```bash
GRADIO_SERVER_PORT=7962 python -m liddia_gui.app
```

Windows PowerShell:

```powershell
$env:GRADIO_SERVER_PORT = "7962"
python -m liddia_gui.app
```

### `No module named 'fire'`

The app or subprocess is using a Python environment without Fire installed. Activate the LIDDIA environment, install `fire`, and relaunch:

```bash
python -m pip install fire
python -m liddia_gui.app
```

### `RDKit unavailable`

The GUI process cannot import RDKit, usually because it is running under a different environment than LIDDIA. Install RDKit in the active environment or relaunch from the correct environment.

Conda is usually the easiest RDKit install path:

```bash
conda install -c conda-forge rdkit
```

### `No module named 'MolKit'`

Docking receptor preparation is missing an AutoDockTools/MGLTools dependency. Vina scores may fail until this environment dependency is fixed. This is a LIDDIA runtime environment issue, not a Gradio rendering issue.

### `protein.pdbqt does not exist`

Usually follows the `MolKit` failure above. Receptor preparation did not produce the expected `protein.pdbqt`, so Vina cannot score the molecule.

### UI freezes during active monitoring

The timer uses a lightweight Monitor refresh path, but heavy docking or model work can still make the machine busy. Use `Load latest / recover` manually if you want full logs; the background timer intentionally avoids rebuilding Results and Trends.

### Windows path notes

Use normal Windows paths in PowerShell, for example:

```powershell
cd C:\Users\<you>\Desktop\LIDDIA\liddia_gui_refactor
python -m liddia_gui.app
```

The GUI stores run artifacts under the repository's `log/` directory regardless of operating system.

## Module Map

- `app.py`: Gradio layout and event wiring.
- `backend.py`: LIDDIA backend adapter and subprocess command.
- `runner.py`: launch, recovery, detached logs, notifications, active lock handling.
- `run_state.py`: `.run.lock` and `run_state.json`.
- `dashboard.py`: named render contracts for Monitor and full review views.
- `parsers.py`: pure run JSON parsing and metric display helpers.
- `molecules.py`: memory/pool loading, RDKit molecule thumbnails, CSV exports.
- `trends.py`: metric trend rows, filtering, plots, iteration rollups.
- `viewer3d.py`: 3Dmol HTML generation and PDBQT pose handling.
- `reports.py`: report bundle export.
- `ui_components.py`: reusable HTML panels.

## Tests

From the repository root:

```bash
python -m pytest liddia_gui_refactor/tests
```

The app should be launched with an environment that has LIDDIA runtime dependencies. Tests can run from any environment that has the test dependencies installed.

## Current Limitations

- Provider switching is UI-only for now; LIDDIA still runs through its current backend behavior.
- Open-source/local LLM support should be added behind a future LIDDIA provider abstraction, not by hard-coding GUI-specific behavior.
- The GUI does not currently stream molecule pool memory mid-action.
- Docking depends on external AutoDockTools/MolKit compatibility.
