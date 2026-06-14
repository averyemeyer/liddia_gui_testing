# LIDDIA GUI architecture

The GUI is intentionally isolated in `liddia_gui_app/`. It treats LIDDIA run artifacts as the source of truth and keeps interface state separate from scientific workflow state.

## Boundaries

- `backend.py` contains version-specific command construction.
- `runner.py` launches detached subprocesses and reconnects to active runs.
- `run_state.py` manages `.run.lock` and per-run `run_state.json`.
- `parsers.py` and `molecules.py` adapt normal LIDDIA outputs for display.
- `dashboard.py` defines monitor and review render contracts.
- `app.py` contains Gradio layout and event wiring.
- `viewer3d.py`, `trends.py`, and `reports.py` provide focused review tools.

Monitor state and review state are independent. A live run can continue refreshing while Results and Trends display another run.

## Upstream Boundary

The application code, tests, launchers, and GUI environment all live under `liddia_gui_app/`. No modules under `liddia/` are modified.

The only core integration change is in the repository-level `run.py`:

- accept `ANTHROPIC_API_KEY` while retaining the original key-file fallback;
- write atomic, incremental run JSON snapshots and runtime timestamps;
- preserve parse failures in run output instead of reporting false completion;
- stop gracefully when the model explicitly declares a satisfied task complete.

These changes allow the GUI to monitor and recover a run without creating a second scientific results format. Removing the incremental snapshot support would reduce live monitoring detail, but completed-run review would still use standard LIDDIA artifacts.

## Persistence

The browser is never the process owner. A run is launched as a detached subprocess, and the GUI reconnects using disk state:

- `log/.run.lock` identifies the active process.
- `log/<run_id>/run_state.json` records GUI recovery metadata.
- `log/<run_id>/<target>.json` remains the LIDDIA run record.
- detached stdout and stderr files preserve diagnostics.

GUI recovery metadata is small and disposable; molecule pools, metrics, structures, and task information continue to come from LIDDIA outputs.
