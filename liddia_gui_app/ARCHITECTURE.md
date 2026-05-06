# LIDDIA GUI architecture

This package is organized around:

1. Modular design
2. Backend adapter boundary for LIDDIA v2/v3
3. Persistent run state so runs continue after the browser page closes

## Legacy references

Earlier GUI scripts are preserved under `../old gui/`. Treat those files as historical references only; the working GUI now lives in the `liddia_gui/` package.

## How to run the modular GUI

```console
cd liddia_gui_app
python -m liddia_gui.app
```

## Important files

- `backend.py`: version-specific LIDDIA assumptions live here.
- `run_state.py`: `.run.lock` and `run_state.json` handling.
- `runner.py`: subprocess launch, recover, and notification logic.
- `logs.py`: detached stdout/stderr log discovery and tailing.
- `parsers.py`: pure run JSON parsing.
- `ui_components.py`: reusable HTML snippets.
- `app.py`: Gradio layout only.

## Current status

- Launches LIDDIA v2 through a backend adapter with target/model/max-iter.
- Recovers active runs from disk with `.run.lock` and per-run `run_state.json`.
- Shows monitor status, elapsed time, progress, action timeline, final pool metrics, requirements, raw JSON, and CLI logs.
- Keeps metric/help copy centralized in parser/UI helpers so future backends can reuse the same surface.
- Includes focused tests for backend command generation, parser table contracts, and log recovery helpers.
