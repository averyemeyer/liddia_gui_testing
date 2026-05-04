# LIDDIA GUI modular refactor

This package starts the refactor for:

1. Modular design
2. Backend adapter boundary for LIDDIA v2/v3
3. Persistent run state so runs continue after the browser page closes

## How to migrate from the flat file

Do not delete the current `liddia_gradio_appv2_flat.py` yet. Treat it as the reference implementation.

Recommended migration order:

1. Drop the `liddia_gui/` package beside `run.py`.
2. Run `python -m liddia_gui.app`.
3. Confirm launch, refresh, lock recovery, and previous-run loading work.
4. Move existing molecule/table functions into `molecules.py`.
5. Move report functions into `reports.py`.
6. Move remaining HTML builders into `ui_components.py`.
7. Only then retire the flat file.

## Important files

- `backend.py`: version-specific LIDDIA assumptions live here.
- `run_state.py`: `.run.lock` and `run_state.json` handling.
- `runner.py`: subprocess launch, recover, and notification logic.
- `logs.py`: detached stdout/stderr log discovery and tailing.
- `parsers.py`: pure run JSON parsing.
- `ui_components.py`: reusable HTML snippets.
- `app.py`: Gradio layout only.

## Current refactor status

- Launches LIDDIA v2 through a backend adapter with target/model/max-iter, skip-docking, and extra CLI args.
- Recovers active runs from disk with `.run.lock` and per-run `run_state.json`.
- Shows monitor status, progress, action timeline, final pool metrics, requirements, raw JSON, and CLI logs.
- Keeps metric/help copy centralized in parser/UI helpers so future backends can reuse the same surface.
- Includes focused tests for backend command generation, parser table contracts, and log recovery helpers.
