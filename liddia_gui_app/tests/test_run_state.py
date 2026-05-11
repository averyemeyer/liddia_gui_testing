from pathlib import Path

import json

from liddia_gui_app.liddia_gui.run_state import read_last_run, write_last_run


def test_last_run_roundtrip(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_json = run_dir / "EGFR.json"
    run_json.write_text("{}")

    write_last_run(run_dir, run_json, tmp_path)

    assert read_last_run(tmp_path) == (run_dir, run_json)


def test_last_run_ignores_missing_paths(tmp_path):
    write_last_run(tmp_path / "missing", tmp_path / "missing" / "EGFR.json", tmp_path)

    assert read_last_run(tmp_path) == (None, None)


def test_read_last_run_falls_back_to_latest_terminal_state(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_json = run_dir / "EGFR.json"
    run_json.write_text("{}")
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "run_dir": str(run_dir),
                "run_json_path": str(run_json),
            }
        )
    )

    assert read_last_run(tmp_path) == (run_dir, run_json)
