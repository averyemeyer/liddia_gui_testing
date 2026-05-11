from pathlib import Path

import json

from liddia_gui_app.liddia_gui.run_state import clear_last_run, dismissed_run_path, last_run_path, read_last_run, write_last_run


def test_last_run_roundtrip(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_json = run_dir / "EGFR.json"
    run_json.write_text("{}")

    write_last_run(run_dir, run_json, tmp_path)

    assert read_last_run(tmp_path) == (run_dir, run_json)


def test_clear_last_run_removes_pointer_and_dismisses_fallback(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_json = run_dir / "EGFR.json"
    run_json.write_text("{}")
    write_last_run(run_dir, run_json, tmp_path)

    clear_last_run(tmp_path)

    assert not last_run_path(tmp_path).exists()
    assert dismissed_run_path(tmp_path).exists()
    assert read_last_run(tmp_path) == (None, None)


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


def test_newer_terminal_state_appears_after_dismissal(tmp_path):
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_json = old_dir / "EGFR.json"
    old_json.write_text("{}")
    old_state = old_dir / "run_state.json"
    old_state.write_text(json.dumps({"status": "failed", "run_dir": str(old_dir), "run_json_path": str(old_json)}))

    clear_last_run(tmp_path)

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_json = new_dir / "EGFR.json"
    new_json.write_text("{}")
    new_state = new_dir / "run_state.json"
    new_state.write_text(json.dumps({"status": "failed", "run_dir": str(new_dir), "run_json_path": str(new_json)}))

    assert read_last_run(tmp_path) == (new_dir, new_json)
