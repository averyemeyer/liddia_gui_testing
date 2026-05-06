from liddia_gui_app.liddia_gui.io_utils import available_run_dirs, latest_json_in_dir, run_dir_choices, safe_read_json


def test_safe_read_json_returns_last_good_data_during_transient_write(tmp_path):
    path = tmp_path / "run.json"
    path.write_text('{"model": "claude-test"}')

    assert safe_read_json(path) == {"model": "claude-test"}

    path.write_text("{")

    assert safe_read_json(path, retries=1) == {"model": "claude-test"}


def test_latest_json_ignores_gui_run_state(tmp_path):
    run_dir = tmp_path / "26-05-04_EGFR"
    run_dir.mkdir()
    state = run_dir / "run_state.json"
    artifact = run_dir / "EGFR.json"
    artifact.write_text('{"model": "claude"}')
    state.write_text('{"status": "running"}')

    assert latest_json_in_dir(run_dir) == artifact
    assert available_run_dirs(tmp_path) == ["26-05-04_EGFR"]


def test_run_dir_choices_include_status_target_and_model(tmp_path):
    run_dir = tmp_path / "26-05-04_EGFR"
    run_dir.mkdir()
    (run_dir / "EGFR.json").write_text(
        '{"model": "claude-test", "success": true, "task": {"target": "EGFR"}}'
    )

    choices = run_dir_choices(tmp_path)

    assert choices[0][1] == "26-05-04_EGFR"
    assert choices[0][0] == "SUCCESS | EGFR | 26-05-04_EGFR | claude-test"


def test_run_dir_choices_prefer_running_state(tmp_path):
    run_dir = tmp_path / "26-05-04_ADRB1"
    run_dir.mkdir()
    (run_dir / "ADRB1.json").write_text('{"model": "claude-test", "success": false}')
    (run_dir / "run_state.json").write_text('{"status": "running", "target": "ADRB1", "model": "claude-live"}')

    choices = run_dir_choices(tmp_path)

    assert choices[0][0] == "RUNNING | ADRB1 | 26-05-04_ADRB1 | claude-test"
