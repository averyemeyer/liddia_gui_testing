import time

from liddia_gui_app.liddia_gui.runner import _active_placeholder_data
from liddia_gui_app.liddia_gui.run_state import ActiveRun, write_run_state


def test_active_placeholder_data_renders_as_running_run_data():
    active = ActiveRun(
        pid=123,
        started_at=time.time() - 5,
        known_dirs=[],
        target="EGFR",
        model="claude-test",
        max_iter=2,
    )

    data = _active_placeholder_data(active)

    assert data["model"] == "claude-test"
    assert data["task"]["target"] == "EGFR"
    assert data["runtime"]["max_iter"] == 2
    assert data["runtime"]["start_time"]


def test_active_placeholder_includes_task_requirements(monkeypatch):
    monkeypatch.setattr(
        "liddia_gui_app.liddia_gui.runner.task_context",
        lambda target, max_iter: {"target": target, "resource": max_iter, "requirements": ["At least 5 molecules"]},
    )
    active = ActiveRun(pid=123, started_at=time.time(), known_dirs=[], target="EGFR", model="claude-test", max_iter=2)

    data = _active_placeholder_data(active)

    assert data["task"]["requirements"] == ["At least 5 molecules"]


def test_run_state_records_log_paths(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    write_run_state(run_dir, status="running", stdout_log="/tmp/out.log", stderr_log="/tmp/err.log")

    text = (run_dir / "run_state.json").read_text()
    assert "/tmp/out.log" in text
    assert "/tmp/err.log" in text
