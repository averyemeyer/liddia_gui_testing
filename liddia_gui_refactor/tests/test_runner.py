import time

from liddia_gui_refactor.liddia_gui.runner import _active_placeholder_data
from liddia_gui_refactor.liddia_gui.run_state import ActiveRun


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
