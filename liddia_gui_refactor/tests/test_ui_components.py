from liddia_gui_refactor.liddia_gui.run_state import ActiveRun
from liddia_gui_refactor.liddia_gui.ui_components import recovery_card


def test_recovery_card_shows_active_run_metadata():
    html = recovery_card(
        ActiveRun(
            pid=123,
            started_at=1.0,
            known_dirs=[],
            target="EGFR",
            model="claude-test",
            max_iter=2,
            stdout_log="/tmp/out.log",
            stderr_log="/tmp/err.log",
        ),
        run_dir="/tmp/run",
        run_json="/tmp/run/EGFR.json",
        is_running=True,
    )

    assert "RUNNING" in html
    assert "EGFR" in html
    assert "/tmp/out.log" in html
