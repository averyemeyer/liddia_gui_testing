from liddia_gui_refactor.liddia_gui.logs import active_log_text, tail_text
from liddia_gui_refactor.liddia_gui.run_state import ActiveRun, write_lock


def test_tail_text_returns_file_end(tmp_path):
    path = tmp_path / "run.log"
    path.write_text("0123456789")

    assert tail_text(path, max_chars=4) == "6789"


def test_active_log_text_uses_lock_paths(tmp_path):
    stdout = tmp_path / "out.log"
    stderr = tmp_path / "err.log"
    stdout.write_text("hello")
    stderr.write_text("boom")
    write_lock(
        ActiveRun(
            pid=123,
            started_at=1.0,
            known_dirs=[],
            stdout_log=str(stdout),
            stderr_log=str(stderr),
        ),
        tmp_path,
    )

    text = active_log_text(tmp_path)

    assert "--- STDOUT ---" in text
    assert "hello" in text
    assert "--- STDERR ---" in text
    assert "boom" in text

