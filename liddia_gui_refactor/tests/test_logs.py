from liddia_gui_refactor.liddia_gui.logs import active_log_text, classify_log_text, log_diagnostics_html, tail_text
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


def test_classify_log_text_detects_missing_dependencies():
    findings = classify_log_text(
        "ModuleNotFoundError: No module named 'fire'\n"
        "ModuleNotFoundError: No module named 'MolKit'\n"
        "RuntimeError: Error: file /tmp/protein.pdbqt does not exist."
    )

    titles = [finding["title"] for finding in findings]
    assert "Missing Fire dependency" in titles
    assert "Missing MolKit dependency" in titles
    assert "Docking receptor was not prepared" in titles


def test_log_diagnostics_html_returns_empty_panel_without_matches():
    assert "No recognized runtime issues" in log_diagnostics_html("plain progress output")
