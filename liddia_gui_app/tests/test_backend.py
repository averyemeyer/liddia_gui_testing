from pathlib import Path
import sys

from liddia_gui_app.liddia_gui.backend import LiddiaV2Backend, RunConfig


def test_v2_backend_builds_base_command():
    backend = LiddiaV2Backend(run_py=Path("/repo/run.py"), log_root=Path("/repo/log"), python_executable=sys.executable)

    cmd = backend.build_command(
        RunConfig(
            target=" EGFR ",
            max_iter=3,
            model=" claude-test ",
        )
    )

    assert cmd[:8] == [
        sys.executable,
        "-u",
        "/repo/run.py",
        "--target",
        "EGFR",
        "--max_iter",
        "3",
        "--model",
    ]
    assert cmd[8:] == ["claude-test"]


def test_v2_backend_prefers_fire_capable_python(monkeypatch):
    monkeypatch.setenv("LIDDIA_RUN_PYTHON", "/tmp/liddia-env/bin/python")

    resolved = LiddiaV2Backend.resolve_python_executable()

    assert resolved == "/tmp/liddia-env/bin/python"
