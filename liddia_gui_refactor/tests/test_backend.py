from pathlib import Path

from liddia_gui_refactor.liddia_gui.backend import LiddiaV2Backend, RunConfig


def test_v2_backend_builds_base_command():
    backend = LiddiaV2Backend(run_py=Path("/repo/run.py"), log_root=Path("/repo/log"))

    cmd = backend.build_command(
        RunConfig(
            target=" EGFR ",
            max_iter=3,
            model=" claude-test ",
        )
    )

    assert cmd[:8] == [
        "python",
        "-u",
        "/repo/run.py",
        "--target",
        "EGFR",
        "--max_iter",
        "3",
        "--model",
    ]
    assert cmd[8:] == ["claude-test"]
