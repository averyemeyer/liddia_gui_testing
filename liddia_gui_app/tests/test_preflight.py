from liddia_gui_app.liddia_gui import preflight


def test_preflight_blocks_missing_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(preflight, "RUN_PY", tmp_path / "run.py")
    monkeypatch.setattr(preflight, "PDB_DIR", tmp_path / "pdb")
    monkeypatch.setattr(preflight, "LOG_ROOT", tmp_path / "log")
    preflight.RUN_PY.write_text("")
    preflight.PDB_DIR.mkdir()
    (preflight.PDB_DIR / "EGFR.pdb").write_text("")

    checks = preflight.run_preflight(target="EGFR", api_key="", module_checker=lambda *_, **__: True)

    assert not preflight.preflight_can_start(checks)
    assert "Anthropic API key" in [check.name for check in checks if check.blocking]


def test_preflight_accepts_env_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(preflight, "RUN_PY", tmp_path / "run.py")
    monkeypatch.setattr(preflight, "PDB_DIR", tmp_path / "pdb")
    monkeypatch.setattr(preflight, "LOG_ROOT", tmp_path / "log")
    preflight.RUN_PY.write_text("")
    preflight.PDB_DIR.mkdir()
    (preflight.PDB_DIR / "EGFR.pdb").write_text("")

    checks = preflight.run_preflight(target="EGFR", api_key="", module_checker=lambda *_, **__: True)

    assert preflight.preflight_can_start(checks)


def test_preflight_blocks_missing_target(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(preflight, "RUN_PY", tmp_path / "run.py")
    monkeypatch.setattr(preflight, "PDB_DIR", tmp_path / "pdb")
    monkeypatch.setattr(preflight, "LOG_ROOT", tmp_path / "log")
    preflight.RUN_PY.write_text("")
    preflight.PDB_DIR.mkdir()

    checks = preflight.run_preflight(target="EGFR", api_key="", module_checker=lambda *_, **__: True)

    assert not preflight.preflight_can_start(checks)
    assert "Target PDB" in [check.name for check in checks if check.blocking]


def test_preflight_html_renders_statuses():
    html = preflight.preflight_html(
        [
            preflight.PreflightCheck("A", "ok", "ready"),
            preflight.PreflightCheck("B", "fail", "missing", "fix it", blocking=True),
        ]
    )

    assert "OK" in html
    assert "FIX" in html
    assert "fix it" in html
