from liddia_gui_app.liddia_gui.artifacts import discover_liddia_artifacts, is_liddia_run_json


def test_artifact_discovery_prefers_liddia_outputs(tmp_path):
    run_dir = tmp_path / "26-05-04_EGFR"
    run_dir.mkdir()
    run_json = run_dir / "EGFR.json"
    gui_json = run_dir / "run_state.json"
    memory = run_dir / "EGFR_memory.pkl"
    pool_csv = run_dir / "MOL005.csv"
    hidden_json = run_dir / ".tmp.json"

    run_json.write_text("{}")
    gui_json.write_text("{}")
    hidden_json.write_text("{}")
    memory.write_bytes(b"memory")
    pool_csv.write_text("SMILES,QED\nCCO,0.7\n")

    artifacts = discover_liddia_artifacts(run_dir)

    assert artifacts is not None
    assert artifacts.run_json == run_json
    assert artifacts.memory_files == (memory,)
    assert artifacts.pool_csvs == (pool_csv,)
    assert is_liddia_run_json(run_json)
    assert not is_liddia_run_json(gui_json)
    assert not is_liddia_run_json(hidden_json)

