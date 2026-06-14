import zipfile
from pathlib import Path

import pandas as pd

from liddia_gui_app.liddia_gui.molecules import download_all_pools_csv, enrich_parsed_with_memory, pool_ids_for_run, pool_stats


class FakeMemory:
    def __init__(self):
        self.history = [{"action_output": "MOL005"}]
        self.stream = {
            "MOL005": {
                "type": "MOL",
                "data": pd.DataFrame(
                    {
                        "SMILES": ["CCO", "CCC", "CCN"],
                        "QED": [0.5, 0.7, 0.9],
                        "Vina Score": [-7.0, -8.0, -9.0],
                    }
                ),
                "metrics": {
                    "size": 3,
                    "diversity": 0.87,
                    "qed": {"min": 0.5, "median": 0.7, "max": 0.9},
                    "vina": {"min": -9.0, "median": -8.0, "max": -7.0},
                },
            }
        }


def test_pool_stats_normalizes_memory_metrics(monkeypatch):
    monkeypatch.setattr("liddia_gui_app.liddia_gui.molecules.load_memory", lambda run_dir: FakeMemory())
    monkeypatch.setattr("liddia_gui_app.liddia_gui.molecules.resolve_run_dir", lambda *_: object())

    stats = pool_stats("/tmp/run", "/tmp/run/EGFR.json", "MOL005")

    assert stats["size"] == 3
    assert stats["diversity"] == "0.87"
    assert stats["metrics"]["QED"]["median"] == "0.70"
    assert stats["metrics"]["Vina Score"]["min"] == "-9.00"


def test_enrich_parsed_with_memory_updates_final_pool(monkeypatch):
    monkeypatch.setattr("liddia_gui_app.liddia_gui.molecules.load_memory", lambda run_dir: FakeMemory())
    monkeypatch.setattr("liddia_gui_app.liddia_gui.molecules.resolve_run_dir", lambda *_: object())

    parsed = {
        "steps": [{"step": 1, "action_output": "MOL005", "pool_stats": {}}],
        "final_pool": {},
    }

    enriched = enrich_parsed_with_memory(parsed, "/tmp/run", "/tmp/run/EGFR.json")

    assert enriched["final_pool"]["pool"] == "MOL005"
    assert enriched["final_pool"]["metrics"]["QED"]["max"] == "0.90"
    assert enriched["steps"][0]["pool_stats"]["metrics"]["Vina Score"]["median"] == "-8.00"


def test_pool_ids_include_stream_pools_without_duplicates(monkeypatch):
    memory = FakeMemory()
    memory.history.append({"action_output": "MOL005"})
    memory.stream["MOL006"] = {"data": pd.DataFrame({"SMILES": ["CO"]})}
    monkeypatch.setattr("liddia_gui_app.liddia_gui.molecules.load_memory", lambda run_dir: memory)
    monkeypatch.setattr("liddia_gui_app.liddia_gui.molecules.resolve_run_dir", lambda *_: object())

    assert pool_ids_for_run("/tmp/run", "") == ["MOL005", "MOL006"]


def test_download_all_pool_csvs_builds_zip_in_loaded_run_folder(tmp_path, monkeypatch):
    memory = FakeMemory()
    memory.stream["MOL006"] = {
        "data": pd.DataFrame({"SMILES": ["CO"], "QED": [0.8]}),
    }
    memory.history.append({"action_output": "MOL006"})
    monkeypatch.setattr("liddia_gui_app.liddia_gui.molecules.load_memory", lambda run_dir: memory)

    run_dir = tmp_path / "sample_run"
    run_dir.mkdir()
    zip_path = download_all_pools_csv(str(run_dir), "")

    assert zip_path is not None
    assert Path(zip_path) == run_dir / "all_pool_csvs.zip"
    assert [path.name for path in run_dir.iterdir()] == ["all_pool_csvs.zip"]
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["MOL005.csv", "MOL006.csv"]
        assert "SMILES,QED,Vina Score" in archive.read("MOL005.csv").decode()
        assert "CO,0.8" in archive.read("MOL006.csv").decode()
