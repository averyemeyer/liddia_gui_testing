import json
import zipfile

import pandas as pd

from liddia_gui_refactor.liddia_gui.reports import build_report_bundle_file


class FakeMemory:
    def __init__(self):
        self.history = [{"action_output": "MOL005"}]
        self.stream = {
            "MOL005": {
                "data": pd.DataFrame({"SMILES": ["CCO"], "QED": [0.7]}),
                "metrics": {"size": 1, "qed": {"min": 0.7, "median": 0.7, "max": 0.7}},
            }
        }


def test_report_bundle_exports_txt_json_and_csv(tmp_path, monkeypatch):
    run_dir = tmp_path / "26-05-04_EGFR"
    run_dir.mkdir()
    run_json = run_dir / "EGFR.json"
    run_json.write_text(json.dumps({"model": "claude-test", "0": {"action": ["GENERATE001", []], "action_output": "MOL005"}, "task": {"target": "EGFR"}}))

    monkeypatch.setattr("liddia_gui_refactor.liddia_gui.molecules.load_memory", lambda run_dir: FakeMemory())

    bundle = build_report_bundle_file(str(run_dir), str(run_json))

    assert bundle
    with zipfile.ZipFile(bundle) as archive:
        assert sorted(archive.namelist()) == ["final_pool_metrics.csv", "run_report.txt", "run_summary.json"]

