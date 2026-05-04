import pandas as pd

from liddia_gui_refactor.liddia_gui.task_context import merge_task_context, task_context


def test_task_context_derives_requirements_from_drug_csv(tmp_path):
    drug_csv = tmp_path / "drugs.csv"
    pd.DataFrame(
        {
            "NAME": ["EGFR", "EGFR"],
            "QED": [0.4, 0.6],
            "SAScore": [2.0, 4.0],
            "Lipinski Rules Followed": [3, 5],
            "Vina Score": [-7.0, -9.0],
        }
    ).to_csv(drug_csv, index=False)

    task = task_context("EGFR", 2, drug_csv)

    assert task["target"] == "EGFR"
    assert task["resource"] == 2
    assert "Vina Score must be lower than -8.00" in task["requirements"]
    assert "QED must be better than 0.50" in task["requirements"]


def test_merge_task_context_preserves_liddia_output(monkeypatch):
    monkeypatch.setattr(
        "liddia_gui_refactor.liddia_gui.task_context.task_context",
        lambda target, max_iter: {"target": target, "resource": max_iter, "requirements": ["fallback"]},
    )

    data = merge_task_context({"task": {"target": "ADRB1", "requirements": ["from run"]}}, "EGFR", 2)

    assert data["task"]["target"] == "ADRB1"
    assert data["task"]["resource"] == 2
    assert data["task"]["requirements"] == ["from run"]
