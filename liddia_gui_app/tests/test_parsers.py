from liddia_gui_app.liddia_gui.parsers import metric_rows, parse_run_data, requirements_rows, timeline_rows


def sample_run():
    return {
        "model": "claude-test",
        "runtime": {"current_iter": 1, "max_iter": 2},
        "0": {
            "action": ["GENERATE001", ["POCKET000"]],
            "action_output": "MOL004",
            "input_goal_prompt": """
- Molecule Set MOL004:
    Size: 100
    Diversity: 0.87
    Vina Score: Range -10.49 to 0.00, Median 0.00
    QED: Range 0.32 to 0.93, Median 0.80
""",
            "goal_response": "Reason: looks good\nAnswer: YES",
        },
        "success": True,
        "task": {
            "target": "ADRB2",
            "requirements": ["At least 5 molecules", "QED must be better than 0.46"],
            "resource": 2,
        },
    }


def test_parse_run_data_extracts_timeline_and_final_metrics():
    parsed = parse_run_data(sample_run())

    assert parsed["success"] is True
    assert parsed["step_count"] == 1
    assert parsed["final_pool"]["pool"] == "MOL004"
    assert parsed["final_pool"]["size"] == 100
    assert parsed["final_pool"]["metrics"]["Vina Score"]["min"] == "-10.49"


def test_table_rows_are_stable_for_gradio_dataframes():
    parsed = parse_run_data(sample_run())

    assert timeline_rows(parsed)[0]["Action"] == "GENERATE001"
    assert timeline_rows(parsed)[0]["Goal"] == "YES"
    assert any(row["Metric"] == "QED" for row in metric_rows(parsed))
    assert requirements_rows(parsed) == [
        {"Requirement": "At least 5 molecules"},
        {"Requirement": "QED must be better than 0.46"},
    ]


def test_placeholder_active_run_does_not_parse_as_idle():
    parsed = parse_run_data(
        {
            "model": "claude-test",
            "runtime": {
                "current_iter": 0,
                "max_iter": 2,
                "start_time": "2026-05-04T10:00:00",
                "updated_at": "2026-05-04T10:00:01",
            },
            "task": {"target": "EGFR", "resource": 2},
            "_placeholder": True,
        }
    )

    assert parsed["runtime"]["start_time"] == "2026-05-04T10:00:00"
    assert parsed["success"] is None
    assert parsed["step_count"] == 0
