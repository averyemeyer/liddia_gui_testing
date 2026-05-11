from liddia_gui_app.liddia_gui.run_config import setup_values_from_run_data


def test_setup_values_from_run_data_prefers_runtime_max_iter():
    values = setup_values_from_run_data(
        {
            "model": "claude-test",
            "runtime": {"max_iter": 10},
            "task": {"target": "ADRB2", "resource": 2},
        }
    )

    assert values.target == "ADRB2"
    assert values.max_iter == 10
    assert values.model == "claude-test"


def test_setup_values_from_run_data_falls_back_to_task_resource():
    values = setup_values_from_run_data({"task": {"target": "EGFR", "resource": "3"}})

    assert values.target == "EGFR"
    assert values.max_iter == 3
    assert values.model is None


def test_setup_values_from_empty_run_data():
    values = setup_values_from_run_data(None)

    assert values.target is None
    assert values.max_iter is None
    assert values.model is None
