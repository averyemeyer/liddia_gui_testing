from liddia_gui_app.liddia_gui.parsers import parse_run_data
from liddia_gui_app.liddia_gui.trends import filter_trend_rows, iteration_rollup, metric_choices, trend_rows


def test_trends_include_metric_choices_and_iteration_rollup():
    parsed = parse_run_data(
        {
            "0": {
                "action": ["GENERATE001", ["POCKET000"]],
                "action_output": "MOL001",
                "input_goal_prompt": "Molecule Set MOL001\nSize: 5\nDiversity: 0.72\nQED: Range 0.4 to 0.8, Median 0.61",
                "goal_response": "Answer: NO",
            },
            "1": {
                "action": ["CODE002", ["MOL001"]],
                "action_output": "MOL002",
                "input_goal_prompt": "Molecule Set MOL002\nSize: 6\nDiversity: 0.81\nQED: Range 0.5 to 0.9, Median 0.73",
                "goal_response": "Answer: YES",
            },
        }
    )

    rows = trend_rows(parsed)
    assert metric_choices(rows) == ["All", "Diversity", "QED"]
    assert filter_trend_rows(rows, "QED")["Median"].tolist() == [0.61, 0.73]

    rollup = iteration_rollup(parsed)
    assert rollup[1]["Pool"] == "MOL002"
    assert rollup[1]["Goal"] == "YES"
    assert rollup[1]["QED"] == "0.73"
