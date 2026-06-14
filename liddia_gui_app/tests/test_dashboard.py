from liddia_gui_app.liddia_gui.dashboard import RESULTS_EMPTY_HTML, DashboardRender


def test_dashboard_render_output_contract_has_named_fields():
    render = DashboardRender.from_snapshot("No run yet.", None, None, None)

    assert len(render.as_outputs()) == 24
    assert render.status_text == "No run yet."
    assert render.results_empty_html == RESULTS_EMPTY_HTML
    assert render.run_dir_state == ""
    assert render.run_json_state == ""


def test_dashboard_render_clears_results_empty_state_when_run_loaded(tmp_path):
    run_json = tmp_path / "run.json"
    render = DashboardRender.from_snapshot("Loaded run.", tmp_path, run_json, {"task": {"target": "EGFR"}})

    assert render.results_empty_html == ""
    assert "Loaded run" in render.results_overview_html
    assert tmp_path.name in render.results_overview_html


def test_dashboard_render_keeps_results_empty_state_without_run_json(tmp_path):
    render = DashboardRender.from_snapshot("No run selected.", tmp_path, None, None)

    assert render.results_empty_html == RESULTS_EMPTY_HTML
