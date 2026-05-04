from liddia_gui_refactor.liddia_gui.dashboard import DashboardRender


def test_dashboard_render_output_contract_has_named_fields():
    render = DashboardRender.from_snapshot("No run yet.", None, None, None)

    assert len(render.as_outputs()) == 20
    assert render.status_text == "No run yet."
    assert render.run_dir_state == ""
    assert render.run_json_state == ""
