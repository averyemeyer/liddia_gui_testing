from types import SimpleNamespace

from liddia_gui_app.liddia_gui.viewer3d import count_pdbqt_poses, extract_pdbqt_pose, extract_pdbqt_vina_score, pdbqt_to_pdb, render_uploaded_structure


PDBQT = """MODEL 1
REMARK VINA RESULT: -7.5 0.0 0.0
ATOM      1  C   LIG A   1      1.000   2.000   3.000  1.00  0.00     0.0 C
ENDMDL
MODEL 2
REMARK VINA RESULT: -8.1 0.0 0.0
ATOM      1  O   LIG A   1      4.000   5.000   6.000  1.00  0.00     0.0 OA
ENDMDL
"""


def test_pdbqt_pose_helpers():
    assert count_pdbqt_poses(PDBQT) == 2
    pose = extract_pdbqt_pose(PDBQT, 2)
    assert pose is not None
    assert extract_pdbqt_vina_score(pose) == -8.1
    pdb = pdbqt_to_pdb(pose)
    assert "ATOM" in pdb
    assert "  O" in pdb


def test_receptor_only_render(tmp_path):
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  C   REC A   1      1.000   2.000   3.000  1.00  0.00           C\nEND\n"
    )
    status, html, badge = render_uploaded_structure(
        None,
        SimpleNamespace(name=str(receptor)),
        "stick",
        "spectrum",
        "surface",
        "blue",
        0.85,
        1,
    )
    assert "Rendered receptor.pdb" in status
    assert "viewer3d-frame" in html
    assert "Receptor surface" in badge
