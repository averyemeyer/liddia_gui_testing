"""Lightweight 3D structure viewer helpers."""
from __future__ import annotations

import json
import html
import re
import time
from pathlib import Path
from typing import Any


SUPPORTED_LIGAND_TYPES = {".pdb": "pdb", ".sdf": "sdf", ".mol2": "mol2", ".pdbqt": "pdbqt"}
SUPPORTED_RECEPTOR_TYPES = {".pdb": "pdb", ".pdbqt": "pdbqt", ".mol2": "mol2"}


def count_pdbqt_poses(model_text: str) -> int:
    blocks = re.findall(r"(?ms)^MODEL\s+\d+.*?^ENDMDL\s*$", model_text or "")
    return len(blocks)


def extract_pdbqt_pose(model_text: str, pose_index: int) -> str | None:
    pose_index = max(1, int(pose_index or 1))
    blocks = re.findall(r"(?ms)^MODEL\s+\d+.*?^ENDMDL\s*$", model_text or "")
    if not blocks:
        return model_text if pose_index == 1 else None
    idx = pose_index - 1
    return blocks[idx] if idx < len(blocks) else None


def extract_pdbqt_vina_score(block_text: str) -> float | None:
    for line in (block_text or "").splitlines():
        if "REMARK VINA RESULT" not in line.upper():
            continue
        match = re.search(r"REMARK\s+VINA\s+RESULT\s*:\s*([-\d\.eE+]+)", line, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
    return None


def pdbqt_pose_rows(model_text: str) -> list[list[int | float | str]]:
    """Return compact pose metadata for a multi-model PDBQT file."""
    rows: list[list[int | float | str]] = []
    pose_count = count_pdbqt_poses(model_text)
    for pose_index in range(1, pose_count + 1):
        pose_text = extract_pdbqt_pose(model_text, pose_index) or ""
        vina_score = extract_pdbqt_vina_score(pose_text)
        rows.append([pose_index, round(vina_score, 3) if vina_score is not None else "—"])
    return rows


def pose_rows_for_upload(ligand_file: Any) -> list[list[int | float | str]]:
    """Read pose metadata from an uploaded PDBQT file."""
    path, text = _read_upload(ligand_file)
    if not path or path.suffix.lower() != ".pdbqt":
        return []
    return pdbqt_pose_rows(text)


def pose_index_from_selection(index: Any) -> int:
    """Convert a Gradio dataframe selection index to a one-based pose index."""
    row_index = index[0] if isinstance(index, (list, tuple)) else index
    try:
        return max(1, int(row_index) + 1)
    except (TypeError, ValueError):
        return 1


def pdbqt_to_pdb(block_text: str) -> str:
    """Convert PDBQT atom records to PDB-ish lines for reliable 3Dmol parsing."""
    out: list[str] = []
    for line in (block_text or "").splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        parts = line.split()
        try:
            serial = int(parts[1])
            atom_name = parts[2][:4]
            res_name = parts[3][:3] if len(parts) > 3 else "LIG"
            has_chain = len(parts) > 9 and parts[4].isalpha() and len(parts[4]) <= 2
            if has_chain:
                chain = parts[4][:1]
                res_seq = int(parts[5])
                x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
            else:
                chain = "A"
                res_seq = int(parts[4]) if len(parts) > 4 else 1
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
        except Exception:
            continue
        atom_type = parts[-1].upper() if parts else ""
        element = {"OA": "O", "NA": "N", "SA": "S", "HD": "H", "A": "C"}.get(atom_type, atom_type[:2] or atom_name[:1] or "C")
        out.append(
            f"ATOM  {serial:>5} {atom_name:<4} {res_name:>3} {chain}{res_seq:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {element:>2}"
        )
    if out:
        out.append("END")
    return "\n".join(out)


def _style_js(style: str, color: str, opacity: float = 1.0) -> str:
    style = (style or "stick").lower()
    color = color or "spectrum"
    schemes = {"spectrum", "greenCarbon", "cyanCarbon", "orangeCarbon", "magentaCarbon", "whiteCarbon"}
    key = "colorscheme" if color in schemes else "color"
    if style == "surface":
        return f"{{surface:{{opacity:{opacity:.3f},{key}:'{color}'}}}}"
    if style == "line":
        return f"{{line:{{linewidth:1.5,{key}:'{color}'}}}}"
    if style == "sphere":
        return f"{{sphere:{{scale:0.28,{key}:'{color}'}}}}"
    if style == "cartoon":
        return f"{{cartoon:{{{key}:'{color}',opacity:{opacity:.3f}}}}}"
    return f"{{stick:{{radius:0.18,{key}:'{color}'}}}}"


def build_3d_html(
    model_text: str | None,
    model_type: str | None,
    *,
    ligand_style: str = "stick",
    ligand_color: str = "spectrum",
    receptor_text: str | None = None,
    receptor_type: str = "pdb",
    receptor_style: str = "cartoon",
    receptor_color: str = "whiteCarbon",
    receptor_opacity: float = 0.85,
) -> str:
    view_id = f"liddia_3d_{int(time.time() * 1000)}"
    ligand_style_js = _style_js(ligand_style, ligand_color)
    receptor_style_js = _style_js(receptor_style, receptor_color, receptor_opacity)
    iframe_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{ margin:0; padding:0; background:#fff; overflow:hidden; }}
    #{view_id} {{ width:100%; height:620px; }}
    #err {{ position:absolute; left:16px; top:16px; color:#991b1b; font:13px system-ui; }}
  </style>
</head>
<body>
  <div id="{view_id}"></div><div id="err"></div>
  <script>
    function loadScript(url) {{
      return new Promise((resolve, reject) => {{
        const script = document.createElement("script");
        script.src = url;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
      }});
    }}
    async function boot() {{
      try {{
        if (!window.$3Dmol) {{
          try {{
            await loadScript("https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js");
          }} catch (e) {{
            await loadScript("https://unpkg.com/3dmol@2.4.2/build/3Dmol-min.js");
          }}
        }}
        if (!window.$3Dmol) {{
          throw new Error("3Dmol.js did not load. Check network access to jsdelivr/unpkg.");
        }}
        const viewer = window.$3Dmol.createViewer(document.getElementById("{view_id}"), {{backgroundColor:"white"}});
      const receptorText = {json.dumps(receptor_text)};
      if (receptorText) {{
        const rec = viewer.addModel(receptorText, {json.dumps(receptor_type)});
        if ({json.dumps(receptor_style == "surface")}) {{
          viewer.addSurface($3Dmol.SurfaceType.VDW, {receptor_style_js}.surface, {{model: rec}});
        }} else {{
          rec.setStyle({{}}, {receptor_style_js});
        }}
      }}
      const ligandText = {json.dumps(model_text)};
      if (ligandText) {{
        const lig = viewer.addModel(ligandText, {json.dumps(model_type)});
        lig.setStyle({{}}, {ligand_style_js});
      }}
      viewer.zoomTo();
      viewer.render();
      }} catch (e) {{
        document.getElementById("err").textContent = "3D viewer failed: " + (e && e.message ? e.message : String(e));
      }}
    }}
    boot();
  </script>
</body>
</html>"""
    escaped_doc = html.escape(iframe_doc, quote=True)
    return (
        "<iframe class='viewer3d-frame' "
        f"srcdoc=\"{escaped_doc}\" "
        "sandbox='allow-scripts allow-same-origin'></iframe>"
    )


def _read_upload(upload: Any) -> tuple[Path, str] | tuple[None, str]:
    if upload is None:
        return None, ""
    path = Path(upload.name)
    try:
        return path, path.read_text(errors="ignore")
    except Exception:
        return path, ""


def render_uploaded_structure(
    ligand_file: Any,
    receptor_file: Any,
    ligand_style: str,
    ligand_color: str,
    receptor_style: str,
    receptor_color: str,
    receptor_opacity: float,
    pose_index: int,
) -> tuple[str, str, str]:
    ligand_path, ligand_text = _read_upload(ligand_file)
    receptor_path, raw_receptor_text = _read_upload(receptor_file)
    if not ligand_path and not receptor_path:
        return "Upload a ligand, pose, or receptor file to render.", "", _badge("No structure loaded")

    model_type = None
    if ligand_path:
        model_type = SUPPORTED_LIGAND_TYPES.get(ligand_path.suffix.lower())
        if not model_type:
            return "Unsupported ligand file type.", "", _badge("Unsupported")
        if not ligand_text.strip():
            return "Ligand file is empty or unreadable.", "", _badge("Read error")

    pose_total = None
    vina_score = None
    if model_type == "pdbqt":
        pose_total = count_pdbqt_poses(ligand_text)
        pose_text = extract_pdbqt_pose(ligand_text, int(pose_index or 1))
        if not pose_text:
            return f"Pose {pose_index} not found.", "", _badge(f"Pose 1-{pose_total}" if pose_total else "Pose not found")
        vina_score = extract_pdbqt_vina_score(pose_text)
        converted = pdbqt_to_pdb(pose_text)
        ligand_text = converted or pose_text
        model_type = "pdb" if converted else "pdbqt"

    receptor_text = None
    receptor_type = "pdb"
    if receptor_path:
        receptor_type = SUPPORTED_RECEPTOR_TYPES.get(receptor_path.suffix.lower())
        if not receptor_type:
            return "Unsupported receptor file type.", "", _badge("Unsupported")
        if not raw_receptor_text.strip():
            return "Receptor file is empty or unreadable.", "", _badge("Read error")
        receptor_text = pdbqt_to_pdb(raw_receptor_text) if receptor_type == "pdbqt" else raw_receptor_text
        if receptor_type == "pdbqt" and receptor_text:
            receptor_type = "pdb"

    html = build_3d_html(
        ligand_text,
        model_type,
        ligand_style=ligand_style,
        ligand_color=ligand_color,
        receptor_text=receptor_text,
        receptor_type=receptor_type,
        receptor_style=receptor_style,
        receptor_color=receptor_color,
        receptor_opacity=float(receptor_opacity or 0.85),
    )
    if pose_total:
        pose_i = max(1, min(int(pose_index or 1), pose_total))
        suffix = f" Vina {vina_score:.2f}" if vina_score is not None else ""
        return (
            f"Rendered pose {pose_i} of {pose_total}.{suffix}",
            html,
            _badge(f"Pose {pose_i}/{pose_total}{' • Vina %.2f' % vina_score if vina_score is not None else ''}"),
        )
    if ligand_path and receptor_path:
        return f"Rendered {ligand_path.name} with {receptor_path.name}.", html, _badge("Ligand + receptor")
    if receptor_path:
        return f"Rendered {receptor_path.name}.", html, _badge("Receptor surface")
    return f"Rendered {ligand_path.name}.", html, _badge("Single structure")


def shift_pose_index(ligand_file: Any, pose_index: int, delta: int) -> int:
    current = max(1, int(pose_index or 1))
    path, text = _read_upload(ligand_file)
    if not path or path.suffix.lower() != ".pdbqt":
        return max(1, current + delta)
    count = count_pdbqt_poses(text)
    if count <= 0:
        return max(1, current + delta)
    return max(1, min(count, current + int(delta)))


def _badge(text: str) -> str:
    return f"<span class='viewer3d-badge'>{text}</span>"
