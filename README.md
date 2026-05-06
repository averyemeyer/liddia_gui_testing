[![arXiv](https://img.shields.io/badge/arXiv-2303.14186-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2502.13959)

**Update: This repository is still in progress.**

# LIDDiA: Language-based Intelligent Drug Discovery Agent

In our [paper](https://arxiv.org/abs/2502.13959), we introduce a LLM-based agent for drug discovery called `LIDDiA` (Language-based Intelligent Drug Discovery Agent). Using `LIDDiA`, you can specify what properties the molecules should have, and `LIDDiA` will run computational tools to generate and evaluate the molecules.

## Quickstart

**Note: The current code for `LIDDiA` randomly sample molecules from TDC ZINC rather than using Pocket2Mol**

The environment dependencies for `conda` are in `environment.yml`.

Set your Anthropic API key via environment variable (recommended):

```console
export ANTHROPIC_API_KEY="sk-ant-..."
```

You can also place the key in `my-anthropic-key.txt` (ignored by git).

```console
python run.py --target EGFR --max_iter 10 --model "claude-3-5-sonnet-20241022"
```

The argument for `--target` must be one of the targets in `dataset/pdb/`. The list of arguments for `--model` is available [here](https://docs.claude.com/en/docs/about-claude/models/overview).

## Setup

Create the environment:

```console
conda env create -f environment.yml
conda activate liddia
```

For GUI users, `liddia_gui_app/environment.yml` is also provided as a GUI-friendly starting point based on the working `liddia-mac` development environment.

## GUI

This fork includes a modular local Gradio GUI in `liddia_gui_app/`. The GUI-specific setup, launch commands, architecture notes, troubleshooting, and tests live in `liddia_gui_app/README.md`.

## Changes From Upstream

This repository is based on the upstream LIDDIA source at [ninglab/LIDDIA](https://github.com/ninglab/LIDDIA/tree/main). Compared with upstream `main`, this working branch adds:

- A modular GUI application under `liddia_gui_app/` for launching, monitoring, recovering, reviewing, and exporting LIDDIA runs.
- GUI-focused launchers for macOS/Linux and Windows, plus a GUI environment file based on the working `liddia-mac` setup.
- A run-artifact reader layer that relies on normal LIDDIA outputs in `log/<run_id>/`, rather than one-off GUI-only data files.
- Focused tests for the GUI adapters, parsers, run-state recovery, report export, trends, molecule tables, logs, and 3D viewer.
- Archived earlier GUI prototypes under `old gui/` so historical versions are preserved but no longer treated as the active app.
- `.gitignore` cleanup for local runtime artifacts such as logs, downloaded data, docking temp files, pickles, Python caches, and API key files.

This branch also contains local changes to `run.py` that make runs easier to monitor from the GUI:

- Anthropic keys can be read from `ANTHROPIC_API_KEY` or `my-anthropic-key.txt`.
- Runs write incremental JSON snapshots with runtime metadata while they are active.
- Runtime metadata includes current iteration, max iterations, timestamps, and elapsed seconds.
- A lightweight heartbeat writes best-effort progress snapshots for recovery after browser/tab closure.
- Model responses that indicate completion without a new action are handled more gracefully.

## Citation

If you use the code in this repository, please cite with the following BibTeX entry:

```
@article{averly2025liddia,
  title={Liddia: Language-based intelligent drug discovery agent},
  author={Averly, Reza and Baker, Frazier N and Watson, Ian A and Ning, Xia},
  journal={arXiv preprint arXiv:2502.13959},
  year={2025}
}
```

If you use the dataset in this repository, please cite the following works as well:

```
@article{gaulton2012chembl,
  title={ChEMBL: a large-scale bioactivity database for drug discovery},
  author={Gaulton, Anna and Bellis, Louisa J and Bento, A Patricia and Chambers, Jon and Davies, Mark and Hersey, Anne and Light, Yvonne and McGlinchey, Shaun and Michalovich, David and Al-Lazikani, Bissan and others},
  journal={Nucleic acids research},
  volume={40},
  number={D1},
  pages={D1100--D1107},
  year={2012},
  publisher={Oxford University Press}
}
```

```
@article{burley2019rcsb,
  title={RCSB Protein Data Bank: biological macromolecular structures enabling research and education in fundamental biology, biomedicine, biotechnology and energy},
  author={Burley, Stephen K and Berman, Helen M and Bhikadiya, Charmi and Bi, Chunxiao and Chen, Li and Di Costanzo, Luigi and Christie, Cole and Dalenberg, Ken and Duarte, Jose M and Dutta, Shuchismita and others},
  journal={Nucleic acids research},
  volume={47},
  number={D1},
  pages={D464--D474},
  year={2019},
  publisher={Oxford University Press}
}
```

## Questions?

Please send an email to averly.1@buckeyemail.osu.edu
