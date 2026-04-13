# IPMR_simulation
This repo contains the metal-artifact simulation data generation and reconstruction

## Environment setup

This repo has been exercised with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Examples:

```bash
python run_single_metal_conebeam.py
python run_two_metal_conebeam.py
```

`run_single_metal.py` currently supports `PHANTOM_NAME = "two_rods"` or `PHANTOM_NAME = "ring_rods"`.
`run_two_metal.py` currently supports `PHANTOM_NAME = "six_rods"`.

Each run script exposes editable config dictionaries near the top:
`MATERIAL_CONFIG`, `PHANTOM_CONFIG`, `CT_MODEL_CONFIG`, `RECON_CONFIG`, and `OUTPUT_CONFIG`.
