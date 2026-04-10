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
python run_single_metal.py
python run_two_metal.py
```

`run_single_metal.py` currently supports `PHANTOM_NAME = "two_rods"` or `PHANTOM_NAME = "ring_rods"`.
`run_two_metal.py` currently supports `PHANTOM_NAME = "six_rods"`.
