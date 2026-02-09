# Environments

This repository uses framework-specific Conda environments:

- `exactbo`: `envs/exactbo.yml` (available)
- `gpugp`: `envs/gpugp.yml` (planned)
- `mobbo`: `envs/mobbo.yml` (planned)

Create and activate:

```bash
conda env create -f envs/exactbo.yml
conda activate tamubo-exactbo
python -m pip install -e .
```
