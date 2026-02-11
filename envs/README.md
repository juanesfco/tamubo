# Environments

There are three supported ways to set up `exactbo`:

1. Conda (most stable): `envs/exactbo.yml`
2. Pip build: `envs/exactbo.txt`
3. Pip fallback without `cupynumeric`: `envs/exactbo_nocp.txt`

## Option 1: Conda (Recommended)

This is the most stable path. It has the same runtime requirements as Legate and needs an NVIDIA GPU.

```bash
conda env create -f envs/exactbo.yml
conda activate tamubo-exactbo
pip install -e .
```

## Option 2: Pip build

```bash
pip install -r envs/exactbo.txt
pip install -e .
```

## Option 3: Pip fallback (no cupynumeric)

If `cupynumeric` cannot be built/resolved, install:

```bash
pip install -r envs/exactbo_nocp.txt
pip install -e .
```
