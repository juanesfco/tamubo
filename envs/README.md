# Environments

Choose the environment based on your workflow:

1. `exactbo` usage: use one of the three options below.
2. Torch-based `bo` usage: use the Docker setup in `envs/pytorch/`.

## ExactBO Environments

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

## Torch BO Environment (Docker)

For files that depend on `torch` (for example `examples/bo/`), use:

```bash
./envs/pytorch/dev.sh shell
```

See [`pytorch/README.md`](pytorch/README.md) for the complete workflow.

Note: this container setup is torch-focused and does not include `cupynumeric`.
