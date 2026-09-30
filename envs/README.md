# Environments

Choose the environment based on your workflow:

1. `exactbo` usage: use one of the three options below.
2. Torch-based `bo` usage: use the Docker setup in `envs/pytorch/`.

## ExactBO Environments

There are three supported ways to set up `exactbo`:

1. Conda (recommended): `envs/exactbo.yml`
2. Pip build: `envs/exactbo.txt`
3. Pip CPU-only fallback (no `cupy`): `envs/exactbo_cpu.txt`

## Option 1: Conda (Recommended)

Needs an NVIDIA GPU. conda-forge's `cupy` brings its own CUDA libraries; the
spec solves on both `linux-64` and `linux-aarch64`.

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

The `cupy-cuda12x` wheel uses the system CUDA 12 runtime, so load it at run
time on clusters (TAMU Vision: `module load CUDA/12.9.1`).

## Option 3: Pip CPU-only fallback

Without a GPU, `backend="auto"` falls back to numpy:

```bash
pip install -r envs/exactbo_cpu.txt
pip install -e .
```

## Torch BO Environment (Docker)

For files that depend on `torch` (for example `examples/bo/`), use:

```bash
./envs/pytorch/dev.sh shell
```

See [`pytorch/README.md`](pytorch/README.md) for the complete workflow.
