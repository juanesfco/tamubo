# Environments

Choose the environment based on your workflow:

1. `exactbo` usage: use one of the three options below.
2. Torch-based `bo` usage: use the Docker setup in `envs/pytorch/`.

## ExactBO Environments

There are four supported ways to set up `exactbo`:

1. Conda (recommended): `envs/exactbo.yml`
2. Pip build: `envs/exactbo.txt` (CUDA 12, e.g. TAMU Vision)
3. Pip build for CUDA 13 (e.g. DGX Spark / GB10, aarch64): `envs/exactbo_cuda13.txt`
4. Pip CPU-only fallback (no `cupy`): `envs/exactbo_cpu.txt`

The default bounds (`bound_method="autobound"`) need `jax` and Google's
`autobound`. `autobound==0.1.7` works with `jax>=0.7.2,<0.10` (0.10 and 0.11
break its Jaxpr interpreter), so every spec pins `jax==0.8.2`. Without them,
pass `bound_method="interval"`.

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

## Option 3: Pip build for CUDA 13 (DGX Spark)

```bash
uv venv envs/venv_spark --python 3.12
source envs/venv_spark/bin/activate
uv pip install -r envs/exactbo_cuda13.txt
uv pip install -e .
python -m pytest experiments/exactbo/experiment3/test_bounds.py
```

## Option 4: Pip CPU-only fallback

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
