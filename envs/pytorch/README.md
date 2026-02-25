# PyTorch BO Container

Use this container for BO examples/workflows that depend on `torch`.

This setup is based on `nvcr.io/nvidia/pytorch:26.01-py3` and maps your host
UID/GID so files created in the container are owned by your local user.

## Files

- `Dockerfile`: NVIDIA PyTorch base image + host-mapped user.
- `compose.yaml`: GPU-enabled service + workspace mount.
- `dev.sh`: helper script for build/run/freeze/install.
- `requirements.user.txt`: reproducible snapshot of `pip --user` packages.

## One-time: login to NGC

```bash
docker login nvcr.io
```

Use username `$oauthtoken` and your NGC API key as password.

## Build and open shell

```bash
./envs/pytorch/dev.sh shell
```

## Install project dependencies in container

```bash
python -m pip install --user -U pip
python -m pip install --user -e /workspace
```

For BoTorch-based scripts, install the Python stack you need:

```bash
python -m pip install --user botorch gpytorch
```

## Reproducible user packages

```bash
./envs/pytorch/dev.sh freeze
./envs/pytorch/dev.sh install
```

## Other commands

```bash
./envs/pytorch/dev.sh build
./envs/pytorch/dev.sh up
./envs/pytorch/dev.sh exec bash
./envs/pytorch/dev.sh down
```
