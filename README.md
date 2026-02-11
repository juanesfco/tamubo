# tamuBO

TAMU Bayesian Optimization tools with multiple framework implementations.

## Frameworks

- `tamubo.exactbo`: exact expected-improvement search with partition-based bounds.
- `tamubo.gpugp`: GPU-oriented GP components.
- `tamubo.mobbo`: multi-objective BO utilities.

## ExactBO Backends

`tamubo.exactbo` now supports a backend selector with CPU fallback:

- `backend="numpy"`: sequential CPU path (current stable implementation).
- `backend="cupynumeric"`: vectorized GPU-oriented path (requires `cupynumeric`).
- `backend="auto"`: picks `cupynumeric` when available, else falls back to `numpy`.

Unified entry point:

```python
from tamubo.exactbo import run_exactbo

result = run_exactbo(
    x0=X0,
    bounds=bounds,
    epsilon=epsilon,
    gp=gp,
    f=f,
    max_iters=10,
    max_partitions=30,
    backend="auto",
)
print(result.backend.selected)
print(result.X.shape, result.y.shape)
```

## How To Run ExactBO

There are multiple ways to run `tamubo.exactbo`:

1. `conda` (recommended, most stable): `envs/exactbo.yml`
2. `pip` build: `envs/exactbo.txt`
3. `pip` fallback without `cupynumeric`: `envs/exactbo_nocp.txt`

### Option 1: Conda (Most Stable)

Use this when possible. It has the same hardware/runtime expectations as Legate and requires an NVIDIA GPU.

```bash
git clone https://github.com/juanesfco/tamubo.git
cd tamubo
conda env create -f envs/exactbo.yml
conda activate tamubo-exactbo
pip install -U pip
pip install -e .
```

### Option 2: Pip Build

```bash
git clone https://github.com/juanesfco/tamubo.git
cd tamubo
python -m venv venvs/tamubo_exactbo
source venvs/tamubo_exactbo/bin/activate
pip install -U pip
pip install -r envs/exactbo.txt
pip install -e .
```

If `cupynumeric` cannot be built/resolved in your environment, use the fallback below.

### Option 3: Pip Fallback Without cupynumeric

```bash
git clone https://github.com/juanesfco/tamubo.git
cd tamubo
python -m venv venvs/tamubo_exactbo
source venvs/tamubo_exactbo/bin/activate
pip install -U pip
pip install -r envs/exactbo_nocp.txt
pip install -e .
```

## Environment Strategy

Use one environment per framework:

- `envs/exactbo.yml` (available)
- `envs/gpugp.yml` (planned)
- `envs/mobbo.yml` (planned)

## Repository Organization

```text
tamubo/
├── development/                  # branch-local exploration (scratch notebooks/scripts)
├── envs/                         # conda environments per framework
│   ├── exactbo.yml
│   ├── exactbo.txt
│   └── exactbo_nocp.txt
├── examples/                     # lightweight runnable demos
│   ├── exactbo/
│   ├── gpugp/
│   └── mobbo/
├── experiments/                  # reproducible studies
│   ├── exactbo/
│   │   ├── experiment1/
│   │   └── experiment2/
│   ├── gpugp/
│   └── mobbo/
├── src/
│   └── tamubo/
│       ├── exactbo/
│       ├── gpugp/
│       └── mobbo/
└── pyproject.toml
```

## Branch Workflow for `development/`

Use `development/` for work-in-progress and keep that work on dedicated branches.

Suggested flow:

1. Create a feature branch (example: `dev/exactbo-cupynumeric`).
2. Iterate in `development/`.
3. Promote mature outputs:
   - reusable library code -> `src/tamubo/`
   - minimal demos -> `examples/`
   - reproducible studies -> `experiments/<framework>/`
4. Merge only curated assets into `main`.

## License

Distributed under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

- **Juan E. Flórez-Coronel** - [juan.florez@tamu.edu](mailto:juan.florez@tamu.edu)
