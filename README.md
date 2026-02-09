# tamuBO

TAMU Bayesian Optimization tools with multiple framework implementations.

## Frameworks

- `tamubo.exactbo`: exact expected-improvement search with partition-based bounds.
- `tamubo.gpugp`: GPU-oriented GP components.
- `tamubo.mobbo`: multi-objective BO utilities.

## Conda Setup (ExactBO)

This repository now uses framework-specific Conda environments.

```bash
git clone https://github.com/juanesfco/tamubo.git
cd tamubo
conda env create -f envs/exactbo.yml
conda activate tamubo-exactbo
python -m pip install -e .
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
│   └── exactbo.yml
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
