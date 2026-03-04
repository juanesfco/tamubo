# Experiment 2: ExactBO vs BoTorch Baselines

This experiment runs one framework per execution (selected in `experiment_config.json`):

- `tamubo.exactbo.exactbo`
- `tamubo.bo.run_botorch_grid_ei`
- `tamubo.bo.run_botorch_optimize_ei`

The output table is written to `results/experiment_results.csv` with one row per BO iteration.

## Files

- `run_experiment.py`: experiment driver
- `experiment_config.json`: framework, problem, and optimizer settings
- `problems.py`: problem registry (`d`, objective, `y*`, defaults for `bounds` and `X0`)
- `results/`: generated CSV table

## Run

From repository root:

```bash
python3 experiments/exactbo/experiment2/run_experiment.py
```

Or with explicit config:

```bash
python3 experiments/exactbo/experiment2/run_experiment.py \
  --config experiments/exactbo/experiment2/experiment_config.json
```

## Notes

- BoTorch workflows require `torch`, `botorch`, and `gpytorch`.
- Use `framework` in `experiment_config.json` with values:
  - `exactbo` (table label `exactBO`)
  - `botorch_grid` (table label `gridBO`)
  - `botorch_optimize` (table label `gradBO`)
