# Experiment 2: ExactBO vs BO Workflow Baselines

This experiment runs the same minimization problem across:

- `tamubo.exactbo.exactbo`
- `tamubo.bo.run_sklearn_grid_ei`
- `tamubo.bo.run_botorch_grid_ei`
- `tamubo.bo.run_botorch_optimize_ei`

The goal is to compare responses from each workflow under shared `X0`, `bounds`, objective, and BO iteration budget.

## Files

- `run_benchmark.py`: benchmark driver
- `benchmark_config.json`: configuration for bounds, initial design, and workflow-specific params
- `results/`: generated benchmark summary and error reports
- `logs/`: optional per-workflow logs (enabled by config)

## Run

From repository root:

```bash
python3 experiments/exactbo/experiment2/run_benchmark.py
```

Or with explicit config:

```bash
python3 experiments/exactbo/experiment2/run_benchmark.py \
  --config experiments/exactbo/experiment2/benchmark_config.json
```

## Notes

- BoTorch workflows require `torch`, `botorch`, and `gpytorch`.
- If BoTorch dependencies are unavailable, the driver still runs and records those workflows as `status=error` in the results JSON.
