# AutoIA 1.0.0 execution policy

## Principle

Execution strategy may change cost, but it may not change proof truth. AutoIA V1 therefore separates trusted proof execution from optional proposal/assist execution.

## CPU proof

The trusted machine-rigorous backend is CPU binary64.

The preferred finite many-box interfaces are:

```python
session.enclose_many(..., workers="auto", parallel="auto")
session.enclose_many_bounds(..., workers="auto", parallel="auto")
```

With both settings on `"auto"`, V1 uses a deterministic safe-auto heuristic:

1. normalize the finite box batch;
2. divide it into chunks (`chunk_size=64` by default);
3. fewer than four chunks run serially;
4. at four or more chunks, use at most one worker per chunk, capped by the available CPU count;
5. prefer process workers when the model is serializable, otherwise threads;
6. explicit `workers=` or `parallel=` choices override the heuristic.

The four-chunk crossover is a performance policy, not a proof assumption. `ManyBoxExecutionInfo.scheduling_note` records the resolved choice.

## GPU proposal/assist

MPS and CUDA are optional non-rigorous proposal devices. `proposal_map(..., device="auto")` deliberately selects CPU in V1. AutoIA does not infer GPU profitability from point count because dispatch, transfer, dtype, and arithmetic intensity are application dependent.

Applications that want a GPU explicitly request:

```python
proposal_map(..., device="mps")
proposal_map(..., device="cuda")
```

Every `ProposalBatchResult` remains `machine_rigorous=False`. A rigorous conclusion requires independent CPU AutoIA verification.

## Evidence motivating the policy

On a 16-core Apple Silicon machine, the release-candidate benchmark showed:

- 100 rigorous boxes: process auto-style execution was slightly slower than serial (0.942x speedup);
- 1,000 rigorous boxes: 5.825x process speedup;
- 10,000 rigorous boxes: 11.192x process speedup;
- the light Torch proposal kernel was faster on CPU than MPS at 100, 1,000, and 10,000 points, although MPS/CPU numerical agreement was at float32 scale and all MPS center proposals were contained by CPU certificates.

The policy therefore automates the part with a stable workload signal (many independent proof boxes) and leaves GPU proposal selection explicit.
