# AutoIA heterogeneous acceleration roadmap

Checkpoint: AutoIA `0.24.0.dev14`

AutoIA separates **trusted proof execution** from **proposal/assist acceleration**.  Fast hardware is useful only if its role in the correctness argument is explicit.

## V1 trust boundary

The V1 machine-rigorous proof backend is audited CPU binary64 arithmetic.  Repeated independent boxes may be proved concurrently on multiple CPU processes or threads; scheduling does not alter the theorem proved for any box.

Optional PyTorch accelerators are exposed through:

```python
available_accelerators()
proposal_map(...)
proposal_box_centers(...)
```

The current device roles are:

- `cpu`: trusted for AutoIA proof;
- `mps`: Apple Metal proposal/assist only;
- `cuda`: NVIDIA proposal/assist only.

Every `ProposalBatchResult` has `machine_rigorous=False`.  GPU results can rank boxes, evaluate center proposals, build nominal preconditioners, or support application algorithms, but a final enclosure must be independently produced or verified by the CPU trusted backend.

## G1 - multicore CPU proof (implemented in V1)

`prepare(model).enclose_many(...)` supports serial, thread, and spawn-process execution.  `workers="auto"` resolves the available CPU count and `parallel="auto"` prefers process workers when the model can be serialized.  Optional `cloudpickle` broadens spawn-process support to many local functions and closures.  Streaming interfaces keep a bounded number of chunks in flight.

Each worker prepares the model once and uses its own audited trusted backend.  `ManyBoxExecutionInfo` records scheduling diagnostics but is not proof data.

## G2 - MPS/CUDA proposal backend (implemented in V1)

Good proposal workloads include:

- batched nominal model values;
- box-center values and ranking scores;
- nominal/reference linear solves or matrix preconditioners supplied by an application;
- root/eigenpair/trajectory proposals;
- controller or global-optimizer heuristics;
- sampling diagnostics and other nontrusted preprocessing.

AutoIA does not automatically translate arbitrary engineering Python into a Torch graph.  Applications that want GPU proposal acceleration supply a Torch-compatible batched proposal function.  The ordinary AutoIA engineering model remains independently certifiable on CPU.

### Apple Metal / MPS

PyTorch MPS provides the practical Apple Silicon interface.  The V1 AutoIA proposal helper defaults to float32 on MPS and rejects a request for float64.  The trusted proof remains CPU binary64.

Use:

```bash
python examples/check_apple_mps.py
```

on Apple Silicon to verify device detection, proposal execution, trust labeling, and independent CPU box containment.

### NVIDIA CUDA

The same proposal interface supports CUDA.  `available_accelerators()` reports the PyTorch-visible device name and compute capability when available.

Use:

```bash
python examples/check_cuda_accelerator.py
```

on an NVIDIA system.  A DGX Spark is a useful V1 CUDA portability/proposal test platform; a data-center GPU should still be tested separately before making architecture-wide performance or future trusted-kernel claims.

## G3 - GPU-assisted proof with CPU verification

A future performance step may let a GPU calculate candidate bounds, compressed matrix summaries, or other expensive proof *proposals*, while a smaller trusted CPU residual/inclusion calculation verifies the final inequality.  Correctness remains CPU-based even if most arithmetic is proposed on the accelerator.

## G4 - trusted CUDA proof backend (research stage)

A trusted CUDA backend is plausible but is a distinct research and audit project.  It would require explicit directed-rounding kernels and independent checks of:

- addition, subtraction, multiplication, division, and square root rounding;
- subnormal and underflow behavior;
- FMA contraction and compiler transformations;
- overflow and infinities;
- reduction trees;
- transcendental implementations;
- compiler/toolkit versions;
- multiple GPU architectures and compute capabilities.

Ordinary PyTorch/CuPy CUDA operations are not automatically trusted proof arithmetic.

## G5 - trusted Metal proof backend (later research)

A trusted Metal/MPS proof path would need equivalent directed-rounding and audit machinery.  Until such primitives exist and are independently validated, Apple GPU execution remains proposal-side.

## Distributed/many-device execution

Multi-GPU or distributed systems can be useful at the orchestration layer because independent box proposals are naturally parallel.  A future system may distribute boxes across nodes/GPUs while returning every accepted certificate through trusted CPU verification.  Fully trusted distributed GPU proof would require the same kernel audit plus deterministic reduction/orchestration rules.
