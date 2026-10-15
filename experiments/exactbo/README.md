# ExactBO Experiments

Experiments for `tamubo.exactbo` live under this directory.

Current contents:

- `experiment1/`: baseline serial ExactBO, good to compare results but noted time unfeasibility
- `experiment2/`: ExactBO benchmark vs gridBO and gradBO
- `experiment3/`: bound tightening for pruning (interval vs AutoBound bounds, EI vs logEI) and the bound soundness tests
- `experiment4/`: stress test of the tighter bounds on TAMU Vision (correctness, multi-GPU, memory, full BO loop, other problems)
- `experiment5/`: length-scale-aware box splitting (`split_ratio`), which makes the uncapped 10-d search converge, and the partition tests
