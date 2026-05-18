# Failure Modes

Use this before claiming an optimization succeeded.

## False Speedups

| Failure | Why it matters | Check |
| --- | --- | --- |
| Warmup included differently | Compile/cache/dataloader startup can dominate | Same warmup and measured steps |
| Batch size changed | Throughput no longer comparable | Compare global and microbatch size |
| Sequence length changed | Transformer cost changes quadratically for attention | Record sequence/context distribution |
| Data order changed | Loss proxy may drift | Seed and sampler check |
| Evaluation changed | Metric comparison invalid | Same eval command and dataset |
| Dropout behavior changed | Train semantics changed | Check train/eval mode and dropout args |
| Precision changed silently | Numerical behavior changed | Record AMP/TF32/FP8 policy |
| Checkpoint disabled | Goodput/safety changed | Report checkpoint cadence and blocking time |
| Compile cold start hidden | Short jobs may be slower | Separate cold start and steady state |
| Dependency fallback | Intended kernel not actually used | Log backend and fallback warnings |

## Numerical Regressions

Watch for:

- NaN/Inf loss or gradients
- Loss curve shift beyond expected noise
- Validation metric drop
- Gradient norm outliers
- Eager vs compiled output mismatch
- Different dropout or mask semantics

## Systems Regressions

Watch for:

- Higher peak memory or fragmentation
- Longer first epoch due to compile/autotune/cache fill
- More CPU RAM from dataloader workers
- More network/storage pressure
- Slower checkpoint resume
- Rank imbalance or NCCL stalls

## Final Report Rule

If a speedup is real but carries risk, say so directly and recommend keeping it behind a flag until longer training validates convergence and resume behavior.
