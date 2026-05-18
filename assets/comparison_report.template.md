# Before/After Comparison Report

Use this format after implementing and validating an optimization.

## Run Identity

- Git commit before:
- Git commit after:
- Command before:
- Command after:
- Config path and hash:
- Conda/pip environment hash:
- `torch.__config__.show()` captured:
- `scripts/collect_env.py` output path:
- Date/time:

## Run Conditions

- Hardware:
- Interconnect/topology:
- GPU count:
- GPU model:
- Driver version:
- PyTorch/CUDA/cuDNN/NCCL versions:
- Dataset or sample size:
- Global batch size:
- Microbatch size:
- Gradient accumulation:
- Sequence length, context length, or input resolution:
- Precision:
- TF32 policy:
- Attention backend:
- Compile mode:
- Distributed strategy:
- Checkpoint cadence:
- Logging/eval cadence:
- Warmup steps:
- Measured steps:
- Seed:

## Invariant Checks

| Invariant | Before | After | Same? |
| --- | --- | --- | --- |
| Global batch size | | | |
| Microbatch size | | | |
| Gradient accumulation | | | |
| Sequence length / image resolution | | | |
| Dataset split and sample count | | | |
| Data order / seed | | | |
| Loss definition | | | |
| Optimizer and scheduler | | | |
| Precision policy | | | |
| Dropout / train-eval mode | | | |
| Evaluation meaning | | | |
| Checkpoint resume behavior | | | |

## Speed Metrics

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Mean step time | | | |
| p50 step time | | | |
| p95 step time | | | |
| Step time std | | | |
| Throughput | | | |
| Throughput/GPU | | | |
| GPU utilization | | | |
| Peak CUDA allocated | | | |
| Peak CUDA reserved | | | |
| Dataloader wait | | | |
| Host-to-device copy time | | | |
| Forward time | | | |
| Backward time | | | |
| Optimizer time | | | |
| Communication time | | | |
| Checkpoint blocking time | | | |
| Eval/logging overhead | | | |
| Checkpoint-stall-adjusted goodput | | | |
| Scaling efficiency | | | |

Change formulas:

- Speedup: `before_mean_step_time / after_mean_step_time`
- Step time reduction: `(before - after) / before * 100%`
- Throughput increase: `(after - before) / before * 100%`
- Scaling efficiency: `multi_gpu_throughput / (single_gpu_throughput * gpu_count)`

## Correctness Metrics

| Metric | Before | After | Difference |
| --- | ---: | ---: | ---: |
| Training loss proxy | | | |
| Validation metric | | | |
| Evaluation loss | | | |
| Gradient norm | | | |
| Eager vs compiled max diff | | | |
| Eager vs compiled mean diff | | | |
| NaN/Inf count | | | |

## Correctness Notes

- Build/test status before:
- Build/test status after:
- Changed numerical precision:
- Changed TF32 or attention backend:
- Changed compile mode:
- Changed effective batch size:
- Changed data ordering:
- Changed checkpoint or evaluation behavior:
- Dependency changes:
- Residual risks:
- Rollback plan:

## Decision

- Accept / reject / keep behind flag:
- Reason:
- Follow-up measurements:
