# Before/After Comparison Report

Use this format after implementing and validating an optimization.

## Run Conditions

- Hardware:
- Interconnect/topology:
- GPU count:
- PyTorch/CUDA versions:
- Command:
- Dataset or sample size:
- Global batch size:
- Microbatch size:
- Sequence length or input resolution:
- Precision:
- TF32 policy:
- Attention backend:
- Compile mode:
- Distributed strategy:
- Warmup steps:
- Measured steps:
- Seed:

## Speed Metrics

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Mean step time | | | |
| p50 step time | | | |
| p95 step time | | | |
| Throughput | | | |
| GPU utilization | | | |
| Peak CUDA memory | | | |
| Dataloader wait | | | |
| Communication time | | | |
| Checkpoint/eval/logging overhead | | | |

Change formulas:

- Speedup: `before_mean_step_time / after_mean_step_time`
- Step time reduction: `(before - after) / before * 100%`
- Throughput increase: `(after - before) / before * 100%`

## Performance Metrics

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Training loss proxy | | | |
| Validation metric | | | |
| Evaluation loss | | | |

## Correctness Notes

- Build/test status before:
- Build/test status after:
- Changed numerical precision:
- Changed TF32 or attention backend:
- Changed compile mode:
- Changed effective batch size:
- Changed data ordering:
- Changed checkpoint or evaluation behavior:
- Residual risks:
