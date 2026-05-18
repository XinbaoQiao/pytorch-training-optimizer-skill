# Compiler and CUDA Graphs

Use this when `torch.compile` or CUDA graph capture may help. Ask for user confirmation before changing code or configs.

## When It Helps

`torch.compile` is most likely to help when the hot path is tensor-heavy, shapes are stable, and the profile shows many small kernels, Python overhead, or unfused compute.

CUDA graphs and `mode="reduce-overhead"` are most relevant when CPU launch overhead creates gaps between kernels. They are less useful when the run is dominated by large kernels, dataloading, communication, or checkpoint stalls.

## Modes

| Mode | Use when | Risk |
| --- | --- | --- |
| default | First compile attempt on stable model code | May be modest if graph breaks remain |
| `fullgraph=True` | Debugging graph breaks | Fails fast; not always production mode |
| `mode="reduce-overhead"` | Many small kernels or launch overhead | Can use more memory and requires stable capture-friendly behavior |
| `mode="max-autotune"` | Matmul/conv-heavy stable shapes | Longer warmup and autotune cost |

## Graph Break Triage

Look for:

- `.item()`, `.cpu()`, `.numpy()`, or data-dependent Python branching
- Printing, logging, progress bars, or warning paths in the hot loop
- Shape-dependent branches or frequent dynamic shapes
- Mutation-heavy code or unsupported custom ops
- Python containers whose structure changes across steps

Debugging pattern:

```python
compiled = torch.compile(model, fullgraph=True)
```

Use the failure to identify the first break. Fix only if the change is safe and user-approved.

## Recompile Triage

If compile appears slow or unstable:

- Separate compile cold start from steady-state timing.
- Check whether input shapes, sequence lengths, batch sizes, or Python guards change frequently.
- Record how many warmup steps are needed before measurement.
- Consider bucketing or padding only if the user accepts the data/compute tradeoff.

## CUDA Graph Compatibility Checklist

- Static or well-bucketed shapes
- Stable control flow
- No hot-path CPU/GPU synchronization
- No changing Python-side data structures in captured regions
- No allocation patterns that break capture
- Dataloader and host-to-device transfer outside captured compute when needed

## Correctness and Rollback

- Keep an eager-mode flag.
- Compare loss proxy and small deterministic outputs when possible.
- Record compile mode and warmup steps in the comparison report.
- Do not report speedup from a run that includes one-time compile cost unless the user cares about short jobs.
