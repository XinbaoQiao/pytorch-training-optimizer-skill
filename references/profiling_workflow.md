# Profiling Workflow

Use this when the project has no reliable baseline yet.

## Timing Harness

Measure steady-state step time:

```python
import time
import torch

warmup_steps = 10
measure_steps = 50
times = []

for step, batch in enumerate(loader):
    if step >= warmup_steps:
        torch.cuda.synchronize()
        start = time.perf_counter()

    loss = train_step(batch)

    if step >= warmup_steps:
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    if step >= warmup_steps + measure_steps:
        break
```

Rules:

- Synchronize only around measurement windows, not inside normal training.
- Exclude compile warmup, dataloader startup, cache fill, and first-epoch one-time work.
- Report p50/p95, not only mean.

## PyTorch Profiler Pass

Use a short scheduled trace:

```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule

prof_schedule = schedule(wait=2, warmup=2, active=4, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    record_shapes=False,
    profile_memory=True,
    with_stack=False,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_trace"),
) as prof:
    for step, batch in enumerate(loader):
        with record_function("train_step"):
            loss = train_step(batch)
        prof.step()
        if step >= 10:
            break
```

Use `record_function` ranges for `data_to_device`, `forward`, `backward`, `optimizer_step`, `logging`, `eval`, and `checkpoint` when the code is easy to annotate.

## What To Look For

- Large CPU gaps before CUDA kernels: dataloader, preprocessing, collation, host-to-device transfer, or Python overhead.
- Many tiny CUDA kernels: Python/launch overhead, missing fusion, compile graph breaks, or scalar tensor operations.
- Heavy `aten::item`, `cudaMemcpy`, or explicit syncs: logging or metric synchronization.
- High memory allocation churn: tensor creation in loops, non-reused buffers, expensive format conversions.
- NCCL-dominated steps: communication-bound distributed training, rank imbalance, or poor overlap.
- Periodic long steps: checkpoint, evaluation, visualization, W&B/TensorBoard, or sample export in the hot path.

## Escalation

- Use Nsight Systems when CPU/GPU overlap, kernel launch gaps, or NCCL scheduling need deeper inspection.
- Use Nsight Compute when one or two kernels dominate and occupancy/memory bandwidth details matter.
- Use framework-specific profilers only after the PyTorch-level trace identifies the suspicious region.
