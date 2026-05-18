# Profiling Workflow

Use this when the project has no reliable baseline yet. Start with lightweight timing; use heavier profilers only when the cheap signal is insufficient.

## Level 0: Lightweight Timing

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
- Report p50/p95 and standard deviation, not only mean.
- Record `tokens/sec/GPU` or `samples/sec/GPU` for distributed runs.
- Record peak allocated and reserved CUDA memory.
- Use `scripts/step_timer.py` to summarize raw timing arrays.

## Level 1: PyTorch Profiler Pass

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

Profiler overhead rules:

- Keep `record_shapes=False` and `with_stack=False` for the first pass.
- Enable `record_shapes`, `with_stack`, or `with_flops` only for targeted passes.
- Avoid making conclusions from a trace that includes compile cold start unless cold start is the actual problem.
- For memory investigations, prefer dedicated memory history/snapshot passes over leaving heavy memory tracing enabled for every run.

## Level 2: Systems and Memory Profiling

Escalate when PyTorch profiler identifies a suspicious region but not the cause:

- Use Nsight Systems for CPU/GPU overlap, kernel launch gaps, NCCL scheduling, dataloader gaps, and host-to-device transfer overlap.
- Use Nsight Compute when one or two kernels dominate and occupancy, memory bandwidth, tensor-core use, or instruction mix matters.
- Use `torch.cuda.memory._record_memory_history()` and memory snapshots for allocation churn or fragmentation investigations.
- For distributed jobs, inspect rank imbalance and rank-local traces; one slow rank can stall the full job.

## What To Look For

- Large CPU gaps before CUDA kernels: dataloader, preprocessing, collation, host-to-device transfer, or Python overhead.
- Many tiny CUDA kernels: Python/launch overhead, missing fusion, compile graph breaks, or scalar tensor operations.
- Heavy `aten::item`, `cudaMemcpy`, or explicit syncs: logging or metric synchronization.
- High memory allocation churn: tensor creation in loops, non-reused buffers, expensive format conversions.
- NCCL-dominated steps: communication-bound distributed training, rank imbalance, or poor overlap.
- Periodic long steps: checkpoint, evaluation, visualization, W&B/TensorBoard, or sample export in the hot path.

## Before Editing

After the trace identifies a likely fix, stop and ask the user to approve the specific change set. Use `references/change_confirmation.md`.
