# Diagnostic Matrix

Use this when the first baseline does not make the bottleneck obvious.

## Baseline Signals

Collect these before changing code when practical:

- Mean, p50, and p95 step time after warmup
- Throughput: `samples/sec`, `tokens/sec`, `images/sec`, or `steps/sec`
- GPU utilization, SM occupancy if available, and memory utilization
- Peak allocated and reserved CUDA memory
- Host-side gap between steps and dataloader wait time
- Forward, backward, optimizer, scheduler, evaluation, checkpoint, and logging time
- Multi-GPU communication time, NCCL timeline, and rank imbalance
- Task metric, validation loss, or short loss-curve proxy

## Classification

| Signal | Likely bottleneck | First checks |
| --- | --- | --- |
| Low GPU utilization, gaps before kernels, CPU busy | Input pipeline | `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, decode cost, storage latency, collation cost |
| High GPU utilization, high step time, stable memory | Compute-bound | BF16/FP16, optimized attention, fused ops, `torch.compile`, batch size |
| High memory use, OOM, tiny microbatches | Memory-bound | Batch sizing, FSDP/ZeRO, optimizer state, activation checkpointing only when activation memory dominates |
| Multi-GPU scaling is poor, NCCL-heavy traces | Communication-bound | FSDP/DDP strategy, bucket/group sizing, accumulation, topology, rank imbalance, overlap |
| Many tiny kernels, high CPU launch overhead | Python or launch overhead | Remove hot-path Python work, vectorize small ops, reduce graph breaks, use `torch.compile` |
| Frequent host-device syncs | Synchronization-bound | Remove per-step `.item()`, `.cpu()`, `.numpy()`, explicit syncs, synchronous logging |
| Periodic long steps | Side-effect overhead | Reduce logging/eval/checkpoint cadence, rank-zero-only work, async writes |
| First trace is inconclusive | Measurement issue | Add profiler ranges, exclude warmup, run enough steady-state steps, compare p50/p95 |

## Fast Triage Commands

- `nvidia-smi dmon -s pucm` for coarse utilization and memory behavior.
- `torch.profiler` for step decomposition and dataloader gaps.
- Nsight Systems for CPU/GPU overlap, kernel launch gaps, and NCCL timelines.
- Nsight Compute for kernel occupancy and attention/operator-level analysis.
- `torch._dynamo.explain` or compile logs when `torch.compile` appears ineffective.

## Routing Questions

Ask at most one short diagnostic question if the user has not provided enough signal:

- For "training is slow": "What are GPU utilization, step time, batch size, and dataloader/augmentation cost?"
- For "low GPU utilization": "Does the profiler show CPU/dataloader gaps or many tiny kernels?"
- For "OOM": "Is memory dominated by parameters/optimizer states, activations, attention, or batch data?"
- For "multi-GPU is slower": "What is single-GPU throughput versus N-GPU throughput, and does the trace show NCCL time or rank imbalance?"
- For "compile did not help": "Are there graph breaks, dynamic shapes, unsupported ops, or one-time compile cost included in the benchmark?"

## Guardrails

- Do not count compile warmup, dataloader startup, or cache population as steady-state training time.
- Do not call a speedup valid if effective batch size, precision policy, dropout behavior, or data order changed unintentionally.
- When changing precision or compiler settings, compare a short loss curve or deterministic smoke metric against the baseline.
