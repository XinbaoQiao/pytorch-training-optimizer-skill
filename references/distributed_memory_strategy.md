# Distributed and Memory Strategy

Use this before changing DDP, FSDP, FSDP2, ZeRO, activation checkpointing, tensor parallelism, pipeline parallelism, or optimizer state placement. Ask for confirmation before editing.

## Strategy Selection

| Symptom | First strategy | Notes |
| --- | --- | --- |
| Single GPU fits and scaling is acceptable | DDP | Simpler and low risk |
| Optimizer states or parameters dominate memory | FSDP/FSDP2 or ZeRO | Verify checkpoint format and optimizer init order |
| Activations dominate memory | Selective activation checkpointing | Trades memory for recompute time |
| Long context attention dominates memory | Attention backend + selective checkpointing | Check masks, dtype, backward |
| Multi-GPU scaling poor | Communication profiling | Inspect NCCL time, overlap, rank imbalance |
| Very large transformer | FSDP2 + tensor/context/pipeline parallel as needed | Use reference stacks as pattern libraries |

## FSDP/FSDP2 Checks

- Effective global batch size stays unchanged unless approved.
- Optimizer initialization order is correct for the chosen API.
- Checkpoint save/load and resume are tested.
- Mixed precision policy is explicit.
- Rank-local metrics are collected.
- Dataloader resume and sample order are preserved if needed.

## ZeRO Checks

- Stage choice matches memory pressure.
- Offload bandwidth does not become the new bottleneck.
- Checkpoint conversion and resume are tested.
- Communication overlap is measured.

## Activation Checkpointing Checks

- Use only when activation memory dominates or sharding alone is insufficient.
- Apply selectively to large blocks instead of blanket wrapping when possible.
- Measure recompute overhead and throughput drop.
- Confirm RNG/dropout behavior stays correct.

## Communication Checks

- Compare single-GPU throughput to N-GPU throughput.
- Report scaling efficiency.
- Inspect NCCL time and rank imbalance.
- Check topology and launch placement.
- Consider gradient accumulation only if the effective batch-size or optimizer-step semantics remain acceptable.

## Confirmation Language

Distributed changes often have hidden consequences. Explain that the change may make larger batches or models fit, but can also change checkpoint format, increase communication overhead, complicate resume, or reduce reproducibility.
