# Large Transformer Training Checklist

Use this when optimizing LLM, diffusion transformer, multimodal transformer, long-context, MoE, SFT, DPO, GRPO, or other large transformer training.

## Core Questions

- What is the target metric: tokens/sec, tokens/sec/GPU, MFU, cost per token, memory headroom, or time to validation target?
- Does the model fit with DDP, or are parameters/optimizer states/activations the limiting factor?
- Is attention time or memory dominant for the actual sequence length distribution?
- Is the input pipeline text-only, multimodal, remote, or decode-heavy?
- Are checkpoint stalls hurting goodput?
- Is the run long enough to amortize compile or autotune warmup?

## Modern Stack Checklist

- BF16 baseline is stable.
- TF32 policy is documented for FP32 paths.
- `torch.compile` considered only after graph-break/recompile risks are understood.
- SDPA/FlashAttention backend verified for masks, dropout, dtype, head dim, GQA/MQA, varlen, and backward.
- FSDP2 or ZeRO considered when optimizer states/parameters dominate memory.
- Selective activation checkpointing considered when activations dominate memory.
- Distributed checkpointing and async checkpointing considered when checkpoint stalls are visible.
- Dataloader is checkpointable or resume-safe if mid-epoch resume matters.
- Structured metrics include tokens/sec/GPU, MFU/TFLOPs when meaningful, memory, NCCL time, dataloader wait, and checkpoint blocking time.
- Post-training kernels such as Liger are considered only after exactness and model support checks.

## Do Not Change Without Approval

- Effective global batch size
- Sequence length or packing behavior
- Data mixture or sampling weights
- Loss function or masking
- Optimizer or scheduler math
- Evaluation protocol
- Checkpoint format or resume semantics
