# Literature and Repository Patterns

Use this file when the task benefits from proven training-system designs. Treat these projects as pattern libraries, not dependencies to add automatically.

## Core Distributed Training

| Source | What to borrow | Use when | Local checks |
| --- | --- | --- | --- |
| TorchTitan | FSDP2, DTensor, tensor/pipeline/context parallelism, `torch.compile`, Float8, distributed async checkpointing, structured metrics | PyTorch-native large-scale training, especially transformer or generative model code | PyTorch version, distributed wrapping order, checkpoint format, loss parity, compile warmup, rank metrics |
| DeepSpeed / ZeRO | ZeRO-1/2/3 state partitioning, ZeRO-Offload, ZeRO-Infinity, ZeRO++ communication reduction | Optimizer states, gradients, or parameters dominate memory; low bandwidth makes communication visible | Effective batch size, optimizer semantics, offload bandwidth, checkpoint compatibility, convergence parity |
| Megatron-Core / NeMo | Tensor, pipeline, sequence, context, and expert parallelism; Transformer Engine FP8; distributed optimizer; resiliency | Model parallelism is needed beyond data parallelism; long context, MoE, or very large transformer workloads | Topology, microbatch schedule, pipeline bubbles, sequence length, activation memory, checkpoint conversion |
| PyTorch FSDP/compile papers and blogs | FSDP plus `torch.compile`, graph-break removal, deterministic dataloader, selective activation checkpointing | Raw PyTorch projects where native APIs are preferred | Graph breaks, data-order equivalence, selective vs full checkpointing, profiler-backed MFU/throughput |

## Precision and Kernels

| Source | What to borrow | Use when | Local checks |
| --- | --- | --- | --- |
| FlashAttention / FA2 / FA3 / FA4 | IO-aware attention, Hopper-specific FA3, Blackwell-oriented FA4 ideas | Attention dominates time or memory, especially long sequence transformer-style models | Hardware generation, CUDA/PyTorch/flash-attn versions, mask support, dropout, dtype, backward support |
| torchao Float8 + FSDP2 | Float8 linear layers and Float8 all-gather for modern GPUs | Hopper/Blackwell training has enough matmul/communication cost to justify FP8 risk | Loss parity, scaling granularity, unsupported ops, communication dtype, evaluation quality |
| Transformer Engine | FP8/MXFP8/NVFP4 paths and fused transformer primitives on NVIDIA GPUs | Existing NVIDIA transformer stack or Megatron/NeMo-style code | GPU generation, AMP integration, checkpoint/state compatibility, convergence parity |
| Liger Kernel | Fused and chunked Triton kernels for RMSNorm, RoPE, SwiGLU, cross entropy, and alignment losses | HF-style LLM training or SFT/post-training spends time or memory in common transformer ops/losses | Model support, exactness tests, Triton/Torch versions, multi-GPU compatibility, convergence parity |
| xFormers | Memory-efficient attention variants and operator alternatives | Older PyTorch versions, diffusion stacks, or nonstandard attention variants need a practical fallback | Binary compatibility, fallback path, shape/mask support, whether SDPA/FlashAttention is already faster |

## Data Pipeline

| Source | What to borrow | Use when | Local checks |
| --- | --- | --- | --- |
| Mosaic StreamingDataset | Deterministic shuffling, cloud streaming, mid-epoch resume, shard-aware throughput | Cloud/object-store data or huge datasets cause startup, resume, or random I/O stalls | Sample order, resume semantics, cache size, shard size, worker count, convergence parity |
| WebDataset | Sequential tar-shard streaming through PyTorch `IterableDataset` | Dataset files are many/small or remote, and random file access is the bottleneck | Shard size, shuffle buffer, distributed shard splitting, decode cost |
| NVIDIA DALI | GPU-accelerated decode/preprocessing, pipeline prefetch, parallel batch processing | Image/video/audio decode or augmentation saturates CPU and leaves GPU idle | Operator coverage, GPU memory cost, augmentation parity, integration with distributed sampler |
| HF streaming datasets | Remote streaming and large public dataset access without full local download | Large text or multimodal corpora cannot be staged locally | Network throughput, shuffling quality, worker fanout, cache behavior |

## Measurement Patterns

- Report `tokens/sec/GPU` or `samples/sec/GPU` alongside step time so multi-GPU changes are comparable.
- Track MFU/TFLOPs when FLOP estimates are meaningful for transformer-style workloads.
- Record GPU memory, dataloader wait, communication time, checkpoint blocking time, and side-effect overhead; a single throughput number hides regressions.
- Compare a short loss curve or metric proxy after precision, kernel, compile, sharding, or dataloader changes.
- Keep exact run conditions: hardware, interconnect, PyTorch/CUDA versions, precision, attention backend, compile mode, distributed strategy, batch size, sequence length, and checkpoint/logging cadence.
- Before adopting a library pattern, ask the user to approve the dependency and explain fallback behavior.

## References

- TorchTitan: https://github.com/pytorch/torchtitan
- PyTorch Float8 + FSDP2 blog: https://pytorch.org/blog/training-using-float8-fsdp2/
- PyTorch FSDP + torch.compile throughput blog: https://pytorch.org/blog/maximizing-training-throughput/
- DeepSpeed: https://github.com/deepspeedai/DeepSpeed
- ZeRO paper: https://arxiv.org/abs/1910.02054
- Megatron-Core: https://developer.nvidia.com/megatron-core
- NVIDIA NeMo: https://github.com/NVIDIA/NeMo
- NVIDIA Transformer Engine: https://github.com/NVIDIA/TransformerEngine
- FlashAttention: https://github.com/Dao-AILab/flash-attention
- FlashAttention-3 paper: https://arxiv.org/abs/2407.08608
- FlashAttention-4 paper: https://arxiv.org/abs/2603.05451
- Liger Kernel: https://github.com/linkedin/Liger-Kernel
- Mosaic Streaming: https://github.com/mosaicml/streaming
- WebDataset: https://rom1504.github.io/webdataset/
- NVIDIA DALI: https://github.com/NVIDIA/DALI
