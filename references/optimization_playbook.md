# Optimization Playbook

Use these patterns after the baseline identifies a likely bottleneck.

## Mixed Precision

Preferred first pass on modern NVIDIA GPUs:

```python
amp_dtype = torch.bfloat16

with torch.autocast(device_type="cuda", dtype=amp_dtype):
    loss = model(batch).loss

loss.backward()
```

Checks:

- Confirm the GPU supports BF16 efficiently.
- Keep stability-sensitive code in FP32 where needed.
- Compare a short loss curve before and after.
- For FP16, use gradient scaling unless the framework already handles it.
- For FP32-heavy matmuls on Ampere or newer GPUs, consider TF32 policy explicitly and record it in the report.

## torch.compile

Use on stable tensor-heavy regions:

```python
model = torch.compile(model, mode="max-autotune")
```

Practical rules:

- Benchmark after warmup and exclude compile cold start.
- Start with the model or hot submodules before compiling the whole training step.
- Watch for graph breaks from Python control flow, dynamic shapes, `.item()`, data-dependent branches, mutation-heavy code, and unsupported custom ops.
- Keep an eager-mode flag for rollback.
- For distributed jobs, compile only after the distributed wrapping order is verified for the local PyTorch version.

## Attention Kernels

Evaluate the best kernel for the actual hardware and shape regime:

- Ampere: SDPA flash backend or FlashAttention-2 may be appropriate.
- Hopper H100/H200: evaluate FlashAttention-3 or another Hopper-optimized path.
- Blackwell B200/GB200: evaluate FlashAttention-4 or the current best Blackwell path, but verify install maturity and backward support.

Benchmark with the project settings:

- Sequence length and context pattern
- Head dimension and number of heads
- Causal/non-causal masks
- Dropout
- BF16/FP16/FP8 policy
- Forward and backward, not only inference

Checklist:

- Verify PyTorch version, CUDA version, GPU capability, and installed `flash-attn` version.
- Confirm tensor layout expected by the chosen backend.
- Confirm mask semantics, dropout, causal mode, and dtype are supported.
- Compare max/mean output difference on a small deterministic input.
- Benchmark enough sequence lengths to avoid optimizing for a non-representative shape.

## FSDP, ZeRO, and Checkpointing

Choose the memory lever by what consumes memory:

- Parameters/gradients/optimizer state dominate: prefer FSDP or ZeRO.
- Activations dominate: use activation checkpointing selectively.
- Both dominate: combine sharding with selective checkpointing and measure recompute cost.

Rules:

- Do not use checkpointing as the first fix just because training OOMs.
- Preserve effective global batch size unless explicitly changing the experiment.
- Confirm checkpoint load/save compatibility after FSDP/ZeRO changes.
- For PyTorch FSDP2, initialize the optimizer after sharding and call the module normally so hooks run.

## Dataloader and Transfers

Typical improvements:

```python
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=num_workers > 0,
    prefetch_factor=4 if num_workers > 0 else None,
)

batch = {
    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    for k, v in batch.items()
}
```

Tune empirically:

- Increase `num_workers` until CPU, memory bandwidth, storage, or Python overhead stops improving.
- Use pinned memory and non-blocking transfers for CUDA tensors.
- Move heavy image/video/3D preprocessing out of the step loop when possible.
- Cache or precompute expensive deterministic transforms when input starvation is visible.
- For large shuffled corpora, avoid per-worker giant Python lists of shuffled pointers; prefer compact deterministic index mappings when possible.

## Logging and Synchronization

Avoid this in every step:

```python
loss_value = loss.item()
logger.log({"loss": loss_value})
```

Prefer cadence-based logging:

```python
loss_accum = loss_accum + loss.detach()

if step % log_every == 0:
    loss_value = (loss_accum / log_every).float().item()
    logger.log({"loss": loss_value}, step=step)
    loss_accum = torch.zeros((), device=device)
```

Rules:

- Keep `.item()`, `.cpu()`, `.numpy()`, and explicit `torch.cuda.synchronize()` out of the hot path unless measuring.
- Log from rank zero unless every rank must emit data.
- Move image/video/sample generation, mesh export, evaluation, and checkpoints to lower cadence or async paths.

## Checkpointing

If checkpoint steps create periodic stalls:

- Save only from the ranks required by the distributed strategy.
- Prefer distributed checkpoint formats for sharded training state.
- Evaluate asynchronous checkpointing when checkpoint cost is visible in the profile.
- Keep recovery safety: do not simply checkpoint less often unless the user accepts the risk.
- Verify resume after changing checkpoint format or save path.

## Frameworks

Use framework features only when they fit the existing project:

- PyTorch Lightning, Accelerate, DeepSpeed, or Megatron-style stacks can reduce boilerplate but are migrations, not default fixes.
- If the project already uses one, prefer its native precision, strategy, profiler, and checkpoint APIs.
- If the project is raw PyTorch research code, prefer small targeted patches before framework refactors.

## Borrowing From Reference Stacks

Use reference repositories as design guides:

- TorchTitan: PyTorch-native FSDP2, tensor/pipeline/context parallelism, `torch.compile`, Float8, distributed checkpointing, and structured training metrics.
- DeepSpeed: ZeRO stages, offload, ZeRO++, sequence parallel variants, and memory/communication tradeoffs.
- Megatron-Core/NeMo: tensor, pipeline, sequence, context, expert parallelism, Transformer Engine FP8, and large-scale resiliency.
- Liger Kernel: fused/chunked Triton kernels for RMSNorm, RoPE, SwiGLU, cross entropy, and post-training losses.
- Mosaic Streaming, WebDataset, DALI: data pipeline fixes when dataloading, decode, augmentation, or cloud/object-store access starves GPUs.

Do not copy stack-specific assumptions blindly. Convert them into local checks: shape support, dtype support, distributed compatibility, checkpoint compatibility, reproducibility, and before/after measurements.
