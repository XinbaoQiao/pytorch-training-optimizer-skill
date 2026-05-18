# Precision Policy

Use this before changing AMP, TF32, FP16, BF16, or FP8 behavior. Ask the user to approve precision changes before editing code.

## Defaults

| Policy | Use when | Checks |
| --- | --- | --- |
| FP32 | Debugging, small models, strict reproducibility | Slowest; useful baseline |
| TF32 | FP32 code on Ampere+ with matmul/conv cost | Slight numeric differences from strict FP32; record policy |
| BF16 | Ampere/Hopper/Blackwell training default candidate | Hardware support, loss parity, optimizer stability |
| FP16 | Legacy GPUs or stacks where BF16 is unavailable | GradScaler/overflow checks, NaN monitoring |
| FP8 | Second-stage optimization after stable BF16 | Hardware/library support, scaling policy, loss/eval parity, distributed compatibility |

## BF16 Checklist

- GPU supports efficient BF16.
- Loss curve or metric proxy stays acceptable.
- Reductions and metrics remain FP32 where needed.
- Optimizer state precision is intentional.
- Checkpoint metadata records precision policy.

## FP16 Checklist

- Use gradient scaling unless the framework handles it.
- Monitor overflow, NaNs, and gradient norm.
- Keep stability-sensitive operations in FP32.
- Do not assume FP16 is faster than BF16 on modern NVIDIA GPUs.

## TF32 Checklist

- Explicitly record `torch.backends.cuda.matmul.allow_tf32` and cuDNN TF32 behavior.
- Explain that TF32 can speed FP32 matrix math but changes exact FP32 reproducibility.
- Compare a loss proxy if the project has tight numerical tolerances.

## FP8 Checklist

- Confirm Hopper/Blackwell or another supported target.
- Choose library path intentionally, such as torchao or Transformer Engine.
- Keep attention in BF16 or another supported policy if the FP8 path does not support it.
- Validate loss/eval parity, gradient norm, and NaN/Inf behavior.
- Verify distributed all-gather/reduce paths and checkpoint compatibility.
- Keep BF16 fallback.

## Reporting

Always include precision, TF32 policy, GradScaler behavior, and any FP8 scaling policy in `references/comparison_report.md`.
