# Attention Backend Matrix

Use this before changing attention code or forcing a backend. Ask the user to approve the exact backend change.

## Backend Matrix

| Backend | Best fit | Main checks | Common failure or fallback signs |
| --- | --- | --- | --- |
| PyTorch SDPA math | Compatibility baseline | dtype, mask semantics, dropout behavior | Slow but usually reliable |
| PyTorch SDPA flash | Ampere+ transformer attention | GPU capability, dtype, head dim, causal/dropout support | Warning or fallback to math/mem-efficient backend |
| PyTorch SDPA memory-efficient | Memory pressure with supported shapes | dtype, mask, dropout, backward support | Different numeric behavior from math backend |
| FlashAttention-2 | Ampere/Ada/Hopper broad use | install compatibility, tensor layout, head dim, dropout, backward | Import/build failure or layout mismatch |
| FlashAttention-3 | Hopper H100/H200-oriented workloads | CUDA version, Hopper GPU, package maturity, dtype, backward | Not available on non-Hopper or unsupported install |
| FlashAttention-4 | Blackwell B200/GB200-oriented workloads | Blackwell GPU, package maturity, PyTorch/CUDA compatibility | Experimental availability or unsupported backward/features |
| cuDNN attention | NVIDIA stack where cuDNN backend is selected | determinism settings, dtype, mask, version | Non-determinism or fallback |
| xFormers | Legacy/diffusion/nonstandard attention fallback | binary compatibility, op support, PyTorch version | Silent fallback or slower than SDPA |

## Project-Specific Inputs To Record

- GPU model and capability
- PyTorch, CUDA, cuDNN, and optional `flash-attn` versions
- Sequence length distribution
- Head dimension and number of heads
- Causal vs non-causal
- Mask type and shape
- Dropout probability and train/eval behavior
- BF16/FP16/FP8/FP32 policy
- GQA/MQA use
- Variable-length or packed sequences
- Forward-only versus forward+backward

## Correctness Checks

- Use a small deterministic input.
- Compare max and mean absolute difference against the current backend.
- Verify dropout handling explicitly. Evaluation paths should pass zero dropout where needed.
- Verify mask semantics; bool and additive masks can differ by implementation.
- Run at least one backward pass if training uses the backend.

## Benchmark Rules

- Benchmark representative sequence lengths, not only one convenient size.
- Include backward time and memory, not only forward latency.
- Record fallback warnings.
- Keep a native SDPA fallback unless the user explicitly wants a hard dependency.
