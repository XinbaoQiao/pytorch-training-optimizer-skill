#!/usr/bin/env python3
"""Probe PyTorch scaled_dot_product_attention behavior on local hardware."""
from __future__ import annotations

import argparse
import json
import time
from typing import Any


def sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(seq_len: int, heads: int, head_dim: int, dtype_name: str, steps: int, causal: bool) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    if device == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    batch = 2
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)

    # Warmup
    for _ in range(5):
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=0.0)
        y.float().sum().backward()
        q.grad = k.grad = v.grad = None

    sync()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=0.0)
        y.float().sum().backward()
        q.grad = k.grad = v.grad = None
        sync()
        times.append(time.perf_counter() - start)

    return {
        "torch_version": torch.__version__,
        "device": device,
        "dtype": str(dtype),
        "seq_len": seq_len,
        "heads": heads,
        "head_dim": head_dim,
        "causal": causal,
        "mean_forward_backward_time": sum(times) / len(times),
        "min_forward_backward_time": min(times),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    print(json.dumps(bench(args.seq_len, args.heads, args.head_dim, args.dtype, args.steps, args.causal), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
