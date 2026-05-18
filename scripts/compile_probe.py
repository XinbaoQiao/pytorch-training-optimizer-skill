#!/usr/bin/env python3
"""Small torch.compile availability and smoke benchmark probe."""
from __future__ import annotations

import argparse
import json
import time
from typing import Any


def sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def run_probe(device: str, mode: str, steps: int, dtype_name: str) -> dict[str, Any]:
    import torch

    if not hasattr(torch, "compile"):
        return {"torch_compile_available": False}

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).to(device=device, dtype=dtype)
    x = torch.randn(32, 1024, device=device, dtype=dtype)

    def bench(fn):
        times = []
        for i in range(steps + 5):
            sync(device)
            start = time.perf_counter()
            y = fn(x)
            loss = y.float().square().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync(device)
            if i >= 5:
                times.append(time.perf_counter() - start)
        return sum(times) / len(times)

    eager_time = bench(model)
    compiled = torch.compile(model, mode=None if mode == "default" else mode)
    compiled_time = bench(compiled)
    return {
        "torch_compile_available": True,
        "torch_version": torch.__version__,
        "device": device,
        "dtype": str(dtype),
        "mode": mode,
        "steps": steps,
        "eager_mean_step_time": eager_time,
        "compiled_mean_step_time": compiled_time,
        "speedup_eager_over_compiled": eager_time / compiled_time if compiled_time else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.device, args.mode, args.steps, args.dtype), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
