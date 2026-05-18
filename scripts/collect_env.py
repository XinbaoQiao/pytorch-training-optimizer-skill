#!/usr/bin/env python3
"""Collect a reproducible PyTorch training environment record.

Usage:
  python scripts/collect_env.py --output env.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

ENV_KEYS = [
    "CUDA_VISIBLE_DEVICES",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_ALGO",
    "TORCH_COMPILE_DEBUG",
    "TORCH_LOGS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
]


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    if shutil.which(cmd[0]) is None:
        return {"available": False, "cmd": cmd}
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=20)
        return {"available": True, "cmd": cmd, "output": out.strip()}
    except Exception as exc:  # pragma: no cover - host dependent
        return {"available": True, "cmd": cmd, "error": repr(exc)}


def collect_torch() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False}
    import torch  # type: ignore

    info: dict[str, Any] = {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "git_version": getattr(getattr(torch, "version", None), "git_version", None),
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None) if hasattr(torch.backends, "cuda") else None,
        "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None) if hasattr(torch.backends, "cudnn") else None,
    }
    try:
        config = torch.__config__.show()
        info["config"] = config if isinstance(config, str) else None
    except Exception as exc:  # pragma: no cover - host dependent
        info["config_error"] = repr(exc)

    devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )
    info["cuda_devices"] = devices
    return info


def collect() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "environment": {key: os.environ.get(key) for key in ENV_KEYS if key in os.environ},
        "torch": collect_torch(),
        "nvidia_smi": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader",
            ]
        ),
        "git": {
            "rev_parse": run_cmd(["git", "rev-parse", "HEAD"]),
            "status_short": run_cmd(["git", "status", "--short"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", help="Write JSON to this path. Defaults to stdout.")
    args = parser.parse_args()
    data = collect()
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
