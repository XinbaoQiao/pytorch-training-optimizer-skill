#!/usr/bin/env python3
"""Utilities and template for a short PyTorch profiler trace."""
from __future__ import annotations

import argparse
import textwrap

TEMPLATE = r"""
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch

prof_schedule = schedule(wait=2, warmup=2, active=4, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    record_shapes=False,
    profile_memory=True,
    with_stack=False,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_trace"),
) as prof:
    for step, batch in enumerate(loader):
        with record_function("data_to_device"):
            batch = move_to_device(batch, device)
        with record_function("forward_backward_optimizer"):
            loss = train_step(batch)
        prof.step()
        if step >= 10:
            break
"""


def make_profiler(trace_dir: str = "./profiler_trace", active: int = 4):
    """Return a PyTorch profiler configured for a short scheduled trace."""
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=2, warmup=2, active=active, repeat=1),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-template", action="store_true", help="Print an integration template.")
    args = parser.parse_args()
    if args.print_template:
        print(textwrap.dedent(TEMPLATE).strip())
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
