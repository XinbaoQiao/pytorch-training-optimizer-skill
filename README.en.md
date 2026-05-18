# PyTorch Training Optimizer Skill

[中文](README.md) | English

## Overview

PyTorch Training Optimizer Skill is a Codex skill for diagnosing and optimizing PyTorch research training code. It focuses on training throughput, GPU utilization, memory efficiency, input pipelines, and multi-GPU scaling.

It is useful for training-system problems such as:

- Diagnosing low GPU utilization, high step time, unstable throughput, or poor multi-GPU scaling
- Evaluating compute-side optimizations such as mixed precision, `torch.compile`, attention backends, and kernel fusion
- Choosing distributed and memory strategies such as DDP, FSDP/FSDP2, ZeRO, and activation checkpointing
- Optimizing input-pipeline bottlenecks from dataloaders, decoding, prefetching, host-to-device transfer, and storage access
- Reducing hot-path overhead from synchronization, logging, evaluation, checkpointing, and visualization
- Validating speedups with before/after metrics while keeping loss, task metrics, batch size, data order, and checkpoint semantics comparable

## Important Constraint: Ask Before Editing Code

This skill proposes diagnosis and changes first. Before editing training code, configs, precision policy, compiler settings, attention backend, dataloaders, checkpoints, or distributed strategy, it must ask the user to confirm the exact change set.

The confirmation request must explain in plain language:

- Which files or commands will change
- Why the change may make training faster
- Which metric is expected to improve
- Possible bad outcomes, such as worse loss, higher memory use, compile cold-start overhead, checkpoint incompatibility, dependency failures, or lower reproducibility
- How to roll back
- How the before/after result will be validated

Only the user-approved items should be implemented.

## Download and Install

Download from GitHub:

```bash
git clone https://github.com/XinbaoQiao/pytorch-training-optimizer-skill.git
```

Install into the user-level Codex skills directory:

```bash
mkdir -p "$HOME/.agents/skills"
cp -R pytorch-training-optimizer-skill "$HOME/.agents/skills/pytorch-training-optimizer"
```

You can also install it inside a repository so the skill is shared by that repo:

```bash
mkdir -p "$REPO_ROOT/.agents/skills"
cp -R pytorch-training-optimizer-skill "$REPO_ROOT/.agents/skills/pytorch-training-optimizer"
```

Admin-level installation path:

```bash
sudo mkdir -p /etc/codex/skills
sudo cp -R pytorch-training-optimizer-skill /etc/codex/skills/pytorch-training-optimizer
```

Older Codex setups may still use:

```bash
${CODEX_HOME:-$HOME/.codex}/skills/pytorch-training-optimizer
```

Restart Codex if the update does not appear.

## Usage

Invoke it directly in Codex:

```text
Use $pytorch-training-optimizer to profile and optimize this PyTorch training script.
```

Or describe a more specific target:

```text
Use $pytorch-training-optimizer to find why my 8xH200 training run has low GPU utilization and propose safe speedups. Ask before editing code.
```

## Contents

- `SKILL.md`: main skill workflow, trigger description, and pre-edit confirmation rule
- `agents/openai.yaml`: Codex UI metadata
- `scripts/collect_env.py`: collect PyTorch/CUDA/NCCL/GPU/environment details
- `scripts/step_timer.py`: summarize step-time mean/p50/p95/std metrics
- `scripts/profiler_trace.py`: generate a PyTorch profiler integration template
- `scripts/compare_runs.py`: compare before/after run records and check experiment invariants
- `scripts/compile_probe.py`: check `torch.compile` availability and run a small compile benchmark
- `scripts/sdpa_probe.py`: check SDPA/attention backend availability and run a small attention benchmark
- `scripts/dataloader_sweep.py`: sweep `num_workers`, `prefetch_factor`, `pin_memory`, and `persistent_workers`
- `scripts/checkpoint_goodput.py`: separate raw throughput from checkpoint-stall-adjusted goodput
- `references/`: profiling, compiler, attention, precision, distributed, dataloader, checkpoint, and failure-mode references
- `assets/`: run record schema and comparison report template
- `tests/`: smoke tests for metadata, references, and script compilation

## Design Principles

This skill prioritizes infrastructure-only optimization: measure a baseline, identify the bottleneck, propose understandable changes, wait for user confirmation, apply the smallest high-leverage patch, then validate with throughput, step time, GPU utilization, memory, and a loss or metric proxy. By default, it does not change data semantics, loss functions, optimizers, effective batch size, evaluation meaning, or checkpoint format.

## References and Integration

The skill incorporates ideas from public training optimization and PyTorch engineering routers, a Flash Attention focused skill, and systems such as TorchTitan, DeepSpeed/ZeRO, Megatron-Core/NeMo, Liger Kernel, Mosaic Streaming, WebDataset, NVIDIA DALI, and official PyTorch profiler, DataLoader, FSDP/compile, and Distributed Checkpoint documentation. The integration emphasizes diagnosis-first routing, bottleneck-specific fixes, shape and correctness validation for attention/backend/kernel changes, and profiling real training stalls from checkpointing, logging, and dataloading.

## Contributors

- [XinbaoQiao](https://github.com/XinbaoQiao): project maintainer
- OpenAI Codex: assisted with organizing, writing, and updating the skill content
