# PyTorch Training Optimizer Skill

[中文](README.md) | English

## Overview

PyTorch Training Optimizer Skill is a Codex skill for diagnosing and optimizing PyTorch research training code. It focuses on training throughput, GPU utilization, memory efficiency, and multi-GPU scaling.

It is useful for training-system problems such as:

- Diagnosing low GPU utilization, high step time, unstable throughput, or poor multi-GPU scaling
- Evaluating compute-side optimizations such as mixed precision, `torch.compile`, attention backends, and kernel fusion
- Choosing distributed and memory strategies such as DDP, FSDP, ZeRO, and activation checkpointing
- Optimizing input-pipeline bottlenecks from dataloaders, decoding, prefetching, host-to-device transfer, and storage access
- Reducing hot-path overhead from synchronization, logging, evaluation, checkpointing, and visualization
- Validating speedups with before/after metrics while keeping loss, task metrics, and checkpoint semantics comparable

## Download and Install

Download from GitHub:

```bash
git clone https://github.com/XinbaoQiao/pytorch-training-optimizer-skill.git
```

Install into your Codex skills directory:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R pytorch-training-optimizer-skill "${CODEX_HOME:-$HOME/.codex}/skills/pytorch-training-optimizer"
```

You can also click `Code -> Download ZIP` on GitHub, unzip the folder, and copy or rename it to:

```bash
${CODEX_HOME:-$HOME/.codex}/skills/pytorch-training-optimizer
```

## Usage

Invoke it directly in Codex:

```text
Use $pytorch-training-optimizer to profile and optimize this PyTorch training script.
```

Or describe a more specific target:

```text
Use $pytorch-training-optimizer to find why my 8xH200 training run has low GPU utilization and propose safe speedups.
```

## Contents

- `SKILL.md`: main skill workflow and trigger description
- `references/diagnostic_matrix.md`: bottleneck classification matrix
- `references/profiling_workflow.md`: PyTorch profiler and timing harness workflow
- `references/optimization_playbook.md`: optimization patterns for mixed precision, compile, attention, FSDP/ZeRO, dataloaders, and logging
- `references/literature_and_repos.md`: patterns distilled from papers and open-source training systems
- `references/comparison_report.md`: before/after performance report template
- `agents/openai.yaml`: Codex UI metadata

## Design Principles

This skill prioritizes infrastructure-only optimization: measure a baseline, identify the bottleneck, apply the smallest high-leverage change, then validate with throughput, step time, GPU utilization, memory, and a loss or metric proxy. By default, it does not change data semantics, loss functions, optimizers, effective batch size, evaluation meaning, or checkpoint format.

## References and Integration

The skill incorporates ideas from public training optimization and PyTorch engineering routers, a Flash Attention focused skill, and systems such as TorchTitan, DeepSpeed/ZeRO, Megatron-Core/NeMo, Liger Kernel, Mosaic Streaming, WebDataset, NVIDIA DALI, and official PyTorch profiler, DataLoader, FSDP/compile, and Distributed Checkpoint documentation. The integration emphasizes diagnosis-first routing, bottleneck-specific fixes, shape and correctness validation for attention/backend/kernel changes, and profiling real training stalls from checkpointing, logging, and dataloading.
