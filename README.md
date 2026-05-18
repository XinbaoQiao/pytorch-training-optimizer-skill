# PyTorch Training Optimizer Skill

中文 | [English](README.en.md)

## 简介

PyTorch Training Optimizer Skill 是一个面向研究训练代码的 Codex skill，用来诊断和优化 PyTorch 训练吞吐、GPU 利用率、显存效率和多卡扩展效率。

它适合处理这些常见问题：

- 训练没有启用 BF16/FP16 混合精度
- 没有使用 `torch.compile`，或存在大量 graph break
- H100/H200/B200 上仍然走通用 attention 路径
- 多卡训练里优先堆 gradient checkpointing，而不是先评估 FSDP/ZeRO
- 每个 step 用 `.item()`、`.cpu()`、`.numpy()` 或同步 logger 记录指标
- dataloader、解码、预取、host-to-device copy 导致 GPU 吃不满
- checkpoint、evaluation、可视化或 logging 在训练热路径里造成周期性长 step

## 下载和安装

从 GitHub 下载：

```bash
git clone https://github.com/XinbaoQiao/pytorch-training-optimizer-skill.git
```

安装到 Codex skill 目录：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R pytorch-training-optimizer-skill "${CODEX_HOME:-$HOME/.codex}/skills/pytorch-training-optimizer"
```

也可以在 GitHub 页面点击 `Code -> Download ZIP`，解压后把文件夹复制或重命名到：

```bash
${CODEX_HOME:-$HOME/.codex}/skills/pytorch-training-optimizer
```

## 使用

在 Codex 中直接调用：

```text
Use $pytorch-training-optimizer to profile and optimize this PyTorch training script.
```

或者给出更具体的目标：

```text
Use $pytorch-training-optimizer to find why my 8xH200 training run has low GPU utilization and propose safe speedups.
```

## 包含内容

- `SKILL.md`: skill 主 workflow 和触发描述
- `references/diagnostic_matrix.md`: 性能瓶颈分类表
- `references/profiling_workflow.md`: PyTorch profiler 和 timing harness 工作流
- `references/optimization_playbook.md`: 混合精度、compile、attention、FSDP/ZeRO、dataloader 和 logging 优化模式
- `references/literature_and_repos.md`: 论文和开源训练系统里的可借鉴模式
- `references/comparison_report.md`: before/after 性能报告模板
- `agents/openai.yaml`: Codex UI 元数据

## 设计原则

这个 skill 优先做 infrastructure-only 优化：测 baseline，定位瓶颈，做最小高收益改动，再用吞吐、step time、GPU utilization、显存和 loss/metric proxy 验证。默认不改变数据、loss、optimizer、effective batch size、评估含义或 checkpoint 格式。

## 参考和整合

整理时参考了公开的 training optimization / PyTorch engineering router、Flash Attention 专项 skill，也进一步吸收了 TorchTitan、DeepSpeed/ZeRO、Megatron-Core/NeMo、Liger Kernel、Mosaic Streaming、WebDataset、NVIDIA DALI 和 PyTorch 官方 profiler、DataLoader、FSDP/compile、Distributed Checkpoint 文档。整合重点是诊断优先、按瓶颈路由、对 attention/backend/kernel 做形状和正确性验证，并把 checkpoint、logging、dataloader 这类真实训练里的 stall 纳入 profiling。
