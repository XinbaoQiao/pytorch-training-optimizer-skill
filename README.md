# PyTorch Training Optimizer Skill

中文 | [English](README.en.md)

## 简介

PyTorch Training Optimizer Skill 是一个面向研究训练代码的 Codex skill，用来诊断和优化 PyTorch 训练吞吐、GPU 利用率、显存效率、输入管线和多卡扩展效率。

它适合处理这些训练系统问题：

- 定位 GPU 利用率低、step time 高、吞吐不稳定或多卡扩展效率差的原因
- 评估混合精度、`torch.compile`、attention backend、kernel fusion 等计算侧优化
- 选择 DDP、FSDP/FSDP2、ZeRO、activation checkpointing 等分布式和显存策略
- 优化 dataloader、解码、预取、host-to-device transfer 和存储访问带来的输入瓶颈
- 减少训练循环里的同步、日志、评估、checkpoint、可视化等热路径开销
- 用 before/after 指标验证优化是否真正提升速度，同时保持 loss、metric、batch size、数据顺序和 checkpoint 语义可比

## 重要约束：改代码前必须确认

这个 skill 默认只先给诊断和改动建议。真正改训练代码、配置、precision policy、compiler 设置、attention backend、dataloader、checkpoint 或分布式策略之前，必须先让用户确认。

确认请求需要通俗说明：

- 准备改什么文件或命令
- 为什么这可能变快
- 预计哪个指标会变好
- 可能出现的坏结果，例如 loss 变差、显存升高、compile 首次启动变慢、checkpoint 不兼容、依赖安装失败、复现性下降等
- 如何回滚
- 如何验证 before/after

用户明确选择要改哪些项之后，才执行对应改动。

## 下载和安装

从 GitHub 下载：

```bash
git clone https://github.com/XinbaoQiao/pytorch-training-optimizer-skill.git
```

安装到用户级 Codex skill 目录：

```bash
mkdir -p "$HOME/.agents/skills"
cp -R pytorch-training-optimizer-skill "$HOME/.agents/skills/pytorch-training-optimizer"
```

也可以安装到某个仓库内，让团队共享这个 skill：

```bash
mkdir -p "$REPO_ROOT/.agents/skills"
cp -R pytorch-training-optimizer-skill "$REPO_ROOT/.agents/skills/pytorch-training-optimizer"
```

管理员级安装路径：

```bash
sudo mkdir -p /etc/codex/skills
sudo cp -R pytorch-training-optimizer-skill /etc/codex/skills/pytorch-training-optimizer
```

较旧的 Codex 设置可能仍使用：

```bash
${CODEX_HOME:-$HOME/.codex}/skills/pytorch-training-optimizer
```

如果更新后没有出现，重启 Codex。

## 使用

在 Codex 中直接调用：

```text
Use $pytorch-training-optimizer to profile and optimize this PyTorch training script.
```

或者给出更具体的目标：

```text
Use $pytorch-training-optimizer to find why my 8xH200 training run has low GPU utilization and propose safe speedups. Ask before editing code.
```

## 包含内容

- `SKILL.md`: skill 主 workflow、触发描述和改代码前确认约束
- `agents/openai.yaml`: Codex UI 元数据
- `scripts/collect_env.py`: 收集 PyTorch/CUDA/NCCL/GPU/环境变量信息
- `scripts/step_timer.py`: 统计 step time 的 mean/p50/p95/std 等指标
- `scripts/profiler_trace.py`: 生成 PyTorch profiler 集成模板
- `scripts/compare_runs.py`: 对比 before/after run record，并检查关键实验条件是否变化
- `scripts/compile_probe.py`: 检查 `torch.compile` 可用性并运行小型 compile benchmark
- `scripts/sdpa_probe.py`: 检查 SDPA/attention backend 可用性并运行小型 attention benchmark
- `scripts/dataloader_sweep.py`: sweep `num_workers`、`prefetch_factor`、`pin_memory`、`persistent_workers`
- `scripts/checkpoint_goodput.py`: 区分 raw throughput 和 checkpoint stall 影响后的 goodput
- `references/`: profiling、compiler、attention、precision、distributed、dataloader、checkpoint、failure modes 等参考文档
- `assets/`: run record schema 和 comparison report 模板
- `tests/`: skill metadata、reference 文件和脚本编译 smoke tests

## 设计原则

这个 skill 优先做 infrastructure-only 优化：测 baseline，定位瓶颈，给出可理解的改动建议，等用户确认后做最小高收益改动，再用吞吐、step time、GPU utilization、显存和 loss/metric proxy 验证。默认不改变数据、loss、optimizer、effective batch size、评估含义或 checkpoint 格式。

## 参考和整合

整理时参考了公开的 training optimization / PyTorch engineering router、Flash Attention 专项 skill，也进一步吸收了 TorchTitan、DeepSpeed/ZeRO、Megatron-Core/NeMo、Liger Kernel、Mosaic Streaming、WebDataset、NVIDIA DALI 和 PyTorch 官方 profiler、DataLoader、FSDP/compile、Distributed Checkpoint 文档。整合重点是诊断优先、按瓶颈路由、对 attention/backend/kernel 做形状和正确性验证，并把 checkpoint、logging、dataloader 这类真实训练里的 stall 纳入 profiling。

## Contributors

- [XinbaoQiao](https://github.com/XinbaoQiao): 项目维护者
- OpenAI Codex: 协助整理、编写和更新 skill 内容
