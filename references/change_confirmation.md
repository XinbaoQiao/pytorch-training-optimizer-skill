# Change Confirmation Gate

Use this before editing training code, launch configs, precision settings, compiler settings, attention backends, dataloaders, checkpoint logic, or distributed strategy.

## Rule

A diagnosis or optimization suggestion is not permission to edit code. Stop before code changes and ask the user to approve a specific change set.

A current-turn instruction that explicitly approves a specific change set counts as approval. Examples:

- Approved: "Apply the dataloader and logging changes only."
- Approved: "Change precision to BF16 and add the timer harness."
- Not approved: "Optimize this training script."
- Not approved: "Find speedups."
- Not approved: "Make it faster".

## Confirmation Request Template

```text
I found [likely bottleneck]. I recommend [change set].

Plain-language expected benefit:
- [Why this should make training faster]
- Expected metric movement: [step time / tokens/sec / GPU utilization / memory / checkpoint stall]

Possible bad outcomes:
- [Numerical/convergence risk]
- [Memory/dependency/checkpoint/reproducibility risk]
- [When this may be slower]

Files or commands I would change:
- [path or command]

Rollback:
- [How to undo]

Validation:
- [Before/after command]
- [Correctness proxy]

Please confirm which option to apply:
A. [lowest-risk change]
B. [higher-impact change]
C. [instrumentation only]
D. Do not change code yet
```

## Plain-Language Risk Translation

| Technical risk | Explain it like this |
| --- | --- |
| BF16/FP16 instability | The run may become faster but the loss curve can shift or produce NaNs if the model is numerically sensitive. |
| TF32 change | Matrix multiplications may be faster, but exact FP32 reproducibility changes slightly. |
| `torch.compile` cold start | The first iterations may become slower while PyTorch compiles graphs; only steady-state speed matters. |
| Graph breaks/recompiles | Compile may not help if the code changes shape/control flow often. |
| Flash/SDPA backend change | Attention can become faster, but masks/dropout/dtypes may behave differently or fall back silently. |
| Dataloader worker increase | GPUs may wait less, but CPU RAM and file-system pressure may increase. |
| Activation checkpointing | Memory may drop, but each step can get slower because some activations are recomputed. |
| FSDP/ZeRO change | Larger models or batches may fit, but checkpoint format and optimizer state handling can change. |
| Async checkpointing | GPU stalls may shrink, but staging memory and recovery logic must be verified. |

## Allowed Without Confirmation

These actions are normally safe before approval:

- Reading files
- Inspecting configs
- Running read-only tests or profiling commands that do not modify the repository
- Creating a proposal
- Adding notes in the response

These actions require confirmation:

- Editing source files or configs
- Adding/removing dependencies
- Changing launch scripts
- Changing data loading, precision, compile, checkpoint, or distributed behavior
- Changing metrics, evaluation, optimizer, scheduler, loss, or model architecture
