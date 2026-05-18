# Checkpoint Goodput

Use this when periodic save/eval/logging stalls hide inside step time. Ask for confirmation before changing checkpoint cadence, format, or async behavior.

## Concepts

- Raw throughput: training work per second during ordinary steps.
- End-to-end throughput: training work per wall-clock second including checkpoint/eval/logging stalls.
- Goodput: useful training progress per wall-clock second after excluding failed/restarted work and including safety overhead.

A run can have fast ordinary steps but poor goodput if checkpoint stalls are long or resume is unreliable.

## What To Measure

- Checkpoint blocking time per save
- Save frequency
- Bytes written
- Rank participation
- Whether optimizer state is sharded or gathered
- Resume time and resume correctness
- Lost work risk if cadence is reduced

Use `scripts/checkpoint_goodput.py` to compute simple raw-vs-goodput estimates from timing JSON.

## Fix Options

| Fix | Upside | Risk |
| --- | --- | --- |
| Rank-zero-only side effects | Less duplicated work | Wrong for sharded states if used blindly |
| Distributed checkpointing | Avoids full state gather | Format/resume complexity |
| Async checkpointing | Reduces GPU blocking time | Staging memory, background I/O pressure |
| Lower checkpoint cadence | Less overhead | More lost work on failure |
| Smaller checkpoint contents | Less I/O | Missing state needed for exact resume |

## Guardrails

- Do not simply checkpoint less often unless the user accepts failure-recovery risk.
- Always test resume after changing format or cadence.
- For sharded training, use a checkpoint path compatible with the distributed strategy.
- Report both ordinary step time and end-to-end goodput.
