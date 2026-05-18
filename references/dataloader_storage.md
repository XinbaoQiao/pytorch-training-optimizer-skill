# Dataloader and Storage Optimization

Use this when the trace shows CPU, dataloader, decode, storage, or host-to-device transfer stalls. Ask for confirmation before editing data code or storage format.

## First Sweep

Measure before changing data semantics:

- `num_workers`: 0, 2, 4, 8, then stop when improvement saturates
- `pin_memory`: off/on
- `persistent_workers`: off/on for `num_workers > 0`
- `prefetch_factor`: 2, 4, 8 for `num_workers > 0`
- Host-to-device `.to(device, non_blocking=True)`
- CPU thread variables such as `OMP_NUM_THREADS` and `MKL_NUM_THREADS`

Use `scripts/dataloader_sweep.py` if the project can provide a dataset factory.

## Signs of Specific Bottlenecks

| Signal | Likely cause | Candidate fix |
| --- | --- | --- |
| GPU waits before each batch | Dataloader/CPU bottleneck | Worker/prefetch/pin sweep, move transforms out of loop |
| CPU saturated | Decode or augmentation | Offline preprocessing, DALI, vectorized transforms |
| Storage read latency high | Random I/O or network FS | Local NVMe cache, shard format, streaming dataset |
| RAM grows with workers | Dataset object copied per worker | Use compact metadata, mmap/Arrow, lazy open files |
| H2D copies visible and blocking | Transfer bottleneck | Pinned memory + non-blocking transfer + overlap |
| Startup/resume slow | Cache fill or shard discovery | Persistent workers, checkpointable dataloader, manifest cache |

## Storage Format Changes

Recommend storage rewrites only after evidence shows file access is the bottleneck.

| Format/tool | Use when | Risks |
| --- | --- | --- |
| WebDataset/tar shards | Many small files or remote sequential streaming | Shuffle buffer and shard splitting must preserve training semantics |
| mmap/Arrow | Large tabular/text metadata and random access | Requires conversion and index compatibility |
| LMDB | Small random reads with local storage | Writer/reader complexity and file locking |
| Mosaic Streaming | Cloud/object store, deterministic resume, large datasets | Dependency and cache behavior |
| DALI | CPU decode/augmentation bottleneck | Operator coverage, GPU memory, augmentation parity |

## Worker Memory Guardrail

Increasing workers can multiply Python object memory. If the dataset stores a large list of filenames, annotations, or token arrays in Python objects, each worker may replicate it. Prefer compact arrays, memory maps, or lazy manifests when worker RAM grows too much.

## Correctness Checks

- Distributed sampler still shards correctly.
- Shuffle quality and sample order are intentionally preserved or intentionally changed.
- Augmentation distribution is unchanged unless approved.
- Resume behavior is unchanged unless approved.
- Effective batch size and drop-last behavior are unchanged.
