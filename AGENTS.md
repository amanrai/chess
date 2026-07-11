# Agent Guidelines

## Trainer checkpointing

All training scripts must support periodic in-epoch checkpointing by default.

Required behavior for every trainer:

- provide a `--checkpoint-dir` option
- provide a `--snapshot-every-batches` option
- default `--snapshot-every-batches` to a nonzero value, currently `5000`
- allow disabling snapshots with `--snapshot-every-batches 0`
- save an end-of-epoch checkpoint
- include at least these fields in checkpoints:
  - `model`
  - `optimizer` when an optimizer exists
  - `args`
  - `epoch`
  - `batch`
  - `global_batch` when batch training is used

Checkpoint filenames should be explicit and sortable, e.g.:

```text
<trainer>_epoch_001_batch_005000.pt
<trainer>_epoch_1.pt
```

Do not add or modify a trainer without preserving this behavior.
