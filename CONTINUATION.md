# Medium 100K continuations on scale

## Identification-only continuation (new run)

Use the original pre-continuation checkpoint, not the regressed `-cont` run.
Prepare once on CPU (no GPU allocation required; allow enough RAM to load and
copy the full optimizer checkpoint):

```bash
python prepare_continuation.py \
  /gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/tc18-medium-100k/last.ckpt \
  --run-id tc18-medium-100k-idonly \
  --additional-steps 200000 --id-only --keep-saved-lr
```

Audio prediction remains a reported metric but is removed from the training
objective. ID and boundary/EOS supervision on clean and degraded views retain
their previous coefficients: `(160 * ID + 2 * boundary) / 562` for the eight-codebook
profile. Audio input tokens are unchanged. The saved LR is held constant without
reheating. Optimizer moments, augmentation, batching and distillation settings
are preserved (the source experiment uses zero distillation weight).

Launch in the para-audio-id environment on four GPUs:

```bash
CONT_ROOT=/gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/tc18-medium-100k-idonly
python train.py "$CONT_ROOT/continuation.yaml" \
  --run-id tc18-medium-100k-idonly \
  --resume --ckpt-path "$CONT_ROOT/continuation-start.ckpt" \
  --devices 4 --wandb-online
```

After a new periodic checkpoint has been saved, subsequent restarts use:

```bash
python train.py "$CONT_ROOT/continuation.yaml" \
  --run-id tc18-medium-100k-idonly --resume --devices 4 --wandb-online
```

Do not prepare again or restart from `continuation-start.ckpt` after progress
has been saved. If interrupted before the first periodic checkpoint, use the
initial launch command. The original checkpoint is never modified.

## Earlier ramp-and-hold continuation (historical)

Prepare once, before launching the four-GPU job:

```bash
python prepare_continuation.py \
  /gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/tc18-medium-100k/last.ckpt \
  --run-id tc18-medium-100k-cont
```

This reads the original checkpoint and writes a separate full-state starting
checkpoint plus `continuation.yaml` in the new run directory. Existing run
directories are rejected. The original file is never written.

Launch on four GPUs:

```bash
ROOT=/gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/tc18-medium-100k-cont
python train.py "$ROOT/continuation.yaml" \
  --run-id tc18-medium-100k-cont \
  --resume --ckpt-path "$ROOT/continuation-start.ckpt" \
  --wandb-online
```

Defaults add 25,000 optimizer updates to the saved global step. The saved LR
ramps linearly to 5e-5 over 2,000 updates, then stays constant. Change the pilot
budget with `--additional-steps`, or use `--target-lr` and `--ramp-steps` at
preparation time. The source optimizer moments, model, loop/sampler state,
RNG, loss, distillation weight, augmentation boundaries and recipes are retained.
The physical layout remains four GPUs, 16 tracks per GPU, accumulation one.
Global steps continue from the source; W&B uses the new run ID. Checkpoint and
monitor intervals remain 10K and 5K respectively.

After interruption, resume the new run's latest saved checkpoint (do not repeat
the preparation command or use the starting checkpoint again):

```bash
python train.py "$ROOT/continuation.yaml" \
  --run-id tc18-medium-100k-cont --resume --wandb-online
```

If interrupted before the first new periodic checkpoint, restart from
`continuation-start.ckpt` using the launch command. Source provenance and the
continuation LR policy are embedded in every new checkpoint.
