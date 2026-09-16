# Short medium 100K continuation on scale

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
