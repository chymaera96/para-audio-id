# tc21: 25K -> 26K continual-indexing pilot

This is independent of `train.py`. It warm-starts the completed tc21 model,
keeps fsq-tc8 frozen, and adds 1,000 audited recordings from outside the source
catalogue, preserving all existing five-digit identifiers. The source checkpoint
is read-only and SHA256-pinned. Each arm has a new update clock, fresh AdamW
moments, constant saved source LR (approximately 1.5e-5), 80 recording selections
per update, clean next-token prediction, identifier weight 16 and boundary loss.
The source's final acoustic distribution applies from update zero, including
the complete training RIR pool. No ranking or distillation objective is added.

| Arm | New/old selections per update | Old replay pool |
| --- | --- | --- |
| A | 80 / 0 | None |
| B | 40 / 40 | All 25,000 original recordings |
| C | 40 / 40 | Fixed seeded subset of 1,000 original recordings |

B and C use identical newcomer draws and augmentation keys at matched updates.
Unreadable training audio is replaced only within its own new/old pool; all
fallbacks are audited. This first pilot uses fresh **audio replay**, not a finite
token replay buffer. Each selected recording still supplies a clean anchor and
secondary observation. All arms target the expanded 26K catalogue at inference,
without task labels, catalogue constraints, or an external retrieval index.

## Prepare once on an allocated GPU

```bash
ROOT=/gpfs/scratch/acw723/para-audio-id
python -u continual_index.py prepare \
  "$ROOT/audio-lm-checkpoints/tc21/last.ckpt" \
  "$ROOT/continual-index/tc21-add1k-v1"
```

Preparation creates an immutable shared manifest, audits newcomers, and chooses
100 old and 100 new monitoring identities with one fixed two-second crop each.
Old monitoring identities are outside C's replay set. All three conditions
(clean, 0dB noise, 0dB noise+RIR) use held-out degradation assets. Their 600 token
sequences and source-model predictions are cached once and shared across arms.
Exact validation starts are excluded from update training. Invalid monitoring
candidates are replaced deterministically and recorded; final counts are exact.
These small probes screen trends, not one-percentage-point significance.

## Train separate runs

```bash
python -u continual_index.py train \
  /gpfs/scratch/acw723/para-audio-id/continual-index/tc21-add1k-v1 \
  --arm A --wandb-online
```

Repeat with `--arm B` and `--arm C`. Defaults are 10K updates, validation at
zero/every 1K updates plus 2.5K/5K/10K milestones, and eight loading workers.
W&B project is `para-audio-id`; group is `tc21-add1k-v1`; stable run IDs are
`tc21-add1k-a`, `tc21-add1k-b`, `tc21-add1k-c`. Only six old/new Top-1 curves,
one forgetting curve, three losses and three update-cost curves are logged.
Full per-query predictions, cross-cohort confusions, replacement audits,
exposure counters, and timing live in the run directory, not additional W&B
plots. No W&B checkpoint artifacts are uploaded.

Checkpoints are under `audio-lm-checkpoints/<run-id>/`: a replaceable full-state
`last.ckpt` and milestones at 1K/2.5K/5K/10K. Milestones initially hard-link the
same file; later atomic replacement of `last.ckpt` preserves them. Resume with
the original command plus `--resume`. It restores weights, optimizer, scaler,
RNG and exact update-indexed sample stream. The run contract must match. For a
short check, `--steps 100 --stop-after 50` followed by `--steps 100 --resume`
tests continuation without changing the endpoint. Sanity runs omit W&B and
use separate IDs such as `tc21-add1k-a-check`.

```bash
python -u continual_index.py evaluate \
  /gpfs/scratch/acw723/para-audio-id/continual-index/tc21-add1k-v1 \
  /gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/tc21-add1k-a/last.ckpt
```

The dedicated evaluator uses the unchanged old/new query pack. Ordinary 25K
evaluation/resume checks are deliberately not weakened to accommodate this
expanded-catalogue protocol. A receives twice B/C's newcomer exposures at equal
updates; compare both GPU-hours and new-recording exposures. Preparation cost
is shared but should be reported separately from per-arm update cost.
