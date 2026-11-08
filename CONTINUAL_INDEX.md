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

## Fixed replay and preservation: D/E

The completed A/B/C pilot motivated broad identity coverage and protection of
degraded old decisions. D/E start again from the original tc21, not A/B/C.
Both use 40 new + 40 old selections per update, identical draws and a shared
immutable replay cache. New observations remain fresh; only old replay is cached.
The existing clean causal audio loss, ID weight 16, boundary loss, constant
source LR, fresh AdamW moments, and saturated augmentation are unchanged.

```bash
PILOT=/gpfs/scratch/acw723/para-audio-id/continual-index/tc21-add1k-v1
python -u continual_index.py prepare-replay "$PILOT" "$PILOT/replay-cache-v1"
python -u continual_index.py train "$PILOT" --arm D --wandb-online
python -u continual_index.py train "$PILOT" --arm E --wandb-online
```

Preparation visits all 25K old recordings four times, with independent training
crop/degradation keys and the original validation-start exclusions. Shards are
atomic and resumable. Existing corrupt-audio replacements remain audited; the
sealed manifest reports actual identity coverage, uncovered IDs, pair counts,
category counts, preparation time, physical bytes, and source-teacher accuracy.
More than 1% uncovered identities fails preparation. D/E sample the same covered
old pool; they never silently substitute fresh old audio for absent cache entries.
The cache and all teacher targets are SHA256-bound to both run contracts.

D uses ground-truth supervision only. E adds source-teacher preservation on old
documents whose original width-10, EOS-scored retrieval is correct. All replay
documents still receive ground-truth supervision; newcomers never receive KD.
The source model is FP32/eval during preparation and is not run during training.
At each of five teacher-forced ID positions and EOS, cache log probabilities for
ten digits, EOS, and the aggregate remaining-vocabulary mass. This grouped KL
retains the full-softmax normalizer but is explicitly not a full-vocabulary KL
or a guarantee of preserving free-running beam rankings.

The default E multiplier is 1.0 (`--preservation-weight`). Per-document KL weights
are `[16,16,16,16,16,1]`, averaged over all batch documents with zero contribution
from new or teacher-incorrect documents, then scaled by 81/282 (the existing
ID-plus-EOS target-family coefficient for 200 audio tokens). Base losses are not
renormalized or reduced. E logs raw preservation loss and eligible fraction.
Both runs use 10K updates and stable W&B IDs `tc21-add1k-d` / `tc21-add1k-e`,
project `para-audio-id`, group `tc21-add1k-replay-v1`. Resume with `--resume`;
changing replay content or preservation weight on resume is rejected.

The raw audio-token payload is about 80 MB; compressed teacher targets add about
29 MB, plus metadata and shard overhead. Actual storage is recorded, not inferred
from these estimates. The tokenizer is frozen; its online FP32 policy remains
identical to the source pilot. Held-out validation is never used to select replay
observations or teacher targets. Use larger fresh-crop evaluation before making
small-difference claims, then test a second disjoint newcomer cohort.
