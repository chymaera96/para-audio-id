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

## Overnight mechanisms: F/G/H/I

Each arm independently starts from original tc21, shares D/E's sealed old replay
cache, and retains the base causal audio, ID16, and boundary objectives. New crops
remain online. No distillation is added to these arms.

| Arm | Mechanism | LR | Sampling |
| --- | --- | --- | --- |
| F | Only MLPs in final four blocks train; all other weights frozen | source ~1.5e-5 | 40 new / 40 old per update |
| G | Project actual Adam parameter displacement against three old-loss gradients | source ~1.5e-5 | 40 new / 40 old, plus references |
| H | Five cycles: 1K newcomer-only updates then 1K old-only updates | source ~1.5e-5 | 80 selections/update from current phase |
| I | Frozen original base + rank32/alpha32 LoRA on every attention/MLP Conv1D projection | 1e-4 | 40 new / 40 old per update |

At 10K updates all arms have 400K main newcomer and 400K main old selections.
G additionally reads 16 old documents per clean/noise/noise+RIR condition per
update (480K reference document draws). These extra references and their cost
must not be hidden in exposure comparisons. The ID16/EOS1 reference loss uses
ground-truth targets, no audio loss and no validation queries. Reference gradients
are evaluated without dropout. The candidate main update still uses the complete
base objective, original clipping, and AdamW. Three normalized halfspaces constrain
the ACTUAL post-Adam displacement; raw-gradient projection would not do this.
An eight-active-set small QP projects in Euclidean parameter space. Adam moments
advance unprojected. Non-finite references fail; infeasible/numerically unreliable
projection reverts the parameter displacement to zero, logged as a fallback.
Actual first-order residuals are checked after parameter rounding (tolerance1e-7).
This is a sampled, first-order constraint, not a global retrieval guarantee.

F's attention and embeddings remain frozen; gradients can still flow through
frozen upper layers to the trainable MLPs. I adds no new vocabulary or task router.
Freezing the base does not guarantee old predictions survive adapter updates.
I exports merged ordinary inference weights AND separate adapters. Resume loads
the original base and restores adapters, never adds them again to merged weights.

Prepare one 1K-old/1K-new fresh-crop evaluation pack (6K queries):

```bash
python -u continual_index.py prepare-fresh "$PILOT" "$PILOT/fresh-1k-v1"
python -u continual_index.py train "$PILOT" --arm G --wandb-online \
  --extra-eval "$PILOT/fresh-1k-v1"
```

Repeat independently for F/H/I (priority G,F,H,I), each 10K updates. W&B project
`para-audio-id`, group `tc21-add1k-mechanisms-v1`, stable IDs `tc21-add1k-f/g/h/i`.
Run contracts pin trainable counts, cache, reference settings, phase length, rank,
LR and fresh pack. F-I use a separate DataLoader generator so creating an iterator
on resume does not consume model RNG. H resumes at its exact phase/update index.
Each phase boundary is evaluated with the existing 600-query probe; checkpoints
retain 1K/2.5K/5K/10K milestones as before, plus every H phase boundary (including
the final newcomer-write state at 9K, before the last replay/repair phase).
The fresh pack is evaluated at endpoint,
saved as `fresh-final.json` and logged as `fresh/...`.

Fresh crops exclude exact starts in every cached old replay observation and the
original probe, and use held-out degradation assets. Their starts are additionally
excluded from F-I's online crop draws, including replacement identities. The pack
has its own source-model baseline, hashes and audit. It is not used to mine replay,
pick gradient references, or construct training targets. Completed comparison
models can be evaluated with:

```bash
python -u continual_index.py evaluate "$PILOT" /path/to/checkpoint.ckpt \
  --fresh "$PILOT/fresh-1k-v1"
```

This writes a separate `.fresh.json`, without overwriting the original probe.
