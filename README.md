# Parametric audio language model

This repository tests whether a decoder-only causal language model can compile a
fixed 100,000-track catalogue into its parameters. Each training document is:

```text
[BOS] audio-token-1 ... audio-token-N [ID] digit-1 ... digit-5 [EOS]
```

Audio tokens are the first two Mel-RVQ codebooks from the frozen
`OpenMuQ/MuQ-large-msd-iter` checkpoint. The causal LM jointly predicts the audio
sequence and the track's arbitrary five-digit code. Identification performs model
generation only: it does not search fingerprints, embeddings, an ANN index, token
shards, a valid-code list, or training audio.

This is LLM-style memorisation through repeated causal continuations. Every clean
five-second segment from a track is a separate document with the same identifier,
and each document contains that identifier only once. Parametric indexing methods
such as DSI are related work, but this system does not use staged DSI training.

The previous continuous MuQ encoder and cross-attention decoder are intentionally
not supported on this branch.

## Installation and catalogue

Install in an isolated environment with matching PyTorch and Torchaudio builds:

```bash
export PYTHONNOUSERSITE=1
python -m pip install -e '.[dev]'
```

Build the existing deterministic 100,000-track/code catalogue if it is not already
present:

```bash
python prepare_catalogue.py \
  /gpfs/scratch/acw723/fma_data/fma_large \
  data/fma_large_100k.jsonl \
  --bad-files data/catalogue_bad_files.jsonl
```

## Mandatory MuQ probe

Do not tokenize the catalogue until this command succeeds on the cluster:

```bash
python probe_muq_tokenizer.py \
  configs/fma_large.yaml \
  /gpfs/scratch/acw723/fma_data/fma_large/000/000002.mp3
```

The probe resolves and records an immutable Hugging Face revision, checks that the
checkpoint contains eight 1,024-entry Mel-RVQ codebooks, proves MuQ's public target
layout is block-major, verifies deterministic extraction, converts the first two
codebooks to time-major tokens, and confirms the complete causal document fits the
512-token context.

Failure is a hard blocker. The pipeline does not substitute rounded continuous
features or another codec.

## Offline catalogue tokenization

After the probe passes:

```bash
python tokenize_catalogue.py configs/fma_large.yaml
```

The default output is:

```text
/gpfs/scratch/acw723/para-audio-id/audio_lm_tokens
```

The corpus contains six canonical five-second documents per catalogue track.
Tokens are stored as atomic, memory-mappable `uint16` shards. Each shard has a
JSONL span index and metadata containing the resolved tokenizer specification and
fingerprint. Reruns skip complete compatible shards, rebuild incomplete shards,
and refuse incompatible cache contents. `tokenization_report.json` accounts for
every intended document and every explicit failure.

Token shards are training artifacts and are never loaded by identification.

For paired-shift training, first copy the exact cohort from the completed
canonical 10K run:

```bash
cp logs/fma-large-audio-lm/<canonical-10k-run>/training_tracks.json \
  data/training_tracks_10k.json
```

Then prepare the two new, role-separated stores:

```bash
python prepare_paired_tokens.py configs/fma_large.yaml
```

This reuses the canonical cache and creates:

```text
/gpfs/scratch/acw723/para-audio-id/audio_lm_tokens_shifted_training_10k
/gpfs/scratch/acw723/para-audio-id/audio_lm_tokens_heldout_evaluation_10k
```

The shifted-training store contains standalone crops at integer noncanonical
starts. The evaluation-only store contains half-offset crops and is rejected by
the training sampler. Both stores record their role, view policy, exact cohort,
tokenizer fingerprint, source duration, and any zero-padding applied to short
recordings. Preparation resumes at complete shard boundaries.

## Training

The default causal LM is a randomly initialized 12-layer GPT-2-style decoder with
hidden size 768, 12 heads, tied embeddings, and no dropout. The vocabulary has
2,048 collision-free audio tokens, `[BOS]`, `[ID]`, ten dedicated digit tokens,
and `[EOS]`.

Training jointly predicts the complete document from the first optimizer step.
It uses one token-level causal mean, with every identifier digit assigned a fixed
weight of 20:

```text
loss = (
  sum(audio token losses)
  + 20 * sum(identifier digit losses)
  + sum([ID] and [EOS] losses)
) / (number of audio tokens + 20 * number of digits + 2)
```

Audio and identifier losses are also logged separately for diagnosis, but neither
is optimized as a separately normalized objective and there is no objective
schedule.
The current experiment reuses the exact 10,000 identities from the canonical run.
In `paired` mode, each track contributes one canonical and one shifted document;
their independently normalized causal losses are averaged equally. Set
`data.view_mode: canonical_only` for the from-scratch ablation, which instead
uses two distinct canonical views with identical batch shape and compute.
Identifier digit targets have a fixed loss weight of 20.
The single-GPU logical batch is 32 tracks × 2 segments: a physical microbatch of
four tracks × two segments with eight gradient-accumulation steps.
Training stops after 100,000 optimizer steps and warms up for exactly 200.
Validation, rotating canonical/held-out greedy probes, and checkpointing run
every 500 steps. Beam probes run every 2,500 steps and at the final step.
Catalogue passes reshuffle identities and advance deterministic per-track view
permutations, but do not control stopping or checkpoint cadence.

```bash
python train.py configs/fma_large.yaml \
  --devices 1 \
  --run-id paired-10k-100k \
  --wandb-online
```

Resume the exact saved model, optimizer, scheduler, loop, sampler pass, and RNG
state:

```bash
python train.py configs/fma_large.yaml \
  --devices 1 \
  --run-id paired-10k-100k \
  --wandb-online \
  --resume
```

Checkpoints are written every 500 optimizer steps under:

```text
/gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/<run-id>/
```

Checkpoints embed the vocabulary, exact cohort, view grids, corpus fingerprint,
tokenizer fingerprint, and code mapping. A canonical-only legacy checkpoint
cannot initialize the paired experiment; interruption resume is accepted only
when all paired-corpus metadata matches.

PyTorch deterministic algorithms and seeded Python, NumPy, Torch, samplers, and
workers are enabled. `deterministic_warn_only: true` reports CUDA operations for
which PyTorch cannot promise bitwise determinism instead of aborting a long run.

## Evaluation and inference

Post-run evaluation reads all six canonical and all five held-out cached positions:

```bash
python evaluate.py \
  /gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/paired-10k-100k/last.ckpt \
  evaluation-paired-10k.json \
  --cohort training \
  --expected-tracks 10000
```

It reports greedy exact accuracy, beam Top-1/5/10, MRR, and protocol validity
separately for canonical views, held-out views, and every individual start.
Batched generation receives only `[BOS] audio [ID]` and produces five digits plus
`[EOS]`; it uses no teacher forcing or catalogue-derived valid-code constraint.

Identification requires only the causal-LM checkpoint, frozen MuQ weights, and
query audio:

```bash
python identify.py \
  /gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/fma-large-audio-lm/last.ckpt \
  query.wav
```

The five-digit exact random baseline is `1e-5`. A functioning pipeline is not
evidence that catalogue acquisition succeeded.

To evaluate the exact 1,000-track training cohort from the preceding run on
shifted clean excerpts at 2.5, 12.5, and 22.5 seconds, with all five identifier
digits generated autoregressively and no teacher-forced scoring:

```bash
python evaluate.py \
  /gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/tc-1k-id20-20k/last.ckpt \
  evaluation-1k-shifted-greedy.json \
  --cohort training \
  --expected-tracks 1000 \
  --greedy-only
```

The expected-track guard prevents accidentally running this test against the
10,000-track checkpoint. The output records the checkpoint cohort, shifted
positions, evaluated/skipped track counts, protocol validity, per-query generated
codes, and free-running greedy exact accuracy.

## Tests

```bash
pytest
ruff check .
```

The default suite uses synthetic tokens for mathematical and storage checks. The
real MuQ integration test is opt-in:

```bash
RUN_MUQ_INTEGRATION=1 MUQ_DEVICE=cuda \
  pytest -m integration tests/test_muq_integration.py -v
```
