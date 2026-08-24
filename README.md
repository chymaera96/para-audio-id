# Parametric audio language model

This repository tests whether a decoder-only causal language model can compile a
fixed 100,000-track catalogue into its parameters. Each training document is:

```text
[BOS] audio-token-1 ... audio-token-N [ID] digit-1 ... digit-5 [EOS]
```

Audio tokens use all eight Mel-RVQ codebooks from the frozen
`OpenMuQ/MuQ-large-msd-iter` checkpoint. The causal LM jointly predicts the audio
sequence and the track's arbitrary five-digit code. Identification performs model
generation only: it does not search fingerprints, embeddings, an ANN index, token
shards, a valid-code list, or training audio.

This is LLM-style memorisation through repeated causal continuations. Every
online two-second crop from a track is a separate document with the same
identifier, and each document contains that identifier only once. Parametric indexing methods
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
layout is block-major, verifies deterministic extraction, converts the selected
codebooks to time-major tokens, and confirms the complete causal document fits the
512-token context.

Failure is a hard blocker. The pipeline does not substitute rounded continuous
features or another codec.

## Historical offline tokenization utilities

The repository still contains the older cache-preparation commands, but current training
does not use canonical, shifted, or half-offset token stores for training or
evaluation. They remain useful only when reproducing tc2–tc6 from Git history.

tc18 uses the existing 25K identity manifest. Prepare it if it is not already
available:

```bash
python prepare_training_cohort.py configs/fma_large.yaml
# Writes or validates data/training_tracks_25k.json.
```

## Training

The default causal LM is a randomly initialized 12-layer GPT-2-style decoder with
hidden size 768, 12 heads, tied embeddings, and no dropout. The vocabulary has
8,192 codebook-separated audio tokens, `[BOS]`, `[ID]`, ten dedicated digit tokens,
and `[EOS]`.

Training samples online two-second crops from the fixed 25K cohort. Each
identity contributes a clean anchor and one secondary view: a distinct clean
crop, an exact same-crop background-noise view, an exact same-crop room-reverb
view, or the combined noise-then-reverb view. One frozen lightweight MuQ call
tokenizes all 80 final waveforms in each physical microbatch.

```text
loss =
    (
        400 * mean(clean audio-token CE)
        + 160 * mean(all digit CE)
        + 2 * mean(all boundary CE)
    ) / 562
    + lambda_KD * identifier-logit distillation
```

For every degraded secondary, tc18 uses its clean anchor as a stop-gradient
teacher at the five causal positions that predict the next identifier digit.
The KL divergence is computed only over the ten digit-token logits with
temperature 2. This uses the existing single decoder pass and adds no model
parameters or inference-time modules. The causal coefficients are approximately
0.709 audio, 0.284 digits, and 0.007 boundaries with identifier
weight 32. Checkpoints record `tc18_two_second_eight_codebook_logit_distillation_v1`.

Eight codebooks produce 400 audio tokens and a 408-token complete causal
document. The existing 512-position table is retained: the inference prompt is
402 tokens before the five generated digits, so no context expansion is needed.

The secondary-view curriculum remains identical to tc15–tc17:

| Raw steps | Clean | Noise | RIR | Noise + RIR | RIR severity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0–10K | 1.00 | 0 | 0 | 0 | disabled |
| 10–30K | 1.00 → 0.40 | 0 → 0.30 | 0 → 0.30 | 0 | mild → moderate |
| 30–60K | 0.40 → 0.10 | 0.30 → 0.35 | 0.30 | 0 → 0.25 | expand to full range |
| 60–225K | 0.10 | 0.35 | 0.30 | 0.25 | full range |

The distillation weight is zero through 15K, ramps to its configured maximum
over 15K–30K, and then remains fixed. Its maximum defaults to 0.1 and can be
set to zero for the matched no-distillation ablation.
RIR severity is ranked by post-peak 99%-energy decay duration. The eligible
training pool expands from its mildest third, through two thirds at 30K, to all
IRs at 60K. Convolution remains full-wet at every severity.

Noisy examples use fixed SNR-bin probabilities `0.40/0.30/0.20/0.10` for
`0–5/5–10/10–20/20–30 dB`; 10% of noisy views are exactly 0 dB. The LR uses
the same schedule: linear warm-up over 500 steps to `3e-4`, hold through 60K,
linear decay to `1.5e-4` at 140K, then cosine decay to `1.5e-5` at 225K.

The single-GPU logical batch is 80 tracks × 2 segments: a physical microbatch of
40 tracks × two segments with two gradient-accumulation steps. Each
microbatch contains 32,000 audio targets and 160 seconds of waveform; each
optimizer update contains 64,000 audio targets and 320 seconds of waveform.

RIR uses full-wet causal convolution with two seconds of preceding audio;
the prefix is discarded after convolution so reverberant tails from preceding
music enter the query. Room IR train/test assets are source- and content-disjoint.

tc18 runs for 225K optimizer steps. Checkpoints and monitoring both run every
2,500 steps. For direct comparison with tc6, the same seeded 100-track cohort is
evaluated using one balanced canonical, integer-shifted, and held-out
half-offset crop per track. All three groups are evaluated clean and at
0/5/10/20/30 dB at step zero, every 2,500 steps, and at completion. Evaluation
crops are two seconds long and are decoded and tokenized online; training still has
no runtime token-store dependency. W&B preserves the causal metrics and adds
only the epoch-level distillation loss, alongside aggregate RIR-only and
noise+RIR beam Top-1. Complete beam
Top-1/5/10 and MRR results are appended to `probe_metrics.jsonl`.

```bash
python train.py configs/fma_large.yaml \
  --decoder small \
  --schedule noise-rir \
  --codebooks 8 \
  --distillation-weight 0.1 \
  --run-id tc18 \
  --devices 1 \
  --wandb-online
```

This branch accepts only `--decoder small`, `--schedule noise-rir`, and
`--codebooks 8`. The resolved tc18 profile and distillation maximum are embedded
in every checkpoint. Use `--distillation-weight 0.0` for the matched ablation.

The only new W&B key is `train/distillation_loss_epoch`; it remains available
when the configured optimization weight is zero.

Resume the exact saved model, optimizer, scheduler, loop, sampler pass, and RNG
state:

```bash
python train.py configs/fma_large.yaml \
  --devices 1 \
  --run-id tc18 \
  --wandb-online \
  --resume
```

On resume, profile values are recovered from the checkpoint. Any non-matching
checkpoint or incompatible explicit override fails before model construction.

Checkpoints are written every 2,500 optimizer steps under:

```text
/gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/<run-id>/
```

Checkpoints embed the vocabulary, exact cohort, resolved profile, random-crop and replacement
policies, fixed monitor manifest, noise/tokenizer fingerprints, sampler state,
RNG state, query specification, and code mapping. Resume is accepted only for
compatible tc18 checkpoints. Backward compatibility with earlier experiments
is intentionally not provided on this trial branch.

PyTorch deterministic algorithms and seeded Python, NumPy, Torch, samplers, and
workers are enabled. `deterministic_warn_only: true` reports CUDA operations for
which PyTorch cannot promise bitwise determinism instead of aborting a long run.

## Evaluation and inference

The paper-facing evaluation samples a deterministic 1,000-track subset from the
checkpoint's embedded training cohort. It evaluates nested 2/5/10-second queries
under eight fixed suites: held-out background noise at 0/5/10/20 dB, with and
without fully wet held-out room convolution. Noise is mixed before RIR, matching
training. Queries use two-second windows with 50% overlap. Identifier
log-probabilities are averaged across all windows at each shared beam prefix
before pruning:

```bash
python evaluate.py \
  /gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/tc18/last.ckpt \
  evaluation-tc18-joint.json \
  --protocol joint-beam
```

This derives catalogue size and decoder dimensions from the checkpoint and
defaults to a seeded 1,000-track sample, recipe and sample seed `1337`, fixed
query lengths `2/5/10`, the eight fixed noise/RIR suites, beam width 10, and
CUDA. Each track reuses one validation-noise recipe and one validation IR across
all durations and SNRs.

The command writes a JSON summary, paper-ready CSV, append-only query JSONL, and
an immutable evaluation manifest. Matching reruns resume completed queries;
checkpoint, tokenizer, noise/IR-manifest, seed, or protocol mismatches fail. Metrics
include beam Top-1/5/10, MRR, evaluated/failed counts, latency, and throughput
for each duration and suite. Runtime failures count as retrieval misses in
the accuracy and MRR denominator rather than being silently excluded.

The earlier fixed 100-track clean/noise/RIR monitor remains available through
the default `--protocol segment` behavior and is unchanged. Training keeps its
own existing monitor path; the paper-facing path is invoked only by this
standalone command.

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
