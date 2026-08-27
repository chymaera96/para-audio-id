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

tc18 supports either the existing 25K identity manifest or a fingerprinted
100K manifest. Prepare the selected cohort if it is not already available:

```bash
python prepare_training_cohort.py configs/fma_large.yaml
# Writes or validates data/training_tracks_25k.json.

python prepare_training_cohort.py configs/fma_large.yaml \
  --database-size 100000
# Writes data/training_tracks_100k.json.
```

## Training

The `scale` branch is a fixed 100K-track GPT-2-medium throughput experiment. The
decoder has 24 layers, hidden size 1024, 16 heads, tied embeddings, and zero
dropout. Two-second crops use all eight MuQ codebooks, producing 400 audio
tokens, 408-token causal documents, and an 8,205-entry vocabulary.

Each identity contributes a clean anchor and either a distinct clean crop or an
exact same-crop noise, room-reverb, or noise-then-reverb view. The causal and
temperature-2 identifier-logit distillation objectives are unchanged from tc18:

```text
loss =
    (
        400 * mean(clean audio-token CE)
        + 160 * mean(all digit CE)
        + 2 * mean(all boundary CE)
    ) / 562
    + lambda_KD * identifier-logit distillation
```

For every degraded secondary, the clean anchor is a stop-gradient
teacher at the five causal positions that predict the next identifier digit.
The KL divergence is computed only over the ten digit-token logits with
temperature 2. This uses the existing single decoder pass and adds no model
parameters or inference-time modules. The causal coefficients are approximately
0.709 audio, 0.284 digits, and 0.007 boundaries with identifier
weight 32. The 512-position table remains sufficient. The training protocol is
`scale_100k_medium_4gpu_eight_codebook_v1`; its base loss protocol remains tc18.

The memory probe selected 16 tracks (32 documents) per GPU. Production training
therefore requires exactly four GPUs and accumulation one, for a global update
of 64 identities and 128 documents. The run preserves 72 million identity
selections, so it lasts 1,125,000 optimizer updates.

All exposure-dependent boundaries are scaled by `ceil(reference_step × 80 / 64)`:

- corruption boundaries: 50K, 150K, and 300K;
- distillation boundaries: 75K and 150K;
- LR hold and linear-decay boundaries: 300K and 700K.

Warm-up remains 500 optimizer steps. Monitoring runs every 5K steps and
checkpoints are saved every 10K steps. RIR remains full-wet with two seconds of
preceding context, and all W&B metric names are unchanged.

RIR uses full-wet causal convolution with two seconds of preceding audio;
the prefix is discarded after convolution so reverberant tails from preceding
music enter the query. Room IR train/test assets are source- and content-disjoint.

```bash
python train.py configs/fma_large.yaml \
  --database-size 100000 \
  --decoder medium \
  --schedule noise-rir \
  --codebooks 8 \
  --distillation-weight 0.1 \
  --run-id scale-100k-medium \
  --devices 4 \
  --wandb-online
```

To measure a safe medium per-GPU microbatch on a one-GPU OnDemand node, run:

```bash
python probe_medium_memory.py configs/fma_large.yaml
```

The probe measured 71.06 documents/s and 16.72% peak headroom at the selected
16-track layout; 20 tracks per GPU ran out of memory. The probe creates no
checkpoints or W&B run. Production batch controls are deliberately not exposed
through the CLI. Use `--distillation-weight 0.0` for the matched ablation.

The only new W&B key is `train/distillation_loss_epoch`; it remains available
when the configured optimization weight is zero.

Resume the exact saved model, optimizer, scheduler, loop, sampler pass, and RNG
state:

```bash
python train.py configs/fma_large.yaml \
  --devices 4 \
  --run-id scale-100k-medium \
  --wandb-online \
  --resume
```

On resume, profile values are recovered from the checkpoint. Any non-matching
checkpoint or incompatible explicit override fails before model construction.

Checkpoints are written every 10,000 optimizer steps under:

```text
/gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/<run-id>/
```

Checkpoints embed the vocabulary, exact cohort, resolved profile, random-crop and replacement
policies, fixed monitor manifest, noise/tokenizer fingerprints, sampler state,
RNG state, query specification, and code mapping. Resume is accepted only for
compatible scale checkpoints. Historical tc18 checkpoints remain loadable for
evaluation but cannot resume this training profile.

PyTorch deterministic algorithms and seeded Python, NumPy, Torch, samplers, and
workers are enabled. `deterministic_warn_only: true` reports CUDA operations for
which PyTorch cannot promise bitwise determinism instead of aborting a long run.

## Evaluation and inference

The paper-facing evaluation samples a deterministic 1,000-track subset from the
checkpoint's embedded training cohort. It evaluates nested 2/3/5/10-second clean and
held-out-room-IR queries using two-second windows with 50% overlap. Identifier
log-probabilities are averaged across all windows at each shared beam prefix
before pruning:

```bash
python evaluate.py \
  /gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/tc11/last.ckpt \
  evaluation-tc11-joint.json \
  --protocol joint-beam
```

This derives catalogue size and decoder dimensions from the checkpoint and
defaults to a seeded 1,000-track sample, recipe
and sample seed `1337`, query lengths `2/3/5/10`, clean and RIR conditions,
beam width 10, and CUDA. The corresponding flags remain available for smaller
diagnostic evaluations.

Use `--conditions clean` for tc9, tc10, or tc11 without touching RIR assets.
`--conditions clean rir` adds held-out room convolution. Historical checkpoints
can use `--rir-training-root` and `--rir-validation-root` overrides.

The command writes a JSON summary, paper-ready CSV, append-only query JSONL, and
an immutable evaluation manifest. Matching reruns resume completed queries;
checkpoint, tokenizer, IR-manifest, seed, or protocol mismatches fail. Metrics
include beam Top-1/5/10, MRR, evaluated/failed counts, latency, and throughput
for each duration and condition. Runtime failures count as retrieval misses in
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
