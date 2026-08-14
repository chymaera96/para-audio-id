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
two-second segment from a track is a separate document with the same identifier,
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

## Historical offline tokenization utilities

The repository still contains the older cache-preparation commands, but current training
does not use canonical, shifted, or half-offset token stores for training or
evaluation. They remain useful only when reproducing tc2–tc6 from Git history.

Set `data.database_size` to `10000`, `25000`, or `100000`, then prepare that
size-specific identity manifest. This never overwrites a different-size manifest:

```bash
python prepare_training_cohort.py configs/fma_large.yaml
# Writes training_tracks_10k.json, training_tracks_25k.json, or
# training_tracks_100k.json according to data.database_size.
```

## Training

The default causal LM is a randomly initialized 12-layer GPT-2-style decoder with
hidden size 768, 12 heads, tied embeddings, and no dropout. The vocabulary has
2,048 collision-free audio tokens, `[BOS]`, `[ID]`, ten dedicated digit tokens,
and `[EOS]`.

Training samples online two-second crops from the configured cohort. Each
identity contributes a clean anchor and one secondary view: a distinct clean
crop, an exact same-crop background-noise view, an exact same-crop room-reverb
view, or the combined noise-then-reverb view. One frozen lightweight MuQ call
tokenizes all twenty final waveforms in each physical microbatch.

```text
loss =
    (
        100 * mean(clean audio-token CE)
        + 40 * mean(all digit CE)
        + 2 * mean(all boundary CE)
    ) / 142
    + lambda_consistency * clean/noisy [ID]-state consistency
```

The consistency loss is one minus cosine similarity between normalized
final-layer `[ID]` states. The clean state is detached for this term; the noisy
state, transformer, and noisy input embeddings retain gradients. Each loss
family is computed as a mean, then recombined with effective coefficients of
approximately 0.704 audio, 0.282 digits, and 0.014 boundaries. Scaling the
identifier digit weight from 20 to 8 preserves tc7's 2.5:1 audio-to-identifier
ratio despite the shorter query. The previous
tc5 weighted-token loss is also logged as a detached comparison metric.
Checkpoints record the `tc5_family_weighted_consistency_v2` loss protocol.

| Raw steps | Clean | Noise | RIR | Noise + RIR | Consistency weight |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0–50K | 1.00 | 0 | 0 | 0 | 0 |
| 50–62.5K | 1-p | p: 0 → 0.75 | 0 | 0 | 0 → 0.10 |
| 62.5–87.5K | 0.25 | 0.75 | 0 | 0 | 0.10 |
| 87.5–100K | 0.25 | 0.75 → 0.55 | 0 → 0.20 | 0 | 0.10 |
| 100–112.5K | 0.25 | 0.55 → 0.35 | 0.20 | 0 → 0.20 | 0.10 |
| 112.5–175K | 0.25 | 0.35 | 0.20 | 0.20 | 0.10 |

Noisy examples use fixed SNR-bin probabilities `0.40/0.30/0.20/0.10` for
`0–5/5–10/10–20/20–30 dB`; 10% of noisy views are exactly 0 dB. The LR uses
ordinary raw-step warm-up and cosine decay without gates, pauses, or recovery.

The single-GPU logical batch is 80 tracks × 2 segments: a physical microbatch of
ten tracks × two segments with eight gradient-accumulation steps. Each physical
microbatch contains 2,000 audio targets and 40 seconds of waveform, matching
tc7's five-second acoustic budget while retaining two-second queries.

RIR uses full-wet causal convolution with two seconds of preceding audio;
the prefix is discarded after convolution so reverberant tails from preceding
music enter the query. Room IR train/test assets are source- and content-disjoint.

The 10K catalogue is the exposure reference: 10K, 25K, and 100K runs use 70K,
175K, and 700K optimizer steps respectively, with every curriculum boundary
scaled by the same factor. Checkpoints remain every 500 steps and monitoring
remains every 2,500 steps. For direct comparison with tc6, the same seeded 100-track cohort is
evaluated using one balanced canonical, integer-shifted, and held-out
half-offset crop per track. All three groups are evaluated clean and at
0/5/10/20/30 dB at step zero, every 2,500 steps, and at completion. Evaluation
crops are two seconds long and are decoded and tokenized online; tc11 still has
no runtime token-store dependency. W&B keeps the tc6/tc9 compact metric names
and adds aggregate RIR-only and noise+RIR beam Top-1, while complete
training metrics retain tc6's `_step` and `_epoch` suffixes. Complete beam
Top-1/5/10 and MRR results are appended to `probe_metrics.jsonl`.

```bash
python train.py configs/fma_large.yaml \
  --decoder small \
  --schedule noise-rir \
  --devices 1 \
  --run-id tc11 \
  --wandb-online
```

`--decoder` accepts `small` (12×768, 12 heads; default) or `medium`
(24×1024, 16 heads). `--schedule` accepts `noise` (default) or `noise-rir`.
A noise-only run does not discover, validate, fingerprint, or decode room-IR
assets. The resolved profile is embedded in every checkpoint.

When degraded pairs exist, W&B logs `train/relative_cosine_margin_step` and
`train/relative_cosine_margin_epoch`, defined as
`(same_track_cosine - different_track_cosine) / (1 - different_track_cosine)`.

Resume the exact saved model, optimizer, scheduler, loop, sampler pass, and RNG
state:

```bash
python train.py configs/fma_large.yaml \
  --devices 1 \
  --run-id tc11 \
  --wandb-online \
  --resume
```

On resume, omit `--decoder` and `--schedule` to recover both from the checkpoint.
An incompatible explicit override fails before model construction.

Checkpoints are written every 500 optimizer steps under:

```text
/gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/<run-id>/
```

Checkpoints embed the vocabulary, exact cohort, resolved profile, random-crop and replacement
policies, fixed monitor manifest, noise/tokenizer fingerprints, sampler state,
RNG state, query specification, and code mapping. Resume is accepted only for
compatible profile checkpoints. Legacy tc11 maps to the equivalent
25K/small/noise-RIR profile and remains resumable. tc9-small and tc10-medium
remain loadable for evaluation but are not training initialization sources.

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

## Clean capacity diagnostics

Capacity experiments use a separate entry point and configuration, leaving the
tc9–tc11 corruption-training interface unchanged. They train on two distinct
clean two-second random crops per identity, with no background-noise or RIR
asset access and no consistency objective.

Choose `data.database_size` from `10000`, `25000`, `50000`, or `100000` in
`configs/capacity.yaml`, then prepare or validate its size-specific manifest:

```bash
python prepare_training_cohort.py configs/capacity.yaml

python diagnose.py configs/capacity.yaml \
  --decoder small \
  --run-id capacity-10k-small \
  --devices 1 \
  --wandb-online
```

The YAML database size can be overridden consistently for preparation and
training, for example:

```bash
python prepare_training_cohort.py configs/capacity.yaml --database-size 50000
python diagnose.py configs/capacity.yaml \
  --database-size 50000 \
  --decoder small \
  --run-id capacity-50k-small \
  --devices 1 \
  --wandb-online
```

Decoder choices are `tiny` (6 layers, width 512, 8 heads), `small` (12/768/12,
the default), and `medium` (24/1024/16). At the default 560 average identity
exposures, the four catalogue sizes resolve to 70K, 175K, 350K, and 700K
optimizer steps. The LR warms up linearly for 200 steps and then remains fixed
at `3e-4`.

Capacity diagnostics use 40 identities per physical microbatch: 40 tracks ×
two clean documents with `accumulate_grad_batches: 2`. This preserves
the original effective batch of 80 tracks/160 documents per update while using
the GPU more efficiently than the former `10 × accumulation 8` partition.

For multiple GPUs, `--devices` automatically repartitions that same global
80-track optimizer batch; it does not increase the scientific batch size or
change the resolved training length. The common layouts are:

| GPUs | Tracks per GPU | Accumulation | Global tracks/update |
|---:|---:|---:|---:|
| 1 | 40 | 2 | 80 |
| 2 | 40 | 1 | 80 |
| 4 | 20 | 1 | 80 |
| 8 | 10 | 1 | 80 |

For example, a two-GPU 100K-medium run is:

```bash
python diagnose.py configs/capacity.yaml \
  --database-size 100000 \
  --decoder medium \
  --run-id capacity-100k-medium \
  --devices 2 \
  --wandb-online
```

The device count is checkpointed. Exact resume therefore uses the same
`--devices` value as the original run.

Existing valid manifests are reused byte-for-byte; incompatible manifests fail
rather than being overwritten. Capacity W&B logging retains the established
clean training and clean probe names and omits undefined corruption metrics.
Resume with:

```bash
python diagnose.py configs/capacity.yaml \
  --run-id capacity-10k-small \
  --devices 1 \
  --wandb-online \
  --resume
```

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
