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
The current experiment samples a seeded 1,000-track subset from all tracks with
six complete cached segments. The exact selected track IDs are saved as
`training_tracks.json` beside the run configuration and embedded in checkpoints;
the full token cache is reused without re-tokenization. Identifier digit targets
have a fixed loss weight of 20 for this run.
The single-GPU logical batch is 32 tracks × 2 segments: a physical microbatch of
four tracks × two segments with eight gradient-accumulation steps.
Training stops after 20,000 optimizer steps, warms up for exactly 200 optimizer
steps, and runs validation, greedy/beam probes, and checkpointing every 500
optimizer steps. Catalogue passes still reshuffle the cached observations, but
they do not control stopping, evaluation, or checkpoint cadence.

```bash
python train.py configs/fma_large.yaml \
  --devices 1 \
  --run-id fma-large-audio-lm \
  --wandb-online
```

Resume the exact saved model, optimizer, scheduler, loop, sampler pass, and RNG
state:

```bash
python train.py configs/fma_large.yaml \
  --devices 1 \
  --run-id fma-large-audio-lm \
  --wandb-online \
  --resume
```

Checkpoints are written after each complete catalogue pass under:

```text
/gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/<run-id>/
```

Configuration and checkpoints are marked `architecture: audio_lm_v1`; missing or
legacy architecture metadata is rejected. Checkpoints embed vocabulary, model
configuration, tokenizer fingerprint, and code-mapping fingerprint, but not
training token shards or catalogue reference features.

PyTorch deterministic algorithms and seeded Python, NumPy, Torch, samplers, and
workers are enabled. `deterministic_warn_only: true` reports CUDA operations for
which PyTorch cannot promise bitwise determinism instead of aborting a long run.

## Evaluation and inference

Evaluation tokenizes shifted clean excerpts online at 2.5, 12.5, and 22.5 seconds:

```bash
python evaluate.py \
  /gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/fma-large-audio-lm/last.ckpt \
  evaluation.json \
  --max-tracks 512
```

It reports audio loss/perplexity, teacher-forced identifier metrics, greedy exact
accuracy, beam Top-1/5/10, MRR, latency, model size, and peak inference memory.
Beam generation searches model continuations only. It requires exactly five
digit tokens after `[ID]`, followed by `[EOS]`; no catalogue-derived valid-code
constraint is used.

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
