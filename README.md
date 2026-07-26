# Parametric audio identification

This repository tests whether a MuQ-based model can memorize a 100,000-recording
catalogue and emit a recording's randomly assigned five-digit code directly from a
short, degraded query. Inference uses no reference audio, fingerprint index, ANN
search, or catalogue-constrained decoder.

## Setup and catalogue

Install the project in an environment with a matching PyTorch/CUDA build:

```bash
pip install -e '.[dev]'
```

Build the deterministic catalogue once. The preparation pass probes files and skips
bad FMA-Large audio before assigning codes:

```bash
python prepare_catalogue.py \
  /gpfs/scratch/acw723/fma_large \
  data/fma_large_100k.jsonl \
  --bad-files data/catalogue_bad_files.jsonl
```

Download the original-resolution degradation datasets and compile them as mono
24 kHz WAV files:

```bash
./prepare-datasets.sh
```

The script downloads TUT Acoustic Scenes 2016, MIT Survey, OpenAIR, Aachen AIR
v1.4, and the Surrey microphone IR collection from their individual official
sources. It applies the `neural-music-fp` selection and split rules, separates
stereo IR channels, and resamples selected files to 24 kHz. Downloads and
extracted originals default to
`/gpfs/scratch/acw723/neural-music-fp-dataset/degradation_sources`; compiled data
goes to the parallel `degradation_24k` tree used by the training config. Optional
positional arguments are `WORK_ROOT OUTPUT_ROOT WORKERS`. Reruns resume downloads
and skip valid compiled outputs. Set `SKIP_DOWNLOAD=1` to recompile already
extracted sources without contacting providers. Exact room/microphone waveform
duplicates are removed before conversion; `source_duplicates.jsonl` records every
removed source and flags any duplicate that crossed the train/test boundary.

The JSONL catalogue uses paths relative to the audio root. Its adjacent metadata
file records the selection and code seeds. Decode failures discovered later by
DataLoader workers are appended under a file lock to
`data/runtime_bad_files.jsonl`; workers skip them for the rest of the run and all
future runs load and ignore them immediately.

## Training

[`configs/fma_large.yaml`](configs/fma_large.yaml) defaults to clean 5-second,
24 kHz queries for the initial memorisation run. All augmentation families are
disabled and their probabilities are zero, so their asset roots are not accessed.

Training uses a deterministic inventory of non-overlapping five-second windows,
plus a duration-anchored tail window when needed. Each exposure presents every
valid catalogue identity once in global batches of eight songs and eight canonical
views per song. On DDP, the global identity count is divided across ranks.

```bash
python train.py configs/fma_large.yaml --run-id first-100k
python train.py configs/fma_large.yaml --run-id first-100k --resume
python train.py configs/fma_large.yaml --run-id first-100k --wandb-online
python train.py configs/fma_large.yaml --run-id first-100k --devices 1
```

Configuration is the source of truth. `--run-id` overrides both the output
subdirectory and W&B run name; `--wandb-online` enables online logging;
`--devices` overrides the GPU count and lets Lightning select single-device or
distributed execution automatically. The same functions can be imported:

```python
from para_audio_id.config import load_config
from para_audio_id.training import train

train(load_config("configs/fma_large.yaml"))
```

Phase 1 keeps MuQ frozen and in evaluation mode for two complete catalogue
exposures. At the start of exposure three, phase 2 unfreezes only its upper quarter
at a 10x lower learning rate. The decoder has no dropout or weight decay for this
memorisation run.

Teacher-forced training and validation metrics are named explicitly. A fixed
512-track clean probe reports greedy exact accuracy every exposure and beam
Top-1/5/10 every five exposures and at the phase transition. Checkpoints are saved
every five exposures under
`/gpfs/scratch/acw723/para-audio-id/checkpoints/<run-id>/` and include the complete
code mapping, segment policy, exposure state, and fixed probe.

## Evaluation and inference

Clean and degraded evaluation use deterministic 10%, 50%, and 90% positions in
every catalogue recording:

```bash
python evaluate.py /gpfs/scratch/acw723/para-audio-id/checkpoints/first-100k/last.ckpt clean.json
python identify.py /gpfs/scratch/acw723/para-audio-id/checkpoints/first-100k/last.ckpt query.wav
```

Evaluation reports greedy Top-1, beam Top-1/5/10, beam reciprocal rank,
teacher-forced per-digit accuracy, and latency. `--max-queries` provides a cheap
diagnostic run. `identify.py` and `para_audio_id.checkpoint.load_network` load only
the model, preprocessing configuration, and small code metadata.

## Tests

```bash
pytest
ruff check .
```

The real-checkpoint MuQ shape and backward smoke test is opt-in because it loads
the full model:

```bash
RUN_MUQ_INTEGRATION=1 pytest -m integration tests/test_muq_integration.py
```
