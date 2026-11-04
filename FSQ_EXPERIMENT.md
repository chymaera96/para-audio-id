# CNN + FSQ pilot

Branch `fsq` starts from `main` at `e87da26` and includes the plotting update
`26beb48`. The local checkout is `/Users/aditya/para-audio-id`, so VS Code uses
the FSQ branch in the original project folder. The canonical Apocrita checkout
is `/data/home/acw723/para-audio-id`, also on `fsq`. Local plots/logs are preserved,
and Git stashes retain the pre-migration changes as backups.
This implements the supplied
`CNN_FSQ_Tokenizer_Pilot.md`, with the previously unspecified choices below.

## Tokenizer experiment

| Setting | Actual configuration |
| --- | --- |
| Music | 10,000 random FMA catalogue tracks; 500 disjoint monitoring tracks |
| Split seed | 1337; recording lists and monitoring crop keys saved in `split.json` |
| Audio | Mono, 24 kHz, two-second targets |
| RIR context | Two seconds of preceding audio, zero-padded at the start of a recording |
| Features | 80 Slaney-normalized mel bands, 30–12,000 Hz; FFT 1024, Hann window, hop 240 |
| Feature scaling | `(log10(max(mel_power, 1e-5)) + 2) / 3`; no per-input gain normalization |
| Mel rate | 100 Hz, exactly 200 frames for two seconds; extra final centered-STFT frame removed |
| Encoder | Conv1D widths 128/192/256; two stride-2 layers; residual blocks with dilations 1/2/4/8 |
| Bottleneck | One group of five FSQ coordinates, levels `[4,4,4,4,4]` |
| Latent rate | 25 Hz, 50 packed symbols per two-second crop |
| Capacity | 1,024 possible symbols, 10 nominal bits/frame, 250 nominal bits/s |
| Decoder | Quantized coordinates only; two residual blocks and two stride-2 transposed convolutions; no encoder skip connections |
| Parameters | 3,141,333 including reconstruction decoder |
| Batch | 64 clean/secondary pairs, i.e. 128 encoder/decoder inputs per update |
| Optimizer | AdamW, LR 3e-4, weight decay 1e-4, gradient clipping 1.0 |
| Schedule | 500-update linear warmup, cosine decay to 1e-5 at 50,000 updates |
| Precision | BF16 mixed precision; log-mels and FSQ grid arithmetic in FP32 |
| Budget | 50,000 updates = 3.2M recording draws, approximately 320 exposures per training track |
| Checkpoints / monitoring | Every 1,000 updates and at the end |

Every pair has a clean input and a secondary input of the same source crop.
The secondary mixture is clean 0.10, noise 0.35, RIR 0.30, noise+RIR 0.25,
using the established final PSI mixture from the outset. Noise SNR bins are
0–5, 5–10, 10–20, 20–30 dB with probabilities .4/.3/.2/.1; the first bin
has a .25 probability of exactly 0 dB. These are the existing PSI SNR settings.
Pitch shifts, time stretching, microphone IRs, and a contrastive objective are
not part of this experiment.

The existing `BackgroundNoiseAssets`, `RoomImpulseResponseAssets`,
`mix_background_noise`, and `convolve_full_wet` are reused. Noise is applied
before RIR convolution. The original RIR samples, including leading delay,
are retained; the existing convolution normalization and crop alignment are
preserved. Clean targets refer to the same nominal source interval; no new
delay compensation is introduced. Existing training/test noise and room pools
remain separate and are fingerprinted in checkpoints.

Both reconstructions target the same clean log-mel:

```text
loss = 0.5 * (L1(reconstruct(clean), clean_mel)
            + L1(reconstruct(secondary), clean_mel))
       + consistency_weight * mean(abs(q_clean - q_secondary))
```

Consistency defaults to **0.0**. If explicitly enabled in a new configuration,
it compares normalized FSQ coordinates, never packed integer IDs. There is no
commitment/codebook loss, recording-ID supervision, or pretrained MuQ encoder
in tokenizer training. Quantization uses bounded scalar grids and
straight-through rounding. Packed IDs use a fixed mixed-radix map with the
first coordinate least significant.

Monitoring fixes the source crop and degradation keys for all four conditions.
Unusable monitoring crops are retried deterministically; unreadable/silent
tracks are replaced from unused catalogue tracks, excluding training and other
monitor identities. Decisions are saved in the split audit. Runtime training
crop replacements are counted in the logs. Resume checks split, configuration,
and degradation fingerprints and continues at the next global sample index.

Metrics include clean-target L1, exact token agreement, coordinate agreement
and distance, token frequencies, occupied symbols, and entropy-derived effective
symbol count. `monitor-latest.pt` contains histograms, source features,
reconstructions, and IDs. W&B receives the scalar metrics and reconstruction
panels. These are diagnostics; downstream identification establishes usefulness.

## Running on Apocrita

Source changes are made locally, committed, and pushed to GitHub. Update the
Apocrita checkout only with `git pull --ff-only origin fsq`; do not edit or copy
source files directly on the cluster. Generated data, logs, and checkpoints
remain runtime artifacts.

On an allocated GPU, with the `para-audio-id` environment:

```bash
cd /data/home/acw723/para-audio-id
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python train_fsq.py configs/fsq_tokenizer.yaml \
  --run-id fsq-pilot-v1 --wandb-online
```

Resume the same checkpoint and W&B run:

```bash
python train_fsq.py configs/fsq_tokenizer.yaml \
  --run-id fsq-pilot-v1 --resume --wandb-online
```

Continue the reconstruction-only `fsq-tc1` snapshot with consistency weight 0.1
in a separate run:

```bash
python -u train_fsq.py configs/fsq_tokenizer.yaml \
  --continue-from /gpfs/scratch/acw723/para-audio-id/fsq-tokenizer-checkpoints/fsq-tc1/step-15000.ckpt \
  --run-id fsq-tc1-cons01 \
  --consistency-weight 0.1 \
  --wandb-online
```

This restores model weights, AdamW moments, and update 15,000 (the last saved
snapshot before the original process was suspended at update 15,975). The existing
50K learning-rate schedule continues, leaving 35K updates; there is no warmup
restart or LR increase. Only the consistency weight and run name change. A new
W&B run is created, and the source checkpoint remains unchanged. To resume this
continuation later, use its saved configuration:

```bash
python -u train_fsq.py \
  /gpfs/scratch/acw723/para-audio-id/fsq-tokenizer-checkpoints/fsq-tc1-cons01/config.yaml \
  --resume --wandb-online
```

The W&B ID is persisted in `wandb-id.txt` and checkpoints. The CLI uses the
same ID with `resume="must"`. Logs, split/asset metadata, and optimizer state
are stored with each run under:

```text
/gpfs/scratch/acw723/para-audio-id/fsq-tokenizer-checkpoints/<run-id>/
```

DataLoader workers use the `spawn` start method, avoiding inherited W&B
telemetry/import hooks and CUDA state. If the first batch fails before a
checkpoint is written, rerun the original command without `--resume`; the
persisted W&B ID reconnects to the same run. Use `--resume` once a checkpoint
exists to restore the model and optimizer as well.

`--sanity-steps 100` creates a separate `-sanity` run, limits monitoring to
16 tracks, and never connects to W&B. `--stop-after N` stops and saves after
N additional updates without changing the full 50K LR schedule, including
when used with `--resume`. A full run uses all 500 monitoring tracks.

For remote access and automatic `ood_jupy*` allocation discovery, see
`~/APOCRITA.md` on the local machine and `scripts/apocrita.py`.

## Frozen tokenizer and PSI stage

### Capacity/rate tokenizer ablations: fsq-tc3 through fsq-tc6

These fresh runs retain tc2's split, 100 Hz mel frontend, augmentation,
reconstruction loss, consistency weight 0.1, and 50K learning-rate schedule.
Each is stopped and saved at 32K updates with `--stop-after 32000`.

| Run | Config | Parallel FSQ groups | Latent frames/s | Tokens per 2 s |
| --- | --- | ---: | ---: | ---: |
| fsq-tc3 | configs/fsq_tc3.yaml | 2 | 25 | 100 |
| fsq-tc4 | configs/fsq_tc4.yaml | 4 | 25 | 200 |
| fsq-tc5 | configs/fsq_tc5.yaml | 1 | 50 | 100 |
| fsq-tc6 | configs/fsq_tc6.yaml | 2 | 50 | 200 |

Each group uses `[4,4,4,4,4]` levels. The 50 Hz models reduce the second
encoder stride from two to one and mirror this in the reconstruction decoder;
mel features are unchanged. Default strides remain `[2,2]`, preserving old
checkpoint shapes. The launch order is tc3, tc5, tc4, tc6, sequentially in
allocation 35273583, with online W&B in `para-audio-id-fsq`.
This stage trains tokenizers only. The current downstream PSI configuration
adapter still restricts frozen FSQ to one group at 25 Hz and needs a separate
update before using these variants. Causal audio next-token prediction remains
part of the intended downstream PSI objective.

### Lower-rate and smaller-encoder ablations: fsq-tc7 through fsq-tc10

Fresh runs use the same split, augmentation, consistency weight 0.1, batch 64,
50K LR schedule and explicit 32K stopping point as tc3--tc6.

| Run | Latent frames/s | Groups | Tokens/2 s | Encoder channels |
| --- | ---: | ---: | ---: | --- |
| fsq-tc7 | 12.5 | 4 | 100 | 128/192/256 |
| fsq-tc8 | 12.5 | 8 | 200 | 128/192/256 |
| fsq-tc9 | 25 | 4 | 200 | 64/96/128 |
| fsq-tc10 | 12.5 | 8 | 200 | 64/96/128 |

The 100 Hz mel frontend is unchanged. `input_pool_stride: 2` applies a fixed
two-frame average before decimation and the first learned convolution; the
two existing stride-two convolutions then produce 12.5 Hz latents. This is a
simple local low-pass operation, not an ideal anti-aliasing filter.

All four runs explicitly retain decoder channels 128/192/256 and decoder
strides 2/2. Low-rate quantized coordinates are linearly interpolated from
12.5 to 25 Hz before that decoder. This interpolation adds no discrete tokens
and no trainable parameters; the exported tokenizer still emits the true low
rate. Smaller encoders do not shrink the reconstruction decoder. Default
pooling and decoder settings preserve earlier checkpoint parameter keys and
forward behaviour.

Configs are `configs/fsq_tc7.yaml` through `configs/fsq_tc10.yaml`. Runs are
sequential in on-demand allocation 35395413, in numeric order, with online
W&B in `para-audio-id-fsq`. Source changes go through Git; the cluster-only
runner is `fsq_tc7_tc10_32k.sh` on Apocrita. No PSI training is launched.

### tc19: 25K medium decoder with fsq-tc2 at 28K

`configs/fsq_psi_tc19.yaml` uses the frozen `fsq-tc2/step-28000.ckpt`
tokenizer and the existing 25K identity manifest. The decoder has 24 layers,
width 1024, and 16 heads. It uses one GPU, 40 recordings/two documents per
microbatch, and accumulation two (80 recordings per optimizer update).
The original 25K schedule runs for 225K updates. PSI identifier-logit
distillation is disabled, matching the previous medium-model command.
W&B project is `para-audio-id`, with run name and ID `tc19`. Checkpoints and
monitoring run every 2,500 updates. FSQ weights remain frozen.

After pulling the branch on Apocrita:

```bash
python prepare_fsq_psi.py \
  /gpfs/scratch/acw723/para-audio-id/fsq-tokenizer-checkpoints/fsq-tc2/step-28000.ckpt \
  --export-only
python -u train.py configs/fsq_psi_tc19.yaml --run-id tc19 --wandb-online
```

Resume with the same command plus `--resume`. W&B keeps the same run ID.

### tc20: PSI-S on the same 25K catalogue

`configs/fsq_psi_tc20.yaml` changes tc19's decoder to small (12 layers, width
768, 12 heads) and its run name/ID to `tc20`. The tokenizer checkpoint, cohort,
batch, loss weights, acoustic curriculum, and 225K schedule are identical.
It trains a fresh PSI decoder and logs to W&B project `para-audio-id`.

```bash
python -u train.py configs/fsq_psi_tc20.yaml --run-id tc20 --wandb-online
```

Resume with the same command plus `--resume`. Checkpoints are saved under
`/gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints/tc20/`.

### Generic small-decoder pilot

After tokenizer training, export a frozen snapshot and generate PSI config:

```bash
python prepare_fsq_psi.py \
  /gpfs/scratch/acw723/para-audio-id/fsq-tokenizer-checkpoints/fsq-pilot-v1/last.ckpt
python train.py configs/fsq_psi.yaml \
  --run-id fsq-psi-25k-small-v1 --wandb-online
```

The export is an immutable `tokenizer-step-N.ckpt`. PSI uses the existing 25K
cohort, small GPT-2 decoder, 225K-update recipe, 80 recording selections per
update, clean/secondary pairs, acoustic curriculum, and causal audio/ID loss.
ID weight 4 preserves the aggregate 2.5:1 audio/identifier ratio. A document has
50 audio tokens and 58 total tokens; the vocabulary has 1,037 entries.
The inherited identifier-logit distillation default is 0.1 and can be set with
`--distillation-weight` as in the existing recipe.

Frozen FSQ inference always uses FP32, including when PSI runs under BF16, to
keep quantization decisions consistent with standalone evaluation. Its weight
hash and feature specification are recorded in the PSI checkpoint; evaluation
loads that exact frozen tokenizer. The full reconstruction decoder remains in
the export but is never called during PSI tokenization. CNN/FSQ parameters are
frozen and excluded from the PSI optimizer.

Compared with eight-stage Mel-RVQ at 25 Hz, this uses 50 rather than 400 input
tokens per crop and 250 rather than 2,000 nominal bits/s. Interpretation must
account for that capacity and sequence-length difference. Tokenizer monitoring
tracks may belong to the later PSI catalogue, as allowed in the pilot.

This setup and its short checks do not establish that the tokenizer improves
PSI retrieval; that requires completing the two training stages and evaluating
the existing shared queries.

## Verification on 2026-11-02

- Allocation 35076673, A100 40 GB, existing `para-audio-id` environment.
- Fresh 100-update tokenizer check, followed by resumes of 20 and 300 updates.
  Final saved update: 420. No W&B connection was used for these checks.
- All 500 monitoring tracks completed each of clean/noise/RIR/noise+RIR
  (2,000 condition queries). One silent monitoring crop needed a deterministic
  retry; that decision is in the split audit.
- The final 300-update window averaged 0.532 seconds/update. A complete 50K run
  therefore projects to roughly 7.4 hours; allocate 10–12 hours for variability
  and overhead. Fixed monitoring crops are loaded once at startup; a complete
  cached 500-track monitoring pass took about 1.6 seconds.
- Training reconstruction loss was 0.434 at update 25 and 0.191 at update 420.
  This checks learning and execution, not convergence or identification quality.
- A frozen snapshot completed three actual small-decoder PSI optimizer updates
  and saved a PSI checkpoint, without W&B.
- Targeted Apocrita tests: 36 passed. Local quantizer/profile/tokenizer/generation
  tests: 47 passed. Static checks and whitespace checks passed.

Verified tokenizer checkpoint:

```text
/gpfs/scratch/acw723/para-audio-id/fsq-tokenizer-checkpoints/fsq-pilot-verified-35076673-sanity/last.ckpt
```

This is a separate sanity run. The production command above starts
`fsq-pilot-v1` fresh with W&B and the full 50K budget.
