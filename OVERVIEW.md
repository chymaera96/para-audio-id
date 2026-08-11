# Experiment overview

This file records the intended meaning of the main W&B/checkpoint run IDs. It
describes the experimental setup, not the result of a run; measured outcomes
should be taken from the corresponding W&B run or evaluation JSON.

## Run lineage

| Run ID | Catalogue | Training views | ID digit weight | Steps | Main change |
| --- | ---: | --- | ---: | ---: | --- |
| `tc2` | Full FMA-large catalogue | Canonical only | 5 | 100K | Replaced the continuous model with Mel-RVQ tokens and a causal transformer |
| `tc3` | 10K tracks | Canonical only | 20 | 50K | Smaller memorisation target and stronger identifier supervision |
| `tc4` | Same 10K tracks | Canonical + integer-shifted | 20 | 100K | Added paired temporal-shift training |
| `tc5` | Same 10K tracks | Clean paired views + online noisy anchor | 20 | 60K | Added the background-noise curriculum |
| `tc6` | Same 10K tracks | Clean paired views + exact clean/noisy pairs | 20 | 70K nominal | Masks noisy audio loss and aligns clean/noisy `[ID]` states |
| `tc7` | Same 10K tracks | Continuous random five-second crops | 20 | 70K | Replaced fixed training grids with online crops |
| `tc8` | Same 10K tracks | Continuous random two-second crops | 8 | 70K | Tests shorter queries at the same audio-to-ID loss ratio |
| `tc9` | Same 10K tracks | Two-second crops, tc7-matched acoustic budget | 8 | 70K | Raises the physical batch from 8 to 20 documents |
| `tc11` | Fresh seeded 25K tracks | Two-second online crops with clean/noise/RIR/combined secondary views | 8 | 175K | Adds full-IR past reverberation and scales the catalogue |

Each entry is a from-scratch model unless explicitly documented otherwise.
In particular, `tc4` is not initialized from `tc3`, `tc5` is not initialized
from `tc4`, and `tc6` is not initialized from `tc5`. Later runs reuse cohorts,
identifier mappings, and token caches—not model weights.

## Shared causal-audio formulation

From `tc2` onward, the system is a discrete-audio causal language model:

```text
[BOS] audio tokens [ID] digit-1 digit-2 digit-3 digit-4 digit-5 [EOS]
```

- Audio is sampled at 24 kHz and tokenized using the frozen
  `OpenMuQ/MuQ-large-msd-iter` Mel-RVQ tokenizer.
- The first two of MuQ's eight 1,024-entry codebooks are serialized in
  time-major, codebook-interleaved order. A five-second crop produces 125 frames
  and 250 audio tokens; tc8's two-second crop produces 50 frames and 100 tokens.
- The vocabulary has 2,061 entries: 2,048 audio tokens, `[BOS]`, `[ID]`, ten
  digit tokens, and `[EOS]`.
- The model is a randomly initialized GPT-2-style causal transformer: 12 layers,
  hidden size 768, 12 attention heads, a 512-token context, tied embeddings, and
  zero dropout. It uses Hugging Face's `GPT2LMHeadModel` implementation but no
  pretrained GPT-2 weights or text tokenizer.
- Training uses ordinary causal shifting. The first digit is predicted from a
  prefix ending in `[ID]`; the five identifier digits are never provided to
  free-running evaluation or identification.
- Identification generates five digits and `[EOS]` using only the query audio,
  MuQ tokenizer weights, and the causal-LM checkpoint. It does not consult the
  catalogue, token shards, a valid-code list, fingerprints, embeddings, or an
  ANN index.
- The common optimizer is AdamW with learning rate `3e-4`, betas
  `[0.9, 0.95]`, zero weight decay, BF16 mixed precision, and gradient clipping
  at `1.0`.
- Through tc8, the single-GPU logical batch is 32 identities with two documents
  each: four identities/eight documents per physical microbatch and eight
  gradient accumulation steps. tc9 increases this to ten identities/twenty
  documents per physical microbatch while retaining accumulation eight.

The arbitrary five-digit code mapping is preserved across the experiments so
changes in behavior can be attributed to the training setup rather than a new
assignment of identifiers.

## `tc2`: Mel-RVQ plus causal transformer

`tc2` is the architectural transition from the earlier continuous MuQ encoder
and cross-attention digit decoder to the present causal-audio formulation.

- Uses the canonical five-second grid at `0, 5, 10, 15, 20, 25` seconds.
- Uses cached `uint16` MuQ tokens rather than running the audio encoder during
  ordinary training.
- Trains one model over the full FMA-large catalogue target.
- Uses identifier digit weight `5`.
- Targets 100,000 optimizer steps with a 2% warm-up followed by cosine decay.
- Introduces autoregressive greedy/beam evaluation: all five digits are
  generated without teacher forcing or catalogue-constrained decoding.

This run establishes whether a causal transformer can learn the basic
audio-token-to-identifier continuation at full-catalogue scale.

## `tc3`: 10K catalogue and identifier weight 20

`tc3` keeps the `tc2` architecture but makes memorisation easier to diagnose.

- Restricts training to a seeded, persisted 10,000-track cohort.
- Reuses the canonical token cache; no re-tokenization is required.
- Persists the exact track IDs and unchanged five-digit mappings in the run
  artifacts and checkpoints.
- Raises identifier digit weight from `5` to `20`, while audio tokens and the
  `[ID]`/`[EOS]` boundaries remain part of the causal objective.
- Uses 200 warm-up steps, step-based training, and a final 50,000-step target.
- Evaluates/checkpoints on a step cadence rather than treating a catalogue pass
  as the stopping unit.

`tc3` is the clean, canonical 10K baseline for the later robustness runs.

## `tc4`: paired shifted-query training

`tc4` tests temporal invariance without adding acoustic degradation. It starts a
new random model on the exact `tc3` 10K identities and codes.

- Canonical positions: `0, 5, 10, 15, 20, 25` seconds.
- Shifted training positions:
  `1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24`
  seconds.
- Held-out evaluation positions:
  `2.5, 7.5, 12.5, 17.5, 22.5` seconds.
- Every physical microbatch contains four identities and, for each identity,
  one cached canonical and one cached shifted document with the same code.
- Canonical and shifted row losses are independently normalized and then
  averaged equally.
- Shifted crops are tokenized as standalone five-second waveforms. Overruns are
  zero-padded and recorded in cache metadata.
- The held-out half-offset cache is physically separate and cannot enter the
  training sampler.
- Uses identifier digit weight `20`, 200 warm-up steps, and 100,000 optimizer
  steps.

This run asks whether exposure to many integer offsets transfers to unseen
half-second offsets while retaining canonical identification.

## `tc5`: background-noise curriculum plus shifted queries

`tc5` starts another randomly initialized model on the same 10K cohort. It
retains the canonical and shifted token caches from `tc4` and creates only noisy
views online.

For each identity, the sampler chooses a cached canonical/shifted pair. One row
is the clean anchor. The second row is either the other cached clean view or a
background-noise version of the anchor:

| Optimizer steps | Noisy-second-view probability | Sampled SNR |
| ---: | ---: | ---: |
| `0–20K` | `0` | Disabled |
| `20–25K` | Linear `0 → 0.25` | `20–30 dB` |
| `25–35K` | Linear `0.25 → 0.50` | `10–30 dB` |
| `35–45K` | Linear `0.50 → 0.75` | `0–30 dB` |
| `45–60K` | `0.75` | `0–30 dB` |

Additional details:

- Only background noise is enabled; there is no RIR, microphone IR,
  consistency loss, contrastive loss, or auxiliary robustness loss.
- Training and validation noise come from disjoint `bg_noise/train` and
  `bg_noise/test` manifests, whose fingerprints are stored with the run.
- Noise is RMS-scaled to the requested SNR while leaving the clean waveform
  unchanged. A shared attenuation is applied only if the mixture would clip.
- Noisy waveforms are tokenized on the training GPU using MuQ's frozen Mel
  frontend, normalization statistics, folding operation, and RVQ. The unused
  Conformer encoder is not loaded in the lightweight online path.
- Startup checks that lightweight tokenization matches the full MuQ path and
  that clean online tokens reproduce the existing canonical and shifted caches.
- Anchor/second-view losses are independently normalized and averaged equally.
  W&B retains aggregate training curves rather than separate canonical,
  shifted, and noisy loss plots.
- Training uses identifier digit weight `20`, 200 warm-up steps, cosine decay
  through step 60,000, and checkpoints every 500 optimizer steps.
- A fixed 100-track monitor runs at step zero, every 2,500 steps, and at the
  final step. It covers canonical, shifted-training, and held-out half-offset
  positions under clean audio and validation noise at `0, 5, 10, 20, 30` dB.
- Silent/corrupt monitoring excerpts retain their cached clean evaluation but
  are skipped for noisy SNR evaluation, where the SNR would be undefined.
- Normal resume restores model, optimizer, scheduler, global step, sampler
  progress, RNG state, curriculum position, and deterministic augmentation
  choices.

The purpose of `tc5` is to learn the clean/shifted mapping first and introduce
acoustic robustness gradually, without transferring weights from a previously
converged clean model.

## `tc6`: noise-consistent identifier states

`tc6` starts from random weights on the same 10K cohort and keeps the tc5
background-noise assets. Its intervention is at the causal identifier boundary:

- Non-noisy identities retain their cached canonical/shifted pair.
- Noisy identities use an exact same-segment clean/noisy pair.
- Noisy audio tokens remain in the input, but their audio next-token targets are
  excluded from the optimized loss.
- Clean and noisy digit and boundary losses remain active.
- The final-layer clean/noisy states at `[ID]` are aligned with a one-sided
  cosine loss; only the clean state is detached for this component.
- Identifier digit weight remains `20`. Audio, digit, and `[ID]`/`[EOS]`
  families are measured independently but recombined using tc5's effective
  coefficients (approximately 0.710/0.284/0.006), preventing a change in
  gradient balance when introducing consistency.

The nominal run is 70K effective steps: 20K clean-only followed by a 50K
noise/consistency curriculum. At 20K, clean integer-shifted teacher-forced exact
accuracy must reach 0.5 before the curriculum clock opens. A clean shifted
beam Top-1 regression can freeze the curriculum, halve consistency weight, and
extend the raw run. The hard ceiling is 120K raw optimizer steps, including the
gate and single recovery allowances. LR decay follows the effective clock and
pauses with the curriculum. Its W&B monitor is beam-only and omits Top-5/10,
MRR, greedy, evaluated/skipped, and redundant per-view/per-SNR cross-product
plots.

The SNR curriculum moves categorical probability mass from easy 20–30 dB noise
toward 0–10 dB noise. In its final phase, approximately 10% of noisy views are
exactly 0 dB. Checkpoints use protocol
`noise_consistency_curriculum_v1`; tc5 checkpoints cannot initialize or resume
tc6. Corrected checkpoints additionally use
`tc5_family_weighted_consistency_v2`. Any tc6 checkpoint created before this
loss marker used the incorrectly rescaled objective and is deliberately
incompatible; tc6 must be restarted from random initialization.

## `tc7`: continuous online crop coverage

`tc7` starts from random weights on the same 10K identities but removes the
canonical/shifted crop grids entirely. Every selected track supplies two
deterministic random five-second crops decoded online. When noise is selected,
the second row is a noisy version of the exact clean anchor crop. All eight
final waveforms are tokenized together by the frozen lightweight MuQ Mel-RVQ
path.

The corrected tc6 loss is unchanged. The schedule is simpler and follows raw
optimizer steps: clean through 20K, a `0→0.75` noise and `0→0.10` consistency
ramp through 25K, then a fixed noisy regime through step 70K. There are no
accuracy gates, effective clocks, pauses, or recovery interventions.

Evaluation preserves tc6's seeded 100-track comparison protocol: one balanced
canonical, integer-shifted, and held-out half-offset position per track, clean
and at `0/5/10/20/30 dB`. These fixed queries are decoded and tokenized online,
so neither tc7 training nor evaluation reads grid-token stores. W&B retains
the retained tc6 metric names—including its `_step` and `_epoch` training
curves—and beam Top-1 probes. Complete beam Top-1/5/10, MRR, position, and
failure records are written to JSONL.

## `tc8`: two-second online random crops

`tc8` starts from random weights and keeps tc7's exact 10K cohort, online crop
sampler, background-noise distribution, consistency objective, raw-step
schedule, logical batch, optimizer, and fixed 100-track evaluation protocol.
The sole scientific intervention is reducing every training and evaluation
query from five seconds to two seconds.

At 25 MuQ frames per second with two selected codebooks, each query supplies
100 audio targets rather than 250. The identifier digit weight is therefore
scaled from 20 to 8:

```text
base loss = (100 * audio CE + 40 * digit CE + 2 * boundary CE) / 142
```

This preserves tc7's 2.5:1 audio-to-identifier ratio. The resulting family
coefficients are approximately 0.704 audio, 0.282 identifier, and 0.014
boundary. Each causal document has 108 tokens and remains within the unchanged
512-position context. tc8 has distinct training and crop protocol markers and
cannot initialize or resume from tc7.

## `tc9`: token-budget-matched two-second crops

`tc9` keeps tc8's two-second query, model, loss, noise curriculum, optimizer,
evaluation protocol, and 70K update count, but increases the physical
microbatch from four to ten tracks. With two documents per track, a tc9
microbatch contains twenty 108-token documents, 2,000 audio targets, and 40
seconds of waveform. This closely matches tc7's eight 258-token documents,
2,000 audio targets, and 40 seconds of waveform per microbatch.

Gradient accumulation remains eight, so tc9 processes 80 track selections and
160 documents per optimizer update. This deliberately supplies 2.5 times as
many identifier sequences as tc7 while matching its acoustic exposure. The
experiment therefore separates tc8's reduced training-data budget from the
intrinsic ambiguity and reduced evidence of a two-second query. Its distinct
`token_budget_matched_two_second_noise_consistency_v1` checkpoint protocol
rejects tc8 and earlier checkpoints.

## `tc11`: 25K noise-and-reverb consistency

`tc11` is a new random initialization over a fresh seeded 25,000-track subset
of the existing 100K catalogue and its unchanged five-digit code mapping. It
keeps tc9's GPT-2-small-style model, two-second crops, 20-document physical
batch, eight accumulation steps, and weighted objective
`(100 audio + 40 digit + 2 boundary) / 142`.

The secondary view follows a four-way 175K-step curriculum: clean through 50K,
noise ramped to 0.75 through 62.5K, then RIR-only and noise+RIR mass introduced
between 87.5K and 112.5K. Consistency ramps from zero to 0.1 over 50K–62.5K.
RIR uses two seconds of preceding music, full-wet convolution with a complete
held-out/train room IR as appropriate, peak normalization, and then discards
the context prefix. Noise+RIR means noise is mixed before convolution.

The existing tc6/tc9 W&B names remain unchanged. Two aggregate beam Top-1
probes are added for RIR-only and noise+RIR; complete condition/SNR details are
stored in probe JSONL and checkpoints. tc11 checkpoints are incompatible with
all earlier protocols.

## Interpretation

The progression isolates eight questions:

1. `tc2`: can the causal formulation memorize audio-token-to-code mappings?
2. `tc3`: does a smaller catalogue and stronger ID supervision make that
   behavior observable?
3. `tc4`: does paired temporal-shift exposure improve position invariance?
4. `tc5`: can online background-noise exposure add acoustic robustness without
   destroying clean and shifted identification?
5. `tc6`: can the causal identifier state become noise invariant without
   rewarding prediction of corruption-dependent noisy audio tokens?
6. `tc7`: can continuous online temporal coverage replace hand-selected crop
   grids while retaining the corrected noise-consistency objective?
7. `tc8`: how much identification and noise robustness remain when acoustic
   evidence is reduced from five seconds to two without changing the relative
   audio-to-identifier supervision strength?
8. `tc9`: does matching tc7's acoustic-token and waveform budget recover the
   two-second model's deficit, or is the remaining difficulty intrinsic to the
   shorter and more ambiguous query?

Teacher-forced digit accuracy is useful for optimization diagnostics, but the
scientific identification result is free-running exact accuracy and beam
Top-k/MRR. The five-digit unconstrained random exact-match baseline is `1e-5`.
