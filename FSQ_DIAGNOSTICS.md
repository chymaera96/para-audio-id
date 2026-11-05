# Frozen FSQ diagnostic pilot

`scripts/fsq_diagnostics.py` is independent of tokenizer training and PSI
training/evaluation. It does not change checkpoints or connect to W&B.

The default pilot uses 100 query identities and 1,000 clean reference excerpts.
All identities are excluded from the tokenizer's 10K training split. Query
identities/crop keys come from the saved monitoring split. The same audio is
prepared once and shared by every model. Invalid distractors are audited and
replaced; an invalid fixed query is an error, not silently replaced.

Queries include exact clean excerpts, 0/5 dB noise, held-out RIR, noise+RIR at
0 dB, offsets +/-10/20/40/80 ms, RIR with +/-40 ms offsets, 1.04 seconds of extra
left context, and a separate non-overlapping excerpt. Existing noise/RIR
loaders and full-wet convolution (including leading IR delays) are reused.
Other excerpts are secondary: a fingerprinter need not map different song
sections close together. Exact clean retrieval is intentionally a self-match
control, not generalisation evidence.

The v2 aligned-context protocol uses a 1.04-second prefix, which is exactly
13/26/52 latent frames at 12.5/25/50 Hz. A 1-second prefix would confound
context changes with a half-frame offset at 12.5 Hz. Temporal readout bins
use equal-time boundaries rounded to the nearest latent frame. Correlations
at common physical lags (80/160/320 ms) complement adjacent-frame statistics.
When comparing the low-rate variants, rerun old models under this protocol
rather than mixing scores from the original pilot. `--queries 250` expands
the same fixed query identity prefix without changing the 1,000-record gallery.

The report includes:

- Bounded encoder coordinates before rounding versus hard FSQ coordinates.
- Gallery-standardized cosine retrieval with global-mean and eight-bin
  temporal readouts, without learned pooling, alignment search or refinement.
- Top-1, rank, hardest-negative margin and distractor-size sensitivity.
- Match recall at a threshold calibrated for 0.1% false matches, using half
  the query identities for calibration and half for testing. This small pilot
  is not a production false-match certification.
- Symbol/coordinate agreement, rounding error, temporal redundancy, code
  entropy, and inter-group mutual information with a shuffled bias control.
- Synchronized FP32 latency for deterministic and ordinary kernels at batch
  sizes 1/16/160; host-transfer timing is separately reported at 1/16.
- Kernel token-equivalence checks on up to 100 queries per audited condition.
  An observed mismatch is reported, not hidden or automatically fixed.

Usage (repeat `--checkpoint` for each model):

```bash
python -u scripts/fsq_diagnostics.py \
  --checkpoint tc3=/absolute/path/fsq-tc3/step-32000.ckpt \
  --output /absolute/path/new-diagnostic-directory
```

Use native training checkpoints, which contain the split; frozen export files
may omit it. The output directory must be new. Results are written after each
model to `REPORT.md` and `results.json`, with a dataset manifest and per-query
CSVs including the strongest wrong recording. No model weights are copied.

These metrics are screening evidence, not PSI accuracy. Coordinate geometry
and fixed pooling may favour some models; compare both readouts, individual
conditions, and latency rather than declaring a winner from one score.
Pre-quantization diagnostics use the bounded continuous coordinates, not the
raw unbounded encoder logits. Correlated frame/group samples make marginal
entropy an unreliable measure of recording-specific information capacity.
