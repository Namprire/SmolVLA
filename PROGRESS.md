# Progress

## DONE

- Inspected the empty repository and installed LeRobot 0.4.4 with SmolVLA support in `.venv`.
- Verified that MPS is available on the macOS host (the restricted command sandbox itself hides MPS).
- Inspected the installed policy, processor, and dataset APIs.
- Confirmed the checkpoint output feature is 7-D and supports explicit `noise` in `predict_action_chunk`.
- Downloaded and inspected real episode 0 metadata/data from `HuggingFaceVLA/smol-libero`.
- Identified that the required dataset is published only as LeRobot v2.1 while installed LeRobot reads v3.0.
- Passed the first hard gate with the real checkpoint, real frame, exact dataset task, official processors, and a finite postprocessed 7-D action.
- Verified actual inference on `mps:0` with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Inspected the installed LeRobot 0.4.4 SmolVLA policy and serialized processor implementations directly.
- Implemented a clean `predict_action(frame, instruction, noise=None, seed=None)` API that copies checkpoint inputs, replaces only `task`, uses the official checkpoint pre/postprocessors, and returns a postprocessed 7-D action plus model inference latency.
- Confirmed the supported explicit-noise path is `SmolVLAPolicy.predict_action_chunk(batch, noise=...)` and documented the checkpoint-specific noise contract: `(1, 50, 32)`, `torch.float32`, on the state/model device.
- Traced `task` through `batch_to_transition` complementary data, batching, newline insertion, and the official SmolVLM tokenizer into language token IDs and boolean attention masks.
- Passed the Phase 4 determinism gate on a real episode-0 frame: five repeated predictions with the same task and explicit noise tensor were bit-for-bit identical (maximum absolute pairwise difference `0`).
- Completed the Phase 5 single-frame controlled language substitution with four exact strings, fixed images/state/checkpoint/processors, and one reused explicit noise tensor.
- Completed the Phase 6 pilot with 9 real observations across episodes 0-2 and early/middle/late positions, producing 36 incrementally written, validated rows.
- Independently revalidated all output artifacts: finite 7-D actions, four conditions per observation, correct progress and strings, one noise identifier per observation, and no duplicate records.
- Completed Phase 7 Stage A on episodes 0-1 with 40 observations and 160 predictions; all integrity gates passed.
- Tested resumability by rerunning Stage A: all 160 valid rows were loaded and skipped, zero predictions were recomputed, the CSV SHA-256 was unchanged, and no duplicates were introduced.
- Completed Phase 7 Stage B by resuming the same CSV for episodes 0-9, producing the full target of 200 observations and 800 predictions.
- Independently validated the main experiment: exactly 20 unique frames per episode, four conditions per observation, exact instructions, deterministic noise reuse, finite 7-D expert/predicted actions, positive timings, broad progress coverage from 0 to 1, one MPS device, and no duplicate keys.
- Completed Phase 8 from the immutable Phase 7 CSV: validated all pairing assumptions, calculated agreement with the recorded expert action, paired language effects, safe translation cosine similarity, latency summaries, and episode-cluster bootstrap intervals.
- Completed Phase 9 with the five specified task-progress bins, aggregate and per-episode stage summaries, and three presentation figures in PNG and PDF formats.
- Confirmed the Phase 8-9 analysis is deterministic and model-free; the SHA-256 of `results/vla_predictions.csv` remained `ab4984aeea52a8ca60c94a5cb19a15d69fd27707f5c5734ddd8bec680f376dc4`.

## CURRENT

- Phases 1-9 are complete and verified. Awaiting inspection and authorization.

## NEXT

- Phase 10 — Gripper Semantics, only after explicit authorization.
- Do not start gripper semantics, release work, stochasticity, MuJoCo, or demos without explicit approval.

## BLOCKERS

- None currently. The dataset format mismatch is handled locally with the converter bundled in LeRobot 0.4.4.

## PHASE 3-6 RESULTS

- Determinism file: `results/determinism_check.csv` (5 rows; maximum absolute repeated-output difference `0`).
- Exact language/API audit: `results/api_inspection.md` and `results/language_conditions.json`.
- Single-frame comparison: `results/language_single_frame.csv` (4 rows).
- Pilot predictions: `results/vla_predictions_pilot.csv` (36 rows = 9 observations x 4 conditions).
- Pilot mean full 7-D distance from Correct: paraphrase `0.09006596`, contradictory `0.12267754`, unrelated `0.44861949`; all three differed from Correct in all 9 observations at a `1e-8` threshold.
- These are controlled-pipeline and small-pilot diagnostics only, not scientific conclusions.

## PHASE 7 RESULTS

- Episodes: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`.
- Sampling: 20 seeded, stratified, unique observations per episode, including real frame 0 and the real maximum frame; 200 observations total.
- Predictions: 800 rows in `results/vla_predictions.csv` with exactly four validated language conditions per observation.
- Actual compute device: `mps:0` with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
- Mean/median model inference latency: `541.0591 ms` / `539.1989 ms`.
- Summed model inference latency: `432.8473 s`.
- Measured Stage A + Stage B wall runtime: `479.4278 s` (7 min 59.4 s), excluding the separate no-op resumability test.
- Selection reconstruction: `results/main_experiment_selection.json` records every local/global frame index, task progress, selection seed, noise seed, and noise hash.
- Final integrity report: `results/main_experiment_validation.json` (`PASS`).
- No deviation from the target experiment size and no scientific language-condition metrics or interpretations were computed.

## PHASE 8-9 RESULTS

- Input validation: `PASS` for 800 rows, 200 paired observations, four conditions per observation, identical within-observation noise and expert actions, finite 7-D vectors, progress in `[0, 1]`, and positive finite timings.
- Paired effects: `results/language_effects.csv` contains 600 alternative-versus-Correct comparisons. Mean full 7-D distances are Paraphrase `0.202755`, Contradictory `0.354315`, and Unrelated `0.806369`.
- Episode-cluster bootstrap (2,000 deterministic replicates) 95% CIs for mean full 7-D distance are Paraphrase `[0.172452, 0.231373]`, Contradictory `[0.297352, 0.414462]`, and Unrelated `[0.761888, 0.855950]`.
- Mean translation / rotation / gripper agreement errors are Correct `0.340237 / 0.070786 / 0.208461`, Paraphrase `0.363251 / 0.071559 / 0.228939`, Contradictory `0.414641 / 0.072025 / 0.276336`, and Unrelated `0.615211 / 0.092736 / 0.656285`.
- Stage sensitivity is condition-specific and non-monotonic. Aggregate and per-episode results are in `results/stage_sensitivity.csv` and `results/stage_sensitivity_by_episode.csv`.
- Translation cosine similarity is undefined for 16/800 rows under the documented `1e-8` near-zero norm rule; these values remain `NaN` and are counted rather than assigned an invented direction.
- Integrity report: `results/metrics_validation.json` (`PASS`). Concise interpretation: `results/phase8_9_findings.md`.

## DISCOVERED ISSUES / CONSTRAINTS

- Installed LeRobot 0.4.4 expects v3.0 datasets, while `HuggingFaceVLA/smol-libero` is published at v2.1. The dataset helper builds isolated local v3.0 views (episodes 0-2 for Phase 6 and 0-9 for Phase 7) and leaves source files untouched.
- Explicit sampler noise uses `max_action_dim=32`, even though the checkpoint action feature is 7-D. Supplying a `(1, 50, 7)` tensor would be incorrect; unpadding to 7-D occurs after flow sampling.
- The installed sampler does not validate a caller-provided noise tensor's shape/dtype/device. The repository API adds strict validation and clones the tensor before inference.
- Model-only MPS latency was measured with `PYTORCH_ENABLE_MPS_FALLBACK=0`; preprocessing and postprocessing are excluded from the reported latency.
- Some postprocessed gripper predictions are slightly outside `[-1, 1]`. They are finite official-unnormalizer outputs and were retained without clipping.
