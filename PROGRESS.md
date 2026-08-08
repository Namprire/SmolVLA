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

## CURRENT

- Phases 1-6 are complete and verified. Work is intentionally stopped after Phase 6 for result inspection.

## NEXT

- Inspect the Phase 3-6 artifacts and methodology before authorizing any later phase.
- Do not start the full batch experiment, release guard, stochasticity analysis, plots, or MuJoCo without explicit approval.

## BLOCKERS

- None currently. The dataset format mismatch is handled locally with the converter bundled in LeRobot 0.4.4.

## PHASE 3-6 RESULTS

- Determinism file: `results/determinism_check.csv` (5 rows; maximum absolute repeated-output difference `0`).
- Exact language/API audit: `results/api_inspection.md` and `results/language_conditions.json`.
- Single-frame comparison: `results/language_single_frame.csv` (4 rows).
- Pilot predictions: `results/vla_predictions_pilot.csv` (36 rows = 9 observations x 4 conditions).
- Pilot mean full 7-D distance from Correct: paraphrase `0.09006596`, contradictory `0.12267754`, unrelated `0.44861949`; all three differed from Correct in all 9 observations at a `1e-8` threshold.
- These are controlled-pipeline and small-pilot diagnostics only, not scientific conclusions.

## DISCOVERED ISSUES / CONSTRAINTS

- Installed LeRobot 0.4.4 expects v3.0 datasets, while `HuggingFaceVLA/smol-libero` is published at v2.1. The Phase 6 helper builds an isolated local v3.0 view of only episodes 0-2 and leaves source files untouched.
- Explicit sampler noise uses `max_action_dim=32`, even though the checkpoint action feature is 7-D. Supplying a `(1, 50, 7)` tensor would be incorrect; unpadding to 7-D occurs after flow sampling.
- The installed sampler does not validate a caller-provided noise tensor's shape/dtype/device. The repository API adds strict validation and clones the tensor before inference.
- Model-only MPS latency was measured with `PYTORCH_ENABLE_MPS_FALLBACK=0`; preprocessing and postprocessing are excluded from the reported latency.
- Some postprocessed gripper predictions are slightly outside `[-1, 1]`. They are finite official-unnormalizer outputs and were retained without clipping.
