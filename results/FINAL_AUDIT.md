FINAL VERDICT: PASS WITH CAVEATS

# Final forensic validation and code-rot audit

## Bottom line: do these results actually make sense?

Yes. Based only on the stored experiment, the evidence supports this cautious statement:

> Under identical visual observations, robot state, preprocessing, model, and explicit flow noise, changing only the language instruction produced measurable changes in SmolVLA's predicted action. Semantically equivalent paraphrases generally caused smaller changes than contradictory instructions, while unrelated instructions caused the largest average changes.

No part of that sentence is contradicted by the stored results. “Generally” and “average” are essential: the expected strict ordering fails at 75/200 observations, and this offline experiment does not establish language understanding, objective action accuracy, closed-loop task success, safety, or generalization beyond this checkpoint/task/sample.

## 1. Raw-result integrity

The six immutable evidence files remained byte-for-byte unchanged. In particular, `results/vla_predictions.csv` retained SHA-256 `ab4984aeea52a8ca60c94a5cb19a15d69fd27707f5c5734ddd8bec680f376dc4`. All before/after hashes are in `final_audit_hashes_before.json` and `post_cleanup_regression.json`.

The primary CSV has exactly 800 rows, 200 unique observations, four exact conditions per observation, episodes 0–9, no duplicate observation-condition keys, finite 7-D predictions and expert actions, consistent expert actions/frame identifiers/progress/noise seed/noise hash within each four-row group, and the exact four instruction strings. Task progress equals `frame_index / max_frame_index`.

Caveat: the CSV does not store image hashes or a per-condition robot-state vector. Their identity cannot be established from CSV columns alone. It is corroborated by the inference source, which reuses one dataset frame and explicit noise tensor for the four condition calls, copies mutable preprocessing inputs, and asserts that the source frame is unchanged; curated image identity was also checked pixel-for-pixel against the local source parquet.

## 2. Independent central-result recomputation

The independent validator imports no project metric or viewer helper and uses pandas/numpy calculations directly over the immutable CSV. Stored dimensions 3–5 are treated as three axis-angle components despite the legacy CSV header names.

Mean full 7-D distance from the Correct prediction:

| Altered condition | Mean | Q25 | Median | Q75 | 10%-tail-trimmed mean |
|---|---:|---:|---:|---:|---:|
| Paraphrase | 0.202755 | 0.091890 | 0.122180 | 0.199275 | 0.138845 |
| Contradictory | 0.354315 | 0.100998 | 0.164170 | 0.301004 | 0.206953 |
| Unrelated | 0.806369 | 0.218061 | 0.479227 | 1.082173 | 0.723061 |

The published approximate means reproduce within four-decimal rounding. The condition order is present in means, quartiles, medians, and trimmed means, so it is not created only by the largest observations. Outliers still materially widen the mean/median gap.

Mean prediction-to-recorded-expert differences also reproduce:

| Condition | Translation L2 | Axis-angle L2 | Gripper absolute |
|---|---:|---:|---:|
| Correct | 0.340237 | 0.070786 | 0.208461 |
| Paraphrase | 0.363251 | 0.071559 | 0.228939 |
| Contradictory | 0.414641 | 0.072025 | 0.276336 |
| Unrelated | 0.615211 | 0.092736 | 0.656285 |

Correct has the greatest mean agreement with the recorded expert action for all three components and also for full 7-D distance and 7-D MAE. These are reference-agreement measurements, not objective correctness or “accuracy.”

## 3. Observation-level ordering and counterexamples

- Paraphrase is closer to Correct than Contradictory in 150/200 observations.
- Paraphrase is closer to Correct than Unrelated in 182/200 observations.
- Contradictory is closer to Correct than Unrelated in 171/200 observations.
- The complete strict order Paraphrase < Contradictory < Unrelated holds in 125/200 and fails in 75/200 (37.5%).

The narrative must therefore remain aggregate. Example 5 is a genuine strong counterexample: Paraphrase distance 2.064586 versus Contradictory 0.064779, a difference of 1.999807. Example 1 also shows that Unrelated can be closer than Paraphrase for an individual observation. These negative cases are retained rather than hidden.

## 4. Component and gripper robustness

Mean language-induced differences from Correct, reported separately because the components have different units/scales:

| Condition | Translation | Axis-angle | Gripper | Full 7-D | Arm-only 6-D |
|---|---:|---:|---:|---:|---:|
| Paraphrase | 0.148509 | 0.014574 | 0.066276 | 0.202755 | 0.149570 |
| Contradictory | 0.208660 | 0.019574 | 0.174892 | 0.354315 | 0.210309 |
| Unrelated | 0.463489 | 0.042330 | 0.479377 | 0.806369 | 0.467043 |

Near-binary gripper flips dominate some individual Euclidean norms. Gripper accounts for more than half of squared full-distance magnitude in 6 Paraphrase, 17 Contradictory, and 47 Unrelated observations; it exceeds 90% in 6, 15, and 18 respectively. Mean full-minus-arm-only distance is 0.053185, 0.144007, and 0.339326.

The conclusion survives exclusion of gripper: arm-only 6-D means remain Paraphrase 0.149570 < Contradictory 0.210309 < Unrelated 0.467043. Thus the main average ordering is not solely a gripper artifact. The strict arm-only order still fails in 75/200 observations.

## 5. Bootstrap validation

The existing implementation is valid. It samples ten `episode_id` values with replacement and concatenates the complete per-episode arrays once for every sampled occurrence. Duplicate sampled episodes therefore contribute multiple times and are not deduplicated. It does not resample individual frames.

An independent 2,000-replicate episode-cluster bootstrap with the documented condition/metric seeds reproduces:

- Paraphrase: mean 0.202755, 95% percentile interval [0.172452, 0.231373].
- Contradictory: mean 0.354315, interval [0.297352, 0.414462].
- Unrelated: mean 0.806369, interval [0.761888, 0.855950].

Only ten clusters are available, so these intervals should not be overinterpreted as broad population guarantees.

## 6. Task-stage validation

Progress bins are left-closed/right-open at 0.0, 0.2, 0.4, 0.6, and 0.8, with progress 1.0 explicitly placed in 80–100%. All 30 altered-condition rows at progress 1.0 are in the final bin. Stage metrics are genuine paired altered-prediction versus Correct-prediction distances.

| Stage | Paraphrase | Contradictory | Unrelated |
|---|---:|---:|---:|
| 0–20% | 0.185594 | 0.434305 | 0.382010 |
| 20–40% | 0.124691 | 0.305601 | 1.311807 |
| 40–60% | 0.117289 | 0.306255 | 0.677254 |
| 60–80% | 0.313960 | 0.326441 | 0.705375 |
| 80–100% | 0.275637 | 0.396262 | 0.979345 |

No monotonic interpretation is warranted. Per-episode analysis shows heterogeneous peak stages. Individual episodes amplify some spikes—especially late Paraphrase sensitivity—but leave-one-episode-out checks show that the main aggregate stage means are not wholly created by one episode. The large Unrelated 20–40% mean remains 1.242–1.375 under leave-one-episode-out recomputation.

## 7. Offline viewer validation

The viewer is correct after a label clarification. For every curated example:

- both packaged/displayed camera arrays are pixel-identical to the exact episode/frame in the local source parquet;
- episode/frame/global/progress metadata and all four CSV rows match;
- all four rows share the same stored noise identifier and expert action;
- the Correct row, exact condition instruction, displayed expert/prediction vectors, rounded table values, and recomputed metrics match;
- selection explanations and exclusion-aware ranks are mathematically true.

The UI now explicitly separates “Selected prediction vs recorded expert” from “Language-induced difference vs Correct prediction.” Its smoke test loads all five examples and switches all four conditions without changing camera pixels.

## 8. Curated-example validation

All five claims are valid over all 200 observations:

1. Example 1 is rank 1/200 for smallest Paraphrase distance (0.031779).
2. Example 2 is rank 1/200 for largest Contradictory distance (2.178876).
3. Example 3 is the largest unused Unrelated distance (2.306573) after excluding examples 1–2; it is also rank 1 before exclusions.
4. Example 4 has mean altered-condition distance 1.407023 at progress 0.888. It is rank 3 among all progress >= 0.8 candidates and rank 1 after excluding examples 1–3.
5. Example 5 maximizes positive Paraphrase-minus-Contradictory difference (1.999807) and is rank 1 even before exclusions.

The selection policy documents duplicate-observation exclusion and deliberately includes a hypothesis-counterexample.

## 9. Gripper semantics

The interpretation remains defensible. Across all 2,612 locally available expert frames, the gripper command is exactly -1 or +1. There are 48 sign transitions. Measured finger aperture decreases after -1→+1 (mean 0.075059 at onset to 0.039398 at +10 frames) and increases after +1→-1 (0.036676 to 0.069481), supporting +1 = closing and -1 = opening.

Normal and retry/regrasp two-camera transition sheets support +1→-1 as opening-command onset. They also show why opening-command onset is not guaranteed successful physical release. No release guard or release detector was added.

## 10. Claims and wording audit

Reader-facing text consistently uses “agreement with recorded expert action,” “recorded expert demonstration,” “language-conditioned/action difference,” and “axis-angle components.” It states that this is an offline evaluation of one pretrained checkpoint on ten locally available episodes and 200 sampled observations, that observations within episodes are temporally related, and that no closed-loop task success or safety is established.

Corrections made:

- viewer metric-family labels were made explicit;
- reader-facing “rotation” shorthand was replaced with “axis-angle” where dimensions 3–5 are described;
- comments now explain the immutable CSV's legacy roll/pitch/yaw headers;
- three machine-specific installed-source paths were replaced with portable `site-packages/...` labels;
- the demo smoke test now covers five examples × four conditions.

No numeric value or raw result was changed.

## 11. Code-rot cleanup

Removed:

- generated `__pycache__`/`.pyc` files;
- obsolete `results/.gitkeep`;
- unused `ACTION_COLUMNS` and `ACTION_DISPLAY_NAMES` constants;
- unused `ACTION_LABELS`, `EXPERT_COLUMNS`, and `PRED_COLUMNS` imports.

Intentionally preserved despite appearing old, redundant, or large:

- pilot, single-frame, determinism, stage, resume, and run-log records, because they preserve experiment provenance;
- source inference scripts, because they document reproducibility even though they must not be rerun for this audit;
- both Phase 8–9 and presentation figure sets, because they are distinct outputs from separate documented pipelines;
- 20 gripper transition contact sheets (about 40 MB), because they are the visual semantic-audit evidence, not stale screenshots;
- the module-level prediction convenience API and historical Phase 10 discussion, because their original compatibility/history role is uncertain.

No byte-identical tracked output duplicates, dead CLI options, `.DS_Store`, temporary debug files, abandoned tests, or active machine-specific paths remain. Detailed classification is in `code_rot_audit.md`.

## 12. Remaining methodological limitations

1. This is offline, frame-level prediction analysis, not closed-loop task-success evaluation.
2. Only one pretrained checkpoint, one task/domain, ten locally available episodes, and manually designed language conditions were studied.
3. Twenty observations sampled along each episode are temporally related. They are not 200 independent robot trials; episode-cluster resampling only partially addresses dependence.
4. Translation, axis-angle, and gripper values have different physical meaning/scales. The 7-D Euclidean norm is not a calibrated physical cost.
5. Sparse near-binary gripper flips materially affect full-distance means, although the arm-only robustness ordering survives.
6. The CSV lacks per-condition state/image hashes, so that part of the control claim depends on audited source logic and local source-frame corroboration.
7. The expert demonstration is a recorded reference, not guaranteed optimal ground truth.
8. Observation-level ordering has many exceptions, and stage patterns are non-monotonic and episode-heterogeneous.

## 13. Post-cleanup regression

All model-free regressions pass: independent validation, existing metrics regeneration, four final PNG/PDF figure pairs, exact viewer audit, five-example/four-condition smoke traversal, curated-example validation, gripper semantics regeneration, static import scan, and `git diff --check`. No model inference, model/data download, training, fine-tuning, simulator work, release guard, or new experiment was performed.
