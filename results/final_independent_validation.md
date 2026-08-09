# Final independent validation

**Result: PASS**

This audit recomputed every result directly from `results/vla_predictions.csv` without importing project metric or viewer helpers. Stored dimensions 3–5 were treated as axis-angle components.

## Central results

| Comparison from Correct prediction | Translation mean | Axis-angle mean | Gripper mean | Arm-only 6-D mean | Full 7-D mean | Q25 | Median | Q75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paraphrase | 0.148509 | 0.014574 | 0.066276 | 0.149570 | 0.202755 | 0.091890 | 0.122180 | 0.199275 |
| Contradictory | 0.208660 | 0.019574 | 0.174892 | 0.210309 | 0.354315 | 0.100998 | 0.164170 | 0.301004 |
| Unrelated | 0.463489 | 0.042330 | 0.479377 | 0.467043 | 0.806369 | 0.218061 | 0.479227 | 1.082173 |

## Agreement with recorded expert action

| Condition | Translation L2 | Axis-angle L2 | Gripper absolute | Full 7-D | 7-D MAE |
|---|---:|---:|---:|---:|---:|
| Correct | 0.340237 | 0.070786 | 0.208461 | 0.511952 | 0.116623 |
| Paraphrase | 0.363251 | 0.071559 | 0.228939 | 0.548614 | 0.124772 |
| Contradictory | 0.414641 | 0.072025 | 0.276336 | 0.635421 | 0.142823 |
| Unrelated | 0.615211 | 0.092736 | 0.656285 | 1.084084 | 0.243027 |

## Bootstrap and ordering

- Paraphrase: episode-cluster mean 0.202755, 95% CI [0.172452, 0.231373]. Duplicate sampled episodes were retained with multiplicity.
- Contradictory: episode-cluster mean 0.354315, 95% CI [0.297352, 0.414462]. Duplicate sampled episodes were retained with multiplicity.
- Unrelated: episode-cluster mean 0.806369, 95% CI [0.761888, 0.855950]. Duplicate sampled episodes were retained with multiplicity.

The strict observation-level ordering Paraphrase < Contradictory < Unrelated holds for 125/200 observations and fails for 75/200. Paraphrase is closer than Contradictory in 150/200 and closer than Unrelated in 182/200.

## Robustness and interpretation

Excluding gripper, arm-only 6-D means are Paraphrase 0.149570, Contradictory 0.210309, and Unrelated 0.467043. The aggregate ordering survives. The strict arm-only observation-level ordering fails in 75/200 observations.

Large near-binary gripper flips can dominate some individual full-norm distances, so translation, axis-angle, gripper, arm-only, and full distances are reported separately. Their raw magnitudes are not physically interchangeable.

Task-stage bins are paired Correct-vs-altered comparisons, include progress 1.0 in the final bin, and show heterogeneous non-monotonic episode patterns. The 200 observations come from ten trajectories and are temporally related, not 200 independent robot trials.

The condition ordering is also present in medians, quartiles, and 10%-per-tail trimmed means, so the aggregate result is not created only by the largest outliers. Outliers and near-binary gripper flips do materially widen the mean/median gap, especially for Contradictory and Unrelated.

## Task-stage means

| Stage | Paraphrase | Contradictory | Unrelated |
|---|---:|---:|---:|
| 0–20% | 0.185594 | 0.434305 | 0.382010 |
| 20–40% | 0.124691 | 0.305601 | 1.311807 |
| 40–60% | 0.117289 | 0.306255 | 0.677254 |
| 60–80% | 0.313960 | 0.326441 | 0.705375 |
| 80–100% | 0.275637 | 0.396262 | 0.979345 |

The CSV itself does not store per-condition robot-state vectors or image hashes. Equality of state and images therefore requires the separate source/viewer asset audit; it cannot be independently proven from CSV columns alone.

Curated example calculations and documented exclusion-aware ranks are in `results/demo_example_audit.csv`.
