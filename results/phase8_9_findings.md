# Phase 8-9 findings

These results describe this checkpoint and the 200 controlled observations only. Changing the language while images, robot state, preprocessing, checkpoint, and explicit flow noise were held fixed produced measurable changes in the predicted action.

## Paired language effects

1. Paraphrase was closer to Correct than Contradictory in 150/200 observations (75.0%) and in 9/10 episode means. Their mean full 7-D distances were 0.2028 and 0.3543, respectively.

2. Paraphrase was closer to Correct than Unrelated in 182/200 observations (91.0%) and in 10/10 episode means. The Unrelated mean full 7-D distance was 0.8064.

3. Translation was the largest typical language-effect component: median translation distances were 0.1194, 0.1635, and 0.4329, while median gripper differences were 0.0045, 0.0048, and 0.0120. Sparse large gripper changes make the mean gripper difference for Unrelated (0.4794) slightly exceed its mean translation distance (0.4635). Axis-angle differences were smallest numerically. Because these components have different semantics/scales, this is not a calibrated cross-component sensitivity ranking.

## Task stage and episode structure

4. Mean full-action language distance varied across the five task-progress bins. The stage means were:

- Paraphrase: 0–20%=0.1856, 20–40%=0.1247, 40–60%=0.1173, 60–80%=0.3140, 80–100%=0.2756.
- Contradictory: 0–20%=0.4343, 20–40%=0.3056, 40–60%=0.3063, 60–80%=0.3264, 80–100%=0.3963.
- Unrelated: 0–20%=0.3820, 20–40%=1.3118, 40–60%=0.6773, 60–80%=0.7054, 80–100%=0.9793.

The pattern is descriptive rather than uniformly monotonic across all conditions; the plotted intervals use episodes—not individual frames—as bootstrap clusters.
Per-episode peak stages were distributed as follows: Paraphrase {'80–100%': 4, '60–80%': 4, '0–20%': 2}; Contradictory {'0–20%': 3, '20–40%': 2, '60–80%': 2, '80–100%': 2, '40–60%': 1}; Unrelated {'20–40%': 5, '80–100%': 4, '40–60%': 1}.

5. Leave-one-episode-out means were checked for every comparison. The largest episode and single-observation effects are reported below. The narrow leave-one-out aggregate ranges indicate that no overall condition mean is explained by only one episode, although stage-specific patterns are heterogeneous:

- Paraphrase: highest episode mean was episode 6 (0.2668; median episode mean 0.2223); leave-one-episode-out aggregate range [0.1956, 0.2103]. The largest observation was 2.0854 at episode 0, frame 241 (progress 0.938).
- Contradictory: highest episode mean was episode 9 (0.5578; median episode mean 0.3615); leave-one-episode-out aggregate range [0.3317, 0.3722]. The largest observation was 2.1789 at episode 2, frame 205 (progress 0.837).
- Unrelated: highest episode mean was episode 4 (0.9720; median episode mean 0.7911); leave-one-episode-out aggregate range [0.7880, 0.8204]. The largest observation was 2.3066 at episode 5, frame 230 (progress 0.827).

## Agreement with the recorded expert action

6. Mean translation / axis-angle / gripper errors by condition were:

- Correct: 0.3402 / 0.0708 / 0.2085; mean translation cosine similarity 0.7096.
- Paraphrase: 0.3633 / 0.0716 / 0.2289; mean translation cosine similarity 0.6943.
- Contradictory: 0.4146 / 0.0720 / 0.2763; mean translation cosine similarity 0.6086.
- Unrelated: 0.6152 / 0.0927 / 0.6563; mean translation cosine similarity 0.3526.

These are agreement measurements against the recorded expert action, not accuracy or objective correctness. Lower errors only mean closer agreement with that recorded demonstration action. Gripper-error means are strongly right-skewed, so the medians and quartiles in summary.csv should be considered alongside the means.

## Latency diagnostic

Overall inference latency was 541.1 ms mean and 539.2 ms median. Condition means were {'Correct': 556.1235181917436, 'Paraphrase': 525.576957880985, 'Contradictory': 541.5318028908223, 'Unrelated': 541.0041716729756}. Conditions were always evaluated in a fixed order, so these differences are order/cache-confounded and are not interpreted as a language effect.

## Episode-aware uncertainty and exceptions

- Paraphrase: mean 0.2028, episode-cluster 95% CI [0.1725, 0.2314].
- Contradictory: mean 0.3543, episode-cluster 95% CI [0.2974, 0.4145].
- Unrelated: mean 0.8064, episode-cluster 95% CI [0.7619, 0.8560].

7. Large (>1 native unit) paired gripper changes occurred in 6/200 Paraphrase, 17/200 Contradictory, and 47/200 Unrelated comparisons. These sparse events create strong mean/median separation and some stage spikes; their semantics are deliberately deferred to Phase 10. Translation cosine similarity was undefined in 16/800 cases because the recorded expert or prediction translation norm was below 1e-08; no direction was invented. Paired observation-level ordering was not universal, and the maximum-distance rows and episode heterogeneity above prevent reducing the result to a strict Paraphrase < Contradictory < Unrelated rule for every observation.

No p-value battery was performed, and no claim is made that the model generally ‘understood’ the language.
