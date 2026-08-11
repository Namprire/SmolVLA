# Phase A spike findings

## Question

Why does the red Unrelated curve rise to approximately 1.31 in the 20–40% normalized trajectory-progress bin?

## Direct answer

The spike is **mixed: gripper-regime flips strongly amplify an independent arm-control action spike**. It is not explained by either component alone.

- The bin contains 39 observations from 10 unique episodes.
- Mean/median full 7-D distance: 1.311807 / 1.065382.
- Mean arm-only 6-D distance: 0.698335.
- Mean translation/orientation-control distance: 0.695722 / 0.046433.
- Mean absolute gripper difference: 0.881961.
- 17/39 observations (43.59%) switch gripper regime between Correct and Unrelated.

On the 17 flip observations, mean full 7-D distance is 2.151368; on the 22 non-flip observations it is 0.663055. The diagnostic exclusion of flip observations therefore lowers the mean from 1.311807 to 0.663055, a decrease of 0.648752 (49.45%). This exclusion is diagnostic only; all observations remain in the official result. Flip observations are 43.59% of the bin but contribute 71.49% of the summed full-distance values.

The arm term also spikes. The 20–40% arm-only mean (0.698335) is 58.67% above the largest Unrelated arm-only mean in any other progress bin (0.440106). Translation (0.695722) accounts for nearly all of that arm-only magnitude; the orientation-control mean is 0.046433. Thus the approximately 1.31 full-norm peak combines unusually large translation-control changes with frequent approximately two-unit gripper sign changes.

## Gripper operational rule

The stored SmolVLA gripper predictions are continuous and are never clipped for this analysis. Based on the existing expert-semantics audit, a value `< 0` is called the opening regime and a value `>= 0` the closing regime. A flip means Correct and Unrelated fall on opposite sides of that shown threshold. This is a command-regime comparison, not evidence of physical release.

## Episode-level check

All 10 episodes contribute to the bin. Episode-mean full distances range from 0.754494 to 2.149458; arm-only episode means range from 0.491705 to 0.828885. Leaving out one episode at a time leaves the aggregate full mean between 1.242003 and 1.375500. The peak is therefore not created by one episode, although flip-heavy episodes have the largest full 7-D means. Episode 3 has no flip in this bin yet still has mean arm-only/full distances of 0.754030/0.754494, reinforcing the independent arm contribution.

The table below reports 10 episode summaries, not independent robot trials:

| episode_id | n_observations_in_bin | mean_full_7d | mean_arm_6d | mean_translation | mean_orientation | mean_gripper_difference | gripper_flip_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4 | 1.734188 | 0.796447 | 0.793614 | 0.055925 | 1.512298 | 3 |
| 1 | 4 | 1.010397 | 0.569862 | 0.564049 | 0.057084 | 0.514316 | 1 |
| 2 | 4 | 1.513203 | 0.797205 | 0.795128 | 0.051488 | 1.002812 | 2 |
| 3 | 4 | 0.754494 | 0.754030 | 0.750624 | 0.064780 | 0.021011 | 0 |
| 4 | 4 | 1.430793 | 0.656268 | 0.655371 | 0.030850 | 1.002097 | 2 |
| 5 | 4 | 1.171678 | 0.801634 | 0.799448 | 0.056066 | 0.498019 | 1 |
| 6 | 4 | 1.794457 | 0.828885 | 0.828473 | 0.023533 | 1.497527 | 3 |
| 7 | 3 | 2.149458 | 0.754013 | 0.752767 | 0.043282 | 2.011584 | 3 |
| 8 | 4 | 0.853403 | 0.491705 | 0.488811 | 0.040410 | 0.524668 | 1 |
| 9 | 4 | 0.915410 | 0.547226 | 0.543193 | 0.040122 | 0.517683 | 1 |

## Reproducibility and scope

- Source: `results/vla_predictions.csv`
- Source SHA-256: `ab4984aeea52a8ca60c94a5cb19a15d69fd27707f5c5734ddd8bec680f376dc4`
- Bins: `[0.0,0.2)`, `[0.2,0.4)`, `[0.4,0.6)`, `[0.6,0.8)`, and `[0.8,1.0]`.
- Episode-cluster figure intervals: 5000 bootstrap replicates, base seed 20260810.
- This is an offline, frame-level paired action analysis. Normalized progress is not asserted to be a semantic manipulation stage.
