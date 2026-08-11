# Phase B findings: systematic multi-prompt robustness

## Observed result

Yes at the aggregate-mean level, but not as a universal observation-level rule. The ordering Paraphrase < Contradictory < Unrelated holds for both full 7-D and arm-only 6-D means; the episode-cluster intervals overlap.

Each value below first averages the 10 prompt variants within an observation; only then are the 20 observation-level values summarized. Prompts on the same frame are not counted as independent robot trials.

- Paraphrase: full 7-D mean 0.288395, episode-cluster 95% interval [0.166934, 0.442699]; arm-only mean 0.142947, interval [0.105710, 0.183088].
- Contradictory: full 7-D mean 0.398144, episode-cluster 95% interval [0.256136, 0.532450]; arm-only mean 0.217074, interval [0.158258, 0.288770].
- Unrelated: full 7-D mean 0.622293, episode-cluster 95% interval [0.432635, 0.836010]; arm-only mean 0.346306, interval [0.221281, 0.510482].

The strict class ordering holds after prompt averaging in 12/20 observations for full 7-D distance and 15/20 for arm-only 6-D distance.

## Answers to the seven analysis questions

1. **Paraphrase versus Contradictory:** Yes at the aggregate full-7D mean: 0.288395 versus 0.398144.
2. **Contradictory versus Unrelated:** Yes at the aggregate full-7D mean: 0.398144 versus 0.622293.
3. **Observation-level ordering:** It holds in 12/20 observations for full 7-D and 15/20 for arm-only 6-D after averaging the 10 prompts in each class.
4. **Large-change paraphrases:** `P03` is the largest Paraphrase at 0.605634; this exceeds the Contradictory class mean and shows that semantic equivalence does not guarantee a small perturbation for every wording.
5. **Small-change unrelated prompts:** `U09` is the smallest Unrelated full-7D prompt at 0.385538, slightly below the Contradictory class mean. Its arm-only mean is 0.330238, so the small full-norm comparison does not imply negligible arm change.
6. **Arm-only result:** The aggregate ordering also holds without the gripper: 0.142947 < 0.217074 < 0.346306. It holds observation-by-observation in 15/20 cases.
7. **20–40% Unrelated effect:** The original sharp global peak does not survive ten-prompt averaging in this selected subset. The bin has mean full/arm distances 0.586713/0.310455 and ranks 2/5 for full and 3/5 for arm. A descriptive full-7D elevation remains, but larger values occur elsewhere and only three selected observations occupy this bin.

Individual-prompt variability is nonzero:

- Paraphrase: minimum `P10` = 0.126193, maximum `P03` = 0.605634, SD across prompt means = 0.136305.
- Contradictory: minimum `C05` = 0.266880, maximum `C01` = 0.555133, SD = 0.077903.
- Unrelated: minimum `U09` = 0.385538, maximum `U01` = 0.950710, SD = 0.178157.

For the selected subset, the multi-prompt Unrelated mean in the 20–40% bin is 0.586713 full 7-D and 0.310455 arm-only 6-D across 3 observations. The largest corresponding means outside that bin are 1.028958 and 0.514035. Therefore, 20–40% is not the largest descriptive bin for both metrics. This is a small selected-subset comparison, not a new task-stage claim.

## Interpretation

The class means test whether the original result was peculiar to one sentence. Aggregate ordering across ten systematically varied sentences supports class-level language sensitivity only where the reported ordering actually holds. The individual-prompt table shows how much specific wording matters and identifies high-change paraphrases and low-change unrelated prompts without treating them as independent trials.

The retained original Drawer prompt `U01` is rank 1/10 within Unrelated, with mean full distance 0.950710. It is therefore unusually strong in this bank. The class-level Unrelated mean nevertheless remains above the other classes after averaging U01–U10, so the aggregate result is not solely a U01 effect; the original single-prompt magnitude was partly wording-dependent.

## Limitations

- Only 20 deterministically selected observations from 10 episodes were evaluated.
- One pretrained model, one demonstrated task, and one domain were studied.
- The bank was manually and systematically authored; it is not a complete or random sample of natural-language instructions.
- Ten prompts on the same observation are nested repeated measurements, not ten independent robot trials.
- This remains offline next-action evaluation, not closed-loop task execution or task-success measurement.
- The progress analysis is descriptive and subsampled; normalized trajectory progress is not a semantic manipulation stage.
- Prompt-specific extremes in this small bank should not be generalized to a population of possible phrasings.
