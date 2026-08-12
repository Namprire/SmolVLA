# Final Findings

**Last reconciled with repository artifacts: 2026-08-11.**

## Research question

How much does changing only the natural-language instruction alter SmolVLA's
predicted action when vision, robot state, preprocessing, model checkpoint, and
explicit flow-matching noise are fixed?

## Evidence covered by this document

The final record now contains three empirical layers and one visualization
layer:

1. **Primary controlled experiment:** 10 episodes, 200 observations, four
   language conditions, and 800 stored predictions.
2. **Stage-spike diagnosis:** a model-free decomposition of the primary
   predictions, with no new inference.
3. **Systematic prompt robustness:** one Correct prompt plus 10 Paraphrase,
   10 Contradictory, and 10 Unrelated variants evaluated on 20 selected
   observations, producing 620 stored predictions.
4. **Bottom-up and Panda visualizations:** interpretations of stored actions.
   These are not closed-loop rollouts or task-success experiments.

## Method and action semantics

The pretrained `HuggingFaceVLA/smolvla_libero` checkpoint was evaluated on
locally available `HuggingFaceVLA/smol-libero` demonstrations. Within every
paired observation, the two camera images, robot state, preprocessing,
checkpoint, and explicit flow-noise tensor were identical; only the task string
changed. Primary inference ran on Apple M2 Max MPS.

The seven action dimensions are three relative Cartesian controller commands,
three axis-angle orientation-control components, and one gripper command. The
immutable CSV retains legacy `roll`, `pitch`, and `yaw` column names for
dimensions 3–5, but the installed LeRobot processor establishes that these are
axis-angle components. Translation, orientation control, and gripper values
have different meanings and scales, so the full 7-D Euclidean norm is a paired
comparison measure rather than a calibrated physical cost.

## Primary controlled result

Mean action distance from the Correct-language prediction:

| Altered condition | Translation | Axis-angle | Gripper | Arm-only 6-D | Full 7-D | Episode-cluster 95% CI for full 7-D |
|---|---:|---:|---:|---:|---:|---:|
| Paraphrase | 0.148509 | 0.014574 | 0.066276 | 0.149570 | **0.202755** | [0.172452, 0.231373] |
| Contradictory | 0.208660 | 0.019574 | 0.174892 | 0.210309 | **0.354315** | [0.297352, 0.414462] |
| Unrelated | 0.463489 | 0.042330 | 0.479377 | 0.467043 | **0.806369** | [0.761888, 0.855950] |

The aggregate ordering is therefore **Paraphrase < Contradictory < Unrelated**.
It survives removal of the gripper channel and is also present in medians,
quartiles, and 10%-per-tail trimmed means, so it is not created only by the
largest outliers.

The ordering is not universal. The strict full 7-D order holds in 125/200
observations and fails in 75/200. Pairwise, Paraphrase is closer to Correct than
Contradictory in 150/200 observations, Paraphrase is closer than Unrelated in
182/200, and Contradictory is closer than Unrelated in 171/200. These exceptions
prevent an observation-level rule or a general claim of language understanding.

## Agreement with the recorded expert action

Mean prediction-to-recorded-expert differences:

| Condition | Translation L2 | Axis-angle L2 | Gripper absolute | Full 7-D |
|---|---:|---:|---:|---:|
| Correct | **0.340237** | **0.070786** | **0.208461** | **0.511952** |
| Paraphrase | 0.363251 | 0.071559 | 0.228939 | 0.548614 |
| Contradictory | 0.414641 | 0.072025 | 0.276336 | 0.635421 |
| Unrelated | 0.615211 | 0.092736 | 0.656285 | 1.084084 |

Correct language has the greatest average agreement with the recorded expert
action on every reported component. This is agreement with one recorded
demonstration action, not objective accuracy and not proof that the
demonstration action is optimal.

## Task-progress result

Mean full 7-D language distance by normalized trajectory-progress bin:

| Progress | Paraphrase | Contradictory | Unrelated |
|---|---:|---:|---:|
| 0–20% | 0.185594 | 0.434305 | 0.382010 |
| 20–40% | 0.124691 | 0.305601 | **1.311807** |
| 40–60% | 0.117289 | 0.306255 | 0.677254 |
| 60–80% | 0.313960 | 0.326441 | 0.705375 |
| 80–100% | 0.275637 | 0.396262 | 0.979345 |

The patterns are condition-specific, non-monotonic, and heterogeneous across
episodes. Normalized progress is not asserted to be a semantic manipulation
stage. Figure intervals resample episodes rather than treating temporally
related frames as independent trials.

## New finding 1: why the Unrelated 20–40% bin spikes

The 1.311807 Unrelated peak has a mixed explanation: frequent gripper-regime
flips amplify an independent translation-control spike.

- The bin contains 39 observations spanning all 10 episodes.
- Mean arm-only 6-D distance is 0.698335; translation alone is 0.695722, while
  orientation-control distance is 0.046433.
- Mean absolute gripper difference is 0.881961.
- Correct and Unrelated fall on opposite sides of the OPEN/CLOSE threshold in
  17/39 observations (43.59%).
- Mean full distance is 2.151368 on flip observations and 0.663055 on non-flip
  observations. Excluding flips diagnostically lowers the bin mean by 49.45%,
  but all observations remain in the official result.
- Leaving out one episode at a time keeps the full mean between 1.242003 and
  1.375500, so the peak is not created by one episode.

Thus neither “only the gripper” nor “only the arm” explains the peak. The arm
term remains unusually large even when no regime flip occurs.

## New finding 2: the result survives multiple prompts, with weaker separation

The robustness extension used a manually audited 31-prompt bank on 20
deterministically selected observations from the same 10 episodes. Each altered
class has 10 wording variants, and each reported class value first averages
those variants within an observation.

| Prompt class | Translation mean | Arm-only 6-D mean | Full 7-D mean | Episode-cluster 95% CI for full 7-D |
|---|---:|---:|---:|---:|
| Paraphrase | 0.141431 | 0.142947 | **0.288395** | [0.166934, 0.442699] |
| Contradictory | 0.214911 | 0.217074 | **0.398144** | [0.256136, 0.532450] |
| Unrelated | 0.343745 | 0.346306 | **0.622293** | [0.432635, 0.836010] |

The aggregate Paraphrase < Contradictory < Unrelated order remains for
translation, arm-only 6-D, and full 7-D distance. However, the episode-cluster
intervals overlap, and the strict order holds in only 12/20 observations for
full 7-D and 15/20 for arm-only 6-D after prompt averaging. This supports a
class-level tendency, not clean separation among all possible wordings.

Specific wording matters materially:

- `P03` (“move the cream cheese box and the butter into the basket”) is the
  largest Paraphrase variant, with mean full distance 0.605634.
- `U01` (“open the top drawer of the cabinet”) is the strongest Unrelated
  variant, with mean full distance 0.950710.
- `U09` (“put the blue cup on the upper shelf”) is the weakest Unrelated
  variant, with mean full distance 0.385538.

The original Drawer prompt is therefore unusually strong, but the ten-prompt
Unrelated class mean remains above the other classes without relying on that
single sentence. In the selected robustness subset, the original sharp 20–40%
Unrelated peak is no longer the largest progress-bin mean after ten-prompt
averaging. This makes the original stage magnitude partly wording-dependent.

## New finding 3: language affects interpretable action primitives

The gripper-semantics audit examined all 2,612 locally available expert frames.
Expert commands are exactly -1 or +1, with 48 sign transitions. Measured finger
aperture decreases after -1→+1 and increases after +1→-1, supporting the
operational interpretation:

- `+1` = CLOSING command.
- `-1` = OPENING command.
- `+1 → -1` = opening-command onset.

SmolVLA predictions remain continuous and are not clipped; `< 0` and `>= 0`
are used only as operational OPENING/CLOSING regimes. Command onset does not
prove physical release.

Two selected prompt-bank observations provide bottom-up examples:

- At episode 3, frame 38, all 31 prompts share the OPENING regime, yet changing
  language changes the stored XYZ proposal. Mean translation distances from
  Correct are 0.203 Paraphrase, 0.245 Contradictory, and 0.461 Unrelated; the
  corresponding class-mean angular differences are 16.3°, 22.8°, and 42.6°.
- At episode 5, frame 230, Correct and all 10 Paraphrases are CLOSING, while
  3/10 Contradictory and 7/10 Unrelated variants are OPENING.

These are deliberately selected, traceable examples. They demonstrate possible
translation and gripper-regime effects, not their frequency in an independent
population of robot trials.

## Visualization evidence, not a new model result

The repository now includes an official Menagerie Franka Panda actuator test,
an articulated Panda reconstruction, synchronized four-condition local
branches, a one-object MuJoCo illustration, and four separate prompt-labeled
pick-and-place GIFs. The prompt-conditioned media load stored predictions; none
of these visualizations runs SmolVLA or LIBERO closed loop.

The Panda joint motion is position-only IK tracking of transformed stored expert
end-effector targets, not recovered LIBERO joint states. Local action arrows and
waypoints use fixed presentation scales. Object attachment and full
grasp-to-basket completion are deterministic visualization logic, not physical
task-success measurements.

The revised Unrelated full pick-and-place GIF displays the requested combined
sentence “put both the cream cheese box and the butter in the basket, and open
the drawer of the cabinet.” No inference was performed for that combined text;
its stored frame-81 proposal remains traceable to the original inferred source
prompt “open the top drawer of the cabinet.” The display edit must not be cited
as a new language-conditioned prediction.

## Consolidated interpretation

Under fixed observations, state, preprocessing, checkpoint, and explicit flow
noise, changing only language measurably changes SmolVLA's predicted next
action. Semantically equivalent paraphrases generally cause smaller changes
than contradictory instructions, and unrelated instructions generally cause
the largest changes. The tendency survives removal of the gripper channel and
averaging across multiple prompt variants, but it has many observation-level
exceptions, overlapping robustness intervals, and meaningful wording
dependence.

The evidence supports controlled language sensitivity in this checkpoint and
task. It does not establish human-like language understanding, physical task
success, closed-loop reliability, or safety.

## Limitations

- The primary experiment covers one pretrained checkpoint, one task/domain,
  10 locally available episodes, and 200 sampled observations.
- Observations within an episode are temporally related; they are not 200
  independent robot trials. Episode-cluster resampling only partially addresses
  this dependence.
- The robustness bank covers 20 deterministically selected observations and 31
  manually authored prompts. It is not a random or complete language sample.
- Ten prompt variants on the same observation are nested repeated measurements,
  not independent trials.
- The recorded expert action is a demonstration reference, not guaranteed
  optimal ground truth.
- Action components have different semantics and scales; the 7-D norm is not a
  calibrated physical cost.
- Sparse near-binary gripper flips materially affect some full-distance means,
  although the arm-only aggregate ordering survives.
- The immutable primary CSV does not contain per-condition image or state
  hashes. Their identity is supported by the audited inference logic and local
  source-frame checks rather than those CSV columns alone.
- No p-value battery was performed, and prompt-specific extremes should not be
  generalized to a population of possible instructions.
- MuJoCo and Panda media are illustrative and do not measure SmolVLA task
  success.

## Primary evidence files

- [vla_predictions.csv](vla_predictions.csv) — immutable 800-row primary
  prediction table.
- [final_independent_validation.md](final_independent_validation.md) —
  independent primary-result recomputation.
- [spike_findings.md](spike_analysis/spike_findings.md) — stage-spike
  decomposition.
- [prompt_bank.md](prompt_robustness/prompt_bank.md) and
  [predictions.csv](prompt_robustness/predictions.csv) — exact 31-prompt
  definitions and 620 robustness predictions.
- [prompt-robustness findings](prompt_robustness/findings.md) — robustness
  analysis and exceptions.
- [bottom-up findings](bottom_up_demo/FINDINGS.md) — translation/gripper
  interpretation and simulation scope.
- [full_pick_place_validation.json](mujoco_panda_prompt_demo/full_pick_place_validation.json)
  — current GIF display/source-prompt provenance and media checks.
- [FINAL_AUDIT.md](FINAL_AUDIT.md) — primary forensic audit and caveats.
