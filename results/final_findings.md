# Research Question

How much does changing only the natural-language instruction alter SmolVLA's predicted action when vision, robot state, preprocessing, model, and flow-matching noise are fixed?

# Method

We evaluated the pretrained `HuggingFaceVLA/smolvla_libero` checkpoint on locally available `HuggingFaceVLA/smol-libero` demonstrations. The controlled evaluation used 10 episodes, 200 observations, four language conditions, and 800 predictions. Within every paired observation, the two camera images, robot state, preprocessing, checkpoint, and explicit flow-noise tensor were identical; only the task string changed. Inference ran on Apple M2 Max MPS.

# Main Results

Mean full 7-D action distance from Correct language, with episode-cluster bootstrap 95% confidence intervals:

- Paraphrase: 0.202755 [0.172452, 0.231373].
- Contradictory: 0.354315 [0.297352, 0.414462].
- Unrelated: 0.806369 [0.761888, 0.855950].

Mean agreement errors against the recorded expert action are reported as translation / axis-angle orientation / gripper absolute error:

- Correct: 0.3402 / 0.0708 / 0.2085.
- Paraphrase: 0.3633 / 0.0716 / 0.2289.
- Contradictory: 0.4146 / 0.0720 / 0.2763.
- Unrelated: 0.6152 / 0.0927 / 0.6563.

Correct-language predictions had the smallest mean error on all three reported components, while Unrelated had the largest. These are agreement measurements against a recorded demonstration, not accuracy or objective correctness.

# Task-Stage Result

Language sensitivity varied descriptively with normalized task progress, and the pattern was condition-specific and non-monotonic:

- Paraphrase: 0–20%=0.1856, 20–40%=0.1247, 40–60%=0.1173, 60–80%=0.3140, 80–100%=0.2756.
- Contradictory: 0–20%=0.4343, 20–40%=0.3056, 40–60%=0.3063, 60–80%=0.3264, 80–100%=0.3963.
- Unrelated: 0–20%=0.3820, 20–40%=1.3118, 40–60%=0.6773, 60–80%=0.7054, 80–100%=0.9793.

Stage intervals use episodes, rather than individual frames, as bootstrap clusters.

# Gripper Semantics

- `+1` is a closing command.
- `-1` is an opening command.
- `+1 -> -1` marks opening-command onset.

Command onset is not equivalent to guaranteed physical release. The audit used temporal finger-state response and two-camera trajectory inspection; retry/regrasp sequences demonstrate why command sign alone is insufficient.

# Interpretation

Under controlled observations and identical flow noise, semantically equivalent paraphrases generally produced smaller action changes than contradictory or unrelated instructions. Unrelated instructions produced the largest average changes in the model's predicted action. Language sensitivity was not constant across trajectory progress. Correct-language predictions showed greater average agreement with the recorded expert demonstration than altered-language conditions.

These statements describe this checkpoint, task, and offline sample. They do not establish human-like language understanding, closed-loop success, or safety.

# Limitations

- Only 10 locally available episodes were evaluated.
- Observations within episodes are temporally related; episode-cluster bootstrap intervals partially address, but do not eliminate, this dependence.
- The expert action is a recorded reference demonstration, not a ground-truth optimal action.
- Action dimensions have different physical meanings and scales; the full 7-D norm is not physically calibrated across components.
- Offline frame-level evaluation does not measure closed-loop task success.
- Only one pretrained VLA checkpoint was studied.
- One task/domain was studied.
- Language conditions were manually designed.
- Physical release cannot be inferred from command sign alone.
