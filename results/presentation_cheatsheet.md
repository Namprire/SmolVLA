# Presentation Cheat Sheet

## The problem

Vision-Language-Action models connect language, images, robot state, and actions. Their behavior under instruction variation matters for reliability.

## The experiment

- Pretrained `HuggingFaceVLA/smolvla_libero`; no training or fine-tuning.
- 200 observations from 10 locally available `smol-libero` episodes.
- Four conditions and 800 stored predictions: Correct, Paraphrase, Contradictory, Unrelated.
- Within each comparison, camera images, robot state, preprocessing, model, and explicit flow noise were fixed. Only the instruction changed.
- Inference ran on Apple M2 Max MPS.

## The big result

Mean full 7-D distance from Correct:

- Paraphrase: **0.2028** (episode-cluster 95% CI [0.1725, 0.2314])
- Contradictory: **0.3543** ([0.2974, 0.4145])
- Unrelated: **0.8064** ([0.7619, 0.8560])

Aggregate result: **Paraphrase < Contradictory < Unrelated**. This ordering is not universal at every observation.

## Why this matters

The pretrained VLA is measurably conditioned on language in this controlled evaluation, and semantically more disruptive instructions generally perturb its predicted actions more strongly. Sensitivity also varies non-monotonically across task progress.

## Robotics part

The expert gripper channel is a persistent binary command. Temporal finger-state response and two-camera inspection establish `+1 = closing`, `-1 = opening`, and `+1 -> -1 = opening-command onset`. Command onset alone does not prove successful physical release.

## Most important limitation

This is offline action analysis, not closed-loop task-success evaluation.

## Likely questions

**Q: Did you train SmolVLA?**  
A: No. We evaluated a pretrained checkpoint.

**Q: How do you know the action differences came from language?**  
A: Images, robot state, preprocessing, model, and explicit flow noise were held fixed within every paired observation; only the task string changed.

**Q: Is the expert action ground truth?**  
A: No. It is the recorded expert demonstration used as a reference for agreement.

**Q: Did you prove safety?**  
A: No. We studied reliability-related behavior and gripper semantics, but did not implement or claim a universal safety mechanism.

**Q: Why are there only 10 episodes?**  
A: The local dataset cache contained 10 usable demonstrations; this limitation is documented.

**Q: Why does unrelated language matter?**  
A: It tests whether semantically irrelevant task conditioning changes the predicted robot action under otherwise identical inputs.

**Q: What is the main takeaway?**  
A: Language changes measurably alter the VLA's action, with semantically equivalent paraphrases generally causing smaller changes than contradictory or unrelated instructions.
