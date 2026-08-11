# Bottom-up presentation notes

## Recommended slide title

**Bottom-up: what does language change?**

## 45–60 second speaker wording

> The earlier experiment started with the full seven-dimensional action. A
> useful criticism was that this was difficult to interpret at the task level,
> so I rebuilt the explanation bottom-up. First, I isolate translation: where
> does the arm want to move? Here all thirty-one prompts have the same gripper
> regime, but the mean direction changes as the language moves from paraphrase
> to contradictory to unrelated. Second, I isolate the gripper decision: should
> the robot keep holding or open? At this stored state, none of the ten
> paraphrases request opening, compared with three of ten contradictory prompts
> and seven of ten unrelated prompts. We randomized prompt execution and reused
> identical flow noise within each observation, so this is not one lucky
> sentence. Third, I combine those two stored decisions in a simple one-object
> MuJoCo visualization. Finally, I return to the original multi-object packing
> task. The simulation is only a short-horizon interpretation of stored actions;
> it is not a closed-loop SmolVLA–LIBERO rollout or a task-success experiment.

## On-screen sequence

- **Level 1 — Translation:** “Where should the arm move?”
- **Level 2 — Gripper:** “Keep holding or open?”
- **Level 3 — One object:** “Combine both decisions once.”
- **Level 4 — Full packing:** “Return to the original task.”

## One-sentence takeaway

**Same observation, model, state, and flow noise; changing only text can alter
both the local movement proposal and the operational hold/open regime.**

## Required qualification

**Stored-action visualization; not closed-loop SmolVLA–LIBERO execution.**
