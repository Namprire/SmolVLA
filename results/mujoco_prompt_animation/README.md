# Illustrative MuJoCo prompt animation

> Illustrative rendering of stored action differences; not closed-loop SmolVLA–LIBERO execution.

The main animation is a synchronized 2×2 comparison of Correct, Paraphrase,
Contradictory, and Unrelated language conditions. Every panel follows the same
stored episode-0 expert-demonstration XYZ backbone. Only the displayed stored
next-action proposal, full 7-D distance, and gripper regime change.

Outputs:

- `videos/prompt_comparison_main.mp4` — 1280×720, 20 fps, 370 frames, 18.5 s
- `videos/prompt_comparison_main.gif` — 960×540 presentation/README version
- `figures/prompt_comparison_storyboard.png` — six-checkpoint contact sheet
- `figures/legend.png` — visual encoding and scope explanation
- `selected_checkpoints.csv` — deterministic eight-checkpoint selection
- `nominal_episode_00.csv` — 258-frame stored EE/state backbone
- `animation_notes.md` — mapping and limitations
- `representative_example_summary.md` — episode-selection rationale
- `validation.json` — exact coverage, hashes, media metadata, and scope checks

Commands:

```bash
.venv/bin/python sim/mujoco_prompt_animation.py --mode preview
.venv/bin/python sim/mujoco_prompt_animation.py --mode render
.venv/bin/python sim/mujoco_prompt_animation.py --mode storyboard
```

The scene uses simple visual proxies. Stored action vectors are relative
controller commands whose physical calibration/reference units are not fully
documented, so the arrow scale is explicitly illustrative. The local actions
are not integrated and the animation does not establish physical task success,
safety, or closed-loop behavior.
