# Panda prompt hero demo

> Illustrative rendering of stored action differences; not closed-loop
> SmolVLA–LIBERO execution.

## Purpose

This 25-second presentation animation connects the real stored LIBERO
observation streams to an illustrative MuJoCo reconstruction using the official
Google DeepMind MuJoCo Menagerie Franka Panda. It communicates the controlled
experiment visually: vision, robot state, checkpoint, preprocessing, and
explicit flow noise are fixed while only the instruction changes.

The visualization does not load SmolVLA, run new inference, install LIBERO, or
integrate stored predictions into a closed-loop rollout.

## Representative episode and checkpoints

Episode 0 is used because it has a complete stored two-camera GIF, 258 expert
end-effector states, two visually reviewed manipulation cycles, and exact
four-condition prediction coverage at 20 evaluated observations. The seven
checkpoints in `selected_checkpoints.csv` deliberately include ordinary frames,
command-transition neighborhoods, strong language effects, and a real
Paraphrase exception. They are not selected only for favorable ordering.

Frame 81 is the hero checkpoint:

| Condition | Full 7-D distance from Correct | Gripper value | Visual regime |
|---|---:|---:|---|
| Correct | 0.000 | +1.017 | CLOSE |
| Paraphrase | 0.056 | +1.021 | CLOSE |
| Contradictory | 0.066 | +1.010 | CLOSE |
| Unrelated | 2.284 | -0.981 | OPEN |

This is one curated real example, not a claim that every observation has this
ordering. The Phase B aggregate and its exceptions remain the scientific
result.

## Scene and provenance

`sim/packing_panda_scene.xml` includes the unmodified vendored Menagerie
`franka_emika_panda/panda.xml`. Repository-specific table, basket, proxy
products, lights, and cameras live in the separate scene file. Product geometry
is deliberately simple and illustrative.

The scene contains:

- the articulated Menagerie Franka Panda and Panda gripper;
- a wood tabletop and basket proxy;
- cream-cheese and butter box proxies;
- two restrained background props;
- a wide hero camera and close branch-comparison camera.

## Fixed coordinate transform

The source is `observation.state[0:3]` from stored episode 0. One transform is
used for all 258 frames and every prompt:

```text
xyz_mujoco = A @ xyz_dataset + b

A = [[0.0, 0.5, 0.0],
     [0.9, 0.0, 0.0],
     [0.0, 0.0, 0.75]]

b = [0.5, -0.10024764910340309, 0.14407656669616698]
```

Dataset `y` maps to Panda-forward `x`, dataset `x` maps to Panda-lateral `y`,
and dataset `z` maps to world `z`. The complete mapped target range is:

```text
x: 0.393468 to 0.635183 m
y: -0.142173 to 0.042173 m
z: 0.480000 to 0.693339 m
```

The machine-readable source ranges and transform are in
`sim/panda_prompt_demo_config.json`. No per-frame or per-prompt transform is
used.

## Panda IK and nominal movement

The stored dataset contains an expert end-effector pose, not the original Panda
joint trajectory. Each transformed XYZ point is therefore treated as an
illustrative position target. Joint configurations are solved sequentially with
a position-only damped least-squares MuJoCo body Jacobian plus nullspace
regularization toward the Menagerie home posture.

- target count: 258
- mean residual: 0.162 mm
- median residual: 0.111 mm
- 95th percentile residual: 0.681 mm
- maximum residual: 1.489 mm

These residuals measure agreement between the transformed visualization target
and the Panda `hand` body position. They do not validate physical task fidelity
or recover the original LIBERO robot joints.

## Stored action visualization

At the selected checkpoint, stored SmolVLA translation components are mapped by
the same axis transform. Two global illustrative scales are documented in the
config:

- action-arrow scale: 0.35
- short local-branch scale: 0.14

All four arrows begin at the same Panda fingertip center. Local branch snippets
use additional position-only IK toward the scaled target. They are reset local
comparisons and are never chained together as a rollout.

Gripper visualization uses the validated operational regime rule:

```text
stored prediction < 0  => OPEN visual target (Panda finger q = 0.040 m)
stored prediction >= 0 => CLOSE visual target (Panda finger q = 0.004 m)
```

The continuous stored gripper values remain in `validation.json`. The visual
finger target is not presented as a direct model output or proof of physical
release.

## Separate full pick-and-place prompt GIFs

Four 8-second GIFs extend the hard-gate-4 visual language into separate,
single-condition animations. Each starts above the cream-cheese box and shows
approach, grasp, lift, one condition-specific stored-action waypoint, scripted
transport to the basket, lowering, opening, a visible object drop, and retreat.

The exact episode-0/frame-81 prompts rendered in the GIFs are:

- Correct: `put both the cream cheese box and the butter in the basket`
- Paraphrase: `place the cream cheese box and the butter together inside the basket`
- Contradictory: `put the cream cheese box in the basket but leave the butter outside the basket`
- Unrelated display text: `put both the cream cheese box and the butter in the basket, and open the drawer of the cabinet`

Only the condition-specific waypoint comes from the stored prediction XYZ.
The complete grasp-to-basket motion is deterministic presentation choreography,
and the release is scripted independently of the stored frame-81 gripper value.
This distinction matters because Correct, Paraphrase, and Contradictory are
CLOSE at the selected frame, while Unrelated is OPEN. The Unrelated prompt also
uses requested combined display text, while its stored frame-81 proposal remains
traceable to the original inferred source prompt, `open the top drawer of the
cabinet`. Its animation intentionally remains an action-difference illustration
rather than a semantic execution claim.

## Video structure

- 0–3 s: stored LIBERO agent/eye-in-hand reference
- 3–6 s: transition to the illustrative Panda reconstruction
- 6–11 s: shared nominal movement
- 11–16 s: four stored action arrows at frame 81
- 16–21 s: synchronized local Panda branches and gripper regimes
- 21–23 s: return to the shared nominal movement
- 23–25 s: controlled-comparison takeaway and disclaimer

## Outputs

- `videos/panda_prompt_demo_teaser.mp4` — concise 10.0 s prompt-first cut,
  240 frames at 24 fps and 1280 x 720
- `videos/panda_prompt_demo_teaser.gif` — recommended short presentation opener;
  150 frames at 15 fps and 1280 x 720
- `figures/panda_prompt_demo_teaser_storyboard.png` — four-frame teaser storyboard
- `teaser_validation.json` — exact teaser timeline, sources, hashes, media
  metadata, and passing checks
- `videos/hard_gate_1_static_scene.gif` — looping packing-scene establishing shot
- `videos/hard_gate_2_nominal_preview.gif` — articulated nominal Panda movement
- `videos/hard_gate_3_four_action_overlay.gif` — pulsing four-action hero comparison
- `videos/hard_gate_4_branch_preview.gif` — synchronized four-branch Panda comparison
- `videos/full_pick_place_correct.gif` — Correct prompt, complete scripted pick-and-place
- `videos/full_pick_place_paraphrase.gif` — Paraphrase prompt, complete scripted pick-and-place
- `videos/full_pick_place_contradictory.gif` — Contradictory prompt, complete scripted pick-and-place
- `videos/full_pick_place_unrelated.gif` — Unrelated prompt, complete scripted pick-and-place
- `full_pick_place_validation.json` — exact prompts/actions, waypoint IK,
  object/basket checks, media metadata, hashes, and passing checks
- `videos/panda_prompt_demo_main.mp4` — 25.0 s, 600 frames, 24 fps,
  1280 x 720 H.264
- `videos/panda_prompt_demo_main.gif` — 300 frames, 960 x 540
- `videos/panda_prompt_demo_main_hq.mp4` — 25.0 s, 600 frames, 24 fps,
  1280 x 720 H.264 using the closer full-arm `hero_hq` camera
- `videos/panda_prompt_demo_main_hq.gif` — recommended presentation GIF;
  375 frames, 1280 x 720, 15 fps, 256 colors, Sierra-2-4A dithering
- `figures/panda_prompt_demo_storyboard.png` — 2880 x 1080
- `figures/panda_prompt_demo_hero_frame.png` — 1280 x 720
- `figures/panda_prompt_demo_storyboard_hq.png` — HQ-camera contact sheet
- `figures/panda_prompt_demo_hero_frame_hq.png` — HQ-camera hero still
- `validation.json` — exact sources, hashes, transform, IK metrics, media
  metadata, and passing checks
- `hq_validation.json` — corresponding checks and metadata for the HQ variant
- `gif_validation.json` — dimensions, frame counts, sizes, and hashes for the
  complete presentation GIF set

The HQ variant changes only presentation framing and media encoding. It uses
the same episode, stored predictions, fixed transform, IK trajectory, visual
action scales, and scientific disclaimer as the standard render.

The short teaser borrows only visual presentation language from the two local
reference videos: prompt-first sparse graphics from `teaser.mp4`, and a stable
wide camera with uninterrupted accelerated robot motion from `hang-towel.mp4`.
It does not borrow their tasks or algorithmic claims. Its nominal motion and
four local proposals use the same stored artifacts as the longer demo.

## Limitations

- This is an illustrative reconstruction, not a closed-loop rollout.
- The source trajectory is an expert EEF target backbone, not recovered Panda
  joint data.
- Orientation-control predictions are retained in the reported 7-D distances
  but are not animated as precise physical wrist rotations.
- Arrow and branch scales are presentation scales, not physical execution
  distances.
- Object attachment follows deterministic task-timeline visualization logic;
  it is not a grasp-dynamics simulation.
- A gripper command regime is not proof that an object was physically released.
