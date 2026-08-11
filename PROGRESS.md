# Progress

## DONE

- Extension Phase A — forensic analysis of the 20–40% Unrelated spike
  - Independently reconstructed all paired action differences from the immutable 800-row prediction CSV; no SmolVLA inference
  - Decomposed translation, orientation-control, gripper, arm-only 6-D, and full 7-D distances for every condition and normalized trajectory-progress bin
  - Audited predicted gripper-regime flips with the explicit threshold `< 0 = opening`, `>= 0 = closing`; continuous outputs remain unclipped
  - Found a mixed explanation: 17/39 regime flips amplify an arm-only spike (full mean 1.311807; arm-only mean 0.698335; no-flip diagnostic full mean 0.663055)
  - Confirmed coverage across all 10 episodes and saved episode-level, observation-level, figure, and written findings artifacts under `results/spike_analysis/`
- Extension Phase B — systematic multi-prompt robustness
  - Audited one exact Correct baseline plus 10 Paraphrase, 10 Contradictory, and 10 Unrelated manipulation prompts
  - Deterministically selected 20 observations (two per episode) spanning progress, expert gripper transitions, stored counterexamples/failures, ordinary controls, and Phase A cases
  - Validated a randomized-order 2 × 31 pilot, including identical per-observation flow noise, immediate writes, frame immutability, exact Correct-baseline reproduction, and a zero-recompute resume check
  - Completed 620 finite predictions with 31 prompt IDs and one noise hash per observation; reused 62 valid pilot rows and computed 558 remaining rows on MPS
  - After averaging 10 variants within each observation, full 7-D means remain ordered: Paraphrase 0.288395, Contradictory 0.398144, Unrelated 0.622293; arm-only means are 0.142947, 0.217074, and 0.346306
  - Ordering is not universal (12/20 observations full 7-D; 15/20 arm-only), U01/Drawer is the strongest Unrelated prompt, and the sharp 20–40% peak does not remain the largest bin after ten-prompt averaging
  - Saved prompt bank, selection, predictions, nested summaries, validations, findings, and three PNG/PDF figures under `results/prompt_robustness/`
- MuJoCo visual/physics scaffold — first hard gate
  - Read the provided DeepMind `tutorial.py` as a native-API reference while excluding its Colab, NVIDIA, CUDA, apt, and EGL setup
  - Installed the official native `mujoco==3.11.0` package in the existing Python 3.11 environment
  - Validated MJCF compilation, `MjData`, `mj_forward`, 625 `mj_step` calls, named body access, `Renderer`, and PNG capture on macOS
  - Used the unset/default MuJoCo rendering backend; the restricted runner lacked a CoreGraphics connection, while the same code passed with normal macOS graphics access
  - Confirmed a free cube fell from z=0.72 m to z=0.0649205 m and saved an inspected initial/final contact sheet under `results/mujoco_smoke/`
  - This is physics/rendering infrastructure only: no basket scenario, empirical release envelope, release guard, SmolVLA load, or LIBERO simulation was added
- MuJoCo stored-action prompt animation — presentation deliverable
  - Selected representative episode 0 because its complete two-camera GIF shows two visually reviewed manipulation cycles and its 20 stored observations retain the aggregate tendency without universal ordering
  - Exported all 258 stored expert EE-XYZ states as the shared nominal backbone and eight deterministic evaluation checkpoints spanning ordinary controls, command-transition neighborhoods, strong Unrelated sensitivity, and real Contradictory/Paraphrase exceptions
  - Built a compact MJCF tabletop scene with a simplified moving arm, gripper, basket, cream-cheese/butter proxies, and context objects
  - Passed the 2.95-second one-prompt MP4 hard gate before rendering the four-prompt comparison
  - Rendered a synchronized 2×2, 1280×720, 18.5-second H.264 animation (370 frames at 20 fps), a 960×540 GIF, a 2880×1080 storyboard, and a 1600×900 legend
  - Displays exact stored translation proposals at global illustrative scale 0.18, continuous gripper values, `<0` opening / `>=0` closing regimes, and full 7-D distance from Correct
  - Local stored predictions are never integrated into the nominal motion; SmolVLA is not loaded and the required non-closed-loop disclaimer is rendered and documented
  - Saved source hashes, exact coverage checks, nominal trajectory, checkpoint table, notes, summary, and passing validation under `results/mujoco_prompt_animation/`
- Official Menagerie Franka Panda — asset/actuator hard gate
  - Verified from the official LIBERO source that its demonstration collection defaults to the Panda robot
  - Vendored the official Google DeepMind MuJoCo Menagerie `franka_emika_panda/` snapshot unchanged at revision `da76818e269b82289eba39808e2fb91d679d6994`, retaining README, changelog, Apache-2.0 license, XML, reference image, and mesh assets
  - Compiled the upstream default scene under MuJoCo 3.11.0 (`nq=9`, `nv=9`, `nu=8`) without modifying its XML
  - Moved all seven arm joints through the upstream position actuators and closed/reopened `finger_joint1` and `finger_joint2` through the upstream gripper actuator
  - Rendered and visually inspected a 3.0-second, 90-frame, 960x720 H.264 animation, GIF, default-scene PNG, and four-state contact sheet
  - Machine checks confirm 7/7 arm joints move by >0.10 rad, finger q spans approximately 0.000097–0.040001, and MuJoCo produced zero warnings
  - Saved exact names, ranges, hashes, source/runtime framebuffer sizes, media metadata, and passing checks under `results/mujoco_panda_hard_gate/`
  - Stopped before custom tabletop composition, coordinate mapping, inverse kinematics, or scene cosmetics as required by the Panda hard gate
- Phases 1–10
  - Real pretrained `HuggingFaceVLA/smolvla_libero` inference on Apple M2 Max MPS
  - Deterministic explicit flow-noise control
  - 10 episodes, 200 paired observations, and 800 predictions
  - Correct / Paraphrase / Contradictory / Unrelated conditions
  - Paired language-effect and recorded-expert agreement metrics
  - Episode-cluster bootstrap uncertainty
  - Task-stage sensitivity analysis
  - Gripper and robot-state semantics audits
- Final analysis figures
  - Four presentation figures in high-resolution PNG and PDF
- Final offline demo
  - Same-observation condition switching from stored predictions
  - Five deterministic curated examples, including an ordering counterexample
  - Packaged two-camera assets; no model load or inference
- Documentation/reproducibility cleanup
  - Final validation, findings summary, presentation cheat sheet, README, and smoke-test record
  - Repository ignore rules and tracked-file audit

## SKIPPED BY SCOPE

- Phase 11 expert release extraction
- Phase 12 release envelope
- Phase 13 release guard
- Phase 14 stochasticity
- MuJoCo release-guard A/B/C scenarios (the native render scaffold now passes)

Reason: The controlled VLA language-sensitivity experiment already provides a complete research contribution, and remaining time is prioritized for rigorous analysis and presentation quality.

## CURRENT

- Extension Phases A and B complete and validated
- Native MuJoCo hard gate, stored-action prompt animation, and official Panda asset/actuator gate complete and validated
- Empirical release guard and its A/B/C scenarios intentionally not started

## NEXT

- User review of the official Panda hard-gate render
- After approval: custom LIBERO-like Panda scene, fixed dataset-to-MuJoCo coordinate transform, and EEF-target IK

## BLOCKERS

- None
