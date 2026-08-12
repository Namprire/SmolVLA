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
- Panda prompt hero demo — polished presentation deliverable
  - Kept all event/release-guard phases paused and performed zero new SmolVLA inference
  - Selected representative episode 0 and seven deterministic checkpoints spanning ordinary motion, command-transition neighborhoods, strong stored language effects, and a real Paraphrase exception
  - Built a separate LIBERO-like scene around the unmodified official Menagerie Panda with a wood table, basket, cream-cheese/butter proxies, restrained context props, cinematic hero camera, and close branch camera
  - Derived one fixed full-episode dataset-to-MuJoCo XYZ transform and reused it for all 258 trajectory frames and all prompt conditions
  - Implemented position-only damped least-squares MuJoCo body-Jacobian IK with home-posture nullspace regularization; mean residual is 0.162 mm and maximum residual is 1.489 mm over 258 targets
  - Passed five incremental visual hard gates: static scene, nominal trajectory preview, four-action checkpoint overlay, synchronized local branches, and final hero render
  - Used real stored frame-81 predictions for the hero comparison: Paraphrase 0.056, Contradictory 0.066, and Unrelated 2.284 full 7-D distance from Correct; only Unrelated switches from CLOSE to OPEN at this curated checkpoint
  - Rendered a 25.0-second, 600-frame, 1280x720 H.264 MP4; 300-frame GIF; 2880x1080 storyboard; and 1280x720 hero still
  - Added animated GIF counterparts for every Panda visual gate (scene, nominal motion, four-action arrows, and local branches)
  - Added a higher-quality primary variant using a closer full-arm camera: 1280x720 at 15 fps with a 256-color Sierra-2-4A-dithered GIF, plus matching MP4, storyboard, hero still, and validation report
  - Added a 10-second prompt-first teaser cut inspired by the visual pacing of the local `teaser.mp4` and `hang-towel.mp4` references: fixed HQ camera, typed canonical instruction, uninterrupted accelerated nominal Panda movement, and the stored frame-81 four-action overlay
  - Added four separate 8-second prompt-labeled pick-and-place GIFs for Correct, Paraphrase, Contradictory, and Unrelated, each showing approach, grasp, lift, its real frame-81 stored-action waypoint, scripted basket transport, visible drop, and retreat
  - Rendered the exact source prompt in every new animation and explicitly separated the stored one-step proposal/gripper value from the hand-authored completion choreography
  - Consolidated fourteen presentation GIFs in one index with a machine-readable audit; retained the smaller 960x540 primary render as a lightweight fallback
  - Rendered the controlled-comparison message and required non-closed-loop disclaimer prominently; predictions are local reset comparisons and are not chained into a rollout
  - Saved scene/config/source hashes, selected checkpoints, nominal IK trajectory, all hard-gate reports, final validation, and notes under `results/mujoco_panda_prompt_demo/`
- Bottom-up professor-feedback extension — complete
  - Reused the immutable 800-row canonical experiment and completed 620-row Phase-B prompt bank; performed no new SmolVLA inference
  - Deterministically selected episode 3/frame 38 for translation: all 31 prompts share the OPENING regime, while mean translation distance is 0.203 Paraphrase, 0.245 Contradictory, and 0.461 Unrelated
  - Deterministically selected episode 5/frame 230 for gripper variation: 0/10 Paraphrases, 3/10 Contradictory prompts, and 7/10 Unrelated prompts request OPENING; Correct remains CLOSING
  - Built a one-object Panda micro-task from four exact stored rows (`CORRECT`, `P07`, `C01`, `U09`) with one fixed transform and one-second local branches
  - Rendered a 16-second 1280x720 bottom-up MP4/GIF, translation and gripper primitive figures, a four-level pipeline graphic, and a one-object storyboard
  - Reported basket-distance and held/not-held outputs only as illustrative short-horizon task proxies; did not claim closed-loop control or task success
  - Saved source hashes, traceable prompt IDs, IK residuals, media metadata, checks, findings, and presentation wording under `results/bottom_up_demo/`
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
- Native MuJoCo hard gates, simplified stored-action animation, official Panda gate, and polished Panda prompt hero demo complete and validated
- Event/release-guard phases remain intentionally paused in favor of the presentation demo

## NEXT

- User review and presentation integration of the final Panda prompt hero demo

## BLOCKERS

- None
