# Official Menagerie Panda hard gate

Status: **PASS**

This gate validates the official `franka_emika_panda` model before any custom
packing-scene composition or inverse kinematics is attempted. It loads the
unmodified upstream `scene.xml`, advances native MuJoCo dynamics through the
official actuators, moves all seven arm joints, and closes and reopens the
official two-finger gripper.

## Provenance

- Source: <https://github.com/google-deepmind/mujoco_menagerie>
- Upstream directory: `franka_emika_panda/`
- Revision: `da76818e269b82289eba39808e2fb91d679d6994`
- Revision date: 2026-08-09
- Model license: Apache License 2.0; see
  `../../sim/assets/mujoco_menagerie/franka_emika_panda/LICENSE`
- Local runtime: Python 3.11, MuJoCo 3.11.0

The complete upstream folder is vendored unchanged at
`sim/assets/mujoco_menagerie/franka_emika_panda/`. Repository-specific Python
and future scene composition remain outside that directory. The source scene's
640 x 480 offscreen framebuffer is raised to 960 x 720 only in the loaded
runtime model; no upstream XML is edited.

## Validation result

- Model dimensions: `nq=9`, `nv=9`, `nu=8`
- Arm joints: `joint1` through `joint7`
- Gripper joints: `finger_joint1`, `finger_joint2`
- Actuators: `actuator1` through `actuator8`
- End-effector bodies: `hand`, `left_finger`, `right_finger`
- Sites in upstream model: none
- Cameras in upstream model: none
- Keyframes: `home`
- Arm joints moving by more than 0.10 rad: 7/7
- Observed finger-position range: approximately 0.000097 to 0.040001
- MuJoCo warnings: 0
- Video: 90 frames, 3.0 seconds, 30 fps, 960 x 720, H.264

See `validation.json` for exact values, hashes, media metadata, and every
machine-checked assertion.

## Outputs

- `panda_default_scene.png` — official home keyframe
- `panda_joint_gripper_hard_gate.mp4` — articulated arm and gripper motion
- `panda_joint_gripper_hard_gate.gif` — presentation/browser preview
- `panda_motion_contact_sheet.png` — four inspected motion states

The animation is an asset and actuator validation only. It does not load
SmolVLA, run LIBERO, reconstruct a dataset trajectory, or claim closed-loop
execution. The custom tabletop/basket scene and EEF-target IK are intentionally
deferred until this gate is reviewed.
