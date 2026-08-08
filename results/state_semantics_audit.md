# Phase 10 state semantics audit

## Eight-dimensional `observation.state`

Installed LeRobot 0.4.4 `lerobot.processor.env_processor.LiberoProcessorStep` constructs the state in this exact order:

- Index 0: `eef_position_x`; local min/max/mean = -0.068381 / 0.174795 / 0.056422.
- Index 1: `eef_position_y`; local min/max/mean = -0.228245 / 0.289008 / 0.045302.
- Index 2: `eef_position_z`; local min/max/mean = 0.445779 / 0.750079 / 0.597711.
- Index 3: `eef_axis_angle_x`; local min/max/mean = 2.894257 / 3.296714 / 3.127819.
- Index 4: `eef_axis_angle_y`; local min/max/mean = -0.633928 / 0.322641 / -0.161840.
- Index 5: `eef_axis_angle_z`; local min/max/mean = -0.520247 / 0.474559 / -0.100476.
- Index 6: `gripper_qpos_0`; local min/max/mean = 0.000930 / 0.040369 / 0.029048.
- Index 7: `gripper_qpos_1`; local min/max/mean = -0.040267 / -0.000688 / -0.029319.

- Indices 0–2 are the absolute observed end-effector Cartesian position from LIBERO `robot0_eef_pos`. XYZ is therefore available for later work; this phase does not extract release coordinates.
- Indices 3–5 are an absolute observed end-effector orientation represented as a three-component axis-angle vector converted from `robot0_eef_quat`. The v2.1 metadata's `roll/pitch/yaw` names are misleading for the current official processor.
- Indices 6–7 are the two observed gripper finger joint positions (`robot0_gripper_qpos`), not two command channels. Their opposing signs make `state[6] - state[7]` a useful aperture proxy for semantic validation.
- Units and reference frame are not declared in the local metadata. They are simulator-native pose coordinates; later geometry work must preserve that coordinate system and should not silently label units.

## Separate joint-state vector

`observation.state.joint` contains seven observed robot joint positions in indices 0–6 (`joint_1` through `joint_7`). It is not used by the validated SmolVLA checkpoint, whose input config includes only the eight-dimensional `observation.state` plus two images.

## Action/state distinction

The seven-dimensional expert `action` is not the eight-dimensional pose state. Under installed `LiberoEnv(control_mode="relative")`, its first six entries are relative end-effector translation/orientation controller commands and index 6 is the binary gripper command. Phase 11 must not use action XYZ as release location; only observed state indices 0–2 are defensible candidate Cartesian positions.
