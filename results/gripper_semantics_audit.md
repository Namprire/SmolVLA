# Phase 10 gripper semantics source audit

## Dataset coverage

- Dataset: `HuggingFaceVLA/smol-libero`, local v2.1 representation at `.cache/huggingface/lerobot_v21/HuggingFaceVLA/smol-libero`.
- `meta/info.json` advertises 50 episodes and 13,021 frames.
- Locally usable parquet data: 10 episodes (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) and 2,612 frames. Episode lengths: {0: 258, 1: 252, 2: 246, 3: 288, 4: 244, 5: 279, 6: 303, 7: 247, 8: 238, 9: 257}.
- Episodes listed in metadata but without local parquet data were not treated as inspectable demonstrations.
- FPS: 20. Images are embedded PNGs from two cameras; no video files are used.

## Feature definitions and source evidence

- `.cache/huggingface/lerobot_v21/HuggingFaceVLA/smol-libero/meta/info.json` names `action` as seven float32 values: `['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']`. Therefore action index 6 is the gripper channel.
- The same metadata names the eight-dimensional state `['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper', 'gripper']` and the separate seven-dimensional joint state `joint_1` through `joint_7`.
- Installed LeRobot 0.4.4 `lerobot.processor.env_processor.LiberoProcessorStep` is more precise than the legacy state labels: it constructs `observation.state` as end-effector position (3), quaternion converted to axis-angle (3), then two gripper `qpos` values. Thus state indices 3–5 are axis-angle components, not roll/pitch/yaw despite the v2.1 metadata labels.
- Installed `lerobot.envs.libero.LiberoEnv` maps camera 1 to LIBERO `agentview_image` and camera 2 to `robot0_eye_in_hand_image`; the official processor flips stored images by 180 degrees. Contact sheets use that same rotation.
- The installed environment defines a seven-dimensional `[-1, 1]` action space and defaults to `control_mode="relative"`, setting the arm controller's `use_delta=True`. It passes the seven-vector directly to LIBERO. The last dimension is therefore the LIBERO gripper command alongside six relative end-effector controller commands, not an observed finger position.
- `get_libero_dummy_action()` returns `[0, 0, 0, 0, 0, 0, -1]`, consistent with `-1` being the non-closing/open command used during settling. The sign interpretation is independently checked below from state response and camera frames.

## Empirical gripper representation

- All 2,612 local expert gripper values are exactly `-1` or `+1`: {'-1.0': 1300, '1.0': 1312}.
- The adjacent-frame delta has only `-2`, `0`, or `+2`: {'-2.0': 24, '0.0': 2554, '2.0': 24}.
- There are 48 nonzero transitions: 24 `-1→+1` and 24 `+1→-1`.
- This is an exactly binary, saturated, temporally persistent open/close controller command. It is not an absolute gripper joint position: the measured finger positions are state indices 6–7 and change gradually after a command transition. The local files do not specify whether LIBERO's lower-level gripper controller realizes that binary command through position, velocity, or force control.
- Mean/median finger-qpos aperture proxy (`state[6] - state[7]`) under command `-1`: 0.072314/0.078425; under `+1`: 0.044547/0.042747.
- After `-1→+1`, mean aperture changes from 0.075059 at command onset to 0.039398 ten frames later. After `+1→-1`, it changes from 0.036676 to 0.069481. This supplies direct measured-state evidence that `+1` closes and `-1` opens.

## Normalization and clipping

- Checkpoint `.cache/huggingface/models/HuggingFaceVLA/smolvla_libero/config.json` maps ACTION to `MEAN_STD` normalization.
- Serialized preprocessor and postprocessor statistics agree. Gripper mean = -0.049640723; gripper std = 0.998767138.
- Native `-1` normalizes to -0.951532; native `+1` normalizes to 1.050936.
- Installed `lerobot.processor.normalize_processor._NormalizationMixin` applies `(x-mean)/(std+eps)` and inverse `x*std+mean`. It contains no clipping in the MEAN_STD path. The official postprocessor likewise does not clip, explaining why predictions may exceed the expert `[-1,1]` support.
- The installed environment declares `[-1,1]` bounds but its `step()` passes the array directly to LIBERO without a LeRobot-side clip. Downstream simulator/controller saturation was not exercised in this phase.

## Documentation ambiguity handled

The dataset card describes actions as “target joint commands,” while the actual seven names, current LeRobot LIBERO environment, and default relative control path identify six relative Cartesian/orientation controller commands plus a binary gripper command. For this audit, installed LeRobot 0.4.4, serialized checkpoint configuration, local metadata, and observed trajectories are treated as the source of truth.
