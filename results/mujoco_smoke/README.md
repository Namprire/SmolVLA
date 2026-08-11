# Native MuJoCo smoke-test gate

This directory records the smallest local macOS pipeline required before a
release-guard scene is scientifically or technically justified:

`MJCF -> MjModel -> MjData -> mj_forward -> mj_step -> Renderer -> PNG`

Run from the repository root:

```bash
.venv/bin/python sim/mujoco_smoke_test.py
```

Validated environment and result:

- Python 3.11.15 on macOS arm64
- native `mujoco==3.11.0`
- MuJoCo rendering backend environment variable unset (default backend)
- timestep 0.002 s; 625 steps; 1.25 simulated seconds
- 2 bodies, 1 free joint, 6 DoFs
- cube position changed from `[0, 0, 0.72]` to approximately
  `[0, 0, 0.0649205]`
- renderer resolution 640 x 480
- hard gate: PASS

The restricted command runner initially produced
`CGLError: invalid CoreGraphics connection`. The identical script passed when
given normal macOS CoreGraphics/OpenGL access; no EGL, CUDA, NVIDIA, or
`MUJOCO_GL` workaround was introduced.

Artifacts:

- `falling_cube_initial.png`
- `falling_cube_final.png`
- `falling_cube_contact_sheet.png`
- `validation.json`

This scaffold does **not** implement an empirical release envelope, guard
decision, basket scenario, closed-loop SmolVLA execution, or LIBERO simulation.
