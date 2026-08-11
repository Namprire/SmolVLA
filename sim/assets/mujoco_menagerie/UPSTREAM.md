# MuJoCo Menagerie attribution

The directory `franka_emika_panda/` is an unmodified vendored snapshot of the
official Google DeepMind MuJoCo Menagerie model:

- Repository: https://github.com/google-deepmind/mujoco_menagerie
- Upstream path: `franka_emika_panda/`
- Commit: `da76818e269b82289eba39808e2fb91d679d6994`
- Commit date: 2026-08-09T12:35:18-07:00
- Vendored on: 2026-08-10
- Local MuJoCo version used for validation: 3.11.0

The upstream `README.md`, `CHANGELOG.md`, `LICENSE`, model XML files, reference
PNG, and mesh assets are retained inside the model directory. The model-level
license is Apache License 2.0; consult
`franka_emika_panda/LICENSE` for the complete terms.

No upstream Panda XML or mesh file has been edited. Repository-specific scene
composition and animation code live outside the vendored directory.

MuJoCo Menagerie citation:

```bibtex
@software{menagerie2022github,
  author = {Zakka, Kevin and Tassa, Yuval and {MuJoCo Menagerie Contributors}},
  title = {MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo},
  url = {https://github.com/google-deepmind/mujoco_menagerie},
  year = {2022}
}
```
