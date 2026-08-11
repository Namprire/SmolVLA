#!/usr/bin/env python3
"""Minimal native-MuJoCo compile, simulation, and rendering smoke test.

This deliberately tests only the local MuJoCo physics/rendering pipeline. It
does not load SmolVLA, LIBERO, a release envelope, or release-guard decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "mujoco_smoke"
)
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_DURATION_SECONDS = 1.25

FALLING_CUBE_MJCF = """
<mujoco model="falling cube smoke test">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.7 0.7 0.7"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.72 0.84 0.95" rgb2="0.96 0.98 1.0"
             width="256" height="1024"/>
    <texture name="floor_grid" type="2d" builtin="checker"
             rgb1="0.83 0.86 0.89" rgb2="0.96 0.97 0.98"
             width="512" height="512"/>
    <material name="floor" texture="floor_grid" texrepeat="4 4"
              texuniform="true" reflectance="0.12"/>
  </asset>

  <worldbody>
    <light name="key" pos="-0.4 -0.5 2.2" dir="0.2 0.2 -1"
           diffuse="0.9 0.9 0.9" specular="0.25 0.25 0.25"/>
    <light name="fill" pos="0.8 -0.2 1.2" dir="-0.4 0 -1"
           diffuse="0.45 0.48 0.52" specular="0.1 0.1 0.1"/>
    <camera name="presentation" pos="0 -1.35 0.75"
            xyaxes="1 0 0 0 0.33 1" fovy="42"/>

    <geom name="floor" type="plane" size="1.2 1.2 0.02"
          material="floor" friction="0.9 0.01 0.001"/>

    <body name="cube" pos="0 0 0.72">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.065 0.065 0.065"
            rgba="0.94 0.30 0.18 1" density="650"
            friction="0.8 0.01 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_SECONDS
    )
    return parser.parse_args()


def save_contact_sheet(
    initial_pixels: np.ndarray,
    final_pixels: np.ndarray,
    output_path: Path,
) -> None:
    initial = Image.fromarray(initial_pixels)
    final = Image.fromarray(final_pixels)
    header_height = 52
    sheet = Image.new(
        "RGB", (initial.width + final.width, initial.height + header_height), "white"
    )
    sheet.paste(initial, (0, header_height))
    sheet.paste(final, (initial.width, header_height))
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 17), "INITIAL - free cube above floor", fill=(25, 35, 48))
    draw.text(
        (initial.width + 18, 17),
        "FINAL - cube settled after mj_step",
        fill=(25, 35, 48),
    )
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Renderer width and height must be positive")
    if args.duration <= 0:
        raise ValueError("Simulation duration must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Hard-gate sequence: MJCF -> MjModel -> MjData -> forward -> step -> render.
    model = mujoco.MjModel.from_xml_string(FALLING_CUBE_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    initial_cube_position = data.body("cube").xpos.copy()
    render_backend_env = os.environ.get("MUJOCO_GL", "unset (MuJoCo default)")

    with mujoco.Renderer(model, args.height, args.width) as renderer:
        renderer.update_scene(data, camera="presentation")
        initial_pixels = renderer.render().copy()

        target_time = data.time + args.duration
        step_count = 0
        while data.time < target_time:
            mujoco.mj_step(model, data)
            step_count += 1

        renderer.update_scene(data, camera="presentation")
        final_pixels = renderer.render().copy()

    final_cube_position = data.body("cube").xpos.copy()
    if not np.isfinite(final_cube_position).all():
        raise RuntimeError("Final cube position is not finite")
    if final_cube_position[2] >= initial_cube_position[2] - 0.25:
        raise RuntimeError("Cube did not fall far enough to validate gravity/stepping")

    initial_path = args.output_dir / "falling_cube_initial.png"
    final_path = args.output_dir / "falling_cube_final.png"
    sheet_path = args.output_dir / "falling_cube_contact_sheet.png"
    Image.fromarray(initial_pixels).save(initial_path)
    Image.fromarray(final_pixels).save(final_path)
    save_contact_sheet(initial_pixels, final_pixels, sheet_path)

    report = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "mujoco_version": mujoco.__version__,
        "render_backend_env": render_backend_env,
        "model_timestep": float(model.opt.timestep),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "nq": int(model.nq),
        "ndof": int(model.nv),
        "step_count": step_count,
        "simulated_seconds": float(data.time),
        "initial_cube_position": initial_cube_position.tolist(),
        "final_cube_position": final_cube_position.tolist(),
        "renderer_width": args.width,
        "renderer_height": args.height,
        "initial_png": str(initial_path),
        "final_png": str(final_path),
        "contact_sheet_png": str(sheet_path),
        "pipeline_validated": True,
    }
    report_path = args.output_dir / "validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Python version: {sys.version.splitlines()[0]}")
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Render backend env: {render_backend_env}")
    print(f"Model timestep: {model.opt.timestep:.6f} s")
    print(f"Bodies: {model.nbody}")
    print(f"Joints: {model.njnt}")
    print(f"DoFs: {model.nv}")
    print(f"Initial cube position: {initial_cube_position.tolist()}")
    print(f"Final cube position: {final_cube_position.tolist()}")
    print(f"Renderer resolution: {args.width}x{args.height}")
    print(f"Simulation steps: {step_count}")
    print(f"Saved initial PNG: {initial_path}")
    print(f"Saved final PNG: {final_path}")
    print(f"Saved contact sheet: {sheet_path}")
    print(f"Saved validation: {report_path}")
    print("Hard gate: PASS")


if __name__ == "__main__":
    main()
