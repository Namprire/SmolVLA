#!/usr/bin/env python3
"""Validate the unmodified official Menagerie Franka Panda in MuJoCo.

This hard gate loads the upstream scene, moves several articulated arm joints
through the official position actuators, closes and reopens the official Panda
gripper, and records a short native-MuJoCo render. It does not load SmolVLA or
attempt to reconstruct a LIBERO trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

import imageio_ffmpeg
import mujoco
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


MODEL_DIR = REPO_ROOT / "sim" / "assets" / "mujoco_menagerie" / "franka_emika_panda"
SCENE_PATH = MODEL_DIR / "scene.xml"
OUTPUT_DIR = REPO_ROOT / "results" / "mujoco_panda_hard_gate"

UPSTREAM_REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
UPSTREAM_REVISION = "da76818e269b82289eba39808e2fb91d679d6994"
WIDTH = 960
HEIGHT = 720
FPS = 30
DURATION_SECONDS = 3.0

HOME_ARM = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
POSE_A = np.array([0.45, -0.35, 0.30, -2.05, 0.25, 1.85, -0.35])
POSE_B = np.array([-0.35, 0.25, -0.25, -1.25, -0.20, 1.25, -1.00])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [
        name
        for index in range(count)
        if (name := mujoco.mj_id2name(model, object_type, index)) is not None
    ]


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def control_target(time_seconds: float) -> tuple[np.ndarray, float, str]:
    if time_seconds < 0.40:
        return HOME_ARM, 255.0, "HOME - GRIPPER OPEN"
    if time_seconds < 1.25:
        alpha = smoothstep((time_seconds - 0.40) / 0.85)
        return (1 - alpha) * HOME_ARM + alpha * POSE_A, 255.0 * (1 - alpha), "MOVE + CLOSE"
    if time_seconds < 1.55:
        return POSE_A, 0.0, "POSE A - GRIPPER CLOSED"
    if time_seconds < 2.55:
        alpha = smoothstep((time_seconds - 1.55) / 1.0)
        return (1 - alpha) * POSE_A + alpha * POSE_B, 255.0 * alpha, "MOVE + OPEN"
    return POSE_B, 255.0, "POSE B - GRIPPER OPEN"


def make_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, 0.0, 0.48])
    camera.distance = 1.75
    camera.azimuth = 135.0
    camera.elevation = -22.0
    return camera


def load_fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    regular_path = font_manager.findfont("DejaVu Sans")
    bold_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    return (
        ImageFont.truetype(bold_path, 31),
        ImageFont.truetype(regular_path, 19),
        ImageFont.truetype(regular_path, 15),
    )


def overlay_frame(
    pixels: np.ndarray,
    time_seconds: float,
    phase: str,
    finger_position: float,
) -> np.ndarray:
    title_font, body_font, small_font = load_fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((18, 16, 625, 105), radius=16, fill=(8, 17, 27, 224))
    draw.text((38, 28), "Official Menagerie Franka Panda", font=title_font, fill="white")
    draw.text(
        (39, 70),
        f"{phase} | t={time_seconds:0.2f}s | finger q={finger_position:0.4f}",
        font=body_font,
        fill=(202, 221, 238),
    )
    draw.rectangle((0, HEIGHT - 42, WIDTH, HEIGHT), fill=(7, 12, 18, 232))
    draw.text(
        (22, HEIGHT - 31),
        "Panda asset: Google DeepMind MuJoCo Menagerie | hard gate only | no SmolVLA",
        font=small_font,
        fill=(230, 237, 244),
    )
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def ffmpeg_writer(path: Path) -> subprocess.Popen[bytes]:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def make_gif(video_path: Path, gif_path: Path) -> None:
    filter_graph = (
        "[0:v]fps=15,scale=720:-2:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-filter_complex",
            filter_graph,
            "-loop",
            "0",
            str(gif_path),
        ],
        check=True,
    )


def make_contact_sheet(frames: list[np.ndarray], path: Path) -> None:
    if len(frames) != 4:
        raise ValueError(f"Expected four contact-sheet frames, got {len(frames)}")
    thumbnails = [
        Image.fromarray(frame).resize((480, 360), Image.Resampling.LANCZOS)
        for frame in frames
    ]
    sheet = Image.new("RGB", (960, 720), (7, 12, 18))
    for index, frame in enumerate(thumbnails):
        sheet.paste(frame, ((index % 2) * 480, (index // 2) * 360))
    sheet.save(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Missing vendored Menagerie scene: {SCENE_PATH}")

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    source_offscreen_size = [int(model.vis.global_.offwidth), int(model.vis.global_.offheight)]
    # Keep the vendored Menagerie XML byte-for-byte unchanged while requesting
    # a presentation-resolution offscreen framebuffer for this process.
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    data = mujoco.MjData(model)
    joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    site_names = names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    camera_names = names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
    keyframe_names = names(model, mujoco.mjtObj.mjOBJ_KEY, model.nkey)
    body_names = names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    gripper_joint_names = [name for name in joint_names if "finger" in name]
    end_effector_body_names = [
        name for name in ("hand", "left_finger", "right_finger") if name in body_names
    ]

    print(f"Python version: {platform.python_version()}")
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Model nq/nv/nu: {model.nq}/{model.nv}/{model.nu}")
    print(f"Joint names: {joint_names}")
    print(f"Actuator names: {actuator_names}")
    print(f"Panda gripper joint names: {gripper_joint_names}")
    print(f"End-effector body names: {end_effector_body_names}")
    print(f"Site names: {site_names}")
    print(f"Available cameras: {camera_names}")
    print(f"Available keyframes: {keyframe_names}")

    if (model.nq, model.nv, model.nu) != (9, 9, 8):
        raise ValueError(f"Unexpected Panda dimensions: {(model.nq, model.nv, model.nu)}")
    if gripper_joint_names != ["finger_joint1", "finger_joint2"]:
        raise ValueError(f"Unexpected Panda gripper joints: {gripper_joint_names}")
    if keyframe_names != ["home"]:
        raise ValueError(f"Unexpected Panda keyframes: {keyframe_names}")

    home_key_id = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    mujoco.mj_forward(model, data)
    camera = make_camera()

    default_png = args.output_dir / "panda_default_scene.png"
    video_path = args.output_dir / "panda_joint_gripper_hard_gate.mp4"
    gif_path = args.output_dir / "panda_joint_gripper_hard_gate.gif"
    contact_sheet_path = args.output_dir / "panda_motion_contact_sheet.png"
    validation_path = args.output_dir / "validation.json"

    writer = ffmpeg_writer(video_path)
    if writer.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")
    qpos_history: list[np.ndarray] = []
    hand_history: list[np.ndarray] = []
    sampled_frames: list[np.ndarray] = []
    warning_count = 0

    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera=camera)
        default_pixels = overlay_frame(
            renderer.render().copy(), 0.0, "UPSTREAM HOME KEYFRAME", float(np.mean(data.qpos[7:9]))
        )
        Image.fromarray(default_pixels).save(default_png)

        for output_index in range(int(round(DURATION_SECONDS * FPS))):
            target_time = output_index / FPS
            while data.time < target_time:
                arm_target, gripper_target, _ = control_target(float(data.time))
                data.ctrl[:7] = arm_target
                data.ctrl[7] = gripper_target
                mujoco.mj_step(model, data)
                warning_count += int(data.warning.number.any())
                qpos_history.append(data.qpos.copy())
                hand_history.append(data.body("hand").xpos.copy())
            _, _, phase = control_target(target_time)
            renderer.update_scene(data, camera=camera)
            pixels = overlay_frame(
                renderer.render().copy(),
                target_time,
                phase,
                float(np.mean(data.qpos[7:9])),
            )
            writer.stdin.write(np.ascontiguousarray(pixels).tobytes())
            if output_index in (0, 32, 48, 89):
                sampled_frames.append(pixels.copy())

    writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    make_gif(video_path, gif_path)
    make_contact_sheet(sampled_frames, contact_sheet_path)

    qpos = np.asarray(qpos_history)
    hand_positions = np.asarray(hand_history)
    arm_motion_range = np.ptp(qpos[:, :7], axis=0)
    finger_min = float(qpos[:, 7:9].min())
    finger_max = float(qpos[:, 7:9].max())
    moved_joint_count = int((arm_motion_range > 0.10).sum())
    mp4_frames, mp4_duration = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    mp4_reader = imageio_ffmpeg.read_frames(str(video_path))
    mp4_metadata = next(mp4_reader)
    mp4_reader.close()
    with Image.open(gif_path) as gif:
        gif_frames = int(gif.n_frames)
        gif_size = list(gif.size)

    checks = {
        "official_scene_compiles": True,
        "model_dimensions_expected": (model.nq, model.nv, model.nu) == (9, 9, 8),
        "all_seven_arm_joints_moved": moved_joint_count == 7,
        "gripper_reached_closed_visual_state": finger_min < 0.01,
        "gripper_reached_open_visual_state": finger_max > 0.035,
        "no_mujoco_warnings": warning_count == 0,
        "mp4_frame_count_expected": int(mp4_frames) == int(DURATION_SECONDS * FPS),
        "mp4_duration_expected": bool(np.isclose(mp4_duration, DURATION_SECONDS)),
        "mp4_resolution_expected": list(mp4_metadata["size"]) == [WIDTH, HEIGHT],
        "gif_nonempty": gif_frames > 0 and gif_path.stat().st_size > 0,
        "png_nonempty": default_png.stat().st_size > 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Panda hard-gate validation failed: {checks}")

    report: dict[str, Any] = {
        "status": "PASS",
        "python_version": platform.python_version(),
        "mujoco_version": mujoco.__version__,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "vendored_model_directory": str(MODEL_DIR),
        "scene_path": str(SCENE_PATH),
        "scene_sha256": sha256_file(SCENE_PATH),
        "panda_xml_sha256": sha256_file(MODEL_DIR / "panda.xml"),
        "license_sha256": sha256_file(MODEL_DIR / "LICENSE"),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "joint_names": joint_names,
        "actuator_names": actuator_names,
        "gripper_joint_names": gripper_joint_names,
        "end_effector_body_names": end_effector_body_names,
        "site_names": site_names,
        "camera_names": camera_names,
        "keyframe_names": keyframe_names,
        "arm_joint_motion_range": arm_motion_range.tolist(),
        "moved_arm_joint_count": moved_joint_count,
        "finger_position_min": finger_min,
        "finger_position_max": finger_max,
        "hand_position_min": hand_positions.min(axis=0).tolist(),
        "hand_position_max": hand_positions.max(axis=0).tolist(),
        "fps": FPS,
        "frame_count": int(mp4_frames),
        "duration_seconds": mp4_duration,
        "resolution": list(mp4_metadata["size"]),
        "source_offscreen_resolution": source_offscreen_size,
        "runtime_offscreen_resolution": [
            int(model.vis.global_.offwidth),
            int(model.vis.global_.offheight),
        ],
        "codec": str(mp4_metadata["codec"]),
        "gif_frame_count": gif_frames,
        "gif_resolution": gif_size,
        "outputs": {
            "default_png": str(default_png),
            "mp4": str(video_path),
            "gif": str(gif_path),
            "contact_sheet": str(contact_sheet_path),
        },
        "smol_vla_model_loaded": False,
        "libero_simulator_loaded": False,
        "checks": checks,
    }
    validation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Panda hard gate: PASS")


if __name__ == "__main__":
    main()
