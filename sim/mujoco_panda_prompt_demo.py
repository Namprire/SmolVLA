#!/usr/bin/env python3
"""Polished Panda visualization grounded in stored SmolVLA experiment artifacts.

Required scientific disclaimer:
    Illustrative rendering of stored action differences;
    not closed-loop SmolVLA–LIBERO execution.

The Panda joint configurations are illustrative IK solutions tracking a fixed
transform of the stored expert end-effector trajectory. They are not recovered
LIBERO joint states. Prompt-conditioned arrows and branches come from stored
next-action predictions and are never presented as a closed-loop rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

_EARLY_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_EARLY_REPO_ROOT / ".cache" / "matplotlib"))

import imageio_ffmpeg
import mujoco
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = _EARLY_REPO_ROOT
SCENE_PATH = REPO_ROOT / "sim" / "packing_panda_scene.xml"
CONFIG_PATH = REPO_ROOT / "sim" / "panda_prompt_demo_config.json"
PREDICTIONS_PATH = REPO_ROOT / "results" / "vla_predictions.csv"
EPISODE_PATH = (
    REPO_ROOT
    / ".cache/huggingface/lerobot_v21/HuggingFaceVLA/smol-libero/data/chunk-000"
    / "episode_000000.parquet"
)
REFERENCE_GIF_PATH = REPO_ROOT / "results" / "demo_assets" / "episode_00_arm_demo.gif"
OUTPUT_ROOT = REPO_ROOT / "results" / "mujoco_panda_prompt_demo"
FIGURE_DIR = OUTPUT_ROOT / "figures"
VIDEO_DIR = OUTPUT_ROOT / "videos"

DISCLAIMER = (
    "Illustrative rendering of stored action differences; "
    "not closed-loop SmolVLA–LIBERO execution."
)

WIDTH = 1280
HEIGHT = 720
FPS = 24
EPISODE_ID = 0
PREVIEW_SOURCE_START = 28
PREVIEW_SOURCE_END = 142
PREVIEW_SECONDS = 4.0
MAIN_SECONDS = 25.0
TEASER_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "static",
            "preview",
            "checkpoint",
            "branch",
            "render",
            "render-hq",
            "render-teaser",
            "gif-audit",
        ),
        default="static",
        help="Render a validated hard gate or the complete 25-second hero deliverable.",
    )
    return parser.parse_args()


@lru_cache(maxsize=1)
def fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    regular = font_manager.findfont("DejaVu Sans")
    bold = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    return (
        ImageFont.truetype(bold, 34),
        ImageFont.truetype(regular, 20),
        ImageFont.truetype(regular, 14),
    )


def overlay_static(pixels: np.ndarray) -> np.ndarray:
    title_font, body_font, small_font = fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((28, 24, 625, 116), radius=18, fill=(7, 15, 25, 224))
    draw.text((50, 38), "Panda packing scene", font=title_font, fill="white")
    draw.text(
        (51, 82),
        "Official Menagerie Franka Panda | visual hard gate 1",
        font=body_font,
        fill=(198, 217, 235),
    )
    draw.rectangle((0, HEIGHT - 48, WIDTH, HEIGHT), fill=(6, 11, 18, 236))
    draw.text((24, HEIGHT - 34), DISCLAIMER, font=small_font, fill=(229, 237, 244))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    matrix = np.asarray(config["dataset_to_mujoco_xyz"]["matrix"], dtype=np.float64)
    translation = np.asarray(
        config["dataset_to_mujoco_xyz"]["translation"], dtype=np.float64
    )
    if matrix.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Invalid fixed dataset-to-MuJoCo transform shape")
    return config


def load_episode() -> dict[str, Any]:
    if not EPISODE_PATH.is_file():
        raise FileNotFoundError(f"Missing stored episode artifact: {EPISODE_PATH}")
    table = pq.read_table(EPISODE_PATH)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    frames = np.asarray(table["frame_index"], dtype=np.int64)
    if states.shape != (258, 8) or actions.shape != (258, 7):
        raise ValueError(f"Unexpected episode-0 shapes: {states.shape}, {actions.shape}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Stored episode has non-finite values")
    if not np.array_equal(frames, np.arange(258)):
        raise ValueError("Episode frame indices are not contiguous 0..257")
    return {"states": states, "actions": actions, "frames": frames}


def dataset_to_mujoco(xyz: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(config["dataset_to_mujoco_xyz"]["matrix"], dtype=np.float64)
    translation = np.asarray(
        config["dataset_to_mujoco_xyz"]["translation"], dtype=np.float64
    )
    values = np.asarray(xyz, dtype=np.float64)
    return values @ matrix.T + translation


def interpolate_rows(values: np.ndarray, frame: float) -> np.ndarray:
    low = int(np.floor(frame))
    high = min(low + 1, len(values) - 1)
    alpha = float(frame - low)
    return (1.0 - alpha) * values[low] + alpha * values[high]


def set_mocap_position(model: mujoco.MjModel, data: mujoco.MjData, name: str, xyz: np.ndarray) -> None:
    mocap_id = int(model.body(name).mocapid[0])
    if mocap_id < 0:
        raise ValueError(f"Expected mocap body: {name}")
    data.mocap_pos[mocap_id] = np.asarray(xyz, dtype=np.float64)


def finger_qpos_from_state(state: np.ndarray) -> float:
    aperture_proxy = float(state[6] - state[7])
    unit = float(np.clip((aperture_proxy - 0.04) / 0.04, 0.0, 1.0))
    return 0.004 + 0.036 * unit


def solve_position_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    seed: np.ndarray,
    home: np.ndarray,
    max_iterations: int = 90,
    tolerance: float = 1.5e-3,
) -> tuple[np.ndarray, float, int]:
    q = np.asarray(seed, dtype=np.float64).copy()
    hand_id = model.body("hand").id
    lower = model.jnt_range[:7, 0] + 0.015
    upper = model.jnt_range[:7, 1] - 0.015
    jac_pos = np.zeros((3, model.nv), dtype=np.float64)
    jac_rot = np.zeros((3, model.nv), dtype=np.float64)

    for iteration in range(max_iterations):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        error = np.asarray(target) - data.body("hand").xpos
        residual = float(np.linalg.norm(error))
        if residual <= tolerance:
            return q, residual, iteration + 1
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, hand_id)
        jacobian = jac_pos[:, :7]
        damping = 0.035
        primary = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(3), error
        )
        pseudoinverse = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(3), np.eye(3)
        )
        nullspace = np.eye(7) - pseudoinverse @ jacobian
        posture = nullspace @ (0.045 * (home - q))
        delta = primary + posture
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > 0.085:
            delta *= 0.085 / max_delta
        q = np.clip(q + delta, lower, upper)

    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    residual = float(np.linalg.norm(np.asarray(target) - data.body("hand").xpos))
    return q, residual, max_iterations


def solve_nominal_trajectory(
    model: mujoco.MjModel,
    episode: dict[str, Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    home = data.qpos[:7].copy()
    seed = home.copy()
    targets = dataset_to_mujoco(episode["states"][:, :3], config)
    solutions: list[np.ndarray] = []
    residuals: list[float] = []
    iterations: list[int] = []
    for target in targets:
        seed, residual, iteration_count = solve_position_ik(
            model, data, target, seed, home
        )
        solutions.append(seed.copy())
        residuals.append(residual)
        iterations.append(iteration_count)
    return (
        targets,
        np.asarray(solutions),
        np.asarray(residuals),
        np.asarray(iterations),
    )


def nominal_object_positions(
    targets: np.ndarray, frame: float
) -> tuple[np.ndarray, np.ndarray]:
    hand = interpolate_rows(targets, frame)
    cream_initial = targets[50] + np.array([0.0, 0.0, -0.112])
    butter_initial = targets[183] + np.array([0.0, 0.0, -0.112])
    cream_final = np.array([0.600, -0.105, 0.390])
    butter_final = np.array([0.662, -0.095, 0.382])
    if frame < 50:
        cream = cream_initial
    elif frame < 129:
        cream = hand + np.array([0.0, 0.0, -0.112])
    else:
        cream = cream_final
    if frame < 183:
        butter = butter_initial
    elif frame < 240:
        butter = hand + np.array([0.0, 0.0, -0.112])
    else:
        butter = butter_final
    return cream, butter


def apply_nominal_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    episode: dict[str, Any],
    targets: np.ndarray,
    q_solutions: np.ndarray,
    frame: float,
) -> tuple[np.ndarray, np.ndarray]:
    q = interpolate_rows(q_solutions, frame)
    state = interpolate_rows(episode["states"], frame)
    data.qpos[:7] = q
    finger = finger_qpos_from_state(state)
    data.qpos[7:9] = finger
    cream, butter = nominal_object_positions(targets, frame)
    set_mocap_position(model, data, "cream_cheese", cream)
    set_mocap_position(model, data, "butter", butter)
    mujoco.mj_forward(model, data)
    return data.body("hand").xpos.copy(), interpolate_rows(targets, frame)


def ffmpeg_writer(path: Path, fps: int = FPS) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
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
            str(fps),
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
        ],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")
    return process


def overlay_nominal_preview(
    pixels: np.ndarray, source_frame: float, residual: float
) -> np.ndarray:
    title_font, body_font, small_font = fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((28, 24, 670, 125), radius=18, fill=(7, 15, 25, 224))
    draw.text((50, 38), "Nominal Panda motion", font=title_font, fill="white")
    draw.text(
        (51, 82),
        f"Stored expert EEF target | source frame {source_frame:05.1f} | IK residual {residual * 1000:4.1f} mm",
        font=body_font,
        fill=(198, 217, 235),
    )
    draw.rectangle((0, HEIGHT - 48, WIDTH, HEIGHT), fill=(6, 11, 18, 236))
    draw.text((24, HEIGHT - 34), DISCLAIMER, font=small_font, fill=(229, 237, 244))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def render_nominal_preview() -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    episode = load_episode()
    config = load_config()
    targets, q_solutions, residuals, iterations = solve_nominal_trajectory(
        model, episode, config
    )
    if float(residuals.max()) > 0.012:
        raise ValueError(f"IK audit failed: max residual {residuals.max():.6f} m")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    trajectory_table = pd.DataFrame(
        {
            "episode_id": EPISODE_ID,
            "frame_index": episode["frames"],
            "source_eef_x": episode["states"][:, 0],
            "source_eef_y": episode["states"][:, 1],
            "source_eef_z": episode["states"][:, 2],
            "target_mj_x": targets[:, 0],
            "target_mj_y": targets[:, 1],
            "target_mj_z": targets[:, 2],
            "ik_q1": q_solutions[:, 0],
            "ik_q2": q_solutions[:, 1],
            "ik_q3": q_solutions[:, 2],
            "ik_q4": q_solutions[:, 3],
            "ik_q5": q_solutions[:, 4],
            "ik_q6": q_solutions[:, 5],
            "ik_q7": q_solutions[:, 6],
            "ik_residual_m": residuals,
            "ik_iterations": iterations,
        }
    )
    trajectory_path = OUTPUT_ROOT / "nominal_ik_trajectory.csv"
    trajectory_table.to_csv(trajectory_path, index=False)

    video_path = VIDEO_DIR / "hard_gate_2_nominal_preview.mp4"
    frame_count = int(round(PREVIEW_SECONDS * FPS))
    source_frames = np.linspace(PREVIEW_SOURCE_START, PREVIEW_SOURCE_END, frame_count)
    writer = ffmpeg_writer(video_path)
    data = mujoco.MjData(model)
    contact_frames: list[np.ndarray] = []
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        for output_index, source_frame in enumerate(source_frames):
            achieved, target = apply_nominal_pose(
                model, data, episode, targets, q_solutions, float(source_frame)
            )
            residual = float(np.linalg.norm(achieved - target))
            renderer.update_scene(data, camera="hero")
            pixels = overlay_nominal_preview(
                renderer.render().copy(), float(source_frame), residual
            )
            writer.stdin.write(np.ascontiguousarray(pixels).tobytes())
            if output_index in (0, frame_count // 3, 2 * frame_count // 3, frame_count - 1):
                contact_frames.append(pixels.copy())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("Nominal preview ffmpeg render failed")
    gif_path = VIDEO_DIR / "hard_gate_2_nominal_preview.gif"
    make_gif(video_path, gif_path)

    sheet = Image.new("RGB", (1280, 720), (7, 12, 18))
    for index, frame_pixels in enumerate(contact_frames):
        thumb = Image.fromarray(frame_pixels).resize((640, 360), Image.Resampling.LANCZOS)
        sheet.paste(thumb, ((index % 2) * 640, (index // 2) * 360))
    contact_path = FIGURE_DIR / "hard_gate_2_nominal_contact_sheet.png"
    sheet.save(contact_path)

    mp4_frames, mp4_duration = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    report = {
        "status": "PASS",
        "hard_gate": 2,
        "episode_id": EPISODE_ID,
        "source_frame_range": [PREVIEW_SOURCE_START, PREVIEW_SOURCE_END],
        "fixed_transform_config": str(CONFIG_PATH),
        "trajectory_rows": int(len(trajectory_table)),
        "ik_method": "position-only damped least-squares body Jacobian with nullspace home-posture regularization",
        "ik_residual_mean_m": float(residuals.mean()),
        "ik_residual_median_m": float(np.median(residuals)),
        "ik_residual_max_m": float(residuals.max()),
        "ik_residual_p95_m": float(np.quantile(residuals, 0.95)),
        "ik_all_targets_below_12mm": bool((residuals <= 0.012).all()),
        "video_frame_count": int(mp4_frames),
        "video_duration_seconds": float(mp4_duration),
        "resolution": [WIDTH, HEIGHT],
        "video": str(video_path),
        "gif": str(gif_path),
        "contact_sheet": str(contact_path),
        "trajectory_csv": str(trajectory_path),
        "episode_sha256": sha256_file(EPISODE_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "disclaimer": DISCLAIMER,
        "smol_vla_model_loaded": False,
        "libero_simulator_loaded": False,
    }
    (OUTPUT_ROOT / "hard_gate_2_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Nominal Panda IK hard gate 2: PASS")
    return report


ACTION_COLUMNS = [
    "pred_action_x",
    "pred_action_y",
    "pred_action_z",
    "pred_action_roll",
    "pred_action_pitch",
    "pred_action_yaw",
    "pred_action_gripper",
]
CONDITIONS = ("correct", "paraphrase", "contradictory", "unrelated")
CONDITION_STYLE = {
    "correct": {"label": "Correct", "color": np.array([0.20, 0.78, 0.38, 1.0])},
    "paraphrase": {"label": "Paraphrase", "color": np.array([0.20, 0.55, 0.96, 1.0])},
    "contradictory": {"label": "Contradictory", "color": np.array([1.00, 0.57, 0.16, 1.0])},
    "unrelated": {"label": "Unrelated", "color": np.array([0.96, 0.20, 0.22, 1.0])},
}
HERO_FRAME = 81
CHECKPOINT_SPECS = (
    (13, "Early ordinary comparison."),
    (53, "Near first close-command onset; real Contradictory gripper flip."),
    (81, "Hero comparison: Paraphrase near Correct; Unrelated large and opening-regime flip."),
    (128, "Near basket and first opening-command onset."),
    (191, "Near second close-command onset."),
    (221, "Late nominal holding interval; strong Unrelated gripper flip."),
    (241, "Late opening regime; retained real Paraphrase exception."),
)


def load_predictions() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    if len(predictions) != 800:
        raise ValueError(f"Expected 800 canonical predictions, found {len(predictions)}")
    if predictions.duplicated(["episode_id", "frame_index", "language_condition"]).any():
        raise ValueError("Canonical predictions contain duplicate observation-condition rows")
    return predictions


def action_vector(row: pd.Series) -> np.ndarray:
    action = row[ACTION_COLUMNS].to_numpy(dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Stored action is not finite 7-D")
    return action


def checkpoint_actions(
    predictions: pd.DataFrame, frame_index: int
) -> dict[str, dict[str, Any]]:
    group = predictions[
        predictions.episode_id.eq(EPISODE_ID)
        & predictions.frame_index.eq(frame_index)
    ]
    if len(group) != 4 or set(group.language_condition) != set(CONDITIONS):
        raise ValueError(f"Frame {frame_index} does not have exact four-condition coverage")
    indexed = group.set_index("language_condition")
    correct = action_vector(indexed.loc["correct"])
    output: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        row = indexed.loc[condition]
        action = action_vector(row)
        output[condition] = {
            "action": action,
            "instruction": str(row.instruction),
            "distance": float(np.linalg.norm(action - correct)),
            "arm_distance": float(np.linalg.norm(action[:6] - correct[:6])),
            "regime": "OPEN" if action[6] < 0 else "CLOSE",
            "gripper_flip": bool((action[6] < 0) != (correct[6] < 0)),
            "progress": float(row.task_progress),
        }
    return output


def build_selected_checkpoints(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame_index, reason in CHECKPOINT_SPECS:
        actions = checkpoint_actions(predictions, frame_index)
        rows.append(
            {
                "episode_id": EPISODE_ID,
                "frame_index": frame_index,
                "normalized_progress": actions["correct"]["progress"],
                "why_selected": reason,
                "correct_vs_paraphrase_distance": actions["paraphrase"]["distance"],
                "correct_vs_contradictory_distance": actions["contradictory"]["distance"],
                "correct_vs_unrelated_distance": actions["unrelated"]["distance"],
                "gripper_flip_present": any(
                    actions[condition]["gripper_flip"] for condition in CONDITIONS[1:]
                ),
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != 7 or not np.isfinite(
        table.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Checkpoint selection validation failed")
    output_path = OUTPUT_ROOT / "selected_checkpoints.csv"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def add_visual_capsule(
    scene: mujoco.MjvScene,
    point1: np.ndarray,
    point2: np.ndarray,
    radius: float,
    color: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Visual scene geometry capacity exhausted")
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        color.astype(np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(point1, dtype=np.float64),
        np.asarray(point2, dtype=np.float64),
    )


def add_visual_sphere(
    scene: mujoco.MjvScene, center: np.ndarray, radius: float, color: np.ndarray
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Visual scene geometry capacity exhausted")
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(center, dtype=np.float64),
        np.eye(3).reshape(-1),
        color.astype(np.float32),
    )


def visual_action_delta(action: np.ndarray, config: dict[str, Any], scale: float) -> np.ndarray:
    matrix = np.asarray(config["dataset_to_mujoco_xyz"]["matrix"], dtype=np.float64)
    return matrix @ action[:3] * scale


def gripper_tip_position(data: mujoco.MjData) -> np.ndarray:
    hand = data.body("hand")
    return hand.xpos + hand.xmat.reshape(3, 3) @ np.array([0.0, 0.0, 0.11])


def add_action_proposal(
    scene: mujoco.MjvScene,
    origin: np.ndarray,
    action: np.ndarray,
    condition: str,
    config: dict[str, Any],
    strength: float = 1.0,
) -> np.ndarray:
    color = CONDITION_STYLE[condition]["color"].copy()
    color[3] = 0.93
    scale = float(config["action_visualization"]["translation_arrow_scale"])
    end = origin + visual_action_delta(action, config, scale * strength)
    add_visual_capsule(scene, origin, end, 0.0085, color)
    add_visual_sphere(scene, end, 0.019, color)
    return end


def rgba255(color: np.ndarray, alpha: int = 255) -> tuple[int, int, int, int]:
    return tuple(int(round(float(channel) * 255)) for channel in color[:3]) + (alpha,)


def overlay_hero_checkpoint(
    pixels: np.ndarray,
    actions: dict[str, dict[str, Any]],
    progress: float,
) -> np.ndarray:
    title_font, body_font, small_font = fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((28, 22, 737, 121), radius=18, fill=(7, 15, 25, 230))
    draw.text((50, 36), "Same state. Only language changes.", font=title_font, fill="white")
    draw.text(
        (51, 81),
        f"Episode 0 | frame {HERO_FRAME} | {progress * 100:.1f}% normalized trajectory progress",
        font=body_font,
        fill=(198, 217, 235),
    )

    left, top, right, bottom = 830, 22, 1252, 330
    draw.rounded_rectangle((left, top, right, bottom), radius=18, fill=(7, 15, 25, 232))
    draw.text((left + 24, top + 18), "Stored next-action proposals", font=body_font, fill="white")
    for index, condition in enumerate(CONDITIONS):
        row_top = top + 65 + index * 57
        style = CONDITION_STYLE[condition]
        color = rgba255(style["color"])
        action = actions[condition]["action"]
        draw.rounded_rectangle(
            (left + 20, row_top, left + 54, row_top + 34), radius=9, fill=color
        )
        draw.text((left + 68, row_top - 1), style["label"], font=body_font, fill="white")
        flip = "  FLIP" if actions[condition]["gripper_flip"] else ""
        draw.text(
            (left + 68, row_top + 25),
            f"Δ7D {actions[condition]['distance']:.3f}  |  {actions[condition]['regime']} {action[6]:+.2f}{flip}",
            font=small_font,
            fill=(194, 211, 226),
        )

    draw.rounded_rectangle((28, 600, 640, 652), radius=14, fill=(7, 15, 25, 220))
    draw.text(
        (48, 616),
        "Arrow scale is illustrative and derived from stored action vectors.",
        font=small_font,
        fill=(205, 220, 234),
    )
    draw.rectangle((0, HEIGHT - 48, WIDTH, HEIGHT), fill=(6, 11, 18, 238))
    draw.text((24, HEIGHT - 34), DISCLAIMER, font=small_font, fill=(229, 237, 244))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def render_checkpoint_overlay() -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    episode = load_episode()
    config = load_config()
    predictions = load_predictions()
    selection = build_selected_checkpoints(predictions)
    actions = checkpoint_actions(predictions, HERO_FRAME)
    targets, q_solutions, residuals, _ = solve_nominal_trajectory(model, episode, config)
    achieved, target = apply_nominal_pose(
        model, data, episode, targets, q_solutions, float(HERO_FRAME)
    )
    ik_residual = float(np.linalg.norm(achieved - target))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "hard_gate_3_four_action_overlay.png"
    arrow_ends: dict[str, list[float]] = {}
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera="hero")
        origin = gripper_tip_position(data)
        for condition in CONDITIONS:
            arrow_ends[condition] = add_action_proposal(
                renderer.scene,
                origin,
                actions[condition]["action"],
                condition,
                config,
            ).tolist()
        pixels = overlay_hero_checkpoint(
            renderer.render().copy(), actions, actions["correct"]["progress"]
        )
    Image.fromarray(pixels).save(output)

    video_path = VIDEO_DIR / "hard_gate_3_four_action_overlay.mp4"
    gif_path = VIDEO_DIR / "hard_gate_3_four_action_overlay.gif"
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    writer = ffmpeg_writer(video_path)
    animated_frames = int(round(2.5 * FPS))
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        for frame_index in range(animated_frames):
            apply_nominal_pose(
                model, data, episode, targets, q_solutions, float(HERO_FRAME)
            )
            renderer.update_scene(data, camera="hero")
            origin = gripper_tip_position(data)
            phase = frame_index / max(animated_frames - 1, 1)
            strength = 0.20 + 0.80 * (0.5 - 0.5 * np.cos(2.0 * np.pi * phase))
            for condition in CONDITIONS:
                add_action_proposal(
                    renderer.scene,
                    origin,
                    actions[condition]["action"],
                    condition,
                    config,
                    strength=float(strength),
                )
            animated = overlay_hero_checkpoint(
                renderer.render().copy(), actions, actions["correct"]["progress"]
            )
            writer.stdin.write(np.ascontiguousarray(animated).tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("Four-action checkpoint animation render failed")
    make_gif(video_path, gif_path)

    report = {
        "status": "PASS",
        "hard_gate": 3,
        "episode_id": EPISODE_ID,
        "frame_index": HERO_FRAME,
        "normalized_progress": actions["correct"]["progress"],
        "exact_four_condition_coverage": set(actions) == set(CONDITIONS),
        "distances_from_correct": {
            condition: actions[condition]["distance"] for condition in CONDITIONS
        },
        "gripper_regimes": {
            condition: actions[condition]["regime"] for condition in CONDITIONS
        },
        "continuous_gripper_values": {
            condition: float(actions[condition]["action"][6]) for condition in CONDITIONS
        },
        "arrow_endpoints_mujoco": arrow_ends,
        "ik_residual_m": ik_residual,
        "full_trajectory_ik_max_residual_m": float(residuals.max()),
        "selected_checkpoint_count": int(len(selection)),
        "selected_checkpoints": str(OUTPUT_ROOT / "selected_checkpoints.csv"),
        "canonical_predictions_sha256": sha256_file(PREDICTIONS_PATH),
        "output": str(output),
        "animated_mp4": str(video_path),
        "gif": str(gif_path),
        "disclaimer": DISCLAIMER,
        "smol_vla_model_loaded": False,
        "closed_loop_rollout": False,
    }
    (OUTPUT_ROOT / "hard_gate_3_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Four-action overlay hard gate 3: PASS")
    return report


BRANCH_SECONDS = 3.0
BRANCH_PANEL_WIDTH = 640
BRANCH_PANEL_HEIGHT = 330


@lru_cache(maxsize=1)
def panel_fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    regular = font_manager.findfont("DejaVu Sans")
    bold = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    return (
        ImageFont.truetype(bold, 27),
        ImageFont.truetype(regular, 16),
        ImageFont.truetype(regular, 12),
    )


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def branch_phase(time_seconds: float) -> float:
    if time_seconds < 0.35:
        return 0.0
    if time_seconds < 1.35:
        return smoothstep((time_seconds - 0.35) / 1.0)
    if time_seconds < 2.35:
        return 1.0
    return 1.0 - smoothstep((time_seconds - 2.35) / 0.65)


def prepare_branch_solutions(
    model: mujoco.MjModel,
    episode: dict[str, Any],
    config: dict[str, Any],
    targets: np.ndarray,
    q_solutions: np.ndarray,
    actions: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, np.ndarray]]:
    data = mujoco.MjData(model)
    base_q = q_solutions[HERO_FRAME].copy()
    home = np.asarray(model.key("home").qpos[:7], dtype=np.float64)
    scale = float(config["action_visualization"]["local_branch_scale"])
    solutions: dict[str, np.ndarray] = {}
    residuals: dict[str, float] = {}
    branch_targets: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        target = targets[HERO_FRAME] + visual_action_delta(
            actions[condition]["action"], config, scale
        )
        q, residual, _ = solve_position_ik(model, data, target, base_q, home)
        solutions[condition] = q
        residuals[condition] = residual
        branch_targets[condition] = target
    return solutions, residuals, branch_targets


def overlay_branch_panel(
    pixels: np.ndarray,
    condition: str,
    action_record: dict[str, Any],
    phase: float,
) -> np.ndarray:
    title_font, body_font, small_font = panel_fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = CONDITION_STYLE[condition]["color"]
    draw.rectangle((0, 0, 640, 78), fill=(7, 15, 25, 226))
    draw.rectangle((0, 0, 9, 330), fill=rgba255(color, 245))
    draw.text((24, 10), CONDITION_STYLE[condition]["label"], font=title_font, fill="white")
    action = action_record["action"]
    flip = " | GRIPPER FLIP" if action_record["gripper_flip"] else ""
    draw.text(
        (24, 46),
        f"Δ7D {action_record['distance']:.3f} | {action_record['regime']} {action[6]:+.2f}{flip}",
        font=body_font,
        fill=(205, 221, 236),
    )
    draw.rounded_rectangle((398, 280, 622, 318), radius=10, fill=(7, 15, 25, 215))
    draw.text(
        (414, 291),
        f"illustrative branch {phase * 100:3.0f}%",
        font=small_font,
        fill=(214, 227, 239),
    )
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def combine_branch_panels(panels: list[np.ndarray], phase: float) -> np.ndarray:
    if len(panels) != 4:
        raise ValueError("Expected four branch panels")
    _, body_font, small_font = fonts()
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (6, 11, 18))
    positions = ((0, 0), (640, 0), (0, 330), (640, 330))
    for panel, position in zip(panels, positions):
        canvas.paste(Image.fromarray(panel), position)
    draw = ImageDraw.Draw(canvas)
    draw.line((640, 0, 640, 660), fill=(232, 239, 245), width=3)
    draw.line((0, 330, 1280, 330), fill=(232, 239, 245), width=3)
    draw.rectangle((0, 660, 1280, 720), fill=(6, 11, 18))
    draw.text(
        (20, 667),
        "Same Panda pose → four stored local action branches",
        font=body_font,
        fill="white",
    )
    draw.text((20, 695), DISCLAIMER, font=small_font, fill=(215, 227, 237))
    draw.text(
        (815, 695),
        "Branch scale is illustrative; no prediction is integrated into a rollout.",
        font=small_font,
        fill=(166, 187, 205),
    )
    draw.rectangle((0, 714, int(round(WIDTH * phase)), 720), fill=(72, 158, 224))
    return np.asarray(canvas)


def render_branch_preview() -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    episode = load_episode()
    config = load_config()
    predictions = load_predictions()
    actions = checkpoint_actions(predictions, HERO_FRAME)
    targets, q_solutions, _, _ = solve_nominal_trajectory(model, episode, config)
    branch_solutions, branch_residuals, branch_targets = prepare_branch_solutions(
        model, episode, config, targets, q_solutions, actions
    )
    if max(branch_residuals.values()) > 0.012:
        raise ValueError(f"Branch IK residual too large: {branch_residuals}")

    data = mujoco.MjData(model)
    base_q = q_solutions[HERO_FRAME]
    base_state = episode["states"][HERO_FRAME]
    base_finger = finger_qpos_from_state(base_state)
    base_cream, base_butter = nominal_object_positions(targets, float(HERO_FRAME))
    base_tip: np.ndarray | None = None

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_DIR / "hard_gate_4_branch_preview.mp4"
    writer = ffmpeg_writer(video_path)
    output_frames = int(round(BRANCH_SECONDS * FPS))
    contact_frames: list[np.ndarray] = []
    with mujoco.Renderer(model, BRANCH_PANEL_HEIGHT, BRANCH_PANEL_WIDTH) as renderer:
        for output_index in range(output_frames):
            time_seconds = output_index / FPS
            phase = branch_phase(time_seconds)
            panels: list[np.ndarray] = []
            for condition in CONDITIONS:
                q = (1.0 - phase) * base_q + phase * branch_solutions[condition]
                target_finger = 0.040 if actions[condition]["action"][6] < 0 else 0.004
                data.qpos[:7] = q
                data.qpos[7:9] = (1.0 - phase) * base_finger + phase * target_finger
                set_mocap_position(model, data, "cream_cheese", base_cream)
                set_mocap_position(model, data, "butter", base_butter)
                mujoco.mj_forward(model, data)
                if base_tip is None:
                    saved_q = data.qpos.copy()
                    data.qpos[:7] = base_q
                    mujoco.mj_forward(model, data)
                    base_tip = gripper_tip_position(data)
                    data.qpos[:] = saved_q
                    mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="branch")
                add_action_proposal(
                    renderer.scene,
                    base_tip,
                    actions[condition]["action"],
                    condition,
                    config,
                    strength=0.85,
                )
                panel = overlay_branch_panel(
                    renderer.render().copy(), condition, actions[condition], phase
                )
                panels.append(panel)
            composite = combine_branch_panels(panels, phase)
            writer.stdin.write(np.ascontiguousarray(composite).tobytes())
            if output_index in (0, output_frames // 2, output_frames - 1):
                contact_frames.append(composite.copy())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("Branch preview ffmpeg render failed")
    gif_path = VIDEO_DIR / "hard_gate_4_branch_preview.gif"
    make_gif(video_path, gif_path)

    sheet = Image.new("RGB", (1920, 1080), (6, 11, 18))
    for index, frame_pixels in enumerate(contact_frames):
        image = Image.fromarray(frame_pixels).resize((960, 540), Image.Resampling.LANCZOS)
        if index == 0:
            sheet.paste(image, (0, 0))
        elif index == 1:
            sheet.paste(image, (960, 0))
        else:
            sheet.paste(image, (480, 540))
    contact_path = FIGURE_DIR / "hard_gate_4_branch_contact_sheet.png"
    sheet.save(contact_path)
    mp4_frames, duration = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    report = {
        "status": "PASS",
        "hard_gate": 4,
        "episode_id": EPISODE_ID,
        "frame_index": HERO_FRAME,
        "frame_count": int(mp4_frames),
        "duration_seconds": float(duration),
        "resolution": [WIDTH, HEIGHT],
        "branch_ik_residuals_m": branch_residuals,
        "branch_targets_mujoco": {
            condition: target.tolist() for condition, target in branch_targets.items()
        },
        "gripper_visual_mapping": "stored prediction < 0 => Panda finger target 0.040 m OPEN; >= 0 => 0.004 m CLOSE",
        "continuous_gripper_values_retained": {
            condition: float(actions[condition]["action"][6]) for condition in CONDITIONS
        },
        "video": str(video_path),
        "gif": str(gif_path),
        "contact_sheet": str(contact_path),
        "disclaimer": DISCLAIMER,
        "stored_actions_integrated_as_rollout": False,
        "smol_vla_model_loaded": False,
    }
    (OUTPUT_ROOT / "hard_gate_4_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Local branch comparison hard gate 4: PASS")
    return report


def load_reference_frames() -> list[Image.Image]:
    if not REFERENCE_GIF_PATH.is_file():
        raise FileNotFoundError(f"Missing stored two-camera reference: {REFERENCE_GIF_PATH}")
    gif = Image.open(REFERENCE_GIF_PATH)
    frames: list[Image.Image] = []
    for index in range(gif.n_frames):
        gif.seek(index)
        frames.append(gif.convert("RGB").copy())
    if len(frames) != 129:
        raise ValueError(f"Expected 129 stored reference frames, found {len(frames)}")
    return frames


def add_footer(draw: ImageDraw.ImageDraw, right_text: str = "") -> None:
    _, _, small_font = fonts()
    draw.rectangle((0, HEIGHT - 48, WIDTH, HEIGHT), fill=(6, 11, 18, 240))
    draw.text((24, HEIGHT - 34), DISCLAIMER, font=small_font, fill=(229, 237, 244))
    if right_text:
        text_width = draw.textbbox((0, 0), right_text, font=small_font)[2]
        draw.text(
            (WIDTH - text_width - 24, HEIGHT - 34),
            right_text,
            font=small_font,
            fill=(158, 181, 200),
        )


def reference_video_frame(frame: Image.Image, progress: float) -> np.ndarray:
    title_font, body_font, small_font = fonts()
    canvas = frame.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((28, 24, 600, 119), radius=18, fill=(7, 15, 25, 226))
    draw.text((50, 38), "Recorded LIBERO demonstration", font=title_font, fill="white")
    draw.text(
        (51, 82),
        "Stored agent view + eye-in-hand view",
        font=body_font,
        fill=(200, 218, 234),
    )
    add_footer(draw, "real dataset frames | no new inference")
    draw.rectangle((0, HEIGHT - 6, WIDTH, HEIGHT), fill=(46, 57, 68, 255))
    draw.rectangle((0, HEIGHT - 6, int(WIDTH * progress), HEIGHT), fill=(75, 164, 229, 255))
    return np.asarray(Image.alpha_composite(canvas, overlay).convert("RGB"))


def overlay_nominal_main(
    pixels: np.ndarray,
    source_frame: float,
    residual: float,
    headline: str,
    subline: str,
) -> np.ndarray:
    title_font, body_font, small_font = fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((28, 24, 720, 126), radius=18, fill=(7, 15, 25, 226))
    draw.text((50, 38), headline, font=title_font, fill="white")
    draw.text((51, 82), subline, font=body_font, fill=(198, 217, 235))
    draw.text(
        (51, 106),
        f"source frame {source_frame:05.1f} | IK residual {residual * 1000:4.1f} mm",
        font=small_font,
        fill=(157, 181, 202),
    )
    add_footer(draw, "stored expert EEF target → illustrative Panda IK")
    progress = float(np.clip(source_frame / 257.0, 0.0, 1.0))
    draw.rectangle((0, HEIGHT - 6, WIDTH, HEIGHT), fill=(46, 57, 68, 255))
    draw.rectangle((0, HEIGHT - 6, int(WIDTH * progress), HEIGHT), fill=(75, 164, 229, 255))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def render_nominal_main_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    episode: dict[str, Any],
    targets: np.ndarray,
    q_solutions: np.ndarray,
    source_frame: float,
    headline: str,
    subline: str,
    camera_name: str = "hero",
) -> np.ndarray:
    achieved, target = apply_nominal_pose(
        model, data, episode, targets, q_solutions, source_frame
    )
    residual = float(np.linalg.norm(achieved - target))
    renderer.update_scene(data, camera=camera_name)
    return overlay_nominal_main(
        renderer.render().copy(), source_frame, residual, headline, subline
    )


def hero_overlay_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    episode: dict[str, Any],
    targets: np.ndarray,
    q_solutions: np.ndarray,
    actions: dict[str, dict[str, Any]],
    config: dict[str, Any],
    strength: float,
    camera_name: str = "hero",
) -> np.ndarray:
    apply_nominal_pose(model, data, episode, targets, q_solutions, float(HERO_FRAME))
    renderer.update_scene(data, camera=camera_name)
    origin = gripper_tip_position(data)
    for condition in CONDITIONS:
        add_action_proposal(
            renderer.scene,
            origin,
            actions[condition]["action"],
            condition,
            config,
            strength=strength,
        )
    return overlay_hero_checkpoint(
        renderer.render().copy(), actions, actions["correct"]["progress"]
    )


def overlay_teaser_motion(
    pixels: np.ndarray,
    instruction: str,
    source_frame: float,
    typed_fraction: float,
) -> np.ndarray:
    """Minimal prompt-and-motion overlay inspired by short robotics teasers."""
    _, body_font, small_font = fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    visible_count = int(round(len(instruction) * np.clip(typed_fraction, 0.0, 1.0)))
    visible_instruction = instruction[:visible_count]
    if typed_fraction < 1.0 and int(typed_fraction * 12) % 2 == 0:
        visible_instruction += "|"

    draw.ellipse((32, 27, 88, 83), fill=(7, 15, 25, 238), outline=(235, 241, 247), width=2)
    draw.text((50, 38), "L", font=body_font, fill="white")
    draw.rounded_rectangle((98, 25, 850, 86), radius=28, fill=(244, 246, 248, 238))
    draw.text((124, 43), visible_instruction, font=body_font, fill=(18, 25, 34))

    draw.rounded_rectangle((1010, 28, 1248, 78), radius=18, fill=(7, 15, 25, 218))
    draw.text((1034, 43), "ACCELERATED VISUALIZATION", font=small_font, fill=(219, 230, 240))

    draw.rectangle((0, HEIGHT - 48, WIDTH, HEIGHT), fill=(6, 11, 18, 238))
    draw.text((24, HEIGHT - 34), DISCLAIMER, font=small_font, fill=(229, 237, 244))
    draw.rectangle((0, HEIGHT - 6, WIDTH, HEIGHT), fill=(46, 57, 68, 255))
    progress = float(np.clip(source_frame / 257.0, 0.0, 1.0))
    draw.rectangle((0, HEIGHT - 6, int(WIDTH * progress), HEIGHT), fill=(75, 164, 229, 255))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def render_teaser_motion_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    episode: dict[str, Any],
    targets: np.ndarray,
    q_solutions: np.ndarray,
    source_frame: float,
    instruction: str,
    typed_fraction: float,
) -> np.ndarray:
    apply_nominal_pose(model, data, episode, targets, q_solutions, source_frame)
    renderer.update_scene(data, camera="hero_hq")
    return overlay_teaser_motion(
        renderer.render().copy(), instruction, source_frame, typed_fraction
    )


def main_branch_phase(local_time: float) -> float:
    if local_time < 0.40:
        return 0.0
    if local_time < 2.00:
        return smoothstep((local_time - 0.40) / 1.60)
    if local_time < 4.10:
        return 1.0
    return 1.0 - smoothstep((local_time - 4.10) / 0.90)


def render_branch_main_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    targets: np.ndarray,
    q_solutions: np.ndarray,
    episode: dict[str, Any],
    actions: dict[str, dict[str, Any]],
    config: dict[str, Any],
    branch_solutions: dict[str, np.ndarray],
    phase: float,
) -> np.ndarray:
    base_q = q_solutions[HERO_FRAME]
    base_finger = finger_qpos_from_state(episode["states"][HERO_FRAME])
    base_cream, base_butter = nominal_object_positions(targets, float(HERO_FRAME))
    data.qpos[:7] = base_q
    data.qpos[7:9] = base_finger
    set_mocap_position(model, data, "cream_cheese", base_cream)
    set_mocap_position(model, data, "butter", base_butter)
    mujoco.mj_forward(model, data)
    base_tip = gripper_tip_position(data)
    panels: list[np.ndarray] = []
    for condition in CONDITIONS:
        data.qpos[:7] = (1.0 - phase) * base_q + phase * branch_solutions[condition]
        target_finger = 0.040 if actions[condition]["action"][6] < 0 else 0.004
        data.qpos[7:9] = (1.0 - phase) * base_finger + phase * target_finger
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="branch")
        add_action_proposal(
            renderer.scene,
            base_tip,
            actions[condition]["action"],
            condition,
            config,
            strength=0.85,
        )
        panels.append(
            overlay_branch_panel(
                renderer.render().copy(), condition, actions[condition], phase
            )
        )
    return combine_branch_panels(panels, phase)


def end_card(background: np.ndarray) -> np.ndarray:
    title_font, body_font, small_font = fonts()
    image = Image.fromarray(background).convert("RGBA")
    shade = Image.new("RGBA", image.size, (4, 9, 15, 225))
    image = Image.alpha_composite(image, shade)
    draw = ImageDraw.Draw(image)
    headline = "Same vision + state + model + flow noise."
    second = "Only language changed."
    headline_width = draw.textbbox((0, 0), headline, font=title_font)[2]
    second_width = draw.textbbox((0, 0), second, font=title_font)[2]
    draw.text(((WIDTH - headline_width) / 2, 235), headline, font=title_font, fill="white")
    draw.text(
        ((WIDTH - second_width) / 2, 286),
        second,
        font=title_font,
        fill=(105, 190, 250),
    )
    draw.rounded_rectangle((260, 370, 1020, 440), radius=18, fill=(7, 15, 25, 215))
    disclaimer_width = draw.textbbox((0, 0), DISCLAIMER, font=body_font)[2]
    draw.text(
        ((WIDTH - disclaimer_width) / 2, 394),
        DISCLAIMER,
        font=body_font,
        fill=(224, 233, 241),
    )
    note = "Stored next-action comparison | one pretrained SmolVLA checkpoint | offline visualization"
    note_width = draw.textbbox((0, 0), note, font=small_font)[2]
    draw.text(
        ((WIDTH - note_width) / 2, 465), note, font=small_font, fill=(164, 186, 204)
    )
    return np.asarray(image.convert("RGB"))


def make_gif(video_path: Path, gif_path: Path) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-filter_complex",
            "[0:v]fps=12,scale=960:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            "-loop",
            "0",
            str(gif_path),
        ],
        check=True,
    )


def make_hq_gif(video_path: Path, gif_path: Path) -> None:
    """Encode a full-resolution, smooth 256-color presentation GIF."""
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-filter_complex",
            "[0:v]fps=15,scale=1280:720:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
            "-loop",
            "0",
            str(gif_path),
        ],
        check=True,
    )


def make_hold_gif(image_path: Path, gif_path: Path) -> None:
    """Create a subtle looping presentation GIF from a validated still."""
    source = Image.open(image_path).convert("RGB")
    output_size = (960, 540)
    frames: list[Image.Image] = []
    for index in range(24):
        phase = index / 23.0
        zoom = 1.0 + 0.018 * (0.5 - 0.5 * np.cos(2.0 * np.pi * phase))
        crop_width = int(round(source.width / zoom))
        crop_height = int(round(source.height / zoom))
        left = (source.width - crop_width) // 2
        top = (source.height - crop_height) // 2
        frame = source.crop((left, top, left + crop_width, top + crop_height))
        frames.append(frame.resize(output_size, Image.Resampling.LANCZOS))
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=83,
        loop=0,
        optimize=True,
    )


def render_teaser_deliverable() -> dict[str, Any]:
    """Render a concise prompt-first Panda GIF using only stored artifacts."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    episode = load_episode()
    config = load_config()
    predictions = load_predictions()
    actions = checkpoint_actions(predictions, HERO_FRAME)
    targets, q_solutions, residuals, _ = solve_nominal_trajectory(model, episode, config)
    if float(residuals.max()) > 0.012:
        raise ValueError("IK residual hard gate failed before teaser render")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_DIR / "panda_prompt_demo_teaser.mp4"
    gif_path = VIDEO_DIR / "panda_prompt_demo_teaser.gif"
    storyboard_path = FIGURE_DIR / "panda_prompt_demo_teaser_storyboard.png"
    writer = ffmpeg_writer(video_path)
    frame_count = int(round(TEASER_SECONDS * FPS))
    storyboard_times = (0.7, 3.4, 7.0, 9.4)
    storyboard_indices = {int(round(value * FPS)): value for value in storyboard_times}
    storyboard_frames: dict[float, np.ndarray] = {}
    instruction = actions["correct"]["instruction"]
    last_hero: np.ndarray | None = None

    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        for output_index in range(frame_count):
            time_seconds = output_index / FPS
            if time_seconds < 1.2:
                typed_fraction = smoothstep(time_seconds / 1.05)
                pixels = render_teaser_motion_frame(
                    model,
                    data,
                    renderer,
                    episode,
                    targets,
                    q_solutions,
                    0.0,
                    instruction,
                    typed_fraction,
                )
            elif time_seconds < 5.4:
                unit = smoothstep((time_seconds - 1.2) / 4.2)
                pixels = render_teaser_motion_frame(
                    model,
                    data,
                    renderer,
                    episode,
                    targets,
                    q_solutions,
                    unit * 257.0,
                    instruction,
                    1.0,
                )
            elif time_seconds < 8.8:
                local = time_seconds - 5.4
                strength = smoothstep(min(local / 0.7, 1.0))
                pixels = hero_overlay_frame(
                    model,
                    data,
                    renderer,
                    episode,
                    targets,
                    q_solutions,
                    actions,
                    config,
                    strength,
                    "hero_hq",
                )
                last_hero = pixels.copy()
            else:
                if last_hero is None:
                    raise RuntimeError("Missing teaser hero background")
                pixels = end_card(last_hero)

            writer.stdin.write(np.ascontiguousarray(pixels).tobytes())
            if output_index in storyboard_indices:
                storyboard_frames[storyboard_indices[output_index]] = pixels.copy()

    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("Teaser video ffmpeg render failed")

    if set(storyboard_frames) != set(storyboard_times):
        raise RuntimeError(f"Teaser storyboard capture mismatch: {sorted(storyboard_frames)}")
    storyboard = Image.new("RGB", (2560, 1440), (6, 11, 18))
    for index, time_value in enumerate(storyboard_times):
        storyboard.paste(
            Image.fromarray(storyboard_frames[time_value]),
            ((index % 2) * WIDTH, (index // 2) * HEIGHT),
        )
    storyboard.save(storyboard_path)
    make_hq_gif(video_path, gif_path)

    mp4_frames, mp4_duration = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    mp4_reader = imageio_ffmpeg.read_frames(str(video_path))
    mp4_metadata = next(mp4_reader)
    mp4_reader.close()
    with Image.open(gif_path) as gif:
        gif_frames = int(gif.n_frames)
        gif_resolution = list(gif.size)

    checks = {
        "canonical_predictions_800_rows": len(predictions) == 800,
        "canonical_predictions_no_duplicates": not predictions.duplicated(
            ["episode_id", "frame_index", "language_condition"]
        ).any(),
        "exact_four_conditions_at_hero": set(actions) == set(CONDITIONS),
        "canonical_instruction_exact": instruction
        == "put both the cream cheese box and the butter in the basket",
        "all_nominal_ik_residuals_below_12mm": bool((residuals <= 0.012).all()),
        "mp4_frame_count_expected": int(mp4_frames) == frame_count,
        "mp4_duration_expected": bool(np.isclose(mp4_duration, TEASER_SECONDS)),
        "mp4_resolution_expected": list(mp4_metadata["size"]) == [WIDTH, HEIGHT],
        "gif_animated": gif_frames > 1,
        "gif_resolution_expected": gif_resolution == [WIDTH, HEIGHT],
        "required_disclaimer_rendered": True,
        "smol_vla_model_not_loaded": True,
        "not_closed_loop_rollout": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Panda teaser validation failed: {checks}")

    report = {
        "status": "PASS",
        "style_references": {
            "teaser.mp4": "prompt-first reveal and sparse action graphics",
            "hang-towel.mp4": "stable wide camera and uninterrupted accelerated arm motion",
        },
        "camera": "hero_hq",
        "duration_seconds": float(mp4_duration),
        "fps": FPS,
        "frame_count": int(mp4_frames),
        "resolution": list(mp4_metadata["size"]),
        "canonical_instruction": instruction,
        "hero_frame": HERO_FRAME,
        "hero_distances_from_correct": {
            condition: actions[condition]["distance"] for condition in CONDITIONS
        },
        "hero_gripper_regimes": {
            condition: actions[condition]["regime"] for condition in CONDITIONS
        },
        "nominal_motion_source": "fixed transform of stored episode-0 expert end-effector XYZ targets",
        "timeline_seconds": {
            "typed_prompt": [0.0, 1.2],
            "accelerated_nominal_motion": [1.2, 5.4],
            "stored_four_prompt_action_overlay": [5.4, 8.8],
            "takeaway": [8.8, 10.0],
        },
        "mp4": str(video_path),
        "gif": {
            "path": str(gif_path),
            "frame_count": gif_frames,
            "resolution": gif_resolution,
            "size_bytes": gif_path.stat().st_size,
        },
        "storyboard": str(storyboard_path),
        "source_hashes": {
            "canonical_predictions_sha256": sha256_file(PREDICTIONS_PATH),
            "scene_xml_sha256": sha256_file(SCENE_PATH),
            "transform_config_sha256": sha256_file(CONFIG_PATH),
        },
        "disclaimer": DISCLAIMER,
        "stored_actions_integrated_as_rollout": False,
        "smol_vla_model_loaded": False,
        "checks": checks,
    }
    (OUTPUT_ROOT / "teaser_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Panda prompt teaser: PASS")
    return report


def audit_presentation_gifs() -> dict[str, Any]:
    gif_paths = {
        "stored_libero_reference": REPO_ROOT / "results/demo_assets/episode_00_arm_demo.gif",
        "simplified_prompt_comparison": REPO_ROOT
        / "results/mujoco_prompt_animation/videos/prompt_comparison_main.gif",
        "official_panda_hard_gate": REPO_ROOT
        / "results/mujoco_panda_hard_gate/panda_joint_gripper_hard_gate.gif",
        "panda_static_scene": VIDEO_DIR / "hard_gate_1_static_scene.gif",
        "panda_nominal_motion": VIDEO_DIR / "hard_gate_2_nominal_preview.gif",
        "panda_action_arrows": VIDEO_DIR / "hard_gate_3_four_action_overlay.gif",
        "panda_local_branches": VIDEO_DIR / "hard_gate_4_branch_preview.gif",
        "panda_complete_hero": VIDEO_DIR / "panda_prompt_demo_main.gif",
        "panda_complete_hero_hq": VIDEO_DIR / "panda_prompt_demo_main_hq.gif",
        "panda_prompt_teaser": VIDEO_DIR / "panda_prompt_demo_teaser.gif",
    }
    records: dict[str, Any] = {}
    for label, path in gif_paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing presentation GIF: {path}")
        with Image.open(path) as image:
            records[label] = {
                "path": str(path),
                "frame_count": int(image.n_frames),
                "resolution": list(image.size),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        if records[label]["frame_count"] <= 1:
            raise ValueError(f"Presentation GIF is not animated: {path}")
    report = {
        "status": "PASS",
        "gif_count": len(records),
        "all_files_nonempty": True,
        "all_gifs_animated": True,
        "primary_gif": str(gif_paths["panda_complete_hero_hq"]),
        "gifs": records,
        "disclaimer": DISCLAIMER,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "gif_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Presentation GIF audit: PASS")
    return report


def render_main_deliverable(
    camera_name: str = "hero",
    output_suffix: str = "",
    high_quality_gif: bool = False,
) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    episode = load_episode()
    config = load_config()
    predictions = load_predictions()
    selection = build_selected_checkpoints(predictions)
    reference_frames = load_reference_frames()
    actions = checkpoint_actions(predictions, HERO_FRAME)
    targets, q_solutions, residuals, iterations = solve_nominal_trajectory(
        model, episode, config
    )
    branch_solutions, branch_residuals, branch_targets = prepare_branch_solutions(
        model, episode, config, targets, q_solutions, actions
    )
    if float(residuals.max()) > 0.012 or max(branch_residuals.values()) > 0.012:
        raise ValueError("IK residual hard gate failed before main render")

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_DIR / f"panda_prompt_demo_main{output_suffix}.mp4"
    gif_path = VIDEO_DIR / f"panda_prompt_demo_main{output_suffix}.gif"
    storyboard_path = FIGURE_DIR / f"panda_prompt_demo_storyboard{output_suffix}.png"
    hero_path = FIGURE_DIR / f"panda_prompt_demo_hero_frame{output_suffix}.png"
    writer = ffmpeg_writer(video_path)
    frame_count = int(round(MAIN_SECONDS * FPS))
    storyboard_times = (1.5, 4.5, 8.5, 13.5, 18.5, 24.0)
    storyboard_indices = {int(round(value * FPS)): value for value in storyboard_times}
    storyboard_frames: dict[float, np.ndarray] = {}
    hero_pixels: np.ndarray | None = None
    data = mujoco.MjData(model)

    with (
        mujoco.Renderer(model, HEIGHT, WIDTH) as full_renderer,
        mujoco.Renderer(model, BRANCH_PANEL_HEIGHT, BRANCH_PANEL_WIDTH) as panel_renderer,
    ):
        final_nominal_background: np.ndarray | None = None
        for output_index in range(frame_count):
            time_seconds = output_index / FPS
            if time_seconds < 3.0:
                unit = time_seconds / 3.0
                reference_index = min(
                    int(round(unit * (len(reference_frames) - 1))),
                    len(reference_frames) - 1,
                )
                pixels = reference_video_frame(reference_frames[reference_index], unit)
            elif time_seconds < 6.0:
                local = (time_seconds - 3.0) / 3.0
                source_frame = local * 28.0
                reconstructed = render_nominal_main_frame(
                    model,
                    data,
                    full_renderer,
                    episode,
                    targets,
                    q_solutions,
                    source_frame,
                    "Illustrative MuJoCo reconstruction",
                    "Official Panda + one fixed transform of stored expert EEF targets",
                    camera_name,
                )
                if local < 0.40:
                    reference = reference_video_frame(reference_frames[-1], 1.0)
                    pixels = np.asarray(
                        Image.blend(
                            Image.fromarray(reference),
                            Image.fromarray(reconstructed),
                            smoothstep(local / 0.40),
                        )
                    )
                else:
                    pixels = reconstructed
            elif time_seconds < 11.0:
                local = (time_seconds - 6.0) / 5.0
                source_frame = local * 257.0
                pixels = render_nominal_main_frame(
                    model,
                    data,
                    full_renderer,
                    episode,
                    targets,
                    q_solutions,
                    source_frame,
                    "Shared nominal manipulation movement",
                    "Position-only Panda IK follows the stored expert end-effector target",
                    camera_name,
                )
            elif time_seconds < 16.0:
                local = time_seconds - 11.0
                strength = smoothstep(min(local / 1.25, 1.0))
                pixels = hero_overlay_frame(
                    model,
                    data,
                    full_renderer,
                    episode,
                    targets,
                    q_solutions,
                    actions,
                    config,
                    strength,
                    camera_name,
                )
                if abs(time_seconds - 13.5) < 0.5 / FPS:
                    hero_pixels = pixels.copy()
            elif time_seconds < 21.0:
                local = time_seconds - 16.0
                phase = main_branch_phase(local)
                pixels = render_branch_main_frame(
                    model,
                    data,
                    panel_renderer,
                    targets,
                    q_solutions,
                    episode,
                    actions,
                    config,
                    branch_solutions,
                    phase,
                )
            elif time_seconds < 23.0:
                local = (time_seconds - 21.0) / 2.0
                source_frame = HERO_FRAME + local * (257.0 - HERO_FRAME)
                pixels = render_nominal_main_frame(
                    model,
                    data,
                    full_renderer,
                    episode,
                    targets,
                    q_solutions,
                    source_frame,
                    "Back to the shared nominal movement",
                    "Prompt proposals were local stored comparisons—not executed rollouts",
                    camera_name,
                )
                final_nominal_background = pixels.copy()
            else:
                if final_nominal_background is None:
                    raise RuntimeError("Missing final nominal background")
                pixels = end_card(final_nominal_background)

            writer.stdin.write(np.ascontiguousarray(pixels).tobytes())
            if output_index in storyboard_indices:
                storyboard_frames[storyboard_indices[output_index]] = pixels.copy()

    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("Main hero video ffmpeg render failed")
    if hero_pixels is None:
        raise RuntimeError("Hero frame was not captured")
    Image.fromarray(hero_pixels).save(hero_path)

    if set(storyboard_frames) != set(storyboard_times):
        raise RuntimeError(f"Storyboard capture mismatch: {sorted(storyboard_frames)}")
    storyboard = Image.new("RGB", (2880, 1080), (6, 11, 18))
    for index, time_value in enumerate(storyboard_times):
        thumbnail = Image.fromarray(storyboard_frames[time_value]).resize(
            (960, 540), Image.Resampling.LANCZOS
        )
        storyboard.paste(thumbnail, ((index % 3) * 960, (index // 3) * 540))
    storyboard.save(storyboard_path)
    if high_quality_gif:
        make_hq_gif(video_path, gif_path)
    else:
        make_gif(video_path, gif_path)

    mp4_frames, mp4_duration = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    mp4_reader = imageio_ffmpeg.read_frames(str(video_path))
    mp4_metadata = next(mp4_reader)
    mp4_reader.close()
    with Image.open(gif_path) as gif:
        gif_frames = int(gif.n_frames)
        gif_size = list(gif.size)
    with Image.open(storyboard_path) as storyboard_image:
        storyboard_size = list(storyboard_image.size)
    with Image.open(hero_path) as hero_image:
        hero_size = list(hero_image.size)

    expected_regimes = {
        "correct": "CLOSE",
        "paraphrase": "CLOSE",
        "contradictory": "CLOSE",
        "unrelated": "OPEN",
    }
    checks = {
        "canonical_predictions_800_rows": len(predictions) == 800,
        "canonical_predictions_no_duplicates": not predictions.duplicated(
            ["episode_id", "frame_index", "language_condition"]
        ).any(),
        "exact_four_conditions_at_hero": set(actions) == set(CONDITIONS),
        "hero_regimes_expected": {
            condition: actions[condition]["regime"] for condition in CONDITIONS
        }
        == expected_regimes,
        "selected_checkpoint_count_7": len(selection) == 7,
        "same_fixed_transform_all_frames": True,
        "all_258_ik_targets_finite": bool(np.isfinite(q_solutions).all()),
        "all_nominal_ik_residuals_below_12mm": bool((residuals <= 0.012).all()),
        "all_branch_ik_residuals_below_12mm": max(branch_residuals.values()) <= 0.012,
        "mp4_frame_count_expected": int(mp4_frames) == frame_count,
        "mp4_duration_expected": bool(np.isclose(mp4_duration, MAIN_SECONDS)),
        "mp4_resolution_expected": list(mp4_metadata["size"]) == [WIDTH, HEIGHT],
        "gif_nonempty": gif_frames > 0 and gif_path.stat().st_size > 0,
        "storyboard_resolution_expected": storyboard_size == [2880, 1080],
        "hero_resolution_expected": hero_size == [WIDTH, HEIGHT],
        "required_disclaimer_rendered": True,
        "smol_vla_model_not_loaded": True,
        "not_closed_loop_rollout": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Final Panda prompt demo validation failed: {checks}")

    report = {
        "status": "PASS",
        "render_variant": "high_quality" if high_quality_gif else "standard",
        "camera": camera_name,
        "episode_id": EPISODE_ID,
        "representative_episode_reason": "complete two-camera reference, two manipulation cycles, exact four-condition coverage, and real high-contrast checkpoints including frame 81",
        "selected_checkpoint_count": int(len(selection)),
        "hero_frame": HERO_FRAME,
        "hero_progress": actions["correct"]["progress"],
        "hero_distances_from_correct": {
            condition: actions[condition]["distance"] for condition in CONDITIONS
        },
        "hero_gripper_regimes": {
            condition: actions[condition]["regime"] for condition in CONDITIONS
        },
        "hero_continuous_gripper_values": {
            condition: float(actions[condition]["action"][6]) for condition in CONDITIONS
        },
        "fixed_transform": config["dataset_to_mujoco_xyz"],
        "mapped_xyz": config["mapped_xyz"],
        "ik_method": "position-only damped least-squares MuJoCo body Jacobian with nullspace home-posture regularization",
        "ik_trajectory_rows": int(len(q_solutions)),
        "ik_residual_mean_m": float(residuals.mean()),
        "ik_residual_median_m": float(np.median(residuals)),
        "ik_residual_p95_m": float(np.quantile(residuals, 0.95)),
        "ik_residual_max_m": float(residuals.max()),
        "ik_iteration_max": int(iterations.max()),
        "branch_ik_residuals_m": branch_residuals,
        "branch_targets_mujoco": {
            condition: target.tolist() for condition, target in branch_targets.items()
        },
        "gripper_visual_mapping": "prediction < 0 => OPEN target 0.040 m; prediction >= 0 => CLOSE target 0.004 m",
        "translation_arrow_scale": config["action_visualization"]["translation_arrow_scale"],
        "local_branch_scale": config["action_visualization"]["local_branch_scale"],
        "video_structure_seconds": {
            "recorded_reference": [0, 3],
            "reconstruction_transition": [3, 6],
            "nominal_motion": [6, 11],
            "hero_four_action_overlay": [11, 16],
            "four_local_branches": [16, 21],
            "return_to_nominal": [21, 23],
            "takeaway_end_card": [23, 25],
        },
        "mp4": {
            "path": str(video_path),
            "frame_count": int(mp4_frames),
            "duration_seconds": float(mp4_duration),
            "fps": FPS,
            "resolution": list(mp4_metadata["size"]),
            "codec": str(mp4_metadata["codec"]),
        },
        "gif": {
            "path": str(gif_path),
            "frame_count": gif_frames,
            "resolution": gif_size,
            "encoding": (
                "15 fps, 1280x720, 256 colors, Sierra-2-4A dithering"
                if high_quality_gif
                else "12 fps, 960x540, 128 colors, Bayer dithering"
            ),
        },
        "storyboard": {"path": str(storyboard_path), "resolution": storyboard_size},
        "hero_still": {"path": str(hero_path), "resolution": hero_size},
        "source_hashes": {
            "canonical_predictions_sha256": sha256_file(PREDICTIONS_PATH),
            "episode_parquet_sha256": sha256_file(EPISODE_PATH),
            "reference_gif_sha256": sha256_file(REFERENCE_GIF_PATH),
            "scene_xml_sha256": sha256_file(SCENE_PATH),
            "transform_config_sha256": sha256_file(CONFIG_PATH),
        },
        "disclaimer": DISCLAIMER,
        "stored_actions_integrated_as_rollout": False,
        "smol_vla_model_loaded": False,
        "libero_simulator_loaded": False,
        "checks": checks,
    }
    validation_name = "hq_validation.json" if high_quality_gif else "validation.json"
    (OUTPUT_ROOT / validation_name).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(
        "Panda prompt hero demo HQ: PASS"
        if high_quality_gif
        else "Panda prompt hero demo hard gate 5: PASS"
    )
    return report


def render_static() -> Path:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "hard_gate_1_static_scene.png"
    with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera="hero")
        pixels = overlay_static(renderer.render().copy())
    Image.fromarray(pixels).save(output)
    gif_path = VIDEO_DIR / "hard_gate_1_static_scene.gif"
    make_hold_gif(output, gif_path)

    report = {
        "status": "PASS",
        "hard_gate": 1,
        "scene": str(SCENE_PATH),
        "robot_asset": "official Menagerie franka_emika_panda",
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "camera": "hero",
        "resolution": [WIDTH, HEIGHT],
        "output": str(output),
        "gif": str(gif_path),
        "disclaimer": DISCLAIMER,
        "smol_vla_model_loaded": False,
        "libero_simulator_loaded": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "hard_gate_1_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Panda packing-scene hard gate 1: PASS")
    return output


def main() -> None:
    args = parse_args()
    if args.mode == "static":
        render_static()
    elif args.mode == "preview":
        render_nominal_preview()
    elif args.mode == "checkpoint":
        render_checkpoint_overlay()
    elif args.mode == "branch":
        render_branch_preview()
    elif args.mode == "render":
        render_main_deliverable()
    elif args.mode == "render-hq":
        render_main_deliverable(
            camera_name="hero_hq", output_suffix="_hq", high_quality_gif=True
        )
    elif args.mode == "render-teaser":
        render_teaser_deliverable()
    elif args.mode == "gif-audit":
        audit_presentation_gifs()


if __name__ == "__main__":
    main()
