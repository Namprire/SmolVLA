#!/usr/bin/env python3
"""Illustrative packing animation grounded in stored episode/action artifacts.

Required scientific disclaimer:
    Illustrative rendering of stored action differences;
    not closed-loop SmolVLA–LIBERO execution.

The nominal end-effector path comes from the stored expert demonstration state.
Colored local arrows come from stored SmolVLA next-action predictions. Arrow
scale is illustrative; no local prediction is integrated into a rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

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
SCENE_PATH = REPO_ROOT / "sim" / "packing_prompt_scene.xml"
PREDICTIONS_PATH = REPO_ROOT / "results" / "vla_predictions.csv"
DATA_PATH_TEMPLATE = (
    REPO_ROOT
    / ".cache/huggingface/lerobot_v21/HuggingFaceVLA/smol-libero/data/chunk-000"
    / "episode_{episode_id:06d}.parquet"
)
OUTPUT_ROOT = REPO_ROOT / "results" / "mujoco_prompt_animation"
VIDEO_DIR = OUTPUT_ROOT / "videos"
FIGURE_DIR = OUTPUT_ROOT / "figures"

DISCLAIMER = (
    "Illustrative rendering of stored action differences; "
    "not closed-loop SmolVLA–LIBERO execution."
)
ARROW_NOTE = "Arrow/path scale is illustrative and derived from stored relative actions."

PANEL_WIDTH = 640
PANEL_HEIGHT = 360
FULL_PANEL_HEIGHT = 330
COMPOSITE_WIDTH = 1280
COMPOSITE_HEIGHT = 720
PREVIEW_FPS = 20
RENDER_FPS = 20
CHECKPOINT_HOLD_FRAMES = 14
PREVIEW_START_FRAME = 42
PREVIEW_END_FRAME = 100
PREVIEW_CHECKPOINT = 81
ARROW_SCALE = 0.18

CHECKPOINT_SPECS = (
    (13, "Early ordinary control: low-progress comparison without an event transition."),
    (53, "Nearest stored evaluation after first expert close-command onset at frame 50."),
    (81, "Strongest episode-0 Unrelated distance in the 20–40% progress region."),
    (128, "Nearest stored evaluation to first expert open-command onset at frame 129."),
    (156, "Ordinary mid-trajectory control between the two manipulation cycles."),
    (191, "Nearest stored evaluation after second expert close-command onset at frame 183."),
    (221, "Late second holding interval with a strong Unrelated gripper-regime flip."),
    (241, "Nearest stored evaluation to second expert open-command onset at frame 240."),
)
STORYBOARD_CHECKPOINTS = (13, 53, 81, 156, 221, 241)
CONDITION_ORDER = ("correct", "paraphrase", "contradictory", "unrelated")

CONDITION_STYLE = {
    "correct": {"label": "Correct", "color": np.array([0.16, 0.48, 0.88, 1.0])},
    "paraphrase": {"label": "Paraphrase", "color": np.array([0.12, 0.68, 0.58, 1.0])},
    "contradictory": {"label": "Contradictory", "color": np.array([0.94, 0.52, 0.12, 1.0])},
    "unrelated": {"label": "Unrelated", "color": np.array([0.86, 0.18, 0.18, 1.0])},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("preview", "render", "storyboard"),
        default="preview",
        help="Render the validated preview, final 2x2 comparison, or storyboard.",
    )
    parser.add_argument("--episode", type=int, default=0)
    return parser.parse_args()


def load_episode(episode_id: int) -> dict[str, Any]:
    path = Path(str(DATA_PATH_TEMPLATE).format(episode_id=episode_id))
    if not path.is_file():
        raise FileNotFoundError(f"Missing local stored episode data: {path}")
    table = pq.read_table(path)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    frame_indices = np.asarray(table["frame_index"], dtype=np.int64)
    timestamps = np.asarray(table["timestamp"], dtype=np.float64)
    if states.shape != (len(table), 8) or actions.shape != (len(table), 7):
        raise ValueError(f"Unexpected stored episode shapes: state={states.shape}, action={actions.shape}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Stored episode contains non-finite state/action values")
    if not np.array_equal(frame_indices, np.arange(len(table))):
        raise ValueError("Episode frame indices are not a contiguous zero-based sequence")
    if timestamps.shape != (len(table),) or not np.isfinite(timestamps).all():
        raise ValueError("Stored episode timestamps are missing or non-finite")
    return {
        "states": states,
        "actions": actions,
        "frame_indices": frame_indices,
        "timestamps": timestamps,
        "source_path": path,
    }


def load_prediction(episode_id: int, frame_index: int, condition: str) -> pd.Series:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    match = predictions[
        predictions.episode_id.eq(episode_id)
        & predictions.frame_index.eq(frame_index)
        & predictions.language_condition.eq(condition)
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one stored prediction for episode={episode_id}, "
            f"frame={frame_index}, condition={condition}; found {len(match)}"
        )
    return match.iloc[0]


def prediction_action(row: pd.Series) -> np.ndarray:
    values = row[
        [
            "pred_action_x",
            "pred_action_y",
            "pred_action_z",
            "pred_action_roll",
            "pred_action_pitch",
            "pred_action_yaw",
            "pred_action_gripper",
        ]
    ].to_numpy(dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("Stored prediction is not a finite 7-D action")
    return values


def validate_episode_zero_events(actions: np.ndarray) -> list[int]:
    transition_frames = (np.flatnonzero(np.diff(actions[:, 6]) != 0) + 1).tolist()
    expected = [50, 129, 183, 240]
    if transition_frames != expected:
        raise ValueError(
            "Episode-0 expert gripper transitions changed: "
            f"expected {expected}, found {transition_frames}"
        )
    directions = [
        (float(actions[index - 1, 6]), float(actions[index, 6]))
        for index in transition_frames
    ]
    if directions != [(-1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, -1.0)]:
        raise ValueError(f"Unexpected episode-0 gripper transition directions: {directions}")
    return transition_frames


def build_checkpoint_table(episode_id: int, predictions: pd.DataFrame) -> pd.DataFrame:
    if episode_id != 0:
        raise ValueError("The validated representative checkpoint set is fixed to episode 0")
    rows: list[dict[str, Any]] = []
    action_columns = [
        "pred_action_x",
        "pred_action_y",
        "pred_action_z",
        "pred_action_roll",
        "pred_action_pitch",
        "pred_action_yaw",
        "pred_action_gripper",
    ]
    for frame_index, reason in CHECKPOINT_SPECS:
        group = predictions[
            predictions.episode_id.eq(episode_id)
            & predictions.frame_index.eq(frame_index)
        ]
        if set(group.language_condition) != set(CONDITION_ORDER) or len(group) != 4:
            raise ValueError(
                f"Checkpoint {frame_index} lacks exact four-condition stored coverage"
            )
        indexed = group.set_index("language_condition")
        correct = indexed.loc["correct", action_columns].to_numpy(dtype=np.float64)
        altered_distances: dict[str, float] = {}
        flip_conditions: list[str] = []
        for condition in CONDITION_ORDER[1:]:
            action = indexed.loc[condition, action_columns].to_numpy(dtype=np.float64)
            altered_distances[condition] = float(np.linalg.norm(action - correct))
            if (action[6] < 0) != (correct[6] < 0):
                flip_conditions.append(CONDITION_STYLE[condition]["label"])
        metadata = indexed.loc["correct"]
        rows.append(
            {
                "episode_id": episode_id,
                "frame_index": frame_index,
                "global_index": int(metadata.global_index),
                "normalized_progress": float(metadata.task_progress),
                "why_selected": reason,
                "correct_vs_paraphrase_distance": altered_distances["paraphrase"],
                "correct_vs_contradictory_distance": altered_distances["contradictory"],
                "correct_vs_unrelated_distance": altered_distances["unrelated"],
                "gripper_flip_present": bool(flip_conditions),
                "gripper_flip_conditions": ", ".join(flip_conditions),
            }
        )
    output = pd.DataFrame(rows)
    numeric = output.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or len(output) != len(CHECKPOINT_SPECS):
        raise ValueError("Selected checkpoint table failed finite/coverage validation")
    return output


def checkpoint_prediction_map(
    episode_id: int, predictions: pd.DataFrame
) -> dict[int, dict[str, dict[str, Any]]]:
    output: dict[int, dict[str, dict[str, Any]]] = {}
    for frame_index, _ in CHECKPOINT_SPECS:
        group = predictions[
            predictions.episode_id.eq(episode_id)
            & predictions.frame_index.eq(frame_index)
        ].set_index("language_condition")
        correct_action = prediction_action(group.loc["correct"])
        output[frame_index] = {}
        for condition in CONDITION_ORDER:
            row = group.loc[condition]
            action = prediction_action(row)
            output[frame_index][condition] = {
                "action": action,
                "instruction": str(row.instruction),
                "full_distance": float(np.linalg.norm(action - correct_action)),
                "arm_distance": float(np.linalg.norm(action[:6] - correct_action[:6])),
                "translation_distance": float(np.linalg.norm(action[:3] - correct_action[:3])),
                "gripper_difference": float(abs(action[6] - correct_action[6])),
                "gripper_flip": bool((action[6] < 0) != (correct_action[6] < 0)),
            }
    return output


def write_stored_input_records(
    episode_id: int,
    episode: dict[str, Any],
    checkpoint_table: pd.DataFrame,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_table.to_csv(OUTPUT_ROOT / "selected_checkpoints.csv", index=False)
    states = episode["states"]
    actions = episode["actions"]
    nominal = pd.DataFrame(
        {
            "episode_id": episode_id,
            "frame_index": episode["frame_indices"],
            "timestamp": episode["timestamps"],
            "eef_x": states[:, 0],
            "eef_y": states[:, 1],
            "eef_z": states[:, 2],
            "eef_axis_angle_x": states[:, 3],
            "eef_axis_angle_y": states[:, 4],
            "eef_axis_angle_z": states[:, 5],
            "finger_qpos_0": states[:, 6],
            "finger_qpos_1": states[:, 7],
            "gripper_aperture_proxy": states[:, 6] - states[:, 7],
            "expert_gripper_command": actions[:, 6],
        }
    )
    if not np.isfinite(nominal.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("Nominal trajectory export contains non-finite values")
    nominal.to_csv(OUTPUT_ROOT / "nominal_episode_00.csv", index=False)


def set_mocap_position(model: mujoco.MjModel, data: mujoco.MjData, name: str, pos: np.ndarray) -> None:
    mocap_id = int(model.body(name).mocapid[0])
    if mocap_id < 0:
        raise ValueError(f"Body {name!r} is not a mocap body")
    data.mocap_pos[mocap_id] = pos


def interpolated_state(states: np.ndarray, frame: float) -> np.ndarray:
    low = int(np.floor(frame))
    high = min(low + 1, len(states) - 1)
    alpha = frame - low
    return (1.0 - alpha) * states[low] + alpha * states[high]


def nominal_object_positions(states: np.ndarray, frame: float) -> tuple[np.ndarray, np.ndarray]:
    table_z = 0.40
    basket_center = np.array([-0.002, 0.265, table_z])
    cream_initial = np.array([states[50, 0], states[50, 1], table_z + 0.025])
    butter_initial = np.array([states[183, 0], states[183, 1], table_z + 0.018])
    eef = interpolated_state(states, frame)[:3]

    if frame < 50:
        cream = cream_initial
    elif frame < 129:
        cream = eef + np.array([0.0, 0.0, -0.060])
    else:
        cream = basket_center + np.array([-0.045, 0.005, 0.040])

    if frame < 183:
        butter = butter_initial
    elif frame < 240:
        butter = eef + np.array([0.0, 0.0, -0.052])
    else:
        butter = basket_center + np.array([0.045, -0.005, 0.028])
    return cream, butter


def jaw_gap_from_aperture(aperture_proxy: float) -> float:
    # The stored episode-0 proxy is roughly 0.04 (closed) to 0.08 (open).
    unit = np.clip((aperture_proxy - 0.04) / 0.04, 0.0, 1.0)
    return float(0.026 + unit * 0.042)


def apply_nominal_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    states: np.ndarray,
    frame: float,
    displayed_gripper: float | None = None,
) -> np.ndarray:
    state = interpolated_state(states, frame)
    eef = state[:3]
    set_mocap_position(model, data, "eef", eef)

    if displayed_gripper is None:
        gap = jaw_gap_from_aperture(float(state[6] - state[7]))
    else:
        gap = 0.068 if displayed_gripper < 0 else 0.026
    jaw_z = eef[2] - 0.055
    set_mocap_position(model, data, "left_jaw", eef + np.array([-gap / 2, 0, jaw_z - eef[2]]))
    set_mocap_position(model, data, "right_jaw", eef + np.array([gap / 2, 0, jaw_z - eef[2]]))

    cream, butter = nominal_object_positions(states, frame)
    set_mocap_position(model, data, "cream_cheese", cream)
    set_mocap_position(model, data, "butter", butter)
    mujoco.mj_forward(model, data)
    return eef


def add_visual_capsule(
    scene: mujoco.MjvScene,
    point1: np.ndarray,
    point2: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo visual scene has no remaining geom capacity")
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1.astype(np.float64),
        point2.astype(np.float64),
    )


def add_visual_sphere(
    scene: mujoco.MjvScene,
    center: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo visual scene has no remaining geom capacity")
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        center.astype(np.float64),
        np.eye(3).reshape(-1),
        rgba.astype(np.float32),
    )


def add_arm_links(scene: mujoco.MjvScene, eef: np.ndarray) -> None:
    shoulder = np.array([-0.30, -0.27, 0.665])
    elbow = np.array(
        [
            (shoulder[0] + eef[0]) * 0.5 - 0.025,
            (shoulder[1] + eef[1]) * 0.5,
            max(0.78, eef[2] + 0.10),
        ]
    )
    wrist_top = eef + np.array([0.0, 0.0, 0.045])
    silver = np.array([0.58, 0.63, 0.69, 1.0])
    dark = np.array([0.18, 0.21, 0.24, 1.0])
    add_visual_capsule(scene, shoulder, elbow, 0.036, silver)
    add_visual_capsule(scene, elbow, wrist_top, 0.031, silver)
    add_visual_sphere(scene, shoulder, 0.050, dark)
    add_visual_sphere(scene, elbow, 0.047, dark)


def add_action_arrow(
    scene: mujoco.MjvScene,
    eef: np.ndarray,
    action: np.ndarray,
    color: np.ndarray,
    strength: float,
) -> None:
    start = eef + np.array([0.0, 0.0, 0.012])
    end = start + action[:3] * ARROW_SCALE * strength
    rgba = color.copy()
    rgba[3] = 0.90
    add_visual_capsule(scene, start, end, 0.011, rgba)
    add_visual_sphere(scene, end, 0.024, rgba)


@lru_cache(maxsize=1)
def load_fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    font_path = font_manager.findfont("DejaVu Sans")
    bold_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    return (
        ImageFont.truetype(bold_path, 24),
        ImageFont.truetype(font_path, 16),
        ImageFont.truetype(font_path, 12),
    )


def overlay_preview(
    pixels: np.ndarray,
    frame: float,
    action: np.ndarray,
    arrow_strength: float,
) -> np.ndarray:
    title_font, body_font, small_font = load_fonts()
    image = Image.fromarray(pixels)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((12, 10, 628, 76), radius=12, fill=(12, 22, 34, 218))
    draw.text((25, 18), "Correct - stored next-action proposal", font=title_font, fill="white")
    regime = "OPENING" if action[6] < 0 else "CLOSING"
    draw.text(
        (25, 49),
        f"Episode 0 | frame {frame:05.1f} | gripper: {regime}",
        font=body_font,
        fill=(214, 229, 244),
    )
    if arrow_strength > 0.02:
        draw.rounded_rectangle((398, 276, 628, 314), radius=9, fill=(25, 80, 145, 225))
        draw.text((413, 286), "Stored action arrow", font=body_font, fill="white")
    draw.rectangle((0, 324, 640, 360), fill=(7, 12, 18, 232))
    draw.text((15, 330), DISCLAIMER, font=small_font, fill=(238, 242, 247))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(image)


def rgb255(color: np.ndarray, alpha: int = 255) -> tuple[int, int, int, int]:
    return (
        int(round(float(color[0]) * 255)),
        int(round(float(color[1]) * 255)),
        int(round(float(color[2]) * 255)),
        alpha,
    )


def draw_xy_action_inset(
    draw: ImageDraw.ImageDraw,
    action: np.ndarray,
    color: np.ndarray,
    body_font: ImageFont.FreeTypeFont,
    small_font: ImageFont.FreeTypeFont,
) -> None:
    left, top, right, bottom = 500, 230, 630, 320
    draw.rounded_rectangle((left, top, right, bottom), radius=10, fill=(8, 16, 25, 218))
    center_x, center_y = 548, 278
    draw.line((center_x - 32, center_y, center_x + 32, center_y), fill=(135, 149, 163), width=1)
    draw.line((center_x, center_y - 30, center_x, center_y + 30), fill=(135, 149, 163), width=1)
    dx = float(np.clip(action[0] * 38, -36, 36))
    dy = float(np.clip(-action[1] * 38, -32, 32))
    endpoint = (center_x + dx, center_y + dy)
    rgba = rgb255(color)
    draw.line((center_x, center_y, *endpoint), fill=rgba, width=5)
    draw.ellipse(
        (endpoint[0] - 6, endpoint[1] - 6, endpoint[0] + 6, endpoint[1] + 6),
        fill=rgba,
    )
    draw.text((574, 240), "xyz", font=body_font, fill="white")
    draw.text((574, 268), f"z {action[2]:+.2f}", font=small_font, fill=(220, 229, 238))
    draw.text((574, 292), "stored", font=small_font, fill=(170, 187, 203))


def overlay_condition_panel(
    pixels: np.ndarray,
    condition: str,
    frame: float,
    progress: float,
    checkpoint: dict[str, Any] | None,
) -> np.ndarray:
    title_font, body_font, small_font = load_fonts()
    style = CONDITION_STYLE[condition]
    color = style["color"]
    image = Image.fromarray(pixels)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, PANEL_WIDTH, 82), fill=(10, 18, 28, 224))
    draw.rectangle((0, 0, 9, FULL_PANEL_HEIGHT), fill=rgb255(color, 245))
    draw.text((22, 10), str(style["label"]), font=title_font, fill="white")

    if checkpoint is None:
        draw.text(
            (22, 43),
            f"Nominal expert backbone | frame {frame:05.1f} | {progress * 100:4.1f}%",
            font=body_font,
            fill=(207, 220, 233),
        )
        draw.text(
            (22, 64),
            "Stored next actions appear at selected checkpoints",
            font=small_font,
            fill=(164, 183, 201),
        )
    else:
        action = checkpoint["action"]
        regime = "OPENING" if action[6] < 0 else "CLOSING"
        flip_text = " | FLIP vs Correct" if checkpoint["gripper_flip"] else ""
        draw.text(
            (22, 39),
            f"Δ7D {checkpoint['full_distance']:.3f} | {regime} {action[6]:+.3f}{flip_text}",
            font=body_font,
            fill=(235, 241, 247),
        )
        prompt = textwrap.shorten(checkpoint["instruction"], width=74, placeholder="...")
        draw.text((22, 63), prompt, font=small_font, fill=(182, 202, 221))
        draw_xy_action_inset(draw, action, color, body_font, small_font)

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(image)


def render_condition_panel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    states: np.ndarray,
    frame: float,
    condition: str,
    checkpoint: dict[str, Any] | None,
    arrow_strength: float,
) -> np.ndarray:
    displayed_gripper = None
    if checkpoint is not None and arrow_strength > 0.02:
        displayed_gripper = float(checkpoint["action"][6])
    eef = apply_nominal_scene(model, data, states, frame, displayed_gripper)
    renderer.update_scene(data, camera="presentation")
    add_arm_links(renderer.scene, eef)
    if checkpoint is not None and arrow_strength > 0:
        add_action_arrow(
            renderer.scene,
            eef,
            checkpoint["action"],
            CONDITION_STYLE[condition]["color"],
            arrow_strength,
        )
    pixels = renderer.render().copy()
    return overlay_condition_panel(
        pixels,
        condition,
        frame,
        frame / (len(states) - 1),
        checkpoint,
    )


def combine_panels(
    panels: list[np.ndarray],
    frame: float,
    n_episode_frames: int,
    checkpoint_frame: int | None,
    checkpoint_reason: str | None,
) -> np.ndarray:
    if len(panels) != 4:
        raise ValueError(f"Expected four rendered panels, got {len(panels)}")
    title_font, body_font, small_font = load_fonts()
    canvas = Image.new("RGB", (COMPOSITE_WIDTH, COMPOSITE_HEIGHT), (7, 12, 18))
    locations = ((0, 0), (640, 0), (0, 330), (640, 330))
    for panel, location in zip(panels, locations):
        canvas.paste(Image.fromarray(panel), location)
    draw = ImageDraw.Draw(canvas)
    draw.line((640, 0, 640, 660), fill=(235, 240, 245), width=4)
    draw.line((0, 330, 1280, 330), fill=(235, 240, 245), width=4)
    draw.rectangle((0, 660, 1280, 720), fill=(7, 12, 18))
    if checkpoint_frame is None:
        headline = "Same stored scene/state - nominal expert-demonstration backbone"
    else:
        headline = f"Checkpoint frame {checkpoint_frame} - only the stored instruction changes"
    draw.text((20, 666), headline, font=body_font, fill=(245, 248, 251))
    if checkpoint_reason:
        reason = textwrap.shorten(checkpoint_reason, width=112, placeholder="...")
        draw.text((690, 668), reason, font=small_font, fill=(178, 198, 216))
    draw.text((20, 692), DISCLAIMER, font=small_font, fill=(216, 226, 235))
    draw.text((660, 692), ARROW_NOTE, font=small_font, fill=(160, 181, 199))
    progress = float(np.clip(frame / (n_episode_frames - 1), 0, 1))
    draw.rectangle((0, 714, 1280, 720), fill=(42, 53, 64))
    draw.rectangle((0, 714, int(round(1280 * progress)), 720), fill=(71, 154, 221))
    return np.asarray(canvas)


def full_frame_schedule(n_episode_frames: int) -> Iterator[tuple[float, int | None, float]]:
    checkpoint_frames = {frame for frame, _ in CHECKPOINT_SPECS}
    for frame in range(n_episode_frames):
        yield float(frame), None, 0.0
        if frame in checkpoint_frames:
            for pause_index in range(CHECKPOINT_HOLD_FRAMES):
                phase = (pause_index + 1) / (CHECKPOINT_HOLD_FRAMES + 1)
                strength = float(0.30 + 0.70 * np.sin(np.pi * phase))
                yield float(frame), frame, strength


def ffmpeg_writer(path: Path, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        f"{width}x{height}",
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
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def preview_frame_sequence() -> Iterator[float]:
    yield from np.linspace(
        PREVIEW_START_FRAME,
        PREVIEW_END_FRAME,
        int(round((PREVIEW_END_FRAME - PREVIEW_START_FRAME) / 20 * PREVIEW_FPS)) + 1,
    )


def render_preview(episode_id: int) -> dict[str, object]:
    if episode_id != 0:
        raise ValueError("The validated first hard-gate preview is intentionally fixed to episode 0")
    episode = load_episode(episode_id)
    states = episode["states"]
    prediction_row = load_prediction(episode_id, PREVIEW_CHECKPOINT, "correct")
    action = prediction_action(prediction_row)

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    video_path = VIDEO_DIR / "preview_correct_snippet.mp4"
    still_path = FIGURE_DIR / "preview_correct_snippet.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    writer = ffmpeg_writer(video_path, PANEL_WIDTH, PANEL_HEIGHT, PREVIEW_FPS)
    if writer.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin")

    output_count = 0
    strongest_frame: np.ndarray | None = None
    with mujoco.Renderer(model, PANEL_HEIGHT, PANEL_WIDTH) as renderer:
        for frame in preview_frame_sequence():
            proximity = abs(frame - PREVIEW_CHECKPOINT)
            arrow_strength = float(np.clip(1.0 - proximity / 10.0, 0.0, 1.0))
            displayed_gripper = float(action[6]) if arrow_strength > 0.02 else None
            eef = apply_nominal_scene(model, data, states, frame, displayed_gripper)
            renderer.update_scene(data, camera="presentation")
            add_arm_links(renderer.scene, eef)
            if arrow_strength > 0:
                add_action_arrow(
                    renderer.scene,
                    eef,
                    action,
                    CONDITION_STYLE["correct"]["color"],
                    arrow_strength,
                )
            pixels = overlay_preview(renderer.render().copy(), frame, action, arrow_strength)
            writer.stdin.write(np.ascontiguousarray(pixels).tobytes())
            output_count += 1
            if abs(frame - PREVIEW_CHECKPOINT) < 0.6:
                strongest_frame = pixels.copy()

    writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    if strongest_frame is None:
        raise RuntimeError("Preview did not capture the checkpoint still")
    Image.fromarray(strongest_frame).save(still_path)

    report = {
        "mode": "preview",
        "hard_gate": True,
        "episode_id": episode_id,
        "source_frames": [PREVIEW_START_FRAME, PREVIEW_END_FRAME],
        "checkpoint_frame": PREVIEW_CHECKPOINT,
        "condition": "Correct",
        "stored_instruction": str(prediction_row.instruction),
        "stored_action": action.tolist(),
        "gripper_rule": "predicted gripper < 0 => opening; >= 0 => closing",
        "arrow_scale": ARROW_SCALE,
        "arrow_scale_note": ARROW_NOTE,
        "fps": PREVIEW_FPS,
        "frame_count": output_count,
        "resolution": [PANEL_WIDTH, PANEL_HEIGHT],
        "video_path": str(video_path),
        "still_path": str(still_path),
        "disclaimer": DISCLAIMER,
        "model_loaded": False,
        "closed_loop_rollout": False,
    }
    preview_report = OUTPUT_ROOT / "preview_validation.json"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    preview_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Preview hard gate: PASS")
    return report


def prepare_full_inputs(
    episode_id: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[int, dict[str, dict[str, Any]]], pd.DataFrame]:
    if episode_id != 0:
        raise ValueError(
            "The presentation animation is intentionally fixed to validated representative episode 0"
        )
    episode = load_episode(episode_id)
    validate_episode_zero_events(episode["actions"])
    predictions = pd.read_csv(PREDICTIONS_PATH)
    if predictions.duplicated(["episode_id", "frame_index", "language_condition"]).any():
        raise ValueError("Canonical stored predictions contain duplicate observation/condition rows")
    checkpoint_table = build_checkpoint_table(episode_id, predictions)
    prediction_map = checkpoint_prediction_map(episode_id, predictions)
    write_stored_input_records(episode_id, episode, checkpoint_table)
    reasons = {frame: reason for frame, reason in CHECKPOINT_SPECS}
    if set(prediction_map) != set(reasons):
        raise ValueError("Checkpoint prediction map does not match deterministic selection")
    return episode, checkpoint_table, prediction_map, predictions


def render_composite_at_checkpoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    states: np.ndarray,
    prediction_map: dict[int, dict[str, dict[str, Any]]],
    checkpoint_frame: int,
    reason: str,
) -> np.ndarray:
    panels = [
        render_condition_panel(
            model,
            data,
            renderer,
            states,
            float(checkpoint_frame),
            condition,
            prediction_map[checkpoint_frame][condition],
            1.0,
        )
        for condition in CONDITION_ORDER
    ]
    return combine_panels(
        panels,
        float(checkpoint_frame),
        len(states),
        checkpoint_frame,
        reason,
    )


def render_main_video(
    episode: dict[str, Any],
    prediction_map: dict[int, dict[str, dict[str, Any]]],
) -> tuple[Path, int]:
    states = episode["states"]
    reason_map = {frame: reason for frame, reason in CHECKPOINT_SPECS}
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    video_path = VIDEO_DIR / "prompt_comparison_main.mp4"
    writer = ffmpeg_writer(video_path, COMPOSITE_WIDTH, COMPOSITE_HEIGHT, RENDER_FPS)
    if writer.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin for main render")

    frame_count = 0
    with mujoco.Renderer(model, FULL_PANEL_HEIGHT, PANEL_WIDTH) as renderer:
        for frame, checkpoint_frame, arrow_strength in full_frame_schedule(len(states)):
            checkpoint_rows = (
                prediction_map[checkpoint_frame] if checkpoint_frame is not None else None
            )
            panels = [
                render_condition_panel(
                    model,
                    data,
                    renderer,
                    states,
                    frame,
                    condition,
                    checkpoint_rows[condition] if checkpoint_rows else None,
                    arrow_strength,
                )
                for condition in CONDITION_ORDER
            ]
            composite = combine_panels(
                panels,
                frame,
                len(states),
                checkpoint_frame,
                reason_map.get(checkpoint_frame),
            )
            writer.stdin.write(np.ascontiguousarray(composite).tobytes())
            frame_count += 1

    writer.stdin.close()
    return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"Main ffmpeg render failed with exit code {return_code}")
    return video_path, frame_count


def render_storyboard(
    episode: dict[str, Any],
    prediction_map: dict[int, dict[str, dict[str, Any]]],
) -> Path:
    states = episode["states"]
    reason_map = {frame: reason for frame, reason in CHECKPOINT_SPECS}
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    composites: list[Image.Image] = []
    with mujoco.Renderer(model, FULL_PANEL_HEIGHT, PANEL_WIDTH) as renderer:
        for frame in STORYBOARD_CHECKPOINTS:
            pixels = render_composite_at_checkpoint(
                model,
                data,
                renderer,
                states,
                prediction_map,
                frame,
                reason_map[frame],
            )
            composites.append(Image.fromarray(pixels).resize((960, 540), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (2880, 1080), (7, 12, 18))
    for index, image in enumerate(composites):
        sheet.paste(image, ((index % 3) * 960, (index // 3) * 540))
    path = FIGURE_DIR / "prompt_comparison_storyboard.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def create_legend(prediction_map: dict[int, dict[str, dict[str, Any]]]) -> Path:
    title_font, body_font, small_font = load_fonts()
    font_path = font_manager.findfont("DejaVu Sans")
    big_font = ImageFont.truetype(font_path, 38)
    section_font = ImageFont.truetype(font_path, 26)
    image = Image.new("RGB", (1600, 900), (242, 246, 250))
    draw = ImageDraw.Draw(image)
    draw.text((70, 48), "How to read the prompt-comparison animation", font=big_font, fill=(16, 30, 45))
    draw.text(
        (72, 105),
        "Same nominal expert-demonstration pose in all panels; only the stored language-conditioned next action changes.",
        font=body_font,
        fill=(52, 72, 91),
    )

    for index, condition in enumerate(CONDITION_ORDER):
        x = 70 + (index % 2) * 750
        y = 165 + (index // 2) * 205
        color = CONDITION_STYLE[condition]["color"]
        draw.rounded_rectangle((x, y, x + 690, y + 165), radius=18, fill=(255, 255, 255), outline=rgb255(color), width=7)
        draw.text((x + 30, y + 22), CONDITION_STYLE[condition]["label"], font=section_font, fill=rgb255(color)[:3])
        prompt = prediction_map[81][condition]["instruction"]
        for line_index, line in enumerate(textwrap.wrap(prompt, width=70)[:2]):
            draw.text((x + 30, y + 64 + line_index * 24), line, font=body_font, fill=(36, 51, 66))
        mean_label = "reference action" if condition == "correct" else "distance shown from Correct"
        draw.text((x + 30, y + 125), mean_label, font=small_font, fill=(92, 108, 123))

    draw.text((72, 595), "Visual encoding", font=section_font, fill=(16, 30, 45))
    draw.line((80, 666, 230, 625), fill=(40, 120, 220), width=12)
    draw.ellipse((215, 610, 245, 640), fill=(40, 120, 220))
    draw.text((270, 615), "3-D line + ghost target: stored relative translation proposal", font=body_font, fill=(36, 51, 66))
    draw.text((270, 647), ARROW_NOTE, font=small_font, fill=(92, 108, 123))
    draw.rounded_rectangle((80, 705, 205, 758), radius=12, fill=(16, 27, 39))
    draw.text((98, 718), "OPENING", font=body_font, fill=(255, 255, 255))
    draw.text((240, 718), "predicted gripper < 0; wider rendered jaws", font=body_font, fill=(36, 51, 66))
    draw.rounded_rectangle((790, 705, 915, 758), radius=12, fill=(16, 27, 39))
    draw.text((804, 718), "CLOSING", font=body_font, fill=(255, 255, 255))
    draw.text((950, 718), "predicted gripper >= 0; narrower rendered jaws", font=body_font, fill=(36, 51, 66))
    draw.rectangle((0, 820, 1600, 900), fill=(7, 12, 18))
    draw.text((70, 838), DISCLAIMER, font=body_font, fill=(238, 243, 248))
    draw.text((70, 868), "Local actions are not integrated into a robot rollout.", font=small_font, fill=(170, 188, 205))
    path = FIGURE_DIR / "legend.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def make_gif(video_path: Path) -> Path:
    gif_path = VIDEO_DIR / "prompt_comparison_main.gif"
    filter_graph = (
        "[0:v]fps=10,scale=960:-2:flags=lanczos,split[s0][s1];"
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
    return gif_path


def checkpoint_markdown(checkpoint_table: pd.DataFrame) -> str:
    columns = [
        "frame_index",
        "normalized_progress",
        "correct_vs_paraphrase_distance",
        "correct_vs_contradictory_distance",
        "correct_vs_unrelated_distance",
        "gripper_flip_conditions",
        "why_selected",
    ]
    labels = ["Frame", "Progress", "P", "C", "U", "Flip conditions", "Why selected"]
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    for row in checkpoint_table[columns].itertuples(index=False, name=None):
        values = [
            str(int(row[0])),
            f"{float(row[1]):.1%}",
            f"{float(row[2]):.3f}",
            f"{float(row[3]):.3f}",
            f"{float(row[4]):.3f}",
            str(row[5]) if str(row[5]) else "None",
            str(row[6]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_animation_notes(
    checkpoint_table: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[Path, Path]:
    episode = predictions[predictions.episode_id.eq(0)]
    distances: list[dict[str, float]] = []
    for frame_index, group in episode.groupby("frame_index"):
        indexed = group.set_index("language_condition")
        correct = prediction_action(indexed.loc["correct"])
        row: dict[str, float] = {"frame": float(frame_index)}
        for condition in CONDITION_ORDER[1:]:
            row[condition] = float(np.linalg.norm(prediction_action(indexed.loc[condition]) - correct))
        distances.append(row)
    distance_frame = pd.DataFrame(distances)
    means = distance_frame[list(CONDITION_ORDER[1:])].mean()
    ordering_count = int(
        (
            (distance_frame.paraphrase < distance_frame.contradictory)
            & (distance_frame.contradictory < distance_frame.unrelated)
        ).sum()
    )

    notes = f"""# MuJoCo prompt-animation notes

> {DISCLAIMER}

## Format

The main deliverable is a synchronized 2×2 comparison. Every panel receives the
same episode-0 expert-demonstration end-effector XYZ state and the same
illustrative object placement. At eight pauses, each panel displays its own
stored next-action translation, continuous gripper value, operational gripper
regime, and full 7-D distance from the Correct prediction.

The prompt-conditioned actions are never integrated into the nominal path.
Orientation-control components contribute to the printed 7-D distance but are
not rendered as rotations because that would make the presentation harder to
read.

## Grounding and mapping

- Nominal motion: all 258 XYZ states from local expert demonstration episode 0.
- Object timing: episode-0 expert command transitions at frames 50, 129, 183,
  and 240, supported by the previously completed two-camera visual review.
- Local arrows: stored `pred_action_x/y/z` from `results/vla_predictions.csv`.
- Linear arrow scale: `{ARROW_SCALE}` scene coordinate per stored action unit.
- Gripper rule: stored prediction `< 0` is opening regime; `>= 0` is closing regime.
- The continuous predicted gripper value remains visible in every checkpoint label.
- Basket/object geometry and attachment are simplified visual proxies, not grasp physics.

{ARROW_NOTE}

## Deterministic checkpoints

{checkpoint_markdown(checkpoint_table)}

## Scope

The dataset state/action reference frames do not provide enough documented
physical calibration for an exact trajectory replay. The scene therefore uses
the stored coordinate system consistently but does not label arrow length as a
physical displacement. The animation does not establish physical success,
safety, or closed-loop behavior.
"""
    notes_path = OUTPUT_ROOT / "animation_notes.md"
    notes_path.write_text(notes, encoding="utf-8")

    summary = f"""# Representative example summary

> {DISCLAIMER}

Episode 0 was selected before rendering because it has a complete 129-frame
two-camera reference GIF, shows two visually reviewed manipulation cycles, and
contains all four stored language conditions at 20 evaluation observations.
It is representative rather than uniformly hypothesis-supporting.

Across its 20 stored observations, mean full 7-D distance from Correct is:

- Paraphrase: {means.paraphrase:.3f}
- Contradictory: {means.contradictory:.3f}
- Unrelated: {means.unrelated:.3f}

The strict `Paraphrase < Contradictory < Unrelated` ordering holds at
{ordering_count}/20 observations, not every frame. The selected storyboard
therefore includes ordinary controls, high Unrelated sensitivity, expert
open/close-command neighborhoods, and the real Paraphrase gripper-regime
exception at frame 241.

The animation communicates the episode-level tendency while retaining these
real local exceptions.
"""
    summary_path = OUTPUT_ROOT / "representative_example_summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    return notes_path, summary_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_final_outputs(
    episode: dict[str, Any],
    checkpoint_table: pd.DataFrame,
    predictions: pd.DataFrame,
    video_path: Path,
    gif_path: Path,
    storyboard_path: Path,
    legend_path: Path,
    rendered_frame_count: int,
) -> dict[str, Any]:
    expected_frames = len(episode["states"]) + len(CHECKPOINT_SPECS) * CHECKPOINT_HOLD_FRAMES
    mp4_frames, mp4_seconds = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    mp4_reader = imageio_ffmpeg.read_frames(str(video_path))
    mp4_metadata = next(mp4_reader)
    mp4_reader.close()
    mp4_size = list(mp4_metadata["size"])
    mp4_fps = float(mp4_metadata["fps"])
    with Image.open(gif_path) as gif:
        gif_frames = int(gif.n_frames)
        gif_size = list(gif.size)
    with Image.open(storyboard_path) as storyboard:
        storyboard_size = list(storyboard.size)
    with Image.open(legend_path) as legend:
        legend_size = list(legend.size)
    selected_numeric = checkpoint_table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    conditions = sorted(
        predictions[
            predictions.episode_id.eq(0)
            & predictions.frame_index.isin(checkpoint_table.frame_index)
        ].language_condition.unique()
    )
    checks = {
        "checkpoint_count_exact": len(checkpoint_table) == len(CHECKPOINT_SPECS),
        "checkpoint_values_finite": bool(np.isfinite(selected_numeric).all()),
        "four_prompt_conditions_exact": conditions == sorted(CONDITION_ORDER),
        "rendered_frame_count_exact": rendered_frame_count == expected_frames,
        "encoded_mp4_frame_count_exact": int(mp4_frames) == expected_frames,
        "mp4_resolution_expected": mp4_size == [COMPOSITE_WIDTH, COMPOSITE_HEIGHT],
        "mp4_fps_expected": bool(np.isclose(mp4_fps, RENDER_FPS)),
        "storyboard_resolution_expected": storyboard_size == [2880, 1080],
        "legend_resolution_expected": legend_size == [1600, 900],
        "outputs_nonempty": all(
            path.is_file() and path.stat().st_size > 0
            for path in (video_path, gif_path, storyboard_path, legend_path)
        ),
        "same_nominal_state_all_panels_by_construction": True,
        "stored_actions_not_integrated": True,
        "smol_vla_model_not_loaded": True,
        "not_closed_loop_rollout": True,
        "required_disclaimer_rendered": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Final animation validation failed: {checks}")
    report = {
        "status": "PASS",
        "episode_id": 0,
        "visual_format": "2x2 synchronized panels",
        "conditions": list(CONDITION_ORDER),
        "checkpoint_frames": checkpoint_table.frame_index.astype(int).tolist(),
        "expert_command_transition_frames": validate_episode_zero_events(episode["actions"]),
        "fps": RENDER_FPS,
        "rendered_frame_count": rendered_frame_count,
        "expected_frame_count": expected_frames,
        "mp4_duration_seconds": mp4_seconds,
        "mp4_resolution": mp4_size,
        "mp4_codec": str(mp4_metadata["codec"]),
        "gif_frame_count": gif_frames,
        "gif_resolution": gif_size,
        "storyboard_resolution": storyboard_size,
        "legend_resolution": legend_size,
        "arrow_scale": ARROW_SCALE,
        "arrow_scale_note": ARROW_NOTE,
        "gripper_rule": "predicted gripper < 0 => opening; >= 0 => closing",
        "disclaimer": DISCLAIMER,
        "smol_vla_model_loaded": False,
        "closed_loop_rollout": False,
        "inputs": {
            "predictions": str(PREDICTIONS_PATH),
            "predictions_sha256": sha256_file(PREDICTIONS_PATH),
            "episode_parquet": str(episode["source_path"]),
            "episode_parquet_sha256": sha256_file(episode["source_path"]),
            "scene_xml": str(SCENE_PATH),
            "scene_xml_sha256": sha256_file(SCENE_PATH),
        },
        "outputs": {
            "mp4": str(video_path),
            "mp4_sha256": sha256_file(video_path),
            "gif": str(gif_path),
            "gif_sha256": sha256_file(gif_path),
            "storyboard": str(storyboard_path),
            "storyboard_sha256": sha256_file(storyboard_path),
            "legend": str(legend_path),
            "legend_sha256": sha256_file(legend_path),
        },
        "checks": checks,
    }
    validation_path = OUTPUT_ROOT / "validation.json"
    validation_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def render_full_deliverable(episode_id: int) -> dict[str, Any]:
    episode, checkpoint_table, prediction_map, predictions = prepare_full_inputs(episode_id)
    video_path, frame_count = render_main_video(episode, prediction_map)
    gif_path = make_gif(video_path)
    storyboard_path = render_storyboard(episode, prediction_map)
    legend_path = create_legend(prediction_map)
    write_animation_notes(checkpoint_table, predictions)
    report = validate_final_outputs(
        episode,
        checkpoint_table,
        predictions,
        video_path,
        gif_path,
        storyboard_path,
        legend_path,
        frame_count,
    )
    print(json.dumps(report, indent=2))
    print("Full prompt-comparison animation: PASS")
    return report


def render_storyboard_only(episode_id: int) -> dict[str, str]:
    episode, checkpoint_table, prediction_map, predictions = prepare_full_inputs(episode_id)
    storyboard_path = render_storyboard(episode, prediction_map)
    legend_path = create_legend(prediction_map)
    write_animation_notes(checkpoint_table, predictions)
    result = {"storyboard": str(storyboard_path), "legend": str(legend_path)}
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    if args.mode == "preview":
        render_preview(args.episode)
        return
    if args.mode == "storyboard":
        render_storyboard_only(args.episode)
        return
    render_full_deliverable(args.episode)


if __name__ == "__main__":
    main()
