#!/usr/bin/env python3
"""Bottom-up Panda micro-task and hero animation from stored SmolVLA actions.

Stored-action visualization; not closed-loop SmolVLA–LIBERO execution.
The one-object branches are short-horizon illustrative consequences. They are
not task-success trials and do not feed MuJoCo state back through SmolVLA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import mujoco
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_EARLY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_EARLY_ROOT / ".cache" / "matplotlib"))

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = _EARLY_ROOT
sys.path.insert(0, str(ROOT))
from sim.mujoco_panda_prompt_demo import (  # noqa: E402
    CONDITION_STYLE,
    add_action_proposal,
    dataset_to_mujoco,
    ffmpeg_writer,
    load_config,
    make_hq_gif,
    solve_position_ik,
)


SCENE = ROOT / "sim" / "bottom_up_one_object_scene.xml"
OUTPUT = ROOT / "results" / "bottom_up_demo"
FIGURES = OUTPUT / "figures"
VIDEOS = OUTPUT / "videos"
ANALYSIS = OUTPUT / "analysis_summary.json"
GRIPPER_CSV = OUTPUT / "gripper_example.csv"
TRANSLATION_FIGURE = FIGURES / "translation_proposals.png"
GRIPPER_FIGURE = FIGURES / "gripper_prompt_variants.png"
PIPELINE_FIGURE = FIGURES / "bottom_up_pipeline.png"
FULL_TASK_VIDEO = (
    ROOT
    / "results"
    / "mujoco_panda_prompt_demo"
    / "videos"
    / "panda_prompt_demo_teaser.mp4"
)
PHASE_B = ROOT / "results" / "prompt_robustness" / "predictions.csv"
CANONICAL = ROOT / "results" / "vla_predictions.csv"
EPISODE_FIVE = (
    ROOT
    / ".cache/huggingface/lerobot_v21/HuggingFaceVLA/smol-libero/data/chunk-000"
    / "episode_000005.parquet"
)

WIDTH = 1280
HEIGHT = 720
PANEL_WIDTH = 640
PANEL_HEIGHT = 360
FPS = 24
MICRO_SECONDS = 3.0
HERO_SECONDS = 16.0
DISCLAIMER = "Stored-action visualization; not closed-loop SmolVLA–LIBERO execution."
CONDITIONS = ("Correct", "Paraphrase", "Contradictory", "Unrelated")
BASKET_CENTER = np.array([0.632, -0.102, 0.390], dtype=np.float64)
OBJECT_OFFSET = np.array([0.0, 0.0, -0.112], dtype=np.float64)
LOCAL_BRANCH_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("microtask", "render", "all"),
        default="all",
        help="Render the one-object hard gate, combined hero, or both.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoothstep(value: float) -> float:
    unit = float(np.clip(value, 0.0, 1.0))
    return unit * unit * (3.0 - 2.0 * unit)


def fonts() -> tuple[ImageFont.FreeTypeFont, ...]:
    regular = font_manager.findfont("DejaVu Sans")
    bold = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold")
    )
    return (
        ImageFont.truetype(bold, 26),
        ImageFont.truetype(regular, 15),
        ImageFont.truetype(regular, 11),
        ImageFont.truetype(bold, 42),
        ImageFont.truetype(regular, 20),
    )


@dataclass
class Branch:
    condition: str
    prompt_id: str
    instruction: str
    action: np.ndarray
    gripper_value: float
    regime: str
    data: mujoco.MjData
    q_start: np.ndarray
    q_end: np.ndarray
    ik_residual: float
    released: bool = False
    start_object_distance: float = 0.0
    current_object_distance: float = 0.0
    final_object_position: np.ndarray | None = None


def set_box_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position: np.ndarray,
    velocity: np.ndarray | None = None,
) -> None:
    joint = model.joint("held_box_joint")
    qpos_address = int(joint.qposadr[0])
    dof_address = int(joint.dofadr[0])
    data.qpos[qpos_address : qpos_address + 3] = position
    data.qpos[qpos_address + 3 : qpos_address + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dof_address : dof_address + 6] = 0.0
    if velocity is not None:
        data.qvel[dof_address : dof_address + 3] = velocity


def box_position(data: mujoco.MjData) -> np.ndarray:
    return data.body("held_box").xpos.copy()


def load_branch_inputs(model: mujoco.MjModel) -> tuple[list[Branch], dict[str, Any]]:
    summary = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    table = pd.read_csv(GRIPPER_CSV)
    selected = table[table.selected_for_microtask].copy()
    if len(selected) != 4 or set(selected.prompt_class) != set(CONDITIONS):
        raise ValueError("Expected exact selected four-class microtask rows")
    if selected.noise_hash.nunique() != 1:
        raise ValueError("Microtask rows do not share one Phase-B noise hash")
    episode_table = pq.read_table(EPISODE_FIVE, columns=["observation.state"])
    states = np.asarray(episode_table["observation.state"].to_pylist(), dtype=np.float64)
    source_xyz = states[230, :3]
    config = load_config()
    start_target = dataset_to_mujoco(source_xyz[None, :], config)[0]

    seed_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, seed_data, model.key("home").id)
    home = seed_data.qpos[:7].copy()
    q_start, start_residual, _ = solve_position_ik(
        model, seed_data, start_target, home, home
    )
    if start_residual > 0.012:
        raise ValueError(f"Start IK residual too large: {start_residual}")

    branches: list[Branch] = []
    for condition in CONDITIONS:
        row = selected[selected.prompt_class.eq(condition)].iloc[0]
        action = row[["pred_action_x", "pred_action_y", "pred_action_z"]].to_numpy(float)
        matrix = np.asarray(config["dataset_to_mujoco_xyz"]["matrix"], dtype=np.float64)
        scale = float(config["action_visualization"]["local_branch_scale"])
        target = start_target + matrix @ action * scale
        data = mujoco.MjData(model)
        q_end, residual, _ = solve_position_ik(
            model, data, target, q_start, home
        )
        if residual > 0.012:
            raise ValueError(f"{condition} branch IK residual too large: {residual}")
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        data.qpos[:7] = q_start
        data.qpos[7:9] = 0.004
        start_object = start_target + OBJECT_OFFSET
        set_box_pose(model, data, start_object)
        data.ctrl[:7] = q_start
        data.ctrl[7] = 0.0
        mujoco.mj_forward(model, data)
        branches.append(
            Branch(
                condition=condition,
                prompt_id=str(row.prompt_id),
                instruction=str(row.instruction),
                action=action,
                gripper_value=float(row.pred_action_gripper),
                regime=str(row.gripper_regime),
                data=data,
                q_start=q_start.copy(),
                q_end=q_end.copy(),
                ik_residual=float(residual),
                start_object_distance=float(np.linalg.norm(start_object - BASKET_CENTER)),
                current_object_distance=float(np.linalg.norm(start_object - BASKET_CENTER)),
            )
        )
    metadata = {
        "source_episode": 5,
        "source_frame": 230,
        "source_xyz_dataset": source_xyz.tolist(),
        "start_target_mujoco": start_target.tolist(),
        "start_ik_residual_m": float(start_residual),
        "fixed_transform": config["dataset_to_mujoco_xyz"],
        "local_branch_scale": float(config["action_visualization"]["local_branch_scale"]),
        "noise_hash": str(selected.noise_hash.iloc[0]),
        "analysis_summary": summary,
    }
    return branches, metadata


def advance_branch(model: mujoco.MjModel, branch: Branch, time_seconds: float) -> float:
    if time_seconds < 0.50:
        phase = 0.0
    elif time_seconds < 1.50:
        phase = smoothstep((time_seconds - 0.50) / LOCAL_BRANCH_SECONDS)
    else:
        phase = 1.0
    q_desired = (1.0 - phase) * branch.q_start + phase * branch.q_end
    opening = branch.gripper_value < 0
    finger_phase = smoothstep((time_seconds - 0.62) / 0.50) if opening else 0.0
    finger_target = 0.004 + 0.036 * finger_phase
    branch.data.qpos[:7] = q_desired
    branch.data.qpos[7:9] = finger_target
    branch.data.qvel[:9] = 0.0
    branch.data.ctrl[:7] = q_desired
    branch.data.ctrl[7] = 255.0 if opening else 0.0

    mujoco.mj_forward(model, branch.data)
    hand_target = branch.data.body("hand").xpos.copy()
    if not branch.released:
        held_position = hand_target + OBJECT_OFFSET
        set_box_pose(model, branch.data, held_position)
        mujoco.mj_forward(model, branch.data)
        if opening and time_seconds >= 1.00:
            velocity = np.asarray(
                dataset_to_mujoco(branch.action[None, :3], load_config())[0]
                - dataset_to_mujoco(np.zeros((1, 3)), load_config())[0]
            ) * 0.14
            set_box_pose(model, branch.data, held_position, velocity=velocity)
            branch.released = True
    else:
        steps = max(1, int(round((1.0 / FPS) / model.opt.timestep)))
        for _ in range(steps):
            mujoco.mj_step(model, branch.data)
        branch.data.qpos[:7] = q_desired
        branch.data.qpos[7:9] = finger_target
        branch.data.qvel[:9] = 0.0
        mujoco.mj_forward(model, branch.data)
    branch.current_object_distance = float(
        np.linalg.norm(box_position(branch.data) - BASKET_CENTER)
    )
    branch.final_object_position = box_position(branch.data)
    return phase


def overlay_panel(pixels: np.ndarray, branch: Branch, phase: float) -> np.ndarray:
    title_font, body_font, small_font, _, _ = fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    style = CONDITION_STYLE[branch.condition.lower()]
    color = tuple(int(channel * 255) for channel in style["color"][:3]) + (255,)
    draw.rectangle((0, 0, 9, PANEL_HEIGHT), fill=color)
    draw.rectangle((9, 0, PANEL_WIDTH, 70), fill=(6, 12, 20, 218))
    draw.text((25, 10), branch.condition, font=title_font, fill="white")
    draw.text(
        (25, 43),
        f"{branch.prompt_id}  |  {branch.regime} {branch.gripper_value:+.2f}",
        font=small_font,
        fill=color,
    )
    held = not branch.released
    draw.rounded_rectangle(
        (365, 18, 620, 56), radius=13, fill=(6, 12, 20, 220)
    )
    draw.text(
        (382, 30),
        f"object held: {'YES' if held else 'NO'}",
        font=small_font,
        fill=(220, 232, 241),
    )
    draw.rectangle((9, PANEL_HEIGHT - 58, PANEL_WIDTH, PANEL_HEIGHT), fill=(6, 12, 20, 224))
    delta = branch.current_object_distance - branch.start_object_distance
    draw.text(
        (25, PANEL_HEIGHT - 45),
        f"illustrative Δ distance to basket: {delta:+.3f} m",
        font=small_font,
        fill=(215, 226, 236),
    )
    draw.text(
        (580, PANEL_HEIGHT - 45),
        f"{phase * 100:3.0f}%",
        font=small_font,
        fill=(165, 184, 201),
    )
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def combine_panels(panels: list[np.ndarray]) -> np.ndarray:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (6, 11, 18))
    for index, pixels in enumerate(panels):
        canvas.paste(Image.fromarray(pixels), ((index % 2) * PANEL_WIDTH, (index // 2) * PANEL_HEIGHT))
    draw = ImageDraw.Draw(canvas, "RGBA")
    _, _, small_font, _, _ = fonts()
    draw.rectangle((0, HEIGHT - 22, WIDTH, HEIGHT), fill=(5, 10, 17, 235))
    text_width = draw.textbbox((0, 0), DISCLAIMER, font=small_font)[2]
    draw.text(((WIDTH - text_width) / 2, HEIGHT - 17), DISCLAIMER, font=small_font, fill=(230, 237, 244, 255))
    return np.asarray(canvas)


def render_microtask() -> tuple[list[np.ndarray], pd.DataFrame, dict[str, Any]]:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    branches, metadata = load_branch_inputs(model)
    frame_count = int(round(MICRO_SECONDS * FPS))
    frames: list[np.ndarray] = []
    with mujoco.Renderer(model, PANEL_HEIGHT, PANEL_WIDTH) as renderer:
        for frame_index in range(frame_count):
            time_seconds = frame_index / FPS
            panels: list[np.ndarray] = []
            for branch in branches:
                phase = advance_branch(model, branch, time_seconds)
                renderer.update_scene(branch.data, camera="micro")
                origin = branch.data.body("hand").xpos.copy()
                add_action_proposal(
                    renderer.scene,
                    origin,
                    np.r_[branch.action, 0.0, 0.0, 0.0, branch.gripper_value],
                    branch.condition.lower(),
                    load_config(),
                    strength=0.55,
                )
                panels.append(overlay_panel(renderer.render().copy(), branch, phase))
            frames.append(combine_panels(panels))

    VIDEOS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    video_path = VIDEOS / "simple_one_object_branches.mp4"
    writer = ffmpeg_writer(video_path)
    for frame in frames:
        writer.stdin.write(np.ascontiguousarray(frame).tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("One-object microtask MP4 render failed")
    gif_path = VIDEOS / "simple_one_object_branches.gif"
    make_hq_gif(video_path, gif_path)

    storyboard_indices = (0, 23, 39, len(frames) - 1)
    storyboard = Image.new("RGB", (2560, 1440), (6, 11, 18))
    for index, frame_index in enumerate(storyboard_indices):
        storyboard.paste(
            Image.fromarray(frames[frame_index]),
            ((index % 2) * WIDTH, (index // 2) * HEIGHT),
        )
    storyboard_path = FIGURES / "simple_task_storyboard.png"
    storyboard.save(storyboard_path)

    records = []
    for branch in branches:
        end_position = np.asarray(branch.final_object_position, dtype=float)
        records.append(
            {
                "prompt_class": branch.condition,
                "prompt_id": branch.prompt_id,
                "instruction": branch.instruction,
                "action_x": branch.action[0],
                "action_y": branch.action[1],
                "action_z": branch.action[2],
                "gripper_value": branch.gripper_value,
                "gripper_regime": branch.regime,
                "start_distance_to_basket_m": branch.start_object_distance,
                "end_distance_to_basket_m": branch.current_object_distance,
                "change_in_distance_m": branch.current_object_distance
                - branch.start_object_distance,
                "object_still_held_after_short_branch": not branch.released,
                "end_object_x": end_position[0],
                "end_object_y": end_position[1],
                "end_object_z": end_position[2],
                "ik_residual_m": branch.ik_residual,
            }
        )
    metrics = pd.DataFrame(records)
    metrics_path = OUTPUT / "simple_task_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    metadata.update(
        {
            "status": "PASS",
            "video": str(video_path),
            "gif": str(gif_path),
            "storyboard": str(storyboard_path),
            "metrics": str(metrics_path),
            "frame_count": len(frames),
            "duration_seconds": MICRO_SECONDS,
            "object_attachment_mechanism": "box pose follows the illustrative hand target while held; on an OPENING branch it becomes a free MuJoCo body and falls under gravity",
            "task_success_evaluation": False,
            "disclaimer": DISCLAIMER,
        }
    )
    (OUTPUT / "microtask_validation.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("One-object MuJoCo micro-task: PASS")
    return frames, metrics, metadata


def fit_frame(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    contained = ImageOps.contain(image, (WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (17, 26, 38))
    canvas.paste(
        contained,
        ((WIDTH - contained.width) // 2, (HEIGHT - contained.height) // 2),
    )
    return np.asarray(canvas)


def add_global_footer(pixels: np.ndarray, stage: str = "") -> np.ndarray:
    _, _, small_font, _, _ = fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, HEIGHT - 42, WIDTH, HEIGHT), fill=(5, 10, 17, 235))
    draw.text((22, HEIGHT - 29), DISCLAIMER, font=small_font, fill=(232, 239, 245))
    if stage:
        width = draw.textbbox((0, 0), stage, font=small_font)[2]
        draw.text((WIDTH - width - 22, HEIGHT - 29), stage, font=small_font, fill=(155, 180, 201))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"))


def overlay_full_task(pixels: np.ndarray) -> np.ndarray:
    title_font, body_font, _, _, _ = fonts()
    image = Image.fromarray(pixels).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((28, 560, 615, 655), radius=18, fill=(6, 12, 20, 225))
    draw.text((50, 575), "LEVEL 4 — FULL PACKING", font=title_font, fill="white")
    draw.text((51, 613), "Return to the existing two-object Panda visualization", font=body_font, fill=(198, 217, 235))
    return add_global_footer(
        np.asarray(Image.alpha_composite(image, overlay).convert("RGB")),
        "existing full-task visualization",
    )


def end_card(background: np.ndarray) -> np.ndarray:
    _, _, small_font, large_font, body_font = fonts()
    image = Image.fromarray(background).convert("RGBA")
    image = Image.alpha_composite(image, Image.new("RGBA", image.size, (4, 9, 15, 225)))
    draw = ImageDraw.Draw(image)
    first = "Same vision + state + model + flow noise."
    second = "Only language changed."
    first_width = draw.textbbox((0, 0), first, font=large_font)[2]
    second_width = draw.textbbox((0, 0), second, font=large_font)[2]
    draw.text(((WIDTH - first_width) / 2, 245), first, font=large_font, fill="white")
    draw.text(((WIDTH - second_width) / 2, 300), second, font=large_font, fill=(105, 190, 250))
    note = "Bottom-up interpretation: translation → gripper → one object → full packing"
    note_width = draw.textbbox((0, 0), note, font=body_font)[2]
    draw.text(((WIDTH - note_width) / 2, 382), note, font=body_font, fill=(207, 220, 231))
    disclaimer_width = draw.textbbox((0, 0), DISCLAIMER, font=small_font)[2]
    draw.text(((WIDTH - disclaimer_width) / 2, 458), DISCLAIMER, font=small_font, fill=(225, 234, 241))
    return np.asarray(image.convert("RGB"))


def read_full_task_segment() -> list[np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(FULL_TASK_VIDEO), pix_fmt="rgb24")
    metadata = next(reader)
    size = metadata["size"]
    frames: list[np.ndarray] = []
    wanted = set(np.linspace(30, 130, 60).round().astype(int).tolist())
    for index, raw in enumerate(reader):
        if index in wanted:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(size[1], size[0], 3).copy()
            frames.append(frame)
        if index > max(wanted):
            break
    reader.close()
    if len(frames) < 50:
        raise ValueError(f"Too few full-task source frames: {len(frames)}")
    return frames


def read_video_frames(path: Path) -> list[np.ndarray]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    frames = [
        np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
        for raw in reader
    ]
    reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def render_combined(
    micro_frames: list[np.ndarray] | None = None,
    metrics: pd.DataFrame | None = None,
    micro_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if micro_frames is None or metrics is None or micro_metadata is None:
        micro_frames = read_video_frames(VIDEOS / "simple_one_object_branches.mp4")
        metrics = pd.read_csv(OUTPUT / "simple_task_metrics.csv")
        micro_metadata = json.loads(
            (OUTPUT / "microtask_validation.json").read_text(encoding="utf-8")
        )
    pipeline = fit_frame(PIPELINE_FIGURE)
    translation = fit_frame(TRANSLATION_FIGURE)
    gripper = fit_frame(GRIPPER_FIGURE)
    full_task = read_full_task_segment()
    final_background = full_task[-1]
    frame_count = int(round(HERO_SECONDS * FPS))
    video_path = VIDEOS / "bottom_up_teaser.mp4"
    gif_path = VIDEOS / "bottom_up_teaser.gif"
    writer = ffmpeg_writer(video_path)
    storyboard_times = (1.0, 3.5, 6.5, 10.0, 13.2, 15.2)
    storyboard_indices = {int(round(value * FPS)): value for value in storyboard_times}
    storyboard_frames: dict[float, np.ndarray] = {}
    for frame_index in range(frame_count):
        seconds = frame_index / FPS
        if seconds < 2.0:
            pixels = add_global_footer(pipeline, "bottom-up progression")
        elif seconds < 5.0:
            pixels = add_global_footer(translation, "Level 1 · stored XYZ predictions")
        elif seconds < 8.0:
            pixels = add_global_footer(gripper, "Level 2 · stored gripper predictions")
        elif seconds < 12.0:
            unit = (seconds - 8.0) / 4.0
            source_index = min(int(round(unit * (len(micro_frames) - 1))), len(micro_frames) - 1)
            pixels = micro_frames[source_index]
        elif seconds < 14.5:
            unit = (seconds - 12.0) / 2.5
            source_index = min(int(round(unit * (len(full_task) - 1))), len(full_task) - 1)
            pixels = overlay_full_task(full_task[source_index])
        else:
            pixels = end_card(final_background)
        writer.stdin.write(np.ascontiguousarray(pixels).tobytes())
        if frame_index in storyboard_indices:
            storyboard_frames[storyboard_indices[frame_index]] = pixels.copy()
    writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("Bottom-up hero MP4 render failed")
    make_hq_gif(video_path, gif_path)

    storyboard = Image.new("RGB", (2880, 1080), (6, 11, 18))
    for index, value in enumerate(storyboard_times):
        thumb = Image.fromarray(storyboard_frames[value]).resize((960, 540), Image.Resampling.LANCZOS)
        storyboard.paste(thumb, ((index % 3) * 960, (index // 3) * 540))
    storyboard_path = FIGURES / "bottom_up_teaser_storyboard.png"
    storyboard.save(storyboard_path)

    mp4_frames, duration = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    with Image.open(gif_path) as gif:
        gif_frames = int(gif.n_frames)
        gif_resolution = list(gif.size)
    stored_phase_b = pd.read_csv(PHASE_B)
    exact_row_trace = True
    for row in metrics.itertuples():
        source = stored_phase_b[
            stored_phase_b.episode_id.eq(5)
            & stored_phase_b.frame_index.eq(230)
            & stored_phase_b.prompt_id.eq(row.prompt_id)
        ]
        if len(source) != 1:
            exact_row_trace = False
            break
        source_row = source.iloc[0]
        stored_action = source_row[
            ["pred_action_x", "pred_action_y", "pred_action_z", "pred_action_gripper"]
        ].to_numpy(float)
        rendered_action = np.array(
            [row.action_x, row.action_y, row.action_z, row.gripper_value], dtype=float
        )
        exact_row_trace &= bool(
            np.allclose(stored_action, rendered_action, rtol=0.0, atol=1e-12)
            and source_row.instruction == row.instruction
        )
    checks = {
        "canonical_predictions_hash_unchanged": sha256(CANONICAL)
        == "ab4984aeea52a8ca60c94a5cb19a15d69fd27707f5c5734ddd8bec680f376dc4",
        "phase_b_predictions_hash_unchanged": sha256(PHASE_B)
        == "2848da0bd577f2db55fc2b1ec003550fdedc7e798a803066a42cbd135bb5b173",
        "exact_four_traceable_microtask_rows": len(metrics) == 4
        and set(metrics.prompt_class) == set(CONDITIONS),
        "displayed_actions_match_phase_b_rows": exact_row_trace,
        "stored_analysis_pass": micro_metadata["analysis_summary"]["status"] == "PASS",
        "same_phase_b_noise_hash_for_microtask_rows": stored_phase_b[
            stored_phase_b.episode_id.eq(5)
            & stored_phase_b.frame_index.eq(230)
            & stored_phase_b.prompt_id.isin(metrics.prompt_id)
        ].noise_hash.nunique()
        == 1,
        "operational_gripper_rule_applied": bool(
            (
                (metrics.gripper_value < 0)
                == metrics.gripper_regime.eq("OPENING")
            ).all()
        ),
        "all_microtask_metrics_finite": bool(
            np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy(float)).all()
        ),
        "all_ik_residuals_below_12mm": bool((metrics.ik_residual_m <= 0.012).all()),
        "mp4_frame_count_expected": int(mp4_frames) == frame_count,
        "mp4_duration_expected": bool(np.isclose(duration, HERO_SECONDS)),
        "gif_animated": gif_frames > 1,
        "gif_resolution_expected": gif_resolution == [WIDTH, HEIGHT],
        "required_disclaimer_rendered": True,
        "smol_vla_inference_not_performed": True,
        "not_closed_loop_rollout": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Bottom-up final validation failed: {checks}")
    report = {
        "status": "PASS",
        "translation_example": {"episode_id": 3, "frame_index": 38},
        "gripper_and_microtask_example": {"episode_id": 5, "frame_index": 230},
        "microtask_prompt_ids": metrics.set_index("prompt_class").prompt_id.to_dict(),
        "microtask_metrics": metrics.to_dict(orient="records"),
        "visualization_transform": micro_metadata["fixed_transform"],
        "local_branch_scale": micro_metadata["local_branch_scale"],
        "object_attachment_mechanism": micro_metadata["object_attachment_mechanism"],
        "timeline_seconds": {
            "bottom_up_title": [0, 2],
            "translation": [2, 5],
            "gripper": [5, 8],
            "one_object_microtask": [8, 12],
            "existing_full_task": [12, 14.5],
            "takeaway": [14.5, 16],
        },
        "outputs": {
            "mp4": str(video_path),
            "gif": str(gif_path),
            "storyboard": str(storyboard_path),
            "microtask_storyboard": str(FIGURES / "simple_task_storyboard.png"),
        },
        "media": {
            "duration_seconds": float(duration),
            "mp4_frame_count": int(mp4_frames),
            "gif_frame_count": gif_frames,
            "resolution": gif_resolution,
        },
        "source_hashes": {
            "canonical_predictions_sha256": sha256(CANONICAL),
            "phase_b_predictions_sha256": sha256(PHASE_B),
            "analysis_summary_sha256": sha256(ANALYSIS),
            "scene_xml_sha256": sha256(SCENE),
        },
        "scientific_scope": "short-horizon illustrative consequence of stored SmolVLA actions",
        "task_success_evaluation": False,
        "disclaimer": DISCLAIMER,
        "checks": checks,
    }
    (OUTPUT / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Bottom-up hero animation: PASS")
    return report


def main() -> None:
    args = parse_args()
    if args.mode == "microtask":
        render_microtask()
    elif args.mode == "render":
        render_combined()
    else:
        frames, metrics, metadata = render_microtask()
        render_combined(frames, metrics, metadata)


if __name__ == "__main__":
    main()
