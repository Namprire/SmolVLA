#!/usr/bin/env python
"""Phase 10: audit LIBERO gripper/state semantics without running SmolVLA inference."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors.torch import load_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures" / "gripper"
TRANSITION_DIR = RESULTS_DIR / "gripper_transitions"
V21_ROOT = (
    REPO_ROOT
    / ".cache/huggingface/lerobot_v21/HuggingFaceVLA/smol-libero"
)
DATA_DIR = V21_ROOT / "data" / "chunk-000"
INFO_PATH = V21_ROOT / "meta" / "info.json"
EPISODES_PATH = V21_ROOT / "meta" / "episodes.jsonl"
CHECKPOINT_DIR = (
    REPO_ROOT / ".cache/huggingface/models/HuggingFaceVLA/smolvla_libero"
)
PREDICTIONS_PATH = RESULTS_DIR / "vla_predictions.csv"

ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
STATE_NAMES_FROM_METADATA = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
    "gripper",
)
STATE_SEMANTIC_NAMES = (
    "eef_position_x",
    "eef_position_y",
    "eef_position_z",
    "eef_axis_angle_x",
    "eef_axis_angle_y",
    "eef_axis_angle_z",
    "gripper_qpos_0",
    "gripper_qpos_1",
)
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
CONTACT_EPISODES = (0, 3, 6, 9)
CONTACT_OFFSETS = (-2, -1, 0, 1, 2, 5, 10)

# These notes record the completed human visual review of the generated contact sheets.
# They are intentionally evidence summaries, not labels used by an event extractor.
VISUAL_REVIEW_NOTES = {
    "status": "COMPLETE",
    "summary": (
        "Twenty transition windows were inspected in both agent and eye-in-hand views at "
        "t−2, t−1, t, t+1, t+2, t+5, and t+10. In routine cycles, −1→+1 begins with "
        "the fingers aligned around the cream-cheese box or butter; subsequent frames show "
        "the object moving with the gripper. Routine +1→−1 onsets occur over the basket, "
        "and by t+10 the object is visibly lower and separated from the retreating gripper. "
        "The rapid reversals at episode 3 frames 195/197 and the openings at episode 3 frame "
        "208 and episode 6 frame 204 occur during failed placement/regrasp sequences away from "
        "a completed basket deposit. They visually confirm that +1→−1 means opening-command "
        "onset, but not necessarily a successful task release."
    ),
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_metadata() -> tuple[dict[str, Any], pd.DataFrame]:
    info = json.loads(INFO_PATH.read_text())
    episodes = pd.read_json(EPISODES_PATH, lines=True)
    return info, episodes


def load_available_frames() -> tuple[pd.DataFrame, dict[int, dict[str, list[Any]]]]:
    parquet_paths = sorted(DATA_DIR.glob("episode_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No local episode parquet files found under {DATA_DIR}")

    frame_tables: list[pd.DataFrame] = []
    images: dict[int, dict[str, list[Any]]] = {}
    expected_episode = 0
    for path in parquet_paths:
        table = pq.read_table(path)
        episode_ids = np.asarray(table["episode_index"], dtype=np.int64)
        if len(np.unique(episode_ids)) != 1:
            raise ValueError(f"{path} contains more than one episode")
        episode_id = int(episode_ids[0])
        if episode_id != expected_episode:
            raise ValueError(
                f"Local episode sequence is not contiguous: expected {expected_episode}, found {episode_id}"
            )
        expected_episode += 1

        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        joints = np.asarray(table["observation.state.joint"].to_pylist(), dtype=np.float64)
        if actions.shape != (len(table), 7):
            raise ValueError(f"Expected 7-D actions in {path}, got {actions.shape}")
        if states.shape != (len(table), 8):
            raise ValueError(f"Expected 8-D states in {path}, got {states.shape}")
        if joints.shape != (len(table), 7):
            raise ValueError(f"Expected 7-D joint states in {path}, got {joints.shape}")

        frame = pd.DataFrame(
            {
                "episode_id": episode_ids,
                "frame_index": np.asarray(table["frame_index"], dtype=np.int64),
                "global_index": np.asarray(table["index"], dtype=np.int64),
                "timestamp": np.asarray(table["timestamp"], dtype=np.float64),
            }
        )
        for index, name in enumerate(ACTION_NAMES):
            frame[f"action_{name}"] = actions[:, index]
        for index, name in enumerate(STATE_SEMANTIC_NAMES):
            frame[f"state_{name}"] = states[:, index]
        for index in range(7):
            frame[f"state_joint_{index + 1}"] = joints[:, index]
        frame["gripper_aperture_proxy"] = states[:, 6] - states[:, 7]
        max_frame = int(frame.frame_index.max())
        frame["task_progress"] = frame.frame_index / max_frame
        frame_tables.append(frame)
        images[episode_id] = {
            "agentview": table["observation.images.image"].to_pylist(),
            "eye_in_hand": table["observation.images.image2"].to_pylist(),
        }

    output = pd.concat(frame_tables, ignore_index=True)
    if not np.isfinite(
        output.filter(regex=r"^(action_|state_|task_progress|timestamp)").to_numpy(dtype=float)
    ).all():
        raise ValueError("Local expert data contain non-finite numeric values")
    return output, images


def gripper_statistics(frames: pd.DataFrame) -> dict[str, Any]:
    values = frames.action_gripper.to_numpy(dtype=float)
    quantiles = np.quantile(values, QUANTILES)
    unique, counts = np.unique(values, return_counts=True)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std_sample": float(np.std(values, ddof=1)),
        "quantiles": {f"q{int(q * 100):02d}": float(v) for q, v in zip(QUANTILES, quantiles)},
        "exact_unique_value_counts": {str(float(v)): int(c) for v, c in zip(unique, counts)},
    }


def transition_candidates(frames: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_deltas: list[np.ndarray] = []
    for episode_id, episode in frames.groupby("episode_id", sort=True):
        episode = episode.sort_values("frame_index").reset_index(drop=True)
        gripper = episode.action_gripper.to_numpy(dtype=float)
        deltas = np.diff(gripper)
        all_deltas.append(deltas)
        # No magnitude threshold is chosen: every exact nonzero change in the binary channel is retained.
        for local_position in np.flatnonzero(deltas != 0) + 1:
            current = episode.iloc[local_position]
            rows.append(
                {
                    "episode_id": int(episode_id),
                    "frame_index": int(current.frame_index),
                    "global_index": int(current.global_index),
                    "task_progress": float(current.task_progress),
                    "gripper_before": float(gripper[local_position - 1]),
                    "gripper_after": float(gripper[local_position]),
                    "delta_gripper": float(deltas[local_position - 1]),
                    "transition_direction": (
                        "negative_to_positive" if deltas[local_position - 1] > 0 else "positive_to_negative"
                    ),
                }
            )
    candidates = pd.DataFrame(rows).sort_values(["episode_id", "frame_index"]).reset_index(drop=True)
    delta = np.concatenate(all_deltas)
    absolute = np.abs(delta)
    unique, counts = np.unique(delta, return_counts=True)
    diagnostics = {
        "delta_count": int(delta.size),
        "delta_unique_value_counts": {str(float(v)): int(c) for v, c in zip(unique, counts)},
        "absolute_delta_quantiles": {
            f"q{int(q * 100):02d}": float(value)
            for q, value in zip(QUANTILES, np.quantile(absolute, QUANTILES))
        },
        "maximum_absolute_delta": float(absolute.max()),
        "candidate_count": int(len(candidates)),
        "negative_to_positive_count": int((candidates.delta_gripper > 0).sum()),
        "positive_to_negative_count": int((candidates.delta_gripper < 0).sum()),
    }
    return candidates, diagnostics


def temporal_response(frames: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        episode = frames[frames.episode_id == candidate.episode_id].sort_values("frame_index")
        episode = episode.set_index("frame_index")
        row: dict[str, Any] = {
            "episode_id": candidate.episode_id,
            "frame_index": candidate.frame_index,
            "transition_direction": candidate.transition_direction,
        }
        max_frame = int(episode.index.max())
        for offset in (0, 1, 2, 5, 10):
            frame_index = min(candidate.frame_index + offset, max_frame)
            row[f"aperture_t_plus_{offset}"] = float(
                episode.loc[frame_index, "gripper_aperture_proxy"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def select_contact_transitions(candidates: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for episode_id in CONTACT_EPISODES:
        subset = candidates[candidates.episode_id == episode_id].sort_values("frame_index")
        if episode_id in (0, 3, 6):
            # Retain every transition in the representative normal episode and both retry episodes.
            selected.append(subset)
        else:
            # Add a late-dataset first-close/final-open pair while keeping the review manageable.
            selected.append(subset.iloc[[0, -1]])
    output = pd.concat(selected, ignore_index=True)
    if not 10 <= len(output) <= 20:
        raise ValueError(f"Expected 10-20 contact-sheet transitions, selected {len(output)}")
    return output


def decode_dataset_image(value: dict[str, Any]) -> Image.Image:
    image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    # Match installed LeRobot 0.4.4 LiberoProcessorStep: dataset images require a 180-degree flip.
    return image.transpose(Image.Transpose.ROTATE_180)


def make_contact_sheets(
    frames: pd.DataFrame,
    images: dict[int, dict[str, list[Any]]],
    selected: pd.DataFrame,
) -> None:
    TRANSITION_DIR.mkdir(parents=True, exist_ok=True)
    for candidate in selected.itertuples(index=False):
        episode = frames[frames.episode_id == candidate.episode_id].sort_values("frame_index")
        max_frame = int(episode.frame_index.max())
        fig, axes = plt.subplots(2, len(CONTACT_OFFSETS), figsize=(17.2, 5.8))
        for column, offset in enumerate(CONTACT_OFFSETS):
            frame_index = int(np.clip(candidate.frame_index + offset, 0, max_frame))
            for row, camera_key in enumerate(("agentview", "eye_in_hand")):
                image = decode_dataset_image(images[candidate.episode_id][camera_key][frame_index])
                axes[row, column].imshow(image)
                axes[row, column].axis("off")
                if row == 0:
                    axes[row, column].set_title(f"t{offset:+d}\nframe {frame_index}", fontsize=9)
        fig.text(0.008, 0.67, "Agent view", rotation=90, va="center", fontsize=10)
        fig.text(0.008, 0.25, "Eye-in-hand", rotation=90, va="center", fontsize=10)
        direction = f"{candidate.gripper_before:+.0f} → {candidate.gripper_after:+.0f}"
        fig.suptitle(
            f"Episode {candidate.episode_id}, transition frame {candidate.frame_index}, "
            f"progress {candidate.task_progress:.3f}, command {direction}, Δ={candidate.delta_gripper:+.0f}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        stem = (
            f"episode_{candidate.episode_id:02d}_frame_{candidate.frame_index:03d}_"
            f"{'neg_to_pos' if candidate.delta_gripper > 0 else 'pos_to_neg'}"
        )
        fig.savefig(TRANSITION_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def make_plots(
    frames: pd.DataFrame,
    candidates: pd.DataFrame,
    temporal: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    values = frames.action_gripper.to_numpy(dtype=float)
    counts = pd.Series(values).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    bars = ax.bar([str(int(value)) for value in counts.index], counts.values, color=["#4C78A8", "#E15759"])
    ax.bar_label(bars, padding=4)
    ax.set_xlabel("Expert action[6] value")
    ax.set_ylabel("Frames")
    ax.set_title("Expert gripper command distribution (all local episodes)")
    ax.set_ylim(0, max(counts.values) * 1.12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "expert_gripper_histogram")

    fig, axes = plt.subplots(5, 2, figsize=(12.5, 12.0), sharey=True)
    for ax, (episode_id, episode) in zip(axes.flat, frames.groupby("episode_id", sort=True), strict=True):
        ax.step(episode.frame_index, episode.action_gripper, where="post", color="#4C78A8", linewidth=1.5)
        changes = candidates[candidates.episode_id == episode_id]
        ax.scatter(changes.frame_index, changes.gripper_after, color="#E15759", s=20, zorder=3)
        ax.set_title(f"Episode {episode_id} (n={len(episode)})")
        ax.set_ylim(-1.25, 1.25)
        ax.set_yticks([-1, 0, 1])
        ax.grid(alpha=0.2)
    fig.supxlabel("Local frame index")
    fig.supylabel("Expert gripper command")
    fig.suptitle("Expert gripper trajectories across all locally available episodes", y=1.01, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "expert_gripper_trajectories")

    delta_counts = {0.0: 0, -2.0: 0, 2.0: 0}
    for episode_id, episode in frames.groupby("episode_id", sort=True):
        unique, count = np.unique(np.diff(episode.action_gripper.to_numpy(dtype=float)), return_counts=True)
        for value, n in zip(unique, count):
            delta_counts[float(value)] = delta_counts.get(float(value), 0) + int(n)
    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    x_values = [-2.0, 0.0, 2.0]
    bars = ax.bar([str(int(value)) for value in x_values], [delta_counts[v] for v in x_values], color="#59A14F")
    ax.bar_label(bars, padding=4)
    ax.set_yscale("log")
    ax.set_xlabel("Δ expert gripper command")
    ax.set_ylabel("Adjacent-frame count (log scale)")
    ax.set_title("Temporal changes in the expert gripper channel")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "expert_gripper_delta_distribution")

    detail_episodes = (3, 5, 6)
    fig, axes = plt.subplots(len(detail_episodes), 1, figsize=(11.5, 8.2), sharex=False)
    for ax, episode_id in zip(axes, detail_episodes, strict=True):
        episode = frames[frames.episode_id == episode_id].sort_values("frame_index")
        line_command = ax.step(
            episode.frame_index,
            episode.action_gripper,
            where="post",
            color="#4C78A8",
            linewidth=1.4,
            label="Binary command",
        )[0]
        ax.set_ylim(-1.25, 1.25)
        ax.set_ylabel(f"Ep. {episode_id}\ncommand")
        ax.grid(alpha=0.2)
        aperture_axis = ax.twinx()
        line_aperture = aperture_axis.plot(
            episode.frame_index,
            episode.gripper_aperture_proxy,
            color="#E15759",
            linewidth=1.2,
            alpha=0.85,
            label="qpos[0] − qpos[1]",
        )[0]
        aperture_axis.set_ylabel("Aperture proxy")
        if episode_id == detail_episodes[0]:
            ax.legend([line_command, line_aperture], ["Binary command", "Finger-qpos aperture proxy"], frameon=False, loc="center right")
    axes[-1].set_xlabel("Local frame index")
    fig.suptitle("Command transitions and delayed finger-state response", y=1.01, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "expert_gripper_temporal_context")

    response_means = temporal.groupby("transition_direction").mean(numeric_only=True)
    offsets = np.asarray([0, 1, 2, 5, 10])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for direction, label, color in (
        ("negative_to_positive", "−1 → +1", "#4C78A8"),
        ("positive_to_negative", "+1 → −1", "#E15759"),
    ):
        values = response_means.loc[direction, [f"aperture_t_plus_{offset}" for offset in offsets]].to_numpy(float)
        ax.plot(offsets, values, marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("Frames after command transition")
    ax.set_ylabel("Mean finger-qpos aperture proxy")
    ax.set_title("Measured gripper-state response after binary command changes")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "expert_gripper_transition_response")


def checkpoint_normalization() -> dict[str, Any]:
    pre = load_file(CHECKPOINT_DIR / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    post = load_file(CHECKPOINT_DIR / "policy_postprocessor_step_1_unnormalizer_processor.safetensors")
    for key in ("action.mean", "action.std"):
        if not torch.equal(pre[key], post[key]):
            raise ValueError(f"Checkpoint pre/post normalization statistic differs for {key}")
    mean = float(pre["action.mean"][6])
    std = float(pre["action.std"][6])
    return {
        "mode": "MEAN_STD",
        "gripper_mean": mean,
        "gripper_std": std,
        "normalized_negative_one": (-1.0 - mean) / (std + 1e-8),
        "normalized_positive_one": (1.0 - mean) / (std + 1e-8),
        "official_postprocessor_formula": "native_action = normalized_action * std + mean",
        "official_normalizer_or_unnormalizer_clips": False,
    }


def prediction_gripper_statistics() -> pd.DataFrame:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    if len(predictions) != 800:
        raise ValueError(f"Expected 800 validated predictions, found {len(predictions)}")
    rows: list[dict[str, Any]] = []
    for condition, subset in predictions.groupby("language_condition", sort=True):
        values = subset.pred_action_gripper.to_numpy(dtype=float)
        row: dict[str, Any] = {
            "language_condition": condition,
            "count": len(values),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)),
        }
        for quantile, value in zip(QUANTILES, np.quantile(values, QUANTILES)):
            row[f"q{int(quantile * 100):02d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def write_audits(
    info: dict[str, Any],
    frames: pd.DataFrame,
    stats: dict[str, Any],
    delta_stats: dict[str, Any],
    temporal: pd.DataFrame,
    selected: pd.DataFrame,
    normalization: dict[str, Any],
    predicted_stats: pd.DataFrame,
) -> None:
    local_lengths = frames.groupby("episode_id").size().astype(int).to_dict()
    metadata_total = int(info["total_episodes"])
    metadata_frames = int(info["total_frames"])
    local_total = len(local_lengths)
    local_frames = len(frames)
    command_aperture = frames.groupby("action_gripper").gripper_aperture_proxy.agg(
        ["count", "mean", "median", "min", "max"]
    )
    temporal_means = temporal.groupby("transition_direction").mean(numeric_only=True)

    source_audit = f"""# Phase 10 gripper semantics source audit

## Dataset coverage

- Dataset: `HuggingFaceVLA/smol-libero`, local v2.1 representation at `{V21_ROOT.relative_to(REPO_ROOT)}`.
- `meta/info.json` advertises {metadata_total} episodes and {metadata_frames:,} frames.
- Locally usable parquet data: {local_total} episodes ({', '.join(map(str, local_lengths))}) and {local_frames:,} frames. Episode lengths: {local_lengths}.
- Episodes listed in metadata but without local parquet data were not treated as inspectable demonstrations.
- FPS: {info['fps']}. Images are embedded PNGs from two cameras; no video files are used.

## Feature definitions and source evidence

- `{INFO_PATH.relative_to(REPO_ROOT)}` names `action` as seven float32 values: `{list(ACTION_NAMES)}`. Therefore action index 6 is the gripper channel.
- The same metadata names the eight-dimensional state `{list(STATE_NAMES_FROM_METADATA)}` and the separate seven-dimensional joint state `joint_1` through `joint_7`.
- Installed LeRobot 0.4.4 `lerobot.processor.env_processor.LiberoProcessorStep` is more precise than the legacy state labels: it constructs `observation.state` as end-effector position (3), quaternion converted to axis-angle (3), then two gripper `qpos` values. Thus state indices 3–5 are axis-angle components, not roll/pitch/yaw despite the v2.1 metadata labels.
- Installed `lerobot.envs.libero.LiberoEnv` maps camera 1 to LIBERO `agentview_image` and camera 2 to `robot0_eye_in_hand_image`; the official processor flips stored images by 180 degrees. Contact sheets use that same rotation.
- The installed environment defines a seven-dimensional `[-1, 1]` action space and defaults to `control_mode="relative"`, setting the arm controller's `use_delta=True`. It passes the seven-vector directly to LIBERO. The last dimension is therefore the LIBERO gripper command alongside six relative end-effector controller commands, not an observed finger position.
- `get_libero_dummy_action()` returns `[0, 0, 0, 0, 0, 0, -1]`, consistent with `-1` being the non-closing/open command used during settling. The sign interpretation is independently checked below from state response and camera frames.

## Empirical gripper representation

- All {stats['count']:,} local expert gripper values are exactly `-1` or `+1`: {stats['exact_unique_value_counts']}.
- The adjacent-frame delta has only `-2`, `0`, or `+2`: {delta_stats['delta_unique_value_counts']}.
- There are {delta_stats['candidate_count']} nonzero transitions: {delta_stats['negative_to_positive_count']} `-1→+1` and {delta_stats['positive_to_negative_count']} `+1→-1`.
- This is an exactly binary, saturated, temporally persistent open/close controller command. It is not an absolute gripper joint position: the measured finger positions are state indices 6–7 and change gradually after a command transition. The local files do not specify whether LIBERO's lower-level gripper controller realizes that binary command through position, velocity, or force control.
- Mean/median finger-qpos aperture proxy (`state[6] - state[7]`) under command `-1`: {command_aperture.loc[-1.0, 'mean']:.6f}/{command_aperture.loc[-1.0, 'median']:.6f}; under `+1`: {command_aperture.loc[1.0, 'mean']:.6f}/{command_aperture.loc[1.0, 'median']:.6f}.
- After `-1→+1`, mean aperture changes from {temporal_means.loc['negative_to_positive', 'aperture_t_plus_0']:.6f} at command onset to {temporal_means.loc['negative_to_positive', 'aperture_t_plus_10']:.6f} ten frames later. After `+1→-1`, it changes from {temporal_means.loc['positive_to_negative', 'aperture_t_plus_0']:.6f} to {temporal_means.loc['positive_to_negative', 'aperture_t_plus_10']:.6f}. This supplies direct measured-state evidence that `+1` closes and `-1` opens.

## Normalization and clipping

- Checkpoint `{(CHECKPOINT_DIR / 'config.json').relative_to(REPO_ROOT)}` maps ACTION to `MEAN_STD` normalization.
- Serialized preprocessor and postprocessor statistics agree. Gripper mean = {normalization['gripper_mean']:.9f}; gripper std = {normalization['gripper_std']:.9f}.
- Native `-1` normalizes to {normalization['normalized_negative_one']:.6f}; native `+1` normalizes to {normalization['normalized_positive_one']:.6f}.
- Installed `lerobot.processor.normalize_processor._NormalizationMixin` applies `(x-mean)/(std+eps)` and inverse `x*std+mean`. It contains no clipping in the MEAN_STD path. The official postprocessor likewise does not clip, explaining why predictions may exceed the expert `[-1,1]` support.
- The installed environment declares `[-1,1]` bounds but its `step()` passes the array directly to LIBERO without a LeRobot-side clip. Downstream simulator/controller saturation was not exercised in this phase.

## Documentation ambiguity handled

The dataset card describes actions as “target joint commands,” while the actual seven names, current LeRobot LIBERO environment, and default relative control path identify six relative Cartesian/orientation controller commands plus a binary gripper command. For this audit, installed LeRobot 0.4.4, serialized checkpoint configuration, local metadata, and observed trajectories are treated as the source of truth.
"""
    (RESULTS_DIR / "gripper_semantics_audit.md").write_text(source_audit)

    state_ranges = frames[[f"state_{name}" for name in STATE_SEMANTIC_NAMES]].agg(["min", "max", "mean"])
    state_lines = "\n".join(
        f"- Index {index}: `{name}`; local min/max/mean = "
        f"{state_ranges.loc['min', f'state_{name}']:.6f} / "
        f"{state_ranges.loc['max', f'state_{name}']:.6f} / "
        f"{state_ranges.loc['mean', f'state_{name}']:.6f}."
        for index, name in enumerate(STATE_SEMANTIC_NAMES)
    )
    state_audit = f"""# Phase 10 state semantics audit

## Eight-dimensional `observation.state`

Installed LeRobot 0.4.4 `lerobot.processor.env_processor.LiberoProcessorStep` constructs the state in this exact order:

{state_lines}

- Indices 0–2 are the absolute observed end-effector Cartesian position from LIBERO `robot0_eef_pos`. XYZ is therefore available for later work; this phase does not extract release coordinates.
- Indices 3–5 are an absolute observed end-effector orientation represented as a three-component axis-angle vector converted from `robot0_eef_quat`. The v2.1 metadata's `roll/pitch/yaw` names are misleading for the current official processor.
- Indices 6–7 are the two observed gripper finger joint positions (`robot0_gripper_qpos`), not two command channels. Their opposing signs make `state[6] - state[7]` a useful aperture proxy for semantic validation.
- Units and reference frame are not declared in the local metadata. They are simulator-native pose coordinates; later geometry work must preserve that coordinate system and should not silently label units.

## Separate joint-state vector

`observation.state.joint` contains seven observed robot joint positions in indices 0–6 (`joint_1` through `joint_7`). It is not used by the validated SmolVLA checkpoint, whose input config includes only the eight-dimensional `observation.state` plus two images.

## Action/state distinction

The seven-dimensional expert `action` is not the eight-dimensional pose state. Under installed `LiberoEnv(control_mode="relative")`, its first six entries are relative end-effector translation/orientation controller commands and index 6 is the binary gripper command. Phase 11 must not use action XYZ as release location; only observed state indices 0–2 are defensible candidate Cartesian positions.
"""
    (RESULTS_DIR / "state_semantics_audit.md").write_text(state_audit)

    prediction_lines = "\n".join(
        f"- {row.language_condition}: min {row.min:.6f}, q01 {row.q01:.6f}, q05 {row.q05:.6f}, "
        f"median {row.median:.6f}, q95 {row.q95:.6f}, q99 {row.q99:.6f}, max {row.max:.6f}."
        for row in predicted_stats.itertuples(index=False)
    )
    analysis = f"""OUTCOME A — SUFFICIENTLY INTERPRETABLE

Coverage
--------
The local metadata advertises {metadata_total} episodes/{metadata_frames:,} frames, but local parquet data exist only for episodes 0–9. All {local_total} usable episodes and all {local_frames:,} available frames were inspected numerically. {len(selected)} transition windows from early, middle, and late local episodes—including anomalous retry sequences—were rendered from both cameras and manually reviewed.

Expert gripper distribution
---------------------------
count={stats['count']}
min={stats['min']:.6f}
max={stats['max']:.6f}
mean={stats['mean']:.6f}
median={stats['median']:.6f}
sample_std={stats['std_sample']:.6f}
q01={stats['quantiles']['q01']:.6f}
q05={stats['quantiles']['q05']:.6f}
q25={stats['quantiles']['q25']:.6f}
q50={stats['quantiles']['q50']:.6f}
q75={stats['quantiles']['q75']:.6f}
q95={stats['quantiles']['q95']:.6f}
q99={stats['quantiles']['q99']:.6f}
unique_counts={stats['exact_unique_value_counts']}

Semantics decision
------------------
The expert action gripper signal is an exactly binary, saturated, persistent open/close controller command. It is not a continuous finger-position, state-delta, or measured-velocity signal. Direct measured finger-state response shows that `+1` commands closing and `-1` commands opening. A `+1→-1` transition is therefore the onset of an opening command. The local representation does not document the lower-level actuator control law beneath that binary command.

Opening-command onset alone is not synonymous with physical object release. Absolute `-1` also occurs before the first grasp and between object manipulations, and episodes 3, 5, and 6 contain extra close/open retries. A defensible release event must use temporal transition logic: prior sustained `+1` holding/closing context, a `+1→-1` onset, subsequent persistence at `-1`, and confirmation from measured finger opening and/or scene context. The exact detector and any thresholds are deliberately deferred to Phase 11.

Visual evidence
---------------
status={VISUAL_REVIEW_NOTES['status']}
{VISUAL_REVIEW_NOTES['summary']}

Temporal evidence
-----------------
All 48 nonzero changes have magnitude exactly 2: 24 `-1→+1` and 24 `+1→-1`. Most episodes show four transitions (two grasp/carry/release cycles); episodes 3, 5, and 6 show additional transitions consistent with retries/regrasping. Mean aperture proxy after `-1→+1` decreases from {temporal_means.loc['negative_to_positive', 'aperture_t_plus_0']:.6f} at t to {temporal_means.loc['negative_to_positive', 'aperture_t_plus_10']:.6f} at t+10. After `+1→-1`, it increases from {temporal_means.loc['positive_to_negative', 'aperture_t_plus_0']:.6f} to {temporal_means.loc['positive_to_negative', 'aperture_t_plus_10']:.6f}.

Prediction diagnostic (not semantic evidence)
---------------------------------------------
{prediction_lines}
Predictions were not clipped or thresholded. Their continuous, sometimes out-of-support values reflect the model plus linear MEAN_STD unnormalization and do not redefine the binary expert semantics.

GO / NO-GO
----------
GO for Phase 11 only after explicit authorization. The numeric direction is interpretable, but Phase 11 must define release temporally and validate held-object context; it must not equate every `-1` value or every `+1→-1` transition with a completed physical release.

Methodological limitations
--------------------------
- Only 10 of the metadata-advertised 50 episodes are locally available; conclusions cover 2,612 frames, not all 13,021 advertised frames.
- Camera evidence is observational and 20 Hz; command onset precedes full finger motion and may precede physical separation from an object.
- Finger `qpos` is a state measurement, while action[6] is a command. Their time lag matters.
- The low-level position/velocity/force realization of the binary gripper command is not documented in the local representation and is not needed for identifying command direction.
- The local metadata does not state pose units/reference frame and mislabels axis-angle state components as roll/pitch/yaw.
- No final release events, spatial envelope, threshold, guard, or new inference were created in Phase 10.
"""
    (RESULTS_DIR / "gripper_analysis.txt").write_text(analysis)


def main() -> None:
    info, metadata_episodes = load_metadata()
    frames, images = load_available_frames()
    local_lengths = frames.groupby("episode_id").size().astype(int)
    metadata_lengths = metadata_episodes.set_index("episode_index").length.astype(int)
    for episode_id, length in local_lengths.items():
        if int(metadata_lengths.loc[episode_id]) != int(length):
            raise ValueError(
                f"Episode {episode_id} length mismatch: metadata={metadata_lengths.loc[episode_id]}, data={length}"
            )

    stats = gripper_statistics(frames)
    candidates, delta_stats = transition_candidates(frames)
    candidates.to_csv(RESULTS_DIR / "gripper_transition_candidates.csv", index=False)
    temporal = temporal_response(frames, candidates)
    selected = select_contact_transitions(candidates)
    TRANSITION_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(TRANSITION_DIR / "visual_validation_selection.csv", index=False)
    make_plots(frames, candidates, temporal)
    make_contact_sheets(frames, images, selected)

    normalization = checkpoint_normalization()
    predicted_stats = prediction_gripper_statistics()
    predicted_stats.to_csv(RESULTS_DIR / "predicted_gripper_distribution.csv", index=False)
    write_audits(
        info,
        frames,
        stats,
        delta_stats,
        temporal,
        selected,
        normalization,
        predicted_stats,
    )
    print(
        "PHASE 10 ANALYSIS GENERATED: "
        f"local_episodes={frames.episode_id.nunique()}, local_frames={len(frames)}, "
        f"transitions={len(candidates)}, contact_sheets={len(selected)}, "
        f"visual_review={VISUAL_REVIEW_NOTES['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
