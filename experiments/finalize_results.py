#!/usr/bin/env python
"""Create submission-ready artifacts from the completed SmolVLA experiment.

This script is deliberately model-free. It validates the stored Phase 7-10
outputs, creates four final figures, selects demo examples, extracts their
camera frames, and writes concise presentation documents. It never imports or
loads SmolVLA and never modifies the validated source CSV files.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import (  # noqa: E402
    COMPARISON_ORDER,
    CONDITION_LABELS,
    CONDITION_ORDER,
    OBSERVATION_KEY,
    STAGE_LABELS,
    build_summary,
    compute_expert_agreement,
    compute_language_effects,
    validate_predictions,
)

RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures" / "final"
DEMO_ASSET_DIR = RESULTS_DIR / "demo_assets"
DATA_DIR = (
    REPO_ROOT
    / ".cache"
    / "huggingface"
    / "lerobot_v21"
    / "HuggingFaceVLA"
    / "smol-libero"
    / "data"
    / "chunk-000"
)

SOURCE_FILES = (
    "vla_predictions.csv",
    "language_effects.csv",
    "expert_agreement.csv",
    "summary.csv",
    "bootstrap_summary.csv",
    "stage_sensitivity.csv",
    "stage_sensitivity_by_episode.csv",
    "phase8_9_findings.md",
    "gripper_analysis.txt",
    "gripper_semantics_audit.md",
    "state_semantics_audit.md",
)

COLORS = {
    "correct": "#3B6EA8",
    "paraphrase": "#3A8D5D",
    "contradictory": "#D17A22",
    "unrelated": "#C84B4B",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sources() -> dict[str, pd.DataFrame]:
    missing = [name for name in SOURCE_FILES if not (RESULTS_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required finalization sources: {missing}")
    return {
        name: pd.read_csv(RESULTS_DIR / name)
        for name in SOURCE_FILES
        if name.endswith(".csv")
    }


def assert_frame_consistent(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    name: str,
    *,
    atol: float = 1e-11,
) -> None:
    if list(actual.columns) != list(expected.columns):
        raise ValueError(f"{name} columns differ from deterministic recomputation")
    if actual.shape != expected.shape:
        raise ValueError(
            f"{name} shape differs: stored={actual.shape}, recomputed={expected.shape}"
        )
    for column in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[column]) and pd.api.types.is_numeric_dtype(
            expected[column]
        ):
            left = actual[column].to_numpy(dtype=float)
            right = expected[column].to_numpy(dtype=float)
            if not np.allclose(left, right, rtol=1e-10, atol=atol, equal_nan=True):
                maximum = float(np.nanmax(np.abs(left - right)))
                raise ValueError(
                    f"{name}.{column} differs from deterministic recomputation; "
                    f"max_abs_difference={maximum}"
                )
        else:
            left = actual[column].fillna("<NA>").astype(str).to_numpy()
            right = expected[column].fillna("<NA>").astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"{name}.{column} differs from deterministic recomputation")


def stage_consistency(
    effects: pd.DataFrame, stage_summary: pd.DataFrame, stage_by_episode: pd.DataFrame
) -> bool:
    progress = effects.task_progress.to_numpy(float)
    staged = effects.copy()
    staged["task_stage_index"] = np.minimum(np.floor(progress * 5).astype(int), 4)
    staged["task_stage"] = [STAGE_LABELS[index] for index in staged.task_stage_index]

    aggregate = (
        staged.groupby(
            ["task_stage", "task_stage_index", "comparison_condition"],
            sort=False,
            observed=True,
        )
        .full_action_language_distance.agg(["count", "mean"])
        .reset_index()
    )
    stored = stage_summary[
        stage_summary.metric == "full_action_language_distance"
    ][
        [
            "task_stage",
            "task_stage_index",
            "comparison_condition",
            "count",
            "mean",
        ]
    ]
    aggregate["comparison_condition"] = aggregate["comparison_condition"].map(
        CONDITION_LABELS
    )
    aggregate = aggregate.sort_values(
        ["task_stage_index", "comparison_condition"]
    ).reset_index(drop=True)
    stored = stored.sort_values(
        ["task_stage_index", "comparison_condition"]
    ).reset_index(drop=True)
    identity_columns = [
        "task_stage",
        "task_stage_index",
        "comparison_condition",
        "count",
    ]
    for column in identity_columns:
        if not np.array_equal(
            aggregate[column].astype(str).to_numpy(),
            stored[column].astype(str).to_numpy(),
        ):
            return False
    if not np.allclose(aggregate["mean"], stored["mean"], rtol=1e-10, atol=1e-11):
        return False

    expected_rows = 10 * 5 * 3 * 4
    return len(stage_by_episode) == expected_rows


def final_validation(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    predictions = tables["vla_predictions.csv"]
    effects = tables["language_effects.csv"]
    expert = tables["expert_agreement.csv"]
    summary = tables["summary.csv"]
    bootstrap = tables["bootstrap_summary.csv"]
    stage_summary = tables["stage_sensitivity.csv"]
    stage_by_episode = tables["stage_sensitivity_by_episode.csv"]

    base_validation = validate_predictions(predictions)
    recomputed_effects = compute_language_effects(predictions)
    recomputed_expert = compute_expert_agreement(predictions)
    recomputed_summary = build_summary(recomputed_expert, recomputed_effects)
    assert_frame_consistent(effects, recomputed_effects, "language_effects.csv")
    assert_frame_consistent(expert, recomputed_expert, "expert_agreement.csv")
    assert_frame_consistent(summary, recomputed_summary, "summary.csv")

    numeric_counts: dict[str, dict[str, int]] = {}
    for name, table in tables.items():
        numeric = table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        numeric_counts[name] = {
            "nan_count_all_columns": int(table.isna().sum().sum()),
            "nonfinite_numeric_count": int((~np.isfinite(numeric)).sum()),
        }

    duplicate_counts = {
        "vla_predictions_observation_condition": int(
            predictions.duplicated(OBSERVATION_KEY + ["language_condition"]).sum()
        ),
        "language_effects_observation_condition": int(
            effects.duplicated(OBSERVATION_KEY + ["comparison_condition"]).sum()
        ),
        "expert_agreement_observation_condition": int(
            expert.duplicated(OBSERVATION_KEY + ["language_condition"]).sum()
        ),
    }

    effect_means = (
        effects.groupby("comparison_condition").full_action_language_distance.mean()
    )
    bootstrap_full = bootstrap[
        bootstrap.metric == "full_action_language_distance"
    ].set_index("comparison_condition")
    agreement_means = expert.groupby("language_condition")[[
        "translation_error_l2",
        "rotation_error_l2",
        "gripper_absolute_error",
    ]].mean()

    audit_text = {
        name: (RESULTS_DIR / name).read_text()
        for name in (
            "gripper_analysis.txt",
            "gripper_semantics_audit.md",
            "state_semantics_audit.md",
        )
    }
    consistency_checks = {
        "predictions_validate_with_phase8_pairing_gate": True,
        "language_effects_match_recomputation": True,
        "expert_agreement_matches_recomputation": True,
        "summary_matches_recomputation": True,
        "stage_summaries_match_effects": stage_consistency(
            effects, stage_summary, stage_by_episode
        ),
        "bootstrap_point_estimates_match_effect_means": all(
            np.isclose(
                effect_means[condition.lower()],
                bootstrap_full.loc[condition, "point_estimate"],
                rtol=1e-10,
                atol=1e-11,
            )
            for condition in ("Paraphrase", "Contradictory", "Unrelated")
        ),
        "mean_language_effect_order_is_paraphrase_contradictory_unrelated": bool(
            effect_means.paraphrase
            < effect_means.contradictory
            < effect_means.unrelated
        ),
        "correct_has_lowest_mean_expert_error_for_all_three_components": all(
            agreement_means.loc["correct", metric]
            == agreement_means[metric].min()
            for metric in agreement_means.columns
        ),
        "gripper_audit_says_positive_is_closing": "`+1` closes"
        in audit_text["gripper_semantics_audit.md"],
        "gripper_audit_says_negative_is_opening": "`-1` opens"
        in audit_text["gripper_semantics_audit.md"],
        "state_audit_identifies_axis_angle": "axis-angle"
        in audit_text["state_semantics_audit.md"],
    }

    expected_undefined_cosine = int(expert.translation_cosine_similarity.isna().sum())
    unexpected_nonfinite = 0
    for name, table in tables.items():
        numeric = table.select_dtypes(include=[np.number])
        if name == "expert_agreement.csv":
            numeric = numeric.drop(columns=["translation_cosine_similarity"])
        unexpected_nonfinite += int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())

    passed = (
        all(value == 0 for value in duplicate_counts.values())
        and expected_undefined_cosine == 16
        and unexpected_nonfinite == 0
        and all(consistency_checks.values())
    )

    report = {
        "validation_result": "PASS" if passed else "FAIL",
        "prediction_rows": int(len(predictions)),
        "observations": int(predictions[OBSERVATION_KEY].drop_duplicates().shape[0]),
        "episodes": sorted(int(value) for value in predictions.episode_id.unique()),
        "episode_count": int(predictions.episode_id.nunique()),
        "language_conditions": list(CONDITION_ORDER),
        "language_condition_count": int(predictions.language_condition.nunique()),
        "duplicate_count": int(sum(duplicate_counts.values())),
        "duplicate_counts_by_key": duplicate_counts,
        "nan_nonfinite_counts_by_file": numeric_counts,
        "expected_undefined_translation_cosine_count": expected_undefined_cosine,
        "unexpected_nan_or_nonfinite_count": unexpected_nonfinite,
        "source_file_sha256": {
            name: sha256(RESULTS_DIR / name) for name in SOURCE_FILES
        },
        "key_result_consistency_checks": consistency_checks,
        "validated_key_results": {
            "mean_full_7d_distance_from_correct": {
                CONDITION_LABELS[condition]: float(effect_means[condition])
                for condition in COMPARISON_ORDER
            },
            "episode_cluster_bootstrap_95ci_full_7d": {
                label: {
                    "low": float(bootstrap_full.loc[label, "ci95_low"]),
                    "high": float(bootstrap_full.loc[label, "ci95_high"]),
                }
                for label in ("Paraphrase", "Contradictory", "Unrelated")
            },
            "mean_expert_agreement_errors": {
                CONDITION_LABELS[condition]: {
                    metric: float(agreement_means.loc[condition, metric])
                    for metric in agreement_means.columns
                }
                for condition in CONDITION_ORDER
            },
        },
        "phase8_validation": base_validation,
        "notes": [
            "The 16 stored NaNs are expected undefined translation cosine similarities under the documented near-zero norm rule.",
            "No validated source CSV was modified by finalization.",
            "Validation does not treat recorded expert actions as optimal ground truth.",
        ],
    }
    (RESULTS_DIR / "final_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if not passed:
        raise RuntimeError("Final validation failed; see results/final_validation.json")
    return report


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_language_difference(effects: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    values = [
        effects.loc[
            effects.comparison_condition == condition,
            "full_action_language_distance",
        ].to_numpy(float)
        for condition in COMPARISON_ORDER
    ]
    episode_means = effects.groupby(
        ["episode_id", "comparison_condition"], as_index=False
    ).full_action_language_distance.mean()
    full_ci = bootstrap[
        bootstrap.metric == "full_action_language_distance"
    ].set_index("comparison_condition")

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    positions = np.arange(1, 4)
    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=180,
    )
    for body, condition in zip(violins["bodies"], COMPARISON_ORDER, strict=True):
        body.set_facecolor(COLORS[condition])
        body.set_edgecolor(COLORS[condition])
        body.set_alpha(0.18)

    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.19,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.8},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch, condition in zip(box["boxes"], COMPARISON_ORDER, strict=True):
        patch.set_facecolor("white")
        patch.set_edgecolor(COLORS[condition])
        patch.set_linewidth(1.4)

    for position, condition in zip(positions, COMPARISON_ORDER, strict=True):
        points = episode_means.loc[
            episode_means.comparison_condition == condition,
            "full_action_language_distance",
        ].to_numpy(float)
        jitter = np.linspace(-0.25, 0.25, len(points))
        ax.scatter(
            position + jitter,
            points,
            s=28,
            marker="o",
            facecolor="white",
            edgecolor=COLORS[condition],
            linewidth=1.1,
            zorder=4,
        )
        label = CONDITION_LABELS[condition]
        mean = float(full_ci.loc[label, "point_estimate"])
        low = float(full_ci.loc[label, "ci95_low"])
        high = float(full_ci.loc[label, "ci95_high"])
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            color=COLORS[condition],
            markeredgecolor="white",
            markersize=8,
            capsize=5,
            linewidth=2.2,
            zorder=5,
        )
        ax.text(
            position,
            high + 0.055,
            f"mean {mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS[condition],
            fontweight="bold",
        )

    ax.set_xticks(positions, [CONDITION_LABELS[c] for c in COMPARISON_ORDER])
    ax.set_ylabel("Full 7-D action distance from Correct")
    ax.set_title(
        "Changing Only the Instruction Changes SmolVLA's Predicted Action",
        pad=15,
    )
    ax.text(
        0.5,
        1.01,
        "200 paired observations; vision, robot state, model, preprocessing, and flow noise fixed",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#555555",
            label="Episode mean",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="#555555",
            markerfacecolor="#555555",
            label="Overall mean ± episode-cluster 95% CI",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "01_language_induced_action_difference")


def figure_expert_agreement(expert: pd.DataFrame) -> None:
    metrics = (
        ("translation_error_l2", "A. Translation", "Translation L2 error"),
        ("rotation_error_l2", "B. Axis-angle orientation", "Axis-angle-component L2 error"),
        ("gripper_absolute_error", "C. Gripper", "Gripper absolute error"),
    )
    episode_means = expert.groupby(
        ["episode_id", "language_condition"], as_index=False
    )[[metric for metric, _, _ in metrics]].mean()

    fig, axes = plt.subplots(1, 3, figsize=(14.3, 4.9))
    x = np.arange(4)
    for ax, (metric, panel_title, ylabel) in zip(axes, metrics, strict=True):
        for position, condition in enumerate(CONDITION_ORDER):
            points = episode_means.loc[
                episode_means.language_condition == condition, metric
            ].to_numpy(float)
            jitter = np.linspace(-0.13, 0.13, len(points))
            ax.scatter(
                position + jitter,
                points,
                s=28,
                facecolor="white",
                edgecolor=COLORS[condition],
                linewidth=1.0,
                alpha=0.95,
                zorder=3,
            )
            mean = float(expert.loc[expert.language_condition == condition, metric].mean())
            ax.scatter(
                position,
                mean,
                marker="D",
                s=62,
                color=COLORS[condition],
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )
            ax.text(
                position,
                mean,
                f" {mean:.3f}",
                ha="left",
                va="bottom",
                fontsize=8.5,
                color=COLORS[condition],
            )
        ax.set_xticks(
            x,
            [CONDITION_LABELS[c] for c in CONDITION_ORDER],
            rotation=24,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.margins(x=0.12, y=0.16)

    fig.suptitle("Agreement with Recorded Expert Action", fontsize=15, y=0.99)
    fig.text(
        0.5,
        0.91,
        "Open circles are episode means; diamonds are overall means. Lower error means closer agreement, not objective correctness.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save_figure(fig, "02_agreement_with_recorded_expert_action")


def figure_stage_sensitivity(stage_summary: pd.DataFrame) -> None:
    metric = "full_action_language_distance"
    fig, ax = plt.subplots(figsize=(9.1, 5.7))
    x = np.arange(len(STAGE_LABELS))
    for condition in COMPARISON_ORDER:
        label = CONDITION_LABELS[condition]
        subset = stage_summary[
            (stage_summary.comparison_condition == label)
            & (stage_summary.metric == metric)
        ].sort_values("task_stage_index")
        mean = subset["mean"].to_numpy(float)
        low = subset.mean_ci95_low.to_numpy(float)
        high = subset.mean_ci95_high.to_numpy(float)
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack([mean - low, high - mean]),
            marker="o",
            markersize=6.5,
            linewidth=2.2,
            capsize=4,
            color=COLORS[condition],
            label=label,
        )
    ax.set_xticks(x, STAGE_LABELS)
    ax.set_xlabel("Normalized task progress")
    ax.set_ylabel("Mean full 7-D action distance from Correct")
    fig.suptitle(
        "Language Sensitivity Varies Across the Manipulation Trajectory",
        fontsize=15,
        y=0.985,
    )
    fig.text(
        0.5,
        0.93,
        "Error bars: 95% percentile intervals from an episode-cluster bootstrap",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, "03_language_sensitivity_across_task_progress")


def decode_image(value: dict[str, Any]) -> Image.Image:
    image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    return image.transpose(Image.Transpose.ROTATE_180)


def read_episode_frame(episode_id: int, frame_index: int) -> tuple[Image.Image, Image.Image]:
    path = DATA_DIR / f"episode_{episode_id:06d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Camera source missing for episode {episode_id}: {path}. "
            "The model is not needed, but the local source parquet is required once to extract demo assets."
        )
    table = pq.read_table(
        path,
        columns=[
            "frame_index",
            "observation.images.image",
            "observation.images.image2",
        ],
    )
    frame_values = np.asarray(table["frame_index"], dtype=np.int64)
    matches = np.flatnonzero(frame_values == frame_index)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for episode {episode_id}, frame {frame_index}; found {len(matches)}"
        )
    row = int(matches[0])
    return (
        decode_image(table["observation.images.image"][row].as_py()),
        decode_image(table["observation.images.image2"][row].as_py()),
    )


def read_episode_actions(episode_id: int) -> pd.DataFrame:
    path = DATA_DIR / f"episode_{episode_id:06d}.parquet"
    table = pq.read_table(path, columns=["frame_index", "action"])
    actions = np.asarray(table["action"].to_pylist(), dtype=float)
    return pd.DataFrame(
        {
            "frame_index": np.asarray(table["frame_index"], dtype=np.int64),
            "gripper_command": actions[:, 6],
        }
    )


def figure_gripper_semantics() -> None:
    episode_id = 0
    transition_frame = 240
    before_frame = transition_frame - 2
    after_frame = transition_frame + 10
    trajectory = read_episode_actions(episode_id)
    before_images = read_episode_frame(episode_id, before_frame)
    after_images = read_episode_frame(episode_id, after_frame)

    fig = plt.figure(figsize=(15.4, 6.5))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.35, 1.35, 1, 1),
        height_ratios=(1, 1),
        wspace=0.09,
        hspace=0.20,
    )
    ax = fig.add_subplot(grid[:, :2])
    frames = trajectory.frame_index.to_numpy(int)
    command = trajectory.gripper_command.to_numpy(float)
    ax.fill_between(
        frames,
        -1.18,
        1.18,
        where=command > 0,
        step="post",
        color=COLORS["paraphrase"],
        alpha=0.08,
        label="+1 closing regime",
    )
    ax.fill_between(
        frames,
        -1.18,
        1.18,
        where=command < 0,
        step="post",
        color=COLORS["correct"],
        alpha=0.07,
        label="−1 opening regime",
    )
    ax.step(frames, command, where="post", color="#303030", linewidth=1.7)
    ax.axvline(
        transition_frame,
        color=COLORS["unrelated"],
        linewidth=2.0,
        linestyle="--",
    )
    ax.scatter(
        [transition_frame],
        [-1],
        color=COLORS["unrelated"],
        s=55,
        zorder=4,
    )
    ax.annotate(
        "+1 → −1\nopening-command onset",
        xy=(transition_frame, -0.92),
        xytext=(180, -0.15),
        arrowprops={"arrowstyle": "->", "color": COLORS["unrelated"]},
        color=COLORS["unrelated"],
        fontsize=10,
        ha="center",
    )
    ax.axvspan(before_frame, after_frame, color=COLORS["unrelated"], alpha=0.05)
    ax.set_ylim(-1.25, 1.25)
    ax.set_yticks([-1, 1], ["−1  Open", "+1  Close"])
    ax.set_xlabel("Expert trajectory frame")
    ax.set_ylabel("Recorded gripper command")
    ax.set_title("Episode 0 expert gripper trajectory", loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False, loc="lower left")

    labels = (
        ("Camera 1 · t−2", before_images[0]),
        ("Camera 1 · t+10", after_images[0]),
        ("Camera 2 · t−2", before_images[1]),
        ("Camera 2 · t+10", after_images[1]),
    )
    for cell, (label, image) in zip(
        (grid[0, 2], grid[0, 3], grid[1, 2], grid[1, 3]), labels, strict=True
    ):
        image_ax = fig.add_subplot(cell)
        image_ax.imshow(image)
        image_ax.set_title(label, fontsize=9)
        image_ax.axis("off")

    fig.suptitle(
        "Temporal and Camera Validation of Gripper-Command Semantics",
        fontsize=15,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.015,
        "+1 → −1 marks opening-command onset; it is not by itself proof of successful physical release.",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )
    save_figure(fig, "04_gripper_semantics_temporal_validation")


def select_demo_examples(effects: pd.DataFrame) -> list[dict[str, Any]]:
    full = effects.pivot_table(
        index=OBSERVATION_KEY + ["task_progress"],
        columns="comparison_condition",
        values="full_action_language_distance",
        aggfunc="first",
    )
    gripper = effects.pivot_table(
        index=OBSERVATION_KEY + ["task_progress"],
        columns="comparison_condition",
        values="gripper_language_difference",
        aggfunc="first",
    )
    selected: list[tuple[str, tuple[Any, ...], str]] = []
    used: set[tuple[int, int, int]] = set()

    def add(selection_type: str, ordered: pd.DataFrame, reason_builder: Any) -> None:
        for key, row in ordered.iterrows():
            observation_key = (int(key[0]), int(key[1]), int(key[2]))
            if observation_key in used:
                continue
            selected.append((selection_type, key, reason_builder(key, row)))
            used.add(observation_key)
            return
        raise RuntimeError(f"No unused observation available for {selection_type}")

    add(
        "paraphrase_close_to_correct",
        full.sort_values("paraphrase", ascending=True),
        lambda key, row: (
            f"Smallest stored Paraphrase distance from Correct ({row.paraphrase:.4f}) "
            "among all 200 observations; included as a low-perturbation example."
        ),
    )
    add(
        "large_contradictory_change",
        full.sort_values("contradictory", ascending=False),
        lambda key, row: (
            f"Largest stored Contradictory distance from Correct ({row.contradictory:.4f}); "
            "shows a substantial controlled language effect."
        ),
    )
    add(
        "large_unrelated_change",
        full.sort_values("unrelated", ascending=False),
        lambda key, row: (
            f"Largest unused Unrelated distance from Correct ({row.unrelated:.4f}); "
            "shows a very large action change without reusing the contradictory example."
        ),
    )
    late = full.reset_index()
    late = late[late.task_progress >= 0.8].copy()
    late["mean_condition_distance"] = late[list(COMPARISON_ORDER)].mean(axis=1)
    late = late.set_index(OBSERVATION_KEY + ["task_progress"])
    add(
        "late_stage_strong_sensitivity",
        late.sort_values("mean_condition_distance", ascending=False),
        lambda key, row: (
            f"Late-stage observation (progress {float(key[3]):.3f}) with a large mean distance "
            f"across altered conditions ({row.mean_condition_distance:.4f}); selected after "
            "excluding observations already used above."
        ),
    )
    counterexample = full.copy()
    counterexample["paraphrase_minus_contradictory"] = (
        counterexample.paraphrase - counterexample.contradictory
    )
    counterexample = counterexample[
        counterexample.paraphrase_minus_contradictory > 0
    ]
    add(
        "ordering_counterexample_and_gripper_change",
        counterexample.sort_values("paraphrase_minus_contradictory", ascending=False),
        lambda key, row: (
            f"Counterexample to observation-level ordering: Paraphrase exceeds Contradictory "
            f"by {row.paraphrase_minus_contradictory:.4f}. It also contains a large "
            "Paraphrase gripper change, preventing an only-hypothesis-supporting selection."
        ),
    )

    output: list[dict[str, Any]] = []
    for index, (selection_type, key, reason) in enumerate(selected, start=1):
        episode_id, frame_index, global_index, task_progress = key
        distances = full.loc[key]
        gripper_values = gripper.loc[key]
        output.append(
            {
                "example_index": index,
                "selection_type": selection_type,
                "episode_id": int(episode_id),
                "frame_index": int(frame_index),
                "global_index": int(global_index),
                "task_progress": float(task_progress),
                "reason": reason,
                "full_action_distance_from_correct": {
                    CONDITION_LABELS[condition]: float(distances[condition])
                    for condition in COMPARISON_ORDER
                },
                "gripper_difference_from_correct": {
                    CONDITION_LABELS[condition]: float(gripper_values[condition])
                    for condition in COMPARISON_ORDER
                },
                "camera_1_asset": (
                    f"results/demo_assets/episode_{int(episode_id):02d}_"
                    f"frame_{int(frame_index):03d}_camera_1.png"
                ),
                "camera_2_asset": (
                    f"results/demo_assets/episode_{int(episode_id):02d}_"
                    f"frame_{int(frame_index):03d}_camera_2.png"
                ),
            }
        )
    return output


def write_demo_examples(effects: pd.DataFrame) -> list[dict[str, Any]]:
    examples = select_demo_examples(effects)
    DEMO_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for example in examples:
        camera_1, camera_2 = read_episode_frame(
            example["episode_id"], example["frame_index"]
        )
        for image, key in ((camera_1, "camera_1_asset"), (camera_2, "camera_2_asset")):
            destination = REPO_ROOT / example[key]
            image.save(destination, format="PNG", optimize=True)
    payload = {
        "source": "results/language_effects.csv",
        "selection_count": len(examples),
        "selection_policy": (
            "Deterministic rank-based selection covering low Paraphrase change, large "
            "Contradictory and Unrelated changes, late-stage sensitivity, and one explicit "
            "observation-level counterexample. Duplicate observations are excluded."
        ),
        "anti_cherry_picking_note": (
            "The selection deliberately includes a counterexample where Paraphrase changes "
            "the action more than Contradictory; aggregate ordering is not claimed for every frame."
        ),
        "examples": examples,
    }
    (RESULTS_DIR / "demo_examples.json").write_text(json.dumps(payload, indent=2) + "\n")
    return examples


def result_tables(
    summary: pd.DataFrame, bootstrap: pd.DataFrame
) -> tuple[dict[str, float], dict[str, tuple[float, float]], dict[str, dict[str, float]]]:
    effect_rows = summary[
        (summary.analysis_family == "paired_language_effect_from_correct")
        & (summary.metric == "full_action_language_distance")
    ].set_index("condition")
    means = {
        label: float(effect_rows.loc[label, "mean"])
        for label in ("Paraphrase", "Contradictory", "Unrelated")
    }
    bootstrap_rows = bootstrap[
        bootstrap.metric == "full_action_language_distance"
    ].set_index("comparison_condition")
    intervals = {
        label: (
            float(bootstrap_rows.loc[label, "ci95_low"]),
            float(bootstrap_rows.loc[label, "ci95_high"]),
        )
        for label in means
    }
    agreement_rows = summary[
        summary.analysis_family == "agreement_with_recorded_expert_action"
    ]
    agreement: dict[str, dict[str, float]] = {}
    for label in ("Correct", "Paraphrase", "Contradictory", "Unrelated"):
        subset = agreement_rows[agreement_rows.condition == label].set_index("metric")
        agreement[label] = {
            metric: float(subset.loc[metric, "mean"])
            for metric in (
                "translation_error_l2",
                "rotation_error_l2",
                "gripper_absolute_error",
            )
        }
    return means, intervals, agreement


def write_final_findings(
    summary: pd.DataFrame, bootstrap: pd.DataFrame, stage_summary: pd.DataFrame
) -> None:
    means, intervals, agreement = result_tables(summary, bootstrap)
    stage_full = stage_summary[
        stage_summary.metric == "full_action_language_distance"
    ]
    stage_lines = []
    for label in ("Paraphrase", "Contradictory", "Unrelated"):
        subset = stage_full[stage_full.comparison_condition == label].sort_values(
            "task_stage_index"
        )
        stage_lines.append(
            f"- {label}: "
            + ", ".join(
                f"{stage}={mean:.4f}"
                for stage, mean in zip(subset.task_stage, subset["mean"], strict=True)
            )
            + "."
        )

    agreement_lines = []
    for label, values in agreement.items():
        agreement_lines.append(
            f"- {label}: {values['translation_error_l2']:.4f} / "
            f"{values['rotation_error_l2']:.4f} / "
            f"{values['gripper_absolute_error']:.4f}."
        )

    text = f"""# Research Question

How much does changing only the natural-language instruction alter SmolVLA's predicted action when vision, robot state, preprocessing, model, and flow-matching noise are fixed?

# Method

We evaluated the pretrained `HuggingFaceVLA/smolvla_libero` checkpoint on locally available `HuggingFaceVLA/smol-libero` demonstrations. The controlled evaluation used 10 episodes, 200 observations, four language conditions, and 800 predictions. Within every paired observation, the two camera images, robot state, preprocessing, checkpoint, and explicit flow-noise tensor were identical; only the task string changed. Inference ran on Apple M2 Max MPS.

# Main Results

Mean full 7-D action distance from Correct language, with episode-cluster bootstrap 95% confidence intervals:

- Paraphrase: {means['Paraphrase']:.6f} [{intervals['Paraphrase'][0]:.6f}, {intervals['Paraphrase'][1]:.6f}].
- Contradictory: {means['Contradictory']:.6f} [{intervals['Contradictory'][0]:.6f}, {intervals['Contradictory'][1]:.6f}].
- Unrelated: {means['Unrelated']:.6f} [{intervals['Unrelated'][0]:.6f}, {intervals['Unrelated'][1]:.6f}].

Mean agreement errors against the recorded expert action are reported as translation / axis-angle orientation / gripper absolute error:

{chr(10).join(agreement_lines)}

Correct-language predictions had the smallest mean error on all three reported components, while Unrelated had the largest. These are agreement measurements against a recorded demonstration, not accuracy or objective correctness.

# Task-Stage Result

Language sensitivity varied descriptively with normalized task progress, and the pattern was condition-specific and non-monotonic:

{chr(10).join(stage_lines)}

Stage intervals use episodes, rather than individual frames, as bootstrap clusters.

# Gripper Semantics

- `+1` is a closing command.
- `-1` is an opening command.
- `+1 -> -1` marks opening-command onset.

Command onset is not equivalent to guaranteed physical release. The audit used temporal finger-state response and two-camera trajectory inspection; retry/regrasp sequences demonstrate why command sign alone is insufficient.

# Interpretation

Under controlled observations and identical flow noise, semantically equivalent paraphrases generally produced smaller action changes than contradictory or unrelated instructions. Unrelated instructions produced the largest average changes in the model's predicted action. Language sensitivity was not constant across trajectory progress. Correct-language predictions showed greater average agreement with the recorded expert demonstration than altered-language conditions.

These statements describe this checkpoint, task, and offline sample. They do not establish human-like language understanding, closed-loop success, or safety.

# Limitations

- Only 10 locally available episodes were evaluated.
- Observations within episodes are temporally related; episode-cluster bootstrap intervals partially address, but do not eliminate, this dependence.
- The expert action is a recorded reference demonstration, not a ground-truth optimal action.
- Action dimensions have different physical meanings and scales; the full 7-D norm is not physically calibrated across components.
- Offline frame-level evaluation does not measure closed-loop task success.
- Only one pretrained VLA checkpoint was studied.
- One task/domain was studied.
- Language conditions were manually designed.
- Physical release cannot be inferred from command sign alone.
"""
    (RESULTS_DIR / "final_findings.md").write_text(text)


def write_cheatsheet(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    means, intervals, _ = result_tables(summary, bootstrap)
    text = f"""# Presentation Cheat Sheet

## The problem

Vision-Language-Action models connect language, images, robot state, and actions. Their behavior under instruction variation matters for reliability.

## The experiment

- Pretrained `HuggingFaceVLA/smolvla_libero`; no training or fine-tuning.
- 200 observations from 10 locally available `smol-libero` episodes.
- Four conditions and 800 stored predictions: Correct, Paraphrase, Contradictory, Unrelated.
- Within each comparison, camera images, robot state, preprocessing, model, and explicit flow noise were fixed. Only the instruction changed.
- Inference ran on Apple M2 Max MPS.

## The big result

Mean full 7-D distance from Correct:

- Paraphrase: **{means['Paraphrase']:.4f}** (episode-cluster 95% CI [{intervals['Paraphrase'][0]:.4f}, {intervals['Paraphrase'][1]:.4f}])
- Contradictory: **{means['Contradictory']:.4f}** ([{intervals['Contradictory'][0]:.4f}, {intervals['Contradictory'][1]:.4f}])
- Unrelated: **{means['Unrelated']:.4f}** ([{intervals['Unrelated'][0]:.4f}, {intervals['Unrelated'][1]:.4f}])

Aggregate result: **Paraphrase < Contradictory < Unrelated**. This ordering is not universal at every observation.

## Why this matters

The pretrained VLA is measurably conditioned on language in this controlled evaluation, and semantically more disruptive instructions generally perturb its predicted actions more strongly. Sensitivity also varies non-monotonically across task progress.

## Robotics part

The expert gripper channel is a persistent binary command. Temporal finger-state response and two-camera inspection establish `+1 = closing`, `-1 = opening`, and `+1 -> -1 = opening-command onset`. Command onset alone does not prove successful physical release.

## Most important limitation

This is offline action analysis, not closed-loop task-success evaluation.

## Likely questions

**Q: Did you train SmolVLA?**  
A: No. We evaluated a pretrained checkpoint.

**Q: How do you know the action differences came from language?**  
A: Images, robot state, preprocessing, model, and explicit flow noise were held fixed within every paired observation; only the task string changed.

**Q: Is the expert action ground truth?**  
A: No. It is the recorded expert demonstration used as a reference for agreement.

**Q: Did you prove safety?**  
A: No. We studied reliability-related behavior and gripper semantics, but did not implement or claim a universal safety mechanism.

**Q: Why are there only 10 episodes?**  
A: The local dataset cache contained 10 usable demonstrations; this limitation is documented.

**Q: Why does unrelated language matter?**  
A: It tests whether semantically irrelevant task conditioning changes the predicted robot action under otherwise identical inputs.

**Q: What is the main takeaway?**  
A: Language changes measurably alter the VLA's action, with semantically equivalent paraphrases generally causing smaller changes than contradictory or unrelated instructions.
"""
    (RESULTS_DIR / "presentation_cheatsheet.md").write_text(text)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    tables = read_sources()
    validation = final_validation(tables)
    examples = write_demo_examples(tables["language_effects.csv"])
    figure_language_difference(
        tables["language_effects.csv"], tables["bootstrap_summary.csv"]
    )
    figure_expert_agreement(tables["expert_agreement.csv"])
    figure_stage_sensitivity(tables["stage_sensitivity.csv"])
    figure_gripper_semantics()
    write_final_findings(
        tables["summary.csv"],
        tables["bootstrap_summary.csv"],
        tables["stage_sensitivity.csv"],
    )
    write_cheatsheet(tables["summary.csv"], tables["bootstrap_summary.csv"])
    print(
        "FINALIZATION PASS: "
        f"validation={validation['validation_result']}, figures=4, "
        f"demo_examples={len(examples)}, model_inference=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
