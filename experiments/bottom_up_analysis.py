#!/usr/bin/env python3
"""Build bottom-up translation and gripper evidence from stored predictions only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PHASE_B = ROOT / "results" / "prompt_robustness" / "predictions.csv"
PROMPT_BANK = ROOT / "results" / "prompt_robustness" / "prompt_bank.json"
CANONICAL = ROOT / "results" / "vla_predictions.csv"
OUTPUT = ROOT / "results" / "bottom_up_demo"
FIGURES = OUTPUT / "figures"
ACTION_COLUMNS = ["pred_action_x", "pred_action_y", "pred_action_z"]
CLASSES = ("Correct", "Paraphrase", "Contradictory", "Unrelated")
COLORS = {
    "Correct": "#31C96B",
    "Paraphrase": "#338CF4",
    "Contradictory": "#FF9229",
    "Unrelated": "#F53538",
}
MICROTASK_PROMPTS = {
    "Correct": "CORRECT",
    "Paraphrase": "P07",
    "Contradictory": "C01",
    "Unrelated": "U09",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate() -> tuple[pd.DataFrame, dict]:
    predictions = pd.read_csv(PHASE_B)
    bank = json.loads(PROMPT_BANK.read_text(encoding="utf-8"))
    if len(predictions) != 620:
        raise ValueError(f"Expected 620 Phase-B rows, found {len(predictions)}")
    key = ["episode_id", "frame_index", "prompt_id"]
    if predictions.duplicated(key).any():
        raise ValueError("Duplicate Phase-B observation/prompt rows")
    if not np.isfinite(
        predictions[ACTION_COLUMNS + ["pred_action_gripper"]].to_numpy(float)
    ).all():
        raise ValueError("Non-finite stored action")
    coverage = predictions.groupby(["episode_id", "frame_index"]).prompt_id.nunique()
    if len(coverage) != 20 or not coverage.eq(31).all():
        raise ValueError("Expected 20 observations with 31 prompts each")
    if not predictions.groupby(["episode_id", "frame_index"]).noise_hash.nunique().eq(1).all():
        raise ValueError("Noise hash is not constant within an observation")
    prompt_map = {item["prompt_id"]: item["instruction"] for item in bank["prompts"]}
    if set(prompt_map) != set(predictions.prompt_id.unique()):
        raise ValueError("Prompt-bank ID coverage mismatch")
    if not all(prompt_map[row.prompt_id] == row.instruction for row in predictions.itertuples()):
        raise ValueError("Stored instruction does not match prompt bank")
    return predictions, bank


def observation_statistics(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (episode_id, frame_index), group in predictions.groupby(
        ["episode_id", "frame_index"], sort=True
    ):
        correct = group[group.prompt_id.eq("CORRECT")].iloc[0]
        correct_vector = correct[ACTION_COLUMNS].to_numpy(float)
        record = {
            "episode_id": int(episode_id),
            "frame_index": int(frame_index),
            "normalized_progress": float(correct.normalized_progress),
            "correct_open": bool(correct.pred_action_gripper < 0),
            "all_same_gripper_regime": bool(
                ((group.pred_action_gripper < 0) == (correct.pred_action_gripper < 0)).all()
            ),
        }
        for prompt_class in CLASSES[1:]:
            subset = group[group.prompt_class.eq(prompt_class)]
            vectors = subset[ACTION_COLUMNS].to_numpy(float)
            mean_vector = vectors.mean(axis=0)
            distances = np.linalg.norm(vectors - correct_vector, axis=1)
            denominator = np.linalg.norm(correct_vector) * np.linalg.norm(mean_vector)
            cosine = float(np.dot(correct_vector, mean_vector) / denominator)
            record[f"{prompt_class}_mean_translation_distance"] = float(distances.mean())
            record[f"{prompt_class}_mean_angle_deg"] = float(
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            )
            record[f"{prompt_class}_open_rate"] = float(
                (subset.pred_action_gripper < 0).mean()
            )
        records.append(record)
    return pd.DataFrame(records)


def select_examples(stats: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    upper_quartile = float(stats.Unrelated_mean_translation_distance.quantile(0.75))
    translation_candidates = stats[
        stats.all_same_gripper_regime
        & (
            stats.Paraphrase_mean_translation_distance
            < stats.Contradictory_mean_translation_distance
        )
        & (
            stats.Contradictory_mean_translation_distance
            < stats.Unrelated_mean_translation_distance
        )
        & (stats.Unrelated_mean_translation_distance <= upper_quartile)
    ]
    translation = translation_candidates.sort_values(
        "Unrelated_mean_angle_deg", ascending=False
    ).iloc[0]

    gripper_candidates = stats[
        (~stats.correct_open)
        & stats.Paraphrase_open_rate.eq(0.0)
        & (
            (stats.Contradictory_open_rate > 0.0)
            | (stats.Unrelated_open_rate > 0.0)
        )
    ].copy()
    gripper_candidates["combined_altered_open_rate"] = (
        gripper_candidates.Contradictory_open_rate
        + gripper_candidates.Unrelated_open_rate
    )
    gripper = gripper_candidates.sort_values(
        ["combined_altered_open_rate", "Unrelated_open_rate"], ascending=False
    ).iloc[0]

    if (int(translation.episode_id), int(translation.frame_index)) != (3, 38):
        raise ValueError("Deterministic translation selection changed")
    if (int(gripper.episode_id), int(gripper.frame_index)) != (5, 230):
        raise ValueError("Deterministic gripper selection changed")
    return translation, gripper


def subset_for(predictions: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    return predictions[
        predictions.episode_id.eq(int(selected.episode_id))
        & predictions.frame_index.eq(int(selected.frame_index))
    ].copy()


def decorate_translation(group: pd.DataFrame) -> pd.DataFrame:
    correct = group[group.prompt_id.eq("CORRECT")].iloc[0]
    correct_vector = correct[ACTION_COLUMNS].to_numpy(float)
    rows: list[pd.DataFrame] = []
    for prompt_class in CLASSES:
        subset = group[group.prompt_class.eq(prompt_class)].copy()
        vectors = subset[ACTION_COLUMNS].to_numpy(float)
        class_mean = vectors.mean(axis=0)
        subset["translation_distance_from_correct"] = np.linalg.norm(
            vectors - correct_vector, axis=1
        )
        denominators = np.linalg.norm(vectors, axis=1) * np.linalg.norm(correct_vector)
        cosine = np.sum(vectors * correct_vector, axis=1) / denominators
        subset["cosine_similarity_to_correct"] = cosine
        subset["angular_difference_deg"] = np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )
        subset["class_mean_x"] = class_mean[0]
        subset["class_mean_y"] = class_mean[1]
        subset["class_mean_z"] = class_mean[2]
        subset["class_mean_translation_distance_from_correct"] = float(
            np.linalg.norm(vectors - correct_vector, axis=1).mean()
        )
        subset["gripper_regime"] = np.where(
            subset.pred_action_gripper < 0, "OPENING", "CLOSING"
        )
        rows.append(subset)
    output = pd.concat(rows, ignore_index=True)
    keep = [
        "episode_id",
        "frame_index",
        "global_index",
        "normalized_progress",
        "prompt_id",
        "prompt_class",
        "instruction",
        "noise_hash",
        *ACTION_COLUMNS,
        "pred_action_gripper",
        "gripper_regime",
        "translation_distance_from_correct",
        "cosine_similarity_to_correct",
        "angular_difference_deg",
        "class_mean_x",
        "class_mean_y",
        "class_mean_z",
        "class_mean_translation_distance_from_correct",
    ]
    return output[keep]


def decorate_gripper(group: pd.DataFrame) -> pd.DataFrame:
    output = group.copy()
    output["gripper_regime"] = np.where(
        output.pred_action_gripper < 0, "OPENING", "CLOSING"
    )
    output["class_open_count"] = output.groupby("prompt_class").pred_action_gripper.transform(
        lambda values: int((values < 0).sum())
    )
    output["class_prompt_count"] = output.groupby("prompt_class").prompt_id.transform("count")
    output["class_open_percentage"] = (
        100.0 * output.class_open_count / output.class_prompt_count
    )
    output["class_close_percentage"] = 100.0 - output.class_open_percentage
    output["selected_for_microtask"] = output.apply(
        lambda row: MICROTASK_PROMPTS.get(row.prompt_class) == row.prompt_id, axis=1
    )
    reasons = {
        "Correct": "single canonical baseline",
        "Paraphrase": "closing variant nearest the Paraphrase mean translation vector",
        "Contradictory": "opening variant nearest the Contradictory mean translation vector",
        "Unrelated": "opening variant nearest the Unrelated mean translation vector",
    }
    output["microtask_selection_reason"] = output.prompt_class.map(reasons)
    keep = [
        "episode_id",
        "frame_index",
        "global_index",
        "normalized_progress",
        "prompt_id",
        "prompt_class",
        "instruction",
        "noise_hash",
        *ACTION_COLUMNS,
        "pred_action_gripper",
        "gripper_regime",
        "class_open_count",
        "class_prompt_count",
        "class_open_percentage",
        "class_close_percentage",
        "selected_for_microtask",
        "microtask_selection_reason",
    ]
    return output[keep]


def style_3d_axis(axis) -> None:
    axis.set_facecolor("#172231")
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((0.10, 0.15, 0.22, 1.0))
        pane.set_edgecolor((0.35, 0.43, 0.52, 0.7))
    axis.tick_params(colors="#C9D7E5", labelsize=9)
    axis.xaxis.label.set_color("#EAF1F7")
    axis.yaxis.label.set_color("#EAF1F7")
    axis.zaxis.label.set_color("#EAF1F7")
    axis.grid(True, color="#536273", alpha=0.35)


def plot_translation(table: pd.DataFrame, path: Path) -> None:
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#111A26")
    axis = fig.add_subplot(111, projection="3d")
    axis.set_position([0.13, 0.12, 0.67, 0.70])
    style_3d_axis(axis)
    all_vectors = table[ACTION_COLUMNS].to_numpy(float)
    for prompt_class in CLASSES:
        subset = table[table.prompt_class.eq(prompt_class)]
        color = COLORS[prompt_class]
        for vector in subset[ACTION_COLUMNS].to_numpy(float):
            axis.quiver(
                0,
                0,
                0,
                *vector,
                color=color,
                alpha=0.17 if prompt_class != "Correct" else 0.95,
                linewidth=1.2 if prompt_class != "Correct" else 3.2,
                arrow_length_ratio=0.09,
            )
        mean_vector = subset[ACTION_COLUMNS].to_numpy(float).mean(axis=0)
        axis.quiver(
            0,
            0,
            0,
            *mean_vector,
            color=color,
            alpha=1.0,
            linewidth=4.0,
            arrow_length_ratio=0.11,
        )
        axis.scatter(*mean_vector, color=color, s=55, depthshade=False)
    limits = np.max(np.abs(np.vstack([all_vectors, np.zeros(3)])), axis=0) * 1.12
    limits = np.maximum(limits, 0.15)
    axis.set_xlim(-limits[0], limits[0])
    axis.set_ylim(-limits[1], limits[1])
    axis.set_zlim(-limits[2], limits[2])
    axis.set_box_aspect((limits[0], limits[1], limits[2]))
    axis.view_init(elev=24, azim=42)
    axis.set_xlabel("Action x", labelpad=10)
    axis.set_ylabel("Action y", labelpad=10)
    axis.set_zlabel("Action z", labelpad=10)
    fig.suptitle(
        "Level 1 — Prompt-conditioned translation proposals",
        color="white",
        fontsize=22,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.905,
        "Episode 3, frame 38 · same observation, state, flow noise, and opening regime",
        ha="center",
        color="#BFD0E0",
        fontsize=12,
    )
    handles = [
        Line2D([0], [0], color=COLORS[label], lw=4, label=label) for label in CLASSES
    ]
    legend = fig.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.965, 0.61),
        frameon=False,
        fontsize=11,
    )
    for text in legend.get_texts():
        text.set_color("white")
    fig.text(
        0.5,
        0.025,
        "Faint arrows: individual prompt variants · bold arrows: class means · stored action coordinates, not physical trajectories",
        ha="center",
        color="#BFD0E0",
        fontsize=10,
    )
    fig.savefig(path, dpi=240, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_gripper(table: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12.8, 7.2), facecolor="#111A26")
    axis.set_facecolor("#172231")
    axis.axvspan(-1.12, 0, color="#F53538", alpha=0.10)
    axis.axvspan(0, 1.12, color="#31C96B", alpha=0.09)
    axis.axvline(0, color="white", lw=2.0, alpha=0.9)
    y_positions = {label: 3 - index for index, label in enumerate(CLASSES)}
    for prompt_class in CLASSES:
        subset = table[table.prompt_class.eq(prompt_class)].sort_values("prompt_id")
        count = len(subset)
        jitter = np.linspace(-0.13, 0.13, count) if count > 1 else np.array([0.0])
        axis.scatter(
            subset.pred_action_gripper,
            y_positions[prompt_class] + jitter,
            s=115 if count == 1 else 78,
            color=COLORS[prompt_class],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )
        open_count = int((subset.pred_action_gripper < 0).sum())
        axis.text(
            1.18,
            y_positions[prompt_class],
            f"OPEN {open_count}/{count}",
            ha="left",
            va="center",
            color=COLORS[prompt_class],
            fontsize=12,
            fontweight="bold",
        )
    axis.text(-0.56, 3.56, "OPENING regime  (< 0)", color="#FFB0B2", fontsize=12, ha="center")
    axis.text(0.56, 3.56, "CLOSING regime  (≥ 0)", color="#A9F0C1", fontsize=12, ha="center")
    axis.set_xlim(-1.12, 1.45)
    axis.set_ylim(-0.55, 3.72)
    axis.set_yticks([y_positions[label] for label in CLASSES], CLASSES)
    axis.set_xlabel("Stored continuous gripper prediction", color="#EAF1F7", fontsize=12)
    axis.tick_params(colors="#DCE7F0", labelsize=11)
    for spine in axis.spines.values():
        spine.set_color("#617184")
    axis.grid(axis="x", color="#536273", alpha=0.35)
    axis.set_title(
        "Level 2 — Can language change HOLD/OPEN?",
        color="white",
        fontsize=22,
        fontweight="bold",
        pad=24,
    )
    axis.text(
        0.5,
        1.015,
        "Episode 5, frame 230 · Correct baseline + ten variants per altered semantic class",
        transform=axis.transAxes,
        ha="center",
        color="#BFD0E0",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.025,
        "Operational rule: prediction < 0 → OPENING; prediction ≥ 0 → CLOSING. Continuous values are not clipped.",
        ha="center",
        color="#BFD0E0",
        fontsize=10,
    )
    fig.savefig(path, dpi=240, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_pipeline(path: Path) -> None:
    fig, axis = plt.subplots(figsize=(16, 9), facecolor="#0E1722")
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")
    axis.text(
        8,
        8.25,
        "Bottom-up: from text to understandable robot behavior",
        ha="center",
        color="white",
        fontsize=26,
        fontweight="bold",
    )
    axis.text(
        8,
        7.72,
        "Start with two stored action primitives, combine them once, then return to the original task",
        ha="center",
        color="#B9CAD9",
        fontsize=13,
    )
    cards = [
        (0.65, "1", "TRANSLATION", "Where should\nthe arm move?", "Stored XYZ"),
        (4.45, "2", "GRIPPER", "Keep holding\nor open?", "Stored continuous value"),
        (8.25, "3", "ONE OBJECT", "Short local\nconsequence", "Illustrative MuJoCo"),
        (12.05, "4", "FULL PACKING", "Two objects\ninto basket", "Existing Panda demo"),
    ]
    card_colors = ("#338CF4", "#31C96B", "#FF9229", "#F53538")
    for index, ((x, level, title, question, evidence), color) in enumerate(
        zip(cards, card_colors)
    ):
        card = FancyBboxPatch(
            (x, 2.25),
            3.25,
            4.5,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            facecolor="#172433",
            edgecolor=color,
            linewidth=2.5,
        )
        axis.add_patch(card)
        axis.text(x + 0.28, 6.28, f"LEVEL {level}", color=color, fontsize=12, fontweight="bold")
        axis.text(x + 0.28, 5.68, title, color="white", fontsize=17, fontweight="bold")
        axis.text(x + 0.28, 4.18, question, color="#E7EFF6", fontsize=17, linespacing=1.45)
        axis.text(x + 0.28, 2.75, evidence, color="#AFC2D3", fontsize=11)
        if index < len(cards) - 1:
            arrow = FancyArrowPatch(
                (x + 3.35, 4.5),
                (x + 3.72, 4.5),
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=2.0,
                color="#D7E3EC",
            )
            axis.add_patch(arrow)
    axis.text(
        8,
        1.35,
        "Empirical evidence: offline stored next-action predictions",
        ha="center",
        color="#DDE8F0",
        fontsize=13,
        fontweight="bold",
    )
    axis.text(
        8,
        0.82,
        "Simulation role: short-horizon interpretation—not closed-loop SmolVLA–LIBERO execution",
        ha="center",
        color="#AFC2D3",
        fontsize=11,
    )
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    predictions, bank = load_and_validate()
    stats = observation_statistics(predictions)
    translation_selected, gripper_selected = select_examples(stats)
    translation = decorate_translation(subset_for(predictions, translation_selected))
    gripper = decorate_gripper(subset_for(predictions, gripper_selected))
    translation.to_csv(OUTPUT / "translation_example.csv", index=False)
    gripper.to_csv(OUTPUT / "gripper_example.csv", index=False)
    plot_translation(translation, FIGURES / "translation_proposals.png")
    plot_gripper(gripper, FIGURES / "gripper_prompt_variants.png")
    plot_pipeline(FIGURES / "bottom_up_pipeline.png")

    class_translation = {
        prompt_class: {
            "mean_vector": translation[translation.prompt_class.eq(prompt_class)][
                ACTION_COLUMNS
            ].mean().tolist(),
            "mean_distance_from_correct": float(
                translation[translation.prompt_class.eq(prompt_class)][
                    "translation_distance_from_correct"
                ].mean()
            ),
            "mean_angle_deg_from_correct": float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(
                                translation[
                                    translation.prompt_class.eq("Correct")
                                ][ACTION_COLUMNS].mean().to_numpy(float),
                                translation[
                                    translation.prompt_class.eq(prompt_class)
                                ][ACTION_COLUMNS].mean().to_numpy(float),
                            )
                            /
                            (
                                np.linalg.norm(
                                    translation[
                                        translation.prompt_class.eq("Correct")
                                    ][ACTION_COLUMNS].mean().to_numpy(float)
                                )
                                * np.linalg.norm(
                                    translation[
                                        translation.prompt_class.eq(prompt_class)
                                    ][ACTION_COLUMNS].mean().to_numpy(float)
                                )
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
            ),
        }
        for prompt_class in CLASSES
    }
    gripper_rates = {
        prompt_class: {
            "n": int(len(gripper[gripper.prompt_class.eq(prompt_class)])),
            "open_count": int(
                (gripper[gripper.prompt_class.eq(prompt_class)].pred_action_gripper < 0).sum()
            ),
            "open_percentage": float(
                100
                * (gripper[gripper.prompt_class.eq(prompt_class)].pred_action_gripper < 0).mean()
            ),
        }
        for prompt_class in CLASSES
    }
    branch_rows = gripper[gripper.selected_for_microtask].copy()
    if set(branch_rows.prompt_class) != set(CLASSES) or len(branch_rows) != 4:
        raise ValueError("Microtask branch selection is not exact four-class coverage")
    summary = {
        "status": "PASS",
        "selection_rules": {
            "translation": "same gripper regime across all 31 prompts; class mean distance ordering P<C<U; exclude observations above the Phase-B 75th percentile of Unrelated translation mean; choose largest Correct-to-Unrelated class-mean angle",
            "gripper": "Correct CLOSING; all ten Paraphrases CLOSING; at least one altered OPENING; choose maximum combined Contradictory+Unrelated opening rate",
        },
        "translation_example": {
            "episode_id": int(translation_selected.episode_id),
            "frame_index": int(translation_selected.frame_index),
            "normalized_progress": float(translation_selected.normalized_progress),
            "all_31_same_gripper_regime": bool(
                translation_selected.all_same_gripper_regime
            ),
            "gripper_regime": translation.gripper_regime.iloc[0],
            "class_results": class_translation,
        },
        "gripper_example": {
            "episode_id": int(gripper_selected.episode_id),
            "frame_index": int(gripper_selected.frame_index),
            "normalized_progress": float(gripper_selected.normalized_progress),
            "class_rates": gripper_rates,
            "microtask_prompt_ids": MICROTASK_PROMPTS,
            "microtask_rows": branch_rows[
                [
                    "prompt_id",
                    "prompt_class",
                    "instruction",
                    *ACTION_COLUMNS,
                    "pred_action_gripper",
                    "gripper_regime",
                ]
            ].to_dict(orient="records"),
        },
        "source_hashes": {
            "canonical_predictions_sha256": sha256(CANONICAL),
            "phase_b_predictions_sha256": sha256(PHASE_B),
            "prompt_bank_sha256": sha256(PROMPT_BANK),
        },
        "phase_b_experiment_seed": bank["experiment_seed"],
        "model_inference_performed": False,
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    selected_text = f"""# Selected bottom-up examples

## Level 1 — translation

- Episode: {int(translation_selected.episode_id)}
- Frame: {int(translation_selected.frame_index)}
- Normalized trajectory progress: {translation_selected.normalized_progress:.3f}
- Operational gripper regime: {translation.gripper_regime.iloc[0]} for all 31 prompts

Selection was deterministic: require one regime across all 31 prompts, require
Paraphrase < Contradictory < Unrelated mean translation distance, exclude the
upper quartile of Unrelated observation means, then choose the largest angular
difference between Correct and the Unrelated class-mean vector. This produces a
clear but non-extreme movement-direction example.

Mean translation distance from Correct:

- Paraphrase: {class_translation['Paraphrase']['mean_distance_from_correct']:.3f}
- Contradictory: {class_translation['Contradictory']['mean_distance_from_correct']:.3f}
- Unrelated: {class_translation['Unrelated']['mean_distance_from_correct']:.3f}

## Level 2 and Level 3 — gripper plus one-object micro-task

- Episode: {int(gripper_selected.episode_id)}
- Frame: {int(gripper_selected.frame_index)}
- Normalized trajectory progress: {gripper_selected.normalized_progress:.3f}

Selection was deterministic: require Correct CLOSING and all ten Paraphrases
CLOSING, require at least one altered opening request, then maximize the sum of
Contradictory and Unrelated opening rates across the ten variants.

Opening frequency:

- Correct: {gripper_rates['Correct']['open_count']}/{gripper_rates['Correct']['n']}
- Paraphrase: {gripper_rates['Paraphrase']['open_count']}/{gripper_rates['Paraphrase']['n']}
- Contradictory: {gripper_rates['Contradictory']['open_count']}/{gripper_rates['Contradictory']['n']}
- Unrelated: {gripper_rates['Unrelated']['open_count']}/{gripper_rates['Unrelated']['n']}

The micro-task uses exact stored rows: CORRECT, P07, C01, and U09. P07 is the
closing Paraphrase nearest its class-mean translation vector. C01 and U09 are
the opening variants nearest their respective class-mean translation vectors.
This branch set is a curated, traceable illustration; the frequency plot shows
how often each regime occurs across all ten variants.
"""
    (OUTPUT / "selected_examples.md").write_text(selected_text, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Bottom-up stored-action analysis: PASS")


if __name__ == "__main__":
    main()
