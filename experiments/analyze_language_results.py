#!/usr/bin/env python
"""Phases 8-9: deterministic paired metrics, episode-aware uncertainty, and figures."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import (  # noqa: E402
    COMPARISON_ORDER,
    CONDITION_LABELS,
    CONDITION_ORDER,
    LANGUAGE_EFFECT_METRICS,
    OBSERVATION_KEY,
    STAGE_LABELS,
    add_task_stage,
    build_summary,
    cluster_bootstrap_mean,
    compute_expert_agreement,
    compute_language_effects,
    validate_predictions,
)

RESULTS_DIR = REPO_ROOT / "results"
INPUT_PATH = RESULTS_DIR / "vla_predictions.csv"
FIGURES_DIR = RESULTS_DIR / "figures"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260809
COSINE_EPSILON = 1e-8

COLORS = {
    "correct": "#4C78A8",
    "paraphrase": "#59A14F",
    "contradictory": "#F28E2B",
    "unrelated": "#E15759",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def explicit_latency_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, subset in [("Overall", data), *[(CONDITION_LABELS[c], data[data.language_condition == c]) for c in CONDITION_ORDER]]:
        values = subset.inference_time_ms.to_numpy(dtype=float)
        rows.append(
            {
                "scope": scope,
                "actual_compute_device": subset.actual_compute_device.iloc[0],
                "count": len(values),
                "mean_inference_time_ms": float(np.mean(values)),
                "median_inference_time_ms": float(np.median(values)),
                "std_inference_time_ms": float(np.std(values, ddof=1)),
                "min_inference_time_ms": float(np.min(values)),
                "max_inference_time_ms": float(np.max(values)),
            }
        )
    return pd.DataFrame(rows)


def build_bootstrap_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(COMPARISON_ORDER):
        subset = effects[effects.comparison_condition == condition]
        for metric_index, metric in enumerate(LANGUAGE_EFFECT_METRICS):
            bootstrap = cluster_bootstrap_mean(
                subset,
                metric,
                replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED + condition_index * 100 + metric_index,
            )
            rows.append(
                {
                    "scope": "overall",
                    "comparison_condition": CONDITION_LABELS[condition],
                    "metric": metric,
                    "observation_count": len(subset),
                    **bootstrap,
                }
            )
    return pd.DataFrame(rows)


def summarize_stages(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    staged = add_task_stage(effects)
    aggregate_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(STAGE_LABELS):
        for condition_index, condition in enumerate(COMPARISON_ORDER):
            subset = staged[
                (staged.task_stage == stage) & (staged.comparison_condition == condition)
            ]
            if subset.empty:
                raise RuntimeError(f"No rows for stage={stage}, condition={condition}")
            for metric_index, metric in enumerate(LANGUAGE_EFFECT_METRICS):
                bootstrap = cluster_bootstrap_mean(
                    subset,
                    metric,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED + 1000 + stage_index * 100 + condition_index * 10 + metric_index,
                )
                aggregate_rows.append(
                    {
                        "task_stage": stage,
                        "task_stage_index": stage_index,
                        "comparison_condition": CONDITION_LABELS[condition],
                        "metric": metric,
                        "count": len(subset),
                        "mean": float(subset[metric].mean()),
                        "median": float(subset[metric].median()),
                        "std": float(subset[metric].std(ddof=1)),
                        "mean_ci95_low": bootstrap["ci95_low"],
                        "mean_ci95_high": bootstrap["ci95_high"],
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "episode_cluster_count": bootstrap["episode_cluster_count"],
                    }
                )
                for episode_id, episode_subset in subset.groupby("episode_id", sort=True):
                    episode_rows.append(
                        {
                            "episode_id": int(episode_id),
                            "task_stage": stage,
                            "task_stage_index": stage_index,
                            "comparison_condition": CONDITION_LABELS[condition],
                            "metric": metric,
                            "count": len(episode_subset),
                            "mean": float(episode_subset[metric].mean()),
                            "median": float(episode_subset[metric].median()),
                            "std": float(episode_subset[metric].std(ddof=1)),
                        }
                    )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(episode_rows)


def plot_language_action_distance(effects: pd.DataFrame) -> None:
    episode_means = effects.groupby(["episode_id", "comparison_condition"], as_index=False)[
        "full_action_language_distance"
    ].mean()
    data = [
        episode_means.loc[
            episode_means.comparison_condition == condition, "full_action_language_distance"
        ].to_numpy()
        for condition in COMPARISON_ORDER
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    box = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for patch, condition in zip(box["boxes"], COMPARISON_ORDER, strict=True):
        patch.set_facecolor(COLORS[condition])
        patch.set_alpha(0.30)
        patch.set_edgecolor(COLORS[condition])
    for index, (values, condition) in enumerate(zip(data, COMPARISON_ORDER, strict=True), start=1):
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(
            index + jitter,
            values,
            color=COLORS[condition],
            edgecolor="white",
            linewidth=0.5,
            s=34,
            zorder=3,
            label="Episode mean" if index == 1 else None,
        )
    ax.set_xticks(range(1, 4), [CONDITION_LABELS[c] for c in COMPARISON_ORDER])
    ax.set_ylabel("Episode-mean full 7-D distance from Correct")
    ax.set_title("Language-induced action distance by condition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "language_action_distance_by_condition")


def plot_expert_agreement(expert: pd.DataFrame) -> None:
    metrics = (
        ("translation_error_l2", "Translation L2 error"),
        ("rotation_error_l2", "Rotation L2 error"),
        ("gripper_absolute_error", "Gripper absolute error"),
    )
    episode_means = expert.groupby(["episode_id", "language_condition"], as_index=False)[
        [metric for metric, _ in metrics]
    ].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        values = [
            episode_means.loc[episode_means.language_condition == condition, metric].to_numpy()
            for condition in CONDITION_ORDER
        ]
        box = ax.boxplot(values, patch_artist=True, widths=0.55, showfliers=False)
        for patch, condition in zip(box["boxes"], CONDITION_ORDER, strict=True):
            patch.set_facecolor(COLORS[condition])
            patch.set_alpha(0.25)
            patch.set_edgecolor(COLORS[condition])
        for index, (points, condition) in enumerate(zip(values, CONDITION_ORDER, strict=True), start=1):
            jitter = np.linspace(-0.10, 0.10, len(points))
            ax.scatter(index + jitter, points, color=COLORS[condition], s=24, zorder=3)
        ax.set_xticks(range(1, 5), [CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=28)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Agreement with the recorded expert action (episode means)", y=1.02)
    fig.tight_layout()
    save_figure(fig, "expert_agreement_by_language")


def plot_stage_sensitivity(stage_summary: pd.DataFrame) -> None:
    metric = "full_action_language_distance"
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(STAGE_LABELS))
    for condition in COMPARISON_ORDER:
        label = CONDITION_LABELS[condition]
        subset = stage_summary[
            (stage_summary.comparison_condition == label) & (stage_summary.metric == metric)
        ].sort_values("task_stage_index")
        mean = subset["mean"].to_numpy(dtype=float)
        lower = mean - subset["mean_ci95_low"].to_numpy(dtype=float)
        upper = subset["mean_ci95_high"].to_numpy(dtype=float) - mean
        ax.errorbar(
            x,
            mean,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS[condition],
            label=label,
        )
    ax.set_xticks(x, STAGE_LABELS)
    ax.set_xlabel("Normalized task stage")
    ax.set_ylabel("Mean full 7-D distance from Correct")
    ax.set_title("Language sensitivity by task stage")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "language_sensitivity_by_task_stage")


def paired_comparison_diagnostics(effects: pd.DataFrame) -> dict[str, Any]:
    pivot = effects.pivot_table(
        index=OBSERVATION_KEY,
        columns="comparison_condition",
        values="full_action_language_distance",
        aggfunc="first",
    )
    diagnostics: dict[str, Any] = {}
    for other in ("contradictory", "unrelated"):
        diagnostics[f"paraphrase_vs_{other}"] = {
            "paraphrase_lower_count": int((pivot.paraphrase < pivot[other]).sum()),
            "observation_count": len(pivot),
            "paraphrase_lower_fraction": float((pivot.paraphrase < pivot[other]).mean()),
            "episode_mean_paraphrase_lower_count": int(
                (
                    pivot.groupby(level="episode_id").paraphrase.mean()
                    < pivot.groupby(level="episode_id")[other].mean()
                ).sum()
            ),
            "episode_count": 10,
        }
    return diagnostics


def episode_outlier_diagnostics(effects: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in COMPARISON_ORDER:
        subset = effects[effects.comparison_condition == condition]
        episode_means = subset.groupby("episode_id").full_action_language_distance.mean()
        maximum_episode = int(episode_means.idxmax())
        leave_one_out = {
            int(episode): float(subset.loc[subset.episode_id != episode, "full_action_language_distance"].mean())
            for episode in sorted(subset.episode_id.unique())
        }
        maximum_row = subset.loc[subset.full_action_language_distance.idxmax()]
        output[condition] = {
            "highest_episode_mean_episode": maximum_episode,
            "highest_episode_mean": float(episode_means.loc[maximum_episode]),
            "median_episode_mean": float(episode_means.median()),
            "leave_one_episode_out_mean_min": float(min(leave_one_out.values())),
            "leave_one_episode_out_mean_max": float(max(leave_one_out.values())),
            "maximum_observation_distance": float(maximum_row.full_action_language_distance),
            "maximum_observation_episode": int(maximum_row.episode_id),
            "maximum_observation_frame": int(maximum_row.frame_index),
            "maximum_observation_task_progress": float(maximum_row.task_progress),
        }
    return output


def write_findings(
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    stage_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    expert: pd.DataFrame,
    stage_by_episode: pd.DataFrame,
    latency: pd.DataFrame,
    cosine_undefined: int,
) -> None:
    def summary_value(family: str, condition: str, metric: str, statistic: str = "mean") -> float:
        row = summary[
            (summary.analysis_family == family)
            & (summary.condition == condition)
            & (summary.metric == metric)
        ]
        return float(row[statistic].iloc[0])

    effect_family = "paired_language_effect_from_correct"
    comparison = paired_comparison_diagnostics(effects)
    episode_diag = episode_outlier_diagnostics(effects)
    effect_means = {
        CONDITION_LABELS[c]: {
            metric: summary_value(effect_family, CONDITION_LABELS[c], metric)
            for metric in LANGUAGE_EFFECT_METRICS
        }
        for c in COMPARISON_ORDER
    }
    agreement = {
        CONDITION_LABELS[c]: {
            metric: summary_value("agreement_with_recorded_expert_action", CONDITION_LABELS[c], metric)
            for metric in (
                "translation_error_l2",
                "rotation_error_l2",
                "gripper_absolute_error",
                "translation_cosine_similarity",
            )
        }
        for c in CONDITION_ORDER
    }
    stage_ranges: dict[str, Any] = {}
    for condition in COMPARISON_ORDER:
        label = CONDITION_LABELS[condition]
        rows = stage_summary[
            (stage_summary.comparison_condition == label)
            & (stage_summary.metric == "full_action_language_distance")
        ].sort_values("task_stage_index")
        stage_ranges[label] = {
            "means": {stage: float(mean) for stage, mean in zip(rows.task_stage, rows["mean"], strict=True)},
            "minimum": float(rows["mean"].min()),
            "maximum": float(rows["mean"].max()),
        }
    bootstrap_full = bootstrap[bootstrap.metric == "full_action_language_distance"]
    bootstrap_lines = [
        f"- {row.comparison_condition}: mean {row.point_estimate:.4f}, episode-cluster 95% CI "
        f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}]."
        for row in bootstrap_full.itertuples(index=False)
    ]
    gripper_over_one = {
        CONDITION_LABELS[condition]: int(
            (
                effects.loc[
                    effects.comparison_condition == condition, "gripper_language_difference"
                ]
                > 1.0
            ).sum()
        )
        for condition in COMPARISON_ORDER
    }
    effect_medians = {
        CONDITION_LABELS[c]: {
            metric: summary_value(effect_family, CONDITION_LABELS[c], metric, "median")
            for metric in LANGUAGE_EFFECT_METRICS
        }
        for c in COMPARISON_ORDER
    }
    stage_peak_counts: dict[str, dict[str, int]] = {}
    full_episode_stages = stage_by_episode[
        stage_by_episode.metric == "full_action_language_distance"
    ]
    for condition in COMPARISON_ORDER:
        label = CONDITION_LABELS[condition]
        pivot = full_episode_stages[
            full_episode_stages.comparison_condition == label
        ].pivot(index="episode_id", columns="task_stage", values="mean")
        stage_peak_counts[label] = {
            str(stage): int(count)
            for stage, count in pivot.idxmax(axis=1).value_counts().items()
        }
    latency_overall = latency[latency.scope == "Overall"].iloc[0]
    latency_conditions = {
        row.scope: float(row.mean_inference_time_ms)
        for row in latency[latency.scope != "Overall"].itertuples(index=False)
    }

    lines = [
        "# Phase 8-9 findings",
        "",
        "These results describe this checkpoint and the 200 controlled observations only. Changing the "
        "language while images, robot state, preprocessing, checkpoint, and explicit flow noise were held "
        "fixed produced measurable changes in the predicted action.",
        "",
        "## Paired language effects",
        "",
        f"1. Paraphrase was closer to Correct than Contradictory in "
        f"{comparison['paraphrase_vs_contradictory']['paraphrase_lower_count']}/200 observations "
        f"({comparison['paraphrase_vs_contradictory']['paraphrase_lower_fraction']:.1%}) and in "
        f"{comparison['paraphrase_vs_contradictory']['episode_mean_paraphrase_lower_count']}/10 episode means. "
        f"Their mean full 7-D distances were {effect_means['Paraphrase']['full_action_language_distance']:.4f} "
        f"and {effect_means['Contradictory']['full_action_language_distance']:.4f}, respectively.",
        "",
        f"2. Paraphrase was closer to Correct than Unrelated in "
        f"{comparison['paraphrase_vs_unrelated']['paraphrase_lower_count']}/200 observations "
        f"({comparison['paraphrase_vs_unrelated']['paraphrase_lower_fraction']:.1%}) and in "
        f"{comparison['paraphrase_vs_unrelated']['episode_mean_paraphrase_lower_count']}/10 episode means. "
        f"The Unrelated mean full 7-D distance was {effect_means['Unrelated']['full_action_language_distance']:.4f}.",
        "",
        "3. Translation was the largest typical language-effect component: median translation distances were "
        f"{effect_medians['Paraphrase']['translation_language_distance']:.4f}, "
        f"{effect_medians['Contradictory']['translation_language_distance']:.4f}, and "
        f"{effect_medians['Unrelated']['translation_language_distance']:.4f}, while median gripper differences "
        f"were {effect_medians['Paraphrase']['gripper_language_difference']:.4f}, "
        f"{effect_medians['Contradictory']['gripper_language_difference']:.4f}, and "
        f"{effect_medians['Unrelated']['gripper_language_difference']:.4f}. Sparse large gripper changes make "
        f"the mean gripper difference for Unrelated ({effect_means['Unrelated']['gripper_language_difference']:.4f}) "
        f"slightly exceed its mean translation distance ({effect_means['Unrelated']['translation_language_distance']:.4f}). "
        "Rotation differences were smallest numerically. Because these components have different semantics/scales, "
        "this is not a calibrated cross-component sensitivity ranking.",
        "",
        "## Task stage and episode structure",
        "",
        "4. Mean full-action language distance varied across the five task-progress bins. The stage means were:",
        "",
    ]
    for label, values in stage_ranges.items():
        formatted = ", ".join(f"{stage}={mean:.4f}" for stage, mean in values["means"].items())
        lines.append(f"- {label}: {formatted}.")
    lines.extend(
        [
            "",
            "The pattern is descriptive rather than uniformly monotonic across all conditions; the plotted "
            "intervals use episodes—not individual frames—as bootstrap clusters.",
            f"Per-episode peak stages were distributed as follows: Paraphrase {stage_peak_counts['Paraphrase']}; "
            f"Contradictory {stage_peak_counts['Contradictory']}; Unrelated {stage_peak_counts['Unrelated']}.",
            "",
            "5. Leave-one-episode-out means were checked for every comparison. The largest episode and "
            "single-observation effects are reported below. The narrow leave-one-out aggregate ranges indicate "
            "that no overall condition mean is explained by only one episode, although stage-specific patterns "
            "are heterogeneous:",
            "",
        ]
    )
    for condition in COMPARISON_ORDER:
        diag = episode_diag[condition]
        lines.append(
            f"- {CONDITION_LABELS[condition]}: highest episode mean was episode "
            f"{diag['highest_episode_mean_episode']} ({diag['highest_episode_mean']:.4f}; median episode mean "
            f"{diag['median_episode_mean']:.4f}); leave-one-episode-out aggregate range "
            f"[{diag['leave_one_episode_out_mean_min']:.4f}, {diag['leave_one_episode_out_mean_max']:.4f}]. "
            f"The largest observation was {diag['maximum_observation_distance']:.4f} at episode "
            f"{diag['maximum_observation_episode']}, frame {diag['maximum_observation_frame']} "
            f"(progress {diag['maximum_observation_task_progress']:.3f})."
        )
    lines.extend(
        [
            "",
            "## Agreement with the recorded expert action",
            "",
            "6. Mean translation / rotation / gripper errors by condition were:",
            "",
        ]
    )
    for label, values in agreement.items():
        lines.append(
            f"- {label}: {values['translation_error_l2']:.4f} / "
            f"{values['rotation_error_l2']:.4f} / {values['gripper_absolute_error']:.4f}; "
            f"mean translation cosine similarity {values['translation_cosine_similarity']:.4f}."
        )
    lines.extend(
        [
            "",
            "These are agreement measurements against the recorded expert action, not accuracy or objective "
            "correctness. Lower errors only mean closer agreement with that recorded demonstration action. "
            "Gripper-error means are strongly right-skewed, so the medians and quartiles in summary.csv should "
            "be considered alongside the means.",
            "",
            "## Latency diagnostic",
            "",
            f"Overall inference latency was {latency_overall.mean_inference_time_ms:.1f} ms mean and "
            f"{latency_overall.median_inference_time_ms:.1f} ms median. Condition means were "
            f"{latency_conditions}. Conditions were always evaluated in a fixed order, so these differences "
            "are order/cache-confounded and are not interpreted as a language effect.",
            "",
            "## Episode-aware uncertainty and exceptions",
            "",
            *bootstrap_lines,
            "",
            f"7. Large (>1 native unit) paired gripper changes occurred in "
            f"{gripper_over_one['Paraphrase']}/200 Paraphrase, "
            f"{gripper_over_one['Contradictory']}/200 Contradictory, and "
            f"{gripper_over_one['Unrelated']}/200 Unrelated comparisons. These sparse events create strong "
            "mean/median separation and some stage spikes; their semantics are deliberately deferred to Phase 10. "
            f"Translation cosine similarity was undefined in {cosine_undefined}/800 cases because the "
            f"recorded expert or prediction translation norm was below {COSINE_EPSILON:g}; no direction was "
            "invented. Paired observation-level ordering was not universal, and the maximum-distance rows and "
            "episode heterogeneity above prevent reducing the result to a strict Paraphrase < Contradictory < "
            "Unrelated rule for every observation.",
            "",
            "No p-value battery was performed, and no claim is made that the model generally ‘understood’ the language.",
        ]
    )
    (RESULTS_DIR / "phase8_9_findings.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    input_hash_before = file_sha256(INPUT_PATH)
    data = pd.read_csv(INPUT_PATH)
    validation = validate_predictions(data)
    print(f"INPUT VALIDATION PASS: {validation}", flush=True)

    expert = compute_expert_agreement(data, cosine_epsilon=COSINE_EPSILON)
    effects = compute_language_effects(data)
    summary = build_summary(expert, effects)
    latency = explicit_latency_summary(data)
    bootstrap = build_bootstrap_summary(effects)
    stage_summary, stage_by_episode = summarize_stages(effects)

    expert.to_csv(RESULTS_DIR / "expert_agreement.csv", index=False)
    effects.to_csv(RESULTS_DIR / "language_effects.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    latency.to_csv(RESULTS_DIR / "latency_summary.csv", index=False)
    bootstrap.to_csv(RESULTS_DIR / "bootstrap_summary.csv", index=False)
    stage_summary.to_csv(RESULTS_DIR / "stage_sensitivity.csv", index=False)
    stage_by_episode.to_csv(RESULTS_DIR / "stage_sensitivity_by_episode.csv", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})
    plot_language_action_distance(effects)
    plot_expert_agreement(expert)
    plot_stage_sensitivity(stage_summary)

    cosine_undefined = int((~expert.translation_cosine_defined).sum())
    write_findings(
        summary,
        effects,
        stage_summary,
        bootstrap,
        expert,
        stage_by_episode,
        latency,
        cosine_undefined,
    )
    input_hash_after = file_sha256(INPUT_PATH)
    if input_hash_after != input_hash_before:
        raise RuntimeError("Source prediction CSV changed during analysis")

    metrics_validation = {
        "validation_result": "PASS",
        "input_file": str(INPUT_PATH),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "input_unchanged": True,
        **validation,
        "expert_agreement_row_count": len(expert),
        "paired_language_effect_row_count": len(effects),
        "summary_row_count": len(summary),
        "bootstrap_summary_row_count": len(bootstrap),
        "stage_sensitivity_row_count": len(stage_summary),
        "stage_sensitivity_by_episode_row_count": len(stage_by_episode),
        "cosine_norm_epsilon": COSINE_EPSILON,
        "translation_cosine_undefined_count": cosine_undefined,
        "cluster_bootstrap": {
            "cluster_unit": "episode_id",
            "episode_count": 10,
            "replicates": BOOTSTRAP_REPLICATES,
            "base_seed": BOOTSTRAP_SEED,
            "confidence_interval": "95% percentile",
        },
        "task_stage_bins": {
            "edges": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "labels": list(STAGE_LABELS),
            "right_open_except_final": True,
            "progress_one_in_final_bin": True,
        },
        "scientific_pairing_key": OBSERVATION_KEY,
        "correct_condition_used_only_as_baseline_in_language_effects": True,
    }
    (RESULTS_DIR / "metrics_validation.json").write_text(
        json.dumps(metrics_validation, indent=2) + "\n"
    )
    print(
        f"ANALYSIS PASS: expert_rows={len(expert)}, paired_effect_rows={len(effects)}, "
        f"stage_rows={len(stage_summary)}, cosine_undefined={cosine_undefined}",
        flush=True,
    )


if __name__ == "__main__":
    main()
