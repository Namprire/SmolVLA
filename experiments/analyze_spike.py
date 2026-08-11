#!/usr/bin/env python
"""Phase A: forensic decomposition of the 20–40% Unrelated-language spike.

This script is deliberately model-free. It reads the immutable 800-row prediction
CSV, reconstructs every altered-vs-Correct paired difference independently, and
writes only new artifacts under results/spike_analysis/.
"""

from __future__ import annotations

import hashlib
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

from src.metrics import validate_predictions  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_PATH = RESULTS_DIR / "vla_predictions.csv"
EXISTING_STAGE_PATH = RESULTS_DIR / "stage_sensitivity.csv"
OUTPUT_DIR = RESULTS_DIR / "spike_analysis"
FIGURE_DIR = OUTPUT_DIR / "figures"

EXPECTED_PREDICTIONS_SHA256 = (
    "ab4984aeea52a8ca60c94a5cb19a15d69fd27707f5c5734ddd8bec680f376dc4"
)
OBSERVATION_KEY = ["episode_id", "frame_index", "global_index"]
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
PREDICTED_COLUMNS = [f"pred_action_{name}" for name in ACTION_NAMES]
CONDITIONS = ("paraphrase", "contradictory", "unrelated")
CONDITION_LABELS = {
    "paraphrase": "Paraphrase",
    "contradictory": "Contradictory",
    "unrelated": "Unrelated",
}
PROGRESS_LABELS = ("0–20%", "20–40%", "40–60%", "60–80%", "80–100%")
COLORS = {
    "paraphrase": "#59A14F",
    "contradictory": "#F28E2B",
    "unrelated": "#E15759",
    "arm": "#4C78A8",
    "gripper": "#E15759",
    "full": "#6B5B95",
}

# Operational threshold for continuous postprocessed predictions. The empirical
# expert semantics audit establishes negative=open and positive=close. No clipping
# is applied; exact zero is assigned to the non-opening/closing regime.
GRIPPER_REGIME_THRESHOLD = 0.0
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_REPLICATES = 5000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regime(value: float) -> str:
    return "opening" if value < GRIPPER_REGIME_THRESHOLD else "closing"


def progress_bin_index(progress: np.ndarray) -> np.ndarray:
    """Left-closed bins, with progress=1 forced into the final bin."""
    return np.minimum(np.floor(progress * 5).astype(int), 4)


def reconstruct_paired_effects(predictions: pd.DataFrame) -> pd.DataFrame:
    """Independently pair each altered action with Correct for one observation."""
    indexed = predictions.set_index(OBSERVATION_KEY + ["language_condition"])
    require(indexed.index.is_unique, "Observation-condition index is not unique")
    rows: list[dict[str, Any]] = []
    observations = predictions[OBSERVATION_KEY].drop_duplicates()
    for key in observations.itertuples(index=False, name=None):
        correct_row = indexed.loc[(*key, "correct")]
        correct_action = correct_row[PREDICTED_COLUMNS].to_numpy(dtype=float)
        for condition in CONDITIONS:
            altered_row = indexed.loc[(*key, condition)]
            altered_action = altered_row[PREDICTED_COLUMNS].to_numpy(dtype=float)
            difference = altered_action - correct_action
            correct_gripper = float(correct_action[6])
            altered_gripper = float(altered_action[6])
            correct_regime = regime(correct_gripper)
            altered_regime = regime(altered_gripper)
            row: dict[str, Any] = {
                "episode_id": int(key[0]),
                "frame_index": int(key[1]),
                "global_index": int(key[2]),
                "task_progress": float(correct_row["task_progress"]),
                "comparison_condition": condition,
                "comparison_label": CONDITION_LABELS[condition],
                "noise_seed": int(correct_row["noise_seed"]),
                "noise_id": str(correct_row["noise_id"]),
                "translation_distance": float(np.linalg.norm(difference[:3])),
                "orientation_control_distance": float(np.linalg.norm(difference[3:6])),
                "gripper_absolute_difference": float(abs(difference[6])),
                "arm_only_6d_distance": float(np.linalg.norm(difference[:6])),
                "full_7d_distance": float(np.linalg.norm(difference)),
                "correct_gripper": correct_gripper,
                "altered_gripper": altered_gripper,
                "correct_gripper_regime": correct_regime,
                "altered_gripper_regime": altered_regime,
                "gripper_regime_threshold": GRIPPER_REGIME_THRESHOLD,
                "gripper_regime_flip": correct_regime != altered_regime,
            }
            for index, name in enumerate(ACTION_NAMES):
                row[f"correct_pred_action_{name}"] = float(correct_action[index])
                row[f"altered_pred_action_{name}"] = float(altered_action[index])
                row[f"signed_difference_{name}"] = float(difference[index])
            rows.append(row)

    effects = pd.DataFrame(rows)
    effects["progress_bin_index"] = progress_bin_index(
        effects.task_progress.to_numpy(dtype=float)
    )
    effects["progress_bin"] = [PROGRESS_LABELS[index] for index in effects.progress_bin_index]
    require(len(effects) == 600, f"Expected 600 paired effects, found {len(effects)}")
    metric_columns = [
        "translation_distance",
        "orientation_control_distance",
        "gripper_absolute_difference",
        "arm_only_6d_distance",
        "full_7d_distance",
    ]
    require(
        np.isfinite(effects[metric_columns].to_numpy(float)).all(),
        "A reconstructed paired metric is non-finite",
    )
    return effects


def cluster_bootstrap_interval(
    subset: pd.DataFrame, metric: str, seed: int
) -> tuple[float, float]:
    episodes = np.asarray(sorted(subset.episode_id.unique()), dtype=int)
    values_by_episode = {
        episode: subset.loc[subset.episode_id == episode, metric].to_numpy(float)
        for episode in episodes
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(episodes, size=len(episodes), replace=True)
        estimates[index] = np.concatenate(
            [values_by_episode[int(episode)] for episode in sampled]
        ).mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def build_component_summary(effects: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "translation_distance",
        "orientation_control_distance",
        "gripper_absolute_difference",
        "arm_only_6d_distance",
        "full_7d_distance",
    ]
    rows: list[dict[str, Any]] = []
    for bin_index, bin_label in enumerate(PROGRESS_LABELS):
        for condition_index, condition in enumerate(CONDITIONS):
            subset = effects[
                (effects.progress_bin_index == bin_index)
                & (effects.comparison_condition == condition)
            ]
            require(not subset.empty, f"Empty subset for {condition}, {bin_label}")
            flips = subset.gripper_regime_flip
            row: dict[str, Any] = {
                "progress_bin": bin_label,
                "progress_bin_index": bin_index,
                "progress_interval": (
                    f"[{bin_index / 5:.1f}, {(bin_index + 1) / 5:.1f})"
                    if bin_index < 4
                    else "[0.8, 1.0]"
                ),
                "comparison_condition": condition,
                "comparison_label": CONDITION_LABELS[condition],
                "n_observations": int(len(subset)),
                "n_unique_episodes": int(subset.episode_id.nunique()),
                "gripper_regime_threshold": GRIPPER_REGIME_THRESHOLD,
                "gripper_regime_rule": "opening if value < 0; closing if value >= 0",
                "gripper_regime_flip_count": int(flips.sum()),
                "gripper_regime_flip_percentage": float(100 * flips.mean()),
            }
            for metric in metric_columns:
                row[f"mean_{metric}"] = float(subset[metric].mean())
                row[f"median_{metric}"] = float(subset[metric].median())
                low, high = cluster_bootstrap_interval(
                    subset,
                    metric,
                    BOOTSTRAP_SEED + bin_index * 100 + condition_index * 10 + metric_columns.index(metric),
                )
                row[f"episode_cluster_ci95_low_{metric}"] = low
                row[f"episode_cluster_ci95_high_{metric}"] = high
            row["mean_full_7d_on_flip_observations"] = (
                float(subset.loc[flips, "full_7d_distance"].mean()) if flips.any() else np.nan
            )
            row["mean_full_7d_excluding_flip_observations"] = float(
                subset.loc[~flips, "full_7d_distance"].mean()
            )
            rows.append(row)
    return pd.DataFrame(rows)


def build_episode_breakdown(target: pd.DataFrame) -> pd.DataFrame:
    output = (
        target.groupby("episode_id", sort=True)
        .agg(
            n_observations_in_bin=("full_7d_distance", "size"),
            mean_full_7d=("full_7d_distance", "mean"),
            mean_arm_6d=("arm_only_6d_distance", "mean"),
            mean_translation=("translation_distance", "mean"),
            mean_orientation=("orientation_control_distance", "mean"),
            mean_gripper_difference=("gripper_absolute_difference", "mean"),
            gripper_flip_count=("gripper_regime_flip", "sum"),
        )
        .reset_index()
    )
    output["episode_id"] = output.episode_id.astype(int)
    output["n_observations_in_bin"] = output.n_observations_in_bin.astype(int)
    output["gripper_flip_count"] = output.gripper_flip_count.astype(int)
    return output


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_progress(summary: pd.DataFrame) -> None:
    panels = (
        ("mean_full_7d_distance", "Full 7-D distance", "A. Full 7-D"),
        ("mean_arm_only_6d_distance", "Arm-only 6-D distance", "B. Arm-only 6-D"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharex=True)
    x = np.arange(len(PROGRESS_LABELS))
    for ax, (metric, ylabel, panel_title) in zip(axes, panels, strict=True):
        for condition in CONDITIONS:
            rows = summary[summary.comparison_condition == condition].sort_values(
                "progress_bin_index"
            )
            mean = rows[metric].to_numpy(float)
            low = rows[f"episode_cluster_ci95_low_{metric.removeprefix('mean_')}"]
            high = rows[f"episode_cluster_ci95_high_{metric.removeprefix('mean_')}"]
            ax.errorbar(
                x,
                mean,
                yerr=np.vstack([mean - low.to_numpy(float), high.to_numpy(float) - mean]),
                marker="o",
                markersize=6,
                linewidth=2.1,
                capsize=3,
                color=COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        ax.axvspan(0.65, 1.35, color="#E15759", alpha=0.07, zorder=0)
        ax.set_xticks(x, PROGRESS_LABELS)
        ax.set_xlabel("Normalized trajectory-progress bin")
        ax.set_ylabel(f"Mean {ylabel} from Correct")
        ax.set_title(panel_title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Language Sensitivity Across Normalized Trajectory Progress", fontsize=15)
    fig.text(
        0.5,
        0.91,
        "Error bars resample the 10 episodes as clusters; the highlighted bin is 20–40%.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, "01_progress_full_vs_arm")


def figure_component_decomposition(target: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    metrics = [
        ("translation_distance", "Translation", COLORS["arm"]),
        ("orientation_control_distance", "Orientation\ncontrol", "#76B7B2"),
        ("gripper_absolute_difference", "Gripper", COLORS["gripper"]),
        ("arm_only_6d_distance", "Arm-only\n6-D", "#2F5597"),
        ("full_7d_distance", "Full\n7-D", COLORS["full"]),
    ]
    means = [float(target[metric].mean()) for metric, _, _ in metrics]
    bars = axes[0].bar(
        np.arange(len(metrics)), means, color=[color for _, _, color in metrics], width=0.7
    )
    axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=3, fontsize=9)
    axes[0].set_xticks(np.arange(len(metrics)), [label for _, label, _ in metrics])
    axes[0].set_ylabel("Mean distance from Correct")
    axes[0].set_title("A. Component magnitudes", loc="left", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].text(
        0.02,
        0.97,
        "Arm/full norms are not additive components.",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8.5,
        color="#555555",
    )

    groups = [target.loc[~target.gripper_regime_flip], target.loc[target.gripper_regime_flip]]
    labels = [f"No regime flip\n(n={len(groups[0])})", f"Regime flip\n(n={len(groups[1])})"]
    x = np.arange(2)
    width = 0.24
    for offset, (metric, label, color) in zip(
        (-width, 0, width),
        (
            ("arm_only_6d_distance", "Arm-only 6-D", COLORS["arm"]),
            ("gripper_absolute_difference", "Gripper", COLORS["gripper"]),
            ("full_7d_distance", "Full 7-D", COLORS["full"]),
        ),
        strict=True,
    ):
        values = [float(group[metric].mean()) for group in groups]
        axes[1].bar(x + offset, values, width=width, label=label, color=color)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Mean distance from Correct")
    axes[1].set_title("B. Effect of gripper-regime flips", loc="left", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle("Why Unrelated Reaches 1.31 in the 20–40% Bin", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "02_unrelated_20_40_component_decomposition")


def figure_episode_breakdown(episodes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.3))
    x = np.arange(len(episodes))
    width = 0.36
    full = ax.bar(
        x - width / 2,
        episodes.mean_full_7d,
        width=width,
        color=COLORS["full"],
        label="Full 7-D",
    )
    ax.bar(
        x + width / 2,
        episodes.mean_arm_6d,
        width=width,
        color=COLORS["arm"],
        label="Arm-only 6-D",
    )
    for bar, row in zip(full, episodes.itertuples(index=False), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{row.gripper_flip_count}/{row.n_observations_in_bin} flips",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7.5,
            color="#555555",
        )
    ax.set_xticks(x, [str(value) for value in episodes.episode_id])
    ax.set_xlabel("Episode ID")
    ax.set_ylabel("Episode-mean distance from Correct")
    ax.set_title("Unrelated Sensitivity in the 20–40% Bin Appears Across Episodes", fontsize=14)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.margins(y=0.18)
    fig.tight_layout()
    save_figure(fig, "03_unrelated_20_40_episode_breakdown")


def write_findings(
    source_hash: str,
    target: pd.DataFrame,
    episodes: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    flips = target.gripper_regime_flip
    full_mean = float(target.full_7d_distance.mean())
    no_flip_mean = float(target.loc[~flips, "full_7d_distance"].mean())
    flip_mean = float(target.loc[flips, "full_7d_distance"].mean())
    reduction = full_mean - no_flip_mean
    reduction_pct = 100 * reduction / full_mean
    flip_sum_share = 100 * target.loc[flips, "full_7d_distance"].sum() / target.full_7d_distance.sum()
    unrelated_bins = summary[summary.comparison_condition == "unrelated"].sort_values(
        "progress_bin_index"
    )
    other_arm_max = float(
        unrelated_bins.loc[
            unrelated_bins.progress_bin_index != 1, "mean_arm_only_6d_distance"
        ].max()
    )
    arm_mean = float(target.arm_only_6d_distance.mean())
    arm_increase_pct = 100 * (arm_mean - other_arm_max) / other_arm_max
    leave_one_out = [
        float(target.loc[target.episode_id != episode, "full_7d_distance"].mean())
        for episode in sorted(target.episode_id.unique())
    ]
    headers = list(episodes.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = []
    for row in episodes.itertuples(index=False, name=None):
        formatted = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        body_rows.append("| " + " | ".join(formatted) + " |")
    episode_table = "\n".join([header_row, separator_row, *body_rows])
    text = f"""# Phase A spike findings

## Question

Why does the red Unrelated curve rise to approximately 1.31 in the 20–40% normalized trajectory-progress bin?

## Direct answer

The spike is **mixed: gripper-regime flips strongly amplify an independent arm-control action spike**. It is not explained by either component alone.

- The bin contains {len(target)} observations from {target.episode_id.nunique()} unique episodes.
- Mean/median full 7-D distance: {full_mean:.6f} / {target.full_7d_distance.median():.6f}.
- Mean arm-only 6-D distance: {arm_mean:.6f}.
- Mean translation/orientation-control distance: {target.translation_distance.mean():.6f} / {target.orientation_control_distance.mean():.6f}.
- Mean absolute gripper difference: {target.gripper_absolute_difference.mean():.6f}.
- {int(flips.sum())}/{len(target)} observations ({100 * flips.mean():.2f}%) switch gripper regime between Correct and Unrelated.

On the {int(flips.sum())} flip observations, mean full 7-D distance is {flip_mean:.6f}; on the {int((~flips).sum())} non-flip observations it is {no_flip_mean:.6f}. The diagnostic exclusion of flip observations therefore lowers the mean from {full_mean:.6f} to {no_flip_mean:.6f}, a decrease of {reduction:.6f} ({reduction_pct:.2f}%). This exclusion is diagnostic only; all observations remain in the official result. Flip observations are {100 * flips.mean():.2f}% of the bin but contribute {flip_sum_share:.2f}% of the summed full-distance values.

The arm term also spikes. The 20–40% arm-only mean ({arm_mean:.6f}) is {arm_increase_pct:.2f}% above the largest Unrelated arm-only mean in any other progress bin ({other_arm_max:.6f}). Translation ({target.translation_distance.mean():.6f}) accounts for nearly all of that arm-only magnitude; the orientation-control mean is {target.orientation_control_distance.mean():.6f}. Thus the approximately 1.31 full-norm peak combines unusually large translation-control changes with frequent approximately two-unit gripper sign changes.

## Gripper operational rule

The stored SmolVLA gripper predictions are continuous and are never clipped for this analysis. Based on the existing expert-semantics audit, a value `< {GRIPPER_REGIME_THRESHOLD:g}` is called the opening regime and a value `>= {GRIPPER_REGIME_THRESHOLD:g}` the closing regime. A flip means Correct and Unrelated fall on opposite sides of that shown threshold. This is a command-regime comparison, not evidence of physical release.

## Episode-level check

All 10 episodes contribute to the bin. Episode-mean full distances range from {episodes.mean_full_7d.min():.6f} to {episodes.mean_full_7d.max():.6f}; arm-only episode means range from {episodes.mean_arm_6d.min():.6f} to {episodes.mean_arm_6d.max():.6f}. Leaving out one episode at a time leaves the aggregate full mean between {min(leave_one_out):.6f} and {max(leave_one_out):.6f}. The peak is therefore not created by one episode, although flip-heavy episodes have the largest full 7-D means. Episode 3 has no flip in this bin yet still has mean arm-only/full distances of {episodes.loc[episodes.episode_id == 3, 'mean_arm_6d'].iloc[0]:.6f}/{episodes.loc[episodes.episode_id == 3, 'mean_full_7d'].iloc[0]:.6f}, reinforcing the independent arm contribution.

The table below reports 10 episode summaries, not independent robot trials:

{episode_table}

## Reproducibility and scope

- Source: `results/vla_predictions.csv`
- Source SHA-256: `{source_hash}`
- Bins: `[0.0,0.2)`, `[0.2,0.4)`, `[0.4,0.6)`, `[0.6,0.8)`, and `[0.8,1.0]`.
- Episode-cluster figure intervals: {BOOTSTRAP_REPLICATES} bootstrap replicates, base seed {BOOTSTRAP_SEED}.
- This is an offline, frame-level paired action analysis. Normalized progress is not asserted to be a semantic manipulation stage.
"""
    (OUTPUT_DIR / "spike_findings.md").write_text(text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    source_hash = sha256(PREDICTIONS_PATH)
    require(
        source_hash == EXPECTED_PREDICTIONS_SHA256,
        f"Immutable prediction source hash changed: {source_hash}",
    )
    predictions = pd.read_csv(PREDICTIONS_PATH)
    validation = validate_predictions(predictions)
    effects = reconstruct_paired_effects(predictions)
    summary = build_component_summary(effects)
    target = effects[
        (effects.comparison_condition == "unrelated") & (effects.progress_bin_index == 1)
    ].copy()
    episodes = build_episode_breakdown(target)

    require(len(target) == 39, f"Expected 39 target observations, found {len(target)}")
    require(target.episode_id.nunique() == 10, "Target bin does not contain all 10 episodes")
    require(not target.duplicated(OBSERVATION_KEY).any(), "Duplicate target observations")
    require(
        not ((target.correct_gripper == GRIPPER_REGIME_THRESHOLD) | (target.altered_gripper == GRIPPER_REGIME_THRESHOLD)).any(),
        "A target gripper value equals the regime threshold; boundary handling needs review",
    )

    existing_stage = pd.read_csv(EXISTING_STAGE_PATH)
    existing_full = existing_stage[
        (existing_stage.task_stage == "20–40%")
        & (existing_stage.comparison_condition == "Unrelated")
        & (existing_stage.metric == "full_action_language_distance")
    ]
    require(len(existing_full) == 1, "Existing stage cross-check row is not unique")
    require(
        np.isclose(
            target.full_7d_distance.mean(), float(existing_full.iloc[0]["mean"]), rtol=0, atol=1e-12
        ),
        "Independent full-distance mean does not reproduce the stored stage result",
    )

    summary.to_csv(OUTPUT_DIR / "spike_component_summary.csv", index=False)
    episodes.to_csv(OUTPUT_DIR / "unrelated_20_40_episode_breakdown.csv", index=False)
    target.sort_values(OBSERVATION_KEY).to_csv(
        OUTPUT_DIR / "unrelated_20_40_observations.csv", index=False
    )
    figure_progress(summary)
    figure_component_decomposition(target)
    figure_episode_breakdown(episodes)
    write_findings(source_hash, target, episodes, summary)

    flips = target.gripper_regime_flip
    print("Phase A spike analysis: PASS")
    print(f"Validated prediction rows: {validation['row_count']}")
    print(f"Source SHA-256: {source_hash}")
    print(f"Unrelated 20–40% observations / episodes: {len(target)} / {target.episode_id.nunique()}")
    print(f"Mean / median full 7-D: {target.full_7d_distance.mean():.9f} / {target.full_7d_distance.median():.9f}")
    print(f"Mean arm-only 6-D: {target.arm_only_6d_distance.mean():.9f}")
    print(f"Mean translation: {target.translation_distance.mean():.9f}")
    print(f"Mean orientation control: {target.orientation_control_distance.mean():.9f}")
    print(f"Mean absolute gripper difference: {target.gripper_absolute_difference.mean():.9f}")
    print(f"Gripper regime flips: {int(flips.sum())}/{len(target)} ({100 * flips.mean():.3f}%)")
    print(f"Mean full 7-D on flip observations: {target.loc[flips, 'full_7d_distance'].mean():.9f}")
    print(f"Mean full 7-D excluding flips: {target.loc[~flips, 'full_7d_distance'].mean():.9f}")
    print(f"Outputs: {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
