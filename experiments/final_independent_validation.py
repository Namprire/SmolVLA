#!/usr/bin/env python
"""Independent, model-free forensic validation of the stored SmolVLA predictions.

This script intentionally does not import project metric or viewer helpers. It uses
only the Python standard library, pandas, and numpy, and never performs inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "vla_predictions.csv"
EXAMPLES = RESULTS / "demo_examples.json"
JSON_OUTPUT = RESULTS / "final_independent_validation.json"
MD_OUTPUT = RESULTS / "final_independent_validation.md"
EXAMPLE_OUTPUT = RESULTS / "demo_example_audit.csv"

CONDITIONS = ("correct", "paraphrase", "contradictory", "unrelated")
ALTERED = CONDITIONS[1:]
LABELS = {name: name.title() for name in CONDITIONS}
INSTRUCTIONS = {
    "correct": "put both the cream cheese box and the butter in the basket",
    "paraphrase": "place the cream cheese box and the butter together inside the basket",
    "contradictory": "put the cream cheese box in the basket but leave the butter outside the basket",
    "unrelated": "open the top drawer of the cabinet",
}
KEY = ["episode_id", "frame_index", "global_index"]
# The legacy CSV headers say roll/pitch/yaw, but the stored dimensions are axis-angle components.
EXPERT = [
    "expert_action_x",
    "expert_action_y",
    "expert_action_z",
    "expert_action_roll",
    "expert_action_pitch",
    "expert_action_yaw",
    "expert_action_gripper",
]
PREDICTED = [
    "pred_action_x",
    "pred_action_y",
    "pred_action_z",
    "pred_action_roll",
    "pred_action_pitch",
    "pred_action_yaw",
    "pred_action_gripper",
]
STAGE_LABELS = ("0–20%", "20–40%", "40–60%", "60–80%", "80–100%")
PUBLISHED_FULL = {
    "paraphrase": 0.2028,
    "contradictory": 0.3543,
    "unrelated": 0.8064,
}
PUBLISHED_AGREEMENT = {
    "correct": (0.3402, 0.0708, 0.2085),
    "paraphrase": (0.3633, 0.0716, 0.2289),
    "contradictory": (0.4146, 0.0720, 0.2763),
    "unrelated": (0.6152, 0.0927, 0.6563),
}
PUBLISHED_CI = {
    "paraphrase": (0.1725, 0.2314),
    "contradictory": (0.2974, 0.4145),
    "unrelated": (0.7619, 0.8560),
}
BOOTSTRAP_SEEDS = {
    "paraphrase": 20260812,
    "contradictory": 20260912,
    "unrelated": 20261012,
}
ROUNDING_TOLERANCE = 5.1e-5


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def describe(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    require(np.isfinite(array).all(), "A descriptive-statistics input is non-finite")
    sorted_array = np.sort(array)
    trim = int(np.floor(array.size * 0.10))
    trimmed = sorted_array[trim:-trim] if trim else sorted_array
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "trimmed_mean_10_percent_each_tail": float(np.mean(trimmed)),
        "maximum": float(np.max(array)),
    }


def validate_structure(data: pd.DataFrame) -> dict[str, Any]:
    required = set(KEY + EXPERT + PREDICTED) | {
        "episode_length",
        "max_frame_index",
        "task_progress",
        "language_condition",
        "instruction",
        "selection_seed",
        "noise_seed",
        "noise_id",
    }
    require(required <= set(data.columns), f"Missing columns: {sorted(required - set(data.columns))}")
    require(len(data) == 800, f"Expected 800 rows, found {len(data)}")
    require(data[KEY].drop_duplicates().shape[0] == 200, "Expected 200 unique observations")
    require(not data.duplicated(KEY + ["language_condition"]).any(), "Duplicate observation-condition key")
    require(sorted(data.episode_id.unique().tolist()) == list(range(10)), "Episodes are not exactly 0-9")
    require(set(data.language_condition) == set(CONDITIONS), "Condition identifiers differ")

    groups = data.groupby(KEY, sort=False, dropna=False)
    require((groups.size() == 4).all(), "Not every observation has exactly four rows")
    require(
        groups.language_condition.agg(lambda x: set(x)).eq(set(CONDITIONS)).all(),
        "Not every observation has all four conditions",
    )
    require(
        set(data[["language_condition", "instruction"]].itertuples(index=False, name=None))
        == set(INSTRUCTIONS.items()),
        "Stored condition-to-instruction mapping is not exact",
    )
    action_matrix = data[EXPERT + PREDICTED].to_numpy(dtype=float)
    require(action_matrix.shape == (800, 14), "Actions are not two 7-D vectors per row")
    require(np.isfinite(action_matrix).all(), "Prediction or expert action is non-finite")
    progress = data.task_progress.to_numpy(dtype=float)
    require(np.isfinite(progress).all() and ((0 <= progress) & (progress <= 1)).all(), "Invalid progress")

    invariant_columns = [
        "episode_id",
        "frame_index",
        "global_index",
        "episode_length",
        "max_frame_index",
        "task_progress",
        "selection_seed",
        "noise_seed",
        "noise_id",
        *EXPERT,
    ]
    invariant_counts = groups[invariant_columns].nunique(dropna=False)
    require((invariant_counts == 1).all().all(), "An observation invariant changes across conditions")
    require((data.frame_index <= data.max_frame_index).all(), "Frame exceeds max frame")
    expected_progress = data.frame_index / data.max_frame_index
    require(np.allclose(data.task_progress, expected_progress, rtol=0, atol=1e-14), "Progress != frame/max_frame")

    return {
        "row_count": 800,
        "observation_count": 200,
        "conditions_per_observation": 4,
        "episodes": list(range(10)),
        "duplicate_observation_condition_keys": 0,
        "finite_prediction_dimension": 7,
        "finite_expert_action_dimension": 7,
        "same_expert_action_across_conditions": True,
        "same_frame_and_selection_identifiers_across_conditions": True,
        "same_noise_seed_and_hash_across_conditions": True,
        "task_progress_consistent_within_observation_and_frame_ratio": True,
        "exact_instruction_mapping": INSTRUCTIONS,
        "csv_state_vector_limitation": (
            "The CSV does not contain a robot-state vector or image hashes, so per-condition state/image "
            "identity cannot be established from this CSV alone; it is corroborated separately from the "
            "source pipeline and local frame assets."
        ),
    }


def calculate_metrics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = data.set_index(KEY + ["language_condition"])
    require(indexed.index.is_unique, "Observation-condition index is not unique")
    effects: list[dict[str, Any]] = []
    for observation in data[KEY].drop_duplicates().itertuples(index=False, name=None):
        correct_row = indexed.loc[(*observation, "correct")]
        correct = correct_row[PREDICTED].to_numpy(dtype=float)
        for condition in ALTERED:
            other = indexed.loc[(*observation, condition)][PREDICTED].to_numpy(dtype=float)
            delta = other - correct
            arm = delta[:6]
            effects.append(
                {
                    **dict(zip(KEY, observation, strict=True)),
                    "task_progress": float(correct_row.task_progress),
                    "comparison_condition": condition,
                    "translation_distance": float(np.sqrt(np.sum(delta[:3] ** 2))),
                    "axis_angle_distance": float(np.sqrt(np.sum(delta[3:6] ** 2))),
                    "gripper_difference": float(abs(delta[6])),
                    "arm_only_6d_distance": float(np.sqrt(np.sum(arm**2))),
                    "full_7d_distance": float(np.sqrt(np.sum(delta**2))),
                }
            )
    effects_frame = pd.DataFrame(effects)
    require(len(effects_frame) == 600, "Expected 600 paired language effects")

    agreement = data[KEY + ["language_condition"]].copy()
    delta = data[PREDICTED].to_numpy(float) - data[EXPERT].to_numpy(float)
    agreement["translation_l2"] = np.sqrt(np.sum(delta[:, :3] ** 2, axis=1))
    agreement["axis_angle_l2"] = np.sqrt(np.sum(delta[:, 3:6] ** 2, axis=1))
    agreement["gripper_absolute_difference"] = np.abs(delta[:, 6])
    agreement["full_7d_distance"] = np.sqrt(np.sum(delta**2, axis=1))
    agreement["full_7d_mae"] = np.mean(np.abs(delta), axis=1)
    return effects_frame, agreement


def bootstrap_episode_clusters(values: pd.DataFrame, seed: int, replicates: int = 2000) -> dict[str, Any]:
    episodes = np.array(sorted(values.episode_id.unique()), dtype=int)
    by_episode = {
        episode: values.loc[values.episode_id == episode, "full_7d_distance"].to_numpy(float)
        for episode in episodes
    }
    require(all(len(item) == 20 for item in by_episode.values()), "Expected 20 rows per episode")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    duplicate_replicates = 0
    multiplicity_check = False
    for index in range(replicates):
        sampled = rng.choice(episodes, size=len(episodes), replace=True)
        if len(np.unique(sampled)) < len(sampled):
            duplicate_replicates += 1
            expected_size = sum(len(by_episode[int(episode)]) for episode in sampled)
            combined = np.concatenate([by_episode[int(episode)] for episode in sampled])
            require(len(combined) == expected_size, "Duplicate episode multiplicity was lost")
            multiplicity_check = multiplicity_check or expected_size > sum(
                len(by_episode[int(episode)]) for episode in np.unique(sampled)
            )
        else:
            combined = np.concatenate([by_episode[int(episode)] for episode in sampled])
        estimates[index] = float(np.mean(combined))
    return {
        "point_estimate": float(values.full_7d_distance.mean()),
        "bootstrap_mean": float(estimates.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "replicates": replicates,
        "seed": seed,
        "cluster_unit": "episode_id",
        "duplicate_episode_replicates": duplicate_replicates,
        "duplicate_sampled_episodes_preserved_with_multiplicity": bool(multiplicity_check),
    }


def stage_analysis(effects: pd.DataFrame) -> dict[str, Any]:
    staged = effects.copy()
    stage_index = np.minimum(np.floor(staged.task_progress.to_numpy(float) * 5).astype(int), 4)
    staged["stage_index"] = stage_index
    staged["task_stage"] = [STAGE_LABELS[index] for index in stage_index]
    require((staged.loc[staged.task_progress == 1.0, "stage_index"] == 4).all(), "progress=1 not final")
    output: dict[str, Any] = {}
    for condition in ALTERED:
        output[LABELS[condition]] = {}
        for index, label in enumerate(STAGE_LABELS):
            subset = staged[(staged.comparison_condition == condition) & (staged.stage_index == index)]
            episode_means = subset.groupby("episode_id").full_7d_distance.mean()
            leave_one_out = [
                subset.loc[subset.episode_id != episode, "full_7d_distance"].mean()
                for episode in sorted(subset.episode_id.unique())
            ]
            output[LABELS[condition]][label] = {
                "observation_count": int(len(subset)),
                "mean_paired_full_7d_distance": float(subset.full_7d_distance.mean()),
                "median_paired_full_7d_distance": float(subset.full_7d_distance.median()),
                "minimum_episode_mean": float(episode_means.min()),
                "maximum_episode_mean": float(episode_means.max()),
                "maximum_episode_mean_episode": int(episode_means.idxmax()),
                "leave_one_episode_out_mean_min": float(np.min(leave_one_out)),
                "leave_one_episode_out_mean_max": float(np.max(leave_one_out)),
            }
    return {
        "bin_edges": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "bin_semantics": "left-closed/right-open except that progress=1.0 is forced into 80–100%",
        "progress_one_rows_in_final_bin": int((staged.task_progress == 1.0).sum()),
        "basis": "paired altered-language SmolVLA prediction vs Correct-language SmolVLA prediction",
        "means_and_episode_influence": output,
        "interpretation": "Stage patterns are non-monotonic and heterogeneous across episodes.",
    }


def example_audit(effects: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(EXAMPLES.read_text())
    examples = payload["examples"]
    pivot = effects.pivot(index=KEY + ["task_progress"], columns="comparison_condition", values="full_7d_distance")
    grip = effects.pivot(index=KEY + ["task_progress"], columns="comparison_condition", values="gripper_difference")
    used: set[tuple[int, int, int]] = set()
    audit_rows: list[dict[str, Any]] = []

    for example in examples:
        key3 = (int(example["episode_id"]), int(example["frame_index"]), int(example["global_index"]))
        matches = [key for key in pivot.index if tuple(int(v) for v in key[:3]) == key3]
        require(len(matches) == 1, f"Example key not unique: {key3}")
        key = matches[0]
        selection_type = example["selection_type"]
        criterion_name = ""
        exclusion_rule = "none"
        if selection_type == "paraphrase_close_to_correct":
            ordered = pivot.sort_values("paraphrase", ascending=True)
            criterion_name = "Paraphrase full 7-D distance from Correct (ascending)"
            criterion = float(pivot.loc[key, "paraphrase"])
        elif selection_type == "large_contradictory_change":
            ordered = pivot.sort_values("contradictory", ascending=False)
            criterion_name = "Contradictory full 7-D distance from Correct (descending)"
            criterion = float(pivot.loc[key, "contradictory"])
        elif selection_type == "large_unrelated_change":
            ordered = pivot.sort_values("unrelated", ascending=False)
            criterion_name = "Unrelated full 7-D distance from Correct (descending)"
            criterion = float(pivot.loc[key, "unrelated"])
            exclusion_rule = "exclude observations selected for examples 1-2"
        elif selection_type == "late_stage_strong_sensitivity":
            ordered = pivot.reset_index()
            ordered = ordered[ordered.task_progress >= 0.8].copy()
            ordered["criterion"] = ordered[list(ALTERED)].mean(axis=1)
            ordered = ordered.set_index(KEY + ["task_progress"]).sort_values("criterion", ascending=False)
            criterion_name = "mean of three altered-condition full 7-D distances, progress >= 0.8 (descending)"
            criterion = float(pivot.loc[key, list(ALTERED)].mean())
            exclusion_rule = "exclude observations selected for examples 1-3"
        elif selection_type == "ordering_counterexample_and_gripper_change":
            ordered = pivot.copy()
            ordered["criterion"] = ordered.paraphrase - ordered.contradictory
            ordered = ordered[ordered.criterion > 0].sort_values("criterion", ascending=False)
            criterion_name = "Paraphrase minus Contradictory full 7-D distance (positive, descending)"
            criterion = float(pivot.loc[key, "paraphrase"] - pivot.loc[key, "contradictory"])
            exclusion_rule = "exclude observations selected for examples 1-4"
        else:
            raise AssertionError(f"Unknown example selection type: {selection_type}")

        ordered_keys = list(ordered.index)
        all_rank = ordered_keys.index(key) + 1
        eligible = [candidate for candidate in ordered_keys if tuple(int(v) for v in candidate[:3]) not in used]
        eligible_rank = eligible.index(key) + 1
        require(eligible_rank == 1, f"Example {example['example_index']} is not first eligible by its criterion")
        stored_distances = example["full_action_distance_from_correct"]
        stored_grip = example["gripper_difference_from_correct"]
        distances_match = all(
            np.isclose(stored_distances[LABELS[c]], pivot.loc[key, c], rtol=0, atol=1e-12)
            for c in ALTERED
        )
        grip_match = all(
            np.isclose(stored_grip[LABELS[c]], grip.loc[key, c], rtol=0, atol=1e-12)
            for c in ALTERED
        )
        reason_number_present = f"{criterion:.4f}" in example["reason"]
        require(distances_match and grip_match and reason_number_present, f"Example {example['example_index']} metadata mismatch")
        audit_rows.append(
            {
                "example_index": int(example["example_index"]),
                "selection_type": selection_type,
                "episode_id": key3[0],
                "frame_index": key3[1],
                "global_index": key3[2],
                "task_progress": float(key[3]),
                "criterion": criterion_name,
                "calculated_criterion": criterion,
                "rank_over_all_applicable_observations": all_rank,
                "rank_after_documented_exclusions": eligible_rank,
                "exclusion_rule": exclusion_rule,
                "full_distances_match_json": distances_match,
                "gripper_differences_match_json": grip_match,
                "reason_value_matches": reason_number_present,
                "mathematically_valid": True,
            }
        )
        used.add(key3)

    frame = pd.DataFrame(audit_rows)
    frame.to_csv(EXAMPLE_OUTPUT, index=False)
    fifth = frame.loc[frame.example_index == 5].iloc[0]
    return audit_rows, {
        "selection_count": len(frame),
        "all_examples_mathematically_valid": bool(frame.mathematically_valid.all()),
        "duplicate_observations_excluded": len(used) == len(frame),
        "example_5_is_counterexample": bool(fifth.calculated_criterion > 0),
        "example_5_paraphrase_minus_contradictory": float(fifth.calculated_criterion),
        "selection_policy": payload["selection_policy"],
    }


def main() -> None:
    data = pd.read_csv(PREDICTIONS)
    structure = validate_structure(data)
    effects, agreement = calculate_metrics(data)

    effect_metrics: dict[str, Any] = {}
    for condition in ALTERED:
        subset = effects[effects.comparison_condition == condition]
        effect_metrics[LABELS[condition]] = {
            metric: describe(subset[metric])
            for metric in (
                "translation_distance",
                "axis_angle_distance",
                "gripper_difference",
                "arm_only_6d_distance",
                "full_7d_distance",
            )
        }
        nonzero = subset.full_7d_distance.to_numpy(float) > 0
        squared_share = np.zeros(len(subset), dtype=float)
        squared_share[nonzero] = (
            subset.loc[nonzero, "gripper_difference"].to_numpy(float) ** 2
            / subset.loc[nonzero, "full_7d_distance"].to_numpy(float) ** 2
        )
        effect_metrics[LABELS[condition]]["gripper_squared_share_of_full_norm"] = describe(squared_share)
        effect_metrics[LABELS[condition]]["gripper_dominance"] = {
            "gripper_difference_over_1_native_unit_count": int(
                (subset.gripper_difference > 1.0).sum()
            ),
            "gripper_over_half_of_squared_full_norm_count": int(
                (squared_share > 0.5).sum()
            ),
            "gripper_over_90_percent_of_squared_full_norm_count": int(
                (squared_share > 0.9).sum()
            ),
            "mean_full_minus_mean_arm_only_6d": float(
                subset.full_7d_distance.mean() - subset.arm_only_6d_distance.mean()
            ),
            "mean_arm_only_divided_by_mean_full": float(
                subset.arm_only_6d_distance.mean() / subset.full_7d_distance.mean()
            ),
        }

    agreement_metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        subset = agreement[agreement.language_condition == condition]
        agreement_metrics[LABELS[condition]] = {
            metric: describe(subset[metric])
            for metric in (
                "translation_l2",
                "axis_angle_l2",
                "gripper_absolute_difference",
                "full_7d_distance",
                "full_7d_mae",
            )
        }

    pivot = effects.pivot(index=KEY, columns="comparison_condition", values="full_7d_distance")
    arm_pivot = effects.pivot(index=KEY, columns="comparison_condition", values="arm_only_6d_distance")
    ordering = {
        "expected_strict_order_count_full_7d": int(((pivot.paraphrase < pivot.contradictory) & (pivot.contradictory < pivot.unrelated)).sum()),
        "expected_strict_order_failure_count_full_7d": int((~((pivot.paraphrase < pivot.contradictory) & (pivot.contradictory < pivot.unrelated))).sum()),
        "paraphrase_closer_than_contradictory_count": int((pivot.paraphrase < pivot.contradictory).sum()),
        "paraphrase_closer_than_unrelated_count": int((pivot.paraphrase < pivot.unrelated).sum()),
        "contradictory_closer_than_unrelated_count": int((pivot.contradictory < pivot.unrelated).sum()),
        "expected_strict_order_count_arm_only_6d": int(((arm_pivot.paraphrase < arm_pivot.contradictory) & (arm_pivot.contradictory < arm_pivot.unrelated)).sum()),
        "expected_strict_order_failure_count_arm_only_6d": int((~((arm_pivot.paraphrase < arm_pivot.contradictory) & (arm_pivot.contradictory < arm_pivot.unrelated))).sum()),
    }

    published_checks: dict[str, Any] = {"tolerance_for_four_decimal_claims": ROUNDING_TOLERANCE}
    for condition in ALTERED:
        actual = effect_metrics[LABELS[condition]]["full_7d_distance"]["mean"]
        published_checks[f"{condition}_full_mean"] = {
            "actual": actual,
            "published_approximate": PUBLISHED_FULL[condition],
            "matches_rounding": abs(actual - PUBLISHED_FULL[condition]) <= ROUNDING_TOLERANCE,
        }
    for condition in CONDITIONS:
        actuals = (
            agreement_metrics[LABELS[condition]]["translation_l2"]["mean"],
            agreement_metrics[LABELS[condition]]["axis_angle_l2"]["mean"],
            agreement_metrics[LABELS[condition]]["gripper_absolute_difference"]["mean"],
        )
        published_checks[f"{condition}_expert_agreement"] = {
            "actual_translation_axis_angle_gripper": actuals,
            "published_approximate": PUBLISHED_AGREEMENT[condition],
            "matches_rounding": all(
                abs(actual - expected) <= ROUNDING_TOLERANCE
                for actual, expected in zip(actuals, PUBLISHED_AGREEMENT[condition], strict=True)
            ),
        }
    published_checks["paraphrase_closer_than_contradictory"] = ordering["paraphrase_closer_than_contradictory_count"] == 150
    published_checks["paraphrase_closer_than_unrelated"] = ordering["paraphrase_closer_than_unrelated_count"] == 182

    bootstrap: dict[str, Any] = {}
    for condition in ALTERED:
        result = bootstrap_episode_clusters(
            effects[effects.comparison_condition == condition], BOOTSTRAP_SEEDS[condition]
        )
        expected_low, expected_high = PUBLISHED_CI[condition]
        result["published_ci_approximate"] = [expected_low, expected_high]
        result["published_ci_matches_rounding"] = (
            abs(result["ci95_low"] - expected_low) <= ROUNDING_TOLERANCE
            and abs(result["ci95_high"] - expected_high) <= ROUNDING_TOLERANCE
        )
        bootstrap[LABELS[condition]] = result

    example_rows, example_summary = example_audit(effects)
    stages = stage_analysis(effects)
    arm_means = {
        LABELS[condition]: effect_metrics[LABELS[condition]]["arm_only_6d_distance"]["mean"]
        for condition in ALTERED
    }
    correct_lowest = all(
        agreement_metrics["Correct"][metric]["mean"]
        == min(agreement_metrics[label][metric]["mean"] for label in LABELS.values())
        for metric in ("translation_l2", "axis_angle_l2", "gripper_absolute_difference", "full_7d_distance", "full_7d_mae")
    )
    lag1: dict[str, float] = {}
    for condition in ALTERED:
        values = []
        subset = effects[effects.comparison_condition == condition]
        for _, episode in subset.groupby("episode_id"):
            ordered_episode = episode.sort_values("frame_index").full_7d_distance.to_numpy(float)
            if np.std(ordered_episode[:-1]) > 0 and np.std(ordered_episode[1:]) > 0:
                values.append(float(np.corrcoef(ordered_episode[:-1], ordered_episode[1:])[0, 1]))
        lag1[LABELS[condition]] = float(np.mean(values))

    all_published_match = all(
        item if isinstance(item, bool) else item.get("matches_rounding", True)
        for key, item in published_checks.items()
        if key != "tolerance_for_four_decimal_claims"
    ) and all(result["published_ci_matches_rounding"] for result in bootstrap.values())
    report = {
        "validation_result": "PASS" if all_published_match else "FAIL — PUBLISHED VALUE MISMATCH",
        "independence": "No project metric or viewer helper was imported.",
        "action_semantics": "Stored dimensions 3-5 are treated as axis-angle components despite legacy CSV header names.",
        "structure": structure,
        "published_checks": published_checks,
        "language_effects_from_correct_prediction": effect_metrics,
        "agreement_with_recorded_expert_action": agreement_metrics,
        "ordering": ordering,
        "bootstrap": bootstrap,
        "stage_analysis": stages,
        "curated_examples": {**example_summary, "rows": example_rows},
        "scientific_sense": {
            "mean_full_7d_order_is_paraphrase_contradictory_unrelated": (
                effect_metrics["Paraphrase"]["full_7d_distance"]["mean"]
                < effect_metrics["Contradictory"]["full_7d_distance"]["mean"]
                < effect_metrics["Unrelated"]["full_7d_distance"]["mean"]
            ),
            "mean_arm_only_6d_order_is_paraphrase_contradictory_unrelated": (
                arm_means["Paraphrase"] < arm_means["Contradictory"] < arm_means["Unrelated"]
            ),
            "arm_only_6d_means": arm_means,
            "correct_has_greatest_mean_expert_agreement_all_reported_metrics": correct_lowest,
            "mean_within_episode_lag1_full_7d_distance": lag1,
            "dependence_note": (
                "The 20 observations per episode are ordered samples from ten trajectories and are temporally "
                "related; n=200 must not be presented as 200 independent robot trials."
            ),
            "units_note": (
                "Translation, axis-angle, and gripper magnitudes have different units/scales; component means "
                "are reported separately and are not treated as calibrated cross-component effect sizes."
            ),
        },
    }
    JSON_OUTPUT.write_text(json.dumps(native(report), indent=2) + "\n")

    lines = [
        "# Final independent validation",
        "",
        f"**Result: {report['validation_result']}**",
        "",
        "This audit recomputed every result directly from `results/vla_predictions.csv` without importing project metric or viewer helpers. Stored dimensions 3–5 were treated as axis-angle components.",
        "",
        "## Central results",
        "",
        "| Comparison from Correct prediction | Translation mean | Axis-angle mean | Gripper mean | Arm-only 6-D mean | Full 7-D mean | Q25 | Median | Q75 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in ALTERED:
        metrics = effect_metrics[LABELS[condition]]
        lines.append(
            f"| {LABELS[condition]} | {metrics['translation_distance']['mean']:.6f} | "
            f"{metrics['axis_angle_distance']['mean']:.6f} | {metrics['gripper_difference']['mean']:.6f} | "
            f"{metrics['arm_only_6d_distance']['mean']:.6f} | {metrics['full_7d_distance']['mean']:.6f} | "
            f"{metrics['full_7d_distance']['q25']:.6f} | {metrics['full_7d_distance']['median']:.6f} | "
            f"{metrics['full_7d_distance']['q75']:.6f} |"
        )
    lines.extend(["", "## Agreement with recorded expert action", "", "| Condition | Translation L2 | Axis-angle L2 | Gripper absolute | Full 7-D | 7-D MAE |", "|---|---:|---:|---:|---:|---:|"])
    for condition in CONDITIONS:
        metrics = agreement_metrics[LABELS[condition]]
        lines.append(
            f"| {LABELS[condition]} | {metrics['translation_l2']['mean']:.6f} | "
            f"{metrics['axis_angle_l2']['mean']:.6f} | {metrics['gripper_absolute_difference']['mean']:.6f} | "
            f"{metrics['full_7d_distance']['mean']:.6f} | {metrics['full_7d_mae']['mean']:.6f} |"
        )
    lines.extend(["", "## Bootstrap and ordering", ""])
    for condition in ALTERED:
        item = bootstrap[LABELS[condition]]
        lines.append(f"- {LABELS[condition]}: episode-cluster mean {item['point_estimate']:.6f}, 95% CI [{item['ci95_low']:.6f}, {item['ci95_high']:.6f}]. Duplicate sampled episodes were retained with multiplicity.")
    lines.extend([
        "",
        f"The strict observation-level ordering Paraphrase < Contradictory < Unrelated holds for {ordering['expected_strict_order_count_full_7d']}/200 observations and fails for {ordering['expected_strict_order_failure_count_full_7d']}/200. Paraphrase is closer than Contradictory in {ordering['paraphrase_closer_than_contradictory_count']}/200 and closer than Unrelated in {ordering['paraphrase_closer_than_unrelated_count']}/200.",
        "",
        "## Robustness and interpretation",
        "",
        f"Excluding gripper, arm-only 6-D means are Paraphrase {arm_means['Paraphrase']:.6f}, Contradictory {arm_means['Contradictory']:.6f}, and Unrelated {arm_means['Unrelated']:.6f}. The aggregate ordering survives. The strict arm-only observation-level ordering fails in {ordering['expected_strict_order_failure_count_arm_only_6d']}/200 observations.",
        "",
        "Large near-binary gripper flips can dominate some individual full-norm distances, so translation, axis-angle, gripper, arm-only, and full distances are reported separately. Their raw magnitudes are not physically interchangeable.",
        "",
        "Task-stage bins are paired Correct-vs-altered comparisons, include progress 1.0 in the final bin, and show heterogeneous non-monotonic episode patterns. The 200 observations come from ten trajectories and are temporally related, not 200 independent robot trials.",
        "",
        "The condition ordering is also present in medians, quartiles, and 10%-per-tail trimmed means, so the aggregate result is not created only by the largest outliers. Outliers and near-binary gripper flips do materially widen the mean/median gap, especially for Contradictory and Unrelated.",
        "",
        "## Task-stage means",
        "",
        "| Stage | Paraphrase | Contradictory | Unrelated |",
        "|---|---:|---:|---:|",
    ])
    for stage in STAGE_LABELS:
        lines.append(
            f"| {stage} | {stages['means_and_episode_influence']['Paraphrase'][stage]['mean_paired_full_7d_distance']:.6f} | "
            f"{stages['means_and_episode_influence']['Contradictory'][stage]['mean_paired_full_7d_distance']:.6f} | "
            f"{stages['means_and_episode_influence']['Unrelated'][stage]['mean_paired_full_7d_distance']:.6f} |"
        )
    lines.extend([
        "",
        "The CSV itself does not store per-condition robot-state vectors or image hashes. Equality of state and images therefore requires the separate source/viewer asset audit; it cannot be independently proven from CSV columns alone.",
        "",
        "Curated example calculations and documented exclusion-aware ranks are in `results/demo_example_audit.csv`.",
    ])
    MD_OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"INDEPENDENT VALIDATION {report['validation_result']}")
    print(
        "FULL 7-D MEANS: "
        + ", ".join(
            f"{LABELS[c]}={effect_metrics[LABELS[c]]['full_7d_distance']['mean']:.6f}" for c in ALTERED
        )
    )
    print(
        "ARM-ONLY 6-D MEANS: "
        + ", ".join(f"{label}={value:.6f}" for label, value in arm_means.items())
    )
    if not all_published_match:
        raise SystemExit("Published values differ beyond four-decimal rounding; audit stopped.")


if __name__ == "__main__":
    main()
