"""Paired metrics for the controlled SmolVLA language experiment."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

CONDITION_ORDER = ("correct", "paraphrase", "contradictory", "unrelated")
COMPARISON_ORDER = CONDITION_ORDER[1:]
CONDITION_LABELS = {
    "correct": "Correct",
    "paraphrase": "Paraphrase",
    "contradictory": "Contradictory",
    "unrelated": "Unrelated",
}
EXPECTED_INSTRUCTIONS = {
    "correct": "put both the cream cheese box and the butter in the basket",
    "paraphrase": "place the cream cheese box and the butter together inside the basket",
    "contradictory": "put the cream cheese box in the basket but leave the butter outside the basket",
    "unrelated": "open the top drawer of the cabinet",
}

# Preserve immutable CSV/derived-table compatibility. The legacy roll/pitch/yaw
# field names represent three stored axis-angle components, not Euler angles.
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
EXPERT_COLUMNS = [f"expert_action_{name}" for name in ACTION_NAMES]
PRED_COLUMNS = [f"pred_action_{name}" for name in ACTION_NAMES]
OBSERVATION_KEY = ["episode_id", "frame_index", "global_index"]
STAGE_LABELS = ("0–20%", "20–40%", "40–60%", "60–80%", "80–100%")
LANGUAGE_EFFECT_METRICS = (
    "translation_language_distance",
    "rotation_language_distance",
    "gripper_language_difference",
    "full_action_language_distance",
)
EXPERT_AGREEMENT_METRICS = (
    "translation_error_l2",
    "rotation_error_l2",
    "gripper_absolute_error",
    "overall_action_mae",
    "translation_cosine_similarity",
)


def validate_predictions(data: pd.DataFrame) -> dict[str, object]:
    """Fail loudly unless every scientific pairing assumption is satisfied."""
    required = {
        *OBSERVATION_KEY,
        "task_progress",
        "language_condition",
        "instruction",
        "noise_seed",
        "noise_id",
        "actual_compute_device",
        "inference_time_ms",
        *EXPERT_COLUMNS,
        *PRED_COLUMNS,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")
    if len(data) != 800:
        raise ValueError(f"Expected 800 prediction rows, found {len(data)}")
    if data.duplicated(OBSERVATION_KEY + ["language_condition"]).any():
        raise ValueError("Duplicate observation/condition keys found")

    observation_count = data[OBSERVATION_KEY].drop_duplicates().shape[0]
    if observation_count != 200:
        raise ValueError(f"Expected 200 observations, found {observation_count}")
    condition_set = set(data["language_condition"].astype(str))
    if condition_set != set(CONDITION_ORDER):
        raise ValueError(
            f"Expected exact stored condition identifiers {set(CONDITION_ORDER)}, found {condition_set}"
        )

    groups = data.groupby(OBSERVATION_KEY, sort=False, dropna=False)
    sizes = groups.size()
    if not (sizes == 4).all():
        raise ValueError(f"Not every observation has four rows: {sizes.value_counts().to_dict()}")
    condition_sets = groups["language_condition"].agg(lambda values: set(values.astype(str)))
    if not all(value == set(CONDITION_ORDER) for value in condition_sets):
        raise ValueError("Not every observation contains the exact four conditions")
    instruction_pairs = set(
        data[["language_condition", "instruction"]].itertuples(index=False, name=None)
    )
    expected_instruction_pairs = set(EXPECTED_INSTRUCTIONS.items())
    if instruction_pairs != expected_instruction_pairs:
        raise ValueError(
            "Condition/instruction mapping differs from the validated language intervention: "
            f"expected {expected_instruction_pairs}, found {instruction_pairs}"
        )
    if not (groups["noise_seed"].nunique(dropna=False) == 1).all():
        raise ValueError("Noise seed differs across conditions within an observation")
    if not (groups["noise_id"].nunique(dropna=False) == 1).all():
        raise ValueError("Noise hash differs across conditions within an observation")
    if not (groups["task_progress"].nunique(dropna=False) == 1).all():
        raise ValueError("task_progress differs across conditions within an observation")
    if not (groups[EXPERT_COLUMNS].nunique(dropna=False) == 1).all().all():
        raise ValueError("Recorded expert action differs across conditions within an observation")

    actions = data[EXPERT_COLUMNS + PRED_COLUMNS].to_numpy(dtype=float)
    if actions.shape != (800, 14):
        raise ValueError(f"Expected two 7-D action vectors per row, got matrix {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Expert or predicted actions contain NaN/infinity")
    progress = data["task_progress"].to_numpy(dtype=float)
    if not np.isfinite(progress).all() or np.any((progress < 0) | (progress > 1)):
        raise ValueError("task_progress is non-finite or outside [0, 1]")
    latency = data["inference_time_ms"].to_numpy(dtype=float)
    if not np.isfinite(latency).all() or np.any(latency <= 0):
        raise ValueError("Inference timing is non-finite or non-positive")
    episodes = sorted(int(value) for value in data["episode_id"].unique())
    if episodes != list(range(10)):
        raise ValueError(f"Expected episodes 0-9, found {episodes}")
    devices = sorted(set(data["actual_compute_device"].astype(str)))
    if devices != ["mps:0"]:
        raise ValueError(f"Expected one validated MPS device, found {devices}")

    return {
        "row_count": len(data),
        "observation_count": observation_count,
        "episode_count": len(episodes),
        "episodes": episodes,
        "rows_per_observation": 4,
        "stored_condition_identifiers": list(CONDITION_ORDER),
        "presentation_condition_labels": CONDITION_LABELS,
        "exact_instruction_mapping": EXPECTED_INSTRUCTIONS,
        "identical_noise_within_observation": True,
        "identical_expert_action_within_observation": True,
        "finite_7d_actions": True,
        "task_progress_in_unit_interval": True,
        "positive_finite_inference_time": True,
        "actual_compute_device": devices[0],
    }


def compute_expert_agreement(data: pd.DataFrame, cosine_epsilon: float = 1e-8) -> pd.DataFrame:
    """Calculate agreement with the recorded expert action for every prediction."""
    output = data.copy()
    expert = output[EXPERT_COLUMNS].to_numpy(dtype=float)
    predicted = output[PRED_COLUMNS].to_numpy(dtype=float)
    delta = predicted - expert
    absolute = np.abs(delta)
    for index, action_name in enumerate(ACTION_NAMES):
        output[f"abs_err_{action_name}"] = absolute[:, index]
    output["translation_error_l2"] = np.linalg.norm(delta[:, :3], axis=1)
    output["rotation_error_l2"] = np.linalg.norm(delta[:, 3:6], axis=1)
    output["gripper_absolute_error"] = absolute[:, 6]
    output["overall_action_mae"] = absolute.mean(axis=1)

    expert_translation = expert[:, :3]
    predicted_translation = predicted[:, :3]
    expert_norm = np.linalg.norm(expert_translation, axis=1)
    predicted_norm = np.linalg.norm(predicted_translation, axis=1)
    valid = (expert_norm >= cosine_epsilon) & (predicted_norm >= cosine_epsilon)
    cosine = np.full(len(output), np.nan, dtype=float)
    cosine[valid] = np.sum(expert_translation[valid] * predicted_translation[valid], axis=1) / (
        expert_norm[valid] * predicted_norm[valid]
    )
    cosine[valid] = np.clip(cosine[valid], -1.0, 1.0)
    output["translation_cosine_similarity"] = cosine
    output["translation_cosine_defined"] = valid
    return output


def compute_language_effects(data: pd.DataFrame) -> pd.DataFrame:
    """Pair each alternative condition with Correct inside the same observation."""
    indexed = data.set_index(OBSERVATION_KEY + ["language_condition"], verify_integrity=True)
    rows: list[dict[str, object]] = []
    for observation_key in data[OBSERVATION_KEY].drop_duplicates().itertuples(index=False, name=None):
        correct = indexed.loc[(*observation_key, "correct")]
        correct_action = correct[PRED_COLUMNS].to_numpy(dtype=float)
        for comparison in COMPARISON_ORDER:
            alternative = indexed.loc[(*observation_key, comparison)]
            alternative_action = alternative[PRED_COLUMNS].to_numpy(dtype=float)
            difference = alternative_action - correct_action
            row: dict[str, object] = {
                "episode_id": int(observation_key[0]),
                "frame_index": int(observation_key[1]),
                "global_index": int(observation_key[2]),
                "task_progress": float(correct["task_progress"]),
                "comparison_condition": comparison,
                "comparison_label": CONDITION_LABELS[comparison],
                "noise_seed": int(correct["noise_seed"]),
                "noise_id": str(correct["noise_id"]),
                "translation_language_distance": float(np.linalg.norm(difference[:3])),
                "rotation_language_distance": float(np.linalg.norm(difference[3:6])),
                "gripper_language_difference": float(abs(difference[6])),
                "full_action_language_distance": float(np.linalg.norm(difference)),
            }
            for index, action_name in enumerate(ACTION_NAMES):
                row[f"signed_diff_{action_name}"] = float(difference[index])
            rows.append(row)
    effects = pd.DataFrame(rows)
    if len(effects) != 600:
        raise RuntimeError(f"Expected 600 paired language-effect rows, got {len(effects)}")
    return effects


def descriptive_statistics(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: np.nan for key in ("count", "mean", "median", "std", "q25", "q75")}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else np.nan,
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def build_summary(expert_agreement: pd.DataFrame, language_effects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in CONDITION_ORDER:
        subset = expert_agreement[expert_agreement["language_condition"] == condition]
        for metric in EXPERT_AGREEMENT_METRICS:
            rows.append(
                {
                    "analysis_family": "agreement_with_recorded_expert_action",
                    "condition": CONDITION_LABELS[condition],
                    "metric": metric,
                    **descriptive_statistics(subset[metric]),
                }
            )
    for condition in COMPARISON_ORDER:
        subset = language_effects[language_effects["comparison_condition"] == condition]
        for metric in LANGUAGE_EFFECT_METRICS:
            rows.append(
                {
                    "analysis_family": "paired_language_effect_from_correct",
                    "condition": CONDITION_LABELS[condition],
                    "metric": metric,
                    **descriptive_statistics(subset[metric]),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_mean(
    data: pd.DataFrame,
    value_column: str,
    episode_column: str = "episode_id",
    replicates: int = 2000,
    seed: int = 20260809,
) -> dict[str, float | int]:
    """Bootstrap episodes as clusters, retaining every selected row per sampled episode."""
    episodes = np.asarray(sorted(data[episode_column].unique()), dtype=int)
    if episodes.size < 2:
        raise ValueError("Cluster bootstrap requires at least two episodes")
    values_by_episode = {
        episode: data.loc[data[episode_column] == episode, value_column].to_numpy(dtype=float)
        for episode in episodes
    }
    if any(values.size == 0 for values in values_by_episode.values()):
        raise ValueError("A bootstrap episode cluster has no values")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(episodes, size=episodes.size, replace=True)
        combined = np.concatenate([values_by_episode[int(episode)] for episode in sampled])
        estimates[index] = np.mean(combined)
    return {
        "point_estimate": float(data[value_column].mean()),
        "bootstrap_mean": float(estimates.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "episode_cluster_count": int(episodes.size),
    }


def add_task_stage(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    progress = output["task_progress"].to_numpy(dtype=float)
    if np.any((progress < 0) | (progress > 1)):
        raise ValueError("Cannot bin task_progress outside [0, 1]")
    stage_index = np.minimum(np.floor(progress * 5).astype(int), 4)
    output["task_stage_index"] = stage_index
    output["task_stage"] = pd.Categorical(
        [STAGE_LABELS[index] for index in stage_index], categories=STAGE_LABELS, ordered=True
    )
    return output
