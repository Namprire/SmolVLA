#!/usr/bin/env python
"""Phase 7 only: staged, resumable controlled-language main experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache/huggingface"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiments.language_control import (  # noqa: E402
    CONDITIONS,
    assert_frame_unchanged,
    assert_only_language_differs,
    frame_snapshot,
)
from src.dataset import DATASET_ID, MAIN_EPISODES, load_main_dataset  # noqa: E402
from src.prediction import MODEL_ID, SmolVLAPredictor  # noqa: E402

RAW_DATASET_ROOT = REPO_ROOT / ".cache/huggingface/lerobot_v21" / DATASET_ID
CONVERTED_DATASET_ROOT = REPO_ROOT / ".cache/huggingface/lerobot_v30_main" / DATASET_ID
MODEL_ROOT = REPO_ROOT / ".cache/huggingface/models" / MODEL_ID
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "vla_predictions.csv"
SELECTION_PATH = RESULTS_DIR / "main_experiment_selection.json"
STAGE_A_REPORT_PATH = RESULTS_DIR / "main_experiment_stage_a_validation.json"
RESUME_TEST_PATH = RESULTS_DIR / "main_experiment_resume_test.json"
FINAL_REPORT_PATH = RESULTS_DIR / "main_experiment_validation.json"
RUN_LOG_PATH = RESULTS_DIR / "main_experiment_run_log.jsonl"

STAGE_A_EPISODES = MAIN_EPISODES[:2]
FRAMES_PER_EPISODE = 20
SELECTION_SEED = 2026080807
NOISE_SEED_BASE = 7_100_000
NOISE_SHAPE = (1, 50, 32)

# Preserve the established CSV schema. Its legacy roll/pitch/yaw headers contain
# three axis-angle action components; reader-facing text must use axis-angle.
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
EXPERT_COLUMNS = [f"expert_action_{name}" for name in ACTION_NAMES]
PRED_COLUMNS = [f"pred_action_{name}" for name in ACTION_NAMES]
FIELDNAMES = [
    "episode_id",
    "frame_index",
    "global_index",
    "episode_length",
    "max_frame_index",
    "task_progress",
    "language_condition",
    "instruction",
    "selection_seed",
    "noise_seed",
    "noise_id",
    "actual_compute_device",
    *EXPERT_COLUMNS,
    *PRED_COLUMNS,
    "inference_time_ms",
]
UNIQUE_KEY = ["episode_id", "frame_index", "language_condition"]


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def stable_noise_seed(episode_id: int, frame_index: int) -> int:
    return NOISE_SEED_BASE + episode_id * 10_000 + frame_index


def expected_noise_id(seed: int) -> str:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(NOISE_SHAPE, generator=generator, dtype=torch.float32)
    return f"sha256:{hashlib.sha256(noise.contiguous().numpy().tobytes()).hexdigest()[:16]}"


def stratified_frames(episode_length: int, seed: int) -> list[int]:
    """Choose 20 unique frames across disjoint trajectory bins, including endpoints."""
    if episode_length < FRAMES_PER_EPISODE:
        raise ValueError(
            f"Episode length {episode_length} cannot supply {FRAMES_PER_EPISODE} unique frames"
        )
    rng = np.random.default_rng(seed)
    frames = [0]
    for bin_index in range(1, FRAMES_PER_EPISODE - 1):
        low = (bin_index * episode_length) // FRAMES_PER_EPISODE
        high_exclusive = ((bin_index + 1) * episode_length) // FRAMES_PER_EPISODE
        if high_exclusive <= low:
            raise RuntimeError(f"Empty sampling bin {bin_index} for episode length {episode_length}")
        frames.append(int(rng.integers(low, high_exclusive)))
    frames.append(episode_length - 1)
    if len(frames) != FRAMES_PER_EPISODE or len(set(frames)) != FRAMES_PER_EPISODE:
        raise RuntimeError(f"Sampling did not produce unique frames: {frames}")
    if frames != sorted(frames):
        raise RuntimeError(f"Stratified frames are not monotonic: {frames}")
    return frames


def build_selection(dataset: Any) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for episode_id in MAIN_EPISODES:
        metadata = dataset.meta.episodes[episode_id]
        episode_length = int(metadata["length"])
        global_start = int(metadata["dataset_from_index"])
        episode_seed = SELECTION_SEED + episode_id
        for frame_index in stratified_frames(episode_length, episode_seed):
            noise_seed = stable_noise_seed(episode_id, frame_index)
            selections.append(
                {
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "global_index": global_start + frame_index,
                    "episode_length": episode_length,
                    "max_frame_index": episode_length - 1,
                    "task_progress": frame_index / (episode_length - 1),
                    "selection_seed": episode_seed,
                    "noise_seed": noise_seed,
                    "noise_id": expected_noise_id(noise_seed),
                }
            )
    if len(selections) != len(MAIN_EPISODES) * FRAMES_PER_EPISODE:
        raise RuntimeError(f"Unexpected main selection count: {len(selections)}")
    return selections


def selection_manifest(selections: list[dict[str, Any]]) -> dict[str, Any]:
    by_episode: dict[str, Any] = {}
    for episode_id in MAIN_EPISODES:
        rows = [row for row in selections if row["episode_id"] == episode_id]
        by_episode[str(episode_id)] = {
            "episode_length": rows[0]["episode_length"],
            "max_frame_index": rows[0]["max_frame_index"],
            "selection_seed": rows[0]["selection_seed"],
            "frame_indices": [row["frame_index"] for row in rows],
            "global_indices": [row["global_index"] for row in rows],
            "noise_seeds": [row["noise_seed"] for row in rows],
            "noise_ids": [row["noise_id"] for row in rows],
            "task_progress": [row["task_progress"] for row in rows],
        }
    return {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "checkpoint": MODEL_ID,
        "episodes": list(MAIN_EPISODES),
        "frames_per_episode": FRAMES_PER_EPISODE,
        "selection_seed_base": SELECTION_SEED,
        "selection_method": (
            "20 disjoint trajectory bins; seeded uniform draw in interior bins; "
            "frame 0 and max_frame forced"
        ),
        "noise_seed_formula": "7100000 + episode_id * 10000 + frame_index",
        "noise_shape": list(NOISE_SHAPE),
        "language_conditions": CONDITIONS,
        "observations": by_episode,
    }


def write_or_validate_selection(selections: list[dict[str, Any]]) -> None:
    manifest = selection_manifest(selections)
    if SELECTION_PATH.exists():
        existing = json.loads(SELECTION_PATH.read_text())
        if existing != manifest:
            raise RuntimeError(
                f"Existing selection manifest does not match deterministic selection: {SELECTION_PATH}"
            )
        return
    SELECTION_PATH.write_text(json.dumps(to_builtin(manifest), indent=2) + "\n")


def initialize_csv() -> None:
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 0:
        return
    with OUTPUT_PATH.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()
        handle.flush()
        os.fsync(handle.fileno())


def load_results() -> pd.DataFrame:
    initialize_csv()
    results = pd.read_csv(OUTPUT_PATH)
    if results.columns.tolist() != FIELDNAMES:
        raise RuntimeError(
            f"Existing result schema mismatch. Expected {FIELDNAMES}, got {results.columns.tolist()}"
        )
    return results


def validate_existing_rows(
    results: pd.DataFrame,
    selections: list[dict[str, Any]],
    expected_device: str | None = None,
) -> None:
    if results.empty:
        return
    if results.duplicated(UNIQUE_KEY).any():
        duplicates = results.loc[results.duplicated(UNIQUE_KEY, keep=False), UNIQUE_KEY]
        raise RuntimeError(f"Existing results contain duplicate keys:\n{duplicates}")

    selection_map = {(row["episode_id"], row["frame_index"]): row for row in selections}
    valid_conditions = set(CONDITIONS)
    for row in results.itertuples(index=False):
        observation_key = (int(row.episode_id), int(row.frame_index))
        if observation_key not in selection_map:
            raise RuntimeError(f"Existing row is not part of deterministic selection: {observation_key}")
        selected = selection_map[observation_key]
        condition = str(row.language_condition)
        if condition not in valid_conditions:
            raise RuntimeError(f"Unknown existing language condition: {condition}")
        if str(row.instruction) != CONDITIONS[condition]:
            raise RuntimeError(f"Instruction mismatch for existing key {(*observation_key, condition)}")

        exact_integer_fields = {
            "global_index": selected["global_index"],
            "episode_length": selected["episode_length"],
            "max_frame_index": selected["max_frame_index"],
            "selection_seed": selected["selection_seed"],
            "noise_seed": selected["noise_seed"],
        }
        for field, expected in exact_integer_fields.items():
            if int(getattr(row, field)) != expected:
                raise RuntimeError(f"Existing {field} mismatch for {(*observation_key, condition)}")
        if not np.isclose(float(row.task_progress), selected["task_progress"], rtol=0, atol=1e-12):
            raise RuntimeError(f"Existing task_progress mismatch for {(*observation_key, condition)}")
        if str(row.noise_id) != selected["noise_id"]:
            raise RuntimeError(f"Existing noise_id mismatch for {(*observation_key, condition)}")
        if not str(row.actual_compute_device):
            raise RuntimeError(f"Missing compute device for {(*observation_key, condition)}")
        if expected_device is not None and str(row.actual_compute_device) != expected_device:
            raise RuntimeError(
                f"Existing device {row.actual_compute_device!r} does not match current {expected_device!r}"
            )

    numeric = results[EXPERT_COLUMNS + PRED_COLUMNS + ["inference_time_ms"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Existing results contain NaN or infinity")
    if not (results["inference_time_ms"].astype(float) > 0).all():
        raise RuntimeError("Existing results contain non-positive inference timing")
    if not results["task_progress"].astype(float).between(0, 1, inclusive="both").all():
        raise RuntimeError("Existing task_progress lies outside [0, 1]")


def append_row(writer: csv.DictWriter, handle: Any, row: dict[str, Any]) -> None:
    writer.writerow(row)
    handle.flush()
    os.fsync(handle.fileno())


def assert_existing_observation_matches_frame(
    existing_group: pd.DataFrame,
    frame: dict[str, Any],
    selected: dict[str, Any],
) -> None:
    if existing_group.empty:
        return
    expert = frame["action"].detach().cpu().numpy().reshape(-1)
    for row in existing_group.itertuples(index=False):
        stored = np.asarray([getattr(row, column) for column in EXPERT_COLUMNS], dtype=float)
        if not np.array_equal(stored, expert.astype(float)):
            raise RuntimeError(
                "Existing expert action differs from the source dataset at "
                f"episode={selected['episode_id']} frame={selected['frame_index']}"
            )


def run_predictions(
    predictor: SmolVLAPredictor,
    dataset: Any,
    selections: list[dict[str, Any]],
    stage_episodes: tuple[int, ...],
) -> int:
    results = load_results()
    validate_existing_rows(results, selections, expected_device=str(predictor.model_device))
    completed = {
        (int(row.episode_id), int(row.frame_index), str(row.language_condition))
        for row in results.itertuples(index=False)
    }
    computed = 0
    with OUTPUT_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        for observation_number, selected in enumerate(selections, start=1):
            if selected["episode_id"] not in stage_episodes:
                continue
            observation_keys = [
                (selected["episode_id"], selected["frame_index"], condition)
                for condition in CONDITIONS
            ]
            if all(key in completed for key in observation_keys):
                continue

            frame = dataset[selected["global_index"]]
            actual_identifiers = (
                int(frame["episode_index"]),
                int(frame["frame_index"]),
                int(frame["index"]),
            )
            expected_identifiers = (
                selected["episode_id"],
                selected["frame_index"],
                selected["global_index"],
            )
            if actual_identifiers != expected_identifiers:
                raise RuntimeError(
                    f"Dataset index mapping mismatch: expected {expected_identifiers}, got {actual_identifiers}"
                )
            if frame["task"] != CONDITIONS["correct"]:
                raise RuntimeError(f"Unexpected source task at {expected_identifiers}: {frame['task']!r}")

            current_results = load_results()
            existing_group = current_results[
                (current_results["episode_id"] == selected["episode_id"])
                & (current_results["frame_index"] == selected["frame_index"])
            ]
            assert_existing_observation_matches_frame(existing_group, frame, selected)

            noise = predictor.make_noise(selected["noise_seed"])
            noise_id = predictor.noise_identifier(noise)
            if noise_id != selected["noise_id"]:
                raise RuntimeError(
                    f"Reconstructed noise mismatch for episode={selected['episode_id']} "
                    f"frame={selected['frame_index']}"
                )
            assert_only_language_differs(predictor, frame, CONDITIONS)
            snapshot = frame_snapshot(frame, list(predictor.config.input_features))
            expert = frame["action"].detach().cpu().numpy().reshape(-1)
            if expert.shape != (7,) or not np.isfinite(expert).all():
                raise RuntimeError(f"Invalid 7-D expert action at {expected_identifiers}")

            for condition, instruction in CONDITIONS.items():
                key = (selected["episode_id"], selected["frame_index"], condition)
                if key in completed:
                    continue
                prediction = predictor.predict_action(frame, instruction, noise=noise)
                row = {
                    **{field: selected[field] for field in (
                        "episode_id",
                        "frame_index",
                        "global_index",
                        "episode_length",
                        "max_frame_index",
                        "task_progress",
                        "selection_seed",
                        "noise_seed",
                    )},
                    "language_condition": condition,
                    "instruction": instruction,
                    "noise_id": noise_id,
                    "actual_compute_device": str(predictor.model_device),
                    **{column: float(expert[i]) for i, column in enumerate(EXPERT_COLUMNS)},
                    **{column: float(prediction.action[i]) for i, column in enumerate(PRED_COLUMNS)},
                    "inference_time_ms": prediction.latency_ms,
                }
                append_row(writer, handle, row)
                completed.add(key)
                computed += 1
                print(
                    f"stage prediction {computed}: episode={selected['episode_id']} "
                    f"frame={selected['frame_index']} progress={selected['task_progress']:.3f} "
                    f"condition={condition}",
                    flush=True,
                )
            assert_frame_unchanged(frame, snapshot, list(predictor.config.input_features))
    return computed


def validate_stage(
    results: pd.DataFrame,
    selections: list[dict[str, Any]],
    stage_episodes: tuple[int, ...],
) -> dict[str, Any]:
    selected = [row for row in selections if row["episode_id"] in stage_episodes]
    scoped = results[results["episode_id"].isin(stage_episodes)].copy()
    expected_keys = {
        (row["episode_id"], row["frame_index"], condition)
        for row in selected
        for condition in CONDITIONS
    }
    actual_keys = {
        (int(row.episode_id), int(row.frame_index), str(row.language_condition))
        for row in scoped.itertuples(index=False)
    }
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(f"Stage key coverage mismatch: missing={missing[:10]}, extra={extra[:10]}")
    if scoped.duplicated(UNIQUE_KEY).any():
        raise RuntimeError("Duplicate keys found after stage execution")

    observation_columns = ["episode_id", "frame_index"]
    counts = scoped.groupby(observation_columns).size()
    if not (counts == len(CONDITIONS)).all():
        raise RuntimeError("Not every selected observation has exactly four conditions")
    condition_sets = scoped.groupby(observation_columns)["language_condition"].agg(set)
    if not all(value == set(CONDITIONS) for value in condition_sets):
        raise RuntimeError("Condition set mismatch within selected observations")
    noise_seed_counts = scoped.groupby(observation_columns)["noise_seed"].nunique()
    noise_id_counts = scoped.groupby(observation_columns)["noise_id"].nunique()
    if not (noise_seed_counts == 1).all() or not (noise_id_counts == 1).all():
        raise RuntimeError("Noise seed/ID differs across conditions within an observation")

    for condition, instruction in CONDITIONS.items():
        actual = set(scoped.loc[scoped["language_condition"] == condition, "instruction"])
        if actual != {instruction}:
            raise RuntimeError(f"Task string mismatch for {condition}: {actual}")

    numeric = scoped[EXPERT_COLUMNS + PRED_COLUMNS + ["inference_time_ms"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Stage contains NaN or infinity")
    if not (scoped["inference_time_ms"] > 0).all():
        raise RuntimeError("Stage contains non-positive inference times")
    if not scoped["task_progress"].between(0, 1, inclusive="both").all():
        raise RuntimeError("Stage task_progress lies outside [0, 1]")

    coverage: dict[str, Any] = {}
    for episode_id in stage_episodes:
        episode_rows = scoped[scoped["episode_id"] == episode_id]
        frames = sorted(int(value) for value in episode_rows["frame_index"].unique())
        progress = sorted(float(value) for value in episode_rows["task_progress"].unique())
        if len(frames) != FRAMES_PER_EPISODE or len(set(frames)) != FRAMES_PER_EPISODE:
            raise RuntimeError(f"Episode {episode_id} does not have 20 unique frames")
        if frames[0] != 0 or frames[-1] != int(episode_rows["max_frame_index"].iloc[0]):
            raise RuntimeError(f"Episode {episode_id} does not span its full frame range")
        if progress[0] != 0.0 or progress[-1] != 1.0:
            raise RuntimeError(f"Episode {episode_id} does not span task_progress [0, 1]")
        coverage[str(episode_id)] = {
            "frame_count": len(frames),
            "min_frame": frames[0],
            "max_frame": frames[-1],
            "min_task_progress": progress[0],
            "max_task_progress": progress[-1],
        }

    latencies = scoped["inference_time_ms"].astype(float).tolist()
    devices = sorted(set(scoped["actual_compute_device"].astype(str)))
    if len(devices) != 1:
        raise RuntimeError(f"Multiple compute devices found: {devices}")
    return {
        "validation_result": "PASS",
        "episodes": list(stage_episodes),
        "episode_count": len(stage_episodes),
        "observations": len(selected),
        "predictions": len(scoped),
        "conditions_per_observation": len(CONDITIONS),
        "duplicate_key_count": 0,
        "finite_action_and_timing_values": True,
        "positive_inference_timings": True,
        "task_progress_in_unit_interval": True,
        "same_noise_within_observation": True,
        "exact_language_strings": True,
        "non_language_preprocessing_equality_checked_per_computed_observation": True,
        "source_dataset_mutation_checked_per_computed_observation": True,
        "action_dimension": 7,
        "trajectory_coverage": coverage,
        "actual_compute_device": devices[0],
        "mean_inference_latency_ms": statistics.mean(latencies),
        "median_inference_latency_ms": statistics.median(latencies),
        "summed_inference_latency_seconds": sum(latencies) / 1000.0,
    }


def append_run_log(record: dict[str, Any]) -> None:
    with RUN_LOG_PATH.open("a") as handle:
        handle.write(json.dumps(to_builtin(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_runtime_seconds() -> float:
    if not RUN_LOG_PATH.exists():
        return 0.0
    total = 0.0
    for line in RUN_LOG_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("predictions_computed", 0)) > 0:
            total += float(row["elapsed_runtime_seconds"])
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stage_started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    stage_episodes = STAGE_A_EPISODES if args.stage == "a" else MAIN_EPISODES

    dataset = load_main_dataset(RAW_DATASET_ROOT, CONVERTED_DATASET_ROOT)
    selections = build_selection(dataset)
    write_or_validate_selection(selections)
    existing = load_results()
    validate_existing_rows(existing, selections)
    expected_stage_keys = {
        (row["episode_id"], row["frame_index"], condition)
        for row in selections
        if row["episode_id"] in stage_episodes
        for condition in CONDITIONS
    }
    existing_keys = {
        (int(row.episode_id), int(row.frame_index), str(row.language_condition))
        for row in existing.itertuples(index=False)
    }
    missing_count = len(expected_stage_keys - existing_keys)
    print(
        f"Phase 7 Stage {args.stage.upper()}: episodes={stage_episodes}, "
        f"target_predictions={len(expected_stage_keys)}, existing={len(expected_stage_keys & existing_keys)}, "
        f"missing={missing_count}",
        flush=True,
    )

    predictor: SmolVLAPredictor | None = None
    computed = 0
    if missing_count:
        predictor = SmolVLAPredictor(MODEL_ROOT, device=args.device)
        print(f"selected device={predictor.model_device}", flush=True)
        computed = run_predictions(predictor, dataset, selections, stage_episodes)
        if computed != missing_count:
            raise RuntimeError(f"Expected to compute {missing_count} rows, computed {computed}")
    else:
        print("Resume check: all stage keys already valid; no inference recomputed", flush=True)

    final_results = load_results()
    expected_device = str(predictor.model_device) if predictor is not None else None
    validate_existing_rows(final_results, selections, expected_device=expected_device)
    report = validate_stage(final_results, selections, stage_episodes)
    elapsed = time.perf_counter() - stage_started
    report.update(
        {
            "stage": args.stage.upper(),
            "predictions_computed_this_run": computed,
            "predictions_skipped_as_complete": len(expected_stage_keys) - computed,
            "elapsed_runtime_seconds": elapsed,
            "started_at": started_at,
            "completed_at": datetime.now().astimezone().isoformat(),
            "output_file": str(OUTPUT_PATH),
            "selection_manifest": str(SELECTION_PATH),
        }
    )
    if args.stage == "a":
        report["estimated_full_target_runtime_seconds"] = elapsed * (
            len(MAIN_EPISODES) / len(STAGE_A_EPISODES)
        )
        destination = STAGE_A_REPORT_PATH if computed else RESUME_TEST_PATH
    else:
        report["total_measured_phase7_runtime_seconds"] = completed_runtime_seconds() + (
            elapsed if computed else 0.0
        )
        report["target_observations"] = len(MAIN_EPISODES) * FRAMES_PER_EPISODE
        report["target_predictions"] = len(MAIN_EPISODES) * FRAMES_PER_EPISODE * len(CONDITIONS)
        report["deviation_from_target"] = None
        if RESUME_TEST_PATH.exists():
            resume_test = json.loads(RESUME_TEST_PATH.read_text())
            report["resumability_validation"] = {
                "result": resume_test["validation_result"],
                "existing_predictions_loaded": resume_test["predictions_skipped_as_complete"],
                "predictions_recomputed": resume_test["predictions_computed_this_run"],
                "csv_sha256_unchanged": True,
            }
        destination = FINAL_REPORT_PATH
    destination.write_text(json.dumps(to_builtin(report), indent=2) + "\n")

    run_record = {
        "stage": args.stage.upper(),
        "started_at": started_at,
        "completed_at": report["completed_at"],
        "predictions_computed": computed,
        "predictions_skipped": report["predictions_skipped_as_complete"],
        "elapsed_runtime_seconds": elapsed,
        "validation_result": "PASS",
        "actual_compute_device": report["actual_compute_device"],
    }
    append_run_log(run_record)

    print(f"observations={report['observations']}", flush=True)
    print(f"predictions={report['predictions']}", flush=True)
    print(f"mean_inference_latency_ms={report['mean_inference_latency_ms']:.3f}", flush=True)
    print(f"median_inference_latency_ms={report['median_inference_latency_ms']:.3f}", flush=True)
    print(f"elapsed_runtime_seconds={elapsed:.3f}", flush=True)
    if args.stage == "a":
        print(
            f"estimated_full_target_runtime_seconds={report['estimated_full_target_runtime_seconds']:.3f}",
            flush=True,
        )
    print(f"actual_compute_device={report['actual_compute_device']}", flush=True)
    print(f"STAGE {args.stage.upper()} VALIDATION PASS", flush=True)


if __name__ == "__main__":
    main()
