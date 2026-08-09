#!/usr/bin/env python
"""Phases 3-6: controlled-noise language intervention and small pilot."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache/huggingface"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import (  # noqa: E402
    SmolVLAPolicy,
    VLAFlowMatching,
)

from src.dataset import DATASET_ID, PILOT_EPISODES, load_pilot_dataset  # noqa: E402
from src.prediction import MODEL_ID, SmolVLAPredictor  # noqa: E402

RAW_DATASET_ROOT = REPO_ROOT / ".cache/huggingface/lerobot_v21" / DATASET_ID
CONVERTED_DATASET_ROOT = REPO_ROOT / ".cache/huggingface/lerobot_v30_pilot" / DATASET_ID
MODEL_ROOT = REPO_ROOT / ".cache/huggingface/models" / MODEL_ID
RESULTS_DIR = REPO_ROOT / "results"

CONDITIONS = {
    "correct": "put both the cream cheese box and the butter in the basket",
    "paraphrase": "place the cream cheese box and the butter together inside the basket",
    "contradictory": "put the cream cheese box in the basket but leave the butter outside the basket",
    "unrelated": "open the top drawer of the cabinet",
}

DETERMINISM_SEED = 314159
PILOT_SEED_BASE = 62000


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def frame_snapshot(frame: dict[str, Any], input_keys: list[str]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"task": frame.get("task")}
    for key in input_keys:
        value = frame[key]
        snapshot[key] = value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
    return snapshot


def assert_frame_unchanged(frame: dict[str, Any], snapshot: dict[str, Any], input_keys: list[str]) -> None:
    if frame.get("task") != snapshot["task"]:
        raise RuntimeError("Prediction mutated the dataset frame's task field")
    for key in input_keys:
        before, after = snapshot[key], frame[key]
        if isinstance(before, torch.Tensor):
            if not torch.equal(before, after.detach().cpu()):
                raise RuntimeError(f"Prediction mutated dataset frame feature {key}")
        elif before != after:
            raise RuntimeError(f"Prediction mutated dataset frame feature {key}")


def assert_only_language_differs(
    predictor: SmolVLAPredictor,
    frame: dict[str, Any],
    conditions: dict[str, str],
) -> dict[str, dict[str, list[Any]]]:
    """Prove official preprocessing changes only language-derived fields."""
    processed = {name: predictor._preprocess(frame, text) for name, text in conditions.items()}
    reference = processed["correct"]
    for name, batch in processed.items():
        for key in predictor.config.input_features:
            left, right = reference[key], batch[key]
            if isinstance(left, torch.Tensor):
                if not torch.equal(left, right):
                    raise RuntimeError(
                        f"Unexplained preprocessing difference for {key}: correct vs {name}"
                    )
            elif left != right:
                raise RuntimeError(f"Unexplained preprocessing difference for {key}: correct vs {name}")

    audit: dict[str, dict[str, list[Any]]] = {}
    for name, batch in processed.items():
        audit[name] = {
            "token_ids": batch["observation.language.tokens"].detach().cpu().reshape(-1).tolist(),
            "attention_mask": batch["observation.language.attention_mask"]
            .detach()
            .cpu()
            .reshape(-1)
            .bool()
            .tolist(),
        }
    return audit


def write_api_inspection(predictor: SmolVLAPredictor, token_audit: dict[str, Any]) -> None:
    policy_source = inspect.getsourcefile(SmolVLAPolicy)
    tokenizer_source = inspect.getsourcefile(
        __import__(
            "lerobot.processor.tokenizer_processor", fromlist=["TokenizerProcessorStep"]
        ).TokenizerProcessorStep
    )
    processor_source = inspect.getsourcefile(
        __import__(
            "lerobot.policies.smolvla.processor_smolvla", fromlist=["SmolVLANewLineProcessor"]
        ).SmolVLANewLineProcessor
    )

    def portable_source(source: str | None) -> str:
        if source is None:
            return "unavailable"
        path = Path(source)
        if "site-packages" in path.parts:
            index = path.parts.index("site-packages")
            return str(Path(*path.parts[index:]))
        return path.name

    lines = [
        "# Installed LeRobot SmolVLA API inspection",
        "",
        f"- LeRobot version: `{importlib.metadata.version('lerobot')}`",
        f"- Policy source inspected: `{portable_source(policy_source)}`",
        f"- SmolVLA processor source inspected: `{portable_source(processor_source)}`",
        f"- Tokenizer processor source inspected: `{portable_source(tokenizer_source)}`",
        f"- `SmolVLAPolicy.predict_action_chunk` signature: `{inspect.signature(SmolVLAPolicy.predict_action_chunk)}`",
        f"- `SmolVLAPolicy.select_action` signature: `{inspect.signature(SmolVLAPolicy.select_action)}`",
        f"- `VLAFlowMatching.sample_actions` signature: `{inspect.signature(VLAFlowMatching.sample_actions)}`",
        f"- `VLAFlowMatching.sample_noise` signature: `{inspect.signature(VLAFlowMatching.sample_noise)}`",
        "",
        "## Task-to-token path",
        "",
        "The caller supplies `task` beside the checkpoint input features in the raw batch dictionary. "
        "`batch_to_transition` removes it from observations and places it in `complementary_data`. "
        "The checkpoint's serialized `to_batch_processor` converts the string to a one-element batch. "
        "`SmolVLANewLineProcessor` appends `\\n`. `TokenizerProcessorStep` reads "
        "`complementary_data['task']` and calls the SmolVLM tokenizer with "
        "`max_length=48`, right padding, `padding='longest'`, truncation, and PyTorch output. "
        "It writes `observation.language.tokens` as `torch.int64` and "
        "`observation.language.attention_mask` as `torch.bool`. The device processor then moves "
        "both to the configured inference device.",
        "",
        "The serialized preprocessor and normalization state from the official checkpoint are used "
        "unchanged except for overriding the runtime device from CUDA to the selected host device.",
        "",
        "## Explicit flow noise",
        "",
        "`SmolVLAPolicy.predict_action_chunk(batch, noise=...)` forwards `noise` through "
        "`_get_action_chunk` to `VLAFlowMatching.sample_actions`. When absent, that sampler calls "
        "`sample_noise` with shape `(batch_size, config.chunk_size, config.max_action_dim)` and "
        "creates a normal `torch.float32` tensor on `state.device`. For this checkpoint and one frame, "
        f"the required explicit tensor is `{predictor.noise_shape}`, `torch.float32`, `{predictor.model_device}`. "
        "The 32-D sampler output is unpadded by the policy to the checkpoint's 7-D action before the "
        "official action unnormalizer runs.",
        "",
        "## Runtime token audit",
        "",
    ]
    for name, text in CONDITIONS.items():
        lines.extend(
            [
                f"- `{name}`: `{text}`",
                f"  - token IDs: `{token_audit[name]['token_ids']}`",
                f"  - attention mask: `{token_audit[name]['attention_mask']}`",
            ]
        )
    (RESULTS_DIR / "api_inspection.md").write_text("\n".join(lines) + "\n")


def action_metrics(action: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    delta = action - correct
    return {
        "translation_distance_from_correct": float(np.linalg.norm(delta[:3])),
        "rotation_distance_from_correct": float(np.linalg.norm(delta[3:6])),
        "gripper_difference_from_correct": float(delta[6]),
        "full_action_distance_from_correct": float(np.linalg.norm(delta)),
    }


def run_determinism(
    predictor: SmolVLAPredictor,
    frame: dict[str, Any],
    noise: torch.Tensor,
    repeats: int,
    tolerance: float,
) -> list[np.ndarray]:
    print("\n=== PHASE 4: EXPLICIT-NOISE DETERMINISM ===", flush=True)
    print(f"repeats={repeats}, tolerance={tolerance:.3e}", flush=True)
    results: list[tuple[np.ndarray, float]] = []
    snapshot = frame_snapshot(frame, list(predictor.config.input_features))
    for repeat in range(repeats):
        prediction = predictor.predict_action(frame, CONDITIONS["correct"], noise=noise)
        results.append((prediction.action, prediction.latency_ms))
        print(
            f"repeat {repeat}: {prediction.action.tolist()} latency_ms={prediction.latency_ms:.1f}",
            flush=True,
        )
    assert_frame_unchanged(frame, snapshot, list(predictor.config.input_features))

    baseline = results[0][0]
    rows: list[dict[str, Any]] = []
    for repeat, (action, latency_ms) in enumerate(results):
        difference = action - baseline
        row: dict[str, Any] = {"repeat": repeat, "latency_ms": latency_ms}
        row.update({f"action_{i}": float(action[i]) for i in range(7)})
        row.update({f"difference_from_repeat_0_{i}": float(difference[i]) for i in range(7)})
        row.update({f"abs_difference_from_repeat_0_{i}": float(abs(difference[i])) for i in range(7)})
        row["max_abs_difference_from_repeat_0"] = float(np.max(np.abs(difference)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "determinism_check.csv", index=False)

    stacked = np.stack([item[0] for item in results])
    per_dimension_range = np.ptp(stacked, axis=0)
    max_abs_difference = float(np.max(np.abs(stacked[:, None, :] - stacked[None, :, :])))
    print(f"per-dimension output ranges: {per_dimension_range.tolist()}", flush=True)
    print(f"maximum absolute pairwise difference: {max_abs_difference:.9g}", flush=True)
    if max_abs_difference > tolerance:
        print("DETERMINISM FAIL: stopping before any language experiment", flush=True)
        raise RuntimeError(
            f"Repeated explicit-noise outputs differ by {max_abs_difference}, above {tolerance}"
        )
    print("DETERMINISM PASS: repeated explicit-noise predictions are essentially identical", flush=True)
    return [item[0] for item in results]


def run_single_frame(
    predictor: SmolVLAPredictor,
    frame: dict[str, Any],
    noise: torch.Tensor,
    token_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    print("\n=== PHASE 5: EXACT LANGUAGE STRINGS ===", flush=True)
    for name, instruction in CONDITIONS.items():
        print(f"{name}: {instruction}", flush=True)

    raw_predictions: dict[str, tuple[np.ndarray, float]] = {}
    for name, instruction in CONDITIONS.items():
        prediction = predictor.predict_action(frame, instruction, noise=noise)
        raw_predictions[name] = (prediction.action, prediction.latency_ms)
    correct = raw_predictions["correct"][0]

    records: list[dict[str, Any]] = []
    print("\n=== PHASE 5: FOUR POSTPROCESSED ACTIONS ===", flush=True)
    for name, instruction in CONDITIONS.items():
        action, latency_ms = raw_predictions[name]
        delta = action - correct
        record: dict[str, Any] = {
            "language_condition": name,
            "instruction": instruction,
            **{f"action_{i}": float(action[i]) for i in range(7)},
            **{f"difference_from_correct_{i}": float(delta[i]) for i in range(7)},
            **action_metrics(action, correct),
            "inference_latency_ms": latency_ms,
        }
        records.append(record)
        print(f"{name}: action={action.tolist()}", flush=True)
        print(f"{name}: difference_from_correct={delta.tolist()}", flush=True)

    pd.DataFrame(records).to_csv(RESULTS_DIR / "language_single_frame.csv", index=False)
    payload = {
        "scope": "single controlled observation; no scientific conclusion is claimed",
        "dataset": DATASET_ID,
        "checkpoint": MODEL_ID,
        "episode_id": int(frame["episode_index"]),
        "frame_index": int(frame["frame_index"]),
        "noise_identifier": predictor.noise_identifier(noise),
        "noise_shape": list(noise.shape),
        "noise_dtype": str(noise.dtype),
        "noise_device": str(noise.device),
        "official_tokenization": token_audit,
        "conditions": records,
    }
    (RESULTS_DIR / "language_conditions.json").write_text(
        json.dumps(json_ready(payload), indent=2) + "\n"
    )
    return records


def select_pilot_observations(dataset: Any) -> list[dict[str, Any]]:
    lengths = {episode: int(dataset.meta.episodes[episode]["length"]) for episode in PILOT_EPISODES}
    starts: dict[int, int] = {}
    offset = 0
    for episode in PILOT_EPISODES:
        starts[episode] = offset
        offset += lengths[episode]

    selections: list[dict[str, Any]] = []
    for episode in PILOT_EPISODES:
        length = lengths[episode]
        for label, local_frame in (
            ("early", 0),
            ("middle", int(round((length - 1) * 0.5))),
            ("late", length - 1),
        ):
            frame = dataset[starts[episode] + local_frame]
            actual_episode = int(frame["episode_index"])
            actual_frame = int(frame["frame_index"])
            if (actual_episode, actual_frame) != (episode, local_frame):
                raise RuntimeError(
                    "Dataset index mapping mismatch: "
                    f"expected {(episode, local_frame)}, got {(actual_episode, actual_frame)}"
                )
            selections.append(
                {
                    "episode_id": episode,
                    "frame_index": local_frame,
                    "position_label": label,
                    "task_progress": local_frame / (length - 1),
                    "episode_length": length,
                    "frame": frame,
                }
            )
    return selections


def run_pilot(predictor: SmolVLAPredictor, dataset: Any) -> pd.DataFrame:
    print("\n=== PHASE 6: NINE-OBSERVATION PILOT ===", flush=True)
    selections = select_pilot_observations(dataset)
    output_path = RESULTS_DIR / "vla_predictions_pilot.csv"
    fieldnames = [
        "episode_id",
        "frame_index",
        "position_label",
        "task_progress",
        "episode_length",
        "language_condition",
        "instruction",
        "noise_seed",
        "noise_identifier",
        *[f"expert_action_{i}" for i in range(7)],
        *[f"predicted_action_{i}" for i in range(7)],
        "inference_latency_ms",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()
        for observation_number, selection in enumerate(selections):
            frame = selection["frame"]
            seed = PILOT_SEED_BASE + observation_number
            noise = predictor.make_noise(seed)
            noise_id = predictor.noise_identifier(noise)
            expert = frame["action"].detach().cpu().numpy().reshape(-1)
            if expert.shape != (7,) or not np.isfinite(expert).all():
                raise RuntimeError(f"Invalid expert action at pilot observation {observation_number}")

            # Audit every observation, not only the Phase 5 frame.
            assert_only_language_differs(predictor, frame, CONDITIONS)
            snapshot = frame_snapshot(frame, list(predictor.config.input_features))
            for condition, instruction in CONDITIONS.items():
                prediction = predictor.predict_action(frame, instruction, noise=noise)
                row: dict[str, Any] = {
                    key: selection[key]
                    for key in (
                        "episode_id",
                        "frame_index",
                        "position_label",
                        "task_progress",
                        "episode_length",
                    )
                }
                row.update(
                    {
                        "language_condition": condition,
                        "instruction": instruction,
                        "noise_seed": seed,
                        "noise_identifier": noise_id,
                        **{f"expert_action_{i}": float(expert[i]) for i in range(7)},
                        **{f"predicted_action_{i}": float(prediction.action[i]) for i in range(7)},
                        "inference_latency_ms": prediction.latency_ms,
                    }
                )
                writer.writerow(row)
                handle.flush()
                print(
                    f"pilot episode={selection['episode_id']} frame={selection['frame_index']} "
                    f"progress={selection['task_progress']:.3f} condition={condition} complete",
                    flush=True,
                )
            assert_frame_unchanged(frame, snapshot, list(predictor.config.input_features))

    pilot = pd.read_csv(output_path)
    validate_pilot(pilot, len(selections))
    print(f"PILOT PASS: {len(pilot)} valid rows ({len(selections)} observations x 4 conditions)", flush=True)
    return pilot


def validate_pilot(pilot: pd.DataFrame, observation_count: int) -> None:
    expected_rows = observation_count * len(CONDITIONS)
    if len(pilot) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} pilot rows, got {len(pilot)}")

    expert_columns = [f"expert_action_{i}" for i in range(7)]
    predicted_columns = [f"predicted_action_{i}" for i in range(7)]
    if len(expert_columns) != 7 or len(predicted_columns) != 7:
        raise RuntimeError("Action column schema is not exactly 7-D")
    if not np.isfinite(pilot[expert_columns + predicted_columns].to_numpy()).all():
        raise RuntimeError("Pilot contains non-finite expert or predicted action values")

    identifier_columns = ["episode_id", "frame_index"]
    counts = pilot.groupby(identifier_columns, sort=False).size()
    if not (counts == 4).all():
        raise RuntimeError(f"Pilot does not have four conditions per observation: {counts.to_dict()}")
    condition_sets = pilot.groupby(identifier_columns)["language_condition"].agg(set)
    if not all(value == set(CONDITIONS) for value in condition_sets):
        raise RuntimeError("Pilot condition identifiers differ between observations")

    for condition, instruction in CONDITIONS.items():
        values = set(pilot.loc[pilot["language_condition"] == condition, "instruction"])
        if values != {instruction}:
            raise RuntimeError(f"Incorrect task strings for condition {condition}: {values}")

    progress_expected = pilot["frame_index"] / (pilot["episode_length"] - 1)
    if not np.allclose(pilot["task_progress"], progress_expected, rtol=0, atol=1e-12):
        raise RuntimeError("Incorrect normalized task_progress values")

    duplicate_key = ["episode_id", "frame_index", "language_condition"]
    if pilot.duplicated(duplicate_key).any():
        raise RuntimeError("Pilot contains duplicate records")
    noise_counts = pilot.groupby(identifier_columns)["noise_identifier"].nunique()
    if not (noise_counts == 1).all():
        raise RuntimeError("Language conditions within an observation did not reuse one noise tensor")


def summarize_language_effects(pilot: pd.DataFrame) -> dict[str, dict[str, float]]:
    predicted_columns = [f"predicted_action_{i}" for i in range(7)]
    summaries: dict[str, list[float]] = {name: [] for name in CONDITIONS if name != "correct"}
    for _, group in pilot.groupby(["episode_id", "frame_index"], sort=False):
        correct = group.loc[group["language_condition"] == "correct", predicted_columns].to_numpy()[0]
        for condition in summaries:
            action = group.loc[group["language_condition"] == condition, predicted_columns].to_numpy()[0]
            summaries[condition].append(float(np.linalg.norm(action - correct)))
    return {
        condition: {
            "observation_count": len(distances),
            "mean_full_7d_distance_from_correct": float(np.mean(distances)),
            "median_full_7d_distance_from_correct": float(np.median(distances)),
            "max_full_7d_distance_from_correct": float(np.max(distances)),
            "nonzero_count_at_1e-8": int(np.sum(np.asarray(distances) > 1e-8)),
        }
        for condition, distances in summaries.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--determinism-repeats", type=int, default=5)
    parser.add_argument("--determinism-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.determinism_repeats < 5:
        raise ValueError("Phase 4 requires at least five repeats")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_pilot_dataset(RAW_DATASET_ROOT, CONVERTED_DATASET_ROOT)
    fixed_frame = dataset[0]
    if fixed_frame["task"] != CONDITIONS["correct"]:
        raise RuntimeError(f"Unexpected real dataset task: {fixed_frame['task']!r}")

    print("=== PHASE 3: LOAD CLEAN PREDICTION API ===", flush=True)
    predictor = SmolVLAPredictor(MODEL_ROOT, device=args.device)
    print(f"device={predictor.device}", flush=True)
    print(f"checkpoint dtypes={predictor.checkpoint_dtype_counts}", flush=True)
    print(
        f"explicit noise spec: shape={predictor.noise_shape}, dtype=torch.float32, "
        f"device={predictor.model_device}",
        flush=True,
    )

    token_audit = assert_only_language_differs(predictor, fixed_frame, CONDITIONS)
    write_api_inspection(predictor, token_audit)
    explicit_noise = predictor.make_noise(DETERMINISM_SEED)
    print(
        f"determinism noise seed={DETERMINISM_SEED}, "
        f"id={predictor.noise_identifier(explicit_noise)}",
        flush=True,
    )

    run_determinism(
        predictor,
        fixed_frame,
        explicit_noise,
        repeats=args.determinism_repeats,
        tolerance=args.determinism_tolerance,
    )
    single_frame_records = run_single_frame(predictor, fixed_frame, explicit_noise, token_audit)
    pilot = run_pilot(predictor, dataset)
    summary = summarize_language_effects(pilot)

    summary_payload = {
        "pilot_row_count": len(pilot),
        "observation_count": pilot[["episode_id", "frame_index"]].drop_duplicates().shape[0],
        "condition_counts": dict(Counter(pilot["language_condition"])),
        "single_frame_full_action_distance_from_correct": {
            row["language_condition"]: row["full_action_distance_from_correct"]
            for row in single_frame_records
        },
        "pilot_full_action_distance_summary": summary,
        "scope": "small controlled pilot; no scientific conclusion is claimed",
    }
    (RESULTS_DIR / "language_pilot_summary.json").write_text(
        json.dumps(json_ready(summary_payload), indent=2) + "\n"
    )
    print("\n=== NUMERIC PILOT SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("STOPPING AFTER PHASE 6 AS REQUESTED", flush=True)


if __name__ == "__main__":
    main()
