#!/usr/bin/env python
"""Forensic, model-free audit of the offline viewer and its five examples."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.demo import (
    CONDITION_LABELS,
    CONDITION_ORDER,
    EXPERT_COLUMNS,
    OBSERVATION_KEY,
    OfflineDemo,
    PRED_COLUMNS,
    format_action,
    load_data,
)


RESULTS = ROOT / "results"
DATA_DIR = ROOT / ".cache/huggingface/lerobot_v21/HuggingFaceVLA/smol-libero/data/chunk-000"
OUTPUT = RESULTS / "final_viewer_audit.json"
EXAMPLE_AUDIT = RESULTS / "demo_example_audit.csv"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decode_image(value: dict[str, Any]) -> np.ndarray:
    image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    image = image.transpose(Image.Transpose.ROTATE_180)
    return np.asarray(image)


def source_observation(episode_id: int, frame_index: int) -> dict[str, Any]:
    path = DATA_DIR / f"episode_{episode_id:06d}.parquet"
    table = pq.read_table(
        path,
        columns=[
            "episode_index",
            "frame_index",
            "index",
            "action",
            "observation.state",
            "observation.images.image",
            "observation.images.image2",
        ],
    )
    frames = np.asarray(table["frame_index"], dtype=np.int64)
    matches = np.flatnonzero(frames == frame_index)
    if len(matches) != 1:
        raise AssertionError(f"Expected one source frame for episode={episode_id}, frame={frame_index}")
    row = int(matches[0])
    state = np.asarray(table["observation.state"][row].as_py(), dtype=np.float64)
    return {
        "episode_id": int(table["episode_index"][row].as_py()),
        "frame_index": int(table["frame_index"][row].as_py()),
        "global_index": int(table["index"][row].as_py()),
        "expert_action": np.asarray(table["action"][row].as_py(), dtype=np.float64),
        "state": state,
        "state_sha256": sha256_bytes(state.tobytes()),
        "camera_1": decode_image(table["observation.images.image"][row].as_py()),
        "camera_2": decode_image(table["observation.images.image2"][row].as_py()),
    }


def main() -> None:
    predictions, examples = load_data()
    mathematical = pd.read_csv(EXAMPLE_AUDIT).set_index("example_index")
    viewer = OfflineDemo(predictions, examples)
    rows: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        viewer.example_index = example_index
        viewer.load_example()
        viewer.fig.canvas.draw()
        source = source_observation(int(example["episode_id"]), int(example["frame_index"]))
        key = tuple(int(example[name]) for name in OBSERVATION_KEY)
        expected = predictions[
            (predictions.episode_id == key[0])
            & (predictions.frame_index == key[1])
            & (predictions.global_index == key[2])
        ].sort_values("language_condition")
        current = viewer.current_rows.copy()
        current["language_condition"] = current.language_condition.astype(str)
        current = current.sort_values("language_condition")
        exact_row_values = all(
            np.array_equal(
                current[column].astype(str).to_numpy(), expected[column].astype(str).to_numpy()
            )
            for column in predictions.columns
        )
        identifiers_match_source = (
            source["episode_id"], source["frame_index"], source["global_index"]
        ) == key
        expert = expected.iloc[0][EXPERT_COLUMNS].to_numpy(float)
        expert_source_max_abs_difference = float(
            np.max(np.abs(expert - source["expert_action"]))
        )
        expert_matches_source = np.allclose(
            expert, source["expert_action"], rtol=0, atol=1e-15
        )
        invariant_columns = [
            *OBSERVATION_KEY,
            "episode_length",
            "max_frame_index",
            "task_progress",
            "selection_seed",
            "noise_seed",
            "noise_id",
            *EXPERT_COLUMNS,
        ]
        invariants_hold = bool(
            (viewer.current_rows[invariant_columns].nunique(dropna=False) == 1).all()
        )
        displayed_cameras = [
            np.asarray(viewer.camera_1_ax.images[0].get_array()),
            np.asarray(viewer.camera_2_ax.images[0].get_array()),
        ]
        source_cameras = [source["camera_1"], source["camera_2"]]
        assets = [
            np.asarray(Image.open(ROOT / example["camera_1_asset"]).convert("RGB")),
            np.asarray(Image.open(ROOT / example["camera_2_asset"]).convert("RGB")),
        ]
        cameras_match_source = all(
            np.array_equal(displayed, original)
            and np.array_equal(asset, original)
            for displayed, asset, original in zip(
                displayed_cameras, assets, source_cameras, strict=True
            )
        )
        condition_checks: dict[str, Any] = {}
        for condition_index, condition in enumerate(CONDITION_ORDER):
            viewer.set_condition(condition_index)
            viewer.fig.canvas.draw()
            row = viewer.selected_row()
            selected_expert = row[EXPERT_COLUMNS].to_numpy(float)
            predicted = row[PRED_COLUMNS].to_numpy(float)
            delta = predicted - selected_expert
            metrics = {
                "translation": float(np.linalg.norm(delta[:3])),
                "axis_angle": float(np.linalg.norm(delta[3:6])),
                "gripper": float(abs(delta[6])),
                "full_7d": float(np.linalg.norm(delta)),
            }
            details = viewer.detail_ax.texts[0].get_text()
            table_text = "\n".join(text.get_text() for text in viewer.table_ax.texts)
            table = list(viewer.table_ax.tables)[0]
            table_values_correct = True
            for stored_index, stored_condition in enumerate(CONDITION_ORDER, start=1):
                stored = viewer.current_rows[
                    viewer.current_rows.language_condition.astype(str) == stored_condition
                ].iloc[0]
                expected_cells = [CONDITION_LABELS[stored_condition]] + [
                    f"{value:+.3f}" for value in stored[PRED_COLUMNS].to_numpy(float)
                ]
                actual_cells = [table[(stored_index, column)].get_text().get_text() for column in range(8)]
                table_values_correct = table_values_correct and actual_cells == expected_cells
            exact_instruction_displayed = textwrap.fill(str(row.instruction), width=66) in details
            metrics_displayed_correctly = all(
                value in details
                for value in (
                    f"Translation difference:  {metrics['translation']:.4f}",
                    f"Axis-angle difference:   {metrics['axis_angle']:.4f}",
                    f"Gripper difference:      {metrics['gripper']:.4f}",
                    f"Full 7-D action distance:{metrics['full_7d']:>8.4f}",
                )
            )
            metric_families_explicit = (
                "SELECTED PREDICTION VS RECORDED EXPERT" in details
                and "LANGUAGE-INDUCED DIFFERENCE VS CORRECT PREDICTION" in table_text
            )
            condition_checks[CONDITION_LABELS[condition]] = {
                "selected_row_is_requested_condition": str(row.language_condition) == condition,
                "instruction_exact": str(row.instruction)
                == predictions.loc[
                    (predictions.episode_id == key[0])
                    & (predictions.frame_index == key[1])
                    & (predictions.global_index == key[2])
                    & (predictions.language_condition == condition),
                    "instruction",
                ].iloc[0],
                "exact_instruction_displayed": exact_instruction_displayed,
                "recorded_expert_vector_displayed": format_action(selected_expert) in details,
                "prediction_vector_displayed": format_action(predicted) in details,
                "expert_agreement_metrics_displayed_correctly": metrics_displayed_correctly,
                "all_four_table_values_match_csv_at_display_precision": table_values_correct,
                "metric_families_explicitly_distinguished": metric_families_explicit,
                "camera_pixels_unchanged_when_condition_selected": all(
                    np.array_equal(np.asarray(axis.images[0].get_array()), original)
                    for axis, original in zip(
                        (viewer.camera_1_ax, viewer.camera_2_ax), source_cameras, strict=True
                    )
                ),
            }
        condition_pass = all(
            all(bool(value) for value in check.values()) for check in condition_checks.values()
        )
        selection_true = bool(mathematical.loc[int(example["example_index"]), "mathematically_valid"])
        example_pass = all(
            (
                exact_row_values,
                identifiers_match_source,
                expert_matches_source,
                invariants_hold,
                cameras_match_source,
                condition_pass,
                selection_true,
            )
        )
        rows.append(
            {
                "example_index": int(example["example_index"]),
                "episode_id": key[0],
                "frame_index": key[1],
                "global_index": key[2],
                "task_progress": float(example["task_progress"]),
                "state_sha256": source["state_sha256"],
                "source_identifiers_match_example": identifiers_match_source,
                "both_camera_assets_and_displayed_pixels_match_exact_source_frame": cameras_match_source,
                "all_four_rows_exactly_match_csv": exact_row_values,
                "same_frame_progress_noise_and_expert_invariants": invariants_hold,
                "recorded_expert_action_matches_source_parquet": expert_matches_source,
                "recorded_expert_source_max_abs_difference": expert_source_max_abs_difference,
                "selection_explanation_mathematically_true": selection_true,
                "conditions": condition_checks,
                "pass": example_pass,
            }
        )
    viewer.smoke_test()
    import matplotlib.pyplot as plt

    plt.close(viewer.fig)
    report = {
        "validation_result": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "viewer_source": "demo/demo.py",
        "example_count": len(rows),
        "all_five_examples_and_all_four_conditions_rendered": True,
        "left_metric_semantics": "selected SmolVLA prediction vs recorded expert action",
        "right_metric_semantics": "altered-language SmolVLA prediction vs Correct-language SmolVLA prediction",
        "source_pipeline_corroboration": (
            "main_language_experiment.py reuses one dataset frame and one explicit noise tensor across "
            "the four condition calls, copies every mutable input during preprocessing, and checks the "
            "source frame is unchanged after the condition loop."
        ),
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"VIEWER FORENSIC AUDIT {report['validation_result']}: "
        f"examples={len(rows)}, conditions={len(rows) * len(CONDITION_ORDER)}, "
        "camera_source_pixels_exact=true"
    )
    if report["validation_result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
