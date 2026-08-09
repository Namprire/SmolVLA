#!/usr/bin/env python
"""Offline viewer for the completed controlled-language experiment.

The viewer reads stored CSV predictions and packaged camera frames only. It
does not import or load SmolVLA and cannot perform inference.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

if "--smoke-test" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Button  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import (  # noqa: E402
    CONDITION_LABELS,
    CONDITION_ORDER,
    EXPERT_COLUMNS,
    OBSERVATION_KEY,
    PRED_COLUMNS,
    validate_predictions,
)

PREDICTIONS_PATH = REPO_ROOT / "results" / "vla_predictions.csv"
EXAMPLES_PATH = REPO_ROOT / "results" / "demo_examples.json"

ACTION_LABELS = (
    "x",
    "y",
    "z",
    "axis-angle-1",
    "axis-angle-2",
    "axis-angle-3",
    "gripper",
)

COLORS = {
    "correct": "#3B6EA8",
    "paraphrase": "#3A8D5D",
    "contradictory": "#D17A22",
    "unrelated": "#C84B4B",
}


def format_action(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:+.4f}" for value in values) + "]"


def load_data() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    validate_predictions(predictions)
    payload = json.loads(EXAMPLES_PATH.read_text())
    examples = payload.get("examples", [])
    if not 3 <= len(examples) <= 5:
        raise ValueError(f"Expected 3-5 curated examples, found {len(examples)}")
    return predictions, examples


class OfflineDemo:
    def __init__(
        self,
        predictions: pd.DataFrame,
        examples: list[dict[str, Any]],
        example_index: int = 0,
    ) -> None:
        self.predictions = predictions
        self.examples = examples
        self.example_index = example_index % len(examples)
        self.condition_index = 0
        self.current_rows = pd.DataFrame()

        self.fig = plt.figure(figsize=(14.4, 9.0), facecolor="white")
        grid = self.fig.add_gridspec(
            2,
            2,
            height_ratios=(1.05, 1),
            left=0.045,
            right=0.975,
            top=0.92,
            bottom=0.16,
            wspace=0.06,
            hspace=0.16,
        )
        self.camera_1_ax = self.fig.add_subplot(grid[0, 0])
        self.camera_2_ax = self.fig.add_subplot(grid[0, 1])
        self.detail_ax = self.fig.add_subplot(grid[1, 0])
        self.table_ax = self.fig.add_subplot(grid[1, 1])
        self.detail_ax.axis("off")
        self.table_ax.axis("off")

        self.condition_buttons: list[Button] = []
        button_width = 0.105
        start_x = 0.31
        for index, condition in enumerate(CONDITION_ORDER):
            axis = self.fig.add_axes(
                [start_x + index * (button_width + 0.008), 0.055, button_width, 0.05]
            )
            button = Button(axis, CONDITION_LABELS[condition])
            button.on_clicked(
                lambda _event, selected=index: self.set_condition(selected)
            )
            self.condition_buttons.append(button)

        previous_axis = self.fig.add_axes([0.05, 0.055, 0.105, 0.05])
        next_axis = self.fig.add_axes([0.165, 0.055, 0.105, 0.05])
        self.previous_button = Button(previous_axis, "← Example")
        self.next_button = Button(next_axis, "Example →")
        self.previous_button.on_clicked(lambda _event: self.change_example(-1))
        self.next_button.on_clicked(lambda _event: self.change_example(1))

        self.help_text = self.fig.text(
            0.975,
            0.072,
            "Keys: 1–4 condition  ·  ←/→ condition  ·  ↑/↓ example",
            ha="right",
            va="center",
            fontsize=9,
            color="#555555",
        )
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load_example()

    @property
    def example(self) -> dict[str, Any]:
        return self.examples[self.example_index]

    @property
    def condition(self) -> str:
        return CONDITION_ORDER[self.condition_index]

    def load_example(self) -> None:
        example = self.example
        mask = np.ones(len(self.predictions), dtype=bool)
        for key in OBSERVATION_KEY:
            mask &= self.predictions[key].to_numpy() == example[key]
        self.current_rows = self.predictions.loc[mask].copy()
        self.current_rows["language_condition"] = pd.Categorical(
            self.current_rows.language_condition,
            categories=CONDITION_ORDER,
            ordered=True,
        )
        self.current_rows = self.current_rows.sort_values("language_condition")
        if len(self.current_rows) != 4:
            raise ValueError(
                f"Curated example {self.example_index + 1} does not have four stored conditions"
            )
        if self.current_rows.noise_id.nunique() != 1:
            raise ValueError("Curated example conditions do not share the same stored noise ID")

        camera_paths = [
            REPO_ROOT / example["camera_1_asset"],
            REPO_ROOT / example["camera_2_asset"],
        ]
        if not all(path.is_file() for path in camera_paths):
            raise FileNotFoundError(
                "Packaged demo camera assets are missing. Run "
                "`.venv/bin/python experiments/finalize_results.py` once."
            )
        images = [np.asarray(Image.open(path).convert("RGB")) for path in camera_paths]
        for axis, image, title in zip(
            (self.camera_1_ax, self.camera_2_ax),
            images,
            ("Camera 1 · agent view", "Camera 2 · eye-in-hand view"),
            strict=True,
        ):
            axis.clear()
            axis.imshow(np.rot90(image, 2))
            axis.set_title(title, fontsize=11, fontweight="bold")
            axis.axis("off")
        self.refresh()

    def selected_row(self) -> pd.Series:
        row = self.current_rows[
            self.current_rows.language_condition.astype(str) == self.condition
        ]
        if len(row) != 1:
            raise ValueError(f"Expected one stored row for condition {self.condition}")
        return row.iloc[0]

    def refresh(self) -> None:
        row = self.selected_row()
        expert = row[EXPERT_COLUMNS].to_numpy(dtype=float)
        predicted = row[PRED_COLUMNS].to_numpy(dtype=float)
        delta = predicted - expert
        translation = float(np.linalg.norm(delta[:3]))
        rotation = float(np.linalg.norm(delta[3:6]))
        gripper = float(abs(delta[6]))
        full = float(np.linalg.norm(delta))

        self.fig.suptitle(
            "SmolVLA Controlled-Language Offline Viewer",
            fontsize=16,
            fontweight="bold",
            y=0.975,
        )
        self.detail_ax.clear()
        self.detail_ax.axis("off")
        label_line = ", ".join(ACTION_LABELS)
        details = (
            f"EXAMPLE {self.example_index + 1}/{len(self.examples)}\n"
            f"Episode {int(row.episode_id)}  ·  frame {int(row.frame_index)}  ·  "
            f"progress {float(row.task_progress):.3f}\n"
            f"Selection: {self.example['selection_type'].replace('_', ' ')}\n"
            f"LANGUAGE CONDITION: {CONDITION_LABELS[self.condition]}\n"
            f"Instruction:\n{textwrap.fill(str(row.instruction), width=66)}\n\n"
            f"Action order: [{label_line}]\n"
            f"Recorded expert 7-D action:\n{format_action(expert)}\n"
            f"Stored SmolVLA predicted 7-D action:\n{format_action(predicted)}\n\n"
            f"SELECTED PREDICTION VS RECORDED EXPERT\n"
            f"Translation difference:  {translation:.4f}\n"
            f"Axis-angle difference:   {rotation:.4f}\n"
            f"Gripper difference:      {gripper:.4f}\n"
            f"Full 7-D action distance:{full:>8.4f}\n"
            f"Stored inference latency:{float(row.inference_time_ms):>8.1f} ms"
        )
        self.detail_ax.text(
            0.01,
            0.99,
            details,
            transform=self.detail_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            family="monospace",
            color="#222222",
        )

        self.table_ax.clear()
        self.table_ax.axis("off")
        table_rows = []
        for condition in CONDITION_ORDER:
            stored = self.current_rows[
                self.current_rows.language_condition.astype(str) == condition
            ].iloc[0]
            vector = stored[PRED_COLUMNS].to_numpy(float)
            table_rows.append([CONDITION_LABELS[condition], *[f"{v:+.3f}" for v in vector]])
        table = self.table_ax.table(
            cellText=table_rows,
            colLabels=["Condition", "x", "y", "z", "aa-1", "aa-2", "aa-3", "grip"],
            cellLoc="center",
            colLoc="center",
            loc="upper center",
            colWidths=[0.22, *([0.11] * 7)],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.8)
        table.scale(1.0, 1.65)
        selected_table_row = self.condition_index + 1
        for (table_row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#D5D5D5")
            if table_row == 0:
                cell.set_facecolor("#EEEEEE")
                cell.set_text_props(fontweight="bold")
            elif table_row == selected_table_row:
                cell.set_facecolor("#E8EEF6")
                cell.set_text_props(fontweight="bold", color=COLORS[self.condition])
            else:
                cell.set_facecolor("white")
        self.table_ax.set_title(
            "All four stored predictions for the same image, state, frame, and noise",
            fontsize=10.5,
            fontweight="bold",
            pad=12,
        )
        reason = textwrap.fill(self.example["reason"], width=76)
        self.table_ax.text(
            0.0,
            0.47,
            "LANGUAGE-INDUCED DIFFERENCE VS CORRECT PREDICTION\n"
            "Why this example was selected:\n" + reason,
            transform=self.table_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#444444",
        )
        self.table_ax.text(
            0.0,
            0.19,
            "Controlled comparison: cameras, robot state, frame, checkpoint, preprocessing, and explicit flow noise are fixed. Only the task string changes.",
            transform=self.table_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            fontweight="bold",
            wrap=True,
            color="#333333",
        )

        for index, button in enumerate(self.condition_buttons):
            button.ax.set_facecolor("#DCE7F4" if index == self.condition_index else "#F5F5F5")
        self.fig.canvas.draw_idle()

    def set_condition(self, index: int) -> None:
        self.condition_index = index % len(CONDITION_ORDER)
        self.refresh()

    def change_example(self, step: int) -> None:
        self.example_index = (self.example_index + step) % len(self.examples)
        self.load_example()

    def on_key(self, event: Any) -> None:
        if event.key in ("right", "]"):
            self.set_condition(self.condition_index + 1)
        elif event.key in ("left", "["):
            self.set_condition(self.condition_index - 1)
        elif event.key == "up":
            self.change_example(-1)
        elif event.key == "down":
            self.change_example(1)
        elif event.key in ("1", "2", "3", "4"):
            self.set_condition(int(event.key) - 1)

    def smoke_test(self) -> None:
        original_example_index = self.example_index
        seen_examples: list[int] = []
        for example_index in range(len(self.examples)):
            self.example_index = example_index
            self.load_example()
            seen_examples.append(int(self.example["example_index"]))
            original_camera_arrays = [
                np.asarray(self.camera_1_ax.images[0].get_array()).copy(),
                np.asarray(self.camera_2_ax.images[0].get_array()).copy(),
            ]
            seen_conditions: list[str] = []
            for condition_index, condition in enumerate(CONDITION_ORDER):
                self.set_condition(condition_index)
                self.fig.canvas.draw()
                if self.selected_row().language_condition != condition:
                    raise RuntimeError(f"Condition switch failed for {condition}")
                seen_conditions.append(condition)
                for axis, original in zip(
                    (self.camera_1_ax, self.camera_2_ax),
                    original_camera_arrays,
                    strict=True,
                ):
                    if not np.array_equal(np.asarray(axis.images[0].get_array()), original):
                        raise RuntimeError("Camera image changed during language-condition switch")
            if seen_conditions != list(CONDITION_ORDER):
                raise RuntimeError(f"Condition cycle incomplete: {seen_conditions}")
        self.example_index = original_example_index
        self.load_example()
        expected_examples = [int(example["example_index"]) for example in self.examples]
        if seen_examples != expected_examples:
            raise RuntimeError(f"Example cycle incomplete: {seen_examples}")
        print(
            "DEMO SMOKE PASS: "
            f"examples={','.join(str(value) for value in seen_examples)}, "
            f"conditions={','.join(CONDITION_ORDER)}, "
            "same_cameras_across_conditions=true, model_inference=0",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View stored SmolVLA predictions for controlled language conditions."
    )
    parser.add_argument(
        "--example-index",
        type=int,
        default=1,
        help="One-based curated example index (default: 1).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Render headlessly and verify switching across all four conditions.",
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="Print the curated examples and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions, examples = load_data()
    if args.list_examples:
        for example in examples:
            print(
                f"{example['example_index']}: episode {example['episode_id']}, "
                f"frame {example['frame_index']}, progress {example['task_progress']:.3f} — "
                f"{example['reason']}"
            )
        return
    if not 1 <= args.example_index <= len(examples):
        raise ValueError(
            f"--example-index must be between 1 and {len(examples)}, got {args.example_index}"
        )
    viewer = OfflineDemo(predictions, examples, example_index=args.example_index - 1)
    if args.smoke_test:
        viewer.smoke_test()
        plt.close(viewer.fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
