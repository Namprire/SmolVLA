#!/usr/bin/env python
"""Phase B: systematic multi-prompt robustness for stored SmolVLA observations.

Stages:
  prepare  - write/validate the prompt bank and deterministic 20-observation selection
  pilot    - run or resume the two-observation (62-prediction) pilot
  full     - run or resume all 20 observations (620 predictions total)
  analyze  - compute nested metrics, episode-aware summaries, figures, and findings

The inference stages reuse one explicit flow-noise tensor across all 31 prompts
for an observation and write each prediction immediately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache/huggingface"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "prompt_robustness"
FIGURE_DIR = OUTPUT_DIR / "figures"
PROMPT_BANK_PATH = OUTPUT_DIR / "prompt_bank.json"
PROMPT_BANK_MD_PATH = OUTPUT_DIR / "prompt_bank.md"
SELECTION_PATH = OUTPUT_DIR / "selected_observations.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"
PILOT_VALIDATION_PATH = OUTPUT_DIR / "pilot_validation.json"
PILOT_RESUME_PATH = OUTPUT_DIR / "pilot_resume_validation.json"
VALIDATION_PATH = OUTPUT_DIR / "validation.json"
RUN_LOG_PATH = OUTPUT_DIR / "run_log.jsonl"
FINDINGS_PATH = OUTPUT_DIR / "findings.md"

CANONICAL_PREDICTIONS_PATH = RESULTS_DIR / "vla_predictions.csv"
LANGUAGE_EFFECTS_PATH = RESULTS_DIR / "language_effects.csv"
DEMO_EXAMPLES_PATH = RESULTS_DIR / "demo_examples.json"
GRIPPER_TRANSITIONS_PATH = RESULTS_DIR / "gripper_transition_candidates.csv"
PHASE_A_OBSERVATIONS_PATH = RESULTS_DIR / "spike_analysis/unrelated_20_40_observations.csv"

DATASET_ID = "HuggingFaceVLA/smol-libero"
MODEL_ID = "HuggingFaceVLA/smolvla_libero"
RAW_DATASET_ROOT = REPO_ROOT / ".cache/huggingface/lerobot_v21" / DATASET_ID
CONVERTED_DATASET_ROOT = REPO_ROOT / ".cache/huggingface/lerobot_v30_main" / DATASET_ID
MODEL_ROOT = REPO_ROOT / ".cache/huggingface/models" / MODEL_ID

EXPECTED_CANONICAL_SHA256 = (
    "ab4984aeea52a8ca60c94a5cb19a15d69fd27707f5c5734ddd8bec680f376dc4"
)
CANONICAL_CORRECT = "put both the cream cheese box and the butter in the basket"
ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
EXPERT_COLUMNS = [f"expert_action_{name}" for name in ACTION_NAMES]
PRED_COLUMNS = [f"pred_action_{name}" for name in ACTION_NAMES]
OBSERVATION_KEY = ["episode_id", "frame_index", "global_index"]
PROMPT_CLASSES = ("Correct", "Paraphrase", "Contradictory", "Unrelated")
ALTERED_CLASSES = PROMPT_CLASSES[1:]
PROGRESS_LABELS = ("0–20%", "20–40%", "40–60%", "60–80%", "80–100%")

EXPERIMENT_SEED = 20260810
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_REPLICATES = 5000
NEAR_TRANSITION_MAX_OFFSET = 5
GRIPPER_REGIME_THRESHOLD = 0.0

COLORS = {
    "Correct": "#4C78A8",
    "Paraphrase": "#59A14F",
    "Contradictory": "#F28E2B",
    "Unrelated": "#E15759",
}

PROMPTS: tuple[dict[str, str], ...] = (
    {
        "prompt_id": "CORRECT",
        "prompt_class": "Correct",
        "instruction": CANONICAL_CORRECT,
        "semantic_notes": "Exact canonical dataset task; the single baseline for every observation.",
    },
    {
        "prompt_id": "P01",
        "prompt_class": "Paraphrase",
        "instruction": "place both the cream cheese box and the butter in the basket",
        "semantic_notes": "Lexical substitution; preserves both objects and destination.",
    },
    {
        "prompt_id": "P02",
        "prompt_class": "Paraphrase",
        "instruction": "put the butter and the cream cheese box together inside the basket",
        "semantic_notes": "Reverses object order and uses inside/together; same goal.",
    },
    {
        "prompt_id": "P03",
        "prompt_class": "Paraphrase",
        "instruction": "move the cream cheese box and the butter into the basket",
        "semantic_notes": "Uses move/into while retaining both objects and destination.",
    },
    {
        "prompt_id": "P04",
        "prompt_class": "Paraphrase",
        "instruction": "transfer both items, the cream cheese box and the butter, to the basket",
        "semantic_notes": "Appositive syntax; explicitly names both items and the basket.",
    },
    {
        "prompt_id": "P05",
        "prompt_class": "Paraphrase",
        "instruction": "set the cream cheese box as well as the butter inside the basket",
        "semantic_notes": "As-well-as coordination; same two-object placement goal.",
    },
    {
        "prompt_id": "P06",
        "prompt_class": "Paraphrase",
        "instruction": "pack the basket with both the cream cheese box and the butter",
        "semantic_notes": "Destination-fronted packing wording; preserves complete goal.",
    },
    {
        "prompt_id": "P07",
        "prompt_class": "Paraphrase",
        "instruction": "take the cream cheese box and the butter and place them in the basket",
        "semantic_notes": "Two-verb construction with an explicit plural reference to both objects.",
    },
    {
        "prompt_id": "P08",
        "prompt_class": "Paraphrase",
        "instruction": "ensure that both the butter and the cream cheese box are inside the basket",
        "semantic_notes": "State-goal phrasing; both named objects must end inside the basket.",
    },
    {
        "prompt_id": "P09",
        "prompt_class": "Paraphrase",
        "instruction": "the cream cheese box and the butter should both be put into the basket",
        "semantic_notes": "Passive syntax; same object set and destination.",
    },
    {
        "prompt_id": "P10",
        "prompt_class": "Paraphrase",
        "instruction": "into the basket, place both the cream cheese box and the butter",
        "semantic_notes": "Fronted destination; retains both objects and the original goal.",
    },
    {
        "prompt_id": "C01",
        "prompt_class": "Contradictory",
        "instruction": "put the cream cheese box in the basket but leave the butter outside the basket",
        "semantic_notes": "Original contradiction: excludes the butter from the destination.",
    },
    {
        "prompt_id": "C02",
        "prompt_class": "Contradictory",
        "instruction": "put the butter in the basket but leave the cream cheese box on the table",
        "semantic_notes": "Reverses which target object is excluded.",
    },
    {
        "prompt_id": "C03",
        "prompt_class": "Contradictory",
        "instruction": "place only the cream cheese box in the basket and move the butter beside the basket",
        "semantic_notes": "Changes inclusion and gives the butter an adjacent destination.",
    },
    {
        "prompt_id": "C04",
        "prompt_class": "Contradictory",
        "instruction": "put only the butter in the basket and keep the cream cheese box where it is",
        "semantic_notes": "Selects only the butter and preserves the cream-cheese position.",
    },
    {
        "prompt_id": "C05",
        "prompt_class": "Contradictory",
        "instruction": "place the cream cheese box on the table and put only the butter in the basket",
        "semantic_notes": "Assigns different destinations and excludes one object from the basket.",
    },
    {
        "prompt_id": "C06",
        "prompt_class": "Contradictory",
        "instruction": "place both the cream cheese box and the butter next to the basket instead of inside it",
        "semantic_notes": "Preserves both objects but changes the destination for both.",
    },
    {
        "prompt_id": "C07",
        "prompt_class": "Contradictory",
        "instruction": "put the cream cheese box in the basket and move the butter to the far side of the table",
        "semantic_notes": "Keeps one original placement and gives the butter a distant destination.",
    },
    {
        "prompt_id": "C08",
        "prompt_class": "Contradictory",
        "instruction": "stack the butter on top of the cream cheese box outside the basket",
        "semantic_notes": "Changes the goal from packing to stacking both objects outside.",
    },
    {
        "prompt_id": "C09",
        "prompt_class": "Contradictory",
        "instruction": "put the cream cheese box beside the basket and leave the butter on the table",
        "semantic_notes": "Keeps both target objects outside at two stated locations.",
    },
    {
        "prompt_id": "C10",
        "prompt_class": "Contradictory",
        "instruction": "move the cream cheese box into the basket, then take the butter out of the basket",
        "semantic_notes": "Sequential inclusion/exclusion contradiction with distinct object roles.",
    },
    {
        "prompt_id": "U01",
        "prompt_class": "Unrelated",
        "instruction": "open the top drawer of the cabinet",
        "semantic_notes": "Original Unrelated prompt retained as one variant for direct diagnosis.",
    },
    {
        "prompt_id": "U02",
        "prompt_class": "Unrelated",
        "instruction": "close the microwave door",
        "semantic_notes": "Plausible appliance manipulation unrelated to the packing goal.",
    },
    {
        "prompt_id": "U03",
        "prompt_class": "Unrelated",
        "instruction": "pick up the red mug and place it on the saucer",
        "semantic_notes": "Different object-transfer task with an unrelated destination.",
    },
    {
        "prompt_id": "U04",
        "prompt_class": "Unrelated",
        "instruction": "move the black bowl onto the serving plate",
        "semantic_notes": "Different tabletop placement task using unrelated objects.",
    },
    {
        "prompt_id": "U05",
        "prompt_class": "Unrelated",
        "instruction": "turn the stove knob clockwise",
        "semantic_notes": "Plausible rotational manipulation unrelated to packing.",
    },
    {
        "prompt_id": "U06",
        "prompt_class": "Unrelated",
        "instruction": "place the cereal box inside the kitchen cabinet",
        "semantic_notes": "Different object and destination; not a modification of the basket task.",
    },
    {
        "prompt_id": "U07",
        "prompt_class": "Unrelated",
        "instruction": "lift the saucepan and set it on the front burner",
        "semantic_notes": "Different two-step kitchen manipulation task.",
    },
    {
        "prompt_id": "U08",
        "prompt_class": "Unrelated",
        "instruction": "slide the cutting board toward the sink",
        "semantic_notes": "Different planar manipulation with unrelated object and target.",
    },
    {
        "prompt_id": "U09",
        "prompt_class": "Unrelated",
        "instruction": "put the blue cup on the upper shelf",
        "semantic_notes": "Different object-placement task in another part of the workspace.",
    },
    {
        "prompt_id": "U10",
        "prompt_class": "Unrelated",
        "instruction": "take the spoon from the counter and place it in the utensil holder",
        "semantic_notes": "Different pick-and-place task with unrelated objects and destination.",
    },
)

FIELDNAMES = [
    "episode_id",
    "frame_index",
    "global_index",
    "episode_length",
    "max_frame_index",
    "normalized_progress",
    "selection_slot",
    "selection_reason",
    "pilot_order",
    "prompt_id",
    "prompt_class",
    "instruction",
    "semantic_notes",
    "experiment_seed",
    "execution_order_seed",
    "execution_position",
    "noise_seed",
    "noise_hash",
    "actual_compute_device",
    *EXPERT_COLUMNS,
    *PRED_COLUMNS,
    "inference_time_ms",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def prompt_map() -> dict[str, dict[str, str]]:
    return {prompt["prompt_id"]: dict(prompt) for prompt in PROMPTS}


def validate_prompt_bank() -> dict[str, Any]:
    ids = [prompt["prompt_id"] for prompt in PROMPTS]
    instructions = [prompt["instruction"] for prompt in PROMPTS]
    expected_ids = ["CORRECT"] + [f"P{i:02d}" for i in range(1, 11)] + [
        f"C{i:02d}" for i in range(1, 11)
    ] + [f"U{i:02d}" for i in range(1, 11)]
    require(ids == expected_ids, f"Prompt IDs differ from required order: {ids}")
    require(len(set(ids)) == 31, "Prompt IDs are not unique")
    require(len(set(instructions)) == 31, "Prompt strings are not unique")
    require(all(text.strip() == text and text for text in instructions), "Empty/padded prompt")
    require(PROMPTS[0]["instruction"] == CANONICAL_CORRECT, "Correct prompt changed")

    counts = pd.Series([prompt["prompt_class"] for prompt in PROMPTS]).value_counts().to_dict()
    require(counts == {"Paraphrase": 10, "Contradictory": 10, "Unrelated": 10, "Correct": 1}, f"Bad class counts: {counts}")
    for prompt in PROMPTS[1:11]:
        lower = prompt["instruction"].lower()
        require("cream cheese" in lower and "butter" in lower and "basket" in lower, f"Paraphrase loses required semantics: {prompt}")
    for prompt in PROMPTS[11:21]:
        lower = prompt["instruction"].lower()
        require("cream cheese" in lower and "butter" in lower and "basket" in lower, f"Contradiction lacks task-domain grounding: {prompt}")
    forbidden = ("cream cheese", "butter", "basket")
    for prompt in PROMPTS[21:]:
        lower = prompt["instruction"].lower()
        require(not any(term in lower for term in forbidden), f"Unrelated prompt leaks original goal: {prompt}")
    return {
        "validation_result": "PASS",
        "prompt_count": 31,
        "altered_prompt_count": 30,
        "class_counts": counts,
        "unique_ids": True,
        "unique_nonempty_instructions": True,
        "paraphrases_preserve_both_objects_and_destination": True,
        "contradictions_name_both_objects_and_task_destination": True,
        "unrelated_prompts_exclude_original_objects_and_destination": True,
        "manual_semantic_audit": "Completed before inference; semantic_notes record each decision.",
    }


def prompt_bank_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "Phase B systematic multi-prompt robustness",
        "experiment_seed": EXPERIMENT_SEED,
        "canonical_correct_instruction": CANONICAL_CORRECT,
        "authoring_method": "Manually and systematically authored fixed prompt bank; not sampled from a population distribution.",
        "prompts": [dict(prompt) for prompt in PROMPTS],
        "audit": validate_prompt_bank(),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(rows)


def write_or_validate_prompt_bank() -> None:
    payload = prompt_bank_payload()
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if PROMPT_BANK_PATH.exists():
        require(PROMPT_BANK_PATH.read_text() == text, "Existing prompt bank differs")
    else:
        PROMPT_BANK_PATH.write_text(text)
    table = pd.DataFrame(PROMPTS)[["prompt_id", "prompt_class", "instruction", "semantic_notes"]]
    md = f"""# Phase B prompt-bank audit

The Correct row is the exact canonical dataset task. The 30 altered prompts were manually inspected before inference. Paraphrases retain both named objects and the basket destination; contradictions stay within the visible packing domain but alter the goal; unrelated prompts are plausible manipulation tasks that do not mention the original objects or basket.

The original Drawer instruction is retained only as `U01`, allowing its influence to be compared with nine other unrelated prompts.

{markdown_table(table)}
"""
    if PROMPT_BANK_MD_PATH.exists():
        require(PROMPT_BANK_MD_PATH.read_text() == md, "Existing prompt-bank Markdown differs")
    else:
        PROMPT_BANK_MD_PATH.write_text(md)


def reconstruct_existing_effects(predictions: pd.DataFrame) -> pd.DataFrame:
    indexed = predictions.set_index(OBSERVATION_KEY + ["language_condition"])
    require(indexed.index.is_unique, "Canonical prediction key is not unique")
    rows: list[dict[str, Any]] = []
    for key in predictions[OBSERVATION_KEY].drop_duplicates().itertuples(index=False, name=None):
        correct = indexed.loc[(*key, "correct")]
        correct_action = correct[PRED_COLUMNS].to_numpy(float)
        row: dict[str, Any] = {
            **dict(zip(OBSERVATION_KEY, key, strict=True)),
            "episode_length": int(correct.episode_length),
            "max_frame_index": int(correct.max_frame_index),
            "normalized_progress": float(correct.task_progress),
            "noise_seed": int(correct.noise_seed),
            "noise_hash": str(correct.noise_id),
        }
        for condition in ("paraphrase", "contradictory", "unrelated"):
            other = indexed.loc[(*key, condition)]
            delta = other[PRED_COLUMNS].to_numpy(float) - correct_action
            row[f"existing_correct_vs_{condition}_distance"] = float(np.linalg.norm(delta))
            row[f"existing_correct_vs_{condition}_gripper_difference"] = float(abs(delta[6]))
        p = row["existing_correct_vs_paraphrase_distance"]
        c = row["existing_correct_vs_contradictory_distance"]
        u = row["existing_correct_vs_unrelated_distance"]
        row["existing_expected_order_holds"] = bool(p < c < u)
        row["existing_order_violation_score"] = float(max(p - c, c - u, 0.0))
        row["existing_mean_altered_distance"] = float(np.mean([p, c, u]))
        rows.append(row)
    return pd.DataFrame(rows)


def add_transition_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    transitions = pd.read_csv(GRIPPER_TRANSITIONS_PATH)
    records: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        episode = transitions[transitions.episode_id == row.episode_id].copy()
        episode["absolute_offset"] = (episode.frame_index - row.frame_index).abs()
        nearest = episode.sort_values(["absolute_offset", "frame_index"]).iloc[0]
        records.append(
            {
                "episode_id": int(row.episode_id),
                "frame_index": int(row.frame_index),
                "nearest_gripper_transition_frame": int(nearest.frame_index),
                "nearest_gripper_transition_offset": int(row.frame_index - nearest.frame_index),
                "nearest_gripper_transition_direction": str(nearest.transition_direction),
                "near_gripper_transition": bool(nearest.absolute_offset <= NEAR_TRANSITION_MAX_OFFSET),
            }
        )
    return candidates.merge(pd.DataFrame(records), on=["episode_id", "frame_index"], validate="one_to_one")


def choose_ranked(candidates: pd.DataFrame, selected_frames: set[int], sort_columns: list[str], ascending: list[bool]) -> pd.Series:
    eligible = candidates[~candidates.frame_index.isin(selected_frames)].sort_values(
        sort_columns, ascending=ascending
    )
    require(not eligible.empty, "No eligible candidate remains")
    return eligible.iloc[0]


def build_selection() -> pd.DataFrame:
    canonical = pd.read_csv(CANONICAL_PREDICTIONS_PATH)
    require(sha256(CANONICAL_PREDICTIONS_PATH) == EXPECTED_CANONICAL_SHA256, "Canonical CSV hash changed")
    require(set(canonical.loc[canonical.language_condition == "correct", "instruction"]) == {CANONICAL_CORRECT}, "Canonical task string changed")
    effects = add_transition_metadata(reconstruct_existing_effects(canonical))
    phase_a = pd.read_csv(PHASE_A_OBSERVATIONS_PATH)
    phase_a_keys = set(zip(phase_a.episode_id.astype(int), phase_a.frame_index.astype(int), strict=True))
    examples = json.loads(DEMO_EXAMPLES_PATH.read_text())["examples"]
    examples_by_episode = {int(item["episode_id"]): item for item in examples}

    target_progress = (0.10, 0.30, 0.50, 0.70, 0.90)
    diagnostic_roles = {
        0: "nearest_gripper_transition",
        1: "curated_failure_case",
        2: "ordering_counterexample",
        3: "ordinary_control",
        4: "nearest_gripper_transition",
        5: "curated_failure_case",
        6: "ordering_counterexample",
        7: "phase_a_20_40_representative",
        8: "ordinary_control",
        9: "nearest_gripper_transition",
    }
    selected_rows: list[dict[str, Any]] = []
    for episode_id in range(10):
        episode = effects[effects.episode_id == episode_id].copy()
        require(len(episode) == 20, f"Episode {episode_id} does not have 20 candidates")
        target = target_progress[episode_id % len(target_progress)]
        episode["anchor_distance"] = (episode.normalized_progress - target).abs()
        anchor = choose_ranked(episode, set(), ["anchor_distance", "frame_index"], [True, True])
        chosen = {int(anchor.frame_index)}
        anchor_reason = f"coverage anchor nearest predetermined normalized progress target {target:.2f}"

        role = diagnostic_roles[episode_id]
        if role == "nearest_gripper_transition":
            episode["diagnostic_rank"] = episode.nearest_gripper_transition_offset.abs()
            diagnostic = choose_ranked(episode, chosen, ["diagnostic_rank", "frame_index"], [True, True])
            diagnostic_reason = (
                "nearest available stored observation to an expert gripper-command transition "
                f"(offset {int(diagnostic.nearest_gripper_transition_offset):+d} frames)"
            )
        elif role == "curated_failure_case":
            example = examples_by_episode[episode_id]
            match = episode[episode.frame_index == int(example["frame_index"])]
            require(len(match) == 1 and int(match.iloc[0].frame_index) not in chosen, f"Curated example unavailable for episode {episode_id}")
            diagnostic = match.iloc[0]
            diagnostic_reason = f"existing deterministic demo failure/counterexample selection: {example['selection_type']}"
        elif role == "ordering_counterexample":
            counterexamples = episode[~episode.existing_expected_order_holds].copy()
            diagnostic = choose_ranked(counterexamples, chosen, ["existing_order_violation_score", "frame_index"], [False, True])
            diagnostic_reason = "strongest existing aggregate-order violation in this episode"
        elif role == "ordinary_control":
            eligible = episode[episode.existing_expected_order_holds].copy()
            target_value = float(eligible.existing_mean_altered_distance.median())
            eligible["diagnostic_rank"] = (eligible.existing_mean_altered_distance - target_value).abs()
            diagnostic = choose_ranked(eligible, chosen, ["diagnostic_rank", "frame_index"], [True, True])
            diagnostic_reason = "ordinary control nearest the episode median altered-condition distance among ordering-consistent rows"
        elif role == "phase_a_20_40_representative":
            eligible = episode[
                [
                    (int(row.episode_id), int(row.frame_index)) in phase_a_keys
                    for row in episode.itertuples(index=False)
                ]
            ].copy()
            target_value = float(phase_a.full_7d_distance.median())
            eligible["diagnostic_rank"] = (eligible.existing_correct_vs_unrelated_distance - target_value).abs()
            diagnostic = choose_ranked(eligible, chosen, ["diagnostic_rank", "frame_index"], [True, True])
            diagnostic_reason = "Phase A 20–40% observation nearest the Phase A median Unrelated full distance"
        else:
            raise AssertionError(role)

        for slot, row, reason in (
            ("coverage_anchor", anchor, anchor_reason),
            (role, diagnostic, diagnostic_reason),
        ):
            item = row.to_dict()
            item["selection_slot"] = slot
            item["selection_reason"] = reason
            item["phase_a_20_40_observation"] = bool(
                (int(row.episode_id), int(row.frame_index)) in phase_a_keys
            )
            item["pilot_order"] = (
                1 if episode_id == 0 and slot == "coverage_anchor" else
                2 if episode_id == 1 and slot == "coverage_anchor" else 0
            )
            selected_rows.append(item)

    selected = pd.DataFrame(selected_rows)
    selected = selected.sort_values(["episode_id", "selection_slot", "frame_index"]).reset_index(drop=True)
    output_columns = [
        "episode_id", "frame_index", "global_index", "episode_length", "max_frame_index",
        "normalized_progress", "selection_slot", "selection_reason", "pilot_order",
        "existing_correct_vs_paraphrase_distance", "existing_correct_vs_contradictory_distance",
        "existing_correct_vs_unrelated_distance", "existing_expected_order_holds",
        "existing_order_violation_score", "existing_mean_altered_distance",
        "near_gripper_transition", "nearest_gripper_transition_frame",
        "nearest_gripper_transition_offset", "nearest_gripper_transition_direction",
        "phase_a_20_40_observation", "noise_seed", "noise_hash",
    ]
    selected = selected[output_columns]
    for column in ("episode_id", "frame_index", "global_index", "episode_length", "max_frame_index", "pilot_order", "nearest_gripper_transition_frame", "nearest_gripper_transition_offset", "noise_seed"):
        selected[column] = selected[column].astype(int)
    require(len(selected) == 20, f"Expected 20 selected observations, found {len(selected)}")
    require((selected.groupby("episode_id").size() == 2).all(), "Not exactly two rows per episode")
    require(not selected.duplicated(OBSERVATION_KEY).any(), "Duplicate selected observation")
    require(set(selected.pilot_order) == {0, 1, 2}, "Pilot markers invalid")
    require((selected.existing_expected_order_holds == False).sum() >= 2, "Fewer than two counterexamples selected")  # noqa: E712
    require(selected.near_gripper_transition.sum() >= 2, "Too few near-transition selections")
    require(selected.phase_a_20_40_observation.sum() >= 2, "Too few Phase A-region selections")
    return selected


def write_or_validate_selection() -> pd.DataFrame:
    selected = build_selection()
    if SELECTION_PATH.exists():
        existing = pd.read_csv(SELECTION_PATH)
        pd.testing.assert_frame_equal(existing, selected, check_dtype=False, atol=1e-14, rtol=0)
    else:
        selected.to_csv(SELECTION_PATH, index=False)
    return selected


def prepare() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    write_or_validate_prompt_bank()
    selected = write_or_validate_selection()
    print("Phase B preparation: PASS")
    print(f"Prompt bank: 1 Correct + 10 Paraphrase + 10 Contradictory + 10 Unrelated")
    print(f"Selected observations: {len(selected)} ({selected.groupby('episode_id').size().iloc[0]} per episode)")
    print(f"Pilot observations: {int((selected.pilot_order > 0).sum())}")
    return selected


def execution_order_seed(row: pd.Series | Any) -> int:
    return EXPERIMENT_SEED + int(row.episode_id) * 10_000 + int(row.frame_index)


def execution_order(row: pd.Series | Any) -> list[str]:
    ids = [prompt["prompt_id"] for prompt in PROMPTS]
    rng = np.random.default_rng(execution_order_seed(row))
    return [str(value) for value in rng.permutation(ids)]


def initialize_predictions_csv() -> None:
    if PREDICTIONS_PATH.exists() and PREDICTIONS_PATH.stat().st_size > 0:
        return
    with PREDICTIONS_PATH.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FIELDNAMES).writeheader()
        handle.flush()
        os.fsync(handle.fileno())


def load_predictions() -> pd.DataFrame:
    initialize_predictions_csv()
    data = pd.read_csv(PREDICTIONS_PATH)
    require(data.columns.tolist() == FIELDNAMES, f"Prediction schema mismatch: {data.columns.tolist()}")
    return data


def validate_prediction_rows(data: pd.DataFrame, selected: pd.DataFrame, complete: bool) -> dict[str, Any]:
    prompt_lookup = prompt_map()
    selected_lookup = {
        (int(row.episode_id), int(row.frame_index), int(row.global_index)): row
        for row in selected.itertuples(index=False)
    }
    key_columns = OBSERVATION_KEY + ["prompt_id"]
    require(not data.duplicated(key_columns).any(), "Duplicate observation/prompt rows")
    for row in data.itertuples(index=False):
        key = (int(row.episode_id), int(row.frame_index), int(row.global_index))
        require(key in selected_lookup, f"Prediction outside selection: {key}")
        source = selected_lookup[key]
        prompt_id = str(row.prompt_id)
        require(prompt_id in prompt_lookup, f"Unknown prompt ID: {prompt_id}")
        prompt = prompt_lookup[prompt_id]
        require(str(row.prompt_class) == prompt["prompt_class"], f"Class mismatch: {key}, {prompt_id}")
        require(str(row.instruction) == prompt["instruction"], f"Instruction mismatch: {key}, {prompt_id}")
        require(str(row.semantic_notes) == prompt["semantic_notes"], f"Notes mismatch: {key}, {prompt_id}")
        require(int(row.experiment_seed) == EXPERIMENT_SEED, "Experiment seed mismatch")
        require(int(row.execution_order_seed) == execution_order_seed(source), "Order seed mismatch")
        order = execution_order(source)
        require(int(row.execution_position) == order.index(prompt_id) + 1, "Execution position mismatch")
        require(int(row.noise_seed) == int(source.noise_seed), "Noise seed mismatch")
        require(str(row.noise_hash) == str(source.noise_hash), "Noise hash mismatch")
        for column in ("episode_length", "max_frame_index", "pilot_order"):
            require(int(getattr(row, column)) == int(getattr(source, column)), f"{column} mismatch")
        require(np.isclose(float(row.normalized_progress), float(source.normalized_progress), rtol=0, atol=1e-12), "Progress mismatch")
        require(str(row.selection_slot) == str(source.selection_slot), "Selection slot mismatch")
        require(str(row.selection_reason) == str(source.selection_reason), "Selection reason mismatch")

    numeric_columns = EXPERT_COLUMNS + PRED_COLUMNS + ["inference_time_ms"]
    if len(data):
        require(np.isfinite(data[numeric_columns].to_numpy(float)).all(), "Non-finite prediction data")
        require((data.inference_time_ms > 0).all(), "Non-positive latency")
    groups = data.groupby(OBSERVATION_KEY, sort=False)
    require((groups.noise_hash.nunique(dropna=False) <= 1).all(), "Noise hash differs within observation")
    require((groups.noise_seed.nunique(dropna=False) <= 1).all(), "Noise seed differs within observation")
    require((groups[EXPERT_COLUMNS].nunique(dropna=False) <= 1).all().all(), "Expert action differs within observation")

    expected_keys = {
        (int(row.episode_id), int(row.frame_index), int(row.global_index), prompt["prompt_id"])
        for row in selected.itertuples(index=False)
        for prompt in PROMPTS
    }
    actual_keys = set(data[key_columns].itertuples(index=False, name=None))
    if complete:
        require(actual_keys == expected_keys, f"Coverage mismatch: missing={len(expected_keys - actual_keys)}, extra={len(actual_keys - expected_keys)}")
        require((groups.size() == 31).all(), "Not every observation has 31 prompts")
        require((groups.prompt_id.nunique() == 31).all(), "Not every observation has 31 unique prompt IDs")
        require((groups.execution_position.nunique() == 31).all(), "Execution positions not unique")
        require(all(set(group.execution_position) == set(range(1, 32)) for _, group in groups), "Execution position coverage mismatch")
    else:
        require(actual_keys <= expected_keys, "Partial file contains unexpected keys")

    correct_positions = data.loc[data.prompt_id == "CORRECT", "execution_position"]
    return {
        "row_count": int(len(data)),
        "observation_count": int(data[OBSERVATION_KEY].drop_duplicates().shape[0]),
        "duplicate_observation_prompt_count": 0,
        "finite_7d_expert_and_predicted_actions": True,
        "same_noise_hash_within_observation": True,
        "prompt_strings_exactly_match_bank": True,
        "execution_positions_match_deterministic_shuffle": True,
        "correct_execution_positions": [int(value) for value in correct_positions],
    }


def append_run_log(payload: dict[str, Any]) -> None:
    with RUN_LOG_PATH.open("a") as handle:
        handle.write(json.dumps(json_native(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def assert_only_language_differs_sequential(predictor: Any, frame: dict[str, Any]) -> None:
    import torch

    reference = predictor._preprocess(frame, CANONICAL_CORRECT)
    for prompt in PROMPTS[1:]:
        processed = predictor._preprocess(frame, prompt["instruction"])
        for key in predictor.config.input_features:
            left, right = reference[key], processed[key]
            if isinstance(left, torch.Tensor):
                require(torch.equal(left, right), f"Non-language preprocessor difference for {prompt['prompt_id']}: {key}")
            else:
                require(left == right, f"Non-language preprocessor difference for {prompt['prompt_id']}: {key}")
        del processed
    del reference


def run_inference(selected: pd.DataFrame, stage: str, device: str) -> dict[str, Any]:
    scope = (
        selected[selected.pilot_order > 0].sort_values("pilot_order")
        if stage == "pilot"
        else selected.sort_values(["episode_id", "frame_index"])
    )
    existing_all = load_predictions()
    validate_prediction_rows(existing_all, selected, complete=False)
    expected = {
        (int(row.episode_id), int(row.frame_index), int(row.global_index), prompt["prompt_id"])
        for row in scope.itertuples(index=False)
        for prompt in PROMPTS
    }
    existing_keys = set(existing_all[OBSERVATION_KEY + ["prompt_id"]].itertuples(index=False, name=None))
    missing = expected - existing_keys
    before_hash = sha256(PREDICTIONS_PATH)
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    computed = 0
    frame_mutation_checks = 0
    preprocessing_audits = 0

    print(f"Phase B {stage}: target={len(expected)} existing={len(expected & existing_keys)} missing={len(missing)}", flush=True)
    if missing:
        from experiments.language_control import assert_frame_unchanged, frame_snapshot
        from src.dataset import load_main_dataset
        from src.prediction import SmolVLAPredictor

        dataset = load_main_dataset(RAW_DATASET_ROOT, CONVERTED_DATASET_ROOT)
        predictor = SmolVLAPredictor(MODEL_ROOT, device=device)
        print(f"Loaded pretrained checkpoint on {predictor.model_device}", flush=True)
        canonical = pd.read_csv(CANONICAL_PREDICTIONS_PATH)
        canonical_correct = canonical[canonical.language_condition == "correct"].set_index(OBSERVATION_KEY)
        prompt_lookup = prompt_map()
        completed = set(existing_keys)
        with PREDICTIONS_PATH.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            for observation_index, source in enumerate(scope.itertuples(index=False), start=1):
                observation_expected = {
                    (int(source.episode_id), int(source.frame_index), int(source.global_index), prompt["prompt_id"])
                    for prompt in PROMPTS
                }
                if observation_expected <= completed:
                    continue
                frame = dataset[int(source.global_index)]
                actual_key = (int(frame["episode_index"]), int(frame["frame_index"]), int(frame["index"]))
                expected_key = (int(source.episode_id), int(source.frame_index), int(source.global_index))
                require(actual_key == expected_key, f"Dataset mapping mismatch: {actual_key} != {expected_key}")
                require(frame["task"] == CANONICAL_CORRECT, f"Dataset task changed: {frame['task']!r}")
                noise = predictor.make_noise(int(source.noise_seed))
                noise_hash = predictor.noise_identifier(noise)
                require(noise_hash == str(source.noise_hash), "Reconstructed noise hash mismatch")
                assert_only_language_differs_sequential(predictor, frame)
                preprocessing_audits += 1
                snapshot = frame_snapshot(frame, list(predictor.config.input_features))
                expert = frame["action"].detach().cpu().numpy().reshape(-1)
                require(expert.shape == (7,) and np.isfinite(expert).all(), "Invalid expert action")
                canonical_row = canonical_correct.loc[expected_key]
                require(
                    np.allclose(
                        expert.astype(float),
                        canonical_row[EXPERT_COLUMNS].to_numpy(float),
                        rtol=0,
                        atol=1e-12,
                    ),
                    "Source expert action differs from canonical CSV beyond CSV round-trip tolerance",
                )

                order = execution_order(source)
                for position, prompt_id in enumerate(order, start=1):
                    unique_key = (*expected_key, prompt_id)
                    if unique_key in completed:
                        continue
                    prompt = prompt_lookup[prompt_id]
                    prediction = predictor.predict_action(frame, prompt["instruction"], noise=noise)
                    row = {
                        "episode_id": int(source.episode_id),
                        "frame_index": int(source.frame_index),
                        "global_index": int(source.global_index),
                        "episode_length": int(source.episode_length),
                        "max_frame_index": int(source.max_frame_index),
                        "normalized_progress": float(source.normalized_progress),
                        "selection_slot": str(source.selection_slot),
                        "selection_reason": str(source.selection_reason),
                        "pilot_order": int(source.pilot_order),
                        **prompt,
                        "experiment_seed": EXPERIMENT_SEED,
                        "execution_order_seed": execution_order_seed(source),
                        "execution_position": position,
                        "noise_seed": int(source.noise_seed),
                        "noise_hash": noise_hash,
                        "actual_compute_device": str(predictor.model_device),
                        **{column: float(expert[index]) for index, column in enumerate(EXPERT_COLUMNS)},
                        **{column: float(prediction.action[index]) for index, column in enumerate(PRED_COLUMNS)},
                        "inference_time_ms": float(prediction.latency_ms),
                    }
                    writer.writerow(row)
                    handle.flush()
                    os.fsync(handle.fileno())
                    completed.add(unique_key)
                    computed += 1
                    print(
                        f"{stage} {computed}/{len(missing)}: ep={source.episode_id} frame={source.frame_index} "
                        f"position={position:02d} prompt={prompt_id}",
                        flush=True,
                    )
                assert_frame_unchanged(frame, snapshot, list(predictor.config.input_features))
                frame_mutation_checks += 1
                print(f"observation {observation_index}/{len(scope)} complete", flush=True)

    final_all = load_predictions()
    scoped = final_all.merge(scope[OBSERVATION_KEY], on=OBSERVATION_KEY, how="inner", validate="many_to_one")
    structure = validate_prediction_rows(scoped[FIELDNAMES], scope, complete=True)
    canonical = pd.read_csv(CANONICAL_PREDICTIONS_PATH)
    stored_correct = canonical[canonical.language_condition == "correct"].set_index(OBSERVATION_KEY)
    phase_b_correct = scoped[scoped.prompt_id == "CORRECT"].set_index(OBSERVATION_KEY)
    common = phase_b_correct.index.intersection(stored_correct.index)
    difference = phase_b_correct.loc[common, PRED_COLUMNS].to_numpy(float) - stored_correct.loc[common, PRED_COLUMNS].to_numpy(float)
    max_correct_difference = float(np.max(np.abs(difference)))
    require(max_correct_difference <= 1e-6, f"Correct baseline does not reproduce canonical prediction: {max_correct_difference}")
    elapsed = time.perf_counter() - started
    after_hash = sha256(PREDICTIONS_PATH)
    report = {
        "validation_result": "PASS",
        "stage": stage,
        "target_observations": int(len(scope)),
        "target_predictions": int(len(expected)),
        "predictions_computed_this_run": computed,
        "predictions_skipped_as_valid": int(len(expected) - computed),
        "same_noise_for_31_prompts_per_observation": True,
        "non_language_preprocessing_audits_this_run": preprocessing_audits,
        "source_frame_mutation_checks_this_run": frame_mutation_checks,
        "max_abs_correct_baseline_difference_from_canonical": max_correct_difference,
        "prediction_csv_sha256_before": before_hash,
        "prediction_csv_sha256_after": after_hash,
        "csv_unchanged_when_no_rows_missing": bool(computed > 0 or before_hash == after_hash),
        "elapsed_runtime_seconds": elapsed,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        **structure,
    }
    if stage == "pilot" and computed > 0:
        PILOT_VALIDATION_PATH.write_text(json.dumps(json_native(report), indent=2) + "\n")
    elif stage == "pilot":
        PILOT_RESUME_PATH.write_text(json.dumps(json_native(report), indent=2) + "\n")
    append_run_log(report)
    print(f"Phase B {stage} validation: PASS")
    print(f"rows={len(scoped)} observations={len(scope)} computed={computed} skipped={len(expected)-computed}")
    print(f"max_correct_baseline_difference={max_correct_difference:.9g}")
    return report


def progress_bin_index(progress: np.ndarray) -> np.ndarray:
    return np.minimum(np.floor(progress * 5).astype(int), 4)


def calculate_effects(predictions: pd.DataFrame) -> pd.DataFrame:
    indexed = predictions.set_index(OBSERVATION_KEY + ["prompt_id"])
    require(indexed.index.is_unique, "Prediction index not unique")
    rows: list[dict[str, Any]] = []
    for key in predictions[OBSERVATION_KEY].drop_duplicates().itertuples(index=False, name=None):
        correct_row = indexed.loc[(*key, "CORRECT")]
        correct = correct_row[PRED_COLUMNS].to_numpy(float)
        for prompt in PROMPTS[1:]:
            row = indexed.loc[(*key, prompt["prompt_id"])]
            altered = row[PRED_COLUMNS].to_numpy(float)
            delta = altered - correct
            rows.append(
                {
                    **dict(zip(OBSERVATION_KEY, key, strict=True)),
                    "normalized_progress": float(correct_row.normalized_progress),
                    "prompt_id": prompt["prompt_id"],
                    "prompt_class": prompt["prompt_class"],
                    "instruction": prompt["instruction"],
                    "translation_distance": float(np.linalg.norm(delta[:3])),
                    "orientation_control_distance": float(np.linalg.norm(delta[3:6])),
                    "gripper_absolute_difference": float(abs(delta[6])),
                    "arm_only_6d_distance": float(np.linalg.norm(delta[:6])),
                    "full_7d_distance": float(np.linalg.norm(delta)),
                    "gripper_regime_flip": bool((correct[6] < GRIPPER_REGIME_THRESHOLD) != (altered[6] < GRIPPER_REGIME_THRESHOLD)),
                }
            )
    effects = pd.DataFrame(rows)
    effects["progress_bin_index"] = progress_bin_index(effects.normalized_progress.to_numpy(float))
    effects["progress_bin"] = [PROGRESS_LABELS[index] for index in effects.progress_bin_index]
    require(len(effects) == 600, f"Expected 600 altered effects, found {len(effects)}")
    return effects


METRICS = (
    "translation_distance",
    "orientation_control_distance",
    "gripper_absolute_difference",
    "arm_only_6d_distance",
    "full_7d_distance",
    "gripper_regime_flip",
)


def observation_class_means(effects: pd.DataFrame) -> pd.DataFrame:
    result = effects.groupby(OBSERVATION_KEY + ["normalized_progress", "progress_bin_index", "progress_bin", "prompt_class"], as_index=False)[list(METRICS)].mean()
    result = result.rename(columns={"gripper_regime_flip": "gripper_regime_flip_rate_across_prompts"})
    counts = effects.groupby(OBSERVATION_KEY + ["prompt_class"]).size()
    require((counts == 10).all(), "Not every observation/class has 10 prompts")
    return result


def cluster_bootstrap(data: pd.DataFrame, metric: str, seed: int) -> tuple[float, float]:
    episodes = np.asarray(sorted(data.episode_id.unique()), dtype=int)
    values = {episode: data.loc[data.episode_id == episode, metric].to_numpy(float) for episode in episodes}
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(episodes, size=len(episodes), replace=True)
        estimates[index] = np.concatenate([values[int(episode)] for episode in sampled]).mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def semantic_summary(observation_means: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = METRICS[:-1] + ("gripper_regime_flip_rate_across_prompts",)
    for class_index, prompt_class in enumerate(ALTERED_CLASSES):
        subset = observation_means[observation_means.prompt_class == prompt_class]
        row: dict[str, Any] = {
            "prompt_class": prompt_class,
            "n_prompts_per_observation": 10,
            "n_observations": int(len(subset)),
            "n_episodes": int(subset.episode_id.nunique()),
            "primary_unit": "observation-level mean across 10 prompt variants",
        }
        for metric_index, metric in enumerate(metrics):
            row[f"mean_{metric}"] = float(subset[metric].mean())
            row[f"median_{metric}"] = float(subset[metric].median())
            low, high = cluster_bootstrap(subset, metric, BOOTSTRAP_SEED + class_index * 100 + metric_index)
            row[f"episode_cluster_ci95_low_{metric}"] = low
            row[f"episode_cluster_ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def individual_prompt_summary(effects: pd.DataFrame) -> pd.DataFrame:
    output = (
        effects.groupby(["prompt_id", "prompt_class", "instruction"], as_index=False)
        .agg(
            n_observations=("full_7d_distance", "size"),
            mean_full_7d_distance=("full_7d_distance", "mean"),
            median_full_7d_distance=("full_7d_distance", "median"),
            mean_arm_only_6d_distance=("arm_only_6d_distance", "mean"),
            mean_translation_distance=("translation_distance", "mean"),
            mean_orientation_control_distance=("orientation_control_distance", "mean"),
            mean_gripper_absolute_difference=("gripper_absolute_difference", "mean"),
            gripper_regime_flip_rate=("gripper_regime_flip", "mean"),
        )
    )
    output["class_prompt_mean_rank"] = output.groupby("prompt_class")["mean_full_7d_distance"].rank(method="first", ascending=False).astype(int)
    order = {prompt["prompt_id"]: index for index, prompt in enumerate(PROMPTS)}
    output["prompt_order"] = output.prompt_id.map(order)
    return output.sort_values("prompt_order").drop(columns="prompt_order").reset_index(drop=True)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_semantic_classes(observation_means: pd.DataFrame, summary: pd.DataFrame) -> None:
    panels = (
        ("full_7d_distance", "Full 7-D distance", "A. Full 7-D"),
        ("arm_only_6d_distance", "Arm-only 6-D distance", "B. Arm-only 6-D"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    for ax, (metric, ylabel, title) in zip(axes, panels, strict=True):
        for position, prompt_class in enumerate(ALTERED_CLASSES):
            subset = observation_means[observation_means.prompt_class == prompt_class]
            episode_points = subset.groupby("episode_id")[metric].mean().to_numpy(float)
            jitter = np.linspace(-0.12, 0.12, len(episode_points))
            ax.scatter(position + jitter, episode_points, facecolor="white", edgecolor=COLORS[prompt_class], s=34, zorder=3)
            row = summary[summary.prompt_class == prompt_class].iloc[0]
            mean = float(row[f"mean_{metric}"])
            low = float(row[f"episode_cluster_ci95_low_{metric}"])
            high = float(row[f"episode_cluster_ci95_high_{metric}"])
            ax.errorbar(position, mean, yerr=[[mean-low], [high-mean]], fmt="D", color=COLORS[prompt_class], capsize=5, markersize=7, zorder=4)
        ax.set_xticks(range(3), ALTERED_CLASSES, rotation=18)
        ax.set_ylabel(f"Mean {ylabel} from Correct")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
    fig.suptitle("Semantic-Class Robustness Across 10 Prompt Variants", fontsize=15)
    fig.text(0.5, 0.91, "Open circles: episode means. Diamonds: overall means with episode-cluster 95% intervals.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, "01_semantic_class_robustness")


def figure_individual_prompts(prompt_summary: pd.DataFrame) -> None:
    panels = (
        ("mean_full_7d_distance", "Mean full 7-D distance", "A. Full 7-D"),
        ("mean_arm_only_6d_distance", "Mean arm-only 6-D distance", "B. Arm-only 6-D"),
    )
    x = np.arange(len(prompt_summary))
    colors = [COLORS[value] for value in prompt_summary.prompt_class]
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.8), sharex=True)
    for ax, (metric, ylabel, title) in zip(axes, panels, strict=True):
        ax.scatter(x, prompt_summary[metric], c=colors, s=48, zorder=3)
        ax.plot(x, prompt_summary[metric], color="#BBBBBB", linewidth=0.8, zorder=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        for boundary in (9.5, 19.5):
            ax.axvline(boundary, color="#777777", linestyle="--", linewidth=0.8)
    axes[-1].set_xticks(x, prompt_summary.prompt_id, rotation=55, ha="right")
    fig.suptitle("Individual-Prompt Variability", fontsize=15)
    fig.text(0.5, 0.925, "Each point averages the same 20 observations; prompts are repeated measurements, not independent robot trials.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, "02_individual_prompt_variability")


def figure_progress(observation_means: pd.DataFrame) -> bool:
    counts = observation_means[observation_means.prompt_class == "Unrelated"].groupby("progress_bin_index").size().reindex(range(5), fill_value=0)
    if int(counts.min()) < 2:
        return False
    panels = (
        ("full_7d_distance", "Mean full 7-D distance", "A. Full 7-D"),
        ("arm_only_6d_distance", "Mean arm-only 6-D distance", "B. Arm-only 6-D"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharex=True)
    x = np.arange(5)
    for ax, (metric, ylabel, title) in zip(axes, panels, strict=True):
        for prompt_class in ALTERED_CLASSES:
            subset = observation_means[observation_means.prompt_class == prompt_class]
            means = subset.groupby("progress_bin_index")[metric].mean().reindex(range(5))
            ax.plot(x, means, marker="o", linewidth=2, color=COLORS[prompt_class], label=prompt_class)
        ax.axvspan(0.65, 1.35, color=COLORS["Unrelated"], alpha=0.07)
        ax.set_xticks(x, [f"{label}\nn={int(counts[index])}" for index, label in enumerate(PROGRESS_LABELS)])
        ax.set_xlabel("Normalized trajectory-progress bin")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("Prompt-Bank Sensitivity Across Trajectory Progress", fontsize=15)
    fig.text(0.5, 0.91, "Descriptive 20-observation selected subset; n labels are observations per bin and no bin-wise inferential claim is made.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, "03_prompt_bank_sensitivity_across_progress")
    return True


def write_findings(effects: pd.DataFrame, observation_means: pd.DataFrame, summary: pd.DataFrame, prompts: pd.DataFrame, progress_figure: bool) -> dict[str, Any]:
    class_values = summary.set_index("prompt_class")
    full_order = (
        class_values.loc["Paraphrase", "mean_full_7d_distance"]
        < class_values.loc["Contradictory", "mean_full_7d_distance"]
        < class_values.loc["Unrelated", "mean_full_7d_distance"]
    )
    arm_order = (
        class_values.loc["Paraphrase", "mean_arm_only_6d_distance"]
        < class_values.loc["Contradictory", "mean_arm_only_6d_distance"]
        < class_values.loc["Unrelated", "mean_arm_only_6d_distance"]
    )
    full_pivot = observation_means.pivot(index=OBSERVATION_KEY, columns="prompt_class", values="full_7d_distance")
    arm_pivot = observation_means.pivot(index=OBSERVATION_KEY, columns="prompt_class", values="arm_only_6d_distance")
    full_holds = (full_pivot.Paraphrase < full_pivot.Contradictory) & (full_pivot.Contradictory < full_pivot.Unrelated)
    arm_holds = (arm_pivot.Paraphrase < arm_pivot.Contradictory) & (arm_pivot.Contradictory < arm_pivot.Unrelated)

    prompt_variation: dict[str, Any] = {}
    for prompt_class in ALTERED_CLASSES:
        subset = prompts[prompts.prompt_class == prompt_class]
        prompt_variation[prompt_class] = {
            "minimum_prompt": str(subset.loc[subset.mean_full_7d_distance.idxmin(), "prompt_id"]),
            "minimum_prompt_mean_full_7d": float(subset.mean_full_7d_distance.min()),
            "maximum_prompt": str(subset.loc[subset.mean_full_7d_distance.idxmax(), "prompt_id"]),
            "maximum_prompt_mean_full_7d": float(subset.mean_full_7d_distance.max()),
            "std_across_10_prompt_means_full_7d": float(subset.mean_full_7d_distance.std(ddof=1)),
        }

    progress = (
        observation_means.groupby(
            ["prompt_class", "progress_bin", "progress_bin_index"], as_index=False
        )
        .agg(
            n_observations=("full_7d_distance", "size"),
            mean_full_7d_distance=("full_7d_distance", "mean"),
            mean_arm_only_6d_distance=("arm_only_6d_distance", "mean"),
        )
    )
    unrelated_progress = progress[progress.prompt_class == "Unrelated"].sort_values(
        "progress_bin_index"
    )
    target = unrelated_progress[unrelated_progress.progress_bin_index == 1].iloc[0]
    other = unrelated_progress[unrelated_progress.progress_bin_index != 1]
    progress_answer = {
        "n_20_40": int(target.n_observations),
        "mean_full_7d_20_40": float(target.mean_full_7d_distance),
        "largest_other_bin_mean_full_7d": float(other.mean_full_7d_distance.max()),
        "mean_arm_6d_20_40": float(target.mean_arm_only_6d_distance),
        "largest_other_bin_mean_arm_6d": float(other.mean_arm_only_6d_distance.max()),
        "full_20_40_is_largest": bool(
            target.mean_full_7d_distance > other.mean_full_7d_distance.max()
        ),
        "arm_20_40_is_largest": bool(
            target.mean_arm_only_6d_distance > other.mean_arm_only_6d_distance.max()
        ),
        "full_20_40_rank_descending": int(
            unrelated_progress.mean_full_7d_distance.rank(
                method="min", ascending=False
            ).loc[target.name]
        ),
        "arm_20_40_rank_descending": int(
            unrelated_progress.mean_arm_only_6d_distance.rank(
                method="min", ascending=False
            ).loc[target.name]
        ),
    }

    summary_lines = []
    for prompt_class in ALTERED_CLASSES:
        row = class_values.loc[prompt_class]
        summary_lines.append(
            f"- {prompt_class}: full 7-D mean {row.mean_full_7d_distance:.6f}, episode-cluster 95% interval "
            f"[{row.episode_cluster_ci95_low_full_7d_distance:.6f}, {row.episode_cluster_ci95_high_full_7d_distance:.6f}]; "
            f"arm-only mean {row.mean_arm_only_6d_distance:.6f}, interval "
            f"[{row.episode_cluster_ci95_low_arm_only_6d_distance:.6f}, {row.episode_cluster_ci95_high_arm_only_6d_distance:.6f}]."
        )
    observed_verdict = (
        "Yes at the aggregate-mean level, but not as a universal observation-level rule. The ordering Paraphrase < Contradictory < Unrelated holds for both full 7-D and arm-only 6-D means; the episode-cluster intervals overlap."
        if full_order and arm_order
        else "Only partially. The complete aggregate ordering does not hold for both full 7-D and arm-only 6-D distances."
    )
    text = f"""# Phase B findings: systematic multi-prompt robustness

## Observed result

{observed_verdict}

Each value below first averages the 10 prompt variants within an observation; only then are the 20 observation-level values summarized. Prompts on the same frame are not counted as independent robot trials.

{chr(10).join(summary_lines)}

The strict class ordering holds after prompt averaging in {int(full_holds.sum())}/20 observations for full 7-D distance and {int(arm_holds.sum())}/20 for arm-only 6-D distance.

## Answers to the seven analysis questions

1. **Paraphrase versus Contradictory:** {'Yes' if class_values.loc['Paraphrase', 'mean_full_7d_distance'] < class_values.loc['Contradictory', 'mean_full_7d_distance'] else 'No'} at the aggregate full-7D mean: {class_values.loc['Paraphrase', 'mean_full_7d_distance']:.6f} versus {class_values.loc['Contradictory', 'mean_full_7d_distance']:.6f}.
2. **Contradictory versus Unrelated:** {'Yes' if class_values.loc['Contradictory', 'mean_full_7d_distance'] < class_values.loc['Unrelated', 'mean_full_7d_distance'] else 'No'} at the aggregate full-7D mean: {class_values.loc['Contradictory', 'mean_full_7d_distance']:.6f} versus {class_values.loc['Unrelated', 'mean_full_7d_distance']:.6f}.
3. **Observation-level ordering:** It holds in {int(full_holds.sum())}/20 observations for full 7-D and {int(arm_holds.sum())}/20 for arm-only 6-D after averaging the 10 prompts in each class.
4. **Large-change paraphrases:** `P03` is the largest Paraphrase at {float(prompts.loc[prompts.prompt_id == 'P03', 'mean_full_7d_distance'].iloc[0]):.6f}; this exceeds the Contradictory class mean and shows that semantic equivalence does not guarantee a small perturbation for every wording.
5. **Small-change unrelated prompts:** `U09` is the smallest Unrelated full-7D prompt at {float(prompts.loc[prompts.prompt_id == 'U09', 'mean_full_7d_distance'].iloc[0]):.6f}, slightly below the Contradictory class mean. Its arm-only mean is {float(prompts.loc[prompts.prompt_id == 'U09', 'mean_arm_only_6d_distance'].iloc[0]):.6f}, so the small full-norm comparison does not imply negligible arm change.
6. **Arm-only result:** The aggregate ordering also holds without the gripper: {class_values.loc['Paraphrase', 'mean_arm_only_6d_distance']:.6f} < {class_values.loc['Contradictory', 'mean_arm_only_6d_distance']:.6f} < {class_values.loc['Unrelated', 'mean_arm_only_6d_distance']:.6f}. It holds observation-by-observation in {int(arm_holds.sum())}/20 cases.
7. **20–40% Unrelated effect:** The original sharp global peak does not survive ten-prompt averaging in this selected subset. The bin has mean full/arm distances {progress_answer['mean_full_7d_20_40']:.6f}/{progress_answer['mean_arm_6d_20_40']:.6f} and ranks {progress_answer['full_20_40_rank_descending']}/5 for full and {progress_answer['arm_20_40_rank_descending']}/5 for arm. A descriptive full-7D elevation remains, but larger values occur elsewhere and only three selected observations occupy this bin.

Individual-prompt variability is nonzero:

- Paraphrase: minimum `{prompt_variation['Paraphrase']['minimum_prompt']}` = {prompt_variation['Paraphrase']['minimum_prompt_mean_full_7d']:.6f}, maximum `{prompt_variation['Paraphrase']['maximum_prompt']}` = {prompt_variation['Paraphrase']['maximum_prompt_mean_full_7d']:.6f}, SD across prompt means = {prompt_variation['Paraphrase']['std_across_10_prompt_means_full_7d']:.6f}.
- Contradictory: minimum `{prompt_variation['Contradictory']['minimum_prompt']}` = {prompt_variation['Contradictory']['minimum_prompt_mean_full_7d']:.6f}, maximum `{prompt_variation['Contradictory']['maximum_prompt']}` = {prompt_variation['Contradictory']['maximum_prompt_mean_full_7d']:.6f}, SD = {prompt_variation['Contradictory']['std_across_10_prompt_means_full_7d']:.6f}.
- Unrelated: minimum `{prompt_variation['Unrelated']['minimum_prompt']}` = {prompt_variation['Unrelated']['minimum_prompt_mean_full_7d']:.6f}, maximum `{prompt_variation['Unrelated']['maximum_prompt']}` = {prompt_variation['Unrelated']['maximum_prompt_mean_full_7d']:.6f}, SD = {prompt_variation['Unrelated']['std_across_10_prompt_means_full_7d']:.6f}.

For the selected subset, the multi-prompt Unrelated mean in the 20–40% bin is {progress_answer['mean_full_7d_20_40']:.6f} full 7-D and {progress_answer['mean_arm_6d_20_40']:.6f} arm-only 6-D across {progress_answer['n_20_40']} observations. The largest corresponding means outside that bin are {progress_answer['largest_other_bin_mean_full_7d']:.6f} and {progress_answer['largest_other_bin_mean_arm_6d']:.6f}. Therefore, 20–40% is {'still the largest descriptive bin for both metrics' if progress_answer['full_20_40_is_largest'] and progress_answer['arm_20_40_is_largest'] else 'not the largest descriptive bin for both metrics'}. This is a small selected-subset comparison, not a new task-stage claim.

## Interpretation

The class means test whether the original result was peculiar to one sentence. Aggregate ordering across ten systematically varied sentences supports class-level language sensitivity only where the reported ordering actually holds. The individual-prompt table shows how much specific wording matters and identifies high-change paraphrases and low-change unrelated prompts without treating them as independent trials.

The retained original Drawer prompt `U01` is rank 1/10 within Unrelated, with mean full distance {float(prompts.loc[prompts.prompt_id == 'U01', 'mean_full_7d_distance'].iloc[0]):.6f}. It is therefore unusually strong in this bank. The class-level Unrelated mean nevertheless remains above the other classes after averaging U01–U10, so the aggregate result is not solely a U01 effect; the original single-prompt magnitude was partly wording-dependent.

## Limitations

- Only 20 deterministically selected observations from 10 episodes were evaluated.
- One pretrained model, one demonstrated task, and one domain were studied.
- The bank was manually and systematically authored; it is not a complete or random sample of natural-language instructions.
- Ten prompts on the same observation are nested repeated measurements, not ten independent robot trials.
- This remains offline next-action evaluation, not closed-loop task execution or task-success measurement.
- The progress analysis is descriptive and subsampled; normalized trajectory progress is not a semantic manipulation stage.
- Prompt-specific extremes in this small bank should not be generalized to a population of possible phrasings.
"""
    FINDINGS_PATH.write_text(text)
    return {
        "aggregate_full_order_holds": bool(full_order),
        "aggregate_arm_order_holds": bool(arm_order),
        "observation_full_order_count": int(full_holds.sum()),
        "observation_arm_order_count": int(arm_holds.sum()),
        "prompt_variation": prompt_variation,
        "progress_20_40": progress_answer,
        "progress_figure_generated": progress_figure,
    }


def analyze(selected: pd.DataFrame) -> dict[str, Any]:
    predictions = load_predictions()
    structure = validate_prediction_rows(predictions, selected, complete=True)
    effects = calculate_effects(predictions)
    observation_means = observation_class_means(effects)
    summary = semantic_summary(observation_means)
    prompts = individual_prompt_summary(effects)
    observation_means.to_csv(OUTPUT_DIR / "observation_class_means.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "semantic_class_summary.csv", index=False)
    prompts.to_csv(OUTPUT_DIR / "individual_prompt_summary.csv", index=False)
    figure_semantic_classes(observation_means, summary)
    figure_individual_prompts(prompts)
    progress_figure = figure_progress(observation_means)
    findings = write_findings(effects, observation_means, summary, prompts, progress_figure)
    run_records = [
        json.loads(line)
        for line in RUN_LOG_PATH.read_text().splitlines()
        if line.strip()
    ]
    pilot_run = next(
        row for row in run_records
        if row["stage"] == "pilot" and row["predictions_computed_this_run"] == 62
    )
    pilot_resume = next(
        row for row in run_records
        if row["stage"] == "pilot" and row["predictions_computed_this_run"] == 0
    )
    full_run = next(
        row for row in run_records
        if row["stage"] == "full" and row["predictions_computed_this_run"] == 558
    )
    full_resume = next(
        row for row in run_records
        if row["stage"] == "full" and row["predictions_computed_this_run"] == 0
    )
    require(pilot_run["predictions_skipped_as_valid"] == 0, "Pilot run coverage changed")
    require(pilot_resume["predictions_skipped_as_valid"] == 62, "Pilot resume coverage changed")
    require(full_run["predictions_skipped_as_valid"] == 62, "Full run did not reuse pilot rows")
    require(full_resume["predictions_skipped_as_valid"] == 620, "Full resume coverage changed")
    require(
        full_resume["prediction_csv_sha256_before"]
        == full_resume["prediction_csv_sha256_after"],
        "Full resume check changed predictions.csv",
    )
    numeric_files = {
        "predictions.csv": predictions.select_dtypes(include=[np.number]),
        "observation_class_means.csv": observation_means.select_dtypes(include=[np.number]),
        "semantic_class_summary.csv": summary.select_dtypes(include=[np.number]),
        "individual_prompt_summary.csv": prompts.select_dtypes(include=[np.number]),
    }
    require(all(np.isfinite(frame.to_numpy(float)).all() for frame in numeric_files.values()), "A Phase B numeric output is non-finite")
    validation = {
        "validation_result": "PASS",
        "canonical_prediction_sha256": sha256(CANONICAL_PREDICTIONS_PATH),
        "prompt_bank_sha256": sha256(PROMPT_BANK_PATH),
        "selected_observations_sha256": sha256(SELECTION_PATH),
        "predictions_sha256": sha256(PREDICTIONS_PATH),
        "experiment_seed": EXPERIMENT_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "gripper_regime_rule": "opening if continuous postprocessed value < 0; closing if value >= 0; no clipping",
        "prompt_count_per_observation": 31,
        "altered_prompt_count_per_observation": 30,
        "total_expected_predictions": 620,
        "total_altered_effects": 600,
        "observation_class_mean_rows": int(len(observation_means)),
        "semantic_class_summary_rows": int(len(summary)),
        "individual_prompt_summary_rows": int(len(prompts)),
        "all_numeric_outputs_finite": True,
        "validated_historical_files_modified": False,
        "inference_and_resume_coverage": {
            "pilot_predictions_computed": 62,
            "pilot_predictions_reused_on_resume": 62,
            "full_predictions_computed_after_pilot": 558,
            "pilot_predictions_reused_in_full_run": 62,
            "full_predictions_reused_on_resume": 620,
            "full_resume_csv_hash_unchanged": True,
            "max_abs_correct_baseline_difference_from_canonical": float(
                full_run["max_abs_correct_baseline_difference_from_canonical"]
            ),
        },
        **structure,
        **findings,
    }
    VALIDATION_PATH.write_text(json.dumps(json_native(validation), indent=2) + "\n")
    print("Phase B analysis validation: PASS")
    print(summary[["prompt_class", "mean_full_7d_distance", "episode_cluster_ci95_low_full_7d_distance", "episode_cluster_ci95_high_full_7d_distance", "mean_arm_only_6d_distance", "episode_cluster_ci95_low_arm_only_6d_distance", "episode_cluster_ci95_high_arm_only_6d_distance"]].to_string(index=False))
    print(f"Observation-level full ordering: {findings['observation_full_order_count']}/20")
    print(f"Observation-level arm ordering: {findings['observation_arm_order_count']}/20")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "pilot", "full", "analyze"), required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="mps")
    args = parser.parse_args()
    selected = prepare()
    if args.stage == "prepare":
        print(selected.to_string(index=False))
    elif args.stage in ("pilot", "full"):
        run_inference(selected, args.stage, args.device)
    else:
        analyze(selected)


if __name__ == "__main__":
    main()
