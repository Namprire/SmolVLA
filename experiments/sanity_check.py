#!/usr/bin/env python
"""Hard-gate smoke test: one real frame to one postprocessed 7-D action."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import DATASET_ID, load_smoke_dataset  # noqa: E402

MODEL_ID = "HuggingFaceVLA/smolvla_libero"
RAW_DATASET_ROOT = REPO_ROOT / ".cache/huggingface/lerobot_v21" / DATASET_ID
CONVERTED_DATASET_ROOT = REPO_ROOT / ".cache/huggingface/lerobot_v30_smoke" / DATASET_ID
MODEL_ROOT = REPO_ROOT / ".cache/huggingface/models" / MODEL_ID

SAFETENSORS_TO_TORCH_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "BOOL": torch.bool,
}


def describe(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
    if isinstance(value, np.ndarray):
        return f"shape={value.shape}, dtype={value.dtype}"
    return f"type={type(value).__name__}, value={value!r}"


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
    return torch.device(requested)


def print_environment(device: torch.device) -> None:
    print("=== ENVIRONMENT ===")
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"macOS/platform: {platform.platform()}")
    print(f"torch version: {torch.__version__}")
    print(f"lerobot version: {importlib.metadata.version('lerobot')}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS availability: {torch.backends.mps.is_available()}")
    print(f"Selected device: {device}")
    print(f"PYTORCH_ENABLE_MPS_FALLBACK: {__import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0')}")


def restore_and_verify_checkpoint_dtypes(policy: SmolVLAPolicy, checkpoint: Path) -> dict[str, int]:
    """Restore stored dtypes after config-only construction and verify them."""
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        stored_dtypes = {key: handle.get_slice(key).get_dtype() for key in handle.keys()}

    live_state = policy.state_dict(keep_vars=True)
    unknown_dtypes = sorted(set(stored_dtypes.values()) - set(SAFETENSORS_TO_TORCH_DTYPE))
    if unknown_dtypes:
        raise TypeError(f"Unsupported checkpoint dtypes: {unknown_dtypes}")

    for key, stored_dtype in stored_dtypes.items():
        if key not in live_state:
            raise KeyError(f"Checkpoint tensor is absent from loaded policy state: {key}")
        target_dtype = SAFETENSORS_TO_TORCH_DTYPE[stored_dtype]
        tensor = live_state[key]
        if tensor.dtype != target_dtype:
            tensor.data = tensor.data.to(dtype=target_dtype)

    mismatches = {
        key: (str(live_state[key].dtype), stored_dtype)
        for key, stored_dtype in stored_dtypes.items()
        if live_state[key].dtype != SAFETENSORS_TO_TORCH_DTYPE[stored_dtype]
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint dtype restoration failed: {mismatches}")

    counts: dict[str, int] = {}
    for stored_dtype in stored_dtypes.values():
        counts[stored_dtype] = counts.get(stored_dtype, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()

    device = select_device(args.device)
    print_environment(device)

    dataset = load_smoke_dataset(RAW_DATASET_ROOT, CONVERTED_DATASET_ROOT)
    if not 0 <= args.frame_index < len(dataset):
        raise IndexError(f"frame-index {args.frame_index} is outside dataset length {len(dataset)}")
    frame = dataset[args.frame_index]

    print("\n=== REAL DATASET SAMPLE ===")
    print(f"Dataset ID: {DATASET_ID}")
    print(f"Dataset length (episode-0 smoke subset): {len(dataset)}")
    print(f"All raw sample keys: {sorted(frame)}")
    for key in dataset.meta.image_keys:
        print(f"Image {key}: {describe(frame[key])}")
    print(f"State shape/values: {tuple(frame['observation.state'].shape)} {frame['observation.state'].tolist()}")
    print(f"Action shape/values: {tuple(frame['action'].shape)} {frame['action'].tolist()}")
    print(f"Actual task text: {frame['task']}")

    config = PreTrainedConfig.from_pretrained(MODEL_ROOT)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = str(device)
    print("\n=== RELEVANT POLICY CONFIG ===")
    for key in (
        "device",
        "input_features",
        "output_features",
        "chunk_size",
        "n_action_steps",
        "max_state_dim",
        "max_action_dim",
        "num_steps",
        "vlm_model_name",
        "tokenizer_max_length",
        "pad_language_to",
        "normalization_mapping",
        "use_amp",
        "load_vlm_weights",
    ):
        print(f"{key}: {getattr(config, key)}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(MODEL_ROOT),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    observation = {key: frame[key] for key in config.input_features}
    observation["task"] = frame["task"]
    processed = preprocessor(observation)

    print("\n=== OFFICIAL PREPROCESSOR OUTPUT ===")
    for key in sorted(processed):
        print(f"Processed {key}: {describe(processed[key])}")
    if OBS_LANGUAGE_TOKENS in processed:
        print(f"Language token IDs: {processed[OBS_LANGUAGE_TOKENS].detach().cpu().tolist()}")
    if OBS_LANGUAGE_ATTENTION_MASK in processed:
        print(
            "Language attention mask: "
            f"{processed[OBS_LANGUAGE_ATTENTION_MASK].detach().cpu().tolist()}"
        )

    print("\n=== REAL POLICY INFERENCE ===")
    started = time.perf_counter()
    # The policy file is self-contained. Avoid downloading 2.03 GB of base VLM
    # initialization weights that this checkpoint immediately overwrites. Strict
    # loading proves every policy state key is supplied by the real checkpoint.
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(MODEL_ROOT, config=config, strict=True)
    dtype_counts = restore_and_verify_checkpoint_dtypes(policy, MODEL_ROOT / "model.safetensors")
    policy.reset()
    model_action = policy.select_action(processed)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    postprocessed_action = postprocessor(model_action).detach().cpu().squeeze(0)
    expert_action = frame["action"].detach().cpu().reshape(-1)

    print(f"Actual inference device: {next(policy.parameters()).device}")
    print(f"Verified checkpoint tensor dtypes: {dtype_counts}")
    print(f"Model-space output: {model_action.detach().cpu().tolist()}")
    print(f"Expert 7-D action: {expert_action.tolist()}")
    print(f"Predicted postprocessed 7-D action: {postprocessed_action.tolist()}")
    print(f"Load + first inference time: {elapsed_ms:.1f} ms")

    assert expert_action.shape == (7,), f"Expected expert action (7,), got {expert_action.shape}"
    assert postprocessed_action.shape == (7,), (
        f"Expected postprocessed action (7,), got {postprocessed_action.shape}"
    )
    assert torch.isfinite(postprocessed_action).all(), "Prediction contains NaN or infinity"
    print("HARD GATE PASS: genuine postprocessed action dimension is 7")


if __name__ == "__main__":
    main()
