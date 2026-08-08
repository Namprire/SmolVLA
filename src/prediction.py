"""Controlled-noise inference API for the real pretrained SmolVLA checkpoint."""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors import safe_open

MODEL_ID = "HuggingFaceVLA/smolvla_libero"

SAFETENSORS_TO_TORCH_DTYPE = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "BOOL": torch.bool,
}


@dataclass(frozen=True)
class Prediction:
    """One final postprocessed action and model-only inference latency."""

    action: np.ndarray
    latency_ms: float


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
    return torch.device(requested)


def restore_and_verify_checkpoint_dtypes(policy: SmolVLAPolicy, checkpoint: Path) -> dict[str, int]:
    """Restore the exact stored dtypes after config-only model construction."""
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        stored_dtypes = {key: handle.get_slice(key).get_dtype() for key in handle.keys()}

    live_state = policy.state_dict(keep_vars=True)
    unknown = sorted(set(stored_dtypes.values()) - set(SAFETENSORS_TO_TORCH_DTYPE))
    if unknown:
        raise TypeError(f"Unsupported checkpoint dtypes: {unknown}")

    for key, stored_dtype in stored_dtypes.items():
        if key not in live_state:
            raise KeyError(f"Checkpoint tensor is absent from loaded policy state: {key}")
        target_dtype = SAFETENSORS_TO_TORCH_DTYPE[stored_dtype]
        if live_state[key].dtype != target_dtype:
            live_state[key].data = live_state[key].data.to(dtype=target_dtype)

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


def _copy_feature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return copy.deepcopy(value)


class SmolVLAPredictor:
    """Load the official checkpoint processors and expose controlled inference."""

    def __init__(self, model_root: Path, device: str = "auto") -> None:
        self.model_root = Path(model_root).resolve()
        self.device = select_device(device)

        config = PreTrainedConfig.from_pretrained(self.model_root)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
        config.device = str(self.device)
        self.config = config

        # Load the serialized official processor pipelines and their checkpoint
        # normalization statistics, changing only the runtime device.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(self.model_root),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

        # This checkpoint is self-contained. Avoid downloading base VLM weights
        # that strict checkpoint loading immediately replaces.
        config.load_vlm_weights = False
        self.policy = SmolVLAPolicy.from_pretrained(self.model_root, config=config, strict=True)
        self.checkpoint_dtype_counts = restore_and_verify_checkpoint_dtypes(
            self.policy, self.model_root / "model.safetensors"
        )
        self.policy.eval()

    @property
    def noise_shape(self) -> tuple[int, int, int]:
        return (1, self.config.chunk_size, self.config.max_action_dim)

    @property
    def model_device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def make_noise(self, seed: int) -> torch.Tensor:
        """Create a reproducible explicit flow-noise tensor for one observation."""
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randn(self.noise_shape, generator=generator, dtype=torch.float32).to(self.model_device)

    def validate_noise(self, noise: torch.Tensor) -> torch.Tensor:
        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"noise must be a torch.Tensor, got {type(noise).__name__}")
        if tuple(noise.shape) != self.noise_shape:
            raise ValueError(f"noise shape must be {self.noise_shape}, got {tuple(noise.shape)}")
        if noise.dtype != torch.float32:
            raise TypeError(f"noise dtype must be torch.float32, got {noise.dtype}")
        if not torch.isfinite(noise).all().item():
            raise ValueError("noise contains NaN or infinity")
        # Accept a reusable CPU tensor for convenience, but always give the
        # installed sampler an isolated float32 tensor on the model/state device.
        return noise.detach().clone().to(device=self.model_device)

    def noise_identifier(self, noise: torch.Tensor) -> str:
        raw = noise.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
        return f"sha256:{hashlib.sha256(raw).hexdigest()[:16]}"

    def tokenize_instruction(self, frame: dict[str, Any], instruction: str) -> tuple[list[int], list[bool]]:
        """Return official processed token IDs/mask for audit documentation."""
        processed = self._preprocess(frame, instruction)
        return (
            processed["observation.language.tokens"].detach().cpu().reshape(-1).tolist(),
            processed["observation.language.attention_mask"]
            .detach()
            .cpu()
            .reshape(-1)
            .bool()
            .tolist(),
        )

    def _preprocess(self, frame: dict[str, Any], instruction: str) -> dict[str, torch.Tensor]:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        missing = [key for key in self.config.input_features if key not in frame]
        if missing:
            raise KeyError(f"Frame is missing checkpoint input features: {missing}")

        # Construct a new mapping containing only checkpoint inputs plus task.
        # Every mutable feature is copied, so neither preprocessing nor policy
        # internals can mutate the dataset-owned frame.
        observation = {key: _copy_feature(frame[key]) for key in self.config.input_features}
        observation["task"] = instruction
        return self.preprocessor(observation)

    def predict_action(
        self,
        frame: dict[str, Any],
        instruction: str,
        noise: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> Prediction:
        """Predict the first postprocessed 7-D action under controlled flow noise.

        Passing ``noise`` is the preferred controlled-intervention path. If only
        ``seed`` is supplied, this method first materializes an explicit tensor
        with :meth:`make_noise`. If both are omitted, the installed sampler draws
        its own noise using its supported default behavior.
        """
        if noise is not None and seed is not None:
            raise ValueError("Pass either explicit noise or seed, not both")

        processed = self._preprocess(frame, instruction)
        if noise is None and seed is not None:
            noise = self.make_noise(seed)
        controlled_noise = None if noise is None else self.validate_noise(noise)

        self.policy.reset()
        if self.device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        action_chunk = self.policy.predict_action_chunk(processed, noise=controlled_noise)
        if self.device.type == "mps":
            torch.mps.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0

        first_model_action = action_chunk[:, 0, :]
        final_action = self.postprocessor(first_model_action).detach().cpu().reshape(-1)
        expected_dim = self.config.output_features["action"].shape[0]
        if final_action.shape != (expected_dim,):
            raise RuntimeError(
                f"Expected postprocessed action shape ({expected_dim},), got {tuple(final_action.shape)}"
            )
        if expected_dim != 7:
            raise RuntimeError(f"This experiment requires a 7-D checkpoint action, got {expected_dim}")
        if not torch.isfinite(final_action).all().item():
            raise RuntimeError("Postprocessed action contains NaN or infinity")
        return Prediction(action=final_action.numpy().copy(), latency_ms=latency_ms)


_DEFAULT_PREDICTOR: SmolVLAPredictor | None = None


def configure_default_predictor(model_root: Path, device: str = "auto") -> SmolVLAPredictor:
    """Configure the module-level API once for scripts or interactive use."""
    global _DEFAULT_PREDICTOR
    _DEFAULT_PREDICTOR = SmolVLAPredictor(model_root=model_root, device=device)
    return _DEFAULT_PREDICTOR


def predict_action(
    frame: dict[str, Any],
    instruction: str,
    noise: torch.Tensor | None = None,
    seed: int | None = None,
) -> Prediction:
    """Module-level API with the requested four-argument prediction signature."""
    if _DEFAULT_PREDICTOR is None:
        raise RuntimeError("Call configure_default_predictor(model_root, device) before predict_action")
    return _DEFAULT_PREDICTOR.predict_action(frame, instruction, noise=noise, seed=seed)
