# From Language to Reliable Action

Evaluating and Guarding SmolVLA for Packing-Style Robotic Manipulation.

This repository contains the verified Phase 1-2 setup/hard gate and the Phase 3-6 controlled-language pilot. It evaluates the pretrained `HuggingFaceVLA/smolvla_libero` checkpoint; it does not train or fine-tune SmolVLA.

## Environment

The verified environment uses stable CPython 3.11 on Apple silicon:

```bash
UV_CACHE_DIR=/private/tmp/smolvla-uv-cache uv venv --python 3.11 .venv
UV_CACHE_DIR=/private/tmp/smolvla-uv-cache uv pip install --python .venv/bin/python -r requirements.txt
```

Verified versions: Python 3.11.15, LeRobot 0.4.4, PyTorch 2.10.0,
torchvision 0.25.0, Transformers 4.57.6, NumPy 2.4.6, pandas 3.0.5,
SciPy 1.17.1, and Matplotlib 3.11.1. The complete record is in
`results/environment.txt`.

## Required hard-gate downloads

Keep Hugging Face caches inside the repository's ignored `.cache/` directory:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

.venv/bin/hf download HuggingFaceVLA/smolvla_libero \
  --include 'config.json' 'model.safetensors' \
  'policy_preprocessor.json' 'policy_postprocessor.json' \
  'policy_preprocessor_step_5_normalizer_processor.safetensors' \
  'policy_postprocessor_step_1_unnormalizer_processor.safetensors' \
  --local-dir .cache/huggingface/models/HuggingFaceVLA/smolvla_libero

.venv/bin/hf download HuggingFaceVLA/smol-libero \
  --repo-type dataset --revision v2.1 \
  --include 'meta/*' 'data/chunk-000/episode_000000.parquet' \
  --local-dir .cache/huggingface/lerobot_v21/HuggingFaceVLA/smol-libero
```

The dataset repository currently exposes only a `v2.1` tag. Installed LeRobot 0.4.4 uses the v3.0 dataset schema, so the smoke test creates an ignored, local v3.0 view of episode 0 using converter functions bundled with that installed release. It never modifies or uploads the source dataset.

## One real postprocessed action

Run on the host so PyTorch can see Metal Performance Shaders:

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python experiments/sanity_check.py --device mps
```

The script prints environment/device information, raw sample keys and values, exact dataset task text, checkpoint configuration, official preprocessor tokens/shapes, the recorded expert action, and the real postprocessed model action. It asserts that the final action is finite and seven-dimensional.

The checkpoint is self-contained. To avoid a redundant 2.03 GB base-VLM
initialization download, the smoke test constructs the architecture from the
checkpoint-specified VLM config, loads `model.safetensors` with `strict=True`,
then restores and verifies every stored tensor dtype (746 BF16 and 42 FP32 in
the inspected checkpoint) before inference. A missing/unexpected state key or
dtype mismatch is a hard failure.

If a specific MPS operation fails, preserve the traceback before testing PyTorch's targeted fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python experiments/sanity_check.py --device mps
```

Use `--device cpu` only if the documented MPS attempts fail. Do not describe a fallback run as pure MPS execution.

## Controlled language intervention (Phases 3-6)

`src/prediction.py` exposes the clean controlled API:

```python
predict_action(frame, instruction, noise=None, seed=None)
```

It copies only the checkpoint input features from the real dataset frame, supplies the requested string as `task`, runs the official serialized checkpoint preprocessor and postprocessor, and returns a finite postprocessed 7-D action with model-only inference latency. Explicit reusable flow noise is preferred. The installed LeRobot 0.4.4 path is `SmolVLAPolicy.predict_action_chunk(batch, noise=...)`; for a single observation this checkpoint requires a `(1, 50, 32)` `torch.float32` tensor on the inference device.

The small pilot uses episodes 0-2 only. Download the two additional parquet files:

```bash
.venv/bin/hf download HuggingFaceVLA/smol-libero \
  --repo-type dataset --revision v2.1 \
  --include 'data/chunk-000/episode_000001.parquet' \
  'data/chunk-000/episode_000002.parquet' \
  --local-dir .cache/huggingface/lerobot_v21/HuggingFaceVLA/smol-libero
```

Then run with MPS fallback disabled:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 \
HF_HOME="$PWD/.cache/huggingface" \
.venv/bin/python experiments/language_control.py --device mps
```

The script gates all language substitutions on five repeated, explicit-noise deterministic predictions. It then writes the four-condition single-frame results and a 9-observation/36-row early-middle-late pilot. It validates exact strings, finite 7-D actions, normalized progress, condition coverage, noise reuse, preprocessing equality outside language fields, and duplicate absence. See `results/api_inspection.md`, `results/determinism_check.csv`, `results/language_conditions.json`, `results/language_single_frame.csv`, `results/vla_predictions_pilot.csv`, and `results/language_pilot_summary.json`.

The Phase 3-6 driver stops after Phase 6. Those outputs prove the controlled intervention pipeline and provide a small pilot only; they are not a full experiment or a scientific conclusion.

## Main controlled language experiment (Phase 7)

The Phase 7 driver uses the unchanged controlled prediction API and is staged and resumable. Stage A covers episodes 0-1; Stage B resumes the same CSV and extends it through episode 9:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HOME="$PWD/.cache/huggingface" \
  .venv/bin/python experiments/main_language_experiment.py --stage a --device mps

PYTORCH_ENABLE_MPS_FALLBACK=0 HF_HOME="$PWD/.cache/huggingface" \
  .venv/bin/python experiments/main_language_experiment.py --stage b --device mps
```

The deterministic selection contains 20 unique, stratified observations per episode and forces the real start/end frames into every episode. `results/main_experiment_selection.json` records all local/global indices, task progress values, selection seeds, noise seeds, and noise hashes.

`results/vla_predictions.csv` is append-only during inference. On resume, the driver validates its complete schema and every existing uniqueness key, instruction, progress value, seed/hash, device, action, and timing before computing only missing rows. The validated Phase 7 run contains 200 observations and 800 predictions across episodes 0-9. Integrity details are in `results/main_experiment_validation.json`.

Phase 7 stops after data generation and integrity validation. It does not calculate language-effect metrics or scientific conclusions.

## Paired metrics and task-stage sensitivity (Phases 8-9)

The Phase 8-9 analysis is deterministic, model-free, and reads the validated Phase 7 CSV without changing it:

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
  .venv/bin/python experiments/analyze_language_results.py
```

The script fails on any violated pairing assumption before computing results. It produces per-prediction agreement with the recorded expert action, 600 paired alternative-versus-Correct effects, condition summaries, latency diagnostics, five-bin task-stage summaries, per-episode stage summaries, and 2,000-replicate episode-cluster bootstrap intervals. Near-zero translation vectors use a documented `1e-8` cosine threshold and remain undefined rather than receiving an invented direction.

The integrity report is `results/metrics_validation.json`; compact numerical findings are in `results/phase8_9_findings.md`. Analysis tables are under `results/`, and the three requested figures are available as PNG and PDF under `results/figures/`.

The analysis stops after Phase 9. It does not interpret gripper release semantics or perform any Phase 10 work.

## Verified hard-gate result

On 2026-08-08, the hard-gate command passed on `mps:0` with fallback disabled.
For episode 0, frame 0, the exact dataset task was:

```text
put both the cream cheese box and the butter in the basket
```

The recorded expert action was:

```text
[0.31874999, -0.24910714, 0.18482143, -0.00535714, 0.12107143, -0.03, -1.0]
```

The checkpoint's official postprocessor produced:

```text
[0.31535950, 0.13197200, -0.08346292, -0.05880658, 0.01600202, -0.02550344, -0.98739660]
```

This is a genuine finite 7-D SmolVLA prediction, not a claim of objective action
correctness. See `results/sanity_check.txt` for the compact terminal record.
