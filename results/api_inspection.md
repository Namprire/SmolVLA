# Installed LeRobot SmolVLA API inspection

- LeRobot version: `0.4.4`
- Policy source inspected: `/Users/nampham/Documents/Blended/SmolVLA/.venv/lib/python3.11/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`
- SmolVLA processor source inspected: `/Users/nampham/Documents/Blended/SmolVLA/.venv/lib/python3.11/site-packages/lerobot/policies/smolvla/processor_smolvla.py`
- Tokenizer processor source inspected: `/Users/nampham/Documents/Blended/SmolVLA/.venv/lib/python3.11/site-packages/lerobot/processor/tokenizer_processor.py`
- `SmolVLAPolicy.predict_action_chunk` signature: `(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None, **kwargs: typing_extensions.Unpack[lerobot.policies.smolvla.modeling_smolvla.ActionSelectKwargs]) -> torch.Tensor`
- `SmolVLAPolicy.select_action` signature: `(self, batch: dict[str, torch.Tensor], noise: torch.Tensor | None = None, **kwargs: typing_extensions.Unpack[lerobot.policies.smolvla.modeling_smolvla.ActionSelectKwargs]) -> torch.Tensor`
- `VLAFlowMatching.sample_actions` signature: `(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs: typing_extensions.Unpack[lerobot.policies.smolvla.modeling_smolvla.ActionSelectKwargs]) -> torch.Tensor`
- `VLAFlowMatching.sample_noise` signature: `(self, shape, device)`

## Task-to-token path

The caller supplies `task` beside the checkpoint input features in the raw batch dictionary. `batch_to_transition` removes it from observations and places it in `complementary_data`. The checkpoint's serialized `to_batch_processor` converts the string to a one-element batch. `SmolVLANewLineProcessor` appends `\n`. `TokenizerProcessorStep` reads `complementary_data['task']` and calls the SmolVLM tokenizer with `max_length=48`, right padding, `padding='longest'`, truncation, and PyTorch output. It writes `observation.language.tokens` as `torch.int64` and `observation.language.attention_mask` as `torch.bool`. The device processor then moves both to the configured inference device.

The serialized preprocessor and normalization state from the official checkpoint are used unchanged except for overriding the runtime device from CUDA to the selected host device.

## Explicit flow noise

`SmolVLAPolicy.predict_action_chunk(batch, noise=...)` forwards `noise` through `_get_action_chunk` to `VLAFlowMatching.sample_actions`. When absent, that sampler calls `sample_noise` with shape `(batch_size, config.chunk_size, config.max_action_dim)` and creates a normal `torch.float32` tensor on `state.device`. For this checkpoint and one frame, the required explicit tensor is `(1, 50, 32)`, `torch.float32`, `mps:0`. The 32-D sampler output is unpadded by the policy to the checkpoint's 7-D action before the official action unnormalizer runs.

## Runtime token audit

- `correct`: `put both the cream cheese box and the butter in the basket`
  - token IDs: `[1078, 1062, 260, 8549, 10421, 3985, 284, 260, 7121, 281, 260, 11831, 198]`
  - attention mask: `[True, True, True, True, True, True, True, True, True, True, True, True, True]`
- `paraphrase`: `place the cream cheese box and the butter together inside the basket`
  - token IDs: `[2361, 260, 8549, 10421, 3985, 284, 260, 7121, 1592, 2972, 260, 11831, 198]`
  - attention mask: `[True, True, True, True, True, True, True, True, True, True, True, True, True]`
- `contradictory`: `put the cream cheese box in the basket but leave the butter outside the basket`
  - token IDs: `[1078, 260, 8549, 10421, 3985, 281, 260, 11831, 564, 3934, 260, 7121, 2856, 260, 11831, 198]`
  - attention mask: `[True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]`
- `unrelated`: `open the top drawer of the cabinet`
  - token IDs: `[6465, 260, 1466, 46520, 282, 260, 19565, 198]`
  - attention mask: `[True, True, True, True, True, True, True, True]`
