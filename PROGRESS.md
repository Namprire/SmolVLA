# Progress

## DONE

- Phases 1–10
  - Real pretrained `HuggingFaceVLA/smolvla_libero` inference on Apple M2 Max MPS
  - Deterministic explicit flow-noise control
  - 10 episodes, 200 paired observations, and 800 predictions
  - Correct / Paraphrase / Contradictory / Unrelated conditions
  - Paired language-effect and recorded-expert agreement metrics
  - Episode-cluster bootstrap uncertainty
  - Task-stage sensitivity analysis
  - Gripper and robot-state semantics audits
- Final analysis figures
  - Four presentation figures in high-resolution PNG and PDF
- Final offline demo
  - Same-observation condition switching from stored predictions
  - Five deterministic curated examples, including an ordering counterexample
  - Packaged two-camera assets; no model load or inference
- Documentation/reproducibility cleanup
  - Final validation, findings summary, presentation cheat sheet, README, and smoke-test record
  - Repository ignore rules and tracked-file audit

## SKIPPED BY SCOPE

- Phase 11 expert release extraction
- Phase 12 release envelope
- Phase 13 release guard
- Phase 14 stochasticity
- MuJoCo

Reason: The controlled VLA language-sensitivity experiment already provides a complete research contribution, and remaining time is prioritized for rigorous analysis and presentation quality.

## CURRENT

- Project complete / presentation-ready

## NEXT

- Presentation

## BLOCKERS

- None
