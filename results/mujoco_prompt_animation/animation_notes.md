# MuJoCo prompt-animation notes

> Illustrative rendering of stored action differences; not closed-loop SmolVLA–LIBERO execution.

## Format

The main deliverable is a synchronized 2×2 comparison. Every panel receives the
same episode-0 expert-demonstration end-effector XYZ state and the same
illustrative object placement. At eight pauses, each panel displays its own
stored next-action translation, continuous gripper value, operational gripper
regime, and full 7-D distance from the Correct prediction.

The prompt-conditioned actions are never integrated into the nominal path.
Orientation-control components contribute to the printed 7-D distance but are
not rendered as rotations because that would make the presentation harder to
read.

## Grounding and mapping

- Nominal motion: all 258 XYZ states from local expert demonstration episode 0.
- Object timing: episode-0 expert command transitions at frames 50, 129, 183,
  and 240, supported by the previously completed two-camera visual review.
- Local arrows: stored `pred_action_x/y/z` from `results/vla_predictions.csv`.
- Linear arrow scale: `0.18` scene coordinate per stored action unit.
- Gripper rule: stored prediction `< 0` is opening regime; `>= 0` is closing regime.
- The continuous predicted gripper value remains visible in every checkpoint label.
- Basket/object geometry and attachment are simplified visual proxies, not grasp physics.

Arrow/path scale is illustrative and derived from stored relative actions.

## Deterministic checkpoints

| Frame | Progress | P | C | U | Flip conditions | Why selected |
|---|---|---|---|---|---|---|
| 13 | 5.1% | 0.114 | 0.080 | 0.399 | None | Early ordinary control: low-progress comparison without an event transition. |
| 53 | 20.6% | 0.188 | 2.051 | 0.191 | Contradictory | Nearest stored evaluation after first expert close-command onset at frame 50. |
| 81 | 31.5% | 0.056 | 0.066 | 2.284 | Unrelated | Strongest episode-0 Unrelated distance in the 20–40% progress region. |
| 128 | 49.8% | 0.086 | 0.101 | 0.109 | None | Nearest stored evaluation to first expert open-command onset at frame 129. |
| 156 | 60.7% | 0.325 | 0.576 | 0.813 | None | Ordinary mid-trajectory control between the two manipulation cycles. |
| 191 | 74.3% | 0.232 | 0.392 | 0.490 | None | Nearest stored evaluation after second expert close-command onset at frame 183. |
| 221 | 86.0% | 0.077 | 0.411 | 2.124 | Unrelated | Late second holding interval with a strong Unrelated gripper-regime flip. |
| 241 | 93.8% | 2.085 | 0.212 | 0.145 | Paraphrase | Nearest stored evaluation to second expert open-command onset at frame 240. |

## Scope

The dataset state/action reference frames do not provide enough documented
physical calibration for an exact trajectory replay. The scene therefore uses
the stored coordinate system consistently but does not label arrow length as a
physical displacement. The animation does not establish physical success,
safety, or closed-loop behavior.
