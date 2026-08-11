# Bottom-up extension findings

## Observed result

This extension decomposes the stored 7-D next action into two directly
interpretable primitives, then combines them once in a short MuJoCo
illustration. No new SmolVLA inference was run.

### 1. What happens to translation when only language changes?

The deterministic translation example is episode 3, frame 38. All 31 Phase-B
prompts produce the same operational gripper regime (OPENING), so the displayed
difference is isolated to translation. The ten prompt variants are shown as
faint arrows and their semantic-class means as bold arrows.

Mean translation distance from the single Correct action:

- Paraphrase: 0.203
- Contradictory: 0.245
- Unrelated: 0.461

The angle between the Correct vector and each class-mean vector is 16.3°,
22.8°, and 42.6°, respectively. In this selected, non-extreme example, changing
only the instruction changes both the magnitude and direction of the stored XYZ
proposal.

Across all 20 Phase-B observations, after first averaging the ten variants
within each observation, mean translation distance was 0.141 for Paraphrase,
0.215 for Contradictory, and 0.344 for Unrelated. The episode-cluster intervals
overlap, so this is aggregate evidence rather than a universal ordering.

### 2. Can language change the predicted opening/closing regime?

Yes. The deterministic gripper example is episode 5, frame 230. Using the
validated operational rule—prediction below zero is OPENING and prediction at
or above zero is CLOSING—the stored prompt-bank results are:

- Correct: 0/1 OPENING; continuous value +1.002
- Paraphrase: 0/10 OPENING
- Contradictory: 3/10 OPENING
- Unrelated: 7/10 OPENING

The continuous values remain visible in `gripper_example.csv`; they were not
clipped before analysis.

### 3. Do the effects survive multiple prompt variants?

For these two examples, yes descriptively. Level 1 shows ten independently
phrased variants per altered class rather than one chosen sentence. Level 2
shows that the altered-prompt opening behavior is not limited to one Unrelated
string: seven of ten Unrelated variants request OPENING at the gripper example.

The broader Phase-B result also retains aggregate
Paraphrase < Contradictory < Unrelated ordering for translation, arm-only 6-D,
and full 7-D distance. It is not universal observation by observation, and the
ten prompts on one frame are repeated measurements rather than independent
robot trials.

### 4. What does the one-object MuJoCo demonstration add?

It converts four exact stored Phase-B rows into a visually readable, one-second
local consequence from one shared simulated state:

- Correct (`CORRECT`, +1.002) and Paraphrase (`P07`, +0.993) stay CLOSING and
  keep the box attached.
- Contradictory (`C01`, -0.978) and Unrelated (`U09`, -0.996) enter OPENING;
  the box becomes a free MuJoCo body and falls under gravity.

The object begins 0.141 m from the illustrative basket center. The final local
distance proxies are 0.192 m (Correct), 0.172 m (P07), 0.078 m (C01), and
0.082 m (U09). Thus the OPENING branches happen to move the box closer to the
basket center in this particular visualization. This is reported rather than
hidden: Correct language is not assumed to be optimal at every isolated frame.

### 5. What does the MuJoCo demonstration NOT establish?

It does not establish closed-loop control, physical task success, model safety,
or agreement with a complete LIBERO rollout. The Panda motion is position-only
IK following a fixed visualization transform. The branch scale is 0.14 for
readability. While held, the object follows the illustrative hand target; on an
OPENING branch it becomes a free body. No MuJoCo state is fed back through
SmolVLA.

The reported distances and held/not-held flag are **illustrative short-horizon
task proxies**, not SmolVLA task-success statistics.

### 6. How does this create a bottom-up progression?

The presentation now starts with the smallest empirical question—where does the
stored action point? It then isolates the gripper threshold—hold or open? The
one-object scene combines only those two primitives. Finally, the existing Panda
packing animation restores the original two-object context. This addresses task
interpretability without replacing the validated offline experiment or claiming
a simulator rollout that was not performed.

> Stored-action visualization; not closed-loop SmolVLA–LIBERO execution.

## Limitations

- The examples are deterministically selected from 20 Phase-B observations,
  one pretrained model, one task, and one domain.
- The prompt bank is manually and systematically authored, not a complete
  language distribution.
- The translation example is deliberately controlled for gripper regime; the
  gripper example is deliberately selected for maximal cross-prompt regime
  variation. Neither is claimed to represent every observation.
- Prompt variants on the same observation are nested repeated measurements.
- The simulation illustrates stored next actions over a short horizon and does
  not measure full-task success.
