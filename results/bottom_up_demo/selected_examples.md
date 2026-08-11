# Selected bottom-up examples

## Level 1 — translation

- Episode: 3
- Frame: 38
- Normalized trajectory progress: 0.132
- Operational gripper regime: OPENING for all 31 prompts

Selection was deterministic: require one regime across all 31 prompts, require
Paraphrase < Contradictory < Unrelated mean translation distance, exclude the
upper quartile of Unrelated observation means, then choose the largest angular
difference between Correct and the Unrelated class-mean vector. This produces a
clear but non-extreme movement-direction example.

Mean translation distance from Correct:

- Paraphrase: 0.203
- Contradictory: 0.245
- Unrelated: 0.461

## Level 2 and Level 3 — gripper plus one-object micro-task

- Episode: 5
- Frame: 230
- Normalized trajectory progress: 0.827

Selection was deterministic: require Correct CLOSING and all ten Paraphrases
CLOSING, require at least one altered opening request, then maximize the sum of
Contradictory and Unrelated opening rates across the ten variants.

Opening frequency:

- Correct: 0/1
- Paraphrase: 0/10
- Contradictory: 3/10
- Unrelated: 7/10

The micro-task uses exact stored rows: CORRECT, P07, C01, and U09. P07 is the
closing Paraphrase nearest its class-mean translation vector. C01 and U09 are
the opening variants nearest their respective class-mean translation vectors.
This branch set is a curated, traceable illustration; the frequency plot shows
how often each regime occurs across all ten variants.
