# Phase B prompt-bank audit

The Correct row is the exact canonical dataset task. The 30 altered prompts were manually inspected before inference. Paraphrases retain both named objects and the basket destination; contradictions stay within the visible packing domain but alter the goal; unrelated prompts are plausible manipulation tasks that do not mention the original objects or basket.

The original Drawer instruction is retained only as `U01`, allowing its influence to be compared with nine other unrelated prompts.

| prompt_id | prompt_class | instruction | semantic_notes |
| --- | --- | --- | --- |
| CORRECT | Correct | put both the cream cheese box and the butter in the basket | Exact canonical dataset task; the single baseline for every observation. |
| P01 | Paraphrase | place both the cream cheese box and the butter in the basket | Lexical substitution; preserves both objects and destination. |
| P02 | Paraphrase | put the butter and the cream cheese box together inside the basket | Reverses object order and uses inside/together; same goal. |
| P03 | Paraphrase | move the cream cheese box and the butter into the basket | Uses move/into while retaining both objects and destination. |
| P04 | Paraphrase | transfer both items, the cream cheese box and the butter, to the basket | Appositive syntax; explicitly names both items and the basket. |
| P05 | Paraphrase | set the cream cheese box as well as the butter inside the basket | As-well-as coordination; same two-object placement goal. |
| P06 | Paraphrase | pack the basket with both the cream cheese box and the butter | Destination-fronted packing wording; preserves complete goal. |
| P07 | Paraphrase | take the cream cheese box and the butter and place them in the basket | Two-verb construction with an explicit plural reference to both objects. |
| P08 | Paraphrase | ensure that both the butter and the cream cheese box are inside the basket | State-goal phrasing; both named objects must end inside the basket. |
| P09 | Paraphrase | the cream cheese box and the butter should both be put into the basket | Passive syntax; same object set and destination. |
| P10 | Paraphrase | into the basket, place both the cream cheese box and the butter | Fronted destination; retains both objects and the original goal. |
| C01 | Contradictory | put the cream cheese box in the basket but leave the butter outside the basket | Original contradiction: excludes the butter from the destination. |
| C02 | Contradictory | put the butter in the basket but leave the cream cheese box on the table | Reverses which target object is excluded. |
| C03 | Contradictory | place only the cream cheese box in the basket and move the butter beside the basket | Changes inclusion and gives the butter an adjacent destination. |
| C04 | Contradictory | put only the butter in the basket and keep the cream cheese box where it is | Selects only the butter and preserves the cream-cheese position. |
| C05 | Contradictory | place the cream cheese box on the table and put only the butter in the basket | Assigns different destinations and excludes one object from the basket. |
| C06 | Contradictory | place both the cream cheese box and the butter next to the basket instead of inside it | Preserves both objects but changes the destination for both. |
| C07 | Contradictory | put the cream cheese box in the basket and move the butter to the far side of the table | Keeps one original placement and gives the butter a distant destination. |
| C08 | Contradictory | stack the butter on top of the cream cheese box outside the basket | Changes the goal from packing to stacking both objects outside. |
| C09 | Contradictory | put the cream cheese box beside the basket and leave the butter on the table | Keeps both target objects outside at two stated locations. |
| C10 | Contradictory | move the cream cheese box into the basket, then take the butter out of the basket | Sequential inclusion/exclusion contradiction with distinct object roles. |
| U01 | Unrelated | open the top drawer of the cabinet | Original Unrelated prompt retained as one variant for direct diagnosis. |
| U02 | Unrelated | close the microwave door | Plausible appliance manipulation unrelated to the packing goal. |
| U03 | Unrelated | pick up the red mug and place it on the saucer | Different object-transfer task with an unrelated destination. |
| U04 | Unrelated | move the black bowl onto the serving plate | Different tabletop placement task using unrelated objects. |
| U05 | Unrelated | turn the stove knob clockwise | Plausible rotational manipulation unrelated to packing. |
| U06 | Unrelated | place the cereal box inside the kitchen cabinet | Different object and destination; not a modification of the basket task. |
| U07 | Unrelated | lift the saucepan and set it on the front burner | Different two-step kitchen manipulation task. |
| U08 | Unrelated | slide the cutting board toward the sink | Different planar manipulation with unrelated object and target. |
| U09 | Unrelated | put the blue cup on the upper shelf | Different object-placement task in another part of the workspace. |
| U10 | Unrelated | take the spoon from the counter and place it in the utensil holder | Different pick-and-place task with unrelated objects and destination. |
