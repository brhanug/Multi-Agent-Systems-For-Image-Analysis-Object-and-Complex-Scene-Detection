# Defence preparation sheet

Thursday 10 December 2026, 08:45–10:00, online (Google Meet). Talk 25–30 minutes, then questions.
Rebuilt 25 November 2026 against the final thesis (`presentation/thesis_v4.pdf`, 143 pages). Every
figure below appears in the thesis or the handout and is held to its result file by the checkers.

---

## The answer in one breath

> "Several agents were meant to catch each other's mistakes: where they disagree, a person looks.
> On this archive that does not work. Most errors are made by all agents together, so disagreement
> never ranks errors better than plain confidence. The system I propose lets each agent report what
> it measures, routes by confidence, and attaches a risk promise from conformal risk control. Inside
> the labelled pool it approves 76.4% of pictures at a promised 5% risk; on random archive pictures
> the promise only holds once it is calibrated on archive pictures."

The contribution is **not a better detector** but a tested account of when an automatic answer can
be trusted, with a stated risk.

---

## Likely questions

### What is the research? What did you improve?

**"You built a pipeline. A thesis is not a pipeline. What is the research?"**
The pipeline is the instrument. The research is the hypothesis it was built to test: that disagreement
between models of different kinds shows where an automatic answer is wrong (an idea from query-by-committee,
Seung et al. 1992, and deep ensembles, Lakshminarayanan et al. 2017). I tested it in a test fixed in writing
before it was run, against people's labels, and it was rejected: disagreement 0.378, confidence 0.825;
within each task 0.78 / 0.82 against 0.84 / 0.83. I explained why (149 of 199 errors are made by all agents
together), audited and replaced the first design because of it, and tested the risk promise on random
archive pictures, where it failed before it was repaired. A pipeline project reports that its system works;
this thesis reports where it does not, and why.

**"What are your contributions?"** (talk slide "What is new")
1. Disagreement between different models does not find errors better than confidence, and why (shared errors).
2. An audit of what an uncertainty score measures: the deployed four signals were two measurements, eight states.
3. Pseudo-label scores hide failure (0.768 against the teacher, 0.146 against people), traced to the teacher
   (126 of 128 misses never labelled); a better teacher: 0.123 to 0.210.
4. A corrected design with a risk promise: 76.4% approved at 5% against 63.0%; a short object list (2.8 classes,
   9.2% missed).
5. The promise tested where it is used: 16.2% approved and wrong with pool calibration; 5.4-5.9% at 80%
   approved with archive calibration. Rule: calibrate on the collection itself.

**"How does this relate to existing work?"** (talk slides "Related research")
- Kim, Im & Mandl 2024 find objects in historical images with transfer learning and pseudo-labels; this thesis
  grades such a student against people and finds the collapse.
- Jiang et al. 2022 and Baek et al. 2022 use disagreement to estimate *how many* answers are wrong, for models of
  the same task; this thesis asks *which picture* is wrong, with models of different kinds.
- Hendrycks & Gimpel 2017: a model's own confidence is a strong baseline; confirmed here.
- Angelopoulos et al. 2024 (conformal risk control) assume new data are like the calibration data; this thesis
  tests that assumption on the archive.
- Piper & Jiang 2026 find models match people only where people agree; the same limit here (kappa 0.34).

### The design

**"Why didn't you train SigLIP and LLaVA on the archive?"** (Section 7.7, backup slide)
I did, in a test fixed in writing before it ran (25 Nov), cross-fitted over five book folds and evaluated on the
same 200 redraws. SigLIP with its learned mapping improves a lot (balanced accuracy 0.329 to 0.517, every redraw;
its untrained largest score already reaches 0.571, Table 6.4), but the system
only moves from 0.543 to 0.570 and the range includes zero; LLaVA does not change (0.505 to 0.507), nor does scene
presence (0.866 to 0.864). The system's mappings already learn from the same labels, and the rare classes are too
small (6 teaching, 32 playing, 51 family pictures). The scene classes stay a shortlist for a person.

**Why keep the old design in the thesis at all?**
Because the thesis is a test of its idea. The old design is what was built and run over all 12,110
pictures; Chapters 2–5 show what it measured and why it fails. Chapter 6 is the design that follows
from that diagnosis. Removing the old design would remove the evidence for the new one.

**What was wrong with the old design?**
Its four signals came from two measurements: three of them were read from one count in the
describer's answer, and the Agreement term was computed from two of the others. So the whole archive
fell into eight states, and every picture in a state got the same decision and the same uncertainty.

**What does the corrected design do differently?**
Three agents, each reporting what it measures: the object finder (8 class scores), the scene matcher
(5 scene scores plus one "no scene" score) and the describer (13 yes-strengths), 27 numbers in all.
Each report is mapped to probabilities by a regression learned on the training books; there is no
Agreement term; confidence, not disagreement, decides what a person checks; the text reader is not
used to decide.

**Do several agents beat one?**
Not clearly. Learned, the combination is barely ahead of the describer (LLaVA) alone: for the 8
object classes both reach a mean AUROC of 0.903. The thesis says so (Section 9.4).

**Why the weights 0.6 / 0.2 / 0.2?**
For "scene at all" they do not matter: equal and learned weights give the same ranking (0.863–0.866)
and approval within a point; the learned shares average 0.43, 0.24 and 0.33. The fixed ones are kept
because the archive tests were fixed with them. For the scene classes they do matter: the learned
mapping reaches a balanced accuracy of 0.544 against 0.401 with fixed weights, so the system uses it.

### Disagreement

**How do you know disagreement does not work?**
A test fixed in writing before it was run, on 300 hand-labelled pictures: disagreement ranks the
system's errors at 0.378, confidence at 0.825 (difference −0.448, [−0.541, −0.354]). Most errors are
shared: on 149 of 199 mistakes all three agents are wrong together.

**0.378 is below chance. Is disagreement anti-informative?**
Careful: that test asked object-class disagreement about scene errors. Within each task disagreement
does carry information, 0.78 for objects and 0.82 for scene presence, but it is still not above
confidence (0.84 and 0.83). Do not call 0.378 "below chance" without that caveat.

### The risk promise

**What exactly is promised?**
That the share of all pictures that are both approved automatically and wrong is at most α, on
average over calibration sets, for new pictures like the calibration pictures. It is not the error
rate among approved pictures: at 70% approved and 5% risk, about 7% of the approved are wrong.

**Is it a guarantee for every batch?**
No. At α = 0.05 the mean realised risk is 0.0498, met without margin; a single split exceeds it in
41.5% of cases. A per-calibration-set promise (Learn then Test) would cost approval; it is discussed,
not used.

**Does the promise hold on the archive?**
Not when calibrated on the labelled pool: on 400 random archive pictures (test fixed before
labelling) 16.2% are approved and wrong. Two causes: the pool mixed two definitions of a scene, and
it was not drawn at random. With one definition of a scene and a cut-off calibrated on archive
pictures, it comes to just above the promised level: 5.4–5.9% at 80% approved on 294 new pictures;
set to 2.5%, the promise gives 4.6%, so it is missed too.

**Why conformal risk control?**
It needs no assumption about the models or the form of the scores, only that new pictures are
exchangeable with the calibration pictures. That is also exactly where it failed on the archive, and
the thesis shows it.

### Data and labels

**Is 249 test pictures enough?**
Not for one split, which is why every headline figure is averaged over 200 book-disjoint redraws,
with the 2.5–97.5% range and the corrected resampled t-test. A difference is called established only
when the whole range lies on one side of zero. The limitation stays: 38 test pictures show no scene,
and "teaching" has only 6 pictures.

**Your labels come from one person.**
Mostly, yes. Two raters agree on scene type at κ = 0.337 (124 pictures); the same rater, rating 143
pictures a second time, reaches κ = 0.36 against the first rating. Most disagreement is about
"drawing": counting it as "no scene" lifts these to 0.50 and 0.58. Scene categories are
interpretations, not facts; the archive tests rest on one person's labels.

**Why does "drawing" count as a scene?**
In the expert labels "drawing" was the class for anything figurative that fits none of the other
four, including sheet music and borders. The thesis fixes one rule (people, animals, a landscape or
a room = scene; sheet music, text, ornaments, maps, a single object = no scene) and reports both
definitions where it matters (Section 7.1, Appendix G.5 on what "drawing" holds).

**The 0.768 → 0.146 collapse: same evaluator?**
Yes, the same Ultralytics validator: 0.768 against the teacher's pseudo-labels, 0.146 against human
boxes. 126 of 128 objects the student misses on pictures the teacher labelled were never in the
teacher's labels. The thesis's own evaluator gives 0.123 for the same detections; a better teacher
(Grounding DINO) raises the student to 0.210.

### Scene and object classes

**Can the system name the scene class?**
It shortlists but does not decide: balanced accuracy 0.544, finding family, playing and landscape in
27–62% of cases. Under a 5% promise only "no scene" and "drawing" are approved reliably (69.1%
approved). On random archive pictures most scenes are playing, not drawing, and balanced accuracy
falls to 0.351 (179 of 444 planned pictures labelled).

**And objects?**
The agents rank the 8 object classes well (mean AUROC 0.903). A promise on missed objects gives each
picture a list of 2.8 classes, missing 9.2% of a picture's objects on average, as promised (10.8% of
all objects), with about 1.56 wrong classes per picture for a person to strike out.

### Tier 2 and transfer

**What do the Tier 2 helpers add?**
Flags and links for a cataloguer; none changes a decision. Contradiction checker: 56% of its flags
name an object that is not there (68% with the retrained finder), against 36.5% of all claims.
Cross-archive linker: 38 of 50 random links at similarity ≥ 0.92 (76%) are real reprints, the rest
share only the style. Inscription reader: not usable as is; of its "real text" only 52.5% comes
from pictures with text.

**Does it transfer to another archive?**
Not unchanged. On 24 Europeana pictures (23 labelled, all scenes) the pipeline misses 65.2% of
confirmed scenes, and the agents span only about one independent direction there (effective rank
1.37 against 2.50 for the deployed signals on Colibri).

**Does restoration help?**
Not measurably for the scene matcher: 0.329 restored against 0.342 original (difference −0.014,
[−0.054, 0.017]). Only the scene matcher was re-scored, so this does not show restoration is
useless for every agent, but no result depends on it.

**Retrieval?**
Judged by a person on 4 queries, Precision@10 is 0.60 for the semantic index against 0.10 for
keywords. The proxy labels used earlier said 1.000; they agree with the person on 71.3% of the 94
judged pictures.

### Practical value and honesty

**What does the archive gain?**
At α = 0.05 the corrected design saves 18.2 hours of full annotation on the 12,110 pictures, at the
price of about 603 wrong labels let through against 201 (pool calibration).

**Did you use AI?**
Yes, as declared in "Use of AI Tools": Claude (Anthropic) through Claude Code for writing and
running the analysis code, checking figures against result files, and drafting and editing text that
I reviewed and revised; and as a pre-sorter that proposed candidates for relabelling (every label
behind a result is mine). Design decisions, research questions, pre-registered tests and
interpretation are mine.

**Can you reproduce the results?**
Every experiment script has a `--verify` flag that recomputes its result and fails on drift; the
whole pipeline was rerun twice from scratch. Three checkers hold the thesis text to the result files
(679 + 113, 43 and 39 figures). The code is public on GitHub, branch `thesis-submission`.

---

## Do not say

| Do not say | Say instead |
|---|---|
| "Disagreement detects errors" | It carries information, never more than confidence (0.78 / 0.82 against 0.84 / 0.83) |
| "0.989 → 0.1462" | 0.768 → 0.146 with one validator; 0.123 → 0.210 with the thesis's evaluator |
| "The promise holds on the archive" | Only when calibrated on archive pictures, and then just above 5% (5.4–5.9%) |
| "P@10 = 1.000" | 0.60 against 0.10, judged by a person on 4 queries |
| "Two independent raters re-checked the negative controls" | I relabelled them; 57.5% of 449 were real scenes |
| "ECE below 0.001", "0.9724 vs 0.9726" | In-sample figures of the old design; held out, U_SAA 0.860 against confidence 0.872 |
| "Tier 3 was run" | Tier 3 (paid frontier models) was designed, not run |
| "A guarantee for every batch" | On average over calibration sets; one split exceeds it 41.5% of the time |

---

## Number card

| | |
|---|---|
| Archive | 12,110 pictures; labelled pool 1,250 from 550 books; split 749 / 252 / 249 |
| Pre-registered test | disagreement 0.378, confidence 0.825; within task 0.78 / 0.82 vs 0.84 / 0.83 |
| Shared errors | 149 of 199 mistakes made by all agents |
| Approval at α = 0.05 | corrected 76.4%, old triage 63.0%, disagreement alone 61.1% (better in 199 of 200 redraws) |
| Approval at α = 0.10 | corrected 89.1%, old 63.0% |
| Realised risk at α = 0.05 | 0.0498 on average; one split exceeds it 41.5% of the time |
| Archive, pool calibration | 16.2% approved and wrong (400 random pictures) |
| Archive, archive calibration | 5.4–5.9% at 80% approved (294 new pictures) |
| Scene presence ranking | old 0.829, corrected 0.866 (88% of redraws) |
| Scene classes | balanced accuracy 0.544 learned, 0.401 fixed, 0.507 LLaVA; archive 0.351 |
| Object classes | AUROC 0.903; list of 2.8 classes, 9.2% missed per picture |
| Collapse | 0.768 → 0.146 (same validator); 0.123 → 0.210 with a better teacher |
| People | κ = 0.337 (two raters, 124); 0.36 self-consistency (143) |
| Review effort | 18.2 h saved at α = 0.05; ≈603 vs 201 wrong labels let through |
| Europeana | n = 24, 65.2% of scenes missed, effective rank 1.37 |
| Retrieval | P@10 0.60 vs 0.10 (4 queries) |

If you do not know a number, say so and offer to look it up.
