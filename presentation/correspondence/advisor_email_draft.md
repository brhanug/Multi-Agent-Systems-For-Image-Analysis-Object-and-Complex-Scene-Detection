# Email to advisors

**To:** Prof. Dr. Thomas Mandl; Shereen Elsayed
**Subject:** Revised thesis and defence slides — and a question on timing

---

## Short version

Dear Prof. Mandl, dear Ms. Elsayed,

Thank you both for the detailed reviews, and I apologise for the slow reply. I wanted to work
through every comment and re-verify the results before answering. Both documents are
attached: the thesis (112 pages) and the defence slides (56, including backup material).

Three things came out of that re-verification, and I would rather raise them now than at the
defence. I found a defect in my own data path, where a fallback had substituted one model's
object count for another's. No result changes, but the description of one input was wrong and
is now corrected. I therefore rebuilt all four signals from the raw model outputs and retested
the central claim without the circularity in the original evaluation; the conclusion held.
Separately, an audit of the evaluation labels found 57.5% of the checked records mislabelled,
and on the corrected pool the risk guarantee holds, certifying 60.6% automatic approval at
91.1% error recall.

I would value your view on whether this is ready to defend, or whether it would be better to
take another one to two months. The two things that would most change what I can claim are
carrying the pipeline end to end on a second archive and finishing the negative-control
annotation — both labelling effort rather than new research.

I am happy to meet in person or online, at short notice.

With kind regards,
Brhanu Atsbaha

---

## Longer version, if you would rather spell the three findings out

Dear Prof. Mandl, dear Ms. Elsayed,

Thank you both for the detailed reviews, and I apologise for the slow reply. I wanted to work
through every comment properly and re-verify the results before answering. Both documents
are attached: the thesis (112 pages) and the defence slides (56, including backup material).

I have gone through your comments individually and can walk you through each one. Three
things came out of the re-verification that I would rather raise now than at the defence.

**I found a defect in my own data path.** The object signal entering the agreement score was
not, as I had written, the spatial detector's box count. The distilled detector's count is zero
for every image in the corpus, so a fallback had silently substituted the vision-language
model's object count instead. That signal and the VLM signal therefore came from one source,
which is why they agreed perfectly — something I had been reporting as a property of the
models rather than of the code. No result changes, since every figure was computed from the
table as it stands, but the description of one input was wrong and is now corrected.

**So I rebuilt the signals and ran the test again.** The raw model outputs had been kept, so I
re-derived all four signals from them, each from a different model, and re-ran the scene
classifier over all 12,110 images rather than the 231 that had genuine scores. The rebuilt
signals are far finer-grained: the object signal now takes 12,100 distinct values instead of
four. I then retested the central claim with the circularity removed, holding one model out so
the predictor and the uncertainty signal share no input, and scoring against human review
rather than a threshold on the agents. The conclusion held: disagreement still does not
outperform a plain confidence baseline. That negative result now survives a test built to
overturn it.

**An error in my evaluation data reversed an earlier finding.** Auditing the negative-control
pool showed that many of its "no scene" labels had never been reviewed by a person; on
re-annotation, 57.5% were wrong. Rebuilt on a corrected pool, the conformal risk guarantee
holds at every level tested, certifying 60.6% automatic approval while still catching 91.1% of
errors, where the earlier evaluation had appeared to show it failing.

I would value your view on whether the work as it stands is ready to defend, or whether it
would be better to take another one to two months. The two things that would most change
what I can claim are carrying the pipeline end to end on a second archive, and finishing the
annotation of the negative-control pool — both are labelling effort rather than new research.

I am happy to meet in person or online, whichever suits you, and can make myself available at
short notice.

With kind regards,
Brhanu Atsbaha
