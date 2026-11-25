# Extension request — draft

**To:** Prof. Dr. Thomas Mandl; Shereen Elsayed
**Subject:** Request for a four-week extension, and the reasons for it

---

Dear Prof. Mandl, dear Ms. Elsayed,

I am writing to ask for a four-week extension. I would rather explain what prompted
it than simply request the time.

Over the past week I completed a verification pass I had been putting off: running all
four Tier 1 models over the entire 12,110-image corpus, rather than relying on the
stored signal table. That run took under five hours, and it changed what I know about
my own results in four ways.

**The object signal was never the detector's output.** The manifest field the fusion
stage reads is zero on all 12,110 rows, so a fallback silently substituted the
vision-language model's object count on every image. I had documented this from the
data; re-executing both code paths confirmed it. The detector had in fact only ever run
on 231 images.

**A second signal is degenerate in the same way.** The VLM agent is defined as the
fraction of four questions answered substantively. On real model output two of those
questions are yes/no by design, so the value is 2/4 on 98.9% of the corpus. It encodes
the question format rather than the image.

**Two decision rules coexist in the pipeline.** A stored fusion column and a
recomputation from the agent columns disagree on 2,217 of 12,110 images (18.3%). Both
have produced numbers in the thesis: the same weighting on the same test partition gives
70 errors under one rule and 56 under the other.

**The headline guarantee rests on five errors.** The conformal result I report — 60.6%
automatic approval at 91.1% error recall, risk 0.0201 under a 0.05 bound — comes from
five errors among 151 auto-approved images. The exact 95% interval on that risk is
[0.013, 0.071]; its upper end exceeds the level being certified.

The consequence is specific. Five of my experiments — AWLF, conformal risk control, the
Tier 1 ablation, the Shapley attribution and the spatial meta-learner — still run on the
superseded inputs, and my central operational claim is not yet securely estimated. I
could defend the thesis as it stands by documenting these as limitations, but I would be
defending a certified guarantee whose interval overruns its own target, and a results
chapter built on inputs I have since shown to be a fallback.

With four weeks I would: complete the annotation that estimates the deployment risk
directly (in progress, 465 images sampled); reconcile the two decision rules and
re-derive the error label and calibration; recompute the five affected experiments on
the measured signals; and revise the affected sections.

I am happy to meet and go through any of this, in person or online, at short notice.

With kind regards,
Brhanu Atsbaha
