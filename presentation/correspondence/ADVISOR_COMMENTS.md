# Advisor comments — resolution tracker

Prof. Dr. Thomas Mandl (M) and Shereen Elsayed (S), September 2026.
Both reviewed an earlier version; several items were already resolved by
intervening work, and are marked as verified rather than newly changed.

## Mandl — slides
| # | Comment | Resolution |
|---|---------|------------|
| M1 | Task and problem on one slide | Slide 1.1 carries both ("The Task" + "Why It's Hard: Domain Shift"); 1.1b is the supporting image evidence |
| M2 | Slide 3 performance — measured by you? | Notation slide now states: *every number in this deck was measured on this project's own data unless a citation is given*. The 0.0047 figure already carried "measured by us" |
| M3 | RQ3 should name the method, not Agent 0 | Already resolved — RQ3 reads "cross-modal contradiction detection" |
| M4 | RQ2 would develop into a good contribution | Agreed; RQ2 (workload reduction via CRC) is the strongest surviving contribution and now leads slide 1.4 |
| M5 | Jargon: review queue, DualCore, DCV, Distribution-Free CRC | Notation slide rewritten to *explain* rather than expand: DCV-MAC described in plain terms, CRC identified as a published standard method (Angelopoulos et al., ICLR 2024) with "distribution-free" glossed |
| M6 | SAA used before explained | Already resolved — Notation is slide 1; SAA defined at first use |
| M7 | More annotation detail in the thesis | Already present: CVAT/domain-expert protocol for C1, separate Label Studio bounding-box task and independent rater for C2 |
| M8 | Why these components? Why Shapley? | **Verified + extended**: all four Tier 1 models carry a "Why this model for this task" justification, and the three Tier 2 agents now carry one too. Plus a "Why Shapley values specifically" paragraph — agents are correlated, so leave-one-out attributes shared contribution arbitrarily; Shapley is the unique attribution satisfying efficiency, symmetry and null-player |
| M9 | student–teacher terminology | Already resolved — the thesis states it is knowledge distillation, names GroundingDINO as teacher and YOLOv11 as student |
| M10 | Slides 12 and 16 point to interesting findings | Those are the pseudo-label collapse and the data-integrity correction, now slides 3.1 and 3.5 |
| M11 | Image examples missing | Slide 1.1b shows 16 real Colibri scans with real detector output; Appendix A.1 and the failure gallery add more |

## Mandl — thesis
| # | Comment | Resolution |
|---|---------|------------|
| M12 | Europeana only in the annex | Already resolved — 11 mentions in Chapter 3 main body (§3.5 pilot, §3.6 precondition), not only the appendix |
| M13 | Annotation ambiguity is about humanities concepts | Already resolved — the text extends the survey's bullet with exactly this point (contested categories: "portrait", "genre scene", "allegorical figure") |
| M14 | State of the art too brief, especially zero-shot | Already resolved — 362-word zero-shot subsection naming WebImageText (400M pairs), ImageNet, COCO and ODinW |
| M15 | Methodology should start with simple approaches | Already resolved — Chapter 2 opens with "Why not a simpler approach first" |
| M16 | Figure 2.2: differences not visible | Already resolved — the figure adds Laplacian edge maps with measured edge variance (1,097 → 4,591, 4.2×), and the caption says outright that the difference is subtle by eye |
| M17 | "degraded scan" — the scan does not degrade | Already resolved — "degraded" now qualifies the *material* ("degraded archival material"), never the scan |
| M18 | Figures on p27 too small | Page numbering has shifted; the two smallest figures enlarged from 0.65 to 0.74 textwidth |
| M19 | §3.4 needs text; efficiency point good | The hardware/efficiency section carries substantive prose on the three-orders-of-magnitude latency span |
| M20 | List of objects and scenes? | Already resolved — all 37 object classes and the 5 scene categories are listed explicitly |

## Shereen
| # | Comment | Resolution |
|---|---------|------------|
| S1 | Slide 5 table does not fit | Resolved — the overrun gate reports 0 across all 53 slides |
| S2 | Slide 10 figure does not fit | Resolved — same gate |
| S3 | Cannot identify task / baselines / contribution / improvement | **New slide 1.4**, "The Contribution, Stated Against the Baselines", answers all four in one table, including the honest answer on improvement: ties on discrimination, wins on calibration and certification |
| S4 | VLM, SAM undefined at first use | Already resolved — Notation slide is slide 1 |
| S5 | Reads as a project presentation, not a thesis contribution | Slide 1.4 closes with the scientific contribution in one line; slide 1.3 now states plainly what was claimed and what had to be withdrawn |
| S6 | Restructure: problem → baselines → contribution → comparison → results | That is the order: 1.1 task, 1.2 baselines, 1.3 + 1.4 contribution, 3.x results. Slide 2 maps the whole chain to report chapters, and Section 3 now follows the report's chapter sequence |


## Verification against the report (rendered PDF, not source)

Each thesis-directed comment was re-checked against the built PDF rather than
the LaTeX, so hyphenation and float placement could not hide a gap:

| Item | Verified by |
|------|-------------|
| M2  | headline figures marked as measured on the Colibri corpus, with chapter pointers |
| M7  | CVAT and Label Studio protocols, "domain expert who reviewed" |
| M8a | seven "Why this model / method for this task" justifications (4 Tier 1, 3 Tier 2) |
| M8b | "Why Shapley values specifically" |
| M9  | "standard knowledge-distillation sense" |
| M12 | Europeana pilot section in Chapter 3 |
| M13 | contested categories, "allegorical figure" |
| M14 | 400 million pairs, ODinW, zero-shot via SigLIP |
| M15 | "Why not a simpler approach first" |
| M16 | edge-response maps, edge variance |
| M17 | "degraded archival material" |
| M19 | three-orders-of-magnitude latency discussion |
| M20 | 37-class object taxonomy, five scene categories |
| S3  | task, baselines, contribution and improvement each stated |
| S4  | abbreviations expanded at first use |

18 of 18 items confirmed present in the report.
