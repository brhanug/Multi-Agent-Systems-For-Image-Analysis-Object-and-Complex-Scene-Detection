# Figure and layout rationale (extracted from LaTeX source comments)

These notes explain non-obvious choices in `thesis_v3_updated.tex`, chiefly in the
DCV-MAC architecture figure. They were inline comments; they live here so the
.tex ships clean while the reasoning survives for anyone editing the figure later.

- SECTION 2: THE PURE PIPELINE ARCHITECTURE
- CHAPTER 3: METHODOLOGY AND SYSTEM ARCHITECTURE
- Tier 1: 4 Validation Core Ensemble Agents (the architectural models)
- Derived Agreement signal -- NOT a fifth model, computed from a1+a3
- Tier 1 Core: two DISTINCT sub-computations sharing a labeled frame
- Tier 3: Dual-LLM Coordinator (selective, only for the ambiguous residual)
- Feeds into the two Tier 1 computations. Each arrow terminates on an explicit
- anchor along the target box's north edge: SAA takes 3 signals, U_SAA takes 4.
- (Do NOT use "(X.south |- box.north)" here -- that only matches the y-level and
- drops a vertical line at X's own x-position, so the arrow never reaches the box.)
- Tier 2 runs on EVERY validated image; Tier 3 sees only the sigma>tau residual
- that Tiers 1-2 could not resolve (matching the tier definitions in the text).
- Only the ambiguous residual reaches Tier 3. The auto-approved majority
- bypasses both the paid coordinator and the human queue -- without this
- bypass the figure would contradict the 60.6% auto-approval result.
- ROUTING ORIGINATES AT TRIAGE, NOT AT A TIER 2 AGENT. Drawing these arrows out
- of cur5/cur2 would say Agent 5 decides what escalates and Agent 2 decides what
- is auto-approved -- which contradicts the Architectural Decoupling Rule stated
- in this same chapter. Tier 2 runs in parallel as enrichment and gates nothing.
- The auto-approved majority bypasses BOTH the paid coordinator and the human
- queue. Without this edge the figure would contradict the 60.6% result.
- CHAPTER 4: SPATIAL GROUNDING COLLAPSE AND SINGLE-AGENT BENCHMARKING
- CHAPTER 5: MULTI-AGENT UNCERTAINTY CORE AND PROBABILISTIC CALIBRATION
- CHAPTER 6: HUMAN-IN-THE-LOOP TRIAGE AND CONFORMAL RISK CONTROL
- CHAPTER 7: DOWNSTREAM CURATION, RETRIEVAL, AND CROSS-ARCHIVE UTILITY
- CHAPTER 8: CONCLUSIONS, ETHICAL CONSIDERATIONS, AND FUTURE WORK
