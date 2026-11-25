# Plan: human relevance judgements for the retrieval pilot (Section 8.4)

Written 21 Nov 2026, before any judgement is made.

## Why
The pilot's relevance label is circular: a picture counts as relevant when the vision-language
model's own scene label matches the query, and the semantic score ranks mainly by that same label,
so Precision@10 = 1.000 is close to guaranteed by construction. A person's judgement replaces it.

## Items
The 10 fixed queries of scripts/core_pipeline/run_semantic_retrieval_eval.py; for each, the top 10
of the semantic index and the top 10 of the caption-keyword baseline, recomputed with that script's
own functions on the same input files. The rebuild must reproduce the stored proxy figures
(mean P@10 1.000 and 0.180) or the run stops. A picture found by both methods for the same query is
judged once. Items are shuffled with seed 20261121 within each query; queries appear in order.

## Judgement (blind)
The author sees the query text and the picture only, never the method, the score or the proxy
label, and answers: relevant / not relevant / unsure. "Relevant": the picture shows what the query
asks for.

## Measures, fixed now
- Human Precision@10 per method and query, unsure counted as not relevant (main figure); also with
  unsure pictures left out.
- Mean over the 10 queries for each method; number of queries on which the semantic index scores
  higher, equal, lower.
- Agreement between the proxy label and the human judgement (share of items, Cohen's kappa).
- With 10 queries the comparison is descriptive; no significance test is reported.
The result replaces the proxy figure in Section 8.4, the RQ4 answer and the contributions, whatever it is.
