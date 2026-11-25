# Plan: what the class "drawing" contains (exploratory, added after the results freeze)

Written 21 Nov 2026, before the analysis is run. Exploratory: no accuracy is claimed and no
reported result changes.

## Why
"Drawing" is defined only as "figurative content that fits none of the other four classes". It
holds 341 of the 506 scene pictures of the labelled pool and 52.3% of the archive's answers, so it
may hide scene types that deserve a class of their own. This analysis shows what it contains, as
evidence for a revised class list (future work and the class-list suggestion owed to Prof. Mandl).

## Pictures
The 10,860 archive pictures outside the labelled pool whose scene-class answer is "drawing"
(results/signal_rebuild/archive_answers.csv, the system's learned joint mapping).

## Method, fixed now
- Features: the stored SigLIP image embeddings (results/signal_rebuild/siglip_embeddings.npz).
- k-means with k = 12 on the normalised embeddings, seed 20261121, 10 initialisations. Cluster
  sizes are also reported for k = 8 and k = 16, to show how much the picture depends on k.
- Each cluster is described by (a) its 12 most distinctive words in the describer's free-text
  answers (relational triplets and object list), by TF-IDF of the cluster against all other
  "drawing" pictures, words in at least 20 pictures, English stop words removed; (b) the mean
  of the system's eight object probabilities; (c) 8 pictures nearest the cluster centre.
- Each cluster gets a short descriptive name, read off (a)-(c). The names are an interpretation,
  marked as such; they are not labels and nothing is scored against them.

## Output
results/signal_rebuild/drawing_clusters.json; a table in Appendix H and a paragraph in Section 9.5.
