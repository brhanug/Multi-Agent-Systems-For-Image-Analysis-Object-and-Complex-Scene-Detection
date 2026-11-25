# Task A2 - remaining negative controls

750 images from the 1,199-row default-labelled block that no one has
reviewed. Every row already covered by the 2026 single-rater audit or the
Label Studio second-rater pass is excluded, so this is disjoint from both.

Question: does the image show a describable scene (person, animal, landscape,
drawing or narrative content)?

`current_label` is the catalogue default, not a human judgement. It is shown
for reference only and is wrong often enough that it should not anchor you:
the two earlier passes found 66.7% and 57.5% of these defaults incorrect.

`cvat_id` indexes the original 2,000-row annotation table, which is how the
labels merge back into the corrected pool.
