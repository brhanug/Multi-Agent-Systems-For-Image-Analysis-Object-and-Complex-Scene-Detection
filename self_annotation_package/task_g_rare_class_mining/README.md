# Task G - rare-class mining (v3)

For every picture, write the object classes you see (person, child, animal, building, weapon, vehicle, tree, text) in `classes_you_see`, and the scene type (teaching, family, playing, landscape, drawing, or none) in `scene_you_see`. Note the minutes each sheet took once, on its first row. Easiest in the thumbnail page (`thumbnail_page/`), which saves into the same columns.

Sources: `prompt` = the top pictures for one rare class or scene by SigLIP and LLaVA together; `cluster` = smallest k-means clusters; `book` = books that look single-content; `random` = the baseline. Comparing the yield of rare content per minute across the four is the result.
