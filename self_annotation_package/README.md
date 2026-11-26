# Annotation hub

Everything open for hand labelling, and which dataset serves which purpose. Every task is built from
the project's own archive scans. Sheet PNGs are archive images and stay out of git.

## Open now

| Task | Question | Amount | Feeds |
|---|---|---|---|
| **A2** remaining negative controls | Does this picture show a scene? | 465 pictures, 31 sheets (batches 20–50, 15 per sheet) | Risk control and the blind-spot result; the test split needs about 80 confirmed negatives and has 65 |
| **F** pseudo-label classes | What is in the red box, and is the box good, loose or wrong? | 20 boxes, 2 sheets | What the teacher's class ids mean; the frozen class list |
| **G** rare-class mining | Which classes do you see? | 716 pictures, 50 sheets (30 cluster, 12 prompt, 8 book) | Rare-class training data, and which search tactic yields most per minute |
| **H** second scene rating | Which scene category, or none? | 150 pictures, 10 sheets | Per-category agreement between two people |
| **E** auto-approved audit | Is the pipeline's answer right? | 420 of 465 left | What automatic approval lets through |

## In Label Studio (running now)

| Task | Project | Pictures |
|---|---|---|
| A2 remaining | http://localhost:8080/projects/13/data | 465 |
| E | http://localhost:8080/projects/12/data | 420 |
| F | http://localhost:8080/projects/9/data | 20 |
| G | http://localhost:8080/projects/10/data | 716 |
| H | http://localhost:8080/projects/11/data | 150 |

Project 8 still holds all 750 A2 pictures including the 285 answered in chat; project 13 holds only
the open ones. Label Studio also records how long each answer takes, which is what the rare-class
comparison needs.

If the pages are blank, the two servers are not running:

```
.venv/bin/python3 scripts/core_pipeline/serve_images_cors.py 9000 &
/data/brhanu/miniconda/bin/python3.13 /data/brhanu/miniconda/bin/label-studio start --port 8080 --no-browser &
```

When you have labelled anything, `scripts/annotation/extract_open_task_labels.py` writes each task's
answers to `labelstudio_labels.csv` in its folder, with the seconds each one took.

**How to label on paper instead.** Open a sheet, read the numbered pictures, and reply with the numbers and your
answers; the answers are recorded exactly as given. Each task folder also has `labeling_sheet.csv`
with one row per picture, if you prefer a spreadsheet. Task A2 sheets ask only for the numbers that
**do** show a scene; everything else on that sheet counts as no scene.

**Order that helps most.** A2 first (it gates the risk-control result), then F (it unblocks the class
list), then H (it needs someone else's time), then G.

## Which dataset for which purpose

The thesis needs several datasets and they are not interchangeable. What makes each one valid is how
its pictures were chosen.

| Purpose | Dataset | How its pictures were chosen | Rule |
|---|---|---|---|
| Model outputs for the whole archive | 12,110 pictures | everything | no labels; never an evaluation set |
| Fitting fusion weights and thresholds | train part of the scene pool (60%) | random, whole books at a time | may include enriched data |
| Choosing the risk-control threshold | calibration part (20%) | random, whole books | must stay random, never enriched |
| Reporting risk and error recall | test part (20%) | random, whole books | must stay random, never enriched, never used to fit anything |
| Per-class detection performance | 300 pictures with human boxes | random from the archive | 136 of them are in the detector's training data, so report the other 164 separately |
| Human agreement | 124 pictures rated by two people | overlap of two independent passes | a category below kappa 0.4 cannot carry a result |
| Training the student detector | teacher pseudo-labels (8,212 and 4,879 pictures) | machine-made | training only, never evaluation |
| Rare-class training data | Task G pictures | deliberately searched | training and diagnosis only |
| Risk control and blind spots | confirmed negatives (319) | random from the archive | keep random: hand-picked negatives would invalidate the bound |

Two rules carry most of the weight:

- **Splitting by book, not by picture.** Pages of one book look alike, so a picture in training and its
  sibling in test would flatter every number. Splits therefore move whole books.
- **Enriched data never reaches calibration or test.** Deliberately searching for horses and buildings
  is the right way to get training data and the wrong way to estimate deployment risk, because the
  archive does not look like the search results.

## Adding a dataset for a new purpose

1. Write down the purpose first, then the sampling: **random** if the numbers will be reported as
   performance or risk, **searched** if the pictures will be trained on.
2. Keep a list of picture ids with the source and the date, so the set can be rebuilt.
3. One row per picture, one column per question, plus who answered. Nothing else in the file.
4. Check overlap against the existing sets before labelling, so a picture never lands in both a
   training set and a test set.
5. When labels come back, they are merged by picture id and the affected results are re-run; the
   committed numbers stay until the new set is complete.

## Labels that are not used in the thesis (checked 26 Nov 2026)

Every other label set in this package is used, in full, by a script behind a thesis figure; their `--verify`
runs reproduce from the labels as they are now. These four are not used, on purpose:

| Labels | What they are | Why they are not in the thesis |
|---|---|---|
| Task E, `task_e_auto_approved_audit/chat_labels.csv` (45 of 465 pictures, 10 Sep) | First batch of an audit of what the deployed design approved automatically | Labelled under the earlier definition of a scene and stopped after 45; superseded by the audit of 247 deployed approvals under one rule (quick labels, Section 7.1). None of the 45 pictures is in the later sets. |
| Task F, `task_f_pseudo_label_classes/` (20 boxes) | Which class each id in the January pseudo-labels meant | A diagnostic of the old label files (ids 2, 7, 8, 9 had two possible names); it informed how those files are read, not a reported figure. |
| Task G and G v3, Label Studio projects 10 and 15 (42 of 548 and 11 of 306 pictures) | Mining rare classes by four search strategies | Stopped early; too few pictures to compare the strategies. |
| Label Studio project 2, "Europeana 30-Image Evaluation" (28 labelled) | Colibri pictures copied under a misleading name by an earlier setup script | Not Europeana data; the thesis uses the genuine Europeana pilot (project 3, 23 of 24). |
