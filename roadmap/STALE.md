# roadmap.json here is behind the published roadmap

As of 2026-09-22 the artifact at https://claude.ai/artifact/YSLhctG5VRpxEjTDXKwY8o is **ahead of this
directory**. It was published from another working copy and contains work this one does not have:

- an **Architecture track** (AR1-AR11) added 22 Sep, with its own tally
- a `future work` status and the `.arch` / `.flow` styles that render it
- revised node text, including the Task F hand-check results (id 9 = animal 4/4, id 7 = child 3/3)

`roadmap.json` and `build_roadmap.py` in this directory know none of that: the builder has no
`future` status and no architecture section, so **running `build_roadmap.py` and publishing the
result would delete all of it**.

Until the newer `roadmap.json` is copied here:

- do not publish `roadmap/roadmap.html`
- to update the artifact, read the live version first, merge onto it, and publish that
  (`roadmap/roadmap_merged.html` is the copy last published, v22)

v22 marks AR1 done (the guarded single construction point in `scripts/signal_rebuild/tier1_signals.py`)
and updates AR2 for the class-id fix.
