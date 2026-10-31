# Plan — Section 4: Final Results

**Key argument:** With the simulator-chosen configs, the hybrids should beat gsplit (and
the baseline) on the settings where gsplit lost. Show real multi-GPU runtime results.

**Status caveat:** both hybrid models may not be built yet. A **separate build loop** will
implement the DP-hybrid and pipeline-hybrid (starting from the `plans/` micro-tests), then
fill `tab:final`. Everything here is scratch until then.

**Structure (done in `final_results.tex`):**
- Intro, build-status caveat, scratch results table, expected-diagram note, expected
  conclusion, placeholder conclusion.

**TODO:**
- Build the hybrids (separate loop), run cases A/B/C, fill `tab:final`.
- Consider replacing the table with a grouped bar chart (per case: DGL / gsplit / DP-hybrid
  / pipeline-hybrid) — decide at write-up time, not now.
- Write the final conclusion: did the hybrids recover the bad cases? did the simulator's
  config ranking match the measured ranking?
