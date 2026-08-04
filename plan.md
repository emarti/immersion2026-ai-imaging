# Doc debt: readme.md / introduction.md updates still pending

This file previously tracked step-by-step build progress and experiment findings for the
original (pre age-band-split) pipeline. That content is superseded: build status lives in
`CLAUDE.md`'s layout table, and findings/run history live in `internal/lab-notes.md`
(its "Findings" section here duplicated `lab-notes.md`'s Run 3/5/7 — see git history for
the original text if needed). This file now tracks a narrower, more concrete thing: the
readme.md/introduction.md updates that a recent code change requires but that were
deliberately deferred, so the work isn't lost or silently re-derived later.

Per CLAUDE.md's own warning (the reason this file was retired the first time): entries
below describe *what changed and why*, not the literal new prose or config numbers to
paste in — quoting specifics here is exactly how the previous version of this file went
stale.

## Why this exists

A recent change (see `internal/lab-notes.md` for the run once it's actually executed)
replaced the single `cohort.age_min`/`age_max` band with two separate bands
(`cohort.age_train` broad, `cohort.age_eval` narrow and shared by validate/test),
consolidated the four `step4a`-`step4d` CNN designs into one `step4-train-network.py`,
added a data-driven `pos_weight` (`training.balanced_loss`) and an AUC metric to step4,
and replaced the never-built `step7-perturb.py` placeholder with a real
`step7-stack-predictors.py` (age/nWBV/CNN predictor ablation). The code and `CLAUDE.md`
are up to date; `readme.md` and `introduction.md` are not yet.

## Pending: age_train / age_eval (was age_min / age_max), and fraction-capped splits

- **readme.md §5.2** (cohort selection): still documents one `age_min`/`age_max` and its
  "Danger — the age (and sex) confound" callout as a single dial. Needs to explain the two
  bands instead: why train stays broad (more data, confound left in on purpose) and eval
  stays narrow (age-matched, honest numbers). `cohort.balance` (strict/label/none) has been
  REMOVED entirely, not just narrowed to the eval pool -- eval balancing is now hardcoded
  (always label-style: equal cdr_negative/cdr_positive, sex free), so any prose describing
  the three balance modes needs to go, not just be reworded.
  - **readme.md §5.5** (splits): `splits:` now only has `validate`/`test` (no `train` key
    -- it was removed as unused). VALIDATE is drawn first, then TEST, each balanced and
    capped at its own fraction -- but that fraction is measured against the size of the
    BROAD age_train-eligible pool, even though the subjects themselves are drawn only from
    the narrow age_eval pool. So the configured fraction (e.g. 0.15) ends up being a much
    bigger share of age_eval's own pool once actually drawn -- step2 prints both
    percentages. TRAIN then takes every age_train-eligible subject not claimed by
    validate/test -- no fraction, no cap (besides `cohort.max_subjects`), so its resulting
    size floats with whatever's left over. Needs a full rewrite of this section, not a
    number update -- the mechanism itself changed twice during development.
  - **readme.md §5.6 / §6**: needs a `training.balanced_loss` bullet, and the
    `splits.yaml`/`step2-cohort_age_histograms.png` bullets need to describe the new
    `meta` fields (`n_train_eligible`, `n_eval_eligible`, `validate_target`,
    `test_target`) and the new 2x2 (not 3-panel) histogram -- panel 4 shows train vs
    validate+test age distribution, to see how much validate/test depletes train inside
    the eval band, with a colour scheme where colour = group (blue = train, red =
    validate+test) and shade = CDR status (dark = CDR+, light = CDR-); panel 2 uses the
    same colour-family/shade idea for sex (green = Male, purple = Female).
- **introduction.md §10**: the age_min-as-single-dial discussion needs reframing around
  two independent bands, each with its own fraction-based cap; the old strict/label/none
  balance-mode discussion is gone.
- **introduction.md §14** ("Trade age range against sample size"): reword — no longer one
  trade-off, now two independent knobs.
- **introduction.md §14** ("Train for balanced accuracy, not plain accuracy"): the
  `pos_weight` bullet should be marked *now implemented* via `training.balanced_loss`
  (mirror the existing "now implemented as step6" phrasing used for Grad-CAM elsewhere in
  this section). Its "switch `cohort.balance` to none for more data" aside is now doubly
  stale -- `cohort.balance` doesn't exist anymore; train is unconditionally unbalanced.
- **introduction.md §12**: optional light-touch mention that this work exists, pointing at
  §14 and `internal/lab-notes.md`.

## Pending: single CNN design (was step4a-4d)

- **readme.md / introduction.md**: both still describe "four designs, diff any two" and
  any per-design comparison framing/tables. Needs a rewrite around the single network --
  no more "compare 4a vs 4c" exercises, since there's only one design to run.

## Pending: step7-stack-predictors.py (replaces the never-built step7-perturb.py)

- **introduction.md §14** ("Ask whether the model adds anything beyond age" sub-bullet,
  under "Use age properly instead of fighting it"): mark as *now implemented* via `step7`,
  mirroring the existing "now implemented as step6" phrasing. Keep the sibling "Give age to
  the model on purpose" sub-bullet (concatenating age into the network's own features)
  explicitly marked as a *different*, still-open alternative -- step7 does the
  separate-predictors-plus-logistic-regression variant, not the concatenate-into-the-network
  variant.
- **readme.md**: add a mention of `step7-stack-predictors.py` (what it answers -- does the
  CNN add anything over age/nWBV alone -- and what it writes) to the main §4 step table --
  it's no longer excluded from `process.sh` (see below), so it belongs in the normal step
  list now, not just an "optional extras" aside.
- **introduction.md**: sweep for any stray `step7-perturb.py` mentions (the placeholder
  this script replaces) and remove/update them.
- **readme.md**: step7 also writes `step7-predictor_correlations.png`, a 4-panel
  scatterplot figure (Pearson r per split in each panel) checking how independent its
  three signals actually are: (1) left vs right hippocampus CNN logit (raw, before the
  two sides are pooled into the single CNN predictor); (2) age vs nWBV; (3) CNN (pooled)
  vs age; (4) CNN (pooled) vs nWBV. Panels 2-4 are NOT raw units -- each variable is
  passed through its own single-variable logistic regression first and plotted on that
  model's pre-sigmoid log-odds scale (same |r| as the raw variables, but the sign can
  flip depending on the fitted coefficient's sign, e.g. nWBV's is expected negative).
  Needs a line in whatever readme.md section documents step7's outputs.
- **readme.md**: `step7-stacking_summary.txt`/`step7-stacking_comparison.png` also now
  report a third quantity per predictor, "bits of information" -- a cross-entropy-based
  LOWER BOUND on mutual information with CDR status (baseline = the cross-entropy of
  always guessing TRAIN's own prevalence, evaluated out-of-sample), with the same
  95%-bootstrap-CI treatment as AUC/balanced accuracy. Needs a line explaining what it
  is and, importantly, that it's a lower bound rather than the true value -- the
  in-script explanation (module docstring + text-summary preamble) has the full
  derivation and caveats to draw from.
- **readme.md**: step7 also writes `step7-roc_curves.png` -- one panel per split
  (train/validate/(test)), all 5 predictors' ROC curves overlaid on each, with AUC in
  the legend label. Needs a line in whatever readme.md section documents step7's
  outputs, alongside the other two figures above.

## Pending: step7 now runs in process.sh; output filenames got step-prefixed; --reveal

- **CLAUDE.md/readme.md §4** (step table): `process.sh` now runs all 7 steps (step7
  included, without `--reveal`, so TEST stays untouched by default). readme.md's step
  table and any "process.sh runs steps 1-6" prose needs updating to say 7.
- **Output filenames**: every root-level report/figure file now carries the step that
  writes it as a prefix -- `dataset_age_histograms.png` -> `step1-dataset_age_histograms.png`,
  `cohort_age_histograms.png` -> `step2-cohort_age_histograms.png`,
  `training_comparison.png`/`training_summary.txt` -> `step5-training_comparison.png`/
  `step5-training_summary.txt`, `gradcam_grid.png` -> `step6-gradcam_grid.png`,
  `stacking_summary.txt`/`stacking_comparison.png` -> `step7-stacking_summary.txt`/
  `step7-stacking_comparison.png`. Per-sample directories (`gradcam/`, `gradcam_context/`,
  `slice_context/`) and core pipeline data (`metadata.csv`, `splits.yaml`,
  `manifest.yaml`, `model_4.pt`, `training_log_4.csv`) were deliberately left unprefixed.
  Any readme.md/introduction.md filename references need updating throughout, not just in
  §5/§6.
- **`--reveal` flag (step5 and step7)**: both scripts default to never touching TEST;
  passing `--reveal` evaluates TEST once (reusing whatever was already fit/trained on
  train, never re-fit or re-tuned on test) and adds extra panels (step5: validate-vs-test
  headline metrics and per-grade accuracy; step7: a third "test" bar group per predictor)
  plus extra rows in the saved text summaries. Needs a mention wherever readme.md
  documents these two scripts, framed as a deliberate, one-time action -- not a routine
  flag to leave on.

## Pending: step5 ROC curve figure + new checkpoint dependency

- **readme.md**: step5 now also writes `step5-roc_curve.png` (validate always, test too
  under `--reveal`), computed by re-running the trained checkpoint fresh rather than
  reading `training_log_4.csv` -- the AUC in that figure's legend is a single
  final-checkpoint evaluation, so it won't exactly match the CSV-derived
  running-tail-mean AUC reported elsewhere in step5's own output. This is also a real
  behavior change worth a line on its own: even step5's *default* (non-`--reveal`) run
  now needs `outputs/model_4.pt` and `manifest.yaml` to exist, not just the CSV log, so
  it can no longer run from the log alone.
- This overlaps with the still-broken readme.md §4/§6 step5 entries noted above
  (pre-step-prefix filenames, stale "all designs" framing from the step4a-4d era) --
  whoever rewrites those should fold this bullet in rather than patching it separately.

## Pending: step3 now copies config.yaml into outputs/

- **readme.md §4 step table / §5**: step3 now also writes `outputs/config.yaml` -- a
  copy of whichever config was actually used (`--config`, or the default), so a run's
  PNGs/manifest always carry a record of the settings that produced them even if the
  repo's own config.yaml is edited afterward. This is unprefixed (like `manifest.yaml`,
  `splits.yaml`, `model_4.pt`) since it's pipeline data, not a report/figure. The step3
  row in readme.md's step table (§4) is itself part of the still-broken table noted
  above (pre-step-prefix filenames) -- fold this in with that rewrite rather than
  patching the row in isolation.
