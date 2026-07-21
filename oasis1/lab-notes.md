# Lab notes — model experiments

Running log of what we tried and how it did, for humans (not parsed by any code).
Each `## Run` below is one sweep of designs; add a new Run section when something
changes (e.g. turning on augmentation) and compare against earlier runs. The design
letters are sometimes redefined between runs — each Run states its own lineup.

**Best so far:** Run 3 · **4c** (narrow 8-16-32, moderate dropout 0.6/0.2, with
augmentation) · balanced accuracy **0.777** (reproduced exactly in Run 5).

**Fixed setup (same across runs unless noted):** OASIS-1, age 62–79, label-balanced
cohort (~110 subjects), left+right hippocampus patches, single fixed slice for
validation/test, AdamW at lr 1e-4, weight decay 1e-4, batch 32, BCE loss. Validation
is 50/50 healthy/demented so chance = 0.50 and accuracy = balanced accuracy.

**Design lineup for Runs 1–2** (three sizes × two dropout levels; Run 3 onward uses a
new lineup — see that section):

| Model | Conv channels | Dropout | ~Params |
|---|---|---|---|
| 4a | 32-64-128 (3 blocks) | 0.5 / 0.2 | 101k |
| 4b | 32-64-128 (3 blocks) | 0.8 / 0.4 | 101k |
| 4c | 16-32-64 (3 blocks) | 0.5 / 0.2 | 28k |
| 4d | 16-32-64 (3 blocks) | 0.8 / 0.4 | 28k |
| 4e | 32-64 (2 blocks) | 0.5 / 0.2 | 23k |
| 4f | 16-32 (2 blocks) | 0.8 / 0.4 | 7k |

---

## Run 1 — 200 epochs, NO random shifts (2026-07-19)

**Change from baseline:** none — this is the reference run. Augmentation OFF
(`apply_random_shifts: false`), 200 epochs.

**Validation results** (mean of the last 50 epochs; bal = balanced accuracy = acc):

| Model | bal | sens | spec | CDR 0.5 (very mild) | CDR 1 (mild) |
|---|---|---|---|---|---|
| 4a baseline | 0.632 | 0.618 | 0.647 | 0.516 | 0.787 |
| 4b baseline + high dropout | 0.656 | 0.694 | 0.619 | 0.606 | 0.840 |
| 4c narrow | 0.696 | 0.800 | 0.591 | 0.740 | 0.900 |
| 4d narrow + high dropout | 0.690 | 0.776 | 0.604 | 0.646 | 0.993 |
| 4e 2-block | 0.667 | 0.721 | 0.614 | 0.554 | 1.000 |
| **4f 2-block narrow + high dropout** | **0.728** | 0.706 | 0.750 | 0.530 | 1.000 |

Per-grade counts (validation): CDR 0.5 = 10 patches / 5 subjects, CDR 1 = 6 patches /
3 subjects, CDR 2 = **none**. So the grade columns are very noisy — trends, not
numbers. (CDR 2 always `n/a`.)

**What we saw**

- **Best overall: 4f** (balanced accuracy 0.728) — and it's the *smallest* model
  (2 blocks, 16-32, ~7k params) with the highest dropout. It also has the most
  balanced sens/spec (0.71 / 0.75), i.e. it isn't just guessing one class.
- **Smaller beats bigger here.** Ranking by balanced accuracy:
  4f (0.728) > 4c (0.696) > 4d (0.690) > 4e (0.667) > 4b (0.656) > 4a (0.632).
  The big baseline (4a) is dead last. Consistent with "data is the bottleneck, large
  nets overfit."
- **High dropout mostly helps.** 4a→4b +0.024, 4e→4f +0.061; the narrow 3-block pair
  was roughly flat (4c 0.696 → 4d 0.690).
- **Most models over-call "demented"** (sens > spec) — e.g. 4c sens 0.80 vs spec 0.59.
  4f is the exception and the most even.
- **Severity matters, as expected:** every model catches CDR 1 (mild) far better than
  CDR 0.5 (very mild). The very-mild cases are the hard ones. Note the tension: 4f wins
  on balanced accuracy but is near-worst on very-mild (CDR 0.5 = 0.53), earning its
  score from specificity + perfect CDR 1. If *catching very-mild disease* is the goal,
  **4c** looks best (CDR 0.5 = 0.74).

**Takeaway going in to the next run:** the lean, regularized models (4f, then 4c/4d)
are the ones to beat. Next: repeat with random-shift augmentation ON and see whether it
lifts the very-mild (CDR 0.5) accuracy in particular.

---

## Run 2 — 200 epochs, WITH random shifts (2026-07-20)

**Change from Run 1:** augmentation ON (`apply_random_shifts: true`, `random_shift: 3`,
`n_shifts: 8`) — each training patch duplicated 8× with random ±3-voxel in-plane
shifts. Everything else identical (200 epochs). Validation/test are never shifted, so
the per-grade counts are unchanged (CDR 0.5 = 10/5, CDR 1 = 6/3, CDR 2 = none).

**Validation results** (mean of the last 50 epochs; Δ = change in bal vs Run 1):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 | Δbal |
|---|---|---|---|---|---|---|
| 4a baseline | 0.630 | 0.708 | 0.552 | 0.644 | 0.813 | −0.00 |
| 4b baseline + high dropout | 0.704 | 0.713 | 0.696 | 0.602 | 0.897 | **+0.05** |
| 4c narrow | 0.727 | 0.708 | 0.746 | 0.608 | 0.873 | **+0.03** |
| **4d narrow + high dropout** | **0.736** | 0.781 | 0.690 | 0.670 | 0.967 | **+0.05** |
| 4e 2-block | 0.711 | 0.659 | 0.764 | 0.518 | 0.893 | **+0.04** |
| 4f 2-block narrow + high dropout | 0.519 | 0.037 | 1.000 | 0.000 | 0.100 | **−0.21** |

**What we saw**

- **Best is now 4d (0.736)** — confirms the hunch. Narrow 3-block + high dropout, and it
  also has the strongest very-mild detection (CDR 0.5 = 0.67) among the good models plus
  near-perfect CDR 1 (0.97). Best all-rounder; now the overall best across both runs.
- **Augmentation helped every mid-size model** by ~0.03–0.05 (4b, 4c, 4d, 4e all up);
  the baseline 4a was flat. So the shifts are doing their job for models with enough
  capacity to use the extra variety.
- **4f collapsed (0.728 → 0.519).** sens 0.037 / spec 1.000 means it degenerated into
  "always say healthy" — it catches almost no disease (CDR 0.5 = 0.00, CDR 1 = 0.10).
  *Why:* 4f is the tiniest net (2-block 16-32, ~7k params) with the highest dropout
  (0.8/0.4). Augmentation makes training harder/more varied; stacked on very high
  dropout + tiny capacity that's **too much regularization**, so it underfit and fell
  back to the trivial constant answer. Augmentation and heavy dropout are both
  regularizers — don't pile both onto a very small model.
- Very-mild (CDR 0.5) is still the hard grade everywhere; mild (CDR 1) is caught well
  (0.81–0.97).

**Takeaway / next ideas**

- **4d is the model to beat.** Sweet spot = moderately small net (narrow 3-block) +
  high dropout + augmentation.
- To rescue the tiny 2-block family (4e/4f) with augmentation, **lower their dropout**
  (e.g. back to 0.5/0.2) so they can still fit — the 4f collapse is over-regularization,
  not a bad architecture per se.
- The ceiling is very-mild (CDR 0.5) detection; to push it, try more data / longer
  training / the ideas in `introduction.md`.

---

## Run 3 — 200 epochs, augmentation on, NEW design lineup (2026-07-20)

**Change from Run 2:** redefined the five designs (letters now mean different nets) to
zoom in around Run 2's winner — **all are 3-block now**, varying **width** (8/16/32
base) and **dropout**. Augmentation still ON; still 200 epochs (now set in
`config.yaml`). Note: Run 3's **4a** *is* Run 2's winning net (16-32-64 @ 0.8/0.4), so
it anchors the two runs together.

**Lineup:**

| Model | Channels | Dropout | ~Params | note |
|---|---|---|---|---|
| 4a | 16-32-64 | 0.8 / 0.4 | 28k | = Run 2 winner |
| 4b | 16-32-64 | 0.9 / 0.5 | 28k | |
| 4c | 8-16-32 | 0.6 / 0.2 | 8k | |
| 4d | 8-16-32 | 0.8 / 0.4 | 8k | |
| 4e | 32-64-128 | 0.8 / 0.4 | 101k | wide "wrong-direction" control |

**Validation results** (mean of the last 50 epochs):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 |
|---|---|---|---|---|---|
| 4a  16-32-64 @0.8/0.4 | 0.736 | 0.781 | 0.690 | 0.670 | 0.967 |
| 4b  16-32-64 @0.9/0.5 | 0.698 | 0.751 | 0.644 | 0.642 | 0.933 |
| **4c  8-16-32 @0.6/0.2** | **0.777** | 0.819 | 0.735 | 0.726 | 0.973 |
| 4d  8-16-32 @0.8/0.4 | 0.623 | 0.801 | 0.445 | 0.682 | 1.000 |
| 4e  32-64-128 @0.8/0.4 | 0.704 | 0.713 | 0.696 | 0.602 | 0.897 |

**What we saw**

- **New best overall: 4c = 0.777** (narrow 8-16-32, *moderate* dropout 0.6/0.2). Beats
  the previous best (Run 2 4d, 0.736) and is the best across all runs. It also leads on
  specificity (0.735), has strong sensitivity (0.819), and the best very-mild detection
  (CDR 0.5 = 0.726). Clean all-round winner.
- **Too much dropout finally loses.** Holding width fixed and raising dropout hurt both
  pairs: 16-32-64 went 0.736 (0.8/0.4) → 0.698 (0.9/0.5); 8-16-32 went **0.777 (0.6/0.2)
  → 0.623 (0.8/0.4)**. Moderate dropout is now the sweet spot.
- **4d collapsed the other way:** spec 0.445 / sens 0.801 — it over-calls *demented*
  (false alarms on healthy), the mirror image of Run 2's 4f (which over-called healthy).
  Same lesson: too much dropout (0.8/0.4) on the tiniest net (8-16-32) destabilizes it,
  just in whichever direction.
- **The field is tight and strong** — excluding 4d, everything lands 0.70–0.78, all well
  above chance. Even the wide "wrong-direction" control 4e (0.704) holds its own; going
  wide didn't clearly lose this time.
- **Consistency check:** 4a here (= Run 2's winning net, unchanged) scored 0.736 —
  exactly matching Run 2's 4d. Good sign the pipeline is stable run-to-run.
- Severity gap persists: CDR 1 (mild) 0.90–1.00 vs CDR 0.5 (very mild) 0.60–0.73; 4c
  leads on very-mild.

**Takeaway / next**

- **4c is the new model to beat** (narrow 8-16-32, moderate 0.6/0.2 dropout, augmentation).
- The dropout story has flipped from the early runs: *moderate* now wins and *very high*
  hurts, especially on the tiny nets. Aim for "small but not over-regularized."
- Very-mild (CDR 0.5) detection is still the ceiling (~0.73 at best) — the next real
  gains likely come from more/cleaner data rather than more architecture/dropout tweaks.

---

## Run 4 — bigger crop + more shifts (2026-07-20)

**Change from Run 3:** same five designs (letters unchanged), but two data knobs turned
up together: **`n_shifts` 8 → 16** (twice as many augmented copies) and a **larger
hippocampus crop** — `ap: [76,152] → [68,160]` and `lr_left: [90,150] → [82,158]` (the
box grew ~16 voxels in each direction, so it now includes more surrounding tissue).
Everything else identical (200 epochs, augmentation on). Validation/test counts unchanged
(CDR 0.5 = 10/5, CDR 1 = 6/3, CDR 2 = none).

**Validation results** (mean of the last 50 epochs; Δbal = change vs Run 3):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 | Δbal |
|---|---|---|---|---|---|---|
| **4a  16-32-64 @0.8/0.4** | **0.681** | 0.690 | 0.672 | 0.550 | 0.923 | −0.055 |
| 4b  16-32-64 @0.9/0.5 | 0.661 | 0.735 | 0.588 | 0.682 | 0.823 | −0.037 |
| 4c  8-16-32 @0.6/0.2 | 0.604 | 0.804 | 0.404 | 0.722 | 0.940 | −0.173 |
| 4d  8-16-32 @0.8/0.4 | 0.514 | 0.850 | 0.177 | 0.786 | 0.957 | −0.109 |
| 4e  32-64-128 @0.8/0.4 | 0.657 | 0.711 | 0.604 | 0.626 | 0.853 | −0.047 |

**What we saw**

- **Yes — every design dropped** (Δbal −0.04 to −0.17). So the two changes made things
  **worse across the board**; Run 3's numbers stand as the better setup. Best so far is
  still Run 3's 4c (0.777); this run's best is only 4a at 0.681.
- **Specificity collapsed everywhere** — the models slid toward **over-calling demented**
  (crying wolf on healthy). spec fell to 0.404 (4c), 0.177 (4d), ~0.60–0.67 for the
  larger nets, while *sensitivity rose* (4d sens 0.850). That imbalance is exactly what
  drags balanced accuracy down.
- **Careful — the per-grade (CDR) numbers went *up*, and that's a trap.** 4d shows CDR 0.5
  = 0.786, CDR 1 = 0.957 — its best-looking grade recall yet. But grade accuracy here is
  just *recall of demented*, so a model that says "demented" more often scores higher on
  it **while its specificity craters**. Balanced accuracy (which counts the healthy side)
  is the honest read, and it fell. Don't be fooled by the grade columns alone.
- **The narrow nets suffered most** (4c −0.173, 4d −0.109); the larger, more-regularized
  4a and 4e held up best. When the input got noisier/looser, the tiny nets destabilized
  first — the flip side of Run 3, where the narrow 4c won on the *tight* crop.
- Run 3's clean winner **4c fell the hardest** (0.777 → 0.604), entirely via specificity
  (0.735 → 0.404). Its moderate dropout that shone on the focused crop couldn't cope once
  the crop let irrelevant tissue back in.

**Likely why / next**

- **Prime suspect: the looser crop.** A bigger box reintroduces non-hippocampus tissue
  and border, diluting the signal and handing the model confounds to over-fit — the
  classic route to a specificity collapse. Heavier augmentation (16 shifts) adds more
  regularization/variety on top and doubles training time.
- **The changes were confounded** — crop *and* n_shifts moved at once, so we can't split
  the blame. Next: change **one at a time** from the Run 3 baseline (revert the crop, keep
  n_shifts=16; then the reverse) to see which knob did the damage.
- Working assumption stands: **tight, hippocampus-focused crops win**; more augmentation
  is not a free lunch here. Revert to Run 3's crop (`ap: [76,152]`, `lr_left: [90,150]`)
  as the baseline to beat.

---

## Run 5 — reproduction of Run 3 (reverted Run 4's changes) (2026-07-20)

**Change from Run 4:** put the two Run 4 knobs back to the Run 3 setup — crop
`ap: [68,160] → [76,152]`, `lr_left: [82,158] → [90,150]`, and `n_shifts: 16 → 8`.
`random_shift` stayed at **3** (same as Run 3). Same five designs. This is a
reproducibility check, not a new idea.

**Result: byte-identical to Run 3.** Every design's acc / sens / spec and per-grade
numbers matched Run 3 to the digit — Δbal = 0.000 across the board:

| Model | bal (= Run 3) | Δ vs Run 3 |
|---|---|---|
| 4a  16-32-64 @0.8/0.4 | 0.736 | 0.000 |
| 4b  16-32-64 @0.9/0.5 | 0.698 | 0.000 |
| **4c  8-16-32 @0.6/0.2** | **0.777** | 0.000 |
| 4d  8-16-32 @0.8/0.4 | 0.623 | 0.000 |
| 4e  32-64-128 @0.8/0.4 | 0.704 | 0.000 |

**What we saw / takeaway**

- **4c remains the best (0.777)**, and the whole sweep reproduced Run 3 exactly.
- **The pipeline is deterministic** — a fixed seed gives identical results run-to-run.
  That's good hygiene, and it confirms **Run 4's degradation was genuinely caused by the
  bigger crop + more shifts**, not run-to-run luck. It also isolates the change: since
  only crop/`n_shifts` differed between Runs 4 and 5, those two knobs own the Run 4 drop.
- **Caveat — identical ≠ robust.** This reproduces because the *seed* is fixed, not
  because 0.777 is a precise number. The ~16-subject validation set still carries roughly
  ±0.1 of sampling noise, so the 4c-over-the-rest gap is not yet established. To trust it,
  vary the seed / use k-fold CV and look at error bars (see the overfitting note from this
  session).

---

## Run 6 — NEW 4-design teaching lineup, final-ish parameters (2026-07-21)

**Big reset from Run 5.** The sweep was repurposed from *optimization* to *teaching*: the
lineup is redefined to **four** designs that sample directions from a baseline, and several
knobs changed at once. **So this is NOT comparable to Runs 1–5** — read it on its own.

**Setup (from `config.yaml`):** cohort age **60–79**, `balance: label` (so validation is
50/50 → acc = balanced acc), seed 42. Crop `ap [74,154]` / `lr_left [88,152]` → patch
**80×64**. Augmentation on, `random_shift 4`, `n_shifts 10`. **100 epochs**, metrics
averaged over the **last 20**. Validation grade counts: CDR 0.5 = 16 patches / 8 subjects,
CDR 1 = **2 / 1**, CDR 2 = none.

**Lineup:**

| Model | Conv blocks | Dropout | ~Params | Direction |
|---|---|---|---|---|
| 4a | 8-16-32 (3) | 0.6 / 0.2 | 8.2k | baseline |
| 4b | 8-16-32 (3) | 0.0 / 0.0 | 8.2k | no dropout (overfit demo) |
| 4c | 16-32-64 (3) | 0.6 / 0.2 | 28k | wider |
| 4d | 8-16 (2) | 0.6 / 0.2 | 2.4k | shallower |

**Validation results** (mean of the last 20 epochs; acc = bal since val is 50/50):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 |
|---|---|---|---|---|---|
| **4a  baseline** | **0.690** | 0.758 | 0.622 | 0.728 | 1.000 |
| 4b  no dropout | 0.581 | 0.667 | 0.494 | 0.634 | 0.925 |
| 4c  wider | 0.674 | 0.697 | 0.650 | 0.659 | 1.000 |
| 4d  shallower | 0.621 | 0.269 | 0.972 | 0.178 | 1.000 |

**What we saw**

- **Baseline 4a wins (0.690)** — the lean, moderately-regularized 3-block net, with the
  most even sens/spec (0.76 / 0.62). Exactly the teaching point: the modest baseline is the
  one to beat.
- **4b (no dropout) is the worst (0.581)** — the overfit demo pays off: strip
  regularization and it's the weakest, leaning toward over-calling demented (spec 0.49).
- **4c (wider) ≈ baseline, slightly below (0.674)** — extra capacity didn't help. Same old
  lesson: bigger doesn't buy you anything on data this small.
- **4d (shallower) degenerated toward "mostly healthy"** (sens 0.27 / spec 0.97, bal
  0.621) — the 2-block net is too weak and tipped into predicting the negative class; it
  barely catches very-mild disease (CDR 0.5 = 0.18). A clean *under*-fit collapse (mirror
  image of the earlier over-call-demented collapses).
- **Ignore the CDR 1 column** — it's 1.000 almost everywhere, but that's **2 patches from
  one subject**, i.e. noise. CDR 0.5 (8 subjects) is the meaningful grade; 4a leads (0.728).

**Caveats / next**

- **Not comparable to earlier runs.** The 8-16-32 @0.6/0.2 net (Run 3's winning "4c", now
  the 4a baseline) scored **0.690 here vs 0.777 in Run 3** — but the crop, cohort, epochs
  (200→100) and averaging window (50→20) all changed, so that drop is setup, not model.
- Short run + tiny val (~9 demented subjects, ±0.1 noise) + the known peak-then-drop
  dynamic (20-epoch tail average). Treat 0.690 as balpark, not precise.
- Planned next: raise `age_min` to ~70 to age-match the groups — expect **less data and a
  lower but more honest** number (see the age-confound note now in the readme/intro).

---

## Run 7 — age-matched cohort (`age_min` 60 → 70) (2026-07-21)

**Change from Run 6:** raised `age_min` **60 → 70** to age-match the healthy and demented
groups (remove the age confound). Everything else identical to Run 6 (same four designs,
crop 80×64, `random_shift 4`, `n_shifts 10`, 100 epochs, last-20 average, `balance: label`
so val is still 50/50). The narrower age band roughly **halves** the very-mild data:
validation grade counts drop to CDR 0.5 = **8 patches / 4 subjects**, CDR 1 = 2 / 1,
CDR 2 = none — so only ~5 demented subjects in validation.

**Validation results** (mean of the last 20 epochs; acc = bal since val is 50/50):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 |
|---|---|---|---|---|---|
| 4a  baseline | 0.588 | 0.510 | 0.665 | 0.456 | 0.725 |
| 4b  no dropout | 0.530 | 0.370 | 0.690 | 0.325 | 0.550 |
| 4c  wider | 0.583 | 0.605 | 0.560 | 0.588 | 0.675 |
| 4d  shallower | 0.600 | 0.200 | 1.000 | 0.250 | 0.000 |

**What we saw**

- **Everything collapsed to near chance (0.53–0.60).** Take the age confound away and the
  numbers fall off a cliff versus Run 6 (0.58–0.69) — the whole field is now barely above
  the 0.50 line and barely distinguishable from *each other*.
- **4d's 0.600 "win" is a mirage.** sens 0.20 / spec 1.00 means it's essentially the
  "always healthy" classifier — it catches almost no disease (CDR 0.5 = 0.25, CDR 1 = 0.00)
  and gets its score for free from specificity. Balanced accuracy still flatters a
  degenerate model when the split is small; read sens/spec, not just bal.
- **Very-mild detection cratered** (CDR 0.5 = 0.25–0.59, was 0.63–0.73) — the grade that
  matters most is exactly the one that suffers when the easy age signal is gone.
- With only ~5 demented validation subjects (±0.15+ noise), the design ranking here is
  essentially noise — don't read anything into 4a vs 4c vs 4d.

**Interpretation (the honest read)**

- The drop is **expected and probably correct**. Two causes are tangled together and can't
  be separated from one run: (1) **less data** — age 70+ roughly halves the cohort, and the
  very-mild grade drops to 4 val subjects; (2) **a genuinely harder problem** — with
  healthy and demented now age-matched, the model can no longer lean on age/sex and must
  read actual atrophy, which for *very-mild* disease is subtle. Both point the same way.
- The takeaway is consistent with the age-confound note now in the readme/intro: part of
  the higher Run 6 numbers (0.69) was the confound leaking in; ~0.55–0.60 age-matched is
  the more honest, if bleaker, picture of what 2-D hippocampus slices alone can do here.
- To tell "too little data" from "genuinely harder" apart you'd need more age-matched
  subjects (more discs / a wider but still-matched band) or the tutorial's heavier
  preprocessing / 3-D signal — not another architecture tweak.

---

<!-- Add the next run below, same format:
## Run 8 — <what changed> (date)
**Change from Run 7:** ...
... setup (config values) ... lineup (if redefined) ... results table ...
what we saw ... takeaway ... Update the "Best so far" line at the top if a new model wins.
-->
