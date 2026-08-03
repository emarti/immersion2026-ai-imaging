# Lab notes — model experiments

Running log of what we tried and how it did, for humans (not parsed by any code).
Each `## Run` below is one sweep of designs; add a new Run section when something
changes (e.g. turning on augmentation) and compare against earlier runs. The design
letters are sometimes redefined between runs — each Run states its own lineup.

**Best so far:** Run 3 · **4c** (narrow 8-16-32, moderate dropout 0.6/0.2, with
augmentation) · balanced accuracy **0.777** (reproduced exactly in Run 5).

**Fixed setup (same across runs unless noted):** OASIS-1, label-balanced cohort,
left+right hippocampus patches, single fixed slice for validation/test, AdamW at lr 1e-4,
weight decay 1e-4, batch 32, BCE loss. Validation is 50/50 CDR-negative/CDR-positive so
chance = 0.50 and accuracy = balanced accuracy. The **age band has changed several times**
(62–79 early on, 70 in Run 7, 60–79 in Runs 8–9, 65–85 in Run 10, 70–90 in Run 11, 60–80 in Run 12) and with it the cohort
size (~110–140 subjects) — each Run states its own.

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
- **Most models over-call "CDR-positive"** (sens > spec) — e.g. 4c sens 0.80 vs spec 0.59.
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
  "always say CDR-negative" — it catches almost no disease (CDR 0.5 = 0.00, CDR 1 = 0.10).
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
- **4d collapsed the other way:** spec 0.445 / sens 0.801 — it over-calls *CDR-positive*
  (false alarms on CDR-negative), the mirror image of Run 2's 4f (which over-called CDR-negative).
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
- **Specificity collapsed everywhere** — the models slid toward **over-calling CDR-positive**
  (crying wolf on CDR-negative). spec fell to 0.404 (4c), 0.177 (4d), ~0.60–0.67 for the
  larger nets, while *sensitivity rose* (4d sens 0.850). That imbalance is exactly what
  drags balanced accuracy down.
- **Careful — the per-grade (CDR) numbers went *up*, and that's a trap.** 4d shows CDR 0.5
  = 0.786, CDR 1 = 0.957 — its best-looking grade recall yet. But grade accuracy here is
  just *recall of CDR-positive*, so a model that says "CDR-positive" more often scores higher on
  it **while its specificity craters**. Balanced accuracy (which counts the CDR-negative side)
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
  regularization and it's the weakest, leaning toward over-calling CDR-positive (spec 0.49).
- **4c (wider) ≈ baseline, slightly below (0.674)** — extra capacity didn't help. Same old
  lesson: bigger doesn't buy you anything on data this small.
- **4d (shallower) degenerated toward "mostly CDR-negative"** (sens 0.27 / spec 0.97, bal
  0.621) — the 2-block net is too weak and tipped into predicting the negative class; it
  barely catches very-mild disease (CDR 0.5 = 0.18). A clean *under*-fit collapse (mirror
  image of the earlier over-call-CDR-positive collapses).
- **Ignore the CDR 1 column** — it's 1.000 almost everywhere, but that's **2 patches from
  one subject**, i.e. noise. CDR 0.5 (8 subjects) is the meaningful grade; 4a leads (0.728).

**Caveats / next**

- **Not comparable to earlier runs.** The 8-16-32 @0.6/0.2 net (Run 3's winning "4c", now
  the 4a baseline) scored **0.690 here vs 0.777 in Run 3** — but the crop, cohort, epochs
  (200→100) and averaging window (50→20) all changed, so that drop is setup, not model.
- Short run + tiny val (~9 CDR-positive subjects, ±0.1 noise) + the known peak-then-drop
  dynamic (20-epoch tail average). Treat 0.690 as balpark, not precise.
- Planned next: raise `age_min` to ~70 to age-match the groups — expect **less data and a
  lower but more honest** number (see the age-confound note now in the readme/intro).

---

## Run 7 — age-matched cohort (`age_min` 60 → 70) (2026-07-21)

**Change from Run 6:** raised `age_min` **60 → 70** to age-match the CDR-negative and CDR-positive
groups (remove the age confound). Everything else identical to Run 6 (same four designs,
crop 80×64, `random_shift 4`, `n_shifts 10`, 100 epochs, last-20 average, `balance: label`
so val is still 50/50). The narrower age band roughly **halves** the very-mild data:
validation grade counts drop to CDR 0.5 = **8 patches / 4 subjects**, CDR 1 = 2 / 1,
CDR 2 = none — so only ~5 CDR-positive subjects in validation.

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
  "always CDR-negative" classifier — it catches almost no disease (CDR 0.5 = 0.25, CDR 1 = 0.00)
  and gets its score for free from specificity. Balanced accuracy still flatters a
  degenerate model when the split is small; read sens/spec, not just bal.
- **Very-mild detection cratered** (CDR 0.5 = 0.25–0.59, was 0.63–0.73) — the grade that
  matters most is exactly the one that suffers when the easy age signal is gone.
- With only ~5 CDR-positive validation subjects (±0.15+ noise), the design ranking here is
  essentially noise — don't read anything into 4a vs 4c vs 4d.

**Interpretation (the honest read)**

- The drop is **expected and probably correct**. Two causes are tangled together and can't
  be separated from one run: (1) **less data** — age 70+ roughly halves the cohort, and the
  very-mild grade drops to 4 val subjects; (2) **a genuinely harder problem** — with
  CDR-negative and CDR-positive now age-matched, the model can no longer lean on age/sex and must
  read actual atrophy, which for *very-mild* disease is subtle. Both point the same way.
- The takeaway is consistent with the age-confound note now in the readme/intro: part of
  the higher Run 6 numbers (0.69) was the confound leaking in; ~0.55–0.60 age-matched is
  the more honest, if bleaker, picture of what 2-D hippocampus slices alone can do here.
- To tell "too little data" from "genuinely harder" apart you'd need more age-matched
  subjects (more discs / a wider but still-matched band) or the tutorial's heavier
  preprocessing / 3-D signal — not another architecture tweak.

---

## Run 8 — much tighter hippocampus crop, back to age 60 (2026-07-29)

**Change from Run 6** (Run 7 was the age-70 detour; this returns to `age_min: 60`): the crop
is **greatly reduced** to focus tightly on the hippocampus, and `random_shift` is eased
4 → 3. Same four-design teaching lineup and cohort as Run 6, so Δ is read against Run 6.

- **Crop:** `ap [74,154] → [80,146]` and `lr_left [88,152] → [92,146]` — patch **80×64 →
  66×54** (H×W), ~30 % less area. Tightest box we've tried, following the standing result
  that tight, hippocampus-focused crops win (Runs 4–5) and the new step3 **context images**
  showing the old box was too large.
- **Augmentation:** `random_shift 4 → 3`; `n_shifts` stays 10.
- **Unchanged:** cohort age **60–79**, `balance: label` (val 50/50 → acc = bal), seed 42,
  100 epochs, metrics over the last 20. (New this run: step3 also writes `slice_context/`.)
- Note: two knobs moved (crop *and* `random_shift`), so a change vs Run 6 can't be split
  between them cleanly — but the crop is by far the bigger move.

**Lineup** (same as Run 6):

| Model | Conv blocks | Dropout | ~Params | Direction |
|---|---|---|---|---|
| 4a | 8-16-32 (3) | 0.6 / 0.2 | 8.2k | baseline |
| 4b | 8-16-32 (3) | 0.0 / 0.0 | 8.2k | no dropout (overfit demo) |
| 4c | 16-32-64 (3) | 0.6 / 0.2 | 28k | wider |
| 4d | 8-16 (2) | 0.6 / 0.2 | 2.4k | shallower |

(Param counts are unchanged despite the smaller patch: the global-average-pool head depends
only on channel counts, not spatial size.)

**Validation results** (mean of the last 20 epochs; acc = bal since val is 50/50):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 | Δbal vs Run 6 |
|---|---|---|---|---|---|---|
| 4a  baseline | 0.656 | 0.733 | 0.578 | 0.700 | 1.000 | −0.034 |
| 4b  no dropout | 0.667 | 0.647 | 0.686 | 0.603 | 1.000 | **+0.086** |
| **4c  wider** | **0.675** | 0.667 | 0.683 | 0.625 | 1.000 | +0.001 |
| 4d  shallower | 0.606 | 0.211 | 1.000 | 0.125 | 0.900 | −0.015 |

Run 6 baselines for Δ: 4a 0.690, 4b 0.581, 4c 0.674, 4d 0.621. Validation grade counts match
Run 6 as expected (crop doesn't change them): CDR 0.5 = 16/8, CDR 1 = 2/1, CDR 2 = none — so
the CDR 1 column (2 patches, 1 subject) is noise; CDR 0.5 (8 subjects) is the meaningful grade.

**What we saw**

- **Slightly worse at the top, and now a 3-way tie.** Best is 4c at 0.675, vs Run 6's best 4a
  0.690 (−0.015). Excluding the degenerate 4d, the field compressed to **0.656–0.675** —
  4a/4b/4c are separated by less than the ±0.1 validation noise, so the ordering is not
  meaningful. The tighter crop did **not** raise the ceiling.
- **Headline: the overfit demo (4b) got rescued.** 4b (no dropout) went from *worst* in Run 6
  (0.581, spec 0.49, over-calling CDR-positive) to **0.667 with balanced sens/spec
  (0.65 / 0.69)** — the biggest mover (+0.086). Interpretation: the tighter box removes much of
  the irrelevant tissue an unregularized net would otherwise memorize, so it overfits far less
  **even with no dropout**. The crop is quietly doing some of dropout's job. (Teaching caveat:
  this makes the "no-dropout ⇒ overfit" lesson *less* dramatic on the tight crop than on Run 6's
  looser one.)
- **Baseline 4a slipped a touch (0.690 → 0.656), via specificity** (0.622 → 0.578) — it now
  leans toward over-calling CDR-positive (sens 0.73 > spec 0.58). Within noise, but the spec
  dip is the one hint the box may be a little *too* tight (losing context around the head).
- **4c wider ≈ flat** (0.674 → 0.675). Extra capacity still buys nothing on data this small; it
  tops the table only by a hair.
- **4d shallower collapsed again** into "always CDR-negative" (sens 0.21 / spec 1.00, CDR 0.5 =
  0.125) — the same underfit degeneracy as Runs 6–7. The 2-block net is simply too weak,
  independent of crop size.

**Hypothesis check / takeaway**

- Predicted "tighter crop → higher specificity → higher bal." **Half-right:** specificity rose
  for 4b and 4c, but the baseline's spec *fell* and overall balanced accuracy did not beat
  Run 6. We've likely hit **diminishing returns** — 66×54 is focused enough that further
  tightening mostly reshuffles within noise, and may start clipping useful context (4a's spec).
  Worth eyeballing the new `slice_context/` images to confirm the box still covers the
  hippocampus and isn't cutting it.
- Net: the tighter, more honest ROI costs ~0.015 at the top but **de-fangs the 4b overfit
  collapse** — a real, interpretable effect. If the teaching goal is "show dropout mattering,"
  Run 6's looser crop demonstrates it better; if it's "an honest, tightly-focused ROI," this
  crop is fine at a small price.
- Ranking is noise-limited (3-way tie among 4a/4b/4c); to actually choose among them you'd need
  seed variation / k-fold error bars. 4d stays the clean underfit example. **Best-ever is still
  Run 3's 4c at 0.777** — unchanged.

---

## Run 9 — nudged the crop back up slightly (the "good" version) (2026-07-30)

**Change from Run 8:** grew the crop a touch to recover the context Run 8's over-tight box was
losing — `ap [80,146] → [80,148]` (+2 rows) and `lr_left [92,146] → [90,148]` (+2 px on each
L-R side): patch **66×54 → 68×58** (H×W). Still tighter than Run 6 (80×64). Everything else
identical to Run 8 (same four designs, age 60–79, `balance: label`, seed 42, `random_shift 3`,
`n_shifts 10`, 100 epochs, last-20 average). **This is the crop we're keeping.**

**Validation results** (mean of the last 20 epochs; acc = bal since val is 50/50):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 | Δbal vs Run 8 |
|---|---|---|---|---|---|---|
| **4a  baseline** | **0.719** | 0.772 | 0.667 | 0.744 | 1.000 | **+0.063** |
| 4b  no dropout | 0.668 | 0.694 | 0.642 | 0.656 | 1.000 | +0.001 |
| 4c  wider | 0.681 | 0.736 | 0.625 | 0.703 | 1.000 | +0.006 |
| 4d  shallower | 0.612 | 0.236 | 0.989 | 0.141 | 1.000 | +0.006 |

Grade counts unchanged (age-60 cohort): CDR 0.5 = 16/8, CDR 1 = 2/1 (noise), CDR 2 = none.

**What we saw**

- **Baseline 4a recovered strongly — best of the teaching runs (0.719).** The +2px-per-side
  nudge lifted 4a **+0.063** over Run 8, almost entirely by **restoring specificity**
  (0.578 → 0.667) — exactly the knob Run 8 flagged as suffering from an over-tight box. 4a now
  also leads very-mild detection (CDR 0.5 = 0.744, its best at age 60) with even sens/spec
  (0.77 / 0.67).
- **Goldilocks crop.** 68×58 beats both neighbours on the baseline: Run 8's too-tight 66×54
  (4a 0.656) and Run 6's looser 80×64 (4a 0.690). Specificity ties the story together — too
  loose lets irrelevant tissue in (Runs 4/6), too tight clips useful context (Run 8); this box
  sits between.
- **4b (no dropout) held its Run 8 gain** (0.668, balanced sens/spec) — the "a focused crop
  regularizes" effect persists; no overfit collapse like Run 6's 4b (0.581).
- **4c wider ≈ flat (0.681)** — still buys nothing over the 8-16-32 baseline. **4d shallower**
  still degenerate ("always CDR-negative", sens 0.24 / spec 0.99, CDR 0.5 = 0.14) — 2-block too
  weak, as ever.

**Takeaway**

- **Keeping this crop (68×58).** Best age-60 baseline so far (4a 0.719) and a clean teaching
  story: baseline > wider > no-dropout ≳ shallower, with 4d the underfit example. Best-ever is
  still Run 3's 4c 0.777 (different, looser-era setup), so the top line stands.
- **Don't trust these numbers.** ~16-subject validation, one fixed seed, peak-then-drop
  dynamics, and we've now hand-tuned the crop against this very set — the 4a/4c/4b spread
  (0.72 / 0.68 / 0.67) is within ±0.1 noise. The honest next step is **error bars, not more
  crop-tuning**: vary the seed and/or use **k-fold cross-validation** (now discussed in
  introduction.md) to see whether 4a's lead survives resampling.

---

## Run 10 — wider, older age band 65–85 (2026-08-01)

**Change from Run 9:** moved the cohort age band to **65–85** (`age_min` 60 → 65,
`age_max` 79 → 85). Motivation came from the step1/step2 **age histograms**: most CDR-positive
subjects fall in this range, so the band is where the data actually lives rather than where we
first guessed. Everything else identical to Run 9 — same four designs, the kept 68×58 crop
(`ap [80,148]`, `lr_left [90,148]`), `balance: label`, seed 42, `random_shift 3`, `n_shifts 10`,
100 epochs, last-20 average.

The wider band **grew the cohort**: **136 subjects** (was ~110), train 96 (48/48 by label,
30M/66F), validate 20 (10/10), test 20 (10/10). Validation grade counts improved too:
CDR 0.5 = **16 patches / 8 subjects**, CDR 1 = **4 / 2**, CDR 2 = none — so 10 CDR-positive
validation subjects, up from 9 in Run 9 and double Run 7's ~5.

**Validation results** (mean of the last 20 epochs; acc = bal since val is 50/50):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 | Δbal vs Run 9 |
|---|---|---|---|---|---|---|
| 4a  baseline | 0.639 | 0.667 | 0.610 | 0.700 | 0.537 | **−0.080** |
| 4b  no dropout | 0.626 | 0.552 | 0.700 | 0.541 | 0.600 | −0.042 |
| **4c  wider** | **0.661** | 0.688 | 0.635 | 0.722 | 0.550 | −0.020 |
| 4d  shallower | 0.650 | 0.440 | 0.860 | 0.438 | 0.450 | **+0.038** |

**What we saw**

- **Everything converged to ~0.63–0.66.** The whole field now sits inside a **0.035** spread,
  where Run 9 spanned 0.107 (0.612–0.719). The designs have become statistically
  indistinguishable from each other.
- **4a's Run 9 lead evaporated** (0.719 → 0.639, the biggest mover), and **4c nominally leads**
  — reversing Run 9's ranking. Do not read this as "4c is better": a 0.022 gap on 20
  validation subjects is nothing.
- **4d stopped being degenerate.** sens 0.44 / spec 0.86 is still lopsided, but far from Run 9's
  0.236 / 0.989 "always CDR-negative" collapse, and it now detects some very-mild cases
  (CDR 0.5 = 0.438, was 0.141). More data helped the weakest model most.
- **CDR 1 detection fell across the board** (0.45–0.60, was 1.000 in Run 9) — but Run 9's
  perfect score was 1 subject / 2 patches. With 2 subjects it is still noise; the Run 9 column
  was never meaningful.

**Interpretation (the honest read)**

- **This is Run 7's lesson at a gentler dose.** Raising `age_min` 60 → 65 partly age-matches the
  groups and removes some of the age signal the model was leaning on. Run 7 (60 → 70) collapsed
  to 0.53–0.60; here, raising `age_max` to 85 at the same time *added* 26 subjects, and the two
  effects roughly offset — a smaller drop (≈0.64) than Run 7's, on more data.
- **The Run 9 ranking did not survive re-rolling the cohort.** Run 9 explicitly warned that the
  crop had been hand-tuned against that validation set and the 4a/4c/4b spread was within noise.
  Change the cohort, and the ordering reshuffles. That is what validation overfitting looks like
  from the outside, and it is the strongest evidence yet for the Run 9 takeaway: **error bars,
  not more tuning.**
- **A wider band is still the right call** on data grounds — more subjects, more CDR-positive
  validation subjects, and a band chosen from the histograms instead of by guesswork. The lower
  headline number is a more honest one, not a worse setup.
- Next: vary the seed and/or run **k-fold cross-validation** before drawing any conclusion about
  which design wins. There is no design story to tell from a 0.035 spread.

**Note:** `config.yaml` has since moved to **68–82** in preparation for the next run; this run's
numbers are from 65–85 (confirmed in `outputs/splits.yaml`).

---

## Run 11 — age band 70–90: the confound removed, and the score collapses (2026-08-01)

**Change from Run 10:** age band **65–85 → 70–90**. Two architecture changes rode along, so
read the comparison carefully: **4c** was widened (16-32-64 → **32-64-128**, plus a 5×5 first
filter) and **4d** was trimmed (64 → **32-unit head**, dropout 0.6/0.2 → **0.4/0.2**). Only
**4a and 4b are unchanged**, so only their deltas isolate the age-band effect. Everything else
as Run 10: 68×58 crop, `balance: label`, seed 42, `random_shift 3`, `n_shifts 10`, 100 epochs,
last-20 average.

**The cohort got BIGGER, not smaller:** **140 subjects** (Run 10: 136). Train 98 (49/49),
validate 20 (10/10), test 22 (11/11). Validation grades: CDR 0.5 = **14 patches / 7 subjects**,
CDR 1 = **4 / 2**, CDR 2 = **2 / 1**.

**Why this band is different — the age gap inverts.** Mean age of each label within the band
(all labelled OASIS-1 `_MR1` subjects):

| band | CDR− mean age | CDR+ mean age | gap |
|---|---|---|---|
| 60–79 (Runs 8–9) | 69.8 | 72.4 | **+2.6 yr** |
| 65–85 (Run 10) | 74.5 | 75.6 | **+1.2 yr** |
| **70–90 (this run)** | **79.1** | **78.1** | **−1.0 yr** |

At 70–90 the CDR-positive group is, if anything, *slightly younger*. There is no age cue left
to lean on.

**Validation results** (mean of the last 20 epochs; acc = bal since val is 50/50):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 | CDR 2 | Δbal vs Run 10 |
|---|---|---|---|---|---|---|---|
| 4a  baseline | 0.523 | 0.615 | 0.430 | 0.496 | 0.863 | 0.950 | **−0.116** |
| 4b  no dropout | 0.540 | 0.605 | 0.475 | 0.489 | 0.850 | 0.925 | **−0.086** |
| 4c  wider *(new arch)* | 0.537 | 0.785 | 0.290 | 0.761 | 0.775 | 0.975 | n/a |
| 4d  shallower *(new arch)* | 0.551 | 0.495 | 0.608 | 0.393 | 0.600 | 1.000 | n/a |

**What we saw**

- **Everything is at chance.** 0.523–0.551 across four very different designs, on a 50/50 split
  where 0.50 is a coin flip. The spread is 0.028 — smaller than Run 10's already-meaningless
  0.035. There is no design story here at all.
- **The two clean comparisons both fell hard.** 4a **−0.116** and 4b **−0.086**, and these are
  the *only* two designs that didn't change, so the drop is attributable to the age band alone.
- **4c's apparent very-mild win is an artifact.** CDR 0.5 = 0.761 looks like the best in the
  table until you read sens 0.785 / spec 0.290 — it is simply answering "CDR-positive" most of
  the time. Same mirage as Run 7's 4d in the opposite direction. Read sens/spec, never a grade
  column on its own.
- **The severe grades are still detected.** 4a scores 0.86 on CDR 1 and 0.95 on CDR 2 while
  sitting at 0.496 — pure chance — on CDR 0.5. Caveat: that is **2 subjects and 1 subject**
  respectively, so it is suggestive, not evidence.

**Interpretation (the honest read)**

- **This is the strongest evidence yet that the age confound was doing the work.** Run 7 raised
  `age_min` 60 → 70 and also *halved* the cohort, so "less data" and "harder problem" could not
  be separated. **Run 11 separates them**: the cohort *grew* (136 → 140), the validation set is
  the same size, and the score still fell ~0.1 to chance. The one thing that changed is that the
  label-vs-age gap went from +1.2 yr to −1.0 yr.
- **But state the conclusion carefully.** "The CNN learns age, not CDR" overshoots. What the
  data supports: *once the age cue is removed, a single 2-D hippocampus patch at this
  resolution carries little detectable signal for **very mild** impairment* — which is where
  almost all of our positives are (7 of 10 validation positives are CDR 0.5). The residual
  skill on CDR 1 and 2 suggests the model does read atrophy when atrophy is gross; there is
  just very little of that in this cohort.
- **Do not over-read 0.523 vs 0.551 either.** With 10 CDR-positive validation subjects, the
  noise is ±0.1+. All four numbers are the same number.
- **Cross-check available.** `jupyter/nWBV.ipynb` runs the same question on the OASIS
  spreadsheet: how well does whole-brain volume predict CDR status, with and without age held
  fixed? If nWBV also collapses inside a narrow age band, that is independent confirmation that
  the confound — not the CNN — was carrying Runs 6–10.

**Takeaway**

- **Keep the 70–90 band for honest evaluation**, and treat every pre-Run-11 number as inflated.
  The bleak ~0.53 is the more truthful picture of what one 2-D slice can do here.
- The next question is not "which architecture", it is **"is there signal at all"** — answer it
  with error bars (repeat seeds / k-fold), not another design tweak. If chance survives
  resampling, the honest write-up is a negative result, which is a perfectly good teaching
  outcome for this project.

---

## Run 12 — age band 60–80: the confound restored, on purpose (2026-08-01)

**Change from Run 11:** age band **70–90 → 60–80**. Nothing else: same four designs (Run 11
already carried the widened 4c and the trimmed 4d, so **all four are directly comparable**),
same 68×58 crop, `balance: label`, seed 42, `random_shift 3`, `n_shifts 10`, 100 epochs,
last-20 average.

**This band was chosen deliberately, and it is not the honest one.** 60–80 restores most of the
age confound that Run 11 had removed. Mean age by label within the band: CDR− **70.8**,
CDR+ **73.3** — a **+2.5 year** gap, against Run 11's **−1.0**. It is kept as a teaching
artefact: a cohort that *looks* reasonable, scores well, and is quietly doing the wrong thing,
for students to find with `jupyter/age_confounder.ipynb`. **Do not present these numbers as the
project's result.** Run 11 is the honest one.

**Cohort:** 132 subjects (Run 11: 140). Train 92 (46/46), validate 20 (10/10), test 20 (10/10).
Validation grades: CDR 0.5 = **18 patches / 9 subjects**, CDR 1 = **2 / 1**, CDR 2 = none.

**Validation results** (mean of the last 20 epochs; acc = bal since val is 50/50):

| Model | bal | sens | spec | CDR 0.5 | CDR 1 | Δbal vs Run 11 |
|---|---|---|---|---|---|---|
| **4a  baseline** | **0.706** | 0.585 | 0.828 | 0.539 | 1.000 | **+0.183** |
| 4b  no dropout | 0.703 | 0.628 | 0.778 | 0.594 | 0.925 | **+0.163** |
| 4c  wider | 0.665 | 0.588 | 0.742 | 0.558 | 0.850 | **+0.128** |
| 4d  shallower | 0.661 | 0.417 | 0.905 | 0.353 | 1.000 | **+0.110** |

**What we saw**

- **Every design jumped, by a lot** (+0.11 to +0.18). Nothing about the *models* changed between
  Run 11 and Run 12 — only which people are in the cohort. That is the entire effect of the age
  band, isolated as cleanly as we are ever going to get it.
- **The designs are still indistinguishable from each other.** Spread 0.661–0.706 = **0.045**,
  on 20 validation subjects. As in Runs 10 and 11, there is no design story — the band moved
  everything together.
- **4b (no dropout) has caught the baseline** (0.703 vs 0.706). The "dropout is the main dial"
  narrative from Runs 1–3 does not survive here.
- **4a's error profile flipped, while its headline barely moved.** In the 60–79 run it was
  sens 0.772 / spec 0.667 (over-calling positive); now it is sens 0.585 / spec 0.828 (over-calling
  negative). Balanced accuracy went 0.719 → 0.706 — essentially unchanged — while
  **very-mild detection fell 0.744 → 0.539**. A stable headline can hide a completely different
  model underneath; read sens/spec and the grade columns, never the one number.
- **4d is still the degenerate one** (sens 0.417 / spec 0.905, CDR 0.5 = 0.353), earning its
  score from specificity as it has all along.

**Interpretation (the honest read)**

- **This is the confound, quantified end to end.** Runs 11 and 12 differ only in age band, and
  the score moves ~0.15. Any claim about what the network "learned" that does not first account
  for age is worthless.
- **Compare against age alone before believing anything.** In this band a logistic regression on
  **age with no image at all** reaches AUC ≈ 0.63, balanced accuracy ≈ 0.61
  (`jupyter/age_confounder.ipynb`). The CNN's 0.706 is roughly **+0.10** on that — a real
  margin, but note the same notebook shows the run-to-run spread at a 20-subject test set is
  about **±0.11**. So the CNN's advantage over "read the birth date" is roughly *one standard
  deviation of the noise*. Suggestive. Not established.
- **What would establish it:** repeat seeds or k-fold, so both the CNN number and the age-only
  number come with error bars, and then check whether the intervals separate. Everything else is
  guessing.

**Takeaway**

- **Config stays at 60–80 for the class**, deliberately, as the thing to catch. Run 11's 70–90
  numbers remain the honest picture of what one 2-D hippocampus patch can do.
- Best-ever is still Run 3's 4c = 0.777 (looser-era setup), so the top line is unchanged — and in
  hindsight that number was almost certainly riding the same age confound at `age_min: 60`.

### Sidebar — the accidental re-run of Run 9 (same day)

Before this run, `process.sh` was run believing the band was 60–80 while `config.yaml` still said
**60–79**. Two things came out of it worth keeping:

- **The pipeline is bit-for-bit reproducible.** 4a and 4b reproduced Run 9 exactly — 0.719 /
  0.772 / 0.667 / 0.744 / 1.000 and 0.668 / 0.694 / 0.642 / 0.656 / 1.000 — along with the grade
  counts. Same config + same seed ⇒ same numbers, to three decimals. That is a free regression
  check: if an "unrelated" change ever moves 4a on an unchanged cohort, something broke.
- **A perfectly controlled architecture comparison.** Because everything else was identical to
  Run 9, the only variables were the two redesigns:
  - **4c** 16-32-64 3×3 → 32-64-128 5×5: **0.681 → 0.651 (−0.030)**, with sens, spec and
    very-mild detection all down. **Going 12× wider made it worse** — the §14 prediction that
    extra capacity does not help on data this scarce, confirmed more sharply than the old 2×
    version ever managed.
  - **4d** head 64 → 32, dropout 0.6/0.2 → 0.4/0.2: **0.612 → 0.635 (+0.023)**, sens 0.236 →
    0.331, CDR 0.5 0.141 → 0.247. Lighter regularization pulled it back from "always
    CDR-negative", but it is still lopsided.
- Lesson for the log: **`outputs/splits.yaml` records the cohort actually used.** Check it
  against `config.yaml` before writing any run up.

---

<!-- Add the next run below, same format:
## Run 13 — <what changed> (date)
**Change from Run 12:** ...
... setup (config values) ... lineup (if redefined) ... results table ...
what we saw ... takeaway ... Update the "Best so far" line at the top if a new model wins.
-->
