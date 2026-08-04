# Introduction — teaching a computer to read a brain MRI

*Class notes. No prior knowledge of AI, programming, or neural networks is assumed. This section was written entirely by Claude, and developed alongside the code.*

These notes explain the **ideas** behind this project and, just as important, **why we
made each choice**. They are the companion to [readme.md](readme.md), which is the
practical "how to run it and what every setting does" guide. Read these notes for
understanding; use the readme when you sit down to run the code.

Watch for the **"Key idea"** boxes: those are the sentences
worth remembering. At the very end there are a few **check-your-understanding**
questions and a **glossary**.

---

## 1. The medical problem we're attacking

**Alzheimer's disease** is the most common cause of dementia. It slowly damages the
brain, and one of its earliest and most visible effects is that a small structure
called the **hippocampus** shrinks (atrophies).

- The **hippocampus** is a seahorse-shaped structure — one in each half of the brain,
  tucked in the temporal lobe near your ears. It is central to forming new memories,
  which is why memory loss is an early Alzheimer's symptom.
- An **MRI** (magnetic resonance imaging) scan is, for our purposes, a **3-D grayscale
  photograph of the brain**. Instead of flat pixels it is made of **voxels** (3-D
  pixels); in our data one voxel represents 1 mm of brain.

If the hippocampus visibly shrinks in Alzheimer's, then maybe a computer can learn to
look at a brain scan and tell whether that shrinkage is present. That is exactly our
task, stated as narrowly and concretely as possible:

> **Key idea.** Given **one 2-D slice** of a brain MRI, output a **single yes/no
> guess**: **CDR-positive** (1) or **CDR-negative** (0). This kind of yes/no prediction is
> called **binary classification**.

To *teach* the computer, we need the correct answer for each scan — the **label**. Ours
comes from the **CDR** (Clinical Dementia Rating), a score a clinician assigns:

| CDR | Meaning | Our label |
|---|---|---|
| 0 | no impairment | 0 (CDR-negative) |
| 0.5, 1, 2 | very mild → moderate dementia | 1 (CDR-positive) |
| (blank) | never assessed (e.g. young subjects) | excluded |

We name the classes by their CDR status — **CDR-negative** (CDR 0) and **CDR-positive**
(CDR ≥ 0.5) — rather than by a disease or severity word: the label is a clinical-severity
score, not a biomarker-confirmed diagnosis, and labelling a person by a disease term is
stigmatizing. Note that label 1
**pools three severities** — very mild (0.5), mild (1), and moderate
(2) — into one CDR-positive class. That's a real simplification: catching moderate
atrophy should be easier than very-mild. step5 therefore reports validation accuracy
**broken down by CDR grade** so you can see this (with a caution: only a handful of
validation subjects fall in each grade, so those per-grade numbers are noisy).

### What our label actually is — and isn't

Look again at what that label is made of, because the distinction below is the single easiest
thing to get wrong when reading a paper in this field — including your own.

**CDR is a rating of observed impairment.** A clinician arrives at it by structured interview,
talking both with the person and with someone who knows them well, and scoring memory,
orientation, judgment, community affairs, home life, and personal care. It describes how
someone is *doing*.

**Alzheimer's disease is a pathology** — amyloid plaques and tau tangles in brain tissue. For
most of the history of the field it could only be confirmed at autopsy; today it can be
inferred from biomarkers (amyloid PET, spinal fluid, and newer blood tests). **OASIS-1
contains none of those.**

These are not the same variable, and they come apart in *both* directions:

- **Impairment without Alzheimer's.** A CDR of 0.5 or 1 can arise from vascular disease, Lewy
  body dementia, depression, thyroid disease, medication side effects, or poor sleep. Our label
  calls all of them CDR-positive, because that is what CDR measures.
- **Alzheimer's without impairment.** In *preclinical* Alzheimer's the pathology is already
  accumulating while cognition still tests normal. That person is CDR 0 — and we label them
  CDR-negative, alongside people with no pathology at all.

So a model trained here is, at its very best, learning to predict **a clinician's rating**. It
is not learning to detect a disease, because we never showed it one.

**This is normal, and it is the interesting part.** Most machine-learning examples you meet
have labels that are correct by definition: a photo of a cat is labelled "cat", and there is
nothing further to check. Biomedical ground truth is rarely like that. The thing you care about
is invasive to measure, or expensive, or only available post-mortem, or simply wasn't recorded
in the study you have. So you train on a **proxy** — a different, available variable that you
hope tracks the one you want. Recognizing that substitution and saying so out loud is not a
caveat you bolt on at the end; it is part of doing the work competently.

Three consequences you can actually feel in this project:

1. **It puts a ceiling on accuracy.** A model cannot be more consistent than the labels it was
   trained on. Where two clinicians would disagree — is this person CDR 0 or CDR 0.5? — there
   is no fact for the network to recover. This is one reason ≈0.78 is a realistic target and
   0.99 would be a sign that something is wrong rather than something is working (§11).
2. **The model inherits whatever the proxy encodes**, including things you did not intend. If
   the CDR-positive people in our sample are also the older people, "predict CDR" and "predict
   age" become partly the same task — which is exactly the confound we run into in §10.
3. **It constrains what you may claim.** "Predicts CDR status from a hippocampus patch" is
   supportable. "Detects Alzheimer's disease" is not, and the gap between those two sentences
   is where a great deal of over-claiming in medical AI lives.

> **Key idea.** We are not predicting Alzheimer's disease. We are predicting a clinician's
> rating of impairment, which is related to it but not the same thing. Know what your label
> really is, and describe your results in those terms — not in the terms you wish you had.

The scans come from **OASIS-1**, a public dataset from Washington University: hundreds
of people, each with an MRI and demographic/clinical information.

---

## 2. What is machine learning?

Before **machine learning**, researchers wrote a list of rules a human wrote: *"if this, do that."* That
works when you can state the rules. But nobody can write down the exact rule for "this
blur of gray pixels means a shrunken hippocampus." The pattern is too subtle and varies
from person to person.

**Machine learning** turns the problem around. Instead of writing rules, you show the
program many examples, each paired with the right answer, and the program
**adjusts itself** until it can produce the right answer on its own.

- The "examples" here are hippocampus images; the "right answers" are the CDR-negative /
  CDR-positive labels. Learning from labeled examples like this is called **supervised
  learning**. (In unsupervised learning, the computer groups similar examples and identifies disparate groups.)
- Inside the program are thousands of adjustable numbers called **parameters** or
  **weights**. "Learning" means slowly changing those numbers to improve the 
  program's guess of the labels better.

### The one thing that makes this hard: generalization

We do **not** want a program that just memorizes the exact images it was shown. A program that memorizes will score perfectly on learned images but will be useless on new data. If we want a
program that can generalize, it needs to correctly identify data it has never seen. Achieving **generalization** drives almost every decision in this project.

To measure generalization honestly, we split our people into **three groups**:

- **Training set** — the examples the program actually learns from.
- **Validation set** — held back during learning. We peek at it to check whether the
  program is generalizing and to choose between different designs.
- **Test set** — locked in a drawer until the very end, looked at **once**, to get an
  unbiased final grade.

Think of studying for an exam:

> **Analogy.** The **training set** is the homework you practice on. The **validation
> set** is a practice exam you take to see how you're doing (and decide what to study
> next). The **test set** is the real, final exam. If you had seen the final exam's
> questions while studying, your grade wouldn't mean anything — which is exactly why the
> test set must stay untouched until the end.

> **Key idea.** Good training performance is easy and not the goal. Good **validation/
> test** performance — doing well on *new* people — is the goal.

---

## 3. Images are grids of numbers

A grayscale image is a **grid of numbers**, one per pixel, where the number is a
brightness from **0 (black)** to **255 (white)**. That's all an image is to a computer.
Our hippocampus patches are grids of roughly **58 × 68** such numbers.

A tiny 3 × 3 corner of an image might look like this (dark background on the left,
bright tissue on the right):

```
  10   12   200
   9  180   210
 190  205   215
```

Everything a neural network does — detecting an edge, deciding "CDR-positive" — is just
arithmetic on grids of numbers like this.

---

## 4. Neural networks and CNNs

### 4.1 A single neuron

The basic unit is a **neuron**: it takes several input numbers, multiplies each by a
weight, adds them up, and passes the result through a simple "activation" step. Stack
many neurons in **layers** — the outputs of one layer feed the next — and you get a
**neural network**. The weights are the parameters that learning adjusts.

### 4.2 Why plain networks are wrong for images

If we connected every pixel to every neuron, an 80 × 64 image (5,120 pixels) would need
millions of weights in the first layer alone, and the network would treat pixel (1,1)
and pixel (30,40) as unrelated — throwing away the fact that **nearby pixels form
shapes**. Images have *local structure*, and we should use it.

### 4.3 Convolution: the key idea of a CNN

A **convolution** uses a small **filter** (also called a kernel) — say 3 × 3 weights —
that **slides across the image**. At each position it multiplies the filter by the
pixels underneath, sums them, and writes one output number. The same filter is reused
at every position, so it learns to detect **one pattern anywhere** in the image (an
edge, a bright blob, a texture).

**Worked example.** Take this vertical-edge filter and this image patch:

```
filter          image patch          multiply & sum
-1 0 +1          10  12  200          (-1·10)+(0·12)+(1·200)
-1 0 +1           9 180  210        + (-1·9) +(0·180)+(1·210)
-1 0 +1         190 205  215        + (-1·190)+(0·205)+(1·215)
                                     = 190 + 201 + 25 = 416  (large!)
```

The output is a **large positive number** because the right side is much brighter than
the left — the filter "fired" on a vertical edge. Over a flat region the same filter
sums to near zero. So one filter turns the image into a map of "where my pattern is."

- A convolution **layer** has *many* filters. Each produces its own output grid, called
  a **channel** (or feature map). If a layer has 32 filters, it outputs 32 channels.
- **Stacking** convolution layers builds a hierarchy: the first layer finds edges; the
  next combines edges into corners and blobs; deeper layers assemble those into
  meaningful structures. This is why **C**onvolutional **N**eural **N**etworks are the
  standard tool for images.

> **Key idea.** A convolution is a small, reusable pattern-detector. Depth lets simple
> patterns combine into complex ones.

### 4.4 Pooling: shrink as you go

After a convolution we usually **pool** to shrink the grid. **Max-pooling** with a
2 × 2 window keeps only the largest value in each 2 × 2 block:

```
 3  9 | 1  2          max of each 2x2 block
 4  6 | 0  8     -->    9  8
------+------           7  5
 7  2 | 5  1
 1  0 | 3  4
```

A 4 × 4 grid becomes 2 × 2 — a quarter of the size. Pooling keeps the strongest
responses while making the network cheaper and less sensitive to the exact position of
a feature. So an image flows through a CNN getting **smaller in space but richer in
channels**, e.g. `1 → 32 → 64 → 128` channels while the grid halves at each step.

### 4.5 Two small helpers: ReLU and BatchNorm

- **ReLU** ("rectified linear unit") is the activation step: it replaces every negative
  number with 0 and keeps positives unchanged (`ReLU(-3)=0`, `ReLU(5)=5`). Without a
  step like this the whole network would not be able to represent complex patterns.
- **BatchNorm** rescales the numbers flowing between layers so they stay in a sane range
  (roughly mean 0, spread 1). It mainly makes training faster and more stable.

### 4.6 From a grid to a decision: GAP, the head, and the logit

After the convolution blocks we have, say, 128 channels each still a small grid. Two
final steps turn that into one yes/no answer:

1. **Global average pooling (GAP)** collapses each channel's whole grid to a **single
   average number**. So "128 channels of size 26 × 22" becomes just **128 numbers**.

   > **Why GAP matters — a parameter comparison.** The old-fashioned alternative is to
   > *flatten* the grid: 128 × 26 × 22 = **73,216** numbers, and a layer connecting
   > those to 64 neurons would need ~4.7 **million** weights. GAP gives 128 numbers, so
   > the same layer needs ~8 **thousand**. GAP also makes the network accept **any**
   > patch size, since the grid is always averaged down to one number per channel.

2. **The classifier head** is a couple of ordinary neuron layers that turn those 128
   numbers into **one output number** called the **logit**. A big positive logit means
   "confidently CDR-positive," a big negative logit "confidently CDR-negative." Passing the
   logit through the **sigmoid** function squashes it to a probability between 0 and 1. We
   predict **CDR-positive when the logit is above 0** (probability above 0.5).

### 4.7 A full pass through design 4a

Putting it together, here is what happens to one hippocampus patch in the reference
network (design **4a**, channels `8→16→32`), with the shape at each step
(`channels × height × width`). The input size below is a **round example** (80 × 64) chosen
so the halving stays tidy; our real patch is ~58 × 68, and — as the note after the table
says — GAP makes the exact height/width irrelevant:

```
input patch            1 x 80 x 64
block 1  conv 1->8,   BN, ReLU, maxpool   ->  8 x 40 x 32
block 2  conv 8->16,  BN, ReLU, maxpool   -> 16 x 20 x 16
block 3  conv 16->32, BN, ReLU, maxpool   -> 32 x 10 x  8
global average pooling                    -> 32 x  1 x  1
flatten                                   -> 32 numbers
dropout, linear 32->64, ReLU, dropout     -> 64 numbers
linear 64->1                              ->  1 logit
```

(The exact heights/widths don't matter — GAP absorbs them — which is why every design
in the sweep works without hand-tuning sizes.)

---

## 5. How a network actually learns

We have a network full of random weights making random guesses. How does it improve?

1. **Loss — measuring wrongness.** A **loss function** scores how far the guesses are
   from the truth (lower = better). For a yes/no question the standard is
   **binary cross-entropy** (`BCEWithLogitsLoss` in the code). Intuitively it punishes
   confident wrong answers a lot and correct confident answers a little.

2. **Gradient descent — rolling downhill.** For each weight, calculus tells us which
   direction (up or down) would *reduce* the loss, and by how much (the **gradient**).
   We nudge every weight a little in the loss-reducing direction, then repeat.

   > **Analogy.** Imagine a hiker on a foggy hillside trying to reach the valley. They
   > can't see far, but they can feel which way is downhill under their feet, and take a
   > small step that way. Repeat thousands of times and they descend. The network is the
   > hiker; the loss is the height; the gradient is the slope under its feet.

3. **Learning rate — how big a step.** The **learning rate** (we use `1e-4` = 0.0001) is
   the size of each step. Too **big** and the hiker overshoots the valley and bounces
   around; too **small** and they crawl and it takes forever. Getting it right matters a
   lot (see §10).

A few more words you'll meet:

- **Batch** — we don't feed one image at a time; we process a small group (here 32) and
  average their gradients per step. That's a **batch**.
- **Epoch** — one full pass over all the training images. Training runs many epochs.
- **Optimizer** — the exact recipe for turning gradients into weight updates. We use
  **AdamW**, a robust modern default.
- **Weight decay** — a gentle pull that keeps weights from growing large, which nudges
  the model toward simpler solutions (a form of regularization, see §8).

---

## 6. Measuring success — and why "accuracy" can lie

**Accuracy** = the fraction of images the model labels correctly. It's the obvious
metric, but it can be dangerously misleading when the two classes are unequal.

> **Example.** Suppose a group is 90 % CDR-negative, 10 % CDR-positive. A lazy model that
> *always* says "CDR-negative," no matter the image, gets **90 % accuracy** — while being
> completely useless (it never catches a single impaired person).

To see through this, we lay out the four possible outcomes in a **confusion matrix**.
Say we evaluate 40 images (20 truly CDR-positive, 20 truly CDR-negative) and the model gets:

```
                    predicted CDR+   predicted CDR−
truly CDR+              12  (TP)           8  (FN)
truly CDR−               3  (FP)          17  (TN)
```

- **TP** true positives (12): impaired, correctly caught.
- **FN** false negatives (8): impaired, wrongly cleared — the dangerous misses.
- **FP** false positives (3): CDR-negative, wrongly alarmed.
- **TN** true negatives (17): CDR-negative, correctly cleared.

From these we compute three honest metrics:

- **Sensitivity** (recall) = TP / (TP + FN) = 12/20 = **0.60** — of the truly CDR-positive,
  how many did we catch?
- **Specificity** = TN / (TN + FP) = 17/20 = **0.85** — of the truly CDR-negative, how many
  did we correctly clear?
- **Balanced accuracy** = average of the two = (0.60 + 0.85)/2 = **0.725** — the fair
  single headline.

Why balanced accuracy is fair: the always-say-CDR-negative model has sensitivity 0 and
specificity 1, so its balanced accuracy is (0 + 1)/2 = **0.5** — exactly "no better than
a coin flip," as it should be.

> **Key idea.** We keep our groups **50/50 CDR-negative/CDR-positive**, so chance is **0.50**
> and plain accuracy already equals balanced accuracy. We still log **sensitivity** and
> **specificity** because they reveal *which way* a model leans (does it miss impaired
> people, or cry wolf on CDR-negative ones?).

One caveat sits underneath all four numbers: they measure **agreement with the label**, and the
label is a clinician's rating, not a ground truth (§1). A "false positive" may be a case where
the model saw something real and the rating didn't, and we have no way here to tell those apart.

All of these judge the model at a single **0.5 threshold**; a threshold-free alternative,
**AUC** (area under the ROC curve), summarizes performance across *all* thresholds — a natural
extension of the plots (see §14).

The step4 scripts print all four every epoch and save them to a CSV; step5 also reports
each design's average over the last 20 epochs.

### Cross-entropy loss and Gibbs' inequality, in more depth

§5 introduced cross-entropy as "punishes confident wrong answers a lot, correct
confident answers a little" — enough to train with. But it isn't an arbitrary choice; it
comes from information theory, and the same idea resurfaces in step7's "bits of
information" diagnostic (`step7-stack-predictors.py`). This subsection works through the
actual formula and the inequality that makes step7's bits numbers meaningful rather than
made up.

**The formula, for one prediction.** Say the true label is `y` (1 for CDR-positive, 0 for
CDR-negative) and the model's predicted probability of CDR-positive is `p`. The per-example
cross-entropy loss (this is exactly what `BCEWithLogitsLoss` computes, using the natural
log `ln`, which gives units called **nats**) is:

```
loss = -[ y * ln(p) + (1 - y) * ln(1 - p) ]
```

Only one of the two terms is ever "active": if `y = 1` the loss is `-ln(p)`; if `y = 0`
it's `-ln(1 - p)`. Either way, the loss is `-ln(probability the model assigned to the
TRUE answer)`. That single fact is the whole intuition from §5, made precise:

- A confident, correct prediction (`p = 0.99` when `y = 1`) gives `-ln(0.99) ≈ 0.01` —
  almost no penalty.
- A confident, WRONG prediction (`p = 0.01` when `y = 1`) gives `-ln(0.01) ≈ 4.6` — a
  large penalty, and as `p → 0` the loss shoots to infinity. There is no probability so
  small that being confidently wrong is "cheap."
- An honest "I don't know" (`p = 0.5`) always gives `-ln(0.5) ≈ 0.69`, regardless of the
  true answer — admitting uncertainty costs a fixed, moderate amount.

Training averages this loss over a batch and nudges the weights to shrink it (§5):
smaller loss means the model's probabilities are, on average, closer to being right, and
confidently so.

**From one prediction to a whole distribution.** Instead of a single 0/1 label, imagine
CDR status has some true underlying rate of being positive in a population, `q` (e.g.
0.5, since our splits are deliberately balanced — §6 above). The model outputs its own
rate, `p`. Averaged over many examples, the formula above computes exactly the **cross-entropy
between `q` and `p`**, written `H(q, p)`. Two special cases of that formula matter:

- **Entropy**, `H(q) = H(q, q)` — the cross-entropy of a distribution *with itself*. It's
  the fewest units needed, on average, to describe an outcome drawn from `q`. A
  population that's 100% one class has `H(q) = 0` — no uncertainty, nothing to describe.
  A 50/50 population has the *maximum* possible entropy for two classes — exactly **1
  bit** (using `log` base 2, the natural unit for "how many yes/no questions does this
  take on average") — you genuinely need a full yes/no answer every time.
- **Cross-entropy**, `H(q, p)` — what you get when you *measure* outcomes from the true
  distribution `q` but *describe* them assuming a possibly-wrong distribution `p` (the
  model's guess). This is what the training loss actually computes, batch by batch.
  (The base of the logarithm only changes the unit — natural log gives nats, what
  PyTorch computes; log base 2 gives bits, what step7 reports. Gibbs' inequality below
  holds in either base; switching base just rescales every term by the same constant.)

**Gibbs' inequality** is the fact that ties these two together:

> `H(q, p) ≥ H(q)`, always — with equality only when `p` and `q` are the exact same
> distribution.

In words: **you can never describe outcomes more efficiently using the wrong
distribution than using the true one.** Guessing anything other than the real
probabilities can only cost extra, never save anything. That extra cost has a name — the
**Kullback-Leibler (KL) divergence**, `KL(q || p) = H(q, p) - H(q)` — and Gibbs'
inequality is exactly the statement `KL(q || p) ≥ 0`.

> **Why it's true (short version).** `KL(q || p)` is the average, over outcomes drawn
> from `q`, of `-log(p/q)`. The function `-log(x)` curves upward (it's *convex*), so by
> Jensen's inequality, the average of `-log` of something is at least `-log` of the
> average of that something. Averaging `p/q` weighted by `q` just gives back the sum of
> `p`, which is 1 (probabilities always sum to 1) — so `KL(q || p) ≥ -log(1) = 0`. Equality
> needs `p/q` to be the *same* constant for every outcome, which — since both distributions
> sum to 1 — forces that constant to be exactly 1, i.e. `p = q`.

**A worked example, in bits.** Suppose the true rate is `q = 0.5` CDR-positive (our
balanced splits), but a miscalibrated model always outputs `p = 0.9` no matter the input
— it's overconfident. Its cross-entropy, averaged over many true 50/50 draws (using
`log` base 2 throughout, so the answer is in bits):

```
H(q, p) = -0.5 * log2(0.9) - 0.5 * log2(0.1)
        ≈  0.5 * 0.152    +  0.5 * 3.322
        ≈  1.74 bits
```

Compare to the true entropy, `H(q) = 1` bit (the same formula with `p = q = 0.5`).
**1.74 > 1**, exactly as Gibbs' inequality demands: the overconfident, miscalibrated
model's cross-entropy overshoots the true uncertainty. The only `p` that would bring
`H(q, p)` back down to exactly 1 bit is `p = 0.5` itself — the true distribution.

**Why any of this matters here:**

- **It's why cross-entropy is the loss, and not just accuracy or something ad hoc.**
  Because of Gibbs' inequality, cross-entropy is *minimized*, over every possible choice
  of `p`, *exactly* when `p = q` — the true probabilities. Training a network to
  minimize cross-entropy is therefore, in principle, training it to output *calibrated,
  honest probabilities* of CDR-positive, not merely the right yes/no answer. Plain
  accuracy has no such property — it's flat almost everywhere, so it gives gradient
  descent nothing to climb down.
- **It's the backbone of step7's "bits of information" diagnostic.**
  `step7-stack-predictors.py` fits a small logistic regression on age, nWBV, and the
  CNN's score, then measures each one's cross-entropy on held-out VALIDATE data (never
  the data it was fit on). Because of Gibbs' inequality, that measured cross-entropy can
  never come in *below* the true conditional entropy of CDR status given that predictor
  — so `(baseline entropy) − (measured cross-entropy)` is a guaranteed **lower bound** on
  how many bits of real information the predictor carries about CDR status, never an
  overestimate. It is not the exact number of bits — a poorly calibrated model
  understates it further — but it can never lie in the *other* direction. See the
  summary step7 prints (`outputs/step7-stacking_summary.txt`) for the actual numbers and
  their confidence intervals.

> **Key idea.** Cross-entropy is minimized only by telling the truth, probabilistically
> speaking — Gibbs' inequality is the reason why. That single fact is both why
> cross-entropy makes a good training loss, and why, measured out of sample after
> training, it can be turned into a lower bound on how much real information a predictor
> carries.

---

## 7. The central challenge: overfitting

There are two opposite ways to fail:

- **Underfitting** — the model is too simple or weak to capture the pattern, so it does
  poorly even on the training data. *(The student who didn't study and fails the
  homework.)*
- **Overfitting** — the model is powerful enough to **memorize** the training images
  (near-perfect training scores) but it latched onto quirks of *those specific brains*,
  so it flops on new people (poor validation scores). *(The student who memorized last
  year's exam answers word-for-word and is lost when the questions change.)*

Balancing these two is the **bias–variance trade-off**, and it is the whole game when
data is scarce — which is exactly our situation.

**Why small data makes overfitting so easy — and a subtle trap.** We have on the order
of a hundred subjects. Worse, the several slices (and left/right patches) we take from
one person are **not independent**: they are near-duplicate views of the *same* brain
with the *same* label.

> **Worked point.** If we have 110 subjects and take, say, 12 patches each, that's 1,320
> training images — but only about **110 independent pieces of information**. The images
> *look* like a lot of data; they are not. The **effective sample size ≈ the number of
> subjects.**

> **Key idea — the most important sentence in these notes.** On a dataset this small,
> **data is the bottleneck, not the network.** A fancier or bigger model cannot invent
> information that isn't in the data — it will just memorize harder.

---

## 8. The toolbox for coping with limited data

Given fixed, scarce data, here are the levers — and where each appears in this project:

- **More data / a bigger cohort.** The most direct fix. Our `cohort.balance: label`
  setting nearly doubles the usable subjects (see §10).
- **Data augmentation.** Create new-ish examples by slightly perturbing existing ones —
  our optional random **shifts** of the crop box — so the model can't rely on the exact
  pixel positions. (`apply_random_shifts` in the config.)
- **Dropout.** The single most important regularizer in this project — during training,
  randomly switch **off** a fraction of the internal numbers on each step so the network
  can't over-rely on any one feature. It's essential enough here that it gets its **own
  section next (§9)**, where our designs' dropout settings are compared.
- **Weight decay.** Keeps weights small → simpler model (already on, `1e-4`).
- **Smaller / narrower models.** Fewer parameters means less room to memorize — but go
  too far and the model **underfits**. The sweep includes tiny models precisely to see
  this edge.
- **Early stopping.** The best validation score often comes *early*; training longer
  after that just deepens overfitting, so you keep the best epoch rather than the last.
- **Cross-validation.** Don't lean on a single train/validation split — on ~110 subjects one
  split is noisy. **k-fold** rotates the validation set so every subject is checked once. It
  doesn't reduce overfitting; it tells you how much to *trust* the number (see the note next).

**A note on trusting the number — cross-validation.** This pipeline uses **one fixed
train/validation/test split**. That's simple to teach, but fragile on ~110 subjects: swap a
few subjects between train and validation and the reported balanced accuracy can wobble by
**±0.1** — which is why the lab notes keep saying "within noise," and why hand-tuning a knob
(like the crop) against that one split is a form of overfitting *to the validation set*. The
standard cure is **k-fold cross-validation**:

- Split the **subjects** (never slices — the same subject-level rule that prevents leakage,
  and the effective-sample-size point from §7) into *k* equal folds.
- Train *k* times; each time hold out a different fold for validation and train on the rest.
- Report the **mean of the *k* scores and their spread** (an error bar), not a single point.

With `k = 5`, every subject is validated exactly once, so you use *all* the data and get a
**mean ± std** instead of one fragile number. Common variants: **stratified** k-fold keeps
each fold class-balanced (important here, so chance stays 0.50); **repeated** k-fold reshuffles
and reruns for tighter error bars; **leave-one-subject-out** takes *k* = number of subjects
(maximal data use, maximal compute); and **nested** cross-validation adds an *inner* loop to
choose hyper-parameters/architecture so that tuning doesn't leak into the final estimate — the
honest way to decide whether 4a really beats 4c. The price is compute (you train *k*× as
often) and a more complex pipeline, which is why this teaching project keeps to one split — but
when you need to trust a *difference* rather than just see one, cross-validation is the right
tool.

---

## 9. Dropout — the regularizer that matters most here

Every lever in §8 helps, but on data this scarce **dropout** is the one that moves the
needle most, so it's worth understanding on its own.

**What it does.** On each training step, dropout looks at the numbers flowing out of a
layer and, at random, sets a fraction `p` of them to **0** for that step (the survivors
are scaled up so the totals stay comparable). A different random subset is dropped every
step. That's the whole mechanism.

> **Worked picture.** After global average pooling design 4a has **32 numbers** (§4.7).
> With `p = 0.5`, on this step roughly **16 of them are randomly zeroed** and the rest
> pass through; on the next step a *different* ~16 are zeroed. The network never gets to
> lean on the same fixed set of features two steps running.

**Why that fights overfitting.** If any single feature might vanish on any given step, the
network **can't lean on it** — it's forced to spread the decision across many features
that each carry a little of the signal. Equivalently, you're training a whole
**ensemble** of slightly different "thinned" sub-networks that all share weights and must
agree; averaging over an ensemble is a classic way to generalize better.

**Two things that trip people up:**

- **Training only.** Dropout is **on while training, fully off while evaluating** (at
  eval every unit is used). This is one reason validation behaves differently from
  training — and why turning dropout *up* lowers training accuracy but can *raise*
  validation accuracy.
- **It removes no parameters.** The model is exactly the same size with `p = 0.2` or
  `p = 0.8`; dropout only *regularizes* how the existing weights are trained. (Contrast
  with making the net *narrower*, which genuinely removes parameters — §8.)

**Our two dials.** Each design has dropout in two places in its head (see the 4a pass in
§4.7): `dropout1` right after GAP, and `dropout2` after the 64-unit layer. We write them
as a pair — the notation **"0.8 / 0.4"** means `p1 = 0.8`, `p2 = 0.4`. The first (on the
post-GAP bottleneck) is the stronger lever.

**Choosing `p` is a dial, not a "more is better."**

- Too **little** dropout → the net memorizes the training brains (overfits).
- Too **much** dropout → the net can't fit the signal at all (**underfits**), and on a
  very small network it can **collapse to a trivial answer** — always guessing one class.

You can watch this in the sweep: designs **4a and 4b** are the same 8-16-32 net and differ
*only* in dropout (4a uses 0.6 / 0.2, 4b none), so the gap between them is the effect of
dropout by itself — expect 4b to overfit (training accuracy climbs high, validation lags).
The running results in `internal/lab-notes.md` also show the *other* failure mode: an 8-16-32 net
pushed to **high** dropout (0.8 / 0.4) saw its *specificity* collapse to 0.445 — it started
crying wolf, calling almost everyone CDR-positive — and a tiny net with high dropout **plus**
augmentation over-regularized the other way into "always CDR-negative." Same lesson from
opposite sides.

> **Key idea.** Dropout is the main tuning dial in this project, and its right value
> depends on the model's size and the amount of data: **match the amount of dropout to
> how much the model could otherwise memorize.** Small, lean nets need *less* of it, not
> more.

---

## 10. The design choices in this project, and why

Almost every choice below is a response to §7: *small, correlated data.*

**2-D slices, not the full 3-D brain.** Serious research (including the tutorial we
follow) feeds the whole 3-D volume to a 3-D CNN. We deliberately use flat 2-D slices
because they are far simpler to understand, quicker to train, and let a beginner see the
entire pipeline end-to-end. It is a weaker signal — an honest trade for clarity.

**Transverse slices at the hippocampus.** Alzheimer's shows up in the hippocampus, so
that's where we look, taking horizontal ("transverse") cross-sections at the height
where the hippocampus sits. We found that height empirically — the hippocampus lives
around **Talairach z ≈ −15 to −25 mm**, which in our atlas is a few slices below the
brain's middle — and confirmed it by actually rendering the slices and looking.

**Crop a small box around the hippocampus — left *and* right.** Feeding the whole slice
makes the network waste effort on irrelevant tissue and black border. Cropping a small
box focuses it on the disease-relevant region. We crop **both** hippocampi and save them
as **separate examples**, which **doubles** the training data for free. To *see* how big
the box is and where it lands, step 3 writes **context images** to
`outputs/slice_context/` for a few subjects — the full axial slice with a rectangle around
each crop window (train shows all the random-shift boxes; val/test the single box). If the
boxes look too large or off-centre, that's your cue to tighten `hippocampus.ap` /
`lr_left`.

> **Why don't we flip the right patch first?** The right box is the left box *mirrored*
> about the midline, and we take the pixels as-is — we do **not** horizontally flip them. So
> the left and right hippocampi reach the *same* network as **mirror images of each other**.
> A fair question: shouldn't we reflect one so both face the same way? We don't, on purpose.
> A CNN is **translation-equivariant** (it finds a learned pattern wherever it sits) but it is
> **not** reflection-invariant — so the network does have to learn the hippocampal pattern in
> both left- and right-handed poses. That's cheap, and actually helpful: the signal we care
> about is **atrophy** (size, shape, surrounding CSF), which looks the same in a mirror, so
> feeding both orientations acts like free **reflection augmentation**. Flipping one to align
> them would be equally valid — just unnecessary — and keeping each patch a *faithful* crop of
> the real slice is what lets step 6 draw its Grad-CAM boxes back onto the actual brain (§13).
> (If you'd rather collapse the two into one answer per person, see "Vote per subject" in §14.)
>
> This box is about the *static* left/right crop-box mirroring, which is always on and applies
> to every split. Separately, TRAIN patches can now *also* be randomly flipped and rotated as
> an explicit, optional augmentation (`hippocampus.apply_random_reflections_rotations` — see
> §14's "Augment with rotations and flips," now implemented). That one deliberately breaks the
> "faithful crop" property above for TRAIN only — validation/test are untouched, so step 6's
> Grad-CAM boxes stay accurate there regardless.

**Split by subject, not by slice.** This one prevents a silent disaster called **data
leakage**. Because a person's patches are near-duplicates, if some of their patches
landed in *training* and others in *test*, the model could "recognize the individual
brain" and score falsely high — it would look brilliant and be worthless on real new
patients. We assign each **whole subject** to one split, so this cannot happen.

**Three balancing modes; default `label`.** Dementia risk depends on age and sex, so a
careless dataset could let the model cheat on a **confound** (e.g. guess from sex
instead of brain shape). The strictest option (`strict`) forces equal Male/Female ×
CDR-negative/CDR-positive groups — fairest, but limited by the rarest group (elderly
*CDR-negative men* are scarce), which throws away roughly half the subjects. We noticed the
data is *already* nearly balanced by label, so `label` mode — equal CDR-negative/CDR-positive,
sex left free — nearly **doubles** the usable subjects while keeping chance at exactly 0.50, at
the mild cost of a possible sex confound. `none` uses everyone. It's a genuine research
trade-off, exposed as a one-word switch.

**Age-match the groups — a word on `age_min`.** Age is the sneakiest **confound** here.
Dementia grows more common with age, so if the cohort reaches down into younger people (a
low `age_min`), the CDR-negative group skews young and the CDR-positive group skews old — and
a network can score well by quietly learning to estimate **age (or sex)**, which track the
label in this sample, instead of reading hippocampal shrinkage. Raising `age_min` (e.g. to
**70**) narrows the cohort to a band where CDR-negative and CDR-positive subjects *overlap* in age,
forcing the model to earn its predictions from the brain itself. The price is real: **far
fewer subjects and noticeably lower accuracy.** That drop is not a bug — it is honest.
Part of the higher accuracy at a low `age_min` was the age/sex confound leaking in, not
disease detection. On small clinical datasets a lower, well-controlled number usually
beats a higher, confounded one. (This is also why we watch **sensitivity/specificity**
and, once the groups are unbalanced, **balanced accuracy** — a model cheating via a
confound often gives itself away by leaning hard toward one class.)

**The design sweep — sampling the design space.** Rather than *tell* you about the
bias–variance trade-off and the effects of dropout, width, and depth, we let you *see*
them. We train a shared **baseline** plus three variants that each change exactly **one**
thing, so every comparison against the baseline isolates a single knob. The four sample
three different directions:

| Design | Conv blocks | Dropout | Direction sampled |
|---|---|---|---|
| 4a | 8→16→32 (3) | 0.6 / 0.2 | baseline (~8.2k params) |
| 4b | 8→16→32 (3) | 0.0 / 0.0 | no dropout → overfitting demo |
| 4c | 32→64→128 (3), 5×5 first | 0.6 / 0.2 | wider (~102k params, 12× the baseline) |
| 4d | 8→16 (2), 32-unit head | 0.4 / 0.2 | smaller (~1.9k params) |

This is a *sample* of the design space, not an exhaustive grid — one step in a few
directions from the baseline. The general lesson we expect, and that you can confirm: on
data this scarce, removing regularization (4b) overfits, going wider (4c) tends not to
help, and going shallower (4d) costs surprisingly little, while the lean baseline holds
its own. No amount of
architecture cleverness beats simply having more/cleaner data. (Running results live in
`internal/lab-notes.md`; to fill in the space *between* these sample points yourself, see §14.)

**Many slices for training, one fixed slice for validation/test.** Training benefits
from variety, so it uses several nearby planes (and optionally random shifts).
Validation and test use a **single fixed** plane so the score is deterministic and
comparable run-to-run — you're measuring the model, not the luck of which slice got
picked.

**Learning rate `1e-4`, and *not* training forever.** Early on we ran 500–1000 epochs
and watched validation accuracy spike early, then drift *back down* while training
accuracy kept climbing — the visual signature of overfitting past the sweet spot. The
reference tutorial reaches its result in only **30 epochs** with a *smaller* learning
rate (`1e-4`) and a tiny batch. Matching that makes learning gentler and steadier, so
the good solution isn't blown past. **More epochs is not the fix; gentler, better-
conditioned training is.**

**Augmentation as an on/off switch.** The random-shift augmentation is controlled by
`apply_random_shifts` and ships **off**, so you can run with it off, then on, and
measure whether it actually helps. Augmentation touches **training only** — validation
and test must remain a fixed, honest target. When we ran exactly that experiment (Run 1
vs Run 2 in `internal/lab-notes.md`), the verdict was sobering: because each training image
becomes `n_shifts` copies (8×), every epoch takes **~8× longer**, yet it **mainly helped
the leaner nets** and **did not appreciably raise** the headline balanced accuracy — and
piled on top of very high dropout in the tiniest net it *over-*regularized and made a
model collapse. Augmentation is still a good habit (it's standard practice and pays off
more with larger, real datasets), but here it is a modest lever, not a magic one.

---

## 11. A yardstick: the tutorial we're based on

This project follows the ARAMIS/Clinica
[Deep Learning for Medical Imaging](https://aramislab.paris.inria.fr/workshops/DL4MI/2022/notebooks/classification.html)
classification lab. Its hippocampus CNN reaches about **0.78 balanced accuracy** on
validation (≈ **0.81** when the left- and right-hippocampus predictions are combined by
voting) — in just **30 epochs**.

Why does theirs do better than we should expect from ours? Three instructive reasons:

1. **Heavier preprocessing.** Their scans are *non-linearly* warped to a common template
   and grey-matter-segmented, so the hippocampus is cleaner and lines up better across
   people. Ours are only *linearly* aligned and skull-stripped.
2. **3-D, not 2-D.** They use the whole hippocampus volume; we use flat slices.
3. **Gentler training.** Low learning rate, small batch, few epochs.

And one reason that caps *both* projects: the CDR label itself is imperfect (§1). Where
clinicians would disagree, no amount of preprocessing or network capacity recovers a fact that
was never in the data — so some of the missing 0.22 is not ours to win.

> **Key idea.** ≈ 0.78 balanced accuracy is a realistic target, and closing the gap
> would come from **better data preparation and going 3-D**, *not* from a bigger network
> or more epochs. This is a teaching pipeline: expect modest, noisy numbers, and treat
> the honest *method* — including its limits — as the real lesson.

---

## 12. What we learned, and what to try next

- **Data quality and quantity dominate model choice.** The fanciest network cannot beat
  the information ceiling of the data.
- **Know what your label actually is.** Ours is a clinical rating standing in for a disease we
  never observed (§1). That bounds both what the model can achieve and what you may claim it
  does — and it is the first question to ask of any biomedical dataset, including your own.
- **Dropout is the main dial.** Moderate dropout on a lean net has been our best
  recipe; too much (especially on a tiny net, or stacked on augmentation) backfires (§9).
- **Augmentation is a modest lever here.** Turning on random shifts costs ~8× the
  training time and, on this small dataset, mainly helps the leaner nets without moving
  the headline much — worth doing as good practice, not as a fix (§10).
- **Measure honestly.** Validation on a handful of subjects is noisy; balanced accuracy
  plus sensitivity/specificity tell you more than raw accuracy; and the best epoch is
  usually *not* the last.
- **Sensible next steps, in rough order of payoff:** pull in more subjects (`label`
  cohort) → turn on augmentation → add early stopping / keep the best-validation epoch →
  combine left+right into one per-subject vote → (to chase the tutorial) heavier
  preprocessing or a 3-D model. Only after all that is settled do you finally spend the
  **test set**, once, for an unbiased final number.
- For concrete, hands-on experiments to run, see §14.

---

## 13. Looking inside the model: Grad-CAM (step 6)

A validation number tells you *how often* the model is right, not *why*. For a medical
model that matters: is it reading the hippocampus, or has it latched onto a scanner
artifact at the edge of the patch? **Grad-CAM** (Gradient-weighted Class Activation
Mapping) is a simple way to look.

This is our first step into **interpretability** — moving past *whether* the model is right
to *what it is actually keying on*. Concretely, Grad-CAM answers: **which parts of this
patch most steer the decision toward CDR-positive (label 1)?**

**The idea.** Take the feature maps `Aₖ` from the **last convolution** — the grids fed
into global average pooling (§4.7). Each lit up wherever some learned pattern was found.
Grad-CAM asks: *how much does each map push the output toward the class I care about?* It
measures that with the gradient — average `∂(score)/∂Aₖ` over the grid to get one weight
`αₖ` per map — then forms a weighted sum and keeps only the positive part:

> `heatmap = ReLU( Σₖ αₖ · Aₖ )`, then upsample to the patch size.

The `ReLU` keeps regions that *raise* the score and drops those that lower it. step 6 shows
**both** directions side by side, in two distinct colormaps: the **CDR-positive** map
`ReLU(+Σₖ αₖ Aₖ)` (the `jet` colormap, "what pushes this patch toward CDR-positive") and the
**CDR-negative** map `ReLU(−Σₖ αₖ Aₖ)` (a distinct `cool` colormap, "what pushes it toward
CDR-negative").

> **Key idea — each map explains a target you choose, not the truth.** Grad-CAM never looks
> at the correct answer; *you* pick which output to explain. So the CDR-positive map for a
> *CDR-negative* patch shows **what would make it look more CDR-positive** — not "why it's
> CDR-negative" (the CDR-negative map answers that). Panel titles carry the true class,
> coloured **blue = CDR-negative, red = CDR-positive**, next to the model's `P(CDR+)`.

**A single-output wrinkle.** With one logit, "evidence for CDR-positive" and "evidence for
CDR-negative" are two sides of one coin: the CDR-negative map is `ReLU(−Σₖ αₖ Aₖ)`. That's
*almost* the negative of the CDR-positive map — but the `ReLU` keeps the opposite lobe, so the
two maps are **complementary**, not a pure sign flip. (Drop the `ReLU` and show a signed map
and they *are* exact negatives.) That is why the two panels usually light up *different*
regions: the CDR-positive (`jet`) panel shows where the CDR-positive-evidence sits, the
CDR-negative (`cool`) panel where the CDR-negative-evidence sits. If they looked identical,
something would be wrong — step 6 renders them as a pair precisely so you can read one against
the other.

**Read it with humility.** The map lives at the last conv layer's resolution — here a
coarse ~10×8 grid blown up to the whole patch — so it localizes *roughly*, not to the
pixel. Treat it as a sanity check ("is attention on the hippocampus?"), not a segmentation.

`step6-gradcam.py` loads a trained checkpoint (`model_4a.pt`, saved by step 4), runs a
sample of patches, and writes **two-panel** overlays (CDR− | CDR+, in two distinct
colormaps) to `outputs/gradcam/` plus a montage. It also redraws each subject's L/R heatmaps back onto the
full axial slice as a CDR−-vs-CDR+ pair (`outputs/gradcam_context/`), so you can see
where in the brain each kind of evidence sits.
**Method:** Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization"* (ICCV 2017); our implementation is inspired by (not copied
from) the original [ramprs/grad-cam](https://github.com/ramprs/grad-cam) and a
[PyTorch backward-hook discussion](https://discuss.pytorch.org/t/grad-cam-implementation-in-pytorch-backward-on-model/3554/7).

---

## 14. Things to play with, or Future Directions

The pipeline is meant to be poked at. Here are experiments you can run — most are a
one-line change to `config.yaml` or to the `Net` class in a `step4?-train-network.py`
file — roughly from easiest to most ambitious. Change **one thing at a time**, re-run,
and compare in `internal/lab-notes.md`.

*Extend the sweep — the four designs only sample a few points; fill in the curve yourself
by editing the `Net` class in a `step4?-train-network.py` file:*

- **Fill in the dropout dial.** The sweep shows dropout only at 0.6 / 0.2 (4a) and off
  (4b). Sweep the `nn.Dropout(...)` values from 0.0 up toward ~0.9 and watch training and
  validation pull apart when it's too low (overfit) and collapse when it's too high.
- **Fill in the width axis.** Designs 4a (8-16-32) and 4c (32-64-128) sample two widths, and
  deliberately far apart ones. Fill in the middle (e.g. 16-32-64) or go narrower still
  (4-8-16) by changing the conv-layer channel counts, and find where adding channels stops
  helping — or starts hurting.
- **Fill in the depth axis.** Designs 4d (2 blocks) and 4a (3 blocks) sample two depths.
  Try 1 block, or 4+, by adding/removing a `conv`+`bn` block (and its pool in `forward`)
  — GAP means you never have to re-tune the classifier sizes.

*Other directions:*

- **Is AI actually useful?** Compare with the original OASIS paper and see whether logistic
regression on brain size, etc. is a competitive or even better than AI.
- **Swap the activation.** Replace `ReLU` with `LeakyReLU` (lets a little negative signal
  through) or `GELU` (a smooth modern default) in the `Net` blocks, and see if training
  is steadier.
- **Resize the hippocampus crop.** Widen or tighten the box via `hippocampus.ap` and
  `lr_left` — more context vs. a tighter focus on the structure. (Re-run step 3.)
- **Trade age range against sample size.** A tighter `cohort.age_min/age_max` makes the
  CDR-negative and CDR-positive groups more comparable but gives you fewer subjects; loosening it
  does the opposite. Where's the sweet spot?
- **Measure how much of the answer is age, not CDR status.** Narrowing the age band (§10) is a
  blunt instrument: it removes the confound *and* shrinks the cohort at the same time, so a drop
  in accuracy has two possible explanations and you cannot tell them apart. Better tools exist,
  and none of them requires abandoning the data you have:
  - **Score the model against age inside the same age bin.** Report balanced accuracy separately
    within narrow strata — 60–65, 65–70, 70–75, … If the model is above chance *overall* but at
    chance *within every bin*, then all of its skill came from telling old people from young
    ones. No retraining needed; you already have per-subject predictions.
  - **Correlate the model's own output with age.** Take the trained network's logit on each
    validation subject and correlate it with that subject's age, *separately within each true
    class*. If the logit climbs with age among the CDR-negative subjects — people for whom age
    should be irrelevant to the answer — the network is reading age off the scan.
  - **Test it on age-matched pairs.** Pair each CDR-positive subject with a CDR-negative one of
    the same age (±1 year) and ask only how often the model ranks the pair correctly. That
    number is age-free by construction, and unlike a narrowed cohort it keeps every subject.
  - **Ask the network to predict age directly.** Train the same architecture to regress age from
    the patch (swap `BCEWithLogitsLoss` for `MSELoss` and the final logit for a number). If it
    predicts age well, then age *is* legible in a hippocampus patch — which tells you the CDR
    classifier had the option of using it. This is a small field in its own right, under the
    name **brain age**.
- **Use age properly instead of fighting it.** Age is genuinely informative about dementia risk;
  the problem is not that the model knows it, but that we cannot tell how much of the score it
  is silently providing. Three ways to make it explicit:
  - **Give age to the model on purpose.** Concatenate age to the GAP feature vector before the
    classifier head (§4.6) — a one-line change to `forward`. Then train three models: age only,
    image only, and age + image. The gap between "age + image" and "age only" is the honest
    answer to *what did the imaging buy us?*, and it is the number a clinical reader would ask
    for. (`jupyter/age_confounder.ipynb` already gives you the age-only leg.)
  - **Ask whether the model adds anything beyond age.** Fit two logistic regressions on the
    validation subjects: `CDR ~ age`, and `CDR ~ age + model_logit`. If the second is not
    meaningfully better than the first, the network contributed nothing that age did not already
    supply — regardless of how good its accuracy looked on its own.
  - **Train the features to be age-blind.** Add a second head that tries to predict age from the
    shared features, and train the backbone *against* it (a gradient-reversal or "adversarial
    de-biasing" setup) so the features it keeps are the ones age cannot explain. The most
    powerful option and much the most work — but it is how the problem is handled properly.
- **Balance sex *and* CDR status.** Switch `cohort.balance` from `label` to `strict` to
  remove a possible sex confound — at the cost of roughly half the data. Does honest
  validation accuracy go up or down?
- **Go 3-D.** Feed the whole hippocampus *volume* to a 3-D CNN instead of flat slices —
  the biggest jump toward the tutorial's numbers, and the biggest code change.
- **Try other architectures.** Add residual/skip connections, an attention block, or
  depthwise-separable convolutions and see whether they help on so little data (usually
  the lean nets still win — a good thing to confirm rather than assume).
- **Try transfer learning.** Instead of training a CNN from scratch, start from a
  network already pretrained on a large natural-image dataset — e.g. **AlexNet**
  (the network), pretrained on **ImageNet** (the 1.2-million-image, 1000-category
  dataset it was trained on) — and adapt it to our task. In practice: keep AlexNet's
  pretrained convolutional layers — the early ones already know generic features like
  edges and blobs, which transfer well across domains — and replace only its final
  classification layer(s) (the "head") with your own, ending in a single logit instead
  of AlexNet's original 1000-way ImageNet output (§4.6). If you freeze the pretrained
  backbone and train only the new head, that's called *feature extraction*; unfreezing
  some of the backbone's later layers too and training them at a small learning rate
  is called *fine-tuning*. Two wrinkles here: AlexNet expects 3-channel color images
  at a fixed size (224×224×3), so our grayscale ~58×68 patches would need resizing and
  channel handling (duplicating the single channel three times, or swapping the first
  layer) to fit; and since AlexNet's features come from natural photos rather than
  MRI, the domain gap means transfer helps less here than in typical vision tasks —
  still worth trying given how starved for data we are (§7).
- **Vote per subject.** Average (or majority-vote) the left- and right-hippocampus
  predictions into **one guess per person**, as the tutorial does — often a free bump.
- **Test-time augmentation.** At evaluation, average the prediction over several shifted
  crops of the *same* patch to smooth out noise.
- **Jitter the intensities.** Add small random brightness/contrast changes to training
  patches (augmentation beyond position).
- **Augment with rotations and flips** — now implemented, as
  `hippocampus.apply_random_reflections_rotations` (step3), layered on top of the
  positional random **shifts** (`apply_random_shifts`). Two details differ from the
  original sketch here: rotations are **exact multiples of 90°**, not a few-degree tilt —
  step3 crops a larger square first and rotates it losslessly, so nothing is ever resized
  or interpolated — and **both** horizontal *and* vertical mirror flips are included (each
  an independent, fixed 25% chance in the code), not just horizontal. None of these three
  transforms is how a brain is actually scanned — no real acquisition is upside-down or
  rotated 90° — but that's beside the point: the reason the free left/right reflection pair
  works (§10) is that the signal we care about, **atrophy**, looks the same reflected, and
  on a small, tightly-cropped, centred patch like ours, *orientation carries essentially no
  diagnostic information either way*. That makes all three transforms cheap, standard
  regularizers for a small dataset, not a compromise on realism — the one real difference
  from the free L/R pair is mechanism, not validity: L/R comes from cropping two genuinely
  different anatomical structures, while these three manufacture synthetic transformed
  copies of the *same* crop, which is exactly how geometric augmentation is meant to work.
  Whether it empirically helps *this* model either way is worth checking against
  `internal/lab-notes.md`. Validation/test stay **un-augmented** for a clean, deterministic
  evaluation, same as they're never shifted.
- **Train for balanced accuracy, not plain accuracy.** We *report* balanced accuracy (§6) but
  we don't *train* for it: `BCEWithLogitsLoss` counts every patch equally, so whichever class
  contributes more patches pulls the gradient harder — plain accuracy's bias, baked into
  training. Worth knowing first: **you cannot optimize balanced accuracy directly.** It is a
  step function — nudge a logit and either nothing changes or a prediction flips — so its
  gradient is zero almost everywhere and gradient descent has nothing to walk down. Every
  classifier is really trained on a smooth *surrogate* and judged on the metric you care about.
  Three ways to close the gap, easiest first:
  1. **Weight the classes in the loss** (one line):
     `nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]))`, where the counts come
     from the training manifest. Each *class* then contributes equally to the loss, which is
     the training-time analogue of balanced accuracy.
  2. **Tune the decision threshold instead** — no change to training at all. The 0.5 cut is a
     convention, not a law: sweep it on the *validation* set and keep whichever value maximizes
     balanced accuracy. Pairs naturally with the ROC-AUC bullet below, which needs the same
     continuous scores.
  3. **Use a differentiable stand-in for the metric** — compute sensitivity and specificity
     from the sigmoid probabilities directly (rather than hard 0/1 counts) and maximize their
     average, or try **focal loss**, which down-weights easy examples. More fiddly, and easy to
     make training unstable.

  **The catch — and the experiment worth running.** As configured this changes almost nothing:
  `cohort.balance: label` already hands us an equal-sized training split, so `pos_weight` ≈ 1.
  The
  payoff is that it lets you *stop balancing*. Switch `cohort.balance` to `none` and you keep
  every eligible subject instead of discarding subjects to force a 50/50 split — more data,
  which §7 and the lab notes both name as the real bottleneck — and use the weighted loss to
  stop the model collapsing onto the majority class. **Does more-but-imbalanced data beat
  less-but-balanced?** Note that validation stops being 50/50 too, so plain accuracy and
  balanced accuracy part ways and §6's metrics stop being interchangeable — read sensitivity
  and specificity, or you will fool yourself exactly as the "always CDR-negative" model does.
- **Change the objective entirely.** Predict the **CDR grade** itself (ordinal / multi-class)
  rather than pooling severities, or regress the **MMSE** score instead of a single yes/no.
- **Report AUC, not just sens/spec.** Add **ROC-AUC** (area under the ROC curve; optionally
  PR-AUC) to step5's printout and plots — a single **threshold-free** summary of how well the
  model *ranks* CDR-positive above CDR-negative, complementing the fixed-0.5-threshold
  sensitivity / specificity / balanced accuracy. The one prerequisite: AUC needs the model's
  **continuous scores** (probabilities), so step4 would log the per-epoch validation
  probabilities (or step5 would recompute them from a saved checkpoint) rather than only the
  thresholded metrics it logs today.
- **Tune the training loop.** A cosine learning-rate schedule with warmup, or a different
  batch size, can matter as much as the architecture.
- **Ensemble the four designs.** Average 4a–4d's predictions — ensembles usually beat any
  single member.
- **Add early stopping.** Keep the *best-validation* checkpoint instead of the last epoch
  (we noted validation peaks early, then drifts down).
- **Look inside with Grad-CAM** — *now implemented as `step6`* (see §13). It overlays a
  heatmap of where the network looks; a great sanity check that it's reading the
  hippocampus and not a border artifact.
- **Feed it more data.** Add more `discs`, widen the cohort, or switch `balance` to
  `none` — remembering that *effective* sample size is the number of subjects (§7). If you go
  the `none` route, pair it with a class-weighted loss (see "Train for balanced accuracy" above)
  so the extra data doesn't just teach the model to always answer with the majority class.

---

## 15. Check your understanding

1. Why do we need a **validation** set *and* a separate **test** set — what goes wrong
   if we tune the model against the test set?
2. Our data has 110 subjects with ~12 patches each. Is the effective amount of
   independent data closer to 1,320 or 110? Why?
3. A model scores **95 % accuracy** on a group that is 95 % CDR-negative. Should you be
   impressed? What single number would tell you more, and why?
4. Designs **4a** and **4b** are the same network except 4a uses dropout and 4b uses
   none. If 4b's *training* accuracy is higher but its *validation* accuracy is lower,
   what does that tell you?
5. Why do we crop the **hippocampus** specifically, instead of feeding the whole slice?
6. Why must all patches from one person go into the **same** train/validate/test split?
7. Our best design reaches about **0.78 balanced accuracy**. Can we report that as "detects
   Alzheimer's disease with 78 % accuracy"? If not, what *can* we honestly say — and what would
   the dataset need to contain for the stronger claim to be available?

*(Answers are throughout the notes: §2, §7, §6, §7 & §9, §10, §10, §1.)*

---

## 16. Glossary

- **Voxel** — a 3-D pixel; here 1 voxel = 1 mm of brain.
- **Slice / plane** — one flat 2-D cross-section of the 3-D brain.
- **Patch** — the small cropped rectangle around one hippocampus that we feed the model.
- **Label** — the correct answer (0 CDR-negative / 1 CDR-positive), derived from the CDR.
- **Ground truth** — the thing you actually want to know. In biomedical imaging it is often
  unavailable (invasive, expensive, post-mortem, or never recorded), which is the whole
  difficulty (§1).
- **Proxy label** — an available variable used *in place of* the ground truth. Ours is CDR,
  standing in for a disease the dataset never measured.
- **Label noise** — disagreement or error in the labels themselves. It caps how well any model
  can score, no matter how good the model is.
- **Inter-rater variability** — how much two qualified people disagree when rating the same
  case. A major source of label noise wherever a human assigns the label.
- **Parameter / weight** — an adjustable number inside the network; "learning" tunes these.
- **Convolution / filter / channel** — a sliding pattern-detector; each filter's output
  grid is a channel.
- **Pooling** — shrinking the grid (we use 2×2 max-pooling).
- **ReLU** — activation that zeroes negatives, keeps positives.
- **Logit** — the network's single raw output; >0 means "predict CDR-positive."
- **Loss** — a number measuring how wrong the predictions are (we minimize it).
- **Epoch** — one full pass over the training data.
- **Learning rate** — the size of each downhill step during training.
- **Overfitting / underfitting** — memorizing quirks vs being too weak to learn at all.
- **Dropout / weight decay / augmentation** — techniques that fight overfitting (dropout
  is the main dial here — §9).
- **Sensitivity / specificity / balanced accuracy** — honest performance metrics (§6).
- **Data leakage** — accidentally letting test information influence training (e.g. the
  same subject in two splits), which inflates scores dishonestly.
- **Cross-validation (k-fold)** — rotating the validation set over *k* subject-level folds
  and averaging, to get a mean ± error bar instead of one noisy split (§8).
- **Grad-CAM** — a heatmap of where the network looked to raise a chosen class score;
  our sanity check that it reads the hippocampus (§13).

---

## 17. Useful references

These notes mention "the reference tutorial" several times (§10, §11, §12, §14); here it
is, alongside other background reading, grouped by topic.

**The tutorial this project benchmarks against**

- [Deep Learning for Medical Imaging — hippocampus classification lab](https://aramislab.paris.inria.fr/workshops/DL4MI/2022/notebooks/classification.html)
  (ARAMIS/Clinica) — the notebook this project's design is compared against throughout,
  most directly in §11.

**Course notes — background on CNNs**

- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
  (Stanford) — particularly [Module 2](https://cs231n.github.io/convolutional-networks/)
  on ConvNets, which covers §4's convolution/pooling material in more depth.
- [EECS 498-007 / 598-005: Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/)
  (Justin Johnson, University of Michigan), with
  [community lecture notes](https://github.com/Andrew-Ng-s-number-one-fan/EECS498-Deep-Learning-for-Computer-Vision).

**Classic papers**

- Krizhevsky, Sutskever, Hinton,
  ["ImageNet Classification with Deep Convolutional Neural Networks"](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
  (AlexNet, NeurIPS 2012) — the paper that made CNNs the default tool for image tasks.
- LeCun, Bengio, Hinton, ["Deep Learning"](https://www.nature.com/articles/nature14539)
  (Nature, 2015) — a short, high-level review of the field.
- Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov,
  ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](https://jmlr.org/papers/v15/srivastava14a.html)
  (JMLR, 2014) — the technique §9 calls "the regularizer that matters most here."
- Ioffe & Szegedy,
  ["Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"](https://arxiv.org/abs/1502.03167)
  (2015) — the BatchNorm layer used throughout (§4.5).

**Medical imaging**

- Yamashita, Nishio, Do, Togashi,
  ["Convolutional Neural Networks: An Overview and Application in Radiology"](https://link.springer.com/article/10.1007/s13244-018-0639-9)
  (Insights into Imaging, 2018) — CNNs specifically in the radiology context this
  project sits in.
- Marcus et al.,
  ["Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young, Middle Aged, Nondemented, and Demented Older Adults"](https://pubmed.ncbi.nlm.nih.gov/17714011/)
  (Journal of Cognitive Neuroscience, 2007) — the paper describing the OASIS-1 dataset
  this project uses.

**What our label is — and what it isn't (§1)**

- Morris,
  ["The Clinical Dementia Rating (CDR): Current Version and Scoring Rules"](https://doi.org/10.1212/WNL.43.11.2412-a)
  (Neurology, 1993) — the CDR itself: what the six categories are and how a clinician
  arrives at a global score. This is the variable we actually train on.
- Sperling et al.,
  ["Toward Defining the Preclinical Stages of Alzheimer's Disease"](https://doi.org/10.1016/j.jalz.2011.03.003)
  (Alzheimer's & Dementia, 2011) — the NIA-AA recommendations describing the stage where
  Alzheimer's pathology is present but cognition still tests normal. Those people are CDR 0,
  which is precisely why a CDR label is not a disease label.
- (See also Marcus et al. 2007 above, which describes how OASIS-1's clinical data was
  collected.)

**Interpretability**

- Selvaraju et al.,
  ["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"](https://arxiv.org/abs/1610.02391)
  (ICCV 2017) — the method behind step 6's Grad-CAM heatmaps (§13).

---

Back to the practical guide: **[readme.md](readme.md)**.
