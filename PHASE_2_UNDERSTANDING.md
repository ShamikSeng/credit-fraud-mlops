# Phase 2 — Understanding Walkthrough

This is a study document, not a changelog (that's `PROJECT_LOG.md`). Read this to be
able to explain Phase 2 out loud, unaided, at the "why does this exist" level — not
the line-by-line level.

---

## 1. The end-to-end flow

Forget the file layout for a second. Here's what actually happens to the data, in order.

### Step 1 — Raw data lands
`creditcard.csv` has 284,807 rows. Each row is one credit card transaction: `Time`
(seconds since the first transaction in the dataset), 28 columns `V1`...`V28` (these
are the *output of a PCA transform* someone already ran on the original raw features —
we never see what those original features were, because that would de-anonymize real
cardholders), `Amount` (transaction amount in whatever currency), and `Class` (0 = legit,
1 = fraud). Only 492 rows are fraud — about 0.17%.

### Step 2 — The split happens *first*, before anything else touches the data
`stratified_split()` divides the 284,807 rows into an 80% training set and a 20% test
set — but it does this in a *stratified* way, meaning it forces both the 80% and the
20% to keep the same ~0.17% fraud rate as the full dataset. Why this matters: if you
split randomly without stratifying, on a rare-event dataset you can easily end up with,
say, 0.24% fraud in train and 0.05% fraud in test just by chance — and now your test
score doesn't mean what you think it means.

This split happens *once*. The 20% test set is set aside and is not touched again until
the very last step (Step 6). Everything in between — imputing, scaling, hyperparameter
tuning — only ever sees the 80% training set.

### Step 3 — The preprocessing pipeline
Two transformers run on the data before it reaches the model:
- **MeanImputer** — fills missing numeric values with the column mean. (This dataset
  actually has zero missing values, so in practice this transformer does nothing right
  now — it's there defensively, in case a batch-upload CSV someone feeds the API later
  has gaps.)
- **RobustScalerTransformer** — rescales `Amount` and `Time` using the median and IQR
  (interquartile range) instead of the mean/std or min/max. Why only these two columns:
  `V1`...`V28` already went through PCA, and PCA output is already roughly centered and
  scaled — scaling it again would be redundant. `Amount` and `Time`, though, are raw and
  `Amount` in particular is heavily skewed (most transactions are small, a few are huge) —
  a plain min-max or mean/std scaler would get badly distorted by those few huge values.
  Median/IQR-based scaling is much more robust to that kind of outlier.

Both of these live inside a single sklearn `Pipeline` object, built by one function
(`build_pipeline()`) that both training and (potentially) inference call. That's the
"single source of truth" — there used to be two hand-copied versions of this
preprocessing in the old repo, which is exactly the kind of thing that quietly drifts
out of sync over time.

### Step 4 — The Hyperopt + cross-validation training loop
This is the part that's easy to get confused about, so slow down here. There are
**two different splitting mechanisms in this system, doing two different jobs**:

- The 80/20 stratified split from Step 2 is the **outer, final holdout**. It exists to
  answer one question at the very end: "how good is my *final* model on data it has
  never seen in any form?"
- Inside the 80% training set, Hyperopt runs a search loop. On *every single trial*
  (every candidate set of hyperparameters it tries), the code doesn't just fit once and
  check a score — it runs 5-fold **StratifiedKFold** cross-validation *within the
  training set only*. That means the 80% training data gets further sliced into 5
  folds; the model trains on 4 folds and validates on the 5th, five times (rotating
  which fold is held out), and the PR-AUC scores from all 5 folds get averaged. That
  averaged number is the score Hyperopt uses to judge whether this trial's
  hyperparameters were good.
- Hyperopt (using the TPE algorithm) does this for 20 trials, each time picking a new
  combination of hyperparameters (max_depth, learning_rate, n_estimators, etc.),
  scoring it via that 5-fold average, and using past results to make smarter guesses
  about where to search next. Whichever trial had the best (highest) average PR-AUC
  wins.

So: the test set never appears anywhere in this loop. The CV folds are a *second,
separate, nested* form of splitting used purely to make the hyperparameter search
robust to noise — a single train/validation split could get lucky or unlucky by chance,
especially with so few fraud examples; averaging over 5 folds smooths that out.

### Step 5 — Final refit
Once Hyperopt has picked the best hyperparameters, the code builds one final pipeline
with those hyperparameters and fits it on the *entire* 80% training set (no more
splitting it into folds — we already know these are good hyperparameters, now we just
want to use all the training data we have).

### Step 6 — The one-time evaluation
*Now*, for the first and only time, the fitted model is run against the 20% test set
from Step 2. This produces the numbers that actually get reported: PR-AUC (the primary
one), plus ROC-AUC, precision, recall, and F1 for context. In this project's actual run:
PR-AUC = 0.8693, ROC-AUC = 0.9809, precision = 0.8632, recall = 0.8367, F1 = 0.8497 —
these are the honest, un-gamed numbers because the test set had never influenced any
decision up to this point.

### Step 7 — Logging
All of this — every trial's hyperparameters and CV score, plus the final model's
metrics and the model artifact itself — gets logged to MLflow. Locally, that means a
folder called `mlruns/` (rather than the original baseline's remote EC2 tracking
server, which is currently unreachable).

### Step 8 — Serving
When a prediction request comes in (`main.py` → `predict.py`), the serving code asks
MLflow: "of all the logged runs in this experiment, which one has the best `pr_auc`?" —
loads that run's saved model, and uses it to predict. The Pydantic schema
(`TransactionFeatures`) is the gatekeeper at the API boundary: it demands all 30 fields
(`Time`, `V1`...`V28`, `Amount`) as floats, so a malformed request (missing field, wrong
type) gets rejected with an HTTP 422 before it ever reaches the model.

### Step 9 — The test gate
`tests/test_prediction.py` doesn't touch MLflow or the CSV at all — it's a pure,
offline check that (a) the Pydantic schema behaves correctly (accepts good input,
rejects bad input) and (b) the stratified split does what it claims (preserves class
ratio, is deterministic given the same random seed). It's a regression gate: if someone
later breaks the schema or the split logic, this catches it immediately without needing
a live model or tracking server.

---

## 2. The four decisions, explained like you're teaching them

### Decision 1 — PR-AUC over ROC-AUC or F1 as the primary metric

**The plain-language version:** with only 0.17% of transactions being fraud, the
dataset is almost entirely made of "easy" negatives (legit transactions that are
obviously legit). ROC-AUC's calculation involves the *true negative rate*, and with
so many negatives, a model can get a very high true-negative rate just by being
lazy and mostly predicting "legit" — that inflates ROC-AUC even for a mediocre model,
because there are so many negatives to get right almost by default. PR-AUC, in
contrast, only cares about precision and recall — both of which are entirely about how
the model handles the *positive* (fraud) class. It never gets a "free pass" from the
huge pile of negatives. So on an imbalanced problem, PR-AUC tells you much more
honestly how good the model is at the thing you actually care about: catching fraud.

F1 is rejected for a different reason: F1 needs you to pick a decision threshold (the
cutoff probability above which you call something "fraud") *before* you can compute
it. PR-AUC is threshold-free — it looks at the model's performance across every
possible threshold and summarizes that into one number. That's more useful during
model *selection/tuning*, because you don't want to be accidentally tuning your model
to look good at one specific threshold that you picked somewhat arbitrarily. (You'd
still pick an actual threshold later, once you're deploying — but that's a business
decision about cost of false positives vs false negatives, not a training-time metric
choice.)

**Likely follow-up:** *"If PR-AUC is threshold-free, how do you actually decide what
threshold to use in production?"*
**How to answer:** PR-AUC is used to pick *which model* is best during training/tuning.
Once you have that model, you'd pick an operating threshold separately, based on the
real-world cost tradeoff — e.g., a false negative (missed fraud) might cost far more
than a false positive (an annoyed customer who gets an extra verification step) — so
you'd look at the precision-recall curve and pick the threshold that hits your
acceptable recall/precision tradeoff for the business, possibly informed by a cost
matrix. This project currently uses a naive 0.5 cutoff for the reported precision/
recall/F1 numbers, which is a simplification — a real deployment would tune this
explicitly.

### Decision 2 — `scale_pos_weight` over resampling (SMOTE / undersampling)

**The plain-language version:** the imbalance problem is: the model sees so few fraud
examples (394 out of ~227,845 in the training set — a ratio of about 1:578) that it
can get a very low loss just by predicting "legit" almost always. There are two broad
families of fixes:
1. *Change the data* — either duplicate/synthesize more fraud examples (oversampling,
   SMOTE) or throw away some legit examples (undersampling) so the classes look more
   balanced to the model.
2. *Change the loss function* — tell the model "when you get a fraud example wrong, it
   costs you ~578x more than getting a legit example wrong." XGBoost supports this
   natively via `scale_pos_weight`.

This project uses option 2. The core reason: resampling methods change the data
distribution the model is trained on, and if you're not extremely careful, that's a
leakage risk — for instance, if you oversample *before* doing your CV split, synthetic
points derived from a fraud case can end up in both the training fold and the
validation fold, so the model gets partial credit for "remembering" a case it
technically shouldn't have seen fully. `scale_pos_weight`, by contrast, doesn't touch
the data at all — it only changes how errors are weighted during training — so there's
no possible leakage vector from it, and it composes cleanly with the StratifiedKFold
CV loop without any extra precautions.

**Likely follow-up:** *"Doesn't `scale_pos_weight` risk pushing the model to
over-predict fraud, hurting precision?"*
**How to answer:** Yes, that's the real tradeoff — weighting the loss this heavily
pushes the model toward higher recall (catching more actual fraud) at some cost to
precision (more false alarms), which is usually the right tradeoff for fraud detection
specifically, since missing real fraud is typically far more costly than a false
alarm that gets manually reviewed. The actual results back this up:
precision = 0.8632 and recall = 0.8367 — both are still high and fairly balanced, so in
this case the weighting didn't overcorrect. If it had overcorrected, you'd see recall
much higher than precision (lots of false alarms), and you could dial
`scale_pos_weight` down from the raw class-ratio value to rebalance.

### Decision 3 — Stratified holdout + StratifiedKFold, and specifically *not* a temporal split

**The plain-language version:** in a *real*, deployed fraud system, a temporal split
(train on older transactions, test on newer ones) is usually the gold standard,
because it mimics the actual deployment reality: your model will always be scoring
transactions that happened *after* the data it was trained on. A random/stratified
split can accidentally leak future information into training in subtle ways (e.g., if
fraud patterns shift over time and your random split mixes future-pattern examples into
your training set, your validation score will look better than what you'd actually get
in production, where you never get to train on the future).

So why does this project deliberately reject that gold-standard approach? Because the
gold standard only pays off when the thing it's protecting against — concept drift over
time — actually exists in a way you can detect and learn from. This dataset:
- only spans about two days total,
- has already had its real features scrubbed through PCA, so there's no interpretable
  "time of day" or "day of week" signal left to even exploit or protect against, and
- only has 492 total fraud examples — a temporal cut would carve those into two even
  smaller, noisier buckets.

Given that, a temporal split here wouldn't buy any real protection against drift (there
isn't enough time range or interpretable signal for drift to show up) — it would mostly
just add random noise to the evaluation by shrinking and skewing how many fraud
examples land in train vs. test. Stratified random splitting, by preserving the class
ratio, gives a more stable and honest read of model quality *for this specific
dataset*. The explicit caveat to hold onto: if this were extended to a longer,
real-world, continuously-arriving transaction stream, temporal splitting would become
the *correct* upgrade, precisely because now there'd be enough time range for drift to
be real and worth guarding against.

**Likely follow-up:** *"How would you actually detect if concept drift became a
problem, if you're not using a temporal split to test for it?"*
**How to answer:** Temporal splitting is one way to *catch* drift during evaluation,
but it's not the only way to *monitor* for it in production — that's actually a
separate concern, handled by drift detection tooling (this repo already has an
EvidentlyAI-based Streamlit app for this, and Phase 3 is scoped to potentially wire
drift monitoring into Prometheus/Grafana). The distinction is: temporal splitting
during *training* evaluation answers "would this modeling approach have generalized
forward in time, historically" — production drift monitoring answers "is the live
data distribution today still similar to what the model was trained on." For this
2-day dataset, the first question isn't answerable meaningfully; the second question
is a live-system concern that belongs in monitoring, not in the train/test split
strategy.

### Decision 4 — Where leakage is prevented, and what the bug would look like otherwise

**The plain-language version:** "leakage" here specifically means: information from
the data you're supposed to be *evaluating on* accidentally influencing something that
was learned *before* evaluation. The two transformers in this pipeline — the mean
imputer and the robust scaler — both have a `fit()` step where they *learn* something
from the data (the imputer learns the column means; the scaler learns the median and
IQR of `Amount`/`Time`), and a separate `transform()` step where they *apply* what they
learned.

The leakage-safe design here is: both transformers live *inside* a single sklearn
`Pipeline` object, alongside the classifier. That means whenever the pipeline is fit —
whether that's on one of the 5 CV folds' training portion, or on the full 80% training
set for the final refit — `fit()` on the imputer/scaler is called *only* on whatever
data is being used as "training" at that moment. The held-out portion (the CV
validation fold, or later the true test set) only ever goes through `transform()`,
never `fit()`. So the median/IQR/mean values baked into the scaler and imputer are
always computed exclusively from data the model hasn't "seen" yet in that round.

**What the bug would look like if this were done wrong:** the classic version of this
mistake is computing the scaler's median/IQR (or a mean, or a min/max) on the *entire*
dataset — train and test combined — *before* splitting. If you did that, the test set's
own values would have quietly influenced the scaling parameters used to transform the
training set (and vice versa). The practical symptom: your reported metrics (PR-AUC,
etc.) would look artificially *better* than what you'd actually get on truly unseen
data in production, because the "test" set was never fully held out — it had already
leaked a little bit of itself into the preprocessing. This is a very easy bug to
introduce by accident (e.g., writing `df['Amount'] = scaler.fit_transform(df[['Amount']])`
on the whole dataframe *before* calling `train_test_split`), and it wouldn't throw any
error — it would just silently make your evaluation numbers optimistic and
untrustworthy.

**Likely follow-up:** *"If I saw suspiciously high performance on this dataset, how
would you check whether it was leakage versus the model genuinely being that good?"*
**How to answer:** A few concrete checks: (1) re-verify that every `fit()` call on a
preprocessing step happens strictly inside the CV loop / on the training split only —
grep for any `.fit(` or `.fit_transform(` call that touches a dataframe before the
train/test split exists. (2) Check whether performance is suspiciously *uniform* across
CV folds — real signal tends to have some fold-to-fold variance; near-identical scores
across all folds can be a leakage tell. (3) Try recomputing the final holdout metric
with a *freshly refit* scaler/imputer fit only on train, applied to test, and confirm
it matches what the pipeline already produces (a sanity re-derivation). In this
project's case specifically, both transformers are wrapped inside one `Pipeline`
object rather than being applied as standalone dataframe operations, which is
structurally what prevents the leak — the pipeline's `fit`/`transform` contract
enforces the ordering, rather than relying on the programmer to remember to do it
correctly by hand each time.

---

## Quick self-check before Phase 3

You should be able to say all four of these out loud, unaided, without looking at this
file:
1. Why PR-AUC, not ROC-AUC/F1 — in one sentence: *because ROC-AUC gets an easy ride from
   the huge negative class, and F1 needs a threshold decided up front.*
2. Why `scale_pos_weight`, not resampling — in one sentence: *it changes the loss, not
   the data, so there's no way for it to leak across CV folds.*
3. Why stratified, not temporal, for this dataset — in one sentence: *there's not enough
   time range or interpretable signal in 2 days of PCA'd data for a temporal split to
   protect against anything real; it would just add noise.*
4. Where leakage is prevented — in one sentence: *the scaler/imputer live inside the
   sklearn Pipeline, so they only ever `fit()` on whatever's currently the training
   portion, never on the held-out fold or test set.*

If any of these feel shaky, re-read that section above before Phase 3.
