# Migrating from lexicon scoring to supervised learning

The pipeline's shape is unchanged. Only the box that turns text into a
number has been replaced.

```
              OLD                                    NEW
  ─────────────────────────────        ─────────────────────────────────
  comment                              comment
    ↓                                    ↓
  lexicon lookup (~20 words)           ML classifier (LogisticRegression)
  + fuzzy match (fuzzywuzzy)             → P(pos), P(neu), P(neg)
  + symbol / scandal / rival rules       → polarity = P(pos) − P(neg)
    ↓                                    ↓                    ∈ [−1, +1]
  unbounded score                      entity layer (KEPT)
    ↓                                    → attribution        ∈ [−1, +1]
  SUM over comments                      → scandal term
    ↓                                    ↓
  sigmoid(sum / 25)                    score = polarity × attribution
    ↓                                          + scandal      ∈ [−1, +1]
  win probability                        ↓
                                       MEAN over comments     ∈ [−1, +1]
                                         ↓
                                       sigmoid(4 × mean)
                                         ↓
                                       win probability + bootstrap CI
```

## Files

| File | Status | Purpose |
|---|---|---|
| `preprocessing.py` | **new** | Script-aware cleaning. Keeps Devanagari. |
| `entities.py` | **new** | Your old rules, repurposed as attribution. |
| `train_sentiment.py` | **new** | Trains + evaluates the classifier. |
| `label_tool.py` | **new** | Terminal annotation tool. |
| `sentiment_model.py` | **replaced** | Same public API, new internals. |
| `requirements.txt` | **replaced** | Now actually installable. |
| `app.py`, `script.js`, `style.css` | **unchanged** | Return shape is preserved. |

## Setup

```bash
pip install -r requirements.txt
export YOUTUBE_API_KEY="your_new_rotated_key"    # never hardcode it again
```

## The three-step workflow

### 1. Label

```bash
python label_tool.py --data election_data.csv --out labelled.csv --n 400
```

Target **1500–2000 labelled comments**. At ~4 seconds per comment that is
roughly two focused hours. It samples Devanagari and Latin evenly, which
matters: your corpus is 35% Devanagari and that is precisely where the old
engine failed.

**Annotation rule:** label the sentiment *of the text*, not who it helps.
`"KP Oli chor ho"` is **negative** even when analysing Balen. Attribution
is `entities.py`'s job, not the classifier's. Mixing the two is the single
most common way to poison a dataset like this.

### 2. Train

```bash
python train_sentiment.py --data labelled.csv --out models/sentiment_clf.pkl
```

Reports CV accuracy against the majority baseline, macro-F1, per-class
recall, a confusion matrix, per-script accuracy, and a calibration curve.
Every number goes in your report.

### 3. Run

```bash
python app.py
```

## Three deliberate deviations from Mohan's method

**Logistic Regression, not Naive Bayes.** Your sigmoid consumes a
continuous score. NB's independence assumption crushes `predict_proba`
toward 0 and 1 — feeding that to a sigmoid re-creates the saturation you
are trying to fix. On the smoke test, LR put only 3.4% of comments at
|polarity| > 0.95, and the reliability curve tracked actual accuracy
closely. That is what makes the score safe to aggregate.

**Character n-grams alongside word n-grams.** `char_wb` (2–5) runs on
Devanagari and Latin through one vectoriser and absorbs romanisation
variance (`ramro` / `raamro` / `ramrro`) as shared substrings — which is
what `fuzzywuzzy` was doing at enormous cost, done properly and ~1000×
faster. This is the fix for the 35% of your corpus the old engine ignored.

**Three classes, not two.** Mohan's dataset is binary. Your doughnut chart
needs Positive/Neutral/Negative, and neutral probability mass usefully
shrinks a comment's influence on the mean.

## Why `mean` instead of `sum`

Measured on your real 1,078-comment corpus, identical scores:

| n comments | `sigmoid(sum/25)` | `sigmoid(4 × mean)` |
|---|---|---|
| 50 | 46.53% | 43.09% |
| 100 | 42.62% | 42.62% |
| 250 | 41.94% | 46.75% |
| 500 | 20.35% | 43.22% |
| 1078 | **4.98%** | **43.20%** |

The old formula's output is a function of sample size, not sentiment. Add
more comments from the same distribution and the "probability" collapses.
The new one is bounded and stable, and reports a bootstrap 95% CI so
sample size shows up as *uncertainty* rather than as a bogus point estimate.

`SIGMOID_GAIN = 4.0` in `sentiment_model.py` sets the spread:
mean +0.25 → 73%, +0.50 → 88%, +1.00 → 98%. Tune it, and say so in your
report — it is a documented hyperparameter, not a magic number.

## Also fixed

- **Evidence sampling is now representative.** The old code showed the 8
  most positive comments when the prediction was favourable and the 8 most
  negative when it was not, making the table circular. It now draws a
  stratified random sample. This was the top item on your own roadmap.
- **API key from `YOUTUBE_API_KEY`.** Rotate the old one; it is public.
- **Errors surface.** The old bare `except: pass` turned quota exhaustion
  into "No data retrieved" with no diagnosis.
- **Caching.** `cache.json` is now actually used, with a 6-hour TTL. At
  ~105 quota units per analysis against a 10,000/day cap, this matters.
- **Negation direction.** Nepali is SOV — `chaina` negates the word
  *before* it. A naive port marks the word after, silently inverting
  negation on every Nepali sentence.

## One caveat, stated plainly

`entities.py` is still hand-curated, and so is the aggregation gain. The
supervised swap removes subjectivity from *polarity* but not from
*attribution*. Say that in your limitations slide before an examiner says
it for you — and note that this is exactly why the hybrid is defensible:
the parts that can be learned are learned, and the parts that require
domain knowledge are declared, documented, and tunable.
