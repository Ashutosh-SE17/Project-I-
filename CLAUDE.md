# Project: Nepali Election Sentiment Analysis

Claude Code reads this file automatically at the start of every session.
Keep it short and current — this is what stops the agent re-deriving the
project from scratch each time.

## What this is

A Flask app that estimates public sentiment toward Nepali political
candidates. It pulls YouTube comments and news headlines, scores each item
with a supervised classifier, and aggregates to a "win probability" shown
on a dashboard with a Folium district map.

Academic project. Currently mid-defense stage.

## Architecture

```
comment / headline
  → preprocessing.clean()      script-aware, keeps Devanagari
  → ML classifier              polarity = P(pos) − P(neg) ∈ [−1,+1]
  → entities.attribution()     who is this about? ∈ [−1,+1]
  → score = polarity × attribution + scandal_term
  → mean over items
  → blend: α·news + (1−α)·comments      α = 0.35, shrunk if few headlines
  → sigmoid(4 × combined) → win probability
```

## Files

| File | Role |
|---|---|
| `manage.py` | CLI entry point for everything. Start here. |
| `preprocessing.py` | Text cleaning. Handles Devanagari + romanized Nepali. |
| `entities.py` | Candidate registry, attribution, scandal/symbol logic. |
| `train_sentiment.py` | Trains and evaluates the classifier. |
| `sentiment_model.py` | `ElectionAnalyzer` — orchestrates fetch → score → aggregate. |
| `label_tool.py` | Terminal annotation tool. |
| `collect_pool.py` | Builds `labelling_pool.csv` from all data sources. |
| `news_source.py` | RSS / Google News headline collection. |
| `app.py` | Flask server. Do not modify without asking. |
| `generate_nepal_map.py` | Regenerates district maps. Do not modify without asking. |

## Commands

```
python manage.py doctor     check environment
python manage.py pool       build labelling pool (--news for headlines)
python manage.py label      annotate
python manage.py audit      annotation consistency check
python manage.py train      train classifier
python manage.py serve      run Flask app
```

## Key design decisions (do not "fix" these)

- **LogisticRegression, not Naive Bayes.** NB probabilities are
  miscalibrated and would saturate the downstream sigmoid.
- **`char_wb` n-grams alongside word n-grams.** This is what makes
  Devanagari and romanized Nepali work through one vectoriser.
- **Mean, not sum, before the sigmoid.** The old code summed, so the
  output scaled with sample size — 1078 comments gave 0.00%.
- **Block weighting for news** (a fixed 35% share), not per-item weights.
  Per-item weighting makes the answer depend on news volume.
- **One pooled model with a `__ch_comment__` / `__ch_news__` marker**, not
  separate models per channel. Data efficiency — see LABELLING_GUIDE.md.
- **Negation direction is script-dependent.** Nepali is SOV: `chaina`
  negates the word BEFORE it. English negates forward.

## Conventions

- Python 3.13, virtualenv at `.venv`
- Secrets in `.env` only. Never hardcode keys. Never commit `.env`.
- Tunable constants live at the top of their module with a comment
  explaining the choice — they get cited in the report as hyperparameters.
- Don't modify `app.py`, `generate_nepal_map.py`, or `static/` without
  asking first.

## Current state

- Labelling pool built: ~1134 comments (722 latin, 346 deva, 66 mixed)
- Labelling: in progress
- Model: not yet trained
- Frontend: does not yet display `confidence_interval`, `channel_scores`,
  `script_breakdown`, `model_metrics`, `top_headlines`

## Known limitations (deliberate, documented in the report)

- YouTube commenters are not a representative sample of the electorate.
  The output is an online sentiment index, not a poll.
- `entities.py` weights are hand-curated.
- `NEWS_SHARE = 0.35` is a judgement call; needs a sensitivity analysis.
- News channel should stay off (`include_news=False`) until headlines are
  labelled and per-channel accuracy is validated.
