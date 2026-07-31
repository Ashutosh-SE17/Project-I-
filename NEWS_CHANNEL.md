# The news channel

## What changed

```
                    OLD                              NEW
        ────────────────────────        ─────────────────────────────────
        YouTube comments (≤250)         YouTube comments (up to 1000)
              ↓                                    ↓
          mean score                         comment_mean
              ↓                                    ↓
           sigmoid                    ┌────── blend ──────┐
                                      │                   │
                            Google News RSS +      news_mean
                            7 Nepali outlet feeds        │
                                      └───────┬───────────┘
                                              ↓
                             combined = α·news + (1−α)·comments
                                              ↓
                                          sigmoid
```

## First: your project never pulled 1000 comments

`fetch_1000_data()` requested 5 videos × 50 comments, single page. **Hard
ceiling of 250.** The name was aspirational.

Now fixed with round-robin pagination across 8 videos until 1000 comments
or the videos run dry. Round-robin rather than draining one video at a
time, so the sample isn't dominated by whichever video happens to be most
commented.

**Quota cost:** the `search` call is 100 units; each 100-comment page is 1
unit. So 1000 comments ≈ **110 units** against a 10,000/day cap — roughly
90 analyses per day. Comment volume was never the expensive part.

## Sources

**Google News RSS** — `news.google.com/rss/search`, queried across two
locales (`NP:ne` and `NP:en`) and three time windows (`when:1d`,
`when:7d`, `when:30d`). The feed caps at ~100 items with no pagination,
so subdividing by time window and merging is the documented way to go
deeper.

**Direct outlet feeds** — Onlinekhabar, Ratopati, Setopati, Kathmandu
Post, Nagarik, Baahrakhari, Himalkhabar. Fetched whole, then filtered by
candidate mention. Cheaper than one query per candidate per outlet.

## Why block weighting, not per-item weighting

The intuitive approach — "one headline is worth 5 comments" — is unstable.
With 1000 comments, 10 headlines at 5× gives news a 4.8% share; 100
headlines gives 33%. The same underlying sentiment produces a different
answer depending on how busy the news cycle was.

So news gets a fixed **share**:

```
combined = α · news_mean + (1 − α) · comment_mean
α        = NEWS_SHARE × min(1, n_news / MIN_NEWS_FULL)
```

with `NEWS_SHARE = 0.35` and `MIN_NEWS_FULL = 15`.

Measured behaviour (1000 comments at mean −0.165, headlines at +0.60):

| headlines | share applied | combined | win % |
|---|---|---|---|
| 0 | 0.00 | −0.165 | 34.06 |
| 3 | 0.07 | −0.112 | 39.02 |
| 8 | 0.19 | −0.022 | 47.76 |
| 15 | 0.35 | +0.103 | **60.12** |
| 40 | 0.35 | +0.103 | **60.12** |

Rows 4 and 5 are identical — 15 headlines and 40 headlines produce the
same answer. That stability is the point. The shrinkage term below 15 stops
three syndicated stories from claiming a third of a national estimate.

Within the news block, each item carries
`weight = source_credibility × recency_decay`, with a 7-day half-life. A
14-day-old item from an unknown outlet lands at 0.175; a fresh Onlinekhabar
item at 1.00.

## Two filters that matter more than they sound

**Syndication dedup.** Google News carries the same story under many
outlets with cosmetically different titles. Keying on a sorted bag of
significant tokens rather than the raw string, three copies of one story
collapse to one — verified on reordered Nepali titles. Without this, one
story silently counts ten times.

**Relevance.** Google News keyword search is loose. A headline only counts
if it actually names the candidate via the alias list in `entities.py`.

## The honest limitations — put these on a slide

**1. It changes what you are measuring.** Comments are public sentiment.
Headlines are editorial framing. The blended number is neither — it's a
composite. Name it accordingly and report `channel_scores` separately in
the dashboard so both are visible. A reader should be able to see when the
press and the public disagree; that divergence is arguably the most
interesting thing this tool can show.

**2. Domain shift is real and will hurt.** A classifier trained on YouTube
comments will misread headlines. Different register entirely —
`"ओली अस्पताल भर्ना"` is negative in tone but not an attack;
`"प्रधानमन्त्रीले उद्घाटन गरे"` is neutral reporting a comment-trained
model may score positive. **Label 200–300 headlines as their own slice**
and report per-channel accuracy separately. Without that, the news score
is unvalidated.

**3. News sentiment ≠ sentiment toward the candidate.** Most political
reporting is event description, not evaluation. Expect the neutral class
to dominate headlines far more than comments, which will drag `news_mean`
toward zero and quietly reduce the news channel's real influence below
35%. Check this once you have data.

**4. RSS is stale.** A July 2026 sampling of Google News RSS found median
item age around 6.6 days, with only 7.6% of items six hours old or newer.
Fine for a trend measure, not for breaking-news response.

**5. `NEWS_SHARE = 0.35` is a judgement call**, as is the outlet
credibility table. Both are declared constants at the top of their files
rather than magic numbers — defend them as documented hyperparameters and
show a sensitivity analysis (how does the answer move at α = 0.2, 0.35,
0.5?). That sensitivity plot is a strong slide and pre-empts the obvious
question.
