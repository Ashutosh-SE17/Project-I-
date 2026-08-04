"""
sentiment_model.py
------------------
Drop-in replacement for the lexicon-based ElectionAnalyzer.

Public API is unchanged:
    ElectionAnalyzer().analyze_candidate(candidate, source) -> dict
so app.py and script.js keep working without modification.

What changed inside:

    OLD:  hand-weighted lexicon -> unbounded sum -> sigmoid(sum / 25)
    NEW:  supervised classifier -> polarity in [-1,+1]
          x entity attribution  -> score in [-1,+1]
          -> MEAN over comments -> sigmoid(gain * mean)

Why mean and not sum: the old sum grew with however many comments the
API happened to return, so win_probability was really a function of
sample size. |sum| > 213 saturates the sigmoid completely, and a real
1078-comment corpus produced sum = -320 -> 0.00%. A mean is bounded in
[-1,+1] by construction and cannot saturate.
"""

import os
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()          # reads .env from the project root
except ImportError:
    pass

import preprocessing as pp
import entities as ent

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'sentiment_clf.pkl'
CACHE_PATH = BASE_DIR / 'cache.json'

# --------------------------------------------------------------------------
# Aggregation hyperparameters -- document these in your report
# --------------------------------------------------------------------------

SIGMOID_GAIN = 4.0        # win_prob = sigmoid(GAIN * mean_score)
                          #   mean +0.00 -> 50%
                          #   mean +0.25 -> 73%
                          #   mean +0.50 -> 88%
                          #   mean +1.00 -> 98%
MIN_COMMENTS = 25         # below this, refuse to report a probability
BOOTSTRAP_N = 1000        # resamples for the confidence interval
CACHE_TTL = 60 * 60 * 6   # seconds

# --- comment volume ---
MAX_VIDEOS = 8            # search costs 100 quota units; comments cost 1/page
MAX_COMMENTS = 1000       # the number the old fetch_1000_data only claimed

# --- news blending ---
# Block weighting: news contributes a fixed SHARE of the final score,
# regardless of how many headlines happen to be found. A per-item weight
# ("one headline = 5 comments") is unstable, because the effective share
# then swings with however many items the feed returned that day.
NEWS_SHARE = 0.35         # headlines carry 35% of the signal, comments 65%
MIN_NEWS_FULL = 15        # below this many headlines, shrink the share
                          # proportionally -- 3 syndicated stories should not
                          # get to move a national estimate by 35%


class ElectionAnalyzer:

    def __init__(self, model_path: Path = MODEL_PATH):
        self.api_key = os.environ.get('YOUTUBE_API_KEY')
        self.youtube = None
        if self.api_key:
            try:
                from googleapiclient.discovery import build
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            except Exception as exc:
                print(f'[warn] YouTube client unavailable: {exc}')
        else:
            print('[warn] YOUTUBE_API_KEY not set -- live fetch disabled, '
                  'cache/CSV only')

        self.model = None
        self.classes = []
        self.model_stats = {}
        if Path(model_path).exists():
            import joblib
            bundle = joblib.load(model_path)
            self.model = bundle['model']
            self.classes = list(self.model.named_steps['clf'].classes_)
            self.model_stats = bundle.get('stats', {})
        else:
            print(f'[warn] no model at {model_path}. Run train_sentiment.py first.')

    # ---------------------------------------------------------------- scoring

    def polarity(self, texts, channel: str = 'comment'):
        """
        Text -> polarity in [-1, +1], via P(positive) - P(negative).

        This is the single method that replaces the entire lexicon. Neutral
        probability mass simply shrinks the magnitude, which is exactly the
        behaviour we want: an uncertain comment should move the aggregate
        less than a confident one.
        """
        if self.model is None:
            raise RuntimeError('No trained model. Run train_sentiment.py.')

        from train_sentiment import mark_channel
        cleaned = [mark_channel(pp.clean(t), channel) for t in texts]
        proba = self.model.predict_proba(cleaned)

        ipos = self.classes.index('positive') if 'positive' in self.classes else None
        ineg = self.classes.index('negative') if 'negative' in self.classes else None

        pos = proba[:, ipos] if ipos is not None else np.zeros(len(cleaned))
        neg = proba[:, ineg] if ineg is not None else np.zeros(len(cleaned))

        pol = pos - neg
        labels = self.model.predict(cleaned)
        return pol, labels, proba

    def score_frame(self, df: pd.DataFrame, candidate: str,
                    channel: str = 'comment',
                    party_hit_weight: float = None) -> pd.DataFrame:
        """Attach polarity, attribution and final score to every item.

        `party_hit_weight` defaults to None, which resolves to the live
        `entities.PARTY_HIT_WEIGHT` -- so every existing caller (app.py,
        analyze_candidate, score_news) is unaffected. It's exposed here so
        validation code can sweep the constant without touching the
        module global.
        """
        pol, labels, _ = self.polarity(df['text'].tolist(), channel=channel)
        df = df.copy()
        df['polarity'] = np.round(pol, 4)
        df['sentiment_label'] = [str(l).capitalize() for l in labels]

        if party_hit_weight is None:
            party_hit_weight = ent.PARTY_HIT_WEIGHT

        attr, scand = [], []
        for text in df['text']:
            info = ent.attribution(text, candidate, party_hit_weight=party_hit_weight)
            attr.append(info['attribution'])
            scand.append(info['scandal'])
        df['attribution'] = attr
        df['scandal'] = scand

        # ---- the composition step ----
        raw = df['polarity'] * df['attribution'] + df['scandal'] * np.sign(
            df['attribution'].replace(0, 1))
        df['sentiment_score'] = np.clip(raw, -1.0, 1.0).round(4)

        df['script'] = df['text'].apply(pp.script_of)
        return df

    # ------------------------------------------------------------ aggregation

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def win_probability(self, comment_scores, news_scores=None,
                        news_weights=None) -> dict:
        """
        Blend the two channels, then map to a probability.

            combined = a * news_mean + (1 - a) * comment_mean
            a        = NEWS_SHARE * min(1, n_news / MIN_NEWS_FULL)

        `a` is a SHARE, not a per-item multiplier. That is deliberate: with
        per-item weights the influence of news swings with however many
        headlines the feed returned, so the same underlying sentiment gives
        a different answer on a slow news day. A share is stable, and the
        shrinkage term stops three syndicated stories from claiming 35% of
        a national estimate.

        Within news, items are weighted by source credibility x recency.
        """
        c = np.asarray(comment_scores, dtype=float)
        c_mean = float(c.mean()) if len(c) else 0.0

        n_news = 0
        n_mean = 0.0
        alpha = 0.0

        if news_scores is not None and len(news_scores):
            n = np.asarray(news_scores, dtype=float)
            w = (np.asarray(news_weights, dtype=float)
                 if news_weights is not None else np.ones(len(n)))
            w = np.clip(w, 1e-6, None)
            n_mean = float(np.average(n, weights=w))
            n_news = int(len(n))
            alpha = NEWS_SHARE * min(1.0, n_news / MIN_NEWS_FULL)

        combined = alpha * n_mean + (1.0 - alpha) * c_mean
        prob = self._sigmoid(SIGMOID_GAIN * combined) * 100.0

        # Bootstrap the CI over BOTH channels so the reported uncertainty
        # reflects how thin the news sample is, not just the comment sample.
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(BOOTSTRAP_N):
            cb = rng.choice(c, size=len(c), replace=True).mean() if len(c) else 0.0
            if n_news:
                idx = rng.integers(0, n_news, n_news)
                nb = float(np.average(np.asarray(news_scores)[idx],
                                      weights=np.clip(np.asarray(news_weights)[idx], 1e-6, None)))
            else:
                nb = 0.0
            boot.append(self._sigmoid(SIGMOID_GAIN * (alpha * nb + (1 - alpha) * cb)) * 100.0)
        lo, hi = np.percentile(boot, [2.5, 97.5])

        return {
            'win_probability': round(prob, 2),
            'mean_score': round(combined, 4),
            'comment_mean': round(c_mean, 4),
            'news_mean': round(n_mean, 4),
            'news_share_applied': round(alpha, 3),
            'ci_low': round(float(lo), 2),
            'ci_high': round(float(hi), 2),
            'n': int(len(c)),
            'n_news': n_news,
        }

    # ------------------------------------------------------------------ data

    def _load_cache(self, key: str):
        if not CACHE_PATH.exists():
            return None
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding='utf-8'))
        except Exception:
            return None
        entry = cache.get(key)
        if isinstance(entry, dict) and 'fetched_at' in entry:
            if time.time() - entry['fetched_at'] > CACHE_TTL:
                return None
            return entry.get('comments')
        if isinstance(entry, list):       # legacy format
            return entry
        return None

    def _save_cache(self, key: str, comments: list) -> None:
        cache = {}
        if CACHE_PATH.exists():
            try:
                cache = json.loads(CACHE_PATH.read_text(encoding='utf-8'))
            except Exception:
                cache = {}
        cache[key] = {'fetched_at': time.time(), 'comments': comments}
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False),
                              encoding='utf-8')

    def fetch_comments(self, search_term: str, max_videos: int = MAX_VIDEOS,
                       max_comments: int = MAX_COMMENTS) -> pd.DataFrame:
        """
        Fetch comments. Errors are reported rather than swallowed -- the old
        bare `except: pass` turned quota exhaustion into a generic
        'No data retrieved' with no way to diagnose it.
        """
        if self.youtube is None:
            return pd.DataFrame(columns=['text'])

        rows, errors = [], []
        try:
            search = self.youtube.search().list(
                q=search_term, part='id,snippet',
                maxResults=max_videos, type='video',
                relevanceLanguage='ne',
            ).execute()
        except Exception as exc:
            raise RuntimeError(f'YouTube search failed: {exc}') from exc

        videos = [i['id'].get('videoId') for i in search.get('items', [])]
        videos = [v for v in videos if v]

        # Round-robin pagination across videos so the sample is not dominated
        # by whichever video happens to have the most comments. Each page is
        # 1 quota unit, so reaching 1000 costs ~10 units on top of the 100
        # the search already cost.
        tokens = {v: None for v in videos}
        exhausted = set()

        while len(rows) < max_comments and len(exhausted) < len(videos):
            for vid in videos:
                if vid in exhausted or len(rows) >= max_comments:
                    continue
                try:
                    kwargs = dict(part='snippet', videoId=vid, maxResults=100,
                                  textFormat='plainText', order='relevance')
                    if tokens[vid]:
                        kwargs['pageToken'] = tokens[vid]
                    resp = self.youtube.commentThreads().list(**kwargs).execute()
                except Exception as exc:
                    errors.append(f'{vid}: {type(exc).__name__}')
                    exhausted.add(vid)
                    continue

                for c in resp.get('items', []):
                    rows.append({
                        'text': c['snippet']['topLevelComment']['snippet']['textDisplay'],
                        'video_id': vid,
                    })

                tokens[vid] = resp.get('nextPageToken')
                if not tokens[vid]:
                    exhausted.add(vid)

        if errors:
            print(f'[warn] {len(errors)} page fetches failed: {errors[:3]}')
        print(f'[info] fetched {len(rows)} comments from {len(videos)} videos')
        return pd.DataFrame(rows[:max_comments])

    # --------------------------------------------------------------- public

    def score_news(self, candidate: str) -> pd.DataFrame:
        """Fetch and score headlines. Returns an empty frame on any failure."""
        try:
            from news_source import NewsCollector
            nc = NewsCollector(verbose=False)
            news = nc.collect(candidate)
        except Exception as exc:
            print(f'[warn] news collection failed: {exc}')
            return pd.DataFrame()

        if news.empty:
            return news
        return self.score_frame(news, candidate, channel='news')

    def analyze_candidate(self, candidate: str, source: str = 'YouTube',
                          include_news: bool = True) -> dict:
        key = ent.resolve(candidate)
        cfg = ent.CANDIDATES.get(key, {})
        search_term = cfg.get('search', candidate)
        display = cfg.get('display', candidate)

        cached = self._load_cache(key)
        if cached:
            df = pd.DataFrame(cached)
        else:
            try:
                df = self.fetch_comments(search_term)
            except RuntimeError as exc:
                return {'error': str(exc)}
            if not df.empty:
                self._save_cache(key, df[['text']].to_dict(orient='records'))

        if df.empty:
            return {'error': f'No comments retrieved for {display}.'}

        df = df.drop_duplicates(subset=['text'])
        df = self.score_frame(df, key)

        if len(df) < MIN_COMMENTS:
            return {'error': f'Only {len(df)} comments found for {display}; '
                             f'need at least {MIN_COMMENTS} for a stable estimate.'}

        news_df = self.score_news(key) if include_news else pd.DataFrame()

        if not news_df.empty:
            agg = self.win_probability(
                df['sentiment_score'].values,
                news_df['sentiment_score'].values,
                news_df['weight'].values,
            )
        else:
            agg = self.win_probability(df['sentiment_score'].values)

        # ---- REPRESENTATIVE sample, not outcome-matched ----
        # The old version showed the 8 most positive comments when the
        # prediction was favourable and the 8 most negative when it was not,
        # which made the evidence table circular. This draws a stratified
        # random sample instead, so the table reflects the real distribution.
        rng = np.random.default_rng(42)
        sample_rows = []
        for label, grp in df.groupby('sentiment_label'):
            share = len(grp) / len(df)
            take = max(1, round(share * 10))
            idx = rng.choice(grp.index, size=min(take, len(grp)), replace=False)
            sample_rows.extend(idx)
        sample = df.loc[sample_rows[:10]]

        counts = df['sentiment_label'].value_counts().to_dict()

        return {
            'candidate': display,
            'win_probability': agg['win_probability'],
            'confidence_interval': [agg['ci_low'], agg['ci_high']],
            'counts': counts,
            'average_score': agg['mean_score'],
            'n_comments': agg['n'],
            'n_headlines': agg.get('n_news', 0),
            'channel_scores': {
                'comments': agg['comment_mean'],
                'news': agg['news_mean'],
                'news_share_applied': agg['news_share_applied'],
            },
            'top_headlines': (
                news_df.nlargest(min(8, len(news_df)), 'weight')
                       [['text', 'outlet', 'sentiment_label',
                         'sentiment_score', 'weight']]
                       .to_dict(orient='records')
                if not news_df.empty else []
            ),
            'trends': {'Initial': 0.0, 'Current': agg['mean_score']},
            'script_breakdown': df.groupby('script')['sentiment_score']
                                  .agg(['count', 'mean']).round(3)
                                  .to_dict(orient='index'),
            'model_metrics': self.model_stats,
            'sample_data': sample[['text', 'sentiment_label', 'sentiment_score',
                                   'polarity', 'attribution']]
                                 .to_dict(orient='records'),
            'sampling_note': 'Stratified random sample, proportional to the '
                             'predicted class distribution.',
        }
