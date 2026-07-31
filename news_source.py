"""
news_source.py
--------------
Headline collection from Google News RSS and direct Nepali outlet feeds.

    from news_source import NewsCollector
    nc = NewsCollector()
    df = nc.collect('kp oli')        # -> DataFrame of headlines

Design notes:

  * Google News RSS caps at roughly 100 items per query with no
    pagination. To go deeper we subdivide by time window (when:1d,
    when:7d, when:30d) and merge, which is the documented workaround.

  * Google News heavily syndicates: the same story appears under many
    outlets with near-identical titles. Without near-duplicate removal a
    single story counts ten times and silently dominates the aggregate.
    Dedup here is on a normalised token signature, not exact string match.

  * A headline only counts if it actually mentions the candidate.
    Google News keyword search is loose and will return tangential items.

  * Each headline carries a weight = source_credibility * recency_decay.
    Old news should not move a live estimate as much as this morning's.
"""

import hashlib
import re
import time
import urllib.parse
from datetime import datetime, timezone

import pandas as pd

import entities as ent
import preprocessing as pp

try:
    import feedparser
except ImportError:  # pragma: no cover
    feedparser = None

# --------------------------------------------------------------------------
# Feeds
#
# Google News search feeds are built per candidate. Direct outlet feeds are
# fetched whole and then filtered by candidate mention -- cheaper and less
# rate-limited than one query per candidate per outlet.
# --------------------------------------------------------------------------

GOOGLE_NEWS = 'https://news.google.com/rss/search'
GOOGLE_LOCALES = [
    {'hl': 'ne',    'gl': 'NP', 'ceid': 'NP:ne'},   # Nepali edition
    {'hl': 'en-US', 'gl': 'NP', 'ceid': 'NP:en'},   # English, Nepal edition
]
TIME_WINDOWS = ['when:1d', 'when:7d', 'when:30d']

# Direct RSS. Weight reflects editorial reach and reliability -- these are
# a judgement call and belong in your report as a documented table, not
# hidden in code.
OUTLET_FEEDS = {
    'onlinekhabar':  {'url': 'https://www.onlinekhabar.com/feed',   'weight': 1.0},
    'ratopati':      {'url': 'https://www.ratopati.com/feed',       'weight': 0.9},
    'setopati':      {'url': 'https://www.setopati.com/feed',       'weight': 1.0},
    'kathmandupost': {'url': 'https://kathmandupost.com/rss',       'weight': 1.0},
    'nagarik':       {'url': 'https://nagariknews.nagariknetwork.com/feed', 'weight': 0.9},
    'baahrakhari':   {'url': 'https://baahrakhari.com/feed',        'weight': 0.8},
    'himalkhabar':   {'url': 'https://www.himalkhabar.com/feed',    'weight': 0.9},
}

DEFAULT_SOURCE_WEIGHT = 0.7   # unknown outlet arriving via Google News
RECENCY_HALFLIFE_DAYS = 7.0   # a 7-day-old headline counts half as much
MIN_TITLE_CHARS = 15


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(entry) -> datetime:
    for key in ('published_parsed', 'updated_parsed'):
        val = entry.get(key)
        if val:
            try:
                return datetime.fromtimestamp(time.mktime(val), tz=timezone.utc)
            except Exception:
                pass
    return _now()


def _recency_weight(published: datetime) -> float:
    age_days = max(0.0, (_now() - published).total_seconds() / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALFLIFE_DAYS)


def _signature(title: str) -> str:
    """
    Near-duplicate key. Google News syndicates one story across many
    outlets with cosmetic title differences, so we key on a sorted bag of
    significant tokens rather than the raw string.
    """
    toks = pp.clean(title).split()
    toks = sorted(set(t for t in toks if len(t) > 2))[:10]
    return hashlib.md5(' '.join(toks).encode('utf-8')).hexdigest()


def _strip_source_suffix(title: str) -> str:
    """Google News appends ' - Outlet Name' to every title."""
    return re.sub(r'\s+-\s+[^-]{2,40}$', '', title).strip()


class NewsCollector:

    def __init__(self, outlet_feeds: dict = None, verbose: bool = True):
        if feedparser is None:
            raise ImportError('feedparser required:  pip install feedparser')
        self.outlet_feeds = outlet_feeds or OUTLET_FEEDS
        self.verbose = verbose

    # ------------------------------------------------------------- fetching

    def _google_news(self, candidate_key: str) -> list:
        cfg = ent.CANDIDATES.get(candidate_key, {})
        terms = [cfg.get('search', candidate_key)]
        # include the Devanagari alias so the Nepali edition returns results
        terms += [a for a in cfg.get('aliases', []) if re.search(r'[\u0900-\u097F]', a)][:1]

        rows = []
        for term in terms:
            for locale in GOOGLE_LOCALES:
                for window in TIME_WINDOWS:
                    q = urllib.parse.quote(f'{term} {window}')
                    url = (f'{GOOGLE_NEWS}?q={q}&hl={locale["hl"]}'
                           f'&gl={locale["gl"]}&ceid={locale["ceid"]}')
                    try:
                        feed = feedparser.parse(url)
                    except Exception as exc:
                        if self.verbose:
                            print(f'  google news failed ({window}): {exc}')
                        continue
                    for e in feed.entries:
                        title = _strip_source_suffix(e.get('title', ''))
                        src = ''
                        if e.get('source'):
                            src = e['source'].get('title', '') if isinstance(
                                e['source'], dict) else str(e['source'])
                        rows.append({
                            'text': title,
                            'link': e.get('link', ''),
                            'outlet': src or 'google_news',
                            'published': _parse_date(e),
                            'feed': 'google_news',
                        })
        return rows

    def _outlet_feed(self, name: str, cfg: dict) -> list:
        try:
            feed = feedparser.parse(cfg['url'])
        except Exception as exc:
            if self.verbose:
                print(f'  {name} failed: {exc}')
            return []
        rows = []
        for e in feed.entries:
            rows.append({
                'text': e.get('title', '').strip(),
                'link': e.get('link', ''),
                'outlet': name,
                'published': _parse_date(e),
                'feed': 'outlet',
            })
        return rows

    # ------------------------------------------------------------ filtering

    @staticmethod
    def _mentions(text: str, candidate_key: str) -> bool:
        cfg = ent.CANDIDATES.get(candidate_key)
        if not cfg:
            return candidate_key.lower() in str(text).lower()
        padded = ' ' + re.sub(r'\s+', ' ', str(text).lower()) + ' '
        for alias in cfg['aliases']:
            a = alias.lower()
            if re.fullmatch(r'[a-z0-9 ]+', a):
                if re.search(r'(?<![a-z])' + re.escape(a) + r'(?![a-z])', padded):
                    return True
            elif a in padded:
                return True
        return False

    # --------------------------------------------------------------- public

    def collect(self, candidate: str, include_outlets: bool = True) -> pd.DataFrame:
        key = ent.resolve(candidate)
        rows = self._google_news(key)
        if self.verbose:
            print(f'  google news: {len(rows)} raw items')

        if include_outlets:
            for name, cfg in self.outlet_feeds.items():
                got = self._outlet_feed(name, cfg)
                if self.verbose and got:
                    print(f'  {name}: {len(got)} items')
                rows.extend(got)

        if not rows:
            return pd.DataFrame(columns=['text', 'outlet', 'published',
                                         'weight', 'link'])

        df = pd.DataFrame(rows)
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'].str.len() >= MIN_TITLE_CHARS]

        # relevance: the headline must actually name the candidate
        before = len(df)
        df = df[df['text'].apply(lambda t: self._mentions(t, key))]
        if self.verbose:
            print(f'  relevance filter: {before} -> {len(df)}')

        if df.empty:
            return df.assign(weight=[])

        # near-duplicate removal (syndication)
        df['_sig'] = df['text'].apply(_signature)
        before = len(df)
        df = df.sort_values('published', ascending=False)
        df = df.drop_duplicates(subset=['_sig']).drop(columns=['_sig'])
        if self.verbose:
            print(f'  dedup: {before} -> {len(df)} unique stories')

        # per-item weight = source credibility x recency decay
        def source_weight(outlet: str) -> float:
            o = str(outlet).lower()
            for name, cfg in self.outlet_feeds.items():
                if name in o or o in name:
                    return cfg['weight']
            return DEFAULT_SOURCE_WEIGHT

        df['source_weight'] = df['outlet'].apply(source_weight)
        df['recency_weight'] = df['published'].apply(_recency_weight)
        df['weight'] = (df['source_weight'] * df['recency_weight']).round(4)

        df['script'] = df['text'].apply(pp.script_of)
        return df.reset_index(drop=True)


if __name__ == '__main__':
    nc = NewsCollector()
    for cand in ['kp oli', 'balen']:
        print(f'\n=== {cand} ===')
        d = nc.collect(cand)
        print(f'{len(d)} headlines')
        if not d.empty:
            for _, r in d.head(8).iterrows():
                print(f'  [{r.weight:.2f}] {r.outlet[:18]:20s} {r.text[:70]}')
