"""
collect_pool.py
---------------
Builds the single de-duplicated pool that label_tool.py draws from.

    # merge what you already have
    python collect_pool.py

    # merge, then top up from YouTube across all five candidates
    python collect_pool.py --fetch --per-candidate 300

    # headlines only, into their own file -- never touches labelling_pool.csv
    python collect_pool.py --news-only

Output: labelling_pool.csv  (columns: text, candidate, script, source)
--news-only output: headline_pool.csv  (columns: text, candidate, outlet,
published, weight, script, channel). Additive: existing rows are kept and
merged with freshly fetched ones, deduped on normalised text.

Why a separate step: your comments currently live in two files with two
different shapes, one of them holding a single candidate. Labelling from
one merged, script-balanced pool is what stops the classifier from
learning "this is a KP Oli video, therefore negative" instead of learning
sentiment.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

import preprocessing as pp

BASE = Path(__file__).resolve().parent


def from_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    out = pd.DataFrame({'text': df[text_col].astype(str)})
    out['candidate'] = df['candidate'] if 'candidate' in df.columns else 'unknown'
    out['source'] = path.name
    out['channel'] = 'comment'
    return out


def from_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        cache = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return pd.DataFrame()

    rows = []
    for key, entry in cache.items():
        items = entry.get('comments', []) if isinstance(entry, dict) else entry
        for it in items:
            text = it.get('text') if isinstance(it, dict) else it
            if text:
                rows.append({'text': str(text), 'candidate': key,
                             'source': path.name, 'channel': 'comment'})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Extra search phrasings per candidate, on top of entities.CANDIDATES[key]
# ['search']. Broadens coverage beyond just-the-name videos: interviews and
# speeches skew more formal/scripted than reaction videos, and the
# Devanagari-spelling query surfaces videos the romanized search misses
# entirely.
# --------------------------------------------------------------------------

EXTRA_QUERIES = {
    'balen': [
        'Balen Shah interview',
        'बालेन शाह',
        'Balen Shah RSP',
        'Balen Shah bhasan',
    ],
    'kp oli': [
        'KP Oli interview',
        'केपी ओली',
        'KP Oli UML',
        'KP Oli bhasan',
    ],
    'gagan thapa': [
        'Gagan Thapa interview',
        'गगन थापा',
        'Gagan Thapa Congress',
        'Gagan Thapa bhasan',
    ],
    'prachanda': [
        'Prachanda interview',
        'प्रचण्ड',
        'Prachanda Maoist Centre',
        'Prachanda bhasan',
    ],
    'harka sampang': [
        'Harka Sampang interview',
        'हर्क सम्पाङ',
        'Harka Sampang Dharan mayor',
        'Harka Sampang bhasan',
    ],
    'rabi lamichhane': [
        'Rabi Lamichhane interview',
        'रवि लामिछाने',
        'Rabi Lamichhane RSP',
        'Rabi Lamichhane bhasan',
    ],
}


def fetch_fresh(per_candidate: int, only: str = None) -> pd.DataFrame:
    """Top up the pool with new comments across every registered candidate,
    querying several phrasings per candidate (name, interview, Devanagari
    spelling, party context, speech) for broader coverage."""
    import entities as ent
    from sentiment_model import ElectionAnalyzer

    if not os.environ.get('YOUTUBE_API_KEY'):
        print('  YOUTUBE_API_KEY not set -- skipping fetch')
        return pd.DataFrame()

    analyzer = ElectionAnalyzer()

    candidates = ent.CANDIDATES
    if only:
        key = ent.resolve(only)
        if key not in candidates:
            print(f'  unknown candidate: {only!r} -- skipping fetch')
            return pd.DataFrame()
        candidates = {key: candidates[key]}

    frames = []
    for key, cfg in candidates.items():
        queries = [cfg['search']] + EXTRA_QUERIES.get(key, [])
        print(f'  fetching {cfg["display"]}...')
        for q in queries:
            try:
                df = analyzer.fetch_comments(
                    q, max_videos=4, max_comments=per_candidate // len(queries))
                df['candidate'] = key
                df['source'] = f'yt:{q}'
                df['channel'] = 'comment'
                frames.append(df[['text', 'candidate', 'source', 'channel']])
                print(f'    {q!r}: {len(df)}')
            except Exception as exc:
                if 'quota' in str(exc).lower():
                    collected = sum(len(f) for f in frames)
                    print(f'  quota exhausted -- stopping fetch, returning '
                          f'{collected} comments collected so far')
                    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                print(f'    {q!r}: failed ({type(exc).__name__})')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_news() -> pd.DataFrame:
    """Build a HEADLINE pool. These need their own labels -- a headline is
    a different register from a comment and the model must see both."""
    import entities as ent
    from news_source import NewsCollector

    nc = NewsCollector(verbose=False)
    frames = []
    for key, cfg in ent.CANDIDATES.items():
        print(f'  headlines for {cfg["display"]}...', end=' ', flush=True)
        try:
            d = nc.collect(key)
            if d.empty:
                print('0')
                continue
            d['candidate'] = key
            d['source'] = 'rss'
            d['channel'] = 'news'
            frames.append(d[['text', 'candidate', 'source', 'channel']])
            print(f'{len(d)}')
        except Exception as exc:
            print(f'failed ({type(exc).__name__})')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


HEADLINE_COLUMNS = ['text', 'candidate', 'outlet', 'published', 'weight',
                    'script', 'channel']


def fetch_all_headlines() -> pd.DataFrame:
    """Fetch headlines for every registered candidate, with the full
    metadata (outlet, published, weight) that the standalone headline pool
    keeps -- this never touches labelling_pool.csv."""
    import entities as ent
    from news_source import NewsCollector

    nc = NewsCollector(verbose=False)
    frames = []
    for key, cfg in ent.CANDIDATES.items():
        print(f'  headlines for {cfg["display"]}...', end=' ', flush=True)
        try:
            d = nc.collect(key)
            if d.empty:
                print('0')
                continue
            d = d.copy()
            d['candidate'] = key
            d['channel'] = 'news'
            frames.append(d[HEADLINE_COLUMNS])
            print(f'{len(d)}')
        except Exception as exc:
            print(f'failed ({type(exc).__name__})')
    return (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=HEADLINE_COLUMNS))


def run_news_only(out_path: Path) -> int:
    """Additive headline collection. Never reads or writes labelling_pool.csv.

    Loads whatever headline_pool.csv already has, fetches fresh headlines
    for all candidates, dedupes on normalised text, and refuses to write if
    the merged result would be smaller than what's already on disk -- that
    would mean something went wrong upstream (e.g. RSS feeds down), not a
    real shrink.
    """
    print('Fetching headlines for all candidates...\n')
    new_df = fetch_all_headlines()

    if out_path.exists():
        existing = pd.read_csv(out_path)
    else:
        existing = pd.DataFrame(columns=HEADLINE_COLUMNS)
    before = len(existing)

    parts = [d for d in (existing, new_df) if not d.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else existing
    combined['text'] = combined['text'].astype(str).str.strip()
    combined['_norm'] = combined['text'].apply(pp.clean)
    combined = combined[combined['_norm'].str.strip().astype(bool)]
    combined = combined.drop_duplicates(subset=['_norm']).drop(columns=['_norm'])
    combined = combined.reset_index(drop=True)
    after = len(combined)

    if after < before:
        print(f'\n  WARNING: merged result ({after}) is smaller than the '
              f'existing {out_path.name} ({before}). Aborting without '
              f'writing -- investigate before re-running.')
        return 1

    combined.to_csv(out_path, index=False)

    print(f'\n{before} existing -> {len(new_df)} fetched -> {after} unique '
          f'-> {out_path.name}\n')
    print('by candidate:')
    print(combined['candidate'].value_counts().to_string())
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='labelling_pool.csv')
    ap.add_argument('--fetch', action='store_true',
                    help='top up comments from YouTube')
    ap.add_argument('--news', action='store_true',
                    help='also build a headline pool (labelled separately '
                         'as channel=news)')
    ap.add_argument('--news-only', action='store_true',
                    help='fetch headlines for all candidates into '
                         'headline_pool.csv only -- never touches '
                         'labelling_pool.csv')
    ap.add_argument('--headline-out', default='headline_pool.csv',
                    help='output path for --news-only')
    ap.add_argument('--per-candidate', type=int, default=1000)
    ap.add_argument('--min-chars', type=int, default=8)
    ap.add_argument('--only', default=None,
                    help='fetch just this one candidate (e.g. --only balen)')
    args = ap.parse_args()

    if args.news_only:
        return run_news_only(BASE / args.headline_out)

    parts = [
        from_csv(BASE / 'election_data.csv'),
        from_cache(BASE / 'cache.json'),
    ]
    if args.fetch:
        parts.append(fetch_fresh(args.per_candidate, args.only))
    if args.news:
        parts.append(fetch_news())

    parts = [p for p in parts if not p.empty]
    if not parts:
        raise SystemExit('No input data found.')

    df = pd.concat(parts, ignore_index=True)
    before = len(df)

    import entities as ent
    df['candidate'] = df['candidate'].apply(
        lambda c: ent.resolve(c) if str(c) != 'unknown' else 'unknown')

    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() >= args.min_chars]

    # de-duplicate on NORMALISED text, so "KP Oli chor!!!" and
    # "kp oli chor" are not labelled twice
    df['_norm'] = df['text'].apply(pp.clean)
    df = df[df['_norm'].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=['_norm']).drop(columns=['_norm'])

    df['script'] = df['text'].apply(pp.script_of)
    df = df.reset_index(drop=True)

    df.to_csv(args.out, index=False)

    print(f'\n{before} raw -> {len(df)} unique comments -> {args.out}\n')
    print('by candidate:')
    print(df['candidate'].value_counts().to_string())
    print('\nby script:')
    print(df['script'].value_counts().to_string())
    print('\nby channel:')
    print(df['channel'].value_counts().to_string())

    if len(df) < 1500:
        print(f'\n  Pool is {len(df)}. Run with --fetch to top up before '
              f'labelling, or label what you have and add more later.')

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
