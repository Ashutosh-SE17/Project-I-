"""
validate_races.py
------------------
Head-to-head backtest: does the sentiment model correctly rank the winner
in five real constituency races, and does weighting party-level mentions
(PARTY_HIT_WEIGHT) help or hurt that call?

    python validate_races.py

Reads:
    entities.RACES          -- five real head-to-head results
    labelling_pool.csv       -- tagged comment pool
    models/sentiment_clf.pkl -- trained classifier (via ElectionAnalyzer)

Writes:
    race_validation.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import entities as ent
from news_source import NewsCollector          # reuse ._mentions, don't
                                                # re-derive alias matching
from sentiment_model import ElectionAnalyzer

if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE = Path(__file__).resolve().parent
POOL_PATH = BASE / 'labelling_pool.csv'
OUT_PATH = BASE / 'race_validation.json'

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

PARTY_WEIGHT_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_PARTY_WEIGHT = ent.PARTY_HIT_WEIGHT   # what the live app uses today


# --------------------------------------------------------------------------
# Comment selection
# --------------------------------------------------------------------------

def build_mention_masks(pool: pd.DataFrame, keys) -> dict:
    """One boolean Series per candidate key: does this row's text mention
    that candidate's aliases? Reuses NewsCollector._mentions -- the exact
    matching already trusted to decide whether a headline is on-topic."""
    texts = pool['text']
    return {k: texts.apply(lambda t: NewsCollector._mentions(t, k)) for k in keys}


def relevant_comments(pool: pd.DataFrame, key_a: str, key_b: str,
                      mention_masks: dict) -> pd.DataFrame:
    """Comments relevant to a race: tagged with either candidate, OR
    mentioning either candidate's aliases regardless of tag. One combined
    set, scored separately per candidate -- the attribution engine (not
    this filter) is what decides who a given comment counts for."""
    tag_mask = pool['candidate'].isin([key_a, key_b])
    mask = tag_mask | mention_masks[key_a] | mention_masks[key_b]
    return pool[mask].drop_duplicates(subset='text').reset_index(drop=True)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def shares(mean_a: float, mean_b: float) -> tuple:
    """[-1,+1] mean scores -> a two-way share that sums to 100%. Shift to
    [0,2] and normalize -- order-preserving, no extra temperature
    hyperparameter to justify. Falls back to 50/50 only in the
    near-impossible case where both means are ~-1."""
    a, b = mean_a + 1.0, mean_b + 1.0
    total = a + b
    if total <= 1e-9:
        return 50.0, 50.0
    return round(100 * a / total, 2), round(100 * b / total, 2)


def score_race(analyzer: ElectionAnalyzer, relevant: pd.DataFrame,
              key_a: str, key_b: str, party_hit_weight: float) -> tuple:
    scored_a = analyzer.score_frame(relevant, key_a, party_hit_weight=party_hit_weight)
    scored_b = analyzer.score_frame(relevant, key_b, party_hit_weight=party_hit_weight)
    return float(scored_a['sentiment_score'].mean()), float(scored_b['sentiment_score'].mean())


# --------------------------------------------------------------------------
# Console tables
# --------------------------------------------------------------------------

def print_selection_table(selection: list) -> None:
    print('Comment selection per race (sanity check)')
    print('-' * 72)
    for row in selection:
        print(f"  {row['constituency']:16s} n_relevant={row['n_relevant']:5d}")
        for cand, n in row['tag_breakdown'].items():
            print(f"      {cand:22s} {n:5d}")
    print()


def print_sweep_table(sweep: list, baseline: dict, constituencies: list) -> None:
    header = '  Party wt  ' + ''.join(f'{c[:14]:>16s}' for c in constituencies) + '   Total'
    print('Winner prediction vs. PARTY_HIT_WEIGHT')
    print('-' * len(header))
    print(header)

    def fmt_row(label, races, total):
        cells = ''.join(f'{"✓" if races[c]["correct"] else "✗":>16s}' for c in constituencies)
        return f'  {label:8s}{cells}   {total}/5'

    print(fmt_row('baseline', baseline['races'], baseline['total_correct']))
    for row in sweep:
        print(fmt_row(f"{row['party_hit_weight']:.1f}", row['races'], row['total_correct']))
    print()


def print_detail_table(detailed: list) -> None:
    print(f'Detailed breakdown at PARTY_HIT_WEIGHT={DEFAULT_PARTY_WEIGHT}')
    print('-' * 100)
    for d in detailed:
        print(f"  {d['constituency']}  (n={d['n_comments']})")
        print(f"    {d['candidate_a']:22s} mean={d['mean_score_a']:+.4f}  "
              f"predicted={d['predicted_share_a']:5.1f}%  actual={d['actual_share_a']:5.1f}%")
        print(f"    {d['candidate_b']:22s} mean={d['mean_score_b']:+.4f}  "
              f"predicted={d['predicted_share_b']:5.1f}%  actual={d['actual_share_b']:5.1f}%")
        mark = '✓' if d['correct'] else '✗'
        print(f"    predicted winner: {d['predicted_winner']:22s} "
              f"actual winner: {d['actual_winner']:22s} [{mark}]")
        print()


# --------------------------------------------------------------------------
# Diagnostic: Sarlahi-4 only
#
# This is the one close, balanced-sample race the model calls wrong
# (§ detailed breakdown). The question is whether that's the attribution
# layer failing to tell the two candidates apart (most comments land on
# the same attribution value for both, or on the CONTEXT_PRIOR fallback
# because neither name was actually detected), or whether attribution is
# doing its job and the comments themselves just don't lean the way the
# vote did.
# --------------------------------------------------------------------------

def diagnose_sarlahi_4(analyzer: ElectionAnalyzer, relevant: pd.DataFrame) -> None:
    key_a, key_b = 'amresh kumar singh', 'gagan thapa'

    scored_a = analyzer.score_frame(relevant, key_a, party_hit_weight=DEFAULT_PARTY_WEIGHT)
    scored_b = analyzer.score_frame(relevant, key_b, party_hit_weight=DEFAULT_PARTY_WEIGHT)

    # own_hits / rival_hits aren't in score_frame's output columns -- pull
    # them straight from attribution() again. Cheap: pure regex over text,
    # no model inference, so re-running it isn't wasteful here.
    info_a = [ent.attribution(t, key_a, party_hit_weight=DEFAULT_PARTY_WEIGHT)
             for t in relevant['text']]
    info_b = [ent.attribution(t, key_b, party_hit_weight=DEFAULT_PARTY_WEIGHT)
             for t in relevant['text']]
    own_a = np.array([i['own_hits'] for i in info_a])
    rival_a = np.array([i['rival_hits'] for i in info_a])
    own_b = np.array([i['own_hits'] for i in info_b])
    rival_b = np.array([i['rival_hits'] for i in info_b])

    n = len(relevant)
    print('=' * 100)
    print(f'DIAGNOSTIC: Sarlahi-4 attribution breakdown ({key_a} vs {key_b}), n={n}')
    print('=' * 100)

    # (a) distribution of attribution values
    print('\n(a) attribution value distribution')
    for label, scored in [(key_a, scored_a), (key_b, scored_b)]:
        attr = scored['attribution']
        n_pos1 = int((attr == 1.0).sum())
        n_neg1 = int((attr == -1.0).sum())
        n_prior = int(np.isclose(attr, ent.CONTEXT_PRIOR, atol=1e-6).sum())
        n_between = n - n_pos1 - n_neg1 - n_prior
        print(f'  {label:22s} +1.0={n_pos1:5d}  -1.0={n_neg1:5d}  '
              f'CONTEXT_PRIOR({ent.CONTEXT_PRIOR:+.2f})={n_prior:5d}  other={n_between:5d}')

    # (b) neither candidate named -> CONTEXT_PRIOR fallback fired
    print('\n(b) neither candidate named (own_hits==0 and rival_hits==0)')
    neither_a = int(((own_a == 0) & (rival_a == 0)).sum())
    neither_b = int(((own_b == 0) & (rival_b == 0)).sum())
    print(f'  {key_a:22s} {neither_a:5d} / {n}  ({100 * neither_a / n:.1f}%)')
    print(f'  {key_b:22s} {neither_b:5d} / {n}  ({100 * neither_b / n:.1f}%)')

    # (c) mean polarity split by attribution strength, per candidate
    print('\n(c) mean polarity where attribution is strongly positive/negative')
    for label, scored in [(key_a, scored_a), (key_b, scored_b)]:
        strong_pos = scored[scored['attribution'] > 0.5]
        strong_neg = scored[scored['attribution'] < -0.5]
        pp_mean = strong_pos['polarity'].mean() if len(strong_pos) else float('nan')
        pn_mean = strong_neg['polarity'].mean() if len(strong_neg) else float('nan')
        print(f'  {label:22s} attribution>+0.5  n={len(strong_pos):5d}  mean_polarity={pp_mean:+.4f}')
        print(f'  {"":22s} attribution<-0.5  n={len(strong_neg):5d}  mean_polarity={pn_mean:+.4f}')

    # (d) the 10 comments where the two candidates' attribution diverges most
    diff = pd.DataFrame({
        'text': relevant['text'].values,
        'attribution_a': scored_a['attribution'].values,
        'attribution_b': scored_b['attribution'].values,
    })
    diff['abs_diff'] = (diff['attribution_a'] - diff['attribution_b']).abs()
    top10 = diff.nlargest(10, 'abs_diff')
    print(f'\n(d) top 10 comments by |attribution_{key_a[:1]} - attribution_{key_b[:1]}|')
    for _, row in top10.iterrows():
        preview = str(row['text'])[:70].replace('\n', ' ')
        print(f"  a={row['attribution_a']:+.2f}  b={row['attribution_b']:+.2f}  {preview}")
    print()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    pool = pd.read_csv(POOL_PATH)
    pool['candidate'] = pool['candidate'].apply(ent.resolve)
    pool['text'] = pool['text'].astype(str)

    analyzer = ElectionAnalyzer()

    race_keys = sorted({k for race in ent.RACES
                        for k in (race['candidate_a'], race['candidate_b'])})
    mention_masks = build_mention_masks(pool, race_keys)

    race_pools = {race['constituency']: relevant_comments(
                      pool, race['candidate_a'], race['candidate_b'], mention_masks)
                  for race in ent.RACES}

    # ---- comment-selection sanity check ----
    selection = []
    for race in ent.RACES:
        relevant = race_pools[race['constituency']]
        selection.append({
            'constituency': race['constituency'],
            'n_relevant': len(relevant),
            'tag_breakdown': relevant['candidate'].value_counts().to_dict(),
        })

    # ---- volume-only baseline (null hypothesis: no sentiment at all) ----
    baseline_races = {}
    baseline_correct = 0
    for race, sel in zip(ent.RACES, selection):
        tags = sel['tag_breakdown']
        n_a = tags.get(race['candidate_a'], 0)
        n_b = tags.get(race['candidate_b'], 0)
        predicted = race['candidate_a'] if n_a >= n_b else race['candidate_b']
        correct = predicted == race['winner']
        baseline_correct += int(correct)
        baseline_races[race['constituency']] = {
            'predicted_winner': predicted, 'correct': correct,
            'n_a': n_a, 'n_b': n_b,
        }
    baseline = {'races': baseline_races, 'total_correct': baseline_correct}

    # ---- party-weight sweep ----
    sweep = []
    means_at_default = {}
    for w in PARTY_WEIGHT_SWEEP:
        row = {'party_hit_weight': w, 'races': {}, 'total_correct': 0}
        for race in ent.RACES:
            relevant = race_pools[race['constituency']]
            mean_a, mean_b = score_race(analyzer, relevant, race['candidate_a'],
                                        race['candidate_b'], w)
            predicted = race['candidate_a'] if mean_a >= mean_b else race['candidate_b']
            correct = predicted == race['winner']
            row['races'][race['constituency']] = {
                'predicted_winner': predicted, 'correct': correct,
            }
            row['total_correct'] += int(correct)
            if w == DEFAULT_PARTY_WEIGHT:
                means_at_default[race['constituency']] = (mean_a, mean_b)
        sweep.append(row)

    # ---- detailed breakdown at the default (production) weight ----
    detailed = []
    for race in ent.RACES:
        relevant = race_pools[race['constituency']]
        if race['constituency'] in means_at_default:
            mean_a, mean_b = means_at_default[race['constituency']]
        else:
            mean_a, mean_b = score_race(analyzer, relevant, race['candidate_a'],
                                        race['candidate_b'], DEFAULT_PARTY_WEIGHT)
        share_a, share_b = shares(mean_a, mean_b)
        actual_a = round(100 * race['votes_a'] / race['total_votes'], 2)
        actual_b = round(100 * race['votes_b'] / race['total_votes'], 2)
        predicted = race['candidate_a'] if mean_a >= mean_b else race['candidate_b']
        detailed.append({
            'constituency': race['constituency'],
            'candidate_a': race['candidate_a'], 'candidate_b': race['candidate_b'],
            'n_comments': len(relevant),
            'mean_score_a': round(mean_a, 4), 'mean_score_b': round(mean_b, 4),
            'predicted_share_a': share_a, 'predicted_share_b': share_b,
            'actual_share_a': actual_a, 'actual_share_b': actual_b,
            'predicted_winner': predicted, 'actual_winner': race['winner'],
            'correct': predicted == race['winner'],
        })

    constituencies = [race['constituency'] for race in ent.RACES]
    print_selection_table(selection)
    print_sweep_table(sweep, baseline, constituencies)
    print_detail_table(detailed)
    diagnose_sarlahi_4(analyzer, race_pools['Sarlahi-4'])

    out = {
        'default_party_hit_weight': DEFAULT_PARTY_WEIGHT,
        'comment_selection': selection,
        'volume_baseline': baseline,
        'party_weight_sweep': sweep,
        'detailed': detailed,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved -> {OUT_PATH.name}')


if __name__ == '__main__':
    main()
