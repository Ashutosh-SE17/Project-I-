"""
validate_parties.py
--------------------
Party-level election backtest.

validate_races.py showed candidate-level scoring is thin: the four newer
opponents are named directly in only 0-19 of 9,123 pool comments. Parties,
by contrast, are discussed heavily via symbols and slogans (SHARED_PARTY_
TERMS hits alone run into the hundreds per race -- see the party-signal
diagnostic in validate_races.py). This tests whether PARTY sentiment
predicts the winning party in each of the five real races, instead of
candidate sentiment predicting the winning candidate.

    python validate_parties.py

Reads:
    entities.RACES / entities.CANDIDATES / entities.SHARED_PARTY_TERMS
    labelling_pool.csv
    models/sentiment_clf.pkl (via ElectionAnalyzer)

Writes:
    party_validation.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import entities as ent
from news_source import NewsCollector          # reuse ._mentions
from sentiment_model import ElectionAnalyzer
from validate_races import build_mention_masks, relevant_comments

if os.name == 'nt':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE = Path(__file__).resolve().parent
POOL_PATH = BASE / 'labelling_pool.csv'
OUT_PATH = BASE / 'party_validation.json'

BOOTSTRAP_N = 1000   # resamples for the 95% CI on mean party polarity
CI_SEED = 42


# --------------------------------------------------------------------------
# Party comment selection
# --------------------------------------------------------------------------

def party_members(party: str) -> list:
    return [k for k, cfg in ent.CANDIDATES.items() if cfg.get('party') == party]


def party_comment_mask(df: pd.DataFrame, party: str, mention_masks: dict) -> pd.Series:
    """Comments containing this party's SHARED_PARTY_TERMS, OR naming any
    candidate registered under this party (regardless of which race that
    candidate belongs to)."""
    terms = ent.SHARED_PARTY_TERMS.get(party, [])
    if terms:
        term_mask = df['text'].apply(lambda t: bool(ent._count_hits(ent._norm(t), terms)))
    else:
        term_mask = pd.Series(False, index=df.index)

    member_mask = pd.Series(False, index=df.index)
    for member in party_members(party):
        member_mask = member_mask | mention_masks.get(
            member, df['text'].apply(lambda t: NewsCollector._mentions(t, member)))
    return term_mask | member_mask


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = CI_SEED) -> tuple:
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def score_party(analyzer: ElectionAnalyzer, texts: pd.Series) -> dict:
    n = len(texts)
    if n == 0:
        return {'n': 0, 'mean_polarity': None, 'ci_low': None, 'ci_high': None}
    pol, _, _ = analyzer.polarity(texts.tolist())
    lo, hi = bootstrap_ci(pol)
    return {'n': int(n), 'mean_polarity': round(float(pol.mean()), 4),
            'ci_low': lo, 'ci_high': hi}


# --------------------------------------------------------------------------
# Console tables
# --------------------------------------------------------------------------

def print_party_scores(title: str, scores: dict) -> None:
    print(title)
    print('-' * 72)
    print(f"  {'party':18s} {'n':>6s} {'mean polarity':>15s} {'95% CI':>18s}")
    for party, s in scores.items():
        if s['n'] == 0:
            print(f"  {party:18s} {'--':>6s} {'no data':>15s} {'--':>18s}")
        else:
            ci = f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]"
            print(f"  {party:18s} {s['n']:6d} {s['mean_polarity']:+15.4f} {ci:>18s}")
    print()


def print_race_table(rows: list) -> None:
    print('Per-race prediction (both scoping variants)')
    print('-' * 100)
    for r in rows:
        print(f"  {r['constituency']}   actual winner party: {r['actual_party']}")
        print(f"    {r['party_a']} ({r['candidate_a']})  vs  {r['party_b']} ({r['candidate_b']})")
        for scope in ('national', 'race_local'):
            sa, sb = r[f'{scope}_score_a'], r[f'{scope}_score_b']
            if not r[f'{scope}_scoreable']:
                print(f"    {scope:10s}  UNSCOREABLE (no data for one or both parties)")
                continue
            mark = '✓' if r[f'{scope}_correct'] else '✗'
            print(f"    {scope:10s}  {r['party_a']}: mean={sa['mean_polarity']:+.4f} n={sa['n']:<5d} "
                  f"{r['party_b']}: mean={sb['mean_polarity']:+.4f} n={sb['n']:<5d}  "
                  f"predicted={r[f'{scope}_predicted']:<16s} [{mark}]")
        print()


def print_totals(totals: dict) -> None:
    print('Totals (5 races)')
    print('-' * 72)
    for name, t in totals.items():
        if 'scoreable' in t:
            suffix = '' if t['scoreable'] == 5 else f"  ({5 - t['scoreable']} unscoreable)"
            print(f"  {name:24s} {t['correct']}/{t['scoreable']} correct{suffix}")
        else:
            print(f"  {name:24s} {t['correct']}/{t['total']} correct")
    print()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    pool = pd.read_csv(POOL_PATH)
    pool['candidate'] = pool['candidate'].apply(ent.resolve)
    pool['text'] = pool['text'].astype(str)

    analyzer = ElectionAnalyzer()

    parties = list(ent.SHARED_PARTY_TERMS.keys())
    race_keys = sorted({k for race in ent.RACES
                        for k in (race['candidate_a'], race['candidate_b'])})
    mention_masks = build_mention_masks(pool, race_keys)

    race_pools = {race['constituency']: relevant_comments(
                      pool, race['candidate_a'], race['candidate_b'], mention_masks,
                      race.get('race_queries'))
                  for race in ent.RACES}

    # ---- national: one score per party, over the whole pool ----
    national_scores = {}
    for party in parties:
        mask = party_comment_mask(pool, party, mention_masks)
        national_scores[party] = score_party(analyzer, pool.loc[mask, 'text'])

    # ---- race-local: one score per party PER RACE, over that race's
    #      relevant set only (tag + alias-mention + shared-query union,
    #      same construction validate_races.py uses) ----
    local_scores = {}
    for race in ent.RACES:
        relevant = race_pools[race['constituency']]
        local_mention_masks = build_mention_masks(relevant, race_keys)
        parties_in_race = {ent.CANDIDATES[race['candidate_a']]['party'],
                           ent.CANDIDATES[race['candidate_b']]['party']}
        local_scores[race['constituency']] = {}
        for party in parties_in_race:
            mask = party_comment_mask(relevant, party, local_mention_masks)
            local_scores[race['constituency']][party] = score_party(
                analyzer, relevant.loc[mask, 'text'])

    # ---- per-race prediction, both scopes, both parties ----
    rows = []
    for race in ent.RACES:
        party_a = ent.CANDIDATES[race['candidate_a']]['party']
        party_b = ent.CANDIDATES[race['candidate_b']]['party']
        actual_party = ent.CANDIDATES[race['winner']]['party']

        row = {
            'constituency': race['constituency'],
            'candidate_a': race['candidate_a'], 'party_a': party_a,
            'candidate_b': race['candidate_b'], 'party_b': party_b,
            'actual_party': actual_party,
        }

        scopes = {'national': national_scores,
                  'race_local': local_scores[race['constituency']]}
        for scope, scores in scopes.items():
            sa = scores.get(party_a, {'n': 0, 'mean_polarity': None, 'ci_low': None, 'ci_high': None})
            sb = scores.get(party_b, {'n': 0, 'mean_polarity': None, 'ci_low': None, 'ci_high': None})
            scoreable = sa['mean_polarity'] is not None and sb['mean_polarity'] is not None
            row[f'{scope}_score_a'] = sa
            row[f'{scope}_score_b'] = sb
            row[f'{scope}_scoreable'] = scoreable
            if scoreable:
                predicted = party_a if sa['mean_polarity'] >= sb['mean_polarity'] else party_b
                row[f'{scope}_predicted'] = predicted
                row[f'{scope}_correct'] = (predicted == actual_party)
            else:
                row[f'{scope}_predicted'] = None
                row[f'{scope}_correct'] = None
        rows.append(row)

    # ---- baselines ----
    # (1) party mention VOLUME only, no sentiment -- reuses the same 'n'
    #     already computed above for each scope.
    volume_totals = {}
    for scope in ('national', 'race_local'):
        correct = scoreable = 0
        for r in rows:
            sa, sb = r[f'{scope}_score_a'], r[f'{scope}_score_b']
            if sa['n'] == 0 and sb['n'] == 0:
                continue
            scoreable += 1
            predicted = r['party_a'] if sa['n'] >= sb['n'] else r['party_b']
            correct += int(predicted == r['actual_party'])
        volume_totals[scope] = {'correct': correct, 'scoreable': scoreable}

    # (2) always predict RSP -- won 3 of 5 races
    always_rsp_correct = sum(1 for r in ent.RACES
                             if ent.CANDIDATES[r['winner']]['party'] == 'rsp')
    always_rsp = {'correct': always_rsp_correct, 'total': len(ent.RACES)}

    # ---- sentiment totals ----
    sentiment_totals = {}
    for scope in ('national', 'race_local'):
        correct = sum(1 for r in rows if r[f'{scope}_correct'])
        scoreable = sum(1 for r in rows if r[f'{scope}_scoreable'])
        sentiment_totals[scope] = {'correct': correct, 'scoreable': scoreable}

    totals = {
        'sentiment_national': sentiment_totals['national'],
        'sentiment_race_local': sentiment_totals['race_local'],
        'volume_baseline_national': volume_totals['national'],
        'volume_baseline_race_local': volume_totals['race_local'],
        'always_predict_rsp': always_rsp,
    }

    # ---- console output ----
    print_party_scores('National party sentiment (whole pool)', national_scores)
    for race in ent.RACES:
        print_party_scores(f"Race-local party sentiment -- {race['constituency']}",
                           local_scores[race['constituency']])
    print_race_table(rows)
    print_totals(totals)

    out = {
        'parties': parties,
        'national_party_scores': national_scores,
        'race_local_party_scores': local_scores,
        'races': rows,
        'totals': totals,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved -> {OUT_PATH.name}')


if __name__ == '__main__':
    main()
