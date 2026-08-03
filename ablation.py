"""
ablation.py
-----------
Compares four sentiment-scoring configurations on the same 854-comment
binary test set (labelled_binary.csv), for the report's ablation study:

    1. lexicon only    -- the old hand-tuned ElectionAnalyzer.get_sentiment()
    2. ML only          -- train_sentiment.build_model(), cross-validated
    3. ML + entity layer -- polarity x attribution, sign() as the class
    4. majority baseline

    python ablation.py

Design notes (see LABELLING_GUIDE.md / plan discussion for the full
reasoning):

- The lexicon and the entity-attribution layer both need a candidate to
  score against, but labelled_binary.csv has no candidate column. It's
  recovered by joining on exact text against labelling_pool.csv, which
  carries the candidate each comment was originally fetched for. Not every
  row matches (text was cleaned differently upstream), so configs 1 and 3
  run on the matched subset while 2 and 4 run on everything -- reported as
  both a "primary" (per-brief) table and a same-N "matched-only" table.

- A lexicon "Neutral" is scored as a plain error (via sklearn's
  labels=['positive','negative'] restriction) rather than remapped by
  score sign, because get_sentiment() returns ("Neutral", 0) from two
  different code paths and the score is exactly 0 in the more common one --
  there's no sign to map.
"""

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import entities as ent
import preprocessing as pp
from train_sentiment import build_model, mark_channel

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'labelled_binary.csv'
POOL_PATH = BASE_DIR / 'labelling_pool.csv'
OLD_MODEL_PATH = BASE_DIR / 'sentiment_model.py.old'
OUT_PATH = BASE_DIR / 'ablation_results.json'

SCRIPTS = ['deva', 'latin', 'mixed']
CLASS_LABELS = ['negative', 'positive']
N_FOLDS = 5
RANDOM_STATE = 42


def load_old_lexicon():
    """
    Load ElectionAnalyzer.get_sentiment from sentiment_model.py.old without
    triggering the live YouTube client its __init__ builds on a hardcoded
    key -- get_sentiment() never touches self.youtube. build() is
    monkeypatched to a no-op only for module exec + construction; the
    lexicon/scandal/symbol data and get_sentiment itself run unmodified.
    """
    import googleapiclient.discovery as gd

    # spec_from_file_location can't infer a loader from the ".old" suffix,
    # so the source loader is supplied explicitly.
    loader = SourceFileLoader('sentiment_model_old', str(OLD_MODEL_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    old_mod = importlib.util.module_from_spec(spec)
    with patch.object(gd, 'build', lambda *a, **k: None):
        loader.exec_module(old_mod)
        analyzer = old_mod.ElectionAnalyzer()
    return analyzer


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['script'] = df['text'].apply(pp.script_of)
    df['clean'] = df['text'].apply(pp.clean)
    df['marked'] = [mark_channel(c, ch) for c, ch in zip(df['clean'], df['channel'])]

    pool = (pd.read_csv(POOL_PATH)[['text', 'candidate']]
            .drop_duplicates(subset=['text']))
    df = df.merge(pool, on='text', how='left')

    matched = int(df['candidate'].notna().sum())
    print(f'Candidate attribution recovered via labelling_pool.csv join: '
          f'{matched}/{len(df)} rows matched, {len(df) - matched} unmatched.')
    return df


def eval_config(name, y_true, y_pred, script=None, labels=CLASS_LABELS) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    per_class = {
        lbl: {'precision': float(p), 'recall': float(r), 'f1': float(f), 'support': int(s)}
        for lbl, p, r, f, s in zip(labels, prec, rec, f1, support)
    }

    result = {
        'name': name,
        'n': int(len(y_true)),
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'per_class': per_class,
    }

    if script is not None:
        script = np.asarray(script)
        per_script = {}
        for s in SCRIPTS:
            mask = script == s
            if mask.sum() == 0:
                continue
            per_script[s] = {
                'n': int(mask.sum()),
                'accuracy': float(accuracy_score(y_true[mask], y_pred[mask])),
                'macro_f1': float(f1_score(y_true[mask], y_pred[mask], labels=labels,
                                           average='macro', zero_division=0)),
            }
        result['per_script'] = per_script

    return result


def neutral_coverage(pred, script) -> dict:
    """
    Fraction of predictions that are 'neutral' -- i.e. the lexicon never
    fired a rule, not that it made a judgement call. Reported separately
    from accuracy because it's a distinct failure mode (non-engagement,
    not misclassification).
    """
    pred = np.asarray(pred)
    script = np.asarray(script)
    per_script = {}
    for s in SCRIPTS:
        mask = script == s
        if mask.sum() == 0:
            continue
        per_script[s] = float((pred[mask] == 'neutral').mean())
    return {'overall': float((pred == 'neutral').mean()), 'per_script': per_script}


def main():
    print('Loading data...')
    df = load_data()

    y_all = df['label'].values
    script_all = df['script'].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    # ---- config 4: majority baseline (N=854) ----
    majority = pd.Series(y_all).value_counts().idxmax()
    pred_baseline = np.full(len(df), majority)
    results['majority_baseline'] = eval_config(
        'Majority baseline', y_all, pred_baseline, script_all)
    print(f"Majority baseline: predicting '{majority}' for every row.")

    # ---- config 2: ML only (N=854) ----
    print(f'Running {N_FOLDS}-fold CV for ML-only config...')
    pred_ml = cross_val_predict(build_model(), df['marked'].values, y_all,
                                cv=skf, n_jobs=-1)
    results['ml_only'] = eval_config('ML only', y_all, pred_ml, script_all)

    # ---- config 3: ML + entity layer, needs out-of-fold probabilities on
    #      the full set (fold-consistent with config 2), then restricted to
    #      the candidate-matched rows since attribution() needs a candidate
    print(f'Running {N_FOLDS}-fold CV (predict_proba) for ML+entity config...')
    proba_all = cross_val_predict(build_model(), df['marked'].values, y_all,
                                  cv=skf, method='predict_proba', n_jobs=-1)
    classes = sorted(set(y_all))          # sklearn always sorts classes_ this way
    ipos, ineg = classes.index('positive'), classes.index('negative')
    df['polarity'] = proba_all[:, ipos] - proba_all[:, ineg]

    matched = df[df['candidate'].notna()].copy()
    matched['attribution'] = [ent.attribution(t, c)['attribution']
                              for t, c in zip(matched['text'], matched['candidate'])]
    combined = matched['polarity'].values * matched['attribution'].values
    # tie-break: attribution == 0 (no candidate/rival mentioned at all) ->
    # default to the majority class, matching config 4's convention
    pred_entity = np.where(combined > 0, 'positive',
                   np.where(combined < 0, 'negative', majority))
    results['ml_entity'] = eval_config(
        'ML + entity layer', matched['label'].values, pred_entity,
        matched['script'].values)

    # ---- config 1: lexicon only (N=806, matched rows -- it needs a candidate) ----
    print('Loading old lexicon (network calls stubbed)...')
    analyzer = load_old_lexicon()
    print(f'Scoring {len(matched)} matched rows with the old lexicon...')
    lex_pred = np.array([
        analyzer.get_sentiment(text, cand)[0].lower()
        for text, cand in zip(matched['text'], matched['candidate'])
    ])
    results['lexicon_only'] = eval_config(
        'Lexicon only', matched['label'].values, lex_pred, matched['script'].values)
    results['lexicon_only']['neutral_coverage'] = neutral_coverage(
        lex_pred, matched['script'].values)

    # ---- matched-only head-to-head (N=806): configs 2 & 4 re-scored on the
    #      same subset configs 1 & 3 were always restricted to ----
    matched_idx = df['candidate'].notna().values
    results_matched_only = {
        'majority_baseline': eval_config(
            'Majority baseline (matched-only)', y_all[matched_idx],
            pred_baseline[matched_idx], script_all[matched_idx]),
        'ml_only': eval_config(
            'ML only (matched-only)', y_all[matched_idx],
            pred_ml[matched_idx], script_all[matched_idx]),
        'ml_entity': results['ml_entity'],
        'lexicon_only': results['lexicon_only'],
    }

    # ---- print summary ----
    def row(r):
        pc = r['per_class']
        return (f"{r['name']:34s} n={r['n']:4d}  acc={r['accuracy']:.3f}  "
                f"macroF1={r['macro_f1']:.3f}  "
                f"F1(pos)={pc['positive']['f1']:.3f}  F1(neg)={pc['negative']['f1']:.3f}")

    order = ['majority_baseline', 'ml_only', 'ml_entity', 'lexicon_only']

    print('\n' + '=' * 88)
    print('PRIMARY RESULTS (per brief -- N differs by config: 854 / 854 / 806 / 806)')
    print('=' * 88)
    for key in order:
        print(row(results[key]))
    cov = results['lexicon_only']['neutral_coverage']
    print(f"\nLexicon 'Neutral' (non-firing) rate: {cov['overall']:.1%} overall")
    for s, v in cov['per_script'].items():
        print(f"    {s:6s} {v:.1%}")

    print('\n' + '=' * 88)
    print(f'MATCHED-ONLY HEAD-TO-HEAD (N={int(matched_idx.sum())}, identical rows every config)')
    print('=' * 88)
    for key in order:
        print(row(results_matched_only[key]))

    # ---- save ----
    out = {
        'primary': results,
        'matched_only_head_to_head': results_matched_only,
        'metadata': {
            'n_total': int(len(df)),
            'n_candidate_matched': int(matched_idx.sum()),
            'n_candidate_unmatched': int(len(df) - matched_idx.sum()),
            'folds': N_FOLDS,
            'random_state': RANDOM_STATE,
            'neutral_handling': "lexicon 'Neutral' predictions are scored as "
                                 "errors via sklearn's labels=['positive','negative'] "
                                 "restriction, not remapped by score sign",
            'config3_tie_break': "when polarity * attribution == 0 (no candidate "
                                  "or rival mentioned), predicted class defaults "
                                  f"to the majority class ('{majority}')",
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSaved -> {OUT_PATH}')


if __name__ == '__main__':
    main()
