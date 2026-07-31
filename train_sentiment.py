"""
train_sentiment.py
------------------
Trains the supervised sentiment classifier that replaces the lexicon.

    python train_sentiment.py --data labelled_comments.csv

Expected CSV columns:
    text   -- the raw comment
    label  -- one of: positive / neutral / negative   (or 1 / 0 / -1)

What this reports that the reference notebook did not:
    * the majority-class baseline (so accuracy means something)
    * macro-F1 and PER-CLASS recall (accuracy alone hides minority-class
      failure -- the reference model scored 49% recall on negatives while
      reporting 77% accuracy)
    * stratified k-fold cross-validation, not a single random split
    * a script-stratified breakdown (Devanagari vs Latin vs mixed), which
      is the specific failure mode this project needs to prove it fixed
    * a calibration check, because the downstream sigmoid consumes
      predict_proba and miscalibrated probabilities corrupt it
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

import preprocessing as pp

LABELS = ['negative', 'neutral', 'positive']
LABEL_TO_INT = {'negative': -1, 'neutral': 0, 'positive': 1}
ALIASES = {
    'positive': 'positive', 'pos': 'positive', '1': 'positive', '1.0': 'positive',
    'neutral': 'neutral', 'neu': 'neutral', '0': 'neutral', '0.0': 'neutral',
    'negative': 'negative', 'neg': 'negative', '-1': 'negative', '-1.0': 'negative',
}


CHANNELS = ['comment', 'news']


def mark_channel(clean_text: str, channel: str) -> str:
    """
    Prefix a channel token so one pooled model can learn channel-specific
    behaviour without needing a separate model per channel.

    The word vectoriser runs ngram_range=(1,2), so this token forms bigrams
    with the first real token of the document. That gives the model both a
    per-channel prior (the unigram) and limited per-channel interaction
    (the bigrams) -- which is most of the benefit of two separate models,
    at a fraction of the labelling cost.
    """
    ch = channel if channel in CHANNELS else 'comment'
    return f'__ch_{ch}__ {clean_text}'


def build_model() -> Pipeline:
    """
    Word features + character features, unioned.

    The char_wb block is the important one: it operates on raw character
    n-grams, so it works on Devanagari and Latin script through the same
    vectoriser, and it absorbs romanisation variance (ramro / raamro /
    ramrro) without any fuzzy-matching step.

    LogisticRegression rather than Naive Bayes because this pipeline's
    output feeds a sigmoid. NB's independence assumption produces
    probabilities crushed against 0 and 1, which would re-introduce the
    saturation problem the aggregation layer is trying to fix.
    """
    features = FeatureUnion([
        ('word', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=30000,
        )),
        ('char', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            min_df=3,
            sublinear_tf=True,
            max_features=50000,
        )),
    ])

    clf = LogisticRegression(
        C=4.0,
        max_iter=2000,
        class_weight='balanced',   # your corpus will be negative-heavy
        solver='lbfgs',
    )

    return Pipeline([('features', features), ('clf', clf)])


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        sys.exit(f"ERROR: {path} must have columns 'text' and 'label'. "
                 f"Found: {list(df.columns)}")

    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(str).str.strip().str.lower().map(ALIASES)
    unknown = df['label'].isna().sum()
    if unknown:
        print(f"  dropping {unknown} rows with unrecognised labels")
        df = df.dropna(subset=['label'])

    df['text'] = df['text'].astype(str)
    if 'channel' not in df.columns:
        df['channel'] = 'comment'
    df['channel'] = (df['channel'].astype(str).str.strip().str.lower()
                       .where(lambda c: c.isin(CHANNELS), 'comment'))
    df['script'] = df['text'].apply(pp.script_of)
    df['clean'] = df['text'].apply(pp.clean)
    df['marked'] = [mark_channel(c, ch)
                    for c, ch in zip(df['clean'], df['channel'])]

    before = len(df)
    df = df[df['clean'].str.strip().astype(bool)]
    df = df.drop_duplicates(subset=['clean', 'label'])
    print(f"  {before} rows -> {len(df)} after cleaning and dedup")
    print(f"  channels: {df['channel'].value_counts().to_dict()}")
    return df.reset_index(drop=True)


def evaluate(df: pd.DataFrame, folds: int = 5) -> dict:
    X, y = df['marked'].values, df['label'].values

    counts = pd.Series(y).value_counts()
    baseline = counts.max() / len(y)
    print(f"\n  class distribution: {counts.to_dict()}")
    print(f"  MAJORITY-CLASS BASELINE: {baseline:.3f}  <- beat this or the model is worthless")

    min_class = counts.min()
    folds = max(2, min(folds, int(min_class)))

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    pred = cross_val_predict(build_model(), X, y, cv=cv, n_jobs=-1)

    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average='macro')

    print(f"\n  {folds}-fold CV accuracy : {acc:.3f}   (baseline {baseline:.3f}, "
          f"lift {acc - baseline:+.3f})")
    print(f"  {folds}-fold CV macro-F1 : {macro_f1:.3f}")
    print("\n" + classification_report(y, pred, digits=3, zero_division=0))

    print("  confusion matrix (rows = true, cols = predicted)")
    labels_present = [l for l in LABELS if l in set(y)]
    cm = confusion_matrix(y, pred, labels=labels_present)
    print("           " + "".join(f"{l[:8]:>10s}" for l in labels_present))
    for name, row in zip(labels_present, cm):
        print(f"  {name:>8s} " + "".join(f"{v:>10d}" for v in row))

    # --- the breakdowns that matter for this project ---
    tmp = df.copy()
    tmp['pred'] = pred

    print("\n  PER-SCRIPT ACCURACY (the Devanagari gap this project exists to close)")
    for script, grp in tmp.groupby('script'):
        s_acc = accuracy_score(grp['label'], grp['pred'])
        s_f1 = f1_score(grp['label'], grp['pred'], average='macro', zero_division=0)
        print(f"    {script:6s}  n={len(grp):5d}   acc={s_acc:.3f}   macro-F1={s_f1:.3f}")

    per_channel = {}
    if tmp['channel'].nunique() > 1:
        print("\n  PER-CHANNEL ACCURACY (does one model really serve both?)")
        for ch, grp in tmp.groupby('channel'):
            c_acc = accuracy_score(grp['label'], grp['pred'])
            c_f1 = f1_score(grp['label'], grp['pred'], average='macro',
                            zero_division=0)
            c_base = grp['label'].value_counts().max() / len(grp)
            per_channel[ch] = {'n': int(len(grp)), 'accuracy': float(c_acc),
                               'macro_f1': float(c_f1), 'baseline': float(c_base)}
            print(f"    {ch:8s}  n={len(grp):5d}   acc={c_acc:.3f}   "
                  f"macro-F1={c_f1:.3f}   (baseline {c_base:.3f}, "
                  f"lift {c_acc - c_base:+.3f})")
            print(f"              class mix: {grp['label'].value_counts().to_dict()}")

        f1s = [v['macro_f1'] for v in per_channel.values()]
        gap = max(f1s) - min(f1s)
        print(f"\n    channel macro-F1 gap: {gap:.3f}")
        if gap > 0.15:
            print("    -> LARGE. The pooled model is not serving both channels.")
            print("       If you have 500+ labels in the weaker channel, retrain")
            print("       with --separate. If not, label more of it first --")
            print("       splitting on thin data will make things worse.")
        else:
            print("    -> acceptable. One pooled model is serving both channels.")

    return {'accuracy': float(acc), 'macro_f1': float(macro_f1),
            'baseline': float(baseline), 'folds': folds, 'n': int(len(df)),
            'per_channel': per_channel}


def check_calibration(df: pd.DataFrame) -> None:
    """
    The downstream sigmoid consumes P(pos) - P(neg). If those probabilities
    are overconfident, win_probability saturates regardless of how good the
    aggregation maths is. This prints the reliability curve.
    """
    X, y = df['marked'].values, df['label'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
    model = build_model().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    classes = list(model.named_steps['clf'].classes_)

    conf = proba.max(axis=1)
    pred = model.predict(Xte)
    correct = (pred == yte).astype(int)

    print("\n  CALIBRATION (predicted confidence vs actual accuracy)")
    bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in bins:
        m = (conf >= lo) & (conf < hi)
        if m.sum() < 5:
            continue
        print(f"    conf {lo:.1f}-{hi:.1f}   n={m.sum():4d}   "
              f"mean_conf={conf[m].mean():.3f}   actual_acc={correct[m].mean():.3f}")

    if 'positive' in classes and 'negative' in classes:
        pol = proba[:, classes.index('positive')] - proba[:, classes.index('negative')]
        print(f"\n  polarity spread: min={pol.min():+.3f} max={pol.max():+.3f} "
              f"mean={pol.mean():+.3f} sd={pol.std():.3f}")
        extreme = float(np.mean(np.abs(pol) > 0.95))
        print(f"  fraction with |polarity| > 0.95: {extreme:.3f}"
              + ("   <- WARNING: overconfident, sigmoid will saturate"
                 if extreme > 0.35 else "   (healthy)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='labelled CSV: text,label')
    ap.add_argument('--out', default='models/sentiment_clf.pkl')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--separate', action='store_true',
                    help='train one model PER CHANNEL instead of a pooled '
                         'model. Only use this if per-channel reporting shows '
                         'a large gap AND you have 500+ labels per channel.')
    ap.add_argument('--calibrate', action='store_true',
                    help='wrap in CalibratedClassifierCV (use if the '
                         'calibration report shows overconfidence)')
    args = ap.parse_args()

    print(f"Loading {args.data}")
    df = load(args.data)

    if len(df) < 100:
        print(f"\n  WARNING: only {len(df)} usable rows. Metrics below are noise. "
              f"Aim for 1500+.")

    stats = evaluate(df, args.folds)
    check_calibration(df)

    if args.separate:
        print("\n  Training SEPARATE models per channel...")
        import joblib
        bundles = {}
        for ch, grp in df.groupby('channel'):
            if len(grp) < 200:
                print(f"    {ch}: only {len(grp)} rows -- too thin, skipping. "
                      f"Label more before splitting.")
                continue
            m = build_model().fit(grp['marked'].values, grp['label'].values)
            bundles[ch] = m
            print(f"    {ch}: trained on {len(grp)} rows")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'per_channel_models': bundles, 'labels': LABELS,
                     'stats': stats, 'mode': 'separate'}, out)
        print(f"  Saved -> {out}")
        return

    print("\n  Fitting final model on all data...")
    model = build_model()
    if args.calibrate:
        base = build_model()
        model = Pipeline([
            ('features', base.named_steps['features']),
            ('clf', CalibratedClassifierCV(base.named_steps['clf'],
                                           method='isotonic', cv=3)),
        ])
    model.fit(df['marked'].values, df['label'].values)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump({'model': model, 'labels': LABELS, 'stats': stats,
                 'mode': 'pooled'}, out)
    print(f"  Saved -> {out}")

    meta = out.with_suffix('.json')
    meta.write_text(json.dumps(stats, indent=2))
    print(f"  Saved -> {meta}   (cite these numbers in your report)")


if __name__ == '__main__':
    main()
