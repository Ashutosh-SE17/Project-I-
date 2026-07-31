"""
label_tool.py
-------------
Terminal annotation tool. This is the one input the supervised model needs
that cannot be automated.

    # 1. build the pool first
    python collect_pool.py

    # 2. first pass -- random, script-balanced
    python label_tool.py --data labelling_pool.csv --out labelled.csv --n 400

    # 3. later passes -- target what the model finds hardest
    python label_tool.py --data labelling_pool.csv --out labelled.csv \
        --n 300 --uncertain

    # 4. measure your own consistency (do this after a few days)
    python label_tool.py --audit --out labelled.csv --n 100

KEYS
    a  negative      s  neutral       d  positive
    x  skip (genuinely undecidable -- not written to file)
    u  undo          q  save and quit

ANNOTATION GUIDELINE
    Label the sentiment OF THE TEXT, not who you think it benefits.
    "KP Oli chor ho" is NEGATIVE even while analysing Balen.
    Attribution is entities.py's job. Mixing the two poisons the dataset.

    negative  attacks, mocks, blames, accuses, insults, sarcastic dismissal
    positive  praises, supports, defends, endorses, agrees
    neutral   factual, a genuine question, off-topic, spam, or too vague

    Question form does not mean neutral -- rhetorical questions are usually
    attacks. Sarcasm is labelled by intent, not by surface wording.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYMAP = {'a': 'negative', 's': 'neutral', 'd': 'positive'}
COLOR = {'negative': '\033[91m', 'neutral': '\033[90m', 'positive': '\033[92m'}
RESET = '\033[0m'


def _load_done(out_path: Path) -> dict:
    """text -> (label, channel)"""
    if not out_path.exists():
        return {}
    prev = pd.read_csv(out_path)
    if 'channel' not in prev.columns:
        prev['channel'] = 'comment'
    return {str(t): (str(l), str(c))
            for t, l, c in zip(prev['text'], prev['label'], prev['channel'])}


def _save(done: dict, out_path: Path) -> pd.DataFrame:
    result = pd.DataFrame({
        'text': list(done.keys()),
        'label': [v[0] for v in done.values()],
        'channel': [v[1] for v in done.values()],
    })
    result.to_csv(out_path, index=False)
    return result


def _prompt(text: str, position: str, script: str, stats: dict) -> str:
    total = sum(stats.values()) or 1
    bar = '  '.join(
        f'{COLOR[k]}{k[:3]} {stats.get(k, 0):4d} ({stats.get(k, 0) / total:4.0%}){RESET}'
        for k in ['negative', 'neutral', 'positive']
    )
    print(f'\n{"-" * 72}')
    print(f'{position}   script={script}   |   {bar}')
    print(f'{"-" * 72}')
    print(text[:500])
    return input('\n  a=neg  s=neu  d=pos  x=skip  u=undo  q=quit  > ').strip().lower()


def run_labelling(args):
    df = pd.read_csv(args.data)
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    df = df.rename(columns={text_col: 'text'})
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 5].drop_duplicates(subset=['text'])

    out_path = Path(args.out)
    done = _load_done(out_path)
    if done:
        print(f'Resuming: {len(done)} already labelled')

    pool = df[~df['text'].isin(done.keys())].copy()
    if pool.empty:
        sys.exit('Nothing left to label. Pool exhausted.')

    import preprocessing as pp
    if 'script' not in pool.columns:
        pool['script'] = pool['text'].apply(pp.script_of)
    if 'channel' not in pool.columns:
        pool['channel'] = 'comment'

    model_path = Path(args.model)
    if args.uncertain and model_path.exists():
        import joblib
        model = joblib.load(model_path)['model']
        proba = model.predict_proba([pp.clean(t) for t in pool['text']])
        # entropy: highest = model is most confused = most informative to label
        pool['priority'] = -(proba * np.log(proba + 1e-12)).sum(axis=1)
        pool = pool.sort_values('priority', ascending=False)
        print('Ordering by model uncertainty (active learning)')
    else:
        if args.uncertain:
            print(f'No model at {model_path} yet -- falling back to random')
        pool = pool.sample(frac=1.0)

    if args.balance_scripts and pool['script'].nunique() > 1:
        per = max(1, args.n // pool['script'].nunique())
        pool = pool.groupby('script', group_keys=False).head(per)
        pool = pool.sample(frac=1.0)

    todo = pool.head(args.n).reset_index(drop=True)

    stats = {}
    for lbl, _ in done.values():
        stats[lbl] = stats.get(lbl, 0) + 1

    print(f'\nLabelling {len(todo)} comments. Ctrl-C is safe -- progress '
          f'is saved on quit.\n')

    history = []
    i = 0
    skipped = 0
    try:
        while i < len(todo):
            row = todo.iloc[i]
            pos = (f'[{i + 1}/{len(todo)}]  total={len(done)}  '
                   f'ch={row["channel"]}')
            choice = _prompt(row['text'], pos, row['script'], stats)

            if choice == 'q':
                break
            if choice == 'x':
                i += 1
                skipped += 1
                continue
            if choice == 'u':
                if history:
                    last = history.pop()
                    entry = done.pop(last, None)
                    if entry:
                        stats[entry[0]] -= 1
                    i -= 1
                    print('  undone')
                continue
            if choice not in KEYMAP:
                print('  use a / s / d / x / u / q')
                continue

            label = KEYMAP[choice]
            done[row['text']] = (label, row['channel'])
            stats[label] = stats.get(label, 0) + 1
            history.append(row['text'])
            i += 1
    except KeyboardInterrupt:
        print('\n  interrupted -- saving')

    result = _save(done, out_path)
    print(f'\nSaved {len(result)} labels -> {out_path}   ({skipped} skipped)')
    print(result['label'].value_counts().to_string())
    if result['channel'].nunique() > 1:
        print()
        print(pd.crosstab(result['channel'], result['label']).to_string())

    counts = result['label'].value_counts()
    print()
    if len(result) < 800:
        print(f'  {len(result)} labels. Target 1500-2000 for a stable model.')
    if len(counts) > 1 and counts.min() / counts.max() < 0.20:
        rare = counts.idxmin()
        print(f'  WARNING: "{rare}" is only {counts.min()} of {len(result)}. '
              f'Go looking for {rare} examples specifically rather than '
              f'labelling more at random -- otherwise the model will never '
              f'learn that class.')


def run_audit(args):
    """
    Re-present already-labelled comments blind and measure agreement with
    your past self. Cohen's kappa above ~0.7 means your guideline is
    consistent enough to train on. Below that, the guideline is too vague
    and more labels will not help.
    """
    out_path = Path(args.out)
    if not out_path.exists():
        sys.exit(f'No labels at {out_path} to audit.')

    prev = pd.read_csv(out_path)
    sample = prev.sample(n=min(args.n, len(prev))).reset_index(drop=True)

    print(f'AUDIT: relabelling {len(sample)} comments you have already done.')
    print('Answer honestly -- do not try to recall your previous choice.\n')

    old, new = [], []
    try:
        for i, row in sample.iterrows():
            print(f'\n{"-" * 72}\n[{i + 1}/{len(sample)}]\n{"-" * 72}')
            print(str(row['text'])[:500])
            choice = input('\n  a=neg  s=neu  d=pos  q=stop  > ').strip().lower()
            if choice == 'q':
                break
            if choice not in KEYMAP:
                continue
            old.append(row['label'])
            new.append(KEYMAP[choice])
    except KeyboardInterrupt:
        pass

    if len(old) < 10:
        sys.exit('\nToo few compared to score.')

    from sklearn.metrics import accuracy_score, cohen_kappa_score
    kappa = cohen_kappa_score(old, new)
    agree = accuracy_score(old, new)

    print(f'\n{"=" * 52}')
    print(f'  n compared        : {len(old)}')
    print(f'  raw agreement     : {agree:.3f}')
    print(f"  Cohen's kappa     : {kappa:.3f}")
    if kappa >= 0.80:
        print('  -> excellent. Your guideline is well defined.')
    elif kappa >= 0.65:
        print('  -> acceptable. Tighten the rules for your weakest class.')
    else:
        print('  -> too low. Rewrite the guideline and relabel; adding more')
        print('     labels at this consistency will not improve the model.')
    print(f'{"=" * 52}')
    print('\n  Put this number in your report. Very few student projects')
    print('  report annotation reliability, and examiners notice.')

    dis = pd.DataFrame({'old': old, 'new': new})
    dis = dis[dis['old'] != dis['new']]
    if not dis.empty:
        print(f'\n  You disagreed with yourself on {len(dis)}:')
        print(dis.groupby(['old', 'new']).size().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='labelling_pool.csv')
    ap.add_argument('--out', default='labelled.csv')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--uncertain', action='store_true',
                    help='prioritise comments the current model is unsure about')
    ap.add_argument('--model', default='models/sentiment_clf.pkl')
    ap.add_argument('--balance-scripts', action='store_true', default=True)
    ap.add_argument('--audit', action='store_true',
                    help='measure agreement with your own past labels')
    args = ap.parse_args()

    if args.audit:
        run_audit(args)
    else:
        run_labelling(args)


if __name__ == '__main__':
    main()
