"""
merge_labels.py
---------------
Combines label files from two or more annotators, and reports
inter-annotator agreement on any comments both people labelled.

    python merge_labels.py labels_ashutosh.csv labels_teammate.csv \
        --out labelled.csv

Why separate files per person:
    A shared labelled.csv creates a git merge conflict every single time
    two people label in parallel, and CSV conflicts are painful to resolve
    by hand. One file per annotator never conflicts. This script merges
    them on demand.

Why the agreement number matters:
    Cohen's kappa between TWO people is much stronger evidence than
    one person agreeing with themselves. If you and your teammate both
    label the same 100 comments and score kappa > 0.7, that is a real
    reliability figure for your report -- and very few undergraduate
    projects have one.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def load(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f'Not found: {path}')
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'label' not in df.columns:
        sys.exit(f'{path} must have columns: text, label')
    if 'channel' not in df.columns:
        df['channel'] = 'comment'
    df['annotator'] = name
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    return df[['text', 'label', 'channel', 'annotator']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', help='one label CSV per annotator')
    ap.add_argument('--out', default='labelled.csv')
    ap.add_argument('--prefer', default=None,
                    help='annotator name whose label wins on disagreement '
                         '(default: drop disagreed items)')
    args = ap.parse_args()

    frames = []
    for f in args.files:
        p = Path(f)
        name = p.stem.replace('labels_', '').replace('labelled_', '')
        d = load(p, name)
        print(f'  {name:15s} {len(d):5d} labels   {d["label"].value_counts().to_dict()}')
        frames.append(d)

    all_df = pd.concat(frames, ignore_index=True)

    # --- find overlaps ---
    counts = all_df.groupby('text')['annotator'].nunique()
    overlap_texts = counts[counts > 1].index
    print(f'\n  total unique comments : {all_df["text"].nunique()}')
    print(f'  labelled by 2+ people : {len(overlap_texts)}')

    disagreed = pd.DataFrame()
    if len(overlap_texts):
        ov = all_df[all_df['text'].isin(overlap_texts)]
        pivot = ov.pivot_table(index='text', columns='annotator',
                               values='label', aggfunc='first')
        pivot = pivot.dropna()

        if pivot.shape[1] >= 2:
            a, b = pivot.columns[0], pivot.columns[1]
            agree = (pivot[a] == pivot[b]).mean()

            from sklearn.metrics import cohen_kappa_score
            kappa = cohen_kappa_score(pivot[a], pivot[b])

            print(f'\n{"=" * 52}')
            print(f'  INTER-ANNOTATOR AGREEMENT  ({a} vs {b})')
            print(f'  n compared      : {len(pivot)}')
            print(f'  raw agreement   : {agree:.3f}')
            print(f"  Cohen's kappa   : {kappa:.3f}")
            if kappa >= 0.80:
                print('  -> excellent. Your guideline is unambiguous.')
            elif kappa >= 0.65:
                print('  -> acceptable. Tighten the rules for your weakest class.')
            else:
                print('  -> too low. Sit down together, agree on the rules,')
                print('     and relabel. Training on inconsistent labels')
                print('     wastes the effort of both of you.')
            print(f'{"=" * 52}')
            print('\n  PUT THIS NUMBER IN YOUR REPORT. Two-annotator kappa is')
            print('  much stronger evidence than one person self-checking.')

            disagreed = pivot[pivot[a] != pivot[b]]
            if not disagreed.empty:
                print(f'\n  You disagreed on {len(disagreed)}:')
                print(disagreed.groupby([a, b]).size().to_string())
                print('\n  Reviewing these together is the fastest way to')
                print('  find the ambiguity in your guideline.')
                Path('disagreements.csv').write_text(
                    disagreed.to_csv(), encoding='utf-8')
                print('  Saved -> disagreements.csv')

    # --- build the merged file ---
    if args.prefer:
        all_df['_rank'] = (all_df['annotator'] != args.prefer).astype(int)
        merged = (all_df.sort_values('_rank')
                        .drop_duplicates(subset=['text'], keep='first')
                        .drop(columns=['_rank']))
        print(f'\n  Disagreements resolved in favour of: {args.prefer}')
    else:
        if not disagreed.empty:
            all_df = all_df[~all_df['text'].isin(disagreed.index)]
            print(f'\n  Dropped {len(disagreed)} disagreed comments '
                  f'(use --prefer NAME to keep them instead)')
        merged = all_df.drop_duplicates(subset=['text'], keep='first')

    out = merged[['text', 'label', 'channel']]
    out.to_csv(args.out, index=False)

    print(f'\n  Merged {len(out)} labels -> {args.out}')
    print(out['label'].value_counts().to_string())
    if out['channel'].nunique() > 1:
        print()
        print(pd.crosstab(out['channel'], out['label']).to_string())


if __name__ == '__main__':
    main()
