#!/usr/bin/env python
"""
manage.py
---------
One entry point for the whole project. Run from the project root.

    python manage.py doctor     check the environment is set up correctly
    python manage.py pool       build the labelling pool (add --news for headlines)
    python manage.py label      start labelling
    python manage.py audit      measure your own annotation consistency
    python manage.py train      train the classifier
    python manage.py maps       regenerate the district maps
    python manage.py serve      run the Flask app
    python manage.py analyze --candidate balen    one-off CLI analysis

Any unrecognised extra arguments are forwarded to the underlying script, so
`python manage.py label --n 500 --uncertain` works as expected.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
MODEL = ROOT / 'models' / 'sentiment_clf.pkl'
POOL = ROOT / 'labelling_pool.csv'
LABELS = ROOT / 'labelled.csv'

OK = '\033[92m[ok]\033[0m'
BAD = '\033[91m[!!]\033[0m'
WARN = '\033[93m[..]\033[0m'
if os.name == 'nt' and not os.environ.get('WT_SESSION'):
    OK, BAD, WARN = '[ok]', '[!!]', '[..]'   # legacy consoles lack ANSI


def run(script: str, extra: list) -> int:
    cmd = [PY, str(ROOT / script)] + extra
    print(f'> {" ".join(cmd[1:])}\n')
    return subprocess.call(cmd, cwd=ROOT)


def doctor(_extra) -> int:
    print('Environment check\n' + '-' * 46)
    problems = 0

    print(f'  python      {sys.version.split()[0]}  ({PY})')
    in_venv = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
    print(f'  {OK if in_venv else WARN} virtual environment '
          f'{"active" if in_venv else "NOT active -- see SETUP.md"}')
    if not in_venv:
        problems += 1

    for mod, pkg in [('flask', 'flask'), ('pandas', 'pandas'),
                     ('sklearn', 'scikit-learn'), ('joblib', 'joblib'),
                     ('feedparser', 'feedparser'), ('dotenv', 'python-dotenv'),
                     ('googleapiclient', 'google-api-python-client')]:
        try:
            __import__(mod)
            print(f'  {OK} {pkg}')
        except ImportError:
            print(f'  {BAD} {pkg} missing   ->  pip install {pkg}')
            problems += 1

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get('YOUTUBE_API_KEY')
    if key:
        print(f'  {OK} YOUTUBE_API_KEY set ({key[:6]}...{key[-4:]})')
    else:
        print(f'  {BAD} YOUTUBE_API_KEY not set -- create a .env file')
        problems += 1

    for label, path in [('district geojson', ROOT / 'Nep_district.geojson'),
                        ('labelling pool', POOL),
                        ('labelled data', LABELS),
                        ('trained model', MODEL)]:
        mark = OK if path.exists() else WARN
        detail = ''
        if path.exists() and path.suffix == '.csv':
            try:
                import pandas as pd
                detail = f'  ({len(pd.read_csv(path))} rows)'
            except Exception:
                pass
        print(f'  {mark} {label:18s} {path.name}{detail}')

    print('-' * 46)
    if problems:
        print(f'{problems} problem(s). Fix these before continuing.')
    else:
        print('Ready.')
    return problems


def pool(extra):
    return run('collect_pool.py', extra)


def label(extra):
    args = extra[:]
    if '--data' not in args:
        if not POOL.exists():
            print(f'{BAD} No {POOL.name}. Run:  python manage.py pool')
            return 1
        args += ['--data', str(POOL)]
    if '--out' not in args:
        args += ['--out', str(LABELS)]
    return run('label_tool.py', args)


def audit(extra):
    args = ['--audit'] + extra
    if '--out' not in args:
        args += ['--out', str(LABELS)]
    return run('label_tool.py', args)


def train(extra):
    args = extra[:]
    if '--data' not in args:
        if not LABELS.exists():
            print(f'{BAD} No {LABELS.name}. Run:  python manage.py label')
            return 1
        args += ['--data', str(LABELS)]
    if '--out' not in args:
        args += ['--out', str(MODEL)]
    return run('train_sentiment.py', args)


def maps(extra):
    return run('generate_nepal_map.py', extra)


def serve(extra):
    if not MODEL.exists():
        print(f'{WARN} No trained model at {MODEL}.')
        print('     The app will start but /analyze will fail.')
        print('     Run:  python manage.py train\n')
    return run('app.py', extra)


def analyze(extra):
    ap = argparse.ArgumentParser(prog='manage.py analyze')
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--no-news', action='store_true')
    args = ap.parse_args(extra)

    sys.path.insert(0, str(ROOT))
    from sentiment_model import ElectionAnalyzer

    a = ElectionAnalyzer()
    res = a.analyze_candidate(args.candidate, include_news=not args.no_news)

    if 'error' in res:
        print(f'{BAD} {res["error"]}')
        return 1

    print(f'\n  {res["candidate"]}')
    print(f'  win probability  {res["win_probability"]}%  '
          f'CI {res["confidence_interval"]}')
    print(f'  comments         {res["n_comments"]}')
    print(f'  headlines        {res.get("n_headlines", 0)}')
    print(f'  channel scores   {res.get("channel_scores", {})}')
    print(f'  class counts     {res["counts"]}')
    return 0


COMMANDS = {
    'doctor': doctor, 'pool': pool, 'label': label, 'audit': audit,
    'train': train, 'maps': maps, 'serve': serve, 'analyze': analyze,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        print(__doc__)
        return 0
    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f'Unknown command "{cmd}". Options: {", ".join(COMMANDS)}')
        return 1
    return COMMANDS[cmd](sys.argv[2:]) or 0


if __name__ == '__main__':
    sys.exit(main())
