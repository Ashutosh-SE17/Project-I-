# Handoff — for the teammate taking over training

Everything you need. Should take about 20 minutes to get running.

You do **not** need a YouTube API key. That's only for fetching new data.
Training works entirely offline from a CSV.

---

## Part 1 — One-time setup

### 1. Get the code

Open a terminal (or VS Code terminal) and run:

```powershell
git clone https://github.com/Ashutosh-SE17/Project-I-.git
cd "Project-I-"
git checkout ml-migration
```

That last line matters — the new code lives on the `ml-migration` branch,
not `main`.

### 2. Make a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:** `python3 -m venv .venv` then `source .venv/bin/activate`

If Windows says *"running scripts is disabled"*, run this once and
answer `Y`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

You'll know it worked when your prompt starts with `(.venv)`.

### 3. Install

```powershell
pip install -r requirements.txt
```

### 4. Create a `.env` file

```powershell
Copy-Item .env.example .env
```

Leave the key as the placeholder. You don't need a real one for training.
`doctor` will flag it — that's fine, ignore that one line.

### 5. Check

```powershell
python manage.py doctor
```

Everything should be `[ok]` except the API key line and the "not yet"
items.

---

## Part 2 — Training

### Get the latest labels

Before every session:

```powershell
git pull
```

### Train

```powershell
python manage.py train
```

Takes 30 seconds to a few minutes.

---

## Part 3 — Reading the output

This is the actual skill. Four numbers matter.

### 1. Accuracy vs baseline

```
MAJORITY-CLASS BASELINE: 0.802
5-fold CV accuracy : 0.877   (baseline 0.802, lift +0.075)
```

**Ignore accuracy on its own.** If 80% of comments are neutral, a model
that answers "neutral" every time scores 80% and is useless. Only the
**lift** matters. Under +0.05 means the model has barely learned anything.

### 2. Macro-F1

```
5-fold CV macro-F1 : 0.769
```

Averages performance across all three classes equally, so a rare class
failing can't hide behind a common class succeeding. **This is the main
number.** Report it.

### 3. Per-class recall

```
              precision    recall  f1-score
    negative      0.663     0.656     0.659
     neutral      0.921     0.934     0.928
     positive     0.900     0.600     0.720
```

Look at the **recall** column. `positive 0.600` means the model misses 40%
of positive comments. If any class is under ~0.5, that class needs more
labelled examples — not more tuning.

### 4. Per-script macro-F1 — the one that matters most here

```
PER-SCRIPT ACCURACY
  deva    n=  312   acc=0.990   macro-F1=0.498
  latin   n=  685   acc=0.819   macro-F1=0.760
```

**This is the whole point of the project.** The old system was effectively
blind to Devanagari. High accuracy with low macro-F1 on `deva` (like
above) means the model learned "Devanagari → always neutral" — the
original bug, reproduced. Accuracy hides it. Macro-F1 exposes it.

**If you see that pattern, say so immediately.** It means more Devanagari
labels are needed, not more training.

### Also watch

```
fraction with |polarity| > 0.95: 0.034   (healthy)
```

Above ~0.35 and it says WARNING. That means the model is overconfident and
the downstream win-probability will saturate. Fix by adding `--calibrate`:

```powershell
python manage.py train --calibrate
```

---

## Part 4 — Working together without breaking things

### Never both edit the same file at once

Talk before either of you edits a shared file.

### If you both label, use separate files

**Do not both write to `labelled.csv`.** You'll get a git conflict every
time and CSV conflicts are horrible to fix.

Instead:

```powershell
python manage.py label --out labels_YOURNAME.csv --n 300
```

Then merge when needed:

```powershell
python merge_labels.py labels_ashutosh.csv labels_YOURNAME.csv --out labelled.csv
```

This also prints **inter-annotator agreement** — Cohen's kappa between the
two of you. Deliberately label ~100 of the same comments so you get this
number. It's strong evidence for the report and almost no student project
has one.

If kappa is below 0.65, stop and agree on the rules before labelling more.
`disagreements.csv` shows exactly where you diverged.

### Use branches

```powershell
git checkout -b training-experiments
# ...work...
git add .
git commit -m "what you did"
git push --set-upstream origin training-experiments
```

Keeps your experiments away from working code. If it goes wrong:

```powershell
git checkout ml-migration
git branch -D training-experiments
```

### Every session

**Start:** `git pull`
**End:** `git add .` → `git commit -m "..."` → `git push`

---

## Docs in the repo

| File | Read it for |
|---|---|
| `CLAUDE.md` | Quick project overview |
| `MIGRATION.md` | Why the ML approach replaced the lexicon |
| `LABELLING_GUIDE.md` | Annotation rules — read before labelling |
| `NEWS_CHANNEL.md` | How headlines are blended in |
| `GIT_CHEATSHEET.md` | Git in plain language |
| `SETUP.md` | Fuller setup detail |

---

## If something breaks

```powershell
python manage.py doctor
```

It usually names the problem. If not, paste the full error into Claude
Code, or message Ashutosh.

**Most common issue:** no `(.venv)` at the start of your prompt. Run
`.venv\Scripts\Activate.ps1` again.
