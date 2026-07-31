# Setup — VS Code on Windows

macOS/Linux differences are noted inline. Should take about 10 minutes.

---

## Step 0 — Back up first

You are replacing `sentiment_model.py` and `requirements.txt`. If your
project isn't already a git repo, make it one **before** copying anything:

```powershell
cd C:\path\to\your\project
git init
git add .
git commit -m "Backup before ML migration"
```

If you already use git, just commit your current state. Either way, you
want a way back.

---

## Step 1 — Where the files go

Everything goes in your **project root**, next to `app.py`:

```
your-project/
├── .vscode/                 ← NEW (tasks, settings, debug configs)
│   ├── tasks.json
│   ├── settings.json
│   ├── launch.json
│   └── extensions.json
├── .env                     ← NEW (you create this from .env.example)
├── .env.example             ← NEW
├── .gitignore               ← NEW
├── manage.py                ← NEW  ← your main entry point
├── preprocessing.py         ← NEW
├── entities.py              ← NEW
├── news_source.py           ← NEW
├── train_sentiment.py       ← NEW
├── label_tool.py            ← NEW
├── collect_pool.py          ← NEW
├── sentiment_model.py       ← REPLACES yours
├── requirements.txt         ← REPLACES yours
│
├── app.py                   ← unchanged
├── generate_nepal_map.py    ← unchanged
├── Nep_district.geojson     ← unchanged
├── election_data.csv        ← unchanged
├── cache.json               ← unchanged
├── templates/
│   └── index.html           ← unchanged
├── static/
│   ├── css/style.css        ← unchanged
│   ├── script.js            ← unchanged
│   └── nepal_map_*.html     ← unchanged
└── models/                  ← created automatically by training
```

**Flat, not in a subfolder.** The modules import each other by name
(`import entities as ent`), so they must sit beside one another.

To copy: download the files, then drag them into the VS Code explorer
panel at the root level. Confirm the overwrite when prompted for
`sentiment_model.py` and `requirements.txt`.

---

## Step 2 — Virtual environment

In VS Code, open the terminal with ``Ctrl+` `` and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:** `python3 -m venv .venv` then `source .venv/bin/activate`

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

You'll know it worked when your prompt shows `(.venv)`.

Then install:

```powershell
pip install -r requirements.txt
```

---

## Step 3 — Point VS Code at the venv

`Ctrl+Shift+P` → **Python: Select Interpreter** → choose the one under
`.\.venv\Scripts\python.exe` (it usually says "Recommended").

This matters. If VS Code uses your global Python, nothing you installed
in step 2 will be visible and every import will fail.

The bundled `settings.json` already points at `.venv/Scripts/python.exe`.
On macOS/Linux change that line to `.venv/bin/python`.

---

## Step 4 — Your API key

Your old key is in your git history and was shared in a chat. **Revoke it**
at <https://console.cloud.google.com/apis/credentials>, then create a new
one.

```powershell
Copy-Item .env.example .env
```

Open `.env` and paste the new key:

```
YOUTUBE_API_KEY=AIza...your_new_key
```

`.env` is gitignored. It will not be committed. `sentiment_model.py`
loads it automatically via `python-dotenv`.

---

## Step 5 — Verify

```powershell
python manage.py doctor
```

You want to see:

```
  [ok] virtual environment active
  [ok] flask
  [ok] pandas
  [ok] scikit-learn
  [ok] joblib
  [ok] feedparser
  [ok] python-dotenv
  [ok] google-api-python-client
  [ok] YOUTUBE_API_KEY set (AIzaSy...abcd)
  [..] labelled data      labelled.csv
  [..] trained model      sentiment_clf.pkl
Ready.
```

The two `[..]` lines are expected — you haven't labelled or trained yet.
Anything marked `[!!]` needs fixing before you continue.

---

## Step 6 — Run it

Every command goes through `manage.py`:

```powershell
python manage.py doctor      # check setup
python manage.py pool        # build the labelling pool
python manage.py pool --news # ...including headlines
python manage.py label       # start labelling
python manage.py audit       # check your annotation consistency
python manage.py train       # train the classifier
python manage.py serve       # run the Flask app
python manage.py maps        # regenerate district maps

python manage.py analyze --candidate balen    # one-off, no browser
```

Extra flags pass straight through:

```powershell
python manage.py label --n 500 --uncertain
python manage.py train --separate
```

---

## Step 7 — Click-to-run (optional but nice)

`Ctrl+Shift+P` → **Tasks: Run Task** gives you a menu:

```
1. Doctor (check setup)
2. Build labelling pool
2b. Build pool + headlines
3. Label comments            ← prompts for batch size
3b. Label (hardest first)
4. Train model               ← Ctrl+Shift+B
5. Run web app
Audit my labels
Regenerate maps
Analyze one candidate        ← prompts for candidate
```

`Ctrl+Shift+B` runs training directly.

**Debugging:** press `F5` and pick a config — Flask app, training,
analysis, or whatever file is open. Breakpoints work normally, including
inside `sentiment_model.py`, which is the fastest way to understand the
scoring composition.

---

## Your first full run

```powershell
python manage.py doctor
python manage.py pool --news
python manage.py label --n 400      # ~30 minutes
python manage.py train
python manage.py serve              # http://127.0.0.1:5001
```

Then iterate:

```powershell
python manage.py label --n 300 --uncertain
python manage.py train
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'sklearn'`**
The venv isn't active or VS Code is on the wrong interpreter. Redo steps
2–3. Check with `python manage.py doctor`.

**`ImportError: cannot import name 'mark_channel'`**
`train_sentiment.py` didn't get copied, or it landed in a subfolder. All
modules must be flat in the project root.

**Terminal opens without `(.venv)`**
Close all terminals and open a new one. VS Code only auto-activates in
terminals opened *after* the interpreter is selected.

**`quotaExceeded` from YouTube**
You've used your 10,000 daily units. Each analysis costs ~110. It resets
at midnight Pacific. The 6-hour cache means repeat analyses of the same
candidate are free.

**Devanagari renders as boxes in the terminal**
Cosmetic, and it will make labelling painful. Use Windows Terminal rather
than the legacy console, and set the VS Code terminal font to one with
Devanagari coverage:
`"terminal.integrated.fontFamily": "Nirmala UI, Consolas"`

**`.venv` shows up in git status**
It's in `.gitignore`. If it was already tracked:
`git rm -r --cached .venv`

---

## One thing to commit deliberately

`.gitignore` excludes `models/` and the generated map HTML — both are
regenerable and large.

It does **not** exclude `labelled.csv`. That file is hours of your own
judgement and is the single most valuable artefact in the project. Commit
it every session:

```powershell
git add labelled.csv
git commit -m "labelling session: +300"
```

If you lose it, you relabel from zero.
