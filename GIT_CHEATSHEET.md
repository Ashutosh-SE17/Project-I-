# Git cheat sheet (plain language)

Keep this open. You only need about 8 commands.

---

## The idea

Think of a video game.

| Game | Git |
|---|---|
| Press "Save" | `git commit` |
| Upload save to the cloud | `git push` |
| Your save history | `git log` |
| Load an old save | `git checkout` |
| A separate save slot so you don't ruin your good save | `git branch` |

Git saves are **permanent**. Once you commit, that version exists
forever and you can always go back to it. That's the whole point.

---

## The save button (do this constantly)

Three commands, always in this order:

```powershell
git add .
git commit -m "what I just did"
git push
```

**What each one does:**

- `git add .` — "Save everything I changed." The `.` means *everything in
  this folder*.
- `git commit -m "..."` — Actually presses save. The text in quotes is
  your label, so future-you knows what this save was.
- `git push` — Uploads it to GitHub. Now it's safe even if your laptop
  dies.

**Do this every time something works.** Not once a day. Every time. It
costs 5 seconds.

Good labels:
```powershell
git commit -m "labelled 300 more comments"
git commit -m "trained model, macro-F1 0.74"
git commit -m "added news feed"
```

Useless labels: `"update"`, `"fix"`, `"asdf"`. You'll thank yourself later.

---

## Seeing your saves

```powershell
git log --oneline
```

Output looks like:

```
a3f9c21 labelled 300 more comments
7b2e410 trained model, macro-F1 0.74
c81d055 added news feed
```

Those short codes (`a3f9c21`) are save IDs. You use them to go back.

```powershell
git status
```

Shows what you've changed but not yet saved. Run this when confused.

---

## Branches: separate save slots

A branch is a **parallel copy of your project.** You can experiment in one
without touching the other.

Your main branch is called `main`. Think of it as "the version that works."

**Never experiment directly on `main`.** Make a branch instead.

### Make a new branch

```powershell
git checkout -b trying-news-feature
```

Read it as: "make a new save slot called `trying-news-feature` and switch
to it."

Now do whatever you want. Break things. `main` is untouched.

### Switch between branches

```powershell
git checkout main                    # back to the safe version
git checkout trying-news-feature     # back to the experiment
```

**Important:** commit before switching, or your changes come with you and
get confusing.

### See your branches

```powershell
git branch
```

The one with `*` is where you are.

### It worked — keep it

```powershell
git checkout main
git merge trying-news-feature
git push
```

"Go to main, bring in the changes from my experiment, upload."

### It failed — throw it away

```powershell
git checkout main
git branch -D trying-news-feature
```

Gone. `main` never knew it happened. **This is the fallback you asked
about.**

---

## Going back when things break

### "I changed one file and want to undo it"

```powershell
git checkout -- filename.py
```

Undoes changes to that one file since your last save.

### "I broke everything, undo all my unsaved changes"

```powershell
git reset --hard
```

Back to your last commit. **Warning:** unsaved work is destroyed. That's
the point, but be sure.

### "My last commit was a mistake"

```powershell
git revert HEAD
```

Creates a *new* save that undoes the last one. Safer than deleting,
because the history stays honest.

### "I want to go look at an old version"

```powershell
git checkout a3f9c21        # use an ID from git log
```

Look around. Then come back:

```powershell
git checkout main
```

---

## Your actual workflow

**Starting something risky:**

```powershell
git checkout main
git add .
git commit -m "before trying X"
git checkout -b trying-X
```

**While working:**

```powershell
git add .
git commit -m "small step that works"
```

**It worked:**

```powershell
git checkout main
git merge trying-X
git push
```

**It failed:**

```powershell
git checkout main
git branch -D trying-X
```

---

## Branches worth making for this project

| Branch | For |
|---|---|
| `main` | Only working code. Never experiment here. |
| `ml-migration` | Swapping the lexicon for the classifier |
| `news-feature` | Adding the RSS channel |
| `labelling` | Your labelling sessions |

---

## Two things people get wrong

**1. Forgetting to push.** `commit` saves to your laptop. `push` uploads
to GitHub. If you only commit and your laptop dies, everything is gone.
Push at the end of every session.

**2. Committing secrets.** Never commit `.env`. Your `.gitignore` already
blocks it. If you ever see `.env` in `git status`, stop and fix it before
committing — anything pushed to GitHub is effectively public forever, even
if you delete it later.

---

## If you get stuck

```powershell
git status
```

It usually tells you what to do next in plain English. When it doesn't,
paste the output into Claude Code and ask.
