# Labelling guide

Keep this open in a second window while you work.

---

## Setup — three commands

```bash
pip install -r requirements.txt

# merge election_data.csv + cache.json into one de-duplicated pool
python collect_pool.py
# -> 1226 raw -> 1134 unique comments -> labelling_pool.csv

# start
python label_tool.py --data labelling_pool.csv --out labelled.csv --n 400
```

Your pool is **1,134 comments** (722 Latin, 346 Devanagari, 66 mixed).
That's below the 1,500–2,000 ideal but enough for a first working model.
Run `python collect_pool.py --fetch` later to top up across all five
candidates.

---

## The one rule that matters

> **Label the sentiment of the text. Not who it helps.**

`"KP Oli chor ho"` is **negative**. Always. Even in a Balen analysis.
`entities.py` handles the flip to Balen's benefit — that's its whole job.

If you label by "who does this help," you fold attribution into the
classifier, the two layers fight each other, and the only fix is
relabelling from zero. This is the most common way a dataset like this
gets ruined.

---

## Decision order

Work down the list. Stop at the first match.

1. **Does it attack, mock, blame, accuse, or insult anyone?** → `negative` (`a`)
2. **Does it praise, support, defend, endorse, or agree?** → `positive` (`d`)
3. **Is it factual, a genuine question, off-topic, or spam?** → `neutral` (`s`)
4. **Truly can't decide after ~10 seconds?** → `skip` (`x`)

Don't agonise. A fast consistent label beats a slow agonised one, because
the model learns your *pattern*, not your individual verdicts.

---

## Worked examples from your actual data

### negative — `a`

| Comment | Why |
|---|---|
| `Budo jindagi nasudrine vo` | Mockery of age and character |
| `kepi bajeko gaf kehi chalden 😊` | Dismissal. The 😊 is sarcastic, not friendly |
| `उखानले देश चल्ला र?` | Rhetorical question = attack, not a question |
| `बतासे, हावा, झुटो, गफाडी, उखान-टुक्के` | String of insults |
| `ए माले पाटिले अब कमेडी शो चलायो भने चल्छ` | Sarcasm — surface is a suggestion, intent is contempt |
| `Muji pata papi buda` | Profanity directed at a person |
| `Aru ko khedo khanne matra tero kam xaen` | Accusation of behaviour |

### positive — `d`

| Comment | Why |
|---|---|
| `Balen ❤❤` | Emoji-only endorsement still counts |
| `हो कुरा त सहि छ` | Agreement with the speaker |
| `Balen dai ramro kaam garyo` | Direct praise |

### neutral — `s`

| Comment | Why |
|---|---|
| `Himali bhegma manxe nahunu chai terai Karan le...` | A causal claim, no evaluation of a person |
| `झापा 5 मा बालेन मात्रै त होइन होला नि` | Observation about a race, no charge either way |
| `Video ko link kata cha?` | Genuine question |
| `Subscribe garnu hola` | Spam |

---

## Five edge cases you will definitely hit

**1. Attacks one, praises another** — `Bolyo Ki Polyo Oli Ba. Ab ki bar Balen Sarkar.`

Pick the **stronger** charge. Here the endorsement is the payload, so
`positive`. If they feel genuinely equal, `skip`. Don't agonise — these
are maybe 3% of comments.

**2. Sarcasm** — label by **intent**, not surface wording.
`वाह क्या राम्रो काम` after a scandal is `negative`. This is where the
model gains the most over your old lexicon, which had no way to see it.
Getting these right is high value.

**3. Rhetorical questions** — question form does **not** mean neutral.
`देश यसरी नै चल्छ?` is an attack.

**4. Profanity** — `negative`, unless it's affectionate (`muji ramro kaam
garyo` exists and is `positive`). Judge the sentence, not the word.

**5. Attacks a third party, not a politician** — a comment attacking
journalists or another commenter is still `negative` in sentiment.
`entities.py` will find no candidate mention and apply the weak context
prior. That's correct behaviour; don't try to compensate by hand.

---

## Session discipline

- **30–40 minutes at a time.** Not one long block. Fatigue makes labels
  drift, and drift is worse than fewer labels.
- **~4–5 seconds per comment.** 1,134 comments ≈ 90 minutes of actual
  work, so about three sessions.
- **Ctrl-C is safe.** Progress saves. Re-running resumes where you left off.
- **Watch the live class counter** in the header. Your KP Oli comments
  are heavily negative — if `positive` stays under 10%, deliberately go
  hunting for positive examples instead of continuing at random. A class
  the model never sees is a class it can never predict.

---

## Quality check — do this, it's a slide

After a few days, when you no longer remember your choices:

```bash
python label_tool.py --audit --out labelled.csv --n 100
```

It re-presents 100 of your own labelled comments blind and reports
**Cohen's kappa** — agreement with your past self.

| kappa | Meaning |
|---|---|
| ≥ 0.80 | Excellent. Guideline is well defined. |
| 0.65–0.80 | Acceptable. Tighten the rules for your weakest class. |
| < 0.65 | Rewrite the guideline and relabel. More labels won't help. |

It also prints *which* pairs you flip-flopped on — usually
neutral↔negative, which tells you exactly which rule to sharpen.

**Report this number.** Almost no undergraduate project measures
annotation reliability, and it pre-empts the "how do you know your labels
are any good?" question before it's asked.

---

## Agreed rules (v2 — after kappa 0.558 review)

An audit against a second labeller came back at kappa 0.558 — below the
0.65 floor. We went through every disagreement together and agreed on the
following four rules. These override the general edge-case guidance above
where they conflict.

**Rule 1 — Mixed attack + praise → label the PRAISE.** The endorsement is
the payload, the attack is setup.

> `केपी चोरभन्दा त प्रचण्ड हजार गुणा best!!` → **positive**, not negative.

> Clarification: this only applies when the praise is **explicitly
> stated**. Mocking someone's opponents is not by itself an endorsement,
> and slogans like `जय जेनजि` or `जय ग्रेटर नेपाल` don't count as praise
> of a candidate. If there's no actual praise of a person in the text,
> it's negative.
>
> `🇳🇵🙏Genz आन्दोलन र सुनसरी घटना एउटै हो भन्ने, एउटै देख्नेको चै
> मानसिक स्थीतीको check गराए हुन्छ।` → **negative**, because it only
> mocks — the defence of GenZ is implied, not stated.

**Rule 2 — Sarcasm is judged on intent, not surface words.** Markers:
`फन्टुस`, exaggerated praise following criticism, 🤣 after mockery.

> `फन्टुस तर्क नगरौ न हो! यत्रो powerful PM भएको मान्छे!` → **negative**,
> despite "powerful PM".

**Rule 3 — Rejection of a party or symbol is NEGATIVE**, even when mildly
phrased.

> `यो बेलुन घण्टी मा आय बाट मलाई rsp मा पर्दैन` → **negative**.

**Rule 4 — Aspirational calls for better governance are POSITIVE**, not
negative — they're not attacks.

> `गृहमन्त्री ले नैतिकता देखाएर राजीनामा दिएर परिपक्व गृहमन्त्री बनाउ` →
> **positive**.

**Also:** a long propaganda-style list (many grievance bullet points) is
labelled by its **dominant sentiment**, not automatically `negative`.

**Rule 5 — If a comment is too garbled or misspelt to understand
confidently, press `x` to skip.** A guessed label teaches the model
noise.

> `सार्वभौमसत्ता राष्ट-राष्ट्रियतामा भनेरअ लेखिदीएकाेभयए...` →
> unparseable, skip it.

---

## The loop

```bash
# 1. label a first batch
python label_tool.py --data labelling_pool.csv --out labelled.csv --n 400

# 2. train
python train_sentiment.py --data labelled.csv --out models/sentiment_clf.pkl

# 3. label another batch -- now targeting what the model finds hardest
python label_tool.py --data labelling_pool.csv --out labelled.csv \
    --n 300 --uncertain

# 4. retrain, compare, repeat
python train_sentiment.py --data labelled.csv --out models/sentiment_clf.pkl
```

`--uncertain` sorts by prediction entropy — the comments the model is most
confused about. Those are worth roughly 2–3× a random comment, so passes
2 and 3 buy more accuracy per minute than pass 1 did.

**Stop when** macro-F1 stops improving between rounds. Usually 1,200–1,800
labels for a task this size. Record the curve (labels vs macro-F1) — it's
a good figure and it justifies where you stopped.

---

## What to watch when you train

Ignore accuracy. Watch these three:

1. **macro-F1 vs the majority baseline.** The script prints the baseline.
   Beating it by less than ~10 points means the model isn't learning much.
2. **Per-class recall on `positive`.** It'll be your rarest class and your
   weakest. Mohan's reference model reported 77% accuracy while getting
   negatives right only 49% of the time — accuracy hid it completely.
3. **Per-script macro-F1.** In my smoke test, Devanagari showed accuracy
   0.990 but macro-F1 0.498 — the model had learned "Devanagari → always
   neutral," which is exactly your old lexicon's bug reproduced. Accuracy
   said everything was fine. macro-F1 caught it. **This is the number that
   proves you fixed the thing this project exists to fix.**

---

# Addendum: two channels, one model

## Do not train separate models

At your data scale that's the wrong call. You'd have ~1,100 comments and
maybe 300 headlines — roughly 100 examples per class for the headline
model. Too thin for stable char n-gram features, and it throws away the
fact that both channels share most of their political vocabulary
(`चोर`, `भ्रष्टाचार`, `विकास`, `राम्रो`).

Instead: **one pooled model with a channel marker.** Every document is
prefixed `__ch_comment__` or `__ch_news__`. Since the word vectoriser runs
bigrams, that token pairs with the first real word, giving the model both
a per-channel prior and limited per-channel interaction terms — most of
the benefit of two models, at a fraction of the labelling cost.

Shared vocabulary trains on all 1,400 examples instead of being split.

## Updated workflow

```bash
# build a comment pool AND a headline pool in one file
python collect_pool.py --news

# label both -- the tool tags each row with its channel automatically
python label_tool.py --data labelling_pool.csv --out labelled.csv --n 400

# train (channel handling is automatic)
python train_sentiment.py --data labelled.csv --out models/sentiment_clf.pkl
```

`labelled.csv` now has three columns: `text`, `label`, `channel`.

Target roughly **1,100 comments + 300 headlines**. Headlines are faster to
label — they're short, formal, and mostly neutral — so 300 costs maybe 20
minutes.

## Labelling headlines: the rule changes slightly

Headlines are event reporting, not opinion. Most are genuinely `neutral`,
and that is the correct label — resist the urge to read implied sentiment
into a factual report.

| Headline | Label | Why |
|---|---|---|
| `ओली सरकारले बजेट पेश गर्‍यो` | neutral | Event report, no evaluation |
| `प्रधानमन्त्री ओली अस्पताल भर्ना` | neutral | Negative *tone*, but not an attack |
| `ओली विरुद्ध भ्रष्टाचार मुद्दा दर्ता` | negative | Reports an accusation against him |
| `ओलीको भाषणमा विपक्षी आक्रोशित` | negative | Reports hostility toward him |
| `Oli's reforms win broad praise` | positive | Reports approval |

The trap: `अस्पताल भर्ना` (hospitalised) *feels* negative and a
comment-trained model will likely score it so. It isn't sentiment toward
the candidate. Getting these right is the main value of labelling
headlines separately.

## What to check after training

The trainer now prints a **PER-CHANNEL** block:

```
comment   n= 1060   acc=0.882   macro-F1=0.787   (baseline 0.802, lift +0.080)
news      n=  248   acc=0.831   macro-F1=0.742   (baseline 0.750, lift +0.081)

channel macro-F1 gap: 0.045
-> acceptable. One pooled model is serving both channels.
```

**Read the gap, not the accuracy.** News will show higher raw accuracy
simply because it's more neutral-heavy — that's the baseline moving, not
the model improving. The lift over each channel's own baseline is the
honest comparison.

| gap | action |
|---|---|
| < 0.15 | Pooled model is fine. Ship it. |
| > 0.15 **and** 500+ labels in the weak channel | Retrain with `--separate` |
| > 0.15 **and** thin labels | Label more of the weak channel. Splitting on thin data makes it worse. |

The escape hatch exists if you need it:

```bash
python train_sentiment.py --data labelled.csv --separate
```

It refuses to split any channel with under 200 rows, which is the guard
that stops you shooting yourself in the foot.

## For your report

This is a good methods paragraph, because the reasoning is the
contribution: you identified a domain-shift risk, chose pooled training
with a channel indicator over separate models on data-efficiency grounds,
and built the per-channel diagnostic that would tell you if that choice
was wrong. That is a more sophisticated answer than either "I trained one
model" or "I trained two."
