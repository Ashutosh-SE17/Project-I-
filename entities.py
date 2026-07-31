"""
entities.py
-----------
The entity / attribution layer.

This is the part of the original rule engine that is KEPT. Its job has
changed, though: it no longer decides whether a comment is positive or
negative (the classifier does that now). It decides WHO the comment is
about, which is the thing a bag-of-words classifier cannot do.

    polarity     -- from the ML model      -- "how positive is this text?"
    attribution  -- from this module       -- "is it aimed at our candidate
                                              or at a rival?"

    score = polarity * attribution + scandal_term

Worked example, analysing BALEN:
    "KP Oli chor ho"      polarity -0.8, attribution -1.0  ->  +0.8
    (a negative comment about a rival is good news for our candidate)

    "Balen chor ho"       polarity -0.8, attribution +1.0  ->  -0.8
"""

import re

# --------------------------------------------------------------------------
# Candidate registry
#
# Add a candidate here and the whole pipeline picks them up. Aliases cover
# Devanagari, romanized, and common misspellings.
# --------------------------------------------------------------------------

CANDIDATES = {
    'balen': {
        'display': 'Balen Shah',
        'search': 'Balen Shah',
        'aliases': ['balen', 'balendra', 'बालेन', 'बालेन्द्र', 'balen shah',
                    'valen', 'baalen'],
        'symbols': ['🔔', 'ghanti', 'घण्टी', 'jay ghanti', 'rsp', 'रास्वपा',
                    'raswapa'],
        'scandals': ['vendor force', 'fatuwari', 'garbage', 'फोहोर',
                     'dharahara', 'bulldozer'],
    },
    'kp oli': {
        'display': 'KP Oli',
        'search': 'KP Oli',
        'aliases': ['kp oli', 'kp', 'oli', 'ओली', 'केपी', 'केपी ओली',
                    'khadga prasad', 'kepi', 'kp baje', 'oli baje'],
        'symbols': ['☀️', '🌞', 'uml', 'एमाले', 'emale', 'lauro', 'लौरो',
                    'surya'],
        'scandals': ['lalita niwas', 'ललिता निवास', 'giribandhu', 'omni',
                     'yeti world', '70 crore', 'baluwatar'],
    },
    'gagan thapa': {
        'display': 'Gagan Thapa',
        'search': 'Gagan Thapa',
        'aliases': ['gagan', 'gagan thapa', 'गगन', 'गगन थापा', 'gagan dai'],
        'symbols': ['🌳', 'nc', 'congress', 'कांग्रेस', 'rukh', 'रुख'],
        'scandals': ['bakhra', 'बाख्रा', 'anudan', 'mcc'],
    },
    'prachanda': {
        'display': 'Prachanda',
        'search': 'Prachanda',
        'aliases': ['prachanda', 'प्रचण्ड', 'dahal', 'दाहाल', 'pushpa kamal',
                    'pk dahal', 'prachand'],
        'symbols': ['🔨', 'maoist', 'माओवादी', 'hathoda', 'हथौडा'],
        'scandals': ['cantonment', 'shibir', 'शिविर', 'ncell', 'lda'],
    },
    'harka sampang': {
        'display': 'Harka Sampang',
        'search': 'Harka Sampang',
        'aliases': ['harka', 'sampang', 'हर्क', 'सम्पाङ', 'harka sampang',
                    'harka raii', 'harka rai'],
        'symbols': ['💧', 'water', 'pani', 'पानी'],
        'scandals': ['pastor', 'gai haney', 'kaku', 'काकु'],
    },
}

# --------------------------------------------------------------------------
# Tunable weights -- every one of these belongs in your report as a
# documented hyperparameter, not a magic number buried in code.
# --------------------------------------------------------------------------

CONTEXT_PRIOR = 0.70      # comment names nobody; it came from this candidate's
                          # video search, so assume it's probably about them
AMBIGUOUS_DAMP = 0.35     # both candidate and rival named -> weak, uncertain
OWN_SCANDAL = -0.45       # our candidate's controversy mentioned
RIVAL_SCANDAL = 0.25      # a rival's controversy mentioned (weaker: attacking
                          # a rival is not the same as endorsing us)
DISRESPECT = -0.30        # disrespectful suffix attached to our candidate

DISRESPECT_SUFFIXES = ['e', 'ie', 'ba', 'baba', 'baje', 'buda', 'budi',
                       'ni', 'niya', 'wa']


def _norm(text: str) -> str:
    return ' ' + re.sub(r'\s+', ' ', str(text).lower().strip()) + ' '


def _count_hits(text_padded: str, terms) -> int:
    """
    Count how many distinct terms appear. Uses word-boundary matching for
    ASCII terms so that 'oli' does not fire inside 'policy', and plain
    substring matching for Devanagari/emoji where \\b is unreliable.
    """
    hits = 0
    for term in terms:
        t = term.lower()
        if re.fullmatch(r'[a-z0-9 ]+', t):
            if re.search(r'(?<![a-z])' + re.escape(t) + r'(?![a-z])', text_padded):
                hits += 1
        else:
            if t in text_padded:
                hits += 1
    return hits


def resolve(candidate: str) -> str:
    """Map any user input ('Balen Shah', 'BALEN') to a registry key."""
    c = str(candidate).strip().casefold()
    if c in CANDIDATES:
        return c
    for key, cfg in CANDIDATES.items():
        if c == cfg['display'].casefold() or c in [a.lower() for a in cfg['aliases']]:
            return key
    for key, cfg in CANDIDATES.items():
        if key in c or c in key:
            return key
    return c  # unknown candidate: attribution falls back to the prior


def attribution(text: str, candidate: str) -> dict:
    """
    Decide who a comment is aimed at.

    Returns a dict with:
        attribution  -- float in [-1, +1]
        scandal      -- float, additive adjustment
        own_hits / rival_hits -- ints, for debugging and for the report
    """
    key = resolve(candidate)
    t = _norm(text)

    own_cfg = CANDIDATES.get(key)
    own_terms = (own_cfg['aliases'] + own_cfg['symbols']) if own_cfg else [key]
    own_hits = _count_hits(t, own_terms)

    rival_hits = 0
    rival_scandal_hit = False
    for rkey, rcfg in CANDIDATES.items():
        if rkey == key:
            continue
        rival_hits += _count_hits(t, rcfg['aliases'] + rcfg['symbols'])
        if not rival_scandal_hit and _count_hits(t, rcfg['scandals']):
            rival_scandal_hit = True

    # --- attribution ---
    if own_hits and not rival_hits:
        attr = 1.0
    elif rival_hits and not own_hits:
        attr = -1.0
    elif own_hits and rival_hits:
        # both named: lean toward whoever dominates, but heavily damped
        attr = AMBIGUOUS_DAMP * (own_hits - rival_hits) / (own_hits + rival_hits)
    else:
        attr = CONTEXT_PRIOR

    # --- scandal term (independent of attribution) ---
    scandal = 0.0
    if own_cfg and _count_hits(t, own_cfg['scandals']):
        scandal += OWN_SCANDAL
    if rival_scandal_hit:
        scandal += RIVAL_SCANDAL

    # --- disrespectful suffix on our candidate's name ---
    if own_cfg:
        flat = t.replace(' ', '')
        for alias in own_cfg['aliases']:
            base = alias.lower().replace(' ', '')
            if len(base) < 3:
                continue
            for suf in DISRESPECT_SUFFIXES:
                if base + suf in flat and base + suf != base:
                    scandal += DISRESPECT
                    break
            else:
                continue
            break

    return {
        'attribution': round(attr, 3),
        'scandal': round(scandal, 3),
        'own_hits': own_hits,
        'rival_hits': rival_hits,
    }


if __name__ == '__main__':
    cases = [
        ('KP Oli chor ho, desh barbad garyo', 'balen'),
        ('Balen dai ramro kaam garyo 🔔', 'balen'),
        ('Balen bhanda Oli ramro cha', 'balen'),
        ('ललिता निवास काण्ड नबिर्सौं', 'kp oli'),
        ('ललिता निवास काण्ड नबिर्सौं', 'balen'),
        ('desh ko awastha naramro cha', 'balen'),
    ]
    for text, cand in cases:
        r = attribution(text, cand)
        print(f'[{cand:12s}] {text[:38]:40s} attr={r["attribution"]:+.2f} '
              f'scandal={r["scandal"]:+.2f} (own={r["own_hits"]} rival={r["rival_hits"]})')
