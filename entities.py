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
        # NOTE: Balen joined RSP in Dec 2025, so the bell / RSP terms are
        # now SHARED with Rabi Lamichhane and no longer identify him alone.
        # They live in SHARED_PARTY_TERMS below instead.
        'symbols': ['balen mayor', 'mayor balen', 'नगर प्रमुख'],
        'scandals': ['vendor force', 'fatuwari', 'garbage', 'फोहोर',
                     'dharahara', 'bulldozer'],
        'party': 'rsp',
        'constituency': 'Jhapa-5',
    },
    'kp oli': {
        'display': 'KP Oli',
        'search': 'KP Oli',
        'aliases': ['kp oli', 'kp', 'oli', 'ओली', 'केपी', 'केपी ओली',
                    'khadga prasad', 'kepi', 'kp baje', 'oli baje'],
        # Generic UML terms (uml/surya/एमाले/☀️...) live in SHARED_PARTY_TERMS
        # only -- NOT duplicated here. 'lauro'/'लौरो' is a personal nickname
        # (cane), not a party symbol, so it stays.
        'symbols': ['lauro', 'लौरो'],
        'scandals': ['lalita niwas', 'ललिता निवास', 'giribandhu', 'omni',
                     'yeti world', '70 crore', 'baluwatar'],
        'party': 'uml',
        'constituency': 'Jhapa-5',
    },
    'gagan thapa': {
        'display': 'Gagan Thapa',
        'search': 'Gagan Thapa',
        'aliases': ['gagan', 'gagan thapa', 'गगन', 'गगन थापा', 'gagan dai'],
        # Generic NC terms live in SHARED_PARTY_TERMS only -- no personal
        # symbol of his own beyond that.
        'symbols': [],
        'scandals': ['bakhra', 'बाख्रा', 'anudan', 'mcc'],
        'party': 'nc',
        'constituency': 'Sarlahi-4',
    },
    'prachanda': {
        'display': 'Prachanda',
        'search': 'Prachanda',
        'aliases': ['prachanda', 'प्रचण्ड', 'dahal', 'दाहाल', 'pushpa kamal',
                    'pk dahal', 'prachand'],
        # Generic Maoist terms live in SHARED_PARTY_TERMS only -- no
        # personal symbol of his own beyond that.
        'symbols': [],
        'scandals': ['cantonment', 'shibir', 'शिविर', 'ncell', 'lda'],
        'party': 'maoist',
        'constituency': 'Eastern Rukum-1',
    },
    'harka sampang': {
        'display': 'Harka Sampang',
        'search': 'Harka Sampang',
        'aliases': ['harka', 'sampang', 'हर्क', 'सम्पाङ', 'harka sampang',
                    'harka raii', 'harka rai'],
        'symbols': ['💧', 'water', 'pani', 'पानी'],
        'scandals': ['pastor', 'gai haney', 'kaku', 'काकु'],
        # Ran for Shram Sanskriti Party, not as an independent -- confirmed
        # by 11 pool comments naming 'श्रम संस्कृति (पार्टी)' directly
        # alongside him (e.g. "...जय हर्क बाद जय श्रम संस्कृति पाटि...").
        'party': 'shram sanskriti',
        'constituency': 'Sunsari-1',
    },
    'rabi lamichhane': {
        'display': 'Rabi Lamichhane',
        'search': 'Rabi Lamichhane',
        'aliases': ['rabi lamichhane', 'ravi lamichhane', 'lamichhane',
                    'रवि लामिछाने', 'रबि लामिछाने', 'लामिछाने',
                    'rabi sir', 'ravi sir', 'rabi dai'],
        # 'rabi' alone is a common given name, so it is deliberately NOT an
        # alias -- it would fire on unrelated people.
        'symbols': ['galaxy 4k', 'sidha kura', 'सिधा कुरा',
                    'sidha kura janata sanga'],
        'scandals': ['cooperative', 'सहकारी', 'sahakari', 'swarnalaxmi',
                     'स्वर्णलक्ष्मी', 'sano paila', 'gb rai', 'गोर्खा मिडिया',
                     'gorkha media', 'citizenship', 'नागरिकता', 'passport',
                     'राहदानी'],
        'party': 'rsp',
        'constituency': 'Chitwan-2',
    },
    # -- validation-set opponents, added for the five-race backtest below --
    # scandals intentionally left empty: not hand-curated yet, and an
    # empty list is correct here, not a placeholder for a guess.
    'mina kharel': {
        'display': 'Mina Kharel',
        'search': 'Mina Kharel',
        # 'mina' alone is a common given name (same reasoning as 'rabi' /
        # 'gagan' above) -- every alias below requires at least two name
        # parts so it can't fire on an unrelated Mina.
        'aliases': ['mina kharel', 'meena kharel', 'मीना खरेल',
                    'mina kharel didi'],
        # Generic NC terms live in SHARED_PARTY_TERMS only -- see the note
        # there on why duplicating them here double-counts a rival hit.
        'symbols': [],
        'scandals': [],
        'party': 'nc',
        'constituency': 'Chitwan-2',
    },
    'goma tamang': {
        'display': 'Goma Tamang',
        'search': 'Goma Tamang',
        # 'goma' alone is a common given name -- same two-name-parts rule.
        'aliases': ['goma tamang', 'गोमा तामाङ', 'goma tamang didi'],
        # Generic RSP terms live in SHARED_PARTY_TERMS only.
        'symbols': [],
        'scandals': [],
        'party': 'rsp',
        'constituency': 'Sunsari-1',
    },
    'amresh kumar singh': {
        'display': 'Amresh Kumar Singh',
        'search': 'Amresh Kumar Singh',
        # 'amresh' alone is a common given name -- same two-name-parts rule.
        # 'singh' alone is excluded too: an extremely common surname on its
        # own, not specific to this candidate.
        'aliases': ['amresh kumar singh', 'amresh singh',
                    'अमरेश कुमार सिंह', 'अमरेश सिंह'],
        # Generic RSP terms live in SHARED_PARTY_TERMS only.
        'symbols': [],
        'scandals': [],
        'party': 'rsp',
        'constituency': 'Sarlahi-4',
    },
    'leelamani gautam': {
        'display': 'Leelamani Gautam',
        'search': 'Leelamani Gautam',
        'aliases': ['leelamani gautam', 'leela mani gautam',
                    'lila mani gautam', 'लीलामणि गौतम', 'लिला मणि गौतम'],
        # Generic UML terms live in SHARED_PARTY_TERMS only.
        'symbols': [],
        'scandals': [],
        'party': 'uml',
        'constituency': 'Eastern Rukum-1',
    },
}

# --------------------------------------------------------------------------
# Real head-to-head results for the five constituencies above, used to
# validate the model's win-probability output against actual outcomes.
# `total_votes` is approximate: the sum of the two listed candidates only
# -- other, minor candidates on the same ballot are not accounted for.
# --------------------------------------------------------------------------

# `race_queries` -- the constituency-name-only YouTube search queries
# (from collect_pool.EXTRA_QUERIES) used for this race, e.g. 'Jhapa 5
# chunav'. These are the queries run for BOTH candidates in a race, so a
# comment they surface can end up tagged to whichever candidate happened
# to be fetched first (pool-wide dedup keeps the first-seen row). Matching
# on the `source` column recovers those comments regardless of which
# candidate's tag they landed under.

RACES = [
    {
        'constituency': 'Jhapa-5',
        'candidate_a': 'balen', 'votes_a': 68348,
        'candidate_b': 'kp oli', 'votes_b': 18734,
        'winner': 'balen',
        'total_votes': 68348 + 18734,
        'race_queries': ['Jhapa 5 chunav'],
    },
    {
        'constituency': 'Chitwan-2',
        'candidate_a': 'rabi lamichhane', 'votes_a': 54402,
        'candidate_b': 'mina kharel', 'votes_b': 14564,
        'winner': 'rabi lamichhane',
        'total_votes': 54402 + 14564,
        'race_queries': ['Chitwan 2 chunav', 'Chitwan 2 nirbachan'],
    },
    {
        'constituency': 'Sunsari-1',
        'candidate_a': 'harka sampang', 'votes_a': 35741,
        'candidate_b': 'goma tamang', 'votes_b': 27249,
        'winner': 'harka sampang',
        'total_votes': 35741 + 27249,
        'race_queries': ['Sunsari 1 chunav', 'Sunsari 1 nirbachan'],
    },
    {
        'constituency': 'Sarlahi-4',
        'candidate_a': 'amresh kumar singh', 'votes_a': 35688,
        'candidate_b': 'gagan thapa', 'votes_b': 22838,
        'winner': 'amresh kumar singh',
        'total_votes': 35688 + 22838,
        'race_queries': ['Sarlahi 4 chunav', 'Sarlahi 4 nirbachan'],
    },
    {
        'constituency': 'Eastern Rukum-1',
        'candidate_a': 'prachanda', 'votes_a': 10240,
        'candidate_b': 'leelamani gautam', 'votes_b': 3462,
        'winner': 'prachanda',
        'total_votes': 10240 + 3462,
        'race_queries': ['Rukum Purba chunav', 'Rukum 1 nirbachan'],
    },
]

# --------------------------------------------------------------------------
# Party-level terms
#
# These identify a PARTY, not a person. Since Balen joined RSP in December
# 2025, the bell symbol and 'rsp' are shared between Balen and Rabi, so
# they can no longer disambiguate the two. Treating them as a personal
# symbol for either one would misattribute every RSP comment.
#
# A party term counts as a WEAK signal for any candidate in that party --
# it narrows the field without naming an individual.
#
# IMPORTANT: these terms must NOT also appear in any individual candidate's
# own 'symbols' list. attribution()'s rival-scan loop sums aliases+symbols
# per OTHER candidate, so the same generic party term duplicated into two
# same-party candidates' symbols gets counted TWICE -- once per candidate,
# as if two individuals were named, when the text really names neither.
# (This is exactly what happened with 'ghanti' being baked into both goma
# tamang's and amresh kumar singh's symbols -- a two-member RSP field
# turned one generic word into rival_hits=2.0 against balen.) Personal
# symbols (a nickname, an office title) are fine; a copy of the party's
# own term list is not -- SHARED_PARTY_TERMS below is the only place that
# should live.
# --------------------------------------------------------------------------

SHARED_PARTY_TERMS = {
    'rsp': ['🔔', 'ghanti', 'घण्टी', 'jay ghanti', 'rsp', 'रास्वपा',
            'raswapa', 'rastriya swatantra', 'राष्ट्रिय स्वतन्त्र'],
    # 'umale' / 'enaile' are common romanized colloquial references to UML
    # ("umale" = UML+ley, "enaile" = a romanized एमाले variant) that don't
    # match the existing 'emale'/'uml' spellings.
    'uml': ['☀️', '🌞', 'uml', 'एमाले', 'emale', 'surya', 'सूर्य',
            'umale', 'enaile'],
    'nc': ['🌳', 'nc', 'congress', 'कांग्रेस', 'rukh', 'रुख'],
    'maoist': ['🔨', 'maoist', 'माओवादी', 'hathoda', 'हथौडा'],
    # Only the Devanagari multi-word phrase is included. The bare word
    # 'shram'/'श्रम' ("labor/effort") is common Nepali vocabulary unrelated
    # to the party -- confirmed by false hits in the pool ("vishram" =
    # rest, "shram gareko" = made an effort). No romanized 'shram
    # sanskriti' spelling appears anywhere in the pool, so it isn't added
    # here either; only what's actually evidenced in the data is included.
    'shram sanskriti': ['श्रम संस्कृति'],
}

PARTY_HIT_WEIGHT = 0.4   # a party mention is weaker evidence than a name

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


def attribution(text: str, candidate: str,
                party_hit_weight: float = PARTY_HIT_WEIGHT,
                context_prior: float = CONTEXT_PRIOR) -> dict:
    """
    Decide who a comment is aimed at.

    `party_hit_weight` and `context_prior` both default to their module
    constants so every existing caller is unaffected; they're exposed as
    parameters so validation code can override them without monkeypatching
    the module globals. `context_prior` in particular assumes the comment
    came from THIS candidate's own search/fetch -- true for the
    single-candidate app flow, false when scoring a shared multi-candidate
    pool (e.g. head-to-head validation), where it should be passed as 0.0.

    Returns a dict with:
        attribution  -- float in [-1, +1]
        scandal      -- float, additive adjustment
        own_hits / rival_hits -- ints, for debugging and for the report
    """
    key = resolve(candidate)
    t = _norm(text)

    own_cfg = CANDIDATES.get(key)
    own_terms = (own_cfg['aliases'] + own_cfg['symbols']) if own_cfg else [key]
    own_hits = float(_count_hits(t, own_terms))

    rival_hits = 0.0
    rival_scandal_hit = False
    for rkey, rcfg in CANDIDATES.items():
        if rkey == key:
            continue
        rival_hits += _count_hits(t, rcfg['aliases'] + rcfg['symbols'])
        if not rival_scandal_hit and _count_hits(t, rcfg['scandals']):
            rival_scandal_hit = True

    # --- party-level terms: weak evidence ---
    # OWN party: skipped once an individual from that party is named --
    # the direct name is already stronger evidence, so the generic party
    # tag adds nothing on top and staying ambiguous here is fine.
    # RIVAL party: always credited, named or not. A comment can both
    # name a specific rival AND separately invoke their party ("KP Oli
    # chor, jay UML") -- that party-level rhetoric is real evidence
    # against us and shouldn't be dropped just because someone from that
    # party happened to be named elsewhere in the same text.
    own_party = own_cfg.get('party') if own_cfg else None
    for party, terms in SHARED_PARTY_TERMS.items():
        if not _count_hits(t, terms):
            continue
        members = [k for k, c in CANDIDATES.items() if c.get('party') == party]
        if party == own_party:
            named = any(_count_hits(t, CANDIDATES[m]['aliases']) for m in members)
            if named:
                continue     # an individual was named; the party tag adds nothing
            # ambiguous between our candidate and their party colleagues,
            # so the evidence is divided among them
            own_hits += party_hit_weight / max(1, len(members))
        else:
            rival_hits += party_hit_weight

    # --- attribution ---
    # Fractional hits mean the evidence was party-level only, which is
    # ambiguous between party colleagues. Confidence is capped accordingly
    # rather than treated as a positive identification.
    if own_hits and not rival_hits:
        attr = 1.0 if own_hits >= 1.0 else min(1.0, own_hits + context_prior * (1 - own_hits))
    elif rival_hits and not own_hits:
        attr = -1.0 if rival_hits >= 1.0 else -min(1.0, rival_hits + 0.2)
    elif own_hits and rival_hits:
        # both named: lean toward whoever dominates, but heavily damped
        attr = AMBIGUOUS_DAMP * (own_hits - rival_hits) / (own_hits + rival_hits)
    else:
        attr = context_prior

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
