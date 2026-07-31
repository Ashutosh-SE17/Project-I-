"""
preprocessing.py
----------------
Script-aware text normalisation for Nepali election comments.

Design notes (why this differs from the reference implementation):

  * Devanagari is PRESERVED. The reference pipeline stripped every
    non-ASCII character, which silently deleted ~35% of a real YouTube
    corpus. Here Devanagari and Latin are both kept and normalised.

  * Word ORDER is preserved. The reference returned `set(words)`, which
    destroyed order (making bigrams meaningless) and destroyed term
    frequency (making TF-IDF degenerate to plain IDF). We return a list
    in original order.

  * Stemming is LIGHT and guarded. Suffix stripping only fires when a
    plausible stem remains, so `rama` is not mangled into `ra`.

  * Negation is marked, not dropped. `ramro chaina` becomes
    `ramro NEG_chaina`-style tagging so the classifier can learn it.
"""

import re
import unicodedata

# --------------------------------------------------------------------------
# Character classes
# --------------------------------------------------------------------------

DEVANAGARI = re.compile(r'[\u0900-\u097F]')
EMOJI = re.compile(
    "[" 
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "]",
    flags=re.UNICODE,
)
URL = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG = re.compile(r'<[^>]+>')
HTML_ENTITY = re.compile(r'&[a-zA-Z]+;|&#\d+;')
MENTION = re.compile(r'@\w+')
REPEAT_CHAR = re.compile(r'(.)\1{2,}')
NON_TOKEN = re.compile(r'[^\w\u0900-\u097F\s]', flags=re.UNICODE)
MULTISPACE = re.compile(r'\s+')

# --------------------------------------------------------------------------
# Romanized Nepali stopwords
#
# Adapted and extended from the approach used in Mohan Bisunke's
# romanized-nepali-text-sentiment-analysis (2023). Reimplemented here
# rather than copied, and paired with a Devanagari equivalent list.
# --------------------------------------------------------------------------

STOPWORDS_LATIN = {
    'ek', 'hai', 'tyo', 'nai', 'k', 'ko', 'ma', 'la', 'lai', 'le', 'ka',
    'ho', 'ehh', 'hmm', 'ha', 'hajur', 'ta', 'haha', 'timi', 'tapai',
    'uni', 'usko', 'usle', 'unle', 'yrr', 'oh', 'ohh', 'hun', 'ra', 'po',
    'pani', 'yesle', 'ni', 'hamro', 'tw', 'tmle', 'tmlai', 'yo', 'chhan',
    'xa', 'pni', 'chha', 'timro', 'maile', 'honi', 'hota', 'ani', 'tara',
    'ki', 'kina', 'jun', 'jasto', 'aba', 'hola', 'hunu', 'garna', 'bhaneko',
    'bhanne', 'hunchha', 'thiyo', 'chha', 'thie', 'huncha',
}

STOPWORDS_DEVA = {
    'र', 'को', 'का', 'की', 'ले', 'लाई', 'मा', 'हो', 'छ', 'छन्', 'हुन्',
    'यो', 'त्यो', 'तर', 'पनि', 'नै', 'कि', 'भने', 'हुन्छ', 'थियो',
    'गर्न', 'भएको', 'हामी', 'तिमी', 'उनी', 'अब', 'के', 'यस', 'त',
}

STOPWORDS = STOPWORDS_LATIN | STOPWORDS_DEVA

# --------------------------------------------------------------------------
# Negation markers -- these are NEVER removed as stopwords
# --------------------------------------------------------------------------

# Nepali is SOV: the negator attaches to the word BEFORE it.
#   "ramro chaina"  = good-not  -> `ramro` is what's negated
# English attaches to the word AFTER it.
#   "not good"                  -> `good` is what's negated
# Getting this backwards (as a naive port would) silently inverts the
# negation on every Nepali sentence, so the two sets are kept separate.

NEGATORS_BACKWARD = {
    # romanized Nepali
    'chaina', 'chhaina', 'xaina', 'hoina', 'hoinaa', 'chhainan', 'xainan',
    'thiena', 'thienan', 'bhaena', 'garena', 'napaeko', 'nabhaeko',
    # Devanagari
    'छैन', 'छैनन्', 'होइन', 'थिएन', 'भएन', 'गरेन', 'नभएको',
}

NEGATORS_FORWARD = {
    'no', 'not', 'never', 'nothing', 'none', 'nahos', 'nagara',
    'dont', 'doesnt', 'didnt', 'cant', 'wont', 'isnt', 'arent',
    'नहोस्', 'नगर', 'कहिल्यै',
}

NEGATORS = NEGATORS_BACKWARD | NEGATORS_FORWARD
NEGATION_SCOPE = 2  # tokens on the relevant side that get the NOT_ prefix

# --------------------------------------------------------------------------
# Light suffix stemmer for romanized Nepali
#
# (suffix, replacement, min_stem_length)
# Ordered longest-first so that `chhaina` is matched before `chha`.
# --------------------------------------------------------------------------

SUFFIX_RULES = [
    ('chhaina', 'xaina', 2),
    ('chaina',  'xaina', 2),
    ('chhau',   'xau',   2),
    ('chhan',   'xan',   2),
    ('chain',   'xaina', 2),
    ('chha',    'xa',    2),
    ('hunxa',   '',      3),
    ('hunxu',   '',      3),
    ('kheri',   '',      3),
    ('haru',    '',      3),
    ('sake',    '',      3),
    ('paxi',    '',      3),
    ('hos',     '',      3),
    ('lai',     '',      3),
    ('ko',      '',      3),
    ('le',      '',      3),
    ('ma',      '',      3),
]


def _normalize_repeats(text: str) -> str:
    """`ramroooo` -> `ramroo` (keep one doubling as an intensity signal)."""
    return REPEAT_CHAR.sub(r'\1\1', text)


def _stem_latin(word: str) -> str:
    """Guarded suffix stripping. Only fires if a real stem survives."""
    for suffix, replacement, min_stem in SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_stem:
            return word[: -len(suffix)] + replacement
    return word


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI.search(str(text)))


def script_of(text: str) -> str:
    """Return 'deva', 'latin', or 'mixed' -- useful for stratified reporting."""
    t = str(text)
    deva = bool(DEVANAGARI.search(t))
    latin = bool(re.search(r'[A-Za-z]', t))
    if deva and latin:
        return 'mixed'
    if deva:
        return 'deva'
    return 'latin'


def extract_emoji(text: str) -> list:
    """Emoji are kept separately -- they carry real signal (🔔 ☀️ 🌳 🔨)."""
    return EMOJI.findall(str(text))


def clean(text: str, stem: bool = True, mark_negation: bool = True) -> str:
    """
    Normalise a raw comment into a token string suitable for vectorising.

    Unlike the reference implementation this keeps Devanagari, keeps word
    order, and keeps term frequency.
    """
    t = str(text)

    # YouTube comment bodies arrive with HTML in them
    t = HTML_TAG.sub(' ', t)
    t = HTML_ENTITY.sub(' ', t)
    t = URL.sub(' ', t)
    t = MENTION.sub(' ', t)

    emoji = extract_emoji(t)

    t = unicodedata.normalize('NFKC', t)
    t = t.lower()
    t = _normalize_repeats(t)
    t = NON_TOKEN.sub(' ', t)
    t = MULTISPACE.sub(' ', t).strip()

    tokens = t.split()

    # --- pass 1: filter + stem, remembering which tokens are negators ---
    kept = []
    for tok in tokens:
        if len(tok) < 2 and not DEVANAGARI.search(tok):
            continue

        is_negator = tok in NEGATORS

        if not is_negator and tok in STOPWORDS:
            continue

        if stem and not DEVANAGARI.search(tok):
            stemmed = _stem_latin(tok)
            # never let stemming destroy a negator
            tok = tok if is_negator else (stemmed or tok)

        kept.append((tok, is_negator))

    # --- pass 2: apply negation in the correct direction ---
    out = [t for t, _ in kept]
    if mark_negation:
        for i, (tok, is_neg) in enumerate(kept):
            if not is_neg:
                continue
            if tok in NEGATORS_BACKWARD:
                lo = max(0, i - NEGATION_SCOPE)
                span = range(lo, i)
            else:
                hi = min(len(kept), i + NEGATION_SCOPE + 1)
                span = range(i + 1, hi)
            for j in span:
                if not kept[j][1] and not out[j].startswith('NOT_'):
                    out[j] = 'NOT_' + out[j]

    # Emoji appended as explicit tokens so the vectoriser can use them
    out.extend('EMO_' + e for e in emoji)

    return ' '.join(out)


# Backwards-compatible alias
clean_sent = clean


if __name__ == '__main__':
    samples = [
        'केपी ओली चोर हो, देश बर्बाद भयो',
        'Balen dai ramro kaam garyo 🔔🔔 jay ghanti',
        'yo manxe ramro chaina',
        'ramro chaina yo manxe',
        'KP Oli चोर ho, dherai naramro',
        'Sala gali garne sabda pani chaina talai',
    ]
    for s in samples:
        print(f'{script_of(s):6s} | {s[:45]:47s} -> {clean(s)}')
