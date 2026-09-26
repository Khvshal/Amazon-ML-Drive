"""
Business name normalization.
Produces multiple representations per the instructions (Section 5):
  name_raw, name_lower, name_unicode_normalized, name_alphanumeric,
  name_core (legal-suffix stripped), name_tokens, name_sorted_tokens, name_compact.

Design decisions:
  - Legal suffix list is intentionally broad (Corp, LLC, Ltd, Pvt, Inc, etc.)
    because Phase 1 showed heavy suffix variation in positive pairs.
  - We do NOT transliterate Indic → Latin here; that's a blocking-time concern.
    Instead, we keep all representations so blocking can choose which to use.
  - 'and' ↔ '&' normalization is applied in name_core since Phase 1 showed
    positive pairs with this variation ('Aarogya & Co' etc.).
"""
import re
from .normalization import unicode_normalize, to_lower, to_alphanumeric, to_compact

# Legal suffixes to strip for name_core.
# Ordered longest-first to avoid partial matches.
LEGAL_SUFFIXES = [
    # English
    'private limited', 'pvt limited', 'pvt ltd', 'pvt. ltd.', 'pvt. ltd',
    'private ltd', 'priv ltd', 'p ltd',
    'limited liability company', 'limited liability partnership',
    'limited partnership', 'limited',
    'incorporated', 'incorporation',
    'corporation', 'corp',
    'enterprises', 'enterprise',
    'associates', 'associate',
    'technologies', 'technology',
    'solutions', 'solution',
    'industries', 'industry',
    'holdings', 'holding',
    'partners', 'partner',
    'services', 'service',
    'ventures', 'venture',
    'consulting', 'consultants', 'consultant',
    'international', 'intl',
    'company', 'co.',
    'llc', 'llp', 'l.l.c.', 'l.l.c', 'l.l.p.',
    'inc.', 'inc', 'ltd.', 'ltd', 'pvt.', 'pvt',
    'pllc', 'p.l.l.c.',
    'p.c.', 'pc',
    'plc', 'p.l.c.',
    'ngo', 'trust', 'foundation',
    'group', 'the',
    # French (for test set France entities)
    'sarl', 'sas', 'sa', 'eurl', 'sasu', 'sci', 'snc',
    'société', 'societe', 'cie',
]

# Pre-compile a pattern that matches any suffix at the end of a string
# after optional punctuation/whitespace
_SUFFIX_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(s) for s in LEGAL_SUFFIXES) + r')\.?\s*$',
    re.IGNORECASE
)


def strip_legal_suffix(name: str) -> str:
    """Iteratively strip legal suffixes from the end of a name.
    Returns the core business name."""
    if not name:
        return ''
    result = name.strip()
    # Strip iteratively (handles 'Foo Pvt Ltd' → 'Foo Pvt' → 'Foo')
    for _ in range(3):  # max 3 passes
        prev = result
        result = _SUFFIX_PATTERN.sub('', result).strip()
        result = result.rstrip('.,;: ')
        if result == prev:
            break
    return result if result else name.strip()  # never return empty


def normalize_ampersand(name: str) -> str:
    """Normalize '&' ↔ 'and' to a consistent form."""
    # Replace '&' with ' and '
    name = re.sub(r'\s*&\s*', ' and ', name)
    return ' '.join(name.split())


def normalize_name(raw_name: str) -> dict:
    """Produce all name representations from a raw business name.
    
    Returns dict with keys:
        name_raw, name_lower, name_unicode_normalized, name_alphanumeric,
        name_core, name_tokens, name_sorted_tokens, name_compact
    """
    if not raw_name or str(raw_name).strip() == '':
        empty = ''
        return {
            'name_raw': '',
            'name_lower': '',
            'name_unicode_normalized': '',
            'name_alphanumeric': '',
            'name_core': '',
            'name_tokens': [],
            'name_sorted_tokens': [],
            'name_compact': '',
        }
    
    raw = str(raw_name).strip()
    lower = to_lower(raw)
    uni_norm = unicode_normalize(raw)
    alphanum = to_alphanumeric(raw)
    
    # Core: strip legal suffixes, normalize &/and, then lowercase
    core = to_lower(normalize_ampersand(strip_legal_suffix(lower)))
    core = core.strip()
    
    # Tokens from the core name (most useful for blocking)
    tokens = core.split() if core else []
    sorted_tokens = sorted(tokens)
    
    compact = to_compact(raw)
    
    return {
        'name_raw': raw,
        'name_lower': lower,
        'name_unicode_normalized': uni_norm,
        'name_alphanumeric': alphanum,
        'name_core': core,
        'name_tokens': tokens,
        'name_sorted_tokens': sorted_tokens,
        'name_compact': compact,
    }
