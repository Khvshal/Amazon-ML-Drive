"""
Business address normalization.
Produces multiple representations per the instructions (Section 5):
  - Punctuation/whitespace/Unicode normalization
  - Tokenization
  - Numeric extraction (all numbers)
  - Postal-code extraction (generic, not country-specific)
  - House-number extraction
  - Rare-token extraction (delegated to blocking stage via tokens)

Design decisions:
  - No country-specific parsing (Section 5 rule). We extract numerics generically.
  - Phase 1 showed addresses get truncated, reordered, have state abbrev changes,
    and have '#' prefixes on house numbers. All handled here.
  - 'NULL' literal in addresses is treated as empty (Phase 1: '3, NULL, NASHIK').
"""
import re
from .normalization import unicode_normalize, to_lower, to_alphanumeric, extract_numeric_tokens


# Common address abbreviation expansions (bidirectional isn't needed —
# we normalize TO the short form for consistency)
ADDR_ABBREVIATIONS = {
    'avenue': 'ave', 'street': 'st', 'road': 'rd', 'drive': 'dr',
    'boulevard': 'blvd', 'lane': 'ln', 'circle': 'cir', 'court': 'ct',
    'place': 'pl', 'terrace': 'ter', 'highway': 'hwy', 'parkway': 'pkwy',
    'square': 'sq', 'trail': 'trl', 'way': 'way',
    'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw',
    'apartment': 'apt', 'suite': 'ste', 'building': 'bldg', 'floor': 'fl',
    'unit': 'unit', 'room': 'rm', 'department': 'dept',
    # Indian address terms
    'nagar': 'nagar', 'marg': 'marg', 'road': 'rd', 'colony': 'colony',
    # State full names → abbreviations (US)
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
    'california': 'ca', 'colorado': 'co', 'connecticut': 'ct',
    'delaware': 'de', 'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi',
    'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me',
    'maryland': 'md', 'massachusetts': 'ma', 'michigan': 'mi',
    'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo',
    'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm',
    'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd',
    'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd',
    'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
    'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
    'wisconsin': 'wi', 'wyoming': 'wy',
    'district of columbia': 'dc',
    # Indian state abbreviations
    'andhra pradesh': 'ap', 'arunachal pradesh': 'ar',
    'assam': 'as', 'bihar': 'br', 'chhattisgarh': 'cg',
    'goa': 'ga', 'gujarat': 'gj', 'haryana': 'hr',
    'himachal pradesh': 'hp', 'jharkhand': 'jh',
    'karnataka': 'ka', 'kerala': 'kl', 'madhya pradesh': 'mp',
    'maharashtra': 'mh', 'manipur': 'mn', 'meghalaya': 'ml',
    'mizoram': 'mz', 'nagaland': 'nl', 'odisha': 'or',
    'punjab': 'pb', 'rajasthan': 'rj', 'sikkim': 'sk',
    'tamil nadu': 'tn', 'telangana': 'tg', 'tripura': 'tr',
    'uttar pradesh': 'up', 'uttarakhand': 'uk',
    'west bengal': 'wb', 'delhi': 'dl',
}

# Pre-compile multi-word abbreviations first, then single-word
_MULTI_WORD_ABBREVS = {k: v for k, v in ADDR_ABBREVIATIONS.items() if ' ' in k}
_SINGLE_WORD_ABBREVS = {k: v for k, v in ADDR_ABBREVIATIONS.items() if ' ' not in k}


def _apply_abbreviations(text: str) -> str:
    """Apply address abbreviation normalization."""
    # Multi-word first (e.g., 'new york' → 'ny' before 'new' gets touched)
    for full, abbr in _MULTI_WORD_ABBREVS.items():
        text = re.sub(r'\b' + re.escape(full) + r'\b', abbr, text)
    # Single-word
    tokens = text.split()
    normalized = []
    for token in tokens:
        normalized.append(_SINGLE_WORD_ABBREVS.get(token, token))
    return ' '.join(normalized)


def normalize_address(raw_address: str) -> dict:
    """Produce all address representations from a raw business address.
    
    Returns dict with keys:
        addr_raw, addr_lower, addr_normalized, addr_alphanumeric,
        addr_tokens, addr_sorted_tokens, addr_numerics, addr_compact
    """
    if not raw_address or str(raw_address).strip() == '' or str(raw_address).strip().upper() == 'NULL':
        return {
            'addr_raw': '',
            'addr_lower': '',
            'addr_normalized': '',
            'addr_alphanumeric': '',
            'addr_tokens': [],
            'addr_sorted_tokens': [],
            'addr_numerics': [],
            'addr_compact': '',
        }
    
    raw = str(raw_address).strip()
    lower = to_lower(raw)
    
    # Remove '#' prefix on house numbers (Phase 1 finding)
    cleaned = re.sub(r'#(\d)', r'\1', lower)
    # Remove 'null' literal tokens
    cleaned = re.sub(r'\bnull\b', '', cleaned, flags=re.IGNORECASE)
    # Normalize punctuation: remove commas between address parts, keep useful chars
    cleaned = re.sub(r'[,;]+', ' ', cleaned)
    cleaned = ' '.join(cleaned.split())
    
    # Apply standard abbreviation normalization
    normalized = _apply_abbreviations(cleaned)
    normalized = ' '.join(normalized.split())
    
    alphanum = to_alphanumeric(raw)
    
    tokens = normalized.split() if normalized else []
    sorted_tokens = sorted(tokens)
    
    # Extract all numeric sequences (house numbers, postal codes, etc.)
    numerics = extract_numeric_tokens(raw)
    
    # Compact: no whitespace, no punctuation
    compact = re.sub(r'[^a-z0-9]', '', normalized)
    
    return {
        'addr_raw': raw,
        'addr_lower': lower,
        'addr_normalized': normalized,
        'addr_alphanumeric': alphanum,
        'addr_tokens': tokens,
        'addr_sorted_tokens': sorted_tokens,
        'addr_numerics': numerics,
        'addr_compact': compact,
    }
