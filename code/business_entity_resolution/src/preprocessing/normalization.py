"""
Base normalization utilities shared across name and address processing.
Design: never destroy raw values — produce multiple representations.
"""
import re
import unicodedata


def unicode_normalize(s: str) -> str:
    """NFC normalization + collapse whitespace."""
    if not s:
        return ''
    s = unicodedata.normalize('NFC', s)
    s = ' '.join(s.split())  # collapse all whitespace
    return s.strip()


def to_lower(s: str) -> str:
    """Lowercase after unicode normalization."""
    return unicode_normalize(s).lower()


def to_alphanumeric(s: str) -> str:
    """Strip everything except letters, digits, and spaces. Lowercase."""
    s = to_lower(s)
    # Keep unicode letters (handles Indic scripts), digits, spaces
    s = re.sub(r'[^\w\s]', ' ', s, flags=re.UNICODE)
    s = ' '.join(s.split())
    return s


def to_compact(s: str) -> str:
    """Remove ALL whitespace and punctuation, lowercase. 
    Useful for exact-match blocking on names like 'B+RetailInc' == 'b+retailinc'."""
    s = to_lower(s)
    s = re.sub(r'[\s\-_.,;:!?\'"()\[\]{}/@#$%^&*+=|\\<>~`]', '', s)
    return s


def extract_numeric_tokens(s: str) -> list:
    """Extract all sequences of digits from a string.
    Returns list of digit strings, preserving order."""
    if not s:
        return []
    return re.findall(r'\d+', s)


def is_latin(s: str) -> bool:
    """Check if string is predominantly Latin script (ASCII letters + common accents).
    Returns True if >50% of letter characters are Latin."""
    if not s:
        return True
    letters = [c for c in s if unicodedata.category(c).startswith('L')]
    if not letters:
        return True
    latin_count = sum(1 for c in letters if ord(c) < 0x0250)  # Basic Latin + Latin Extended-A
    return latin_count > len(letters) * 0.5
