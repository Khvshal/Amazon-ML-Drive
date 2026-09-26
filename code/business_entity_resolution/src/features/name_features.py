"""
Pairwise name feature extraction.
Computes string similarity, token overlap, edit distances, and length metrics
between S1 and candidate entity names.
"""
from typing import Dict, Any, List
import rapidfuzz.distance.Levenshtein as lev
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.fuzz as fuzz


def _get_char_ngrams(s: str, n: int = 3) -> set:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def _jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return float(intersection) / float(union) if union > 0 else 0.0


def _dice(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    intersection = len(s1 & s2)
    total = len(s1) + len(s2)
    return (2.0 * intersection) / float(total) if total > 0 else 0.0


def extract_name_features(e1: Dict[str, Any], e2: Dict[str, Any]) -> Dict[str, float]:
    """Extract pairwise name features between entity 1 (S1) and entity 2 (candidate).
    
    Args:
        e1: S1 normalized entity dict
        e2: S2/S3 candidate normalized entity dict
        
    Returns:
        Dict of numeric features
    """
    raw1 = e1.get('name_raw', '') or ''
    raw2 = e2.get('name_raw', '') or ''
    lower1 = e1.get('name_lower', '') or ''
    lower2 = e2.get('name_lower', '') or ''
    core1 = e1.get('name_core', '') or ''
    core2 = e2.get('name_core', '') or ''
    sorted1 = e1.get('name_sorted_tokens', []) or []
    sorted2 = e2.get('name_sorted_tokens', []) or []
    
    tokens1 = set(e1.get('name_tokens', []) or [])
    tokens2 = set(e2.get('name_tokens', []) or [])
    
    # Missingness flags
    empty1 = int(len(lower1.strip()) == 0)
    empty2 = int(len(lower2.strip()) == 0)
    
    # Exact equalities
    exact_raw = float(raw1 == raw2 and not empty1)
    exact_lower = float(lower1 == lower2 and not empty1)
    exact_core = float(core1 == core2 and len(core1.strip()) > 0)
    exact_sorted = float(sorted1 == sorted2 and len(sorted1) > 0)
    
    # String edit similarities (normalized 0.0 to 1.0)
    lev_ratio = float(lev.normalized_similarity(lower1, lower2)) if not (empty1 and empty2) else 0.0
    jw_sim = float(jw.similarity(lower1, lower2)) if not (empty1 and empty2) else 0.0
    token_sort = float(fuzz.token_sort_ratio(lower1, lower2)) / 100.0 if not (empty1 and empty2) else 0.0
    token_set = float(fuzz.token_set_ratio(lower1, lower2)) / 100.0 if not (empty1 and empty2) else 0.0
    
    # Core name similarities (suffix-stripped)
    core_lev = float(lev.normalized_similarity(core1, core2)) if core1 and core2 else 0.0
    core_jw = float(jw.similarity(core1, core2)) if core1 and core2 else 0.0
    
    # N-gram overlap
    ngrams1 = _get_char_ngrams(lower1, 3)
    ngrams2 = _get_char_ngrams(lower2, 3)
    char_jaccard = _jaccard(ngrams1, ngrams2)
    char_dice = _dice(ngrams1, ngrams2)
    
    # Token overlap
    token_jaccard = _jaccard(tokens1, tokens2)
    token_dice = _dice(tokens1, tokens2)
    common_tokens = len(tokens1 & tokens2)
    min_len = min(len(tokens1), len(tokens2))
    token_overlap_ratio = float(common_tokens) / float(min_len) if min_len > 0 else 0.0
    
    # Length metrics
    len1 = len(lower1)
    len2 = len(lower2)
    len_diff = float(abs(len1 - len2))
    len_ratio = float(min(len1, len2)) / float(max(len1, len2)) if max(len1, len2) > 0 else 1.0
    token_count_diff = float(abs(len(tokens1) - len(tokens2)))
    
    return {
        'name_exact_raw': exact_raw,
        'name_exact_lower': exact_lower,
        'name_exact_core': exact_core,
        'name_exact_sorted': exact_sorted,
        'name_lev_sim': lev_ratio,
        'name_jw_sim': jw_sim,
        'name_token_sort_ratio': token_sort,
        'name_token_set_ratio': token_set,
        'name_core_lev_sim': core_lev,
        'name_core_jw_sim': core_jw,
        'name_char_jaccard_3gram': char_jaccard,
        'name_char_dice_3gram': char_dice,
        'name_token_jaccard': token_jaccard,
        'name_token_dice': token_dice,
        'name_common_tokens_count': float(common_tokens),
        'name_token_overlap_ratio': token_overlap_ratio,
        'name_len_diff': len_diff,
        'name_len_ratio': len_ratio,
        'name_token_count_diff': token_count_diff,
        'name_both_missing': float(empty1 and empty2),
        'name_one_missing': float((empty1 and not empty2) or (empty2 and not empty1)),
    }
