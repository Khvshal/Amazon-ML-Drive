"""
Pairwise address feature extraction.
Computes string similarity, numeric token overlap, postal code matching,
and missingness signals between S1 and candidate addresses.
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


def extract_address_features(e1: Dict[str, Any], e2: Dict[str, Any]) -> Dict[str, float]:
    """Extract pairwise address features between entity 1 (S1) and entity 2 (candidate).
    
    Args:
        e1: S1 normalized entity dict
        e2: S2/S3 candidate normalized entity dict
        
    Returns:
        Dict of numeric features
    """
    raw1 = e1.get('addr_raw', '') or ''
    raw2 = e2.get('addr_raw', '') or ''
    norm1 = e1.get('addr_normalized', '') or ''
    norm2 = e2.get('addr_normalized', '') or ''
    
    tokens1 = set(e1.get('addr_tokens', []) or [])
    tokens2 = set(e2.get('addr_tokens', []) or [])
    
    nums1 = set(e1.get('addr_numerics', []) or [])
    nums2 = set(e2.get('addr_numerics', []) or [])
    
    empty1 = int(len(norm1.strip()) == 0)
    empty2 = int(len(norm2.strip()) == 0)
    
    # Exact matches
    exact_raw = float(raw1 == raw2 and not empty1)
    exact_norm = float(norm1 == norm2 and not empty1)
    
    # String edit similarities
    lev_ratio = float(lev.normalized_similarity(norm1, norm2)) if not (empty1 or empty2) else 0.0
    jw_sim = float(jw.similarity(norm1, norm2)) if not (empty1 or empty2) else 0.0
    token_sort = float(fuzz.token_sort_ratio(norm1, norm2)) / 100.0 if not (empty1 or empty2) else 0.0
    token_set = float(fuzz.token_set_ratio(norm1, norm2)) / 100.0 if not (empty1 or empty2) else 0.0
    
    # N-gram overlap
    ngrams1 = _get_char_ngrams(norm1, 3)
    ngrams2 = _get_char_ngrams(norm2, 3)
    char_jaccard = _jaccard(ngrams1, ngrams2) if not (empty1 or empty2) else 0.0
    char_dice = _dice(ngrams1, ngrams2) if not (empty1 or empty2) else 0.0
    
    # Token overlap
    token_jaccard = _jaccard(tokens1, tokens2) if not (empty1 or empty2) else 0.0
    token_dice = _dice(tokens1, tokens2) if not (empty1 or empty2) else 0.0
    common_tokens = len(tokens1 & tokens2) if not (empty1 or empty2) else 0
    min_len = min(len(tokens1), len(tokens2)) if not (empty1 or empty2) else 0
    token_overlap_ratio = float(common_tokens) / float(min_len) if min_len > 0 else 0.0
    
    # Numeric components (house numbers, postal codes, pin codes)
    num_jaccard = _jaccard(nums1, nums2) if nums1 and nums2 else 0.0
    num_common = len(nums1 & nums2) if nums1 and nums2 else 0
    has_shared_num = float(num_common > 0)
    exact_nums = float(nums1 == nums2 and len(nums1) > 0)
    
    # Length metrics
    len1 = len(norm1)
    len2 = len(norm2)
    len_diff = float(abs(len1 - len2))
    len_ratio = float(min(len1, len2)) / float(max(len1, len2)) if max(len1, len2) > 0 else (1.0 if empty1 and empty2 else 0.0)
    
    return {
        'addr_exact_raw': exact_raw,
        'addr_exact_norm': exact_norm,
        'addr_lev_sim': lev_ratio,
        'addr_jw_sim': jw_sim,
        'addr_token_sort_ratio': token_sort,
        'addr_token_set_ratio': token_set,
        'addr_char_jaccard_3gram': char_jaccard,
        'addr_char_dice_3gram': char_dice,
        'addr_token_jaccard': token_jaccard,
        'addr_token_dice': token_dice,
        'addr_common_tokens_count': float(common_tokens),
        'addr_token_overlap_ratio': token_overlap_ratio,
        'addr_num_jaccard': num_jaccard,
        'addr_num_common_count': float(num_common),
        'addr_has_shared_num': has_shared_num,
        'addr_exact_numerics': exact_nums,
        'addr_len_diff': len_diff,
        'addr_len_ratio': len_ratio,
        'addr_s2s3_missing': float(empty2),
        'addr_both_missing': float(empty1 and empty2),
    }
