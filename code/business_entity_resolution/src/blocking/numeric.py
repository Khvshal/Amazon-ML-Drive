"""
Block D — Numeric/postal code blocking.
Matches on shared numeric tokens between addresses.
Phase 1 showed numerics are preserved even in truncated addresses
('3, NULL, NASHIK' still has '3'). Country-scoped.

Design: generic numeric overlap, no country-specific postal code parsing.
Uses inverted index on numeric tokens.
"""
from collections import defaultdict


def block_numeric(s1_entities: list, s2s3_entities: list,
                  numerics_field: str = 'addr_numerics',
                  country_field: str = 'country',
                  min_shared_numerics: int = 2,
                  min_numeric_length: int = 2,
                  max_candidates_per_numeric: int = 500) -> dict:
    """Numeric blocking: match on shared address numeric tokens.
    
    For each S1 entity, find S2/S3 entities sharing at least
    `min_shared_numerics` numeric tokens of length >= `min_numeric_length`.
    
    Shorter numerics (1-digit like '3') are ignored by default because
    they're too common. Longer numerics (postal codes, house numbers
    like '26', '3901') are more discriminative.
    
    Args:
        s1_entities: S1 entity dicts
        s2s3_entities: S2+S3 entity dicts
        numerics_field: field containing list of numeric strings
        country_field: country field
        min_shared_numerics: minimum shared numeric tokens to qualify
        min_numeric_length: minimum length of numeric token to consider
        max_candidates_per_numeric: cap per numeric token to avoid explosion
    
    Returns:
        dict mapping s1_entity_id -> set of candidate entity_ids
    """
    # Build inverted index: (country, numeric_string) -> [entity_ids]
    index = defaultdict(list)
    for ent in s2s3_entities:
        country = ent.get(country_field, '')
        numerics = set(ent.get(numerics_field, []))
        for num in numerics:
            if len(num) >= min_numeric_length:
                index[(country, num)].append(ent['entity_id'])
    
    # Filter overly common numerics
    index = {k: v for k, v in index.items() if len(v) <= max_candidates_per_numeric}
    
    candidates = {}
    for ent in s1_entities:
        s1_id = ent['entity_id']
        country = ent.get(country_field, '')
        numerics = set(ent.get(numerics_field, []))
        
        candidate_counts = defaultdict(int)
        for num in numerics:
            if len(num) >= min_numeric_length:
                for cand_id in index.get((country, num), []):
                    candidate_counts[cand_id] += 1
        
        candidates[s1_id] = {
            cid for cid, count in candidate_counts.items()
            if count >= min_shared_numerics
        }
    
    return candidates
