"""
Blocks A & B — Exact name matching.
  Block A: exact match on name_lower (normalized lowercase name)
  Block B: exact match on name_core (legal-suffix-stripped name)

Both are country-scoped (Phase 1: 100% of true matches share country).
Uses inverted indexes for O(1) lookup — no Cartesian product.

Expected recall (Phase 1):
  ~15.8% of positive pairs have exact normalized name match (Block A ceiling).
  Block B should capture more by ignoring suffix differences.
"""
from collections import defaultdict


def build_exact_index(entities: list, key_field: str, country_field: str = 'country') -> dict:
    """Build an inverted index: (country, key_value) -> list of entity_ids.
    
    Args:
        entities: list of dicts with at least 'entity_id', key_field, country_field
        key_field: which normalized name field to index on
        country_field: country field name
    
    Returns:
        dict mapping (country, key_value) -> [entity_id, ...]
    """
    index = defaultdict(list)
    for ent in entities:
        key = ent.get(key_field, '')
        country = ent.get(country_field, '')
        if key:  # skip empty keys
            index[(country, key)].append(ent['entity_id'])
    return dict(index)


def block_exact(s1_entities: list, s2s3_entities: list, 
                key_field: str, country_field: str = 'country') -> dict:
    """Run exact blocking: for each S1 entity, find all S2/S3 entities
    with the same (country, key_field) value.
    
    Args:
        s1_entities: list of S1 entity dicts (with normalized fields)
        s2s3_entities: list of S2+S3 entity dicts
        key_field: which field to block on (e.g. 'name_lower', 'name_core')
        country_field: country field name
    
    Returns:
        dict mapping s1_entity_id -> set of candidate entity_ids
    """
    # Build index on S2+S3
    index = build_exact_index(s2s3_entities, key_field, country_field)
    
    candidates = {}
    for ent in s1_entities:
        s1_id = ent['entity_id']
        key = ent.get(key_field, '')
        country = ent.get(country_field, '')
        
        if key:
            matches = index.get((country, key), [])
            candidates[s1_id] = set(matches)
        else:
            candidates[s1_id] = set()
    
    return candidates


def block_sorted_tokens(s1_entities: list, s2s3_entities: list,
                        tokens_field: str = 'name_sorted_tokens',
                        country_field: str = 'country') -> dict:
    """Block on sorted token tuple — catches word reordering.
    e.g., 'Horizon Agricultural Enterprises' == 'Enterprises Horizon Agricultural'
    
    Args:
        s1_entities: list of S1 entity dicts
        s2s3_entities: list of S2+S3 entity dicts
        tokens_field: which sorted-tokens field to use
        country_field: country field name
    
    Returns:
        dict mapping s1_entity_id -> set of candidate entity_ids
    """
    # Build index on S2+S3: (country, sorted_token_tuple) -> [entity_ids]
    index = defaultdict(list)
    for ent in s2s3_entities:
        tokens = tuple(ent.get(tokens_field, []))
        country = ent.get(country_field, '')
        if tokens:  # skip empty
            index[(country, tokens)].append(ent['entity_id'])
    index = dict(index)
    
    candidates = {}
    for ent in s1_entities:
        s1_id = ent['entity_id']
        tokens = tuple(ent.get(tokens_field, []))
        country = ent.get(country_field, '')
        
        if tokens:
            matches = index.get((country, tokens), [])
            candidates[s1_id] = set(matches)
        else:
            candidates[s1_id] = set()
    
    return candidates
