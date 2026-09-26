"""
Block C — Rare token blocking.
Matches S1 entities to S2/S3 entities that share rare name tokens.
Tokens are weighted by inverse document frequency; rare tokens are more
discriminative. This catches partial name matches and typo-resilient matches
where at least one distinctive word is preserved.

Country-scoped. Uses inverted indexes.
"""
from collections import defaultdict
import math


def compute_token_idf(entities: list, tokens_field: str = 'name_tokens',
                      country_field: str = 'country') -> dict:
    """Compute IDF for each token within each country.
    
    Returns:
        dict mapping (country, token) -> idf_score
    """
    # Count documents per country
    country_doc_count = defaultdict(int)
    # Count documents containing each token per country
    token_doc_count = defaultdict(int)
    
    for ent in entities:
        country = ent.get(country_field, '')
        tokens = set(ent.get(tokens_field, []))  # unique tokens per doc
        country_doc_count[country] += 1
        for token in tokens:
            token_doc_count[(country, token)] += 1
    
    idf = {}
    for (country, token), df in token_doc_count.items():
        n = country_doc_count[country]
        idf[(country, token)] = math.log((n + 1) / (df + 1)) + 1  # smoothed IDF
    
    return idf


def build_token_index(entities: list, tokens_field: str = 'name_tokens',
                      country_field: str = 'country',
                      idf: dict = None, min_idf: float = 3.0) -> dict:
    """Build inverted index: (country, token) -> [entity_ids] for rare tokens only.
    
    Args:
        entities: list of entity dicts
        tokens_field: which token field to use
        country_field: country field name
        idf: pre-computed IDF dict; if None, index all tokens
        min_idf: minimum IDF to consider a token "rare" enough to index
    
    Returns:
        dict mapping (country, token) -> [entity_id, ...]
    """
    index = defaultdict(list)
    for ent in entities:
        country = ent.get(country_field, '')
        tokens = set(ent.get(tokens_field, []))
        for token in tokens:
            if len(token) < 2:  # skip single-char tokens
                continue
            if idf is not None:
                token_idf = idf.get((country, token), 0)
                if token_idf < min_idf:
                    continue
            index[(country, token)].append(ent['entity_id'])
    return dict(index)


def block_rare_tokens(s1_entities: list, s2s3_entities: list,
                      tokens_field: str = 'name_tokens',
                      country_field: str = 'country',
                      min_idf: float = 3.0,
                      min_shared_rare: int = 1,
                      max_candidates_per_token: int = 100) -> dict:
    """Rare-token blocking: match on shared rare name tokens.
    
    For each S1 entity, find S2/S3 entities that share at least
    `min_shared_rare` rare tokens (tokens with IDF >= min_idf).
    
    Tokens that appear in too many entities (> max_candidates_per_token)
    are skipped to avoid explosion from common-but-technically-rare tokens.
    
    Args:
        s1_entities: S1 entity dicts
        s2s3_entities: S2+S3 entity dicts
        tokens_field: which token field to use
        country_field: country field
        min_idf: minimum IDF threshold
        min_shared_rare: minimum number of shared rare tokens to be a candidate
        max_candidates_per_token: skip tokens that index more than this many entities
    
    Returns:
        dict mapping s1_entity_id -> set of candidate entity_ids
    """
    # Compute IDF on the combined S2+S3 corpus (that's what we're searching)
    idf = compute_token_idf(s2s3_entities, tokens_field, country_field)
    
    # Build inverted index on S2+S3 for rare tokens only
    raw_index = build_token_index(s2s3_entities, tokens_field, country_field, idf, min_idf)
    
    # Filter out overly-common tokens
    index = {k: v for k, v in raw_index.items() if len(v) <= max_candidates_per_token}
    
    candidates = {}
    for ent in s1_entities:
        s1_id = ent['entity_id']
        country = ent.get(country_field, '')
        tokens = set(ent.get(tokens_field, []))
        
        # Collect candidates from each rare token
        candidate_counts = defaultdict(int)
        for token in tokens:
            if len(token) < 2:
                continue
            token_idf = idf.get((country, token), 0)
            if token_idf < min_idf:
                continue
            for cand_id in index.get((country, token), []):
                candidate_counts[cand_id] += 1
        
        # Keep candidates with enough shared rare tokens
        candidates[s1_id] = {
            cid for cid, count in candidate_counts.items()
            if count >= min_shared_rare
        }
    
    return candidates
