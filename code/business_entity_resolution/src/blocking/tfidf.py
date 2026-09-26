"""
Blocks E, F, G — TF-IDF based blocking.
  Block E: Character n-gram TF-IDF on names (analyzer='char_wb', ngram_range=(2,5))
  Block F: Token/word TF-IDF on names (analyzer='word')
  Block G: Character n-gram TF-IDF on addresses

Uses sklearn's TfidfVectorizer + sparse cosine nearest-neighbor retrieval.
Country-scoped to reduce search space.

This is the primary fuzzy blocking strategy — captures typos, abbreviations,
partial matches, and character-level variations that exact blocks miss.
Critical for the ~80% of positive pairs without exact name matches.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack, csr_matrix
from collections import defaultdict


def _tfidf_block_within_country(
    s1_texts: list,
    s1_ids: list,
    s2s3_texts: list,
    s2s3_ids: list,
    analyzer: str = 'char_wb',
    ngram_range: tuple = (2, 5),
    min_df: int = 2,
    max_df: float = 0.95,
    top_k: int = 10,
    min_similarity: float = 0.1,
) -> dict:
    """Run TF-IDF blocking for one country's worth of entities.
    
    Fits TF-IDF on S2+S3 texts, transforms both S1 and S2+S3,
    then finds top_k most similar S2+S3 candidates per S1 entity.
    
    Args:
        s1_texts: list of text strings for S1 entities
        s1_ids: corresponding S1 entity IDs
        s2s3_texts: list of text strings for S2+S3 entities
        s2s3_ids: corresponding entity IDs
        analyzer: 'char_wb' for character n-grams, 'word' for tokens
        ngram_range: n-gram range for the vectorizer
        min_df: minimum document frequency
        max_df: maximum document frequency
        top_k: number of candidates per S1 entity
        min_similarity: minimum cosine similarity threshold
    
    Returns:
        dict mapping s1_entity_id -> set of candidate entity_ids
    """
    if not s2s3_texts or not s1_texts:
        return {sid: set() for sid in s1_ids}
    
    # Filter out empty texts
    valid_s2s3 = [(text, eid) for text, eid in zip(s2s3_texts, s2s3_ids) if text.strip()]
    valid_s1 = [(text, eid) for text, eid in zip(s1_texts, s1_ids) if text.strip()]
    
    if not valid_s2s3:
        return {sid: set() for sid in s1_ids}
    
    s2s3_t = [t for t, _ in valid_s2s3]
    s2s3_i = [i for _, i in valid_s2s3]
    s1_t = [t for t, _ in valid_s1]
    s1_i = [i for _, i in valid_s1]
    
    # Fit on S2+S3 corpus
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        dtype=np.float32,
    )
    
    try:
        s2s3_matrix = vectorizer.fit_transform(s2s3_t)
    except ValueError:
        # All texts might be too short or min_df too high
        return {sid: set() for sid in s1_ids}
    
    s1_matrix = vectorizer.transform(s1_t)
    
    # Batch cosine similarity — process in chunks to manage memory
    candidates = {sid: set() for sid in s1_ids}
    
    batch_size = 500  # Process 500 S1 entities at a time
    for start in range(0, len(s1_t), batch_size):
        end = min(start + batch_size, len(s1_t))
        batch_matrix = s1_matrix[start:end]
        
        # Sparse dot product for cosine similarity
        sim_matrix = batch_matrix.dot(s2s3_matrix.T)
        
        # Extract top_k per row directly from sparse CSR non-zeros (1000x faster, zero dense memory)
        sim_matrix = sim_matrix.tocsr()
        for local_idx in range(sim_matrix.shape[0]):
            r_start = sim_matrix.indptr[local_idx]
            r_end = sim_matrix.indptr[local_idx + 1]
            n_nonzeros = r_end - r_start
            if n_nonzeros == 0:
                continue
            r_data = sim_matrix.data[r_start:r_end]
            r_indices = sim_matrix.indices[r_start:r_end]
            
            if n_nonzeros <= top_k:
                mask = r_data >= min_similarity
                top_indices = r_indices[mask]
            else:
                part = np.argpartition(r_data, -top_k)[-top_k:]
                valid = r_data[part] >= min_similarity
                top_indices = r_indices[part[valid]]
            
            s1_id = s1_i[start + local_idx]
            for idx in top_indices:
                candidates[s1_id].add(s2s3_i[idx])
    
    return candidates


def block_tfidf(s1_entities: list, s2s3_entities: list,
                text_field: str,
                country_field: str = 'country',
                analyzer: str = 'char_wb',
                ngram_range: tuple = (2, 5),
                min_df: int = 2,
                max_df: float = 0.95,
                top_k: int = 10,
                min_similarity: float = 0.1) -> dict:
    """TF-IDF blocking, country-scoped.
    
    Groups entities by country, then runs TF-IDF retrieval within each country.
    
    Args:
        s1_entities: S1 entity dicts (must have 'entity_id', text_field, country_field)
        s2s3_entities: S2+S3 entity dicts
        text_field: which text field to vectorize (e.g. 'name_lower', 'addr_normalized')
        country_field: country field
        analyzer: 'char_wb' for char n-grams, 'word' for word tokens
        ngram_range: n-gram range
        min_df: min document frequency
        max_df: max document frequency (fraction)
        top_k: candidates per S1 entity
        min_similarity: minimum cosine similarity
    
    Returns:
        dict mapping s1_entity_id -> set of candidate entity_ids
    """
    # Group by country
    s1_by_country = defaultdict(lambda: ([], []))
    for ent in s1_entities:
        country = ent.get(country_field, '')
        texts, ids = s1_by_country[country]
        texts.append(ent.get(text_field, ''))
        ids.append(ent['entity_id'])
    
    s2s3_by_country = defaultdict(lambda: ([], []))
    for ent in s2s3_entities:
        country = ent.get(country_field, '')
        texts, ids = s2s3_by_country[country]
        texts.append(ent.get(text_field, ''))
        ids.append(ent['entity_id'])
    
    all_candidates = {}
    
    for country in s1_by_country:
        s1_texts, s1_ids = s1_by_country[country]
        s2s3_texts, s2s3_ids = s2s3_by_country.get(country, ([], []))
        
        country_candidates = _tfidf_block_within_country(
            s1_texts, s1_ids, s2s3_texts, s2s3_ids,
            analyzer=analyzer, ngram_range=ngram_range,
            min_df=min_df, max_df=max_df,
            top_k=top_k, min_similarity=min_similarity,
        )
        all_candidates.update(country_candidates)
    
    return all_candidates
