"""
Blocking evaluation — test each block individually on a training subsample.

Per instructions Section 6: evaluate each block's recall and candidate count
on its own before combining. Uses a small S1-entity-level validation split.

Usage: python code/business_entity_resolution/src/blocking_eval.py
"""
import sys
import os
import json
import time
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.loading import load_tsv, parse_matched_ids
from src.preprocessing.names import normalize_name
from src.preprocessing.addresses import normalize_address
from src.blocking.exact import block_exact, block_sorted_tokens
from src.blocking.rare_tokens import block_rare_tokens
from src.blocking.numeric import block_numeric
from src.blocking.tfidf import block_tfidf
from src.blocking.candidate_generator import (
    union_candidates, cap_candidates, evaluate_blocking, format_blocking_report
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Dataset'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'experiments', 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== Configuration =====
# Sample size: use a small subset for fast iteration
N_SAMPLE_S1 = 5000       # S1 entities to evaluate on
MAX_S2S3_PER_COUNTRY = 30000  # S2+S3 entities per country (for speed & memory)
SEED = 42

print("=" * 60, flush=True)
print("BLOCKING EVALUATION — TRAINING SUBSAMPLE", flush=True)
print("=" * 60, flush=True)

# ===== 1 & 2. Streaming Data Sampling =====
def load_eval_sample(base_dir, n_sample_s1=5000, max_s2s3_per_country=30000, seed=42):
    rng = np.random.RandomState(seed)
    
    print("\n[1] Scanning ground truth line-by-line...", flush=True)
    gt_path = os.path.join(base_dir, 'train', 'train_ground_truth.tsv')
    s1_with_matches = []
    s1_singletons = []
    gt_dict = {}
    
    t0 = time.time()
    with open(gt_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            s1_id = parts[0]
            raw_matches = parts[1] if len(parts) > 1 else ''
            matched = parse_matched_ids(raw_matches)
            gt_dict[s1_id] = set(matched)
            if len(matched) > 0:
                s1_with_matches.append(s1_id)
            else:
                s1_singletons.append(s1_id)
                
    print(f"  Scanned GT in {time.time()-t0:.1f}s. Entities with matches: {len(s1_with_matches)}, Singletons: {len(s1_singletons)}", flush=True)
    
    n_with = min(int(n_sample_s1 * 0.95), len(s1_with_matches))
    n_single = min(n_sample_s1 - n_with, len(s1_singletons))
    sampled_s1_ids = set(
        list(rng.choice(s1_with_matches, n_with, replace=False)) +
        list(rng.choice(s1_singletons, n_single, replace=False))
    )
    
    sample_gt = {sid: gt_dict[sid] for sid in sampled_s1_ids}
    needed_s2s3_ids = set()
    for sid in sampled_s1_ids:
        needed_s2s3_ids |= sample_gt[sid]
    
    print(f"  Sampled {len(sampled_s1_ids)} S1 entities, {len(needed_s2s3_ids)} true S2/S3 matches", flush=True)
    del gt_dict, s1_with_matches, s1_singletons
    
    print("\n[2] Extracting sampled S1 entities from train_source1...", flush=True)
    t0 = time.time()
    s1_rows = []
    s1_path = os.path.join(base_dir, 'train', 'train_source1.tsv')
    for chunk in pd.read_csv(s1_path, sep='\t', chunksize=200000, dtype=str, keep_default_na=False):
        subset = chunk[chunk['entity_id'].isin(sampled_s1_ids)]
        if len(subset) > 0:
            s1_rows.append(subset)
        if sum(len(c) for c in s1_rows) >= len(sampled_s1_ids):
            break
    s1_df = pd.concat(s1_rows, ignore_index=True)
    countries = set(s1_df['country'].unique())
    print(f"  Loaded {len(s1_df)} S1 rows in {time.time()-t0:.1f}s ({countries})", flush=True)
    
    print("\n[3] Extracting S2 & S3 pool (true matches + negatives)...", flush=True)
    t0 = time.time()
    s2s3_rows = []
    country_counts = {c: 0 for c in countries}
    
    for src_num in (2, 3):
        src_path = os.path.join(base_dir, 'train', f'train_source{src_num}.tsv')
        for chunk in pd.read_csv(src_path, sep='\t', chunksize=200000, dtype=str, keep_default_na=False):
            matches_chunk = chunk[chunk['entity_id'].isin(needed_s2s3_ids)]
            if len(matches_chunk) > 0:
                s2s3_rows.append(matches_chunk)
            
            for c in countries:
                if country_counts[c] < max_s2s3_per_country:
                    c_negatives = chunk[(chunk['country'] == c) & (~chunk['entity_id'].isin(needed_s2s3_ids))]
                    needed_count = max_s2s3_per_country - country_counts[c]
                    sample_n = min(len(c_negatives), min(5000, needed_count))
                    if sample_n > 0:
                        sample_part = c_negatives.sample(n=sample_n, random_state=rng)
                        s2s3_rows.append(sample_part)
                        country_counts[c] += sample_n
                        
    s2s3_df = pd.concat(s2s3_rows, ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"  Loaded {len(s2s3_df)} S2+S3 pool entities in {time.time()-t0:.1f}s", flush=True)
    return s1_df, s2s3_df, sample_gt

s1_sample_df, s2s3_df, sample_gt = load_eval_sample(BASE, N_SAMPLE_S1, MAX_S2S3_PER_COUNTRY, SEED)

# ===== 4. Normalize all entities =====
print("\n[4] Normalizing entities...", flush=True)
t0 = time.time()

def normalize_entity(d):
    """Produce a normalized entity dict from an entity dict."""
    name_fields = normalize_name(d.get('business_name', ''))
    addr_fields = normalize_address(d.get('business_address', ''))
    return {
        'entity_id': d['entity_id'],
        'country': d.get('country', ''),
        **name_fields,
        **addr_fields,
    }

s1_entities = [normalize_entity(d) for d in s1_sample_df.to_dict('records')]
s2s3_entities = [normalize_entity(d) for d in s2s3_df.to_dict('records')]

print(f"  Normalized {len(s1_entities)} S1 + {len(s2s3_entities)} S2/S3 in {time.time()-t0:.1f}s", flush=True)

# ===== 5. Run each block and evaluate =====
print("\n[5] Running blocking passes...")
all_metrics = []

# Block A: Exact normalized name
print("\n  --- Block A: Exact name_lower ---")
t0 = time.time()
block_a = block_exact(s1_entities, s2s3_entities, key_field='name_lower')
m = evaluate_blocking(block_a, sample_gt, 'A_exact_name_lower')
m['runtime_sec'] = round(time.time() - t0, 2)
print(format_blocking_report(m))
all_metrics.append(m)

# Block B: Exact core name (suffix-stripped)
print("\n  --- Block B: Exact name_core ---")
t0 = time.time()
block_b = block_exact(s1_entities, s2s3_entities, key_field='name_core')
m = evaluate_blocking(block_b, sample_gt, 'B_exact_name_core')
m['runtime_sec'] = round(time.time() - t0, 2)
print(format_blocking_report(m))
all_metrics.append(m)

# Block B2: Sorted tokens
print("\n  --- Block B2: Sorted name tokens ---")
t0 = time.time()
block_b2 = block_sorted_tokens(s1_entities, s2s3_entities, tokens_field='name_sorted_tokens')
m = evaluate_blocking(block_b2, sample_gt, 'B2_sorted_tokens')
m['runtime_sec'] = round(time.time() - t0, 2)
print(format_blocking_report(m))
all_metrics.append(m)

# Block C: Rare name tokens
print("\n  --- Block C: Rare name tokens ---")
t0 = time.time()
block_c = block_rare_tokens(s1_entities, s2s3_entities,
                            tokens_field='name_tokens', min_idf=3.0,
                            min_shared_rare=1, max_candidates_per_token=100)
m = evaluate_blocking(block_c, sample_gt, 'C_rare_tokens')
m['runtime_sec'] = round(time.time() - t0, 2)
print(format_blocking_report(m))
all_metrics.append(m)

# Block D: Address numerics
print("\n  --- Block D: Address numerics ---")
t0 = time.time()
block_d = block_numeric(s1_entities, s2s3_entities,
                        numerics_field='addr_numerics', min_shared_numerics=2,
                        min_numeric_length=2, max_candidates_per_numeric=500)
m = evaluate_blocking(block_d, sample_gt, 'D_address_numerics')
m['runtime_sec'] = round(time.time() - t0, 2)
print(format_blocking_report(m))
all_metrics.append(m)

# Block E: Character TF-IDF on names
print("\n  --- Block E: Char TF-IDF name (top 10) ---")
t0 = time.time()
block_e = block_tfidf(s1_entities, s2s3_entities,
                      text_field='name_lower', analyzer='char_wb',
                      ngram_range=(2, 5), min_df=2, top_k=10, min_similarity=0.15)
m = evaluate_blocking(block_e, sample_gt, 'E_char_tfidf_name')
m['runtime_sec'] = round(time.time() - t0, 2)
print(format_blocking_report(m))
all_metrics.append(m)

# Block F: Token TF-IDF on names
print("\n  --- Block F: Word TF-IDF name (top 10) ---")
t0 = time.time()
block_f = block_tfidf(s1_entities, s2s3_entities,
                      text_field='name_core', analyzer='word',
                      ngram_range=(1, 2), min_df=2, top_k=10, min_similarity=0.15)
m = evaluate_blocking(block_f, sample_gt, 'F_word_tfidf_name')
m['runtime_sec'] = round(time.time() - t0, 2)
print(format_blocking_report(m))
all_metrics.append(m)

# Block G: Character TF-IDF on addresses
print("\n  --- Block G: Char TF-IDF address (top 10) ---")
t0 = time.time()
block_g = block_tfidf(s1_entities, s2s3_entities,
                      text_field='addr_normalized', analyzer='char_wb',
                      ngram_range=(2, 5), min_df=2, top_k=10, min_similarity=0.15)
m = evaluate_blocking(block_g, sample_gt, 'G_char_tfidf_addr')
m['runtime_sec'] = round(time.time() - t0, 2)
print(format_blocking_report(m))
all_metrics.append(m)

# ===== 6. Union of all blocks =====
print("\n  --- UNION of all blocks ---")
t0 = time.time()
union_all = union_candidates(block_a, block_b, block_b2, block_c, block_d,
                             block_e, block_f, block_g)
m = evaluate_blocking(union_all, sample_gt, 'UNION_all')
m['runtime_sec'] = round(time.time() - t0, 2)
print(format_blocking_report(m))
all_metrics.append(m)

# Capped union
print("\n  --- UNION capped at 50 ---")
union_capped = cap_candidates(union_all, max_per_s1=50)
m = evaluate_blocking(union_capped, sample_gt, 'UNION_capped_50')
print(format_blocking_report(m))
all_metrics.append(m)

# ===== 7. Save results =====
results_path = os.path.join(RESULTS_DIR, 'blocking_eval_sample.json')
with open(results_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"\n\nResults saved to: {results_path}")
print("Done!")
