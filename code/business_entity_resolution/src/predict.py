"""
Prediction and Submission Generation Pipeline.
Processes country-by-country to operate within memory constraints.
Generates:
  - output/candidate_pairs.tsv
  - output/matching_results.tsv
Then validates using utils/validate_submission.py.

Usage:
  python code/business_entity_resolution/src/predict.py --model-path experiments/models/lgbm_quick_test.joblib
"""
import sys
import os
import gc
import argparse
import time
import subprocess
from typing import Dict, List, Set, Any
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.loading import load_tsv
from src.preprocessing.names import normalize_name
from src.preprocessing.addresses import normalize_address
from src.blocking.exact import block_exact, block_sorted_tokens
from src.blocking.rare_tokens import block_rare_tokens
from src.blocking.numeric import block_numeric
from src.blocking.tfidf import block_tfidf
from src.blocking.candidate_generator import union_candidates, cap_candidates, prioritized_cap_candidates
from src.features.pair_features import extract_pair_features, build_pair_feature_matrix
from src.models.lightgbm_model import LightGBMMatcher
from src.decision import EntityDecisionEngine

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATASET_DIR = os.path.join(BASE, 'Dataset')
OUTPUT_DIR = os.path.join(BASE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate submission files for test set")
    parser.add_argument('--model-path', type=str, required=True, help="Path to trained matcher joblib model")
    parser.add_argument('--test-dir', type=str, default=os.path.join(DATASET_DIR, 'test'), help="Path to test dataset")
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help="Output directory for submission files")
    parser.add_argument('--candidate-cap', type=int, default=40, help="Max candidates per S1 passed to model")
    parser.add_argument('--threshold', type=float, default=0.70, help="Decision threshold")
    parser.add_argument('--limit-s1', type=int, default=None, help="Diagnostic limit on S1 entities (None for full test set)")
    parser.add_argument('--batch-s1', type=int, default=5000, help="Batch size for S1 processing")
    return parser.parse_args()


def normalize_records(records: list) -> dict:
    """Normalize records and return dict mapping entity_id -> entity dict."""
    entities = {}
    for d in records:
        eid = d['entity_id']
        name_fields = normalize_name(d.get('business_name', ''))
        addr_fields = normalize_address(d.get('business_address', ''))
        entities[eid] = {
            'entity_id': eid,
            'country': d.get('country', ''),
            **name_fields,
            **addr_fields,
        }
    return entities


def load_s2s3_for_country(test_dir: str, country: str, max_records: int = None) -> dict:
    """Stream and normalize S2+S3 entities for a specific country."""
    print(f"  [Load S2/S3] Streaming {country} entities...", flush=True)
    t0 = time.time()
    records = []
    
    for src_num in (2, 3):
        path = os.path.join(test_dir, f'test_source{src_num}.tsv')
        for chunk in pd.read_csv(path, sep='\t', chunksize=200000, dtype=str, keep_default_na=False):
            subset = chunk[chunk['country'] == country]
            if len(subset) > 0:
                records.extend(subset.to_dict('records'))
            if max_records and len(records) >= max_records:
                records = records[:max_records]
                break
        if max_records and len(records) >= max_records:
            break
            
    entities = normalize_records(records)
    print(f"  [Load S2/S3] Loaded and normalized {len(entities)} {country} entities in {time.time()-t0:.1f}s", flush=True)
    del records
    gc.collect()
    return entities


def run_country_blocking(s1_entities: list, s2s3_entities: list, cap: int = 40) -> dict:
    """Run full blocking union country-scoped with prioritized consensus capping."""
    ba = block_exact(s1_entities, s2s3_entities, key_field='name_lower')
    bb = block_exact(s1_entities, s2s3_entities, key_field='name_core')
    bb2 = block_sorted_tokens(s1_entities, s2s3_entities, tokens_field='name_sorted_tokens')
    bc = block_rare_tokens(s1_entities, s2s3_entities, tokens_field='name_tokens', min_idf=3.0, max_candidates_per_token=50)
    bd = block_numeric(s1_entities, s2s3_entities, numerics_field='addr_numerics', min_shared_numerics=2)
    be = block_tfidf(s1_entities, s2s3_entities, text_field='name_lower', analyzer='char_wb', ngram_range=(2, 5), top_k=10, min_similarity=0.15)
    bf = block_tfidf(s1_entities, s2s3_entities, text_field='name_core', analyzer='word', ngram_range=(1, 2), top_k=10, min_similarity=0.15)
    bg = block_tfidf(s1_entities, s2s3_entities, text_field='addr_normalized', analyzer='char_wb', ngram_range=(2, 5), top_k=10, min_similarity=0.15)
    
    blocks_dict = {'A': ba, 'B': bb, 'B2': bb2, 'C': bc, 'D': bd, 'E': be, 'F': bf, 'G': bg}
    if cap > 0:
        return prioritized_cap_candidates(blocks_dict, max_per_s1=cap)
    return union_candidates(ba, bb, bb2, bc, bd, be, bf, bg)


def main():
    args = parse_args()
    t_start = time.time()
    
    print("=" * 70, flush=True)
    print("AMAZON ML CHALLENGE — SUBMISSION PREDICTION PIPELINE", flush=True)
    print("=" * 70, flush=True)
    
    # 1. Load trained model
    print(f"\n[1] Loading trained model from: {args.model_path}...", flush=True)
    matcher = LightGBMMatcher.load(args.model_path)
    print("  Model loaded successfully.", flush=True)
    
    # 2. Load test S1
    print(f"\n[2] Loading test S1 entities from: {args.test_dir}...", flush=True)
    t0 = time.time()
    s1_path = os.path.join(args.test_dir, 'test_source1.tsv')
    s1_df = load_tsv(s1_path)
    all_s1_ids = list(s1_df['entity_id'])
    
    if args.limit_s1 is not None:
        s1_df = s1_df.head(args.limit_s1)
        print(f"  DIAGNOSTIC MODE: processing first {len(s1_df)} S1 entities", flush=True)
        
    print(f"  Loaded {len(s1_df)} S1 entities in {time.time()-t0:.1f}s", flush=True)
    
    # Setup output files
    cand_out_path = os.path.join(args.output_dir, 'candidate_pairs.tsv')
    match_out_path = os.path.join(args.output_dir, 'matching_results.tsv')
    
    # If not in limit mode, write headers fresh
    f_cand = open(cand_out_path, 'w', encoding='utf-8')
    f_match = open(match_out_path, 'w', encoding='utf-8')
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    
    decision_engine = EntityDecisionEngine(
        base_threshold=args.threshold,
        singleton_cutoff=0.25,
        multi_match_relative_margin=0.20,
    )
    
    # Group S1 by country
    s1_by_country = {}
    for row in s1_df.to_dict('records'):
        c = row.get('country', '')
        s1_by_country.setdefault(c, []).append(row)
        
    print(f"\n[3] S1 Country Distribution to process: { {c: len(l) for c, l in s1_by_country.items()} }", flush=True)
    
    total_pairs_scored = 0
    total_matches_predicted = 0
    processed_s1_ids = set()
    
    # Process country by country
    for country, s1_records in s1_by_country.items():
        print(f"\n--- Processing Country: {country} ({len(s1_records)} S1 entities) ---", flush=True)
        
        # In diagnostic limit mode, cap S2/S3 pool to speed up testing
        max_s2s3 = 10000 if args.limit_s1 is not None else None
        s2s3_dict = load_s2s3_for_country(args.test_dir, country, max_records=max_s2s3)
        s2s3_entities_list = list(s2s3_dict.values())
        
        # Process S1 in batches
        batch_size = args.batch_s1
        for i in range(0, len(s1_records), batch_size):
            s1_batch_records = s1_records[i:i+batch_size]
            s1_dict = normalize_records(s1_batch_records)
            s1_entities_list = list(s1_dict.values())
            
            # Blocking
            candidates = run_country_blocking(s1_entities_list, s2s3_entities_list, cap=args.candidate_cap)
            
            # Feature extraction
            X_batch, _, pair_ids = build_pair_feature_matrix(candidates, s1_dict, s2s3_dict)
            
            # Model scoring
            probs = matcher.predict_proba(X_batch) if len(X_batch) > 0 else np.array([])
            total_pairs_scored += len(probs)
            
            # Group by S1
            s1_scored = {e['entity_id']: [] for e in s1_entities_list}
            for (sid, cid), p in zip(pair_ids, probs):
                s1_scored[sid].append((cid, float(p)))
                
            # Decision engine
            decisions = decision_engine.predict_entities(s1_scored)
            
            # Write immediately to TSV
            for eid in [r['entity_id'] for r in s1_batch_records]:
                processed_s1_ids.add(eid)
                cands = candidates.get(eid, set())
                cand_str = ",".join(sorted(cands))
                f_cand.write(f"{eid}\t{cand_str}\n")
                
                matches = decisions.get(eid, set()) & cands
                total_matches_predicted += len(matches)
                match_str = ",".join(sorted(matches))
                f_match.write(f"{eid}\t{match_str}\n")
                
            f_cand.flush()
            f_match.flush()
            print(f"  Processed {min(i+batch_size, len(s1_records))}/{len(s1_records)} S1 entities in {country}", flush=True)
            
            del s1_dict, s1_entities_list, candidates, X_batch, probs, s1_scored, decisions
            gc.collect()
            
        del s2s3_dict, s2s3_entities_list
        gc.collect()
        
    # If in limit mode, write remaining test S1 entities as singletons to satisfy full schema validation
    if args.limit_s1 is not None:
        remaining_s1 = [sid for sid in all_s1_ids if sid not in processed_s1_ids]
        print(f"\n  Writing remaining {len(remaining_s1)} S1 entities as singletons for full submission validator...", flush=True)
        for sid in remaining_s1:
            f_cand.write(f"{sid}\t\n")
            f_match.write(f"{sid}\t\n")
            
    f_cand.close()
    f_match.close()
    
    print(f"\nWrote submission files:")
    print(f"  - {cand_out_path}")
    print(f"  - {match_out_path}")
    print(f"Total pairs scored: {total_pairs_scored}, Total matches predicted: {total_matches_predicted}")
    
    # 8. Validate submission using utils/validate_submission.py
    print("\n[8] Running submission validator...", flush=True)
    validator_path = os.path.join(BASE, 'utils', 'validate_submission.py')
    cmd = [
        sys.executable,
        validator_path,
        '--matching', match_out_path,
        '--candidate', cand_out_path,
        '--test-dir', args.test_dir,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
        
    print(f"\nPipeline finished in {time.time()-t_start:.1f}s!")


if __name__ == '__main__':
    main()
