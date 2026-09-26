"""
Training and Validation Pipeline for Business Entity Resolution.
Implements:
  - S1-entity-level train/val split (Phase 7)
  - Multi-pass blocking candidate retrieval (Phase 3)
  - Hard negative mining from blocking candidates (Phase 4)
  - Pairwise feature extraction (Phase 5)
  - LightGBM model training with validation early-stopping (Phase 6)
  - Threshold sweeping for Macro F_0.5 optimization (Phase 8)
  - Entity-level margin decision layer (Phase 9)
  - Comprehensive error analysis (Phase 11)
  - Machine-readable experiment tracking (Phase 15)

Usage:
  python code/business_entity_resolution/src/train.py --sample-s1 6000
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any, Optional

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.loading import parse_matched_ids
from src.preprocessing.names import normalize_name
from src.preprocessing.addresses import normalize_address
from src.blocking.exact import block_exact, block_sorted_tokens
from src.blocking.rare_tokens import block_rare_tokens
from src.blocking.numeric import block_numeric
from src.blocking.tfidf import block_tfidf
from src.blocking.candidate_generator import union_candidates, cap_candidates
from src.features.pair_features import extract_pair_features, build_pair_feature_matrix
from src.models.lightgbm_model import LightGBMMatcher
from src.models.threshold import evaluate_predictions, sweep_thresholds
from src.decision import EntityDecisionEngine
from src.error_analysis import analyze_errors, format_error_summary

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def resolve_dataset_dir(custom_path: Optional[str] = None) -> str:
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)
    for candidate in ['dataset', 'Dataset', 'DATASET']:
        p = os.path.join(REPO_ROOT, candidate)
        if os.path.isdir(p):
            return p
    for candidate in [
        '/content/drive/MyDrive/Amazon-ML-Drive/dataset',
        '/content/drive/MyDrive/Amazon-ML-Drive/Dataset',
        '/content/dataset',
        '/content/Dataset',
    ]:
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(REPO_ROOT, 'Dataset')

BASE = resolve_dataset_dir()
EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'experiments'))
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, 'results')
MODELS_DIR = os.path.join(EXPERIMENTS_DIR, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train business entity resolution matcher")
    parser.add_argument('--dataset-dir', type=str, default=None, help="Path to Dataset/ or dataset/ directory")
    parser.add_argument('--sample-s1', type=int, default=6000, help="Number of S1 entities to sample for train+val")
    parser.add_argument('--val-ratio', type=float, default=0.25, help="Ratio of sampled S1 entities for validation")
    parser.add_argument('--max-s2s3-country', type=int, default=30000, help="Negative candidates per country in pool")
    parser.add_argument('--candidate-cap', type=int, default=40, help="Max candidates per S1 passed to matcher")
    parser.add_argument('--exp-name', type=str, default="lgbm_baseline_v1", help="Experiment identifier")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_dataset_sample(base_dir: str, n_s1: int, max_s2s3_per_country: int, seed: int):
    """Load S1 sample and S2/S3 candidate pool with memory-efficient streaming."""
    rng = np.random.RandomState(seed)
    
    print("\n[Step 1] Loading ground truth...", flush=True)
    t0 = time.time()
    gt_path = os.path.join(base_dir, 'train', 'train_ground_truth.tsv')
    s1_with_matches = []
    s1_singletons = []
    gt_dict = {}
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        f.readline()
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
                
    print(f"  Scanned GT in {time.time()-t0:.1f}s. Total S1: {len(gt_dict)}", flush=True)
    
    # Stratified sample: 95% with matches, 5% singletons
    n_with = min(int(n_s1 * 0.95), len(s1_with_matches))
    n_single = min(n_s1 - n_with, len(s1_singletons))
    sampled_s1_ids = list(rng.choice(s1_with_matches, n_with, replace=False)) + \
                     list(rng.choice(s1_singletons, n_single, replace=False))
    rng.shuffle(sampled_s1_ids)
    sampled_s1_set = set(sampled_s1_ids)
    
    needed_s2s3 = set()
    for sid in sampled_s1_set:
        needed_s2s3 |= gt_dict[sid]
    print(f"  Sampled {len(sampled_s1_set)} S1 entities ({len(needed_s2s3)} true matches)", flush=True)
    
    print("\n[Step 2] Extracting S1 rows...", flush=True)
    t0 = time.time()
    s1_rows = []
    s1_path = os.path.join(base_dir, 'train', 'train_source1.tsv')
    for chunk in pd.read_csv(s1_path, sep='\t', chunksize=200000, dtype=str, keep_default_na=False):
        subset = chunk[chunk['entity_id'].isin(sampled_s1_set)]
        if len(subset) > 0:
            s1_rows.append(subset)
        if sum(len(c) for c in s1_rows) >= len(sampled_s1_set):
            break
    s1_df = pd.concat(s1_rows, ignore_index=True)
    countries = set(s1_df['country'].unique())
    print(f"  Loaded {len(s1_df)} S1 rows in {time.time()-t0:.1f}s", flush=True)
    
    print("\n[Step 3] Extracting S2 & S3 candidate pool...", flush=True)
    t0 = time.time()
    s2s3_rows = []
    country_counts = {c: 0 for c in countries}
    
    for src_num in (2, 3):
        src_path = os.path.join(base_dir, 'train', f'train_source{src_num}.tsv')
        for chunk in pd.read_csv(src_path, sep='\t', chunksize=200000, dtype=str, keep_default_na=False):
            m_chunk = chunk[chunk['entity_id'].isin(needed_s2s3)]
            if len(m_chunk) > 0:
                s2s3_rows.append(m_chunk)
            for c in countries:
                if country_counts[c] < max_s2s3_per_country:
                    c_neg = chunk[(chunk['country'] == c) & (~chunk['entity_id'].isin(needed_s2s3))]
                    needed_n = max_s2s3_per_country - country_counts[c]
                    sample_n = min(len(c_neg), min(5000, needed_n))
                    if sample_n > 0:
                        s2s3_rows.append(c_neg.sample(n=sample_n, random_state=rng))
                        country_counts[c] += sample_n
                        
    s2s3_df = pd.concat(s2s3_rows, ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"  Loaded {len(s2s3_df)} S2+S3 pool entities in {time.time()-t0:.1f}s", flush=True)
    
    sample_gt = {sid: gt_dict[sid] for sid in sampled_s1_set}
    return s1_df, s2s3_df, sample_gt, sampled_s1_ids


def normalize_records(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Normalize records and return dict mapping entity_id -> entity dict."""
    entities = {}
    for d in df.to_dict('records'):
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


def run_blocking(s1_entities: List[Dict], s2s3_entities: List[Dict], cap: int = 40) -> Dict[str, Set[str]]:
    """Run full multi-pass blocking union and cap."""
    print("  - Running Block A (exact lower)...", flush=True)
    ba = block_exact(s1_entities, s2s3_entities, key_field='name_lower')
    print("  - Running Block B (exact core)...", flush=True)
    bb = block_exact(s1_entities, s2s3_entities, key_field='name_core')
    print("  - Running Block B2 (sorted tokens)...", flush=True)
    bb2 = block_sorted_tokens(s1_entities, s2s3_entities, tokens_field='name_sorted_tokens')
    print("  - Running Block C (rare tokens)...", flush=True)
    bc = block_rare_tokens(s1_entities, s2s3_entities, tokens_field='name_tokens', min_idf=3.0, max_candidates_per_token=50)
    print("  - Running Block D (numerics)...", flush=True)
    bd = block_numeric(s1_entities, s2s3_entities, numerics_field='addr_numerics', min_shared_numerics=2)
    print("  - Running Block E (char TF-IDF name)...", flush=True)
    be = block_tfidf(s1_entities, s2s3_entities, text_field='name_lower', analyzer='char_wb', ngram_range=(2, 5), top_k=10, min_similarity=0.15)
    print("  - Running Block F (word TF-IDF name)...", flush=True)
    bf = block_tfidf(s1_entities, s2s3_entities, text_field='name_core', analyzer='word', ngram_range=(1, 2), top_k=10, min_similarity=0.15)
    print("  - Running Block G (char TF-IDF address)...", flush=True)
    bg = block_tfidf(s1_entities, s2s3_entities, text_field='addr_normalized', analyzer='char_wb', ngram_range=(2, 5), top_k=10, min_similarity=0.15)
    
    union = union_candidates(ba, bb, bb2, bc, bd, be, bf, bg)
    if cap > 0:
        capped = cap_candidates(union, max_per_s1=cap)
        return capped
    return union


def main():
    args = parse_args()
    t_start = time.time()
    
    print("=" * 70, flush=True)
    print(f"AMAZON ML CHALLENGE — TRAINING & EVALUATION PIPELINE [{args.exp_name}]", flush=True)
    print("=" * 70, flush=True)
    
    dataset_dir = resolve_dataset_dir(args.dataset_dir)
    print(f"\n[Dataset] Resolved dataset directory: {dataset_dir}", flush=True)
    if not os.path.exists(os.path.join(dataset_dir, 'train', 'train_ground_truth.tsv')):
        raise FileNotFoundError(
            f"Could not locate 'train/train_ground_truth.tsv' in '{dataset_dir}'.\n"
            f"Please ensure the dataset is uploaded and specify its location using: --dataset-dir <path_to_dataset>"
        )
    # 1. Load data
    s1_df, s2s3_df, ground_truth, sampled_s1_ids = load_dataset_sample(
        dataset_dir, args.sample_s1, args.max_s2s3_country, args.seed
    )
    
    # 2. Phase 7: Split S1 entities at entity-level (NO pair leakage)
    val_size = int(len(sampled_s1_ids) * args.val_ratio)
    val_s1_ids = set(sampled_s1_ids[:val_size])
    train_s1_ids = set(sampled_s1_ids[val_size:])
    
    gt_train = {sid: ground_truth[sid] for sid in train_s1_ids}
    gt_val = {sid: ground_truth[sid] for sid in val_s1_ids}
    
    print(f"\n[Phase 7] S1-Entity Split: {len(train_s1_ids)} Train S1, {len(val_s1_ids)} Val S1", flush=True)
    
    # 3. Normalize
    print("\n[Phase 2] Normalizing entities...", flush=True)
    t0 = time.time()
    s1_dict = normalize_records(s1_df)
    s2s3_dict = normalize_records(s2s3_df)
    print(f"  Normalized in {time.time()-t0:.1f}s", flush=True)
    
    s1_train_entities = [s1_dict[sid] for sid in train_s1_ids]
    s1_val_entities = [s1_dict[sid] for sid in val_s1_ids]
    s2s3_entities_list = list(s2s3_dict.values())
    
    # 4. Phase 3: Multi-pass blocking independently for train and val
    print("\n[Phase 3] Generating training candidates via blocking...", flush=True)
    t0 = time.time()
    train_candidates = run_blocking(s1_train_entities, s2s3_entities_list, cap=args.candidate_cap)
    print(f"  Generated {sum(len(c) for c in train_candidates.values())} train pairs in {time.time()-t0:.1f}s", flush=True)
    
    print("\n[Phase 3] Generating validation candidates via blocking...", flush=True)
    t0 = time.time()
    val_candidates = run_blocking(s1_val_entities, s2s3_entities_list, cap=args.candidate_cap)
    print(f"  Generated {sum(len(c) for c in val_candidates.values())} val pairs in {time.time()-t0:.1f}s", flush=True)
    
    # Measure blocking recall on validation
    val_total_true = sum(len(v) for v in gt_val.values())
    val_recov_true = sum(len(gt_val[sid] & val_candidates.get(sid, set())) for sid in val_s1_ids)
    val_blocking_recall = float(val_recov_true) / float(val_total_true) if val_total_true > 0 else 0.0
    print(f"\n  Validation Blocking Recall (capped @ {args.candidate_cap}): {val_blocking_recall:.4f} ({val_recov_true}/{val_total_true})", flush=True)
    
    # 5. Phase 5: Pairwise feature matrices
    print("\n[Phase 5] Extracting pairwise features for training...", flush=True)
    t0 = time.time()
    X_train, y_train, train_pairs = build_pair_feature_matrix(train_candidates, s1_dict, s2s3_dict, gt_train)
    pos_ct = int(np.sum(y_train))
    neg_ct = len(y_train) - pos_ct
    print(f"  Train matrix: {X_train.shape[0]} pairs, {X_train.shape[1]} features in {time.time()-t0:.1f}s", flush=True)
    print(f"  Class balance: {pos_ct} positives, {neg_ct} hard negatives (pos ratio: {pos_ct/len(y_train):.3f})", flush=True)
    
    print("\n[Phase 5] Extracting pairwise features for validation...", flush=True)
    t0 = time.time()
    X_val, y_val, val_pairs = build_pair_feature_matrix(val_candidates, s1_dict, s2s3_dict, gt_val)
    print(f"  Val matrix: {X_val.shape[0]} pairs, {X_val.shape[1]} features in {time.time()-t0:.1f}s", flush=True)
    
    # 6. Phase 6: Train LightGBM model
    print("\n[Phase 6] Training LightGBM Matcher...", flush=True)
    t0 = time.time()
    matcher = LightGBMMatcher(
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        early_stopping_rounds=30,
    )
    matcher.fit(X_train, y_train, X_val, y_val)
    print(f"  Model trained in {time.time()-t0:.1f}s", flush=True)
    
    # Feature importances
    top_feats = sorted(matcher.get_feature_importances().items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n  Top 10 Most Important Features:")
    for fn, imp in top_feats:
        print(f"    - {fn:30s}: {imp:.1f}")
        
    # 7. Phase 8: Predict probabilities and sweep thresholds for Macro F_0.5
    print("\n[Phase 8] Predicting validation probabilities and sweeping thresholds...", flush=True)
    val_probs = matcher.predict_proba(X_val)
    
    scored_val_pairs = []
    val_pair_scores_by_s1 = {sid: [] for sid in val_s1_ids}
    scored_pairs_dict = {}
    
    for (sid, cid), prob in zip(val_pairs, val_probs):
        p_val = float(prob)
        scored_val_pairs.append((sid, cid, p_val))
        val_pair_scores_by_s1[sid].append((cid, p_val))
        scored_pairs_dict[(sid, cid)] = p_val
        
    best_thresh, best_metrics, history = sweep_thresholds(gt_val, scored_val_pairs)
    print(f"\n  Optimal Threshold: {best_thresh:.3f}", flush=True)
    print(f"  Macro F_0.5:      {best_metrics['macro_f05']:.4f}", flush=True)
    print(f"  Macro Precision:  {best_metrics['macro_precision']:.4f}", flush=True)
    print(f"  Macro Recall:     {best_metrics['macro_recall']:.4f}", flush=True)
    print(f"  Singleton F_0.5:  {best_metrics['singleton_f05']:.4f} (on {best_metrics['singleton_count']} singletons)", flush=True)
    
    # 8. Phase 9: Apply Entity Decision Layer
    print("\n[Phase 9] Applying Entity-Level Decision Engine...", flush=True)
    decision_engine = EntityDecisionEngine(
        base_threshold=best_thresh,
        singleton_cutoff=0.25,
        multi_match_relative_margin=0.20,
    )
    final_val_predictions = decision_engine.predict_entities(val_pair_scores_by_s1)
    decision_metrics = evaluate_predictions(gt_val, final_val_predictions)
    print(f"  Post-Decision Macro F_0.5:     {decision_metrics['macro_f05']:.4f}", flush=True)
    print(f"  Post-Decision Macro Precision: {decision_metrics['macro_precision']:.4f}", flush=True)
    print(f"  Post-Decision Macro Recall:    {decision_metrics['macro_recall']:.4f}", flush=True)
    print(f"  Post-Decision Singleton F_0.5: {decision_metrics['singleton_f05']:.4f}", flush=True)
    
    # 9. Phase 11: Error Analysis
    print("\n[Phase 11] Running validation error analysis...", flush=True)
    err_report = analyze_errors(
        gt_val, final_val_predictions, val_candidates, scored_pairs_dict, s1_dict, s2s3_dict
    )
    print(format_error_summary(err_report), flush=True)
    
    # 10. Phase 15: Save experiment record
    runtime_total = round(time.time() - t_start, 1)
    exp_record = {
        'experiment': args.exp_name,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'sample_s1_total': args.sample_s1,
        'train_s1_count': len(train_s1_ids),
        'val_s1_count': len(val_s1_ids),
        'val_blocking_recall': round(val_blocking_recall, 4),
        'candidate_cap': args.candidate_cap,
        'model': 'LightGBM',
        'optimal_threshold': round(best_thresh, 3),
        'raw_macro_f05': round(best_metrics['macro_f05'], 4),
        'macro_f05': round(decision_metrics['macro_f05'], 4),
        'macro_precision': round(decision_metrics['macro_precision'], 4),
        'macro_recall': round(decision_metrics['macro_recall'], 4),
        'singleton_f05': round(decision_metrics['singleton_f05'], 4),
        'non_singleton_f05': round(decision_metrics['non_singleton_f05'], 4),
        'val_tp': err_report['tp_count'],
        'val_fp': err_report['fp_count'],
        'val_fn': err_report['fn_count'],
        'runtime_sec': runtime_total,
    }
    
    results_file = os.path.join(RESULTS_DIR, f"{args.exp_name}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(exp_record, f, indent=2)
    print(f"\nExperiment record written to: {results_file}", flush=True)
    
    # Save model
    model_path = os.path.join(MODELS_DIR, f"{args.exp_name}.joblib")
    matcher.save(model_path)
    print(f"Model saved to: {model_path}", flush=True)
    print("\nTraining & evaluation completed successfully!", flush=True)


if __name__ == '__main__':
    main()
