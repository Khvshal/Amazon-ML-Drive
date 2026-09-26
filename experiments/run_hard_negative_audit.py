"""
Run Hard-Negative Analysis for Baseline Audit.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'business_entity_resolution')))

from src.evaluation.audit_baseline import load_audit_data, normalize_records, execute_full_blocking, BASE
from src.blocking.candidate_generator import prioritized_cap_candidates
from src.features.pair_features import build_pair_feature_matrix
from src.models.lightgbm_model import LightGBMMatcher

def main():
    s1_df, s2s3_df, ground_truth, sampled_s1_ids = load_audit_data(BASE, 1500, seed=42)
    
    rng_split = np.random.RandomState(42)
    singletons = [sid for sid in sampled_s1_ids if len(ground_truth[sid]) == 0]
    non_singletons = [sid for sid in sampled_s1_ids if len(ground_truth[sid]) > 0]
    val_single = set(rng_split.choice(singletons, max(1, int(len(singletons) * 0.35)), replace=False)) if singletons else set()
    val_non_single = set(rng_split.choice(non_singletons, int(len(non_singletons) * 0.35), replace=False))
    val_s1_ids = val_single | val_non_single
    train_s1_ids = set(sampled_s1_ids) - val_s1_ids
    
    gt_train = {sid: ground_truth[sid] for sid in train_s1_ids}
    gt_val = {sid: ground_truth[sid] for sid in val_s1_ids}
    
    s1_dict = normalize_records(s1_df)
    s2s3_dict = normalize_records(s2s3_df)
    
    train_blocks = execute_full_blocking([s1_dict[sid] for sid in train_s1_ids], list(s2s3_dict.values()))
    val_blocks = execute_full_blocking([s1_dict[sid] for sid in val_s1_ids], list(s2s3_dict.values()))
    
    train_cands = prioritized_cap_candidates(train_blocks, max_per_s1=40)
    val_cands = prioritized_cap_candidates(val_blocks, max_per_s1=40)
    
    X_train, y_train, train_pairs = build_pair_feature_matrix(train_cands, s1_dict, s2s3_dict, gt_train)
    X_val, y_val, val_pairs = build_pair_feature_matrix(val_cands, s1_dict, s2s3_dict, gt_val)
    
    matcher = LightGBMMatcher(learning_rate=0.05, num_leaves=31, max_depth=7, n_estimators=400, early_stopping_rounds=30)
    matcher.fit(X_train, y_train, X_val, y_val)
    
    val_probs = matcher.predict_proba(X_val)
    
    # Analyze negative pairs (y_val == 0)
    neg_indices = np.where(y_val == 0)[0]
    neg_scores = val_probs[neg_indices]
    
    print("=" * 60)
    print("HARD NEGATIVE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Total Negative Pairs evaluated: {len(neg_indices)}")
    print(f"  Score < 0.05:  {np.sum(neg_scores < 0.05):6d} ({100*np.mean(neg_scores < 0.05):.2f}%) [Easy Negatives]")
    print(f"  Score 0.05-0.2:{np.sum((neg_scores >= 0.05) & (neg_scores < 0.20)):6d} ({100*np.mean((neg_scores >= 0.05) & (neg_scores < 0.20)):.2f}%) [Moderate Negatives]")
    print(f"  Score 0.2-0.5: {np.sum((neg_scores >= 0.20) & (neg_scores < 0.50)):6d} ({100*np.mean((neg_scores >= 0.20) & (neg_scores < 0.50)):.2f}%) [Semi-Hard Negatives]")
    print(f"  Score 0.5-0.8: {np.sum((neg_scores >= 0.50) & (neg_scores < 0.80)):6d} ({100*np.mean((neg_scores >= 0.50) & (neg_scores < 0.80)):.2f}%) [Hard Negatives]")
    print(f"  Score >= 0.80: {np.sum(neg_scores >= 0.80):6d} ({100*np.mean(neg_scores >= 0.80):.2f}%) [Extreme False Positives]")
    print(f"Mean score on negatives: {np.mean(neg_scores):.4f}")
    print(f"99th percentile negative score: {np.percentile(neg_scores, 99):.4f}")
    
    # Inspect top 10 hardest negatives
    hard_idx = neg_indices[np.argsort(neg_scores)[::-1][:10]]
    print("\nTop 5 Hardest Negatives:")
    for idx in hard_idx[:5]:
        sid, cid = val_pairs[idx]
        sc = val_probs[idx]
        e1 = s1_dict[sid]
        e2 = s2s3_dict[cid]
        print(f"  Score: {sc:.4f} | S1: '{e1['name_raw']}' @ '{e1['addr_raw']}' vs Cand: '{e2['name_raw']}' @ '{e2['addr_raw']}'")

if __name__ == '__main__':
    main()
