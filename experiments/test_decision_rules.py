"""
Experiment OPT-2: Decision Engine Optimization.
Tests different decision rules:
  1. K1: Pure Global Threshold (no margin filtering)
  2. K2: Threshold + Strict Singleton Guard (top score must reach threshold)
  3. K3: Adaptive Margin (relative margin 0.18 only applied if cand_score < 0.70)
  4. K4: Relaxed Margin (relative margin 0.35)
  5. K5: V1 Margin (relative margin 0.18, baseline)
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
from src.models.threshold import sweep_thresholds, evaluate_predictions

class AdaptiveDecisionEngine:
    def __init__(self, base_threshold=0.82, rule='k1'):
        self.base_threshold = base_threshold
        self.rule = rule
        
    def decide(self, candidate_scores):
        if not candidate_scores:
            return set()
        cands = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        top_cand, top_score = cands[0]
        
        if top_score < self.base_threshold:
            return set()
            
        above_thresh = [(cid, s) for cid, s in cands if s >= self.base_threshold]
        if len(above_thresh) <= 1:
            return {above_thresh[0][0]} if above_thresh else set()
            
        if self.rule == 'k1_pure_threshold':
            return {cid for cid, s in above_thresh}
        elif self.rule == 'k3_adaptive_margin_70':
            # Candidates >= 0.70 are always kept if above threshold; below 0.70 require top_score - 0.20
            res = set()
            for cid, s in above_thresh:
                if s >= 0.70 or s >= (top_score - 0.20):
                    res.add(cid)
            return res
        elif self.rule == 'k4_relaxed_margin_35':
            floor = max(self.base_threshold, top_score - 0.35)
            return {cid for cid, s in above_thresh if s >= floor}
        elif self.rule == 'k5_v1_baseline_margin_18':
            floor = max(self.base_threshold, top_score - 0.18)
            return {cid for cid, s in above_thresh if s >= floor}
        return {cid for cid, s in above_thresh}


def main():
    print("Loading audit data for 1500 S1 sample...", flush=True)
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
    scored_val_pairs = [(sid, cid, float(p)) for (sid, cid), p in zip(val_pairs, val_probs)]
    
    val_scores_by_s1 = {sid: [] for sid in val_s1_ids}
    for sid, cid, p in scored_val_pairs:
        val_scores_by_s1[sid].append((cid, p))
        
    best_thresh, _, _ = sweep_thresholds(gt_val, scored_val_pairs)
    print(f"Optimal Base Threshold: {best_thresh:.3f}\n")
    
    rules = [
        ('k5_v1_baseline_margin_18', 'V1 Baseline: Strict Margin (0.18)'),
        ('k4_relaxed_margin_35', 'Relaxed Margin (0.35)'),
        ('k3_adaptive_margin_70', 'Adaptive Margin (Keep >= 0.70; margin 0.20 for <0.70)'),
        ('k1_pure_threshold', 'Pure Threshold (No margin filtering above thresh)'),
    ]
    
    results = []
    for r_key, r_desc in rules:
        engine = AdaptiveDecisionEngine(base_threshold=best_thresh, rule=r_key)
        preds = {sid: engine.decide(val_scores_by_s1[sid]) for sid in val_s1_ids}
        metrics = evaluate_predictions(gt_val, preds)
        
        fp = sum(len(preds[sid] - gt_val[sid]) for sid in val_s1_ids)
        fn = sum(len(gt_val[sid] - preds[sid]) for sid in val_s1_ids)
        
        results.append({
            'Decision Rule': r_desc,
            'Macro F0.5': metrics['macro_f05'],
            'Precision': metrics['macro_precision'],
            'Recall': metrics['macro_recall'],
            'Singleton F0.5': metrics['singleton_f05'],
            'FP': fp,
            'FN': fn,
        })
        
    res_df = pd.DataFrame(results)
    print("=" * 80)
    print("DECISION LAYER COMPARISON RESULTS")
    print("=" * 80)
    print(res_df.to_string(index=False))
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results', 'decision_layer'))
    os.makedirs(out_dir, exist_ok=True)
    res_df.to_json(os.path.join(out_dir, 'decision_rules_comparison.json'), indent=2, orient='records')
    res_df.to_markdown(os.path.join(out_dir, 'decision_rules_comparison.md'), index=False)
    print(f"\nSaved report to: {os.path.join(out_dir, 'decision_rules_comparison.md')}")

if __name__ == '__main__':
    main()
