"""
Full End-to-End Evaluation of Experiment OPT-1 (Prioritized Capping) vs V1 Baseline.
Trains LightGBM on Train S1, tests on Val S1 (entity-level split), sweeps threshold, evaluates Macro F0.5.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'business_entity_resolution')))

from src.evaluation.audit_baseline import load_audit_data, normalize_records, execute_full_blocking, BASE
from src.blocking.candidate_generator import cap_candidates
from src.features.pair_features import build_pair_feature_matrix
from src.models.lightgbm_model import LightGBMMatcher
from src.models.threshold import sweep_thresholds, evaluate_predictions
from src.decision import EntityDecisionEngine

def prioritized_cap(blocks_dict, max_per_s1=40):
    weights = {
        'A': 100.0,   # Exact name
        'B': 80.0,    # Core name
        'B2': 60.0,   # Sorted tokens
        'E': 45.0,    # Name char TF-IDF
        'F': 40.0,    # Name word TF-IDF
        'D': 35.0,    # Numeric address
        'G': 30.0,    # Address char TF-IDF
        'C': 20.0,    # Rare tokens
    }
    
    all_s1 = set()
    for b_key in weights:
        all_s1.update(blocks_dict[b_key].keys())
        
    capped = {}
    for sid in all_s1:
        cand_scores = {}
        for b_key, w in weights.items():
            b_cands = blocks_dict[b_key].get(sid, set())
            for cid in b_cands:
                cand_scores[cid] = cand_scores.get(cid, 0.0) + w
                
        if len(cand_scores) <= max_per_s1:
            capped[sid] = set(cand_scores.keys())
        else:
            sorted_cands = sorted(cand_scores.keys(), key=lambda c: (cand_scores[c], c), reverse=True)
            capped[sid] = set(sorted_cands[:max_per_s1])
            
    return capped


def main():
    print("Loading audit data for 1500 S1 sample...", flush=True)
    s1_df, s2s3_df, ground_truth, sampled_s1_ids = load_audit_data(BASE, 1500, seed=42)
    
    # Entity-level split stratified by singletons
    rng_split = np.random.RandomState(42)
    singletons = [sid for sid in sampled_s1_ids if len(ground_truth[sid]) == 0]
    non_singletons = [sid for sid in sampled_s1_ids if len(ground_truth[sid]) > 0]
    val_single = set(rng_split.choice(singletons, max(1, int(len(singletons) * 0.35)), replace=False)) if singletons else set()
    val_non_single = set(rng_split.choice(non_singletons, int(len(non_singletons) * 0.35), replace=False))
    val_s1_ids = val_single | val_non_single
    train_s1_ids = set(sampled_s1_ids) - val_s1_ids
    
    gt_train = {sid: ground_truth[sid] for sid in train_s1_ids}
    gt_val = {sid: ground_truth[sid] for sid in val_s1_ids}
    
    print(f"Split: {len(train_s1_ids)} Train S1, {len(val_s1_ids)} Val S1 ({len(val_single)} val singletons)", flush=True)
    
    print("Normalizing entities...", flush=True)
    s1_dict = normalize_records(s1_df)
    s2s3_dict = normalize_records(s2s3_df)
    
    s1_train_ents = [s1_dict[sid] for sid in train_s1_ids]
    s1_val_ents = [s1_dict[sid] for sid in val_s1_ids]
    s2s3_ents_list = list(s2s3_dict.values())
    
    print("Running blocking passes for train...", flush=True)
    train_blocks = execute_full_blocking(s1_train_ents, s2s3_ents_list)
    print("Running blocking passes for validation...", flush=True)
    val_blocks = execute_full_blocking(s1_val_ents, s2s3_ents_list)
    
    # Compare Configurations
    configs = [
        ("V1 Baseline: Arbitrary Cap @ 40",
         cap_candidates(train_blocks['union_uncapped'], max_per_s1=40),
         cap_candidates(val_blocks['union_uncapped'], max_per_s1=40)),
        ("OPT-1: Prioritized Cap @ 40",
         prioritized_cap(train_blocks, max_per_s1=40),
         prioritized_cap(val_blocks, max_per_s1=40)),
        ("OPT-1: Prioritized Cap @ 50",
         prioritized_cap(train_blocks, max_per_s1=50),
         prioritized_cap(val_blocks, max_per_s1=50)),
    ]
    
    results = []
    
    for config_name, train_cands, val_cands in configs:
        print("\n" + "=" * 60)
        print(f"Evaluating: {config_name}")
        print("=" * 60)
        
        # Build features
        X_train, y_train, train_pairs = build_pair_feature_matrix(train_cands, s1_dict, s2s3_dict, gt_train)
        X_val, y_val, val_pairs = build_pair_feature_matrix(val_cands, s1_dict, s2s3_dict, gt_val)
        
        # Train LightGBM
        matcher = LightGBMMatcher(learning_rate=0.05, num_leaves=31, max_depth=7, n_estimators=400, early_stopping_rounds=30)
        matcher.fit(X_train, y_train, X_val, y_val)
        
        val_probs = matcher.predict_proba(X_val)
        scored_val_pairs = [(sid, cid, float(p)) for (sid, cid), p in zip(val_pairs, val_probs)]
        
        val_scores_by_s1 = {sid: [] for sid in val_s1_ids}
        for sid, cid, p in scored_val_pairs:
            val_scores_by_s1[sid].append((cid, p))
            
        best_thresh, best_metrics, _ = sweep_thresholds(gt_val, scored_val_pairs)
        
        # Decision engine
        engine = EntityDecisionEngine(base_threshold=best_thresh, singleton_cutoff=0.25, multi_match_relative_margin=0.20)
        preds = engine.predict_entities(val_scores_by_s1)
        final_metrics = evaluate_predictions(gt_val, preds)
        
        # Count FP and FN
        fp_ct = sum(len(preds[sid] - gt_val[sid]) for sid in val_s1_ids)
        fn_ct = sum(len(gt_val[sid] - preds[sid]) for sid in val_s1_ids)
        
        results.append({
            'Configuration': config_name,
            'Macro F0.5': final_metrics['macro_f05'],
            'Precision': final_metrics['macro_precision'],
            'Recall': final_metrics['macro_recall'],
            'Singleton F0.5': final_metrics['singleton_f05'],
            'FP': fp_ct,
            'FN': fn_ct,
            'Threshold': round(best_thresh, 3),
            'Candidates/S1': round(len(val_pairs) / len(val_s1_ids), 1),
        })
        print(f"Result: Macro F0.5 = {final_metrics['macro_f05']:.4f} (P={final_metrics['macro_precision']:.4f}, R={final_metrics['macro_recall']:.4f}) | FP={fp_ct}, FN={fn_ct}")
        
    print("\n" + "=" * 80)
    print("FINAL EXPERIMENTAL COMPARISON SUMMARY")
    print("=" * 80)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    # Save results to json & md
    res_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results', 'blocking'))
    os.makedirs(res_dir, exist_ok=True)
    res_df.to_json(os.path.join(res_dir, 'opt1_capping_comparison.json'), indent=2, orient='records')
    res_df.to_markdown(os.path.join(res_dir, 'opt1_capping_comparison.md'), index=False)
    print(f"\nSaved comparison report to: {os.path.join(res_dir, 'opt1_capping_comparison.md')}")

if __name__ == '__main__':
    main()
