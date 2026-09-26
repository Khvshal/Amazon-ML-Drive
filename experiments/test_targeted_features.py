"""
Experiment OPT-3: Error-Targeted Features Evaluation.
Compares:
  - 51 Baseline Features
  - 54 Features (adding has_empty_addr, name_exact_with_empty_addr, name_addr_contradiction)
Evaluates on the exact same 1500 S1 entity-level stratified split.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'business_entity_resolution')))

from src.evaluation.audit_baseline import load_audit_data, normalize_records, execute_full_blocking, BASE
from src.blocking.candidate_generator import prioritized_cap_candidates
from src.features.pair_features import extract_pair_features
from src.models.lightgbm_model import LightGBMMatcher
from src.models.threshold import sweep_thresholds, evaluate_predictions
from src.decision import EntityDecisionEngine

def extract_features_v2(e1, e2):
    feat = extract_pair_features(e1, e2)
    
    # 3 Error-targeted features
    addr1_len = len(e1.get('addr_normalized', ''))
    addr2_len = len(e2.get('addr_normalized', ''))
    has_empty = float(addr1_len == 0 or addr2_len == 0)
    
    feat['has_empty_addr'] = has_empty
    feat['name_exact_with_empty_addr'] = float(feat['name_exact_core'] > 0 and has_empty > 0)
    feat['name_addr_contradiction'] = float(feat['addr_token_set_ratio'] >= 0.90 and feat['name_token_set_ratio'] <= 0.25)
    return feat

def build_matrix(cand_pairs, s1_dict, s2s3_dict, gt_dict, v2=False):
    rows = []
    y_list = []
    pairs = []
    
    for sid, cands in cand_pairs.items():
        e1 = s1_dict[sid]
        true_set = gt_dict.get(sid, set()) if gt_dict else set()
        for cid in cands:
            e2 = s2s3_dict[cid]
            if v2:
                f = extract_features_v2(e1, e2)
            else:
                f = extract_pair_features(e1, e2)
            rows.append(f)
            y_list.append(1 if cid in true_set else 0)
            pairs.append((sid, cid))
            
    df = pd.DataFrame(rows).fillna(0.0)
    return df, np.array(y_list), pairs


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
    
    results = []
    
    for v2_flag, name in [(False, 'OPT-1: 51 Baseline Features'), (True, 'OPT-3: 54 Targeted Features')]:
        print(f"\nEvaluating: {name}...")
        X_train, y_train, train_pairs = build_matrix(train_cands, s1_dict, s2s3_dict, gt_train, v2=v2_flag)
        X_val, y_val, val_pairs = build_matrix(val_cands, s1_dict, s2s3_dict, gt_val, v2=v2_flag)
        
        matcher = LightGBMMatcher(learning_rate=0.05, num_leaves=31, max_depth=7, n_estimators=400, early_stopping_rounds=30)
        matcher.fit(X_train, y_train, X_val, y_val)
        val_probs = matcher.predict_proba(X_val)
        
        scored_val = [(sid, cid, float(p)) for (sid, cid), p in zip(val_pairs, val_probs)]
        val_scores = {sid: [] for sid in val_s1_ids}
        for sid, cid, p in scored_val:
            val_scores[sid].append((cid, p))
            
        best_thresh, _, _ = sweep_thresholds(gt_val, scored_val)
        engine = EntityDecisionEngine(base_threshold=best_thresh, singleton_cutoff=0.25, multi_match_relative_margin=0.20)
        preds = engine.predict_entities(val_scores)
        metrics = evaluate_predictions(gt_val, preds)
        
        fp = sum(len(preds[sid] - gt_val[sid]) for sid in val_s1_ids)
        fn = sum(len(gt_val[sid] - preds[sid]) for sid in val_s1_ids)
        
        results.append({
            'Configuration': name,
            'Features': X_train.shape[1],
            'Macro F0.5': metrics['macro_f05'],
            'Precision': metrics['macro_precision'],
            'Recall': metrics['macro_recall'],
            'Singleton F0.5': metrics['singleton_f05'],
            'FP': fp,
            'FN': fn,
            'Threshold': round(best_thresh, 3),
        })
        print(f"Result: Macro F0.5 = {metrics['macro_f05']:.4f} (P={metrics['macro_precision']:.4f}, R={metrics['macro_recall']:.4f}) | FP={fp}, FN={fn}")
        
    print("\n" + "=" * 80)
    print("FEATURE OPTIMIZATION SUMMARY (OPT-3)")
    print("=" * 80)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'results', 'features'))
    os.makedirs(out_dir, exist_ok=True)
    res_df.to_json(os.path.join(out_dir, 'opt3_feature_comparison.json'), indent=2, orient='records')
    res_df.to_markdown(os.path.join(out_dir, 'opt3_feature_comparison.md'), index=False)
    print(f"\nSaved report to: {os.path.join(out_dir, 'opt3_feature_comparison.md')}")

if __name__ == '__main__':
    main()
