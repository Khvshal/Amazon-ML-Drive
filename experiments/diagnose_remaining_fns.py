"""
Diagnose the remaining 79 False Negatives under OPT-1.
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
    
    gt_val = {sid: ground_truth[sid] for sid in val_s1_ids}
    s1_dict = normalize_records(s1_df)
    s2s3_dict = normalize_records(s2s3_df)
    
    train_blocks = execute_full_blocking([s1_dict[sid] for sid in train_s1_ids], list(s2s3_dict.values()))
    val_blocks = execute_full_blocking([s1_dict[sid] for sid in val_s1_ids], list(s2s3_dict.values()))
    
    train_cands = prioritized_cap_candidates(train_blocks, max_per_s1=40)
    val_cands = prioritized_cap_candidates(val_blocks, max_per_s1=40)
    
    X_train, y_train, train_pairs = build_pair_feature_matrix(train_cands, s1_dict, s2s3_dict, {sid: ground_truth[sid] for sid in train_s1_ids})
    X_val, y_val, val_pairs = build_pair_feature_matrix(val_cands, s1_dict, s2s3_dict, gt_val)
    
    matcher = LightGBMMatcher(learning_rate=0.05, num_leaves=31, max_depth=7, n_estimators=400, early_stopping_rounds=30)
    matcher.fit(X_train, y_train, X_val, y_val)
    val_probs = matcher.predict_proba(X_val)
    
    # Map pair -> prob
    pair_prob = {pair: float(p) for pair, p in zip(val_pairs, val_probs)}
    
    thresh = 0.820
    
    fn_list = []
    for sid in val_s1_ids:
        true_set = gt_val[sid]
        cand_set = val_cands.get(sid, set())
        for cid in true_set:
            if cid not in cand_set:
                fn_list.append({
                    's1_id': sid, 'cand_id': cid, 'type': 'BLOCKING_MISS',
                    'score': None, 's1_name': s1_dict[sid]['name_raw'], 'cand_name': s2s3_dict[cid]['name_raw'],
                    's1_addr': s1_dict[sid]['addr_raw'], 'cand_addr': s2s3_dict[cid]['addr_raw'],
                })
            else:
                sc = pair_prob.get((sid, cid), 0.0)
                if sc < thresh:
                    fn_list.append({
                        's1_id': sid, 'cand_id': cid, 'type': 'MODEL_UNDERCONFIDENT',
                        'score': round(sc, 4), 's1_name': s1_dict[sid]['name_raw'], 'cand_name': s2s3_dict[cid]['name_raw'],
                        's1_addr': s1_dict[sid]['addr_raw'], 'cand_addr': s2s3_dict[cid]['addr_raw'],
                    })
                    
    fn_df = pd.DataFrame(fn_list)
    print(f"\nTotal False Negatives at threshold {thresh}: {len(fn_df)}")
    print(fn_df['type'].value_counts())
    
    print("\nScore distribution for MODEL_UNDERCONFIDENT:")
    scs = fn_df[fn_df['score'].notnull()]['score']
    print(f"  0.70 - 0.82: {sum((scs >= 0.70) & (scs < 0.82))} pairs (very close to threshold!)")
    print(f"  0.50 - 0.70: {sum((scs >= 0.50) & (scs < 0.70))} pairs")
    print(f"  0.20 - 0.50: {sum((scs >= 0.20) & (scs < 0.50))} pairs")
    print(f"  < 0.20:      {sum(scs < 0.20)} pairs")
    
    print("\nSample of False Negatives scoring < 0.20:")
    for _, r in fn_df[fn_df['score'] < 0.20].head(5).iterrows():
        print(f"  Score: {r['score']} | S1: '{r['s1_name']}' @ '{r['s1_addr']}' vs Cand: '{r['cand_name']}' @ '{r['cand_addr']}'")
        
    print("\nSample of False Negatives scoring 0.50 - 0.82:")
    for _, r in fn_df[(fn_df['score'] >= 0.50) & (fn_df['score'] < 0.82)].head(5).iterrows():
        print(f"  Score: {r['score']} | S1: '{r['s1_name']}' @ '{r['s1_addr']}' vs Cand: '{r['cand_name']}' @ '{r['cand_addr']}'")

if __name__ == '__main__':
    main()
