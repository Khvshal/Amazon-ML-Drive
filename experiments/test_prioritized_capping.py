"""
Test Prioritized Score-Based Capping vs Arbitrary Capping on Validation Set.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'code', 'business_entity_resolution')))

from src.evaluation.audit_baseline import load_audit_data, normalize_records, execute_full_blocking, BASE
from src.blocking.candidate_generator import cap_candidates

def prioritized_cap(blocks_dict, max_per_s1=40):
    # Weights reflecting precision and recall strength
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
    
    # Collect all s1_ids
    all_s1 = set()
    for b_key in weights:
        all_s1.update(blocks_dict[b_key].keys())
        
    capped = {}
    for sid in all_s1:
        # Score each candidate for this S1
        cand_scores = {}
        for b_key, w in weights.items():
            b_cands = blocks_dict[b_key].get(sid, set())
            for cid in b_cands:
                cand_scores[cid] = cand_scores.get(cid, 0.0) + w
                
        if len(cand_scores) <= max_per_s1:
            capped[sid] = set(cand_scores.keys())
        else:
            # Sort descending by score, tie-break by cid
            sorted_cands = sorted(cand_scores.keys(), key=lambda c: (cand_scores[c], c), reverse=True)
            capped[sid] = set(sorted_cands[:max_per_s1])
            
    return capped


def main():
    print("Loading audit data for 1500 S1 sample...", flush=True)
    s1_df, s2s3_df, ground_truth, sampled_s1_ids = load_audit_data(BASE, 1500, seed=42)
    
    # Split
    rng_split = np.random.RandomState(42)
    singletons = [sid for sid in sampled_s1_ids if len(ground_truth[sid]) == 0]
    non_singletons = [sid for sid in sampled_s1_ids if len(ground_truth[sid]) > 0]
    val_single = set(rng_split.choice(singletons, max(1, int(len(singletons) * 0.35)), replace=False)) if singletons else set()
    val_non_single = set(rng_split.choice(non_singletons, int(len(non_singletons) * 0.35), replace=False))
    val_s1_ids = val_single | val_non_single
    gt_val = {sid: ground_truth[sid] for sid in val_s1_ids}
    
    print("Normalizing entities...", flush=True)
    s1_dict = normalize_records(s1_df)
    s2s3_dict = normalize_records(s2s3_df)
    
    s1_val_ents = [s1_dict[sid] for sid in val_s1_ids]
    s2s3_ents_list = list(s2s3_dict.values())
    
    print("Running blocking passes for validation...", flush=True)
    val_blocks = execute_full_blocking(s1_val_ents, s2s3_ents_list)
    uncapped = val_blocks['union_uncapped']
    
    arbitrary_40 = cap_candidates(uncapped, max_per_s1=40)
    arbitrary_50 = cap_candidates(uncapped, max_per_s1=50)
    prioritized_40 = prioritized_cap(val_blocks, max_per_s1=40)
    prioritized_50 = prioritized_cap(val_blocks, max_per_s1=50)
    
    total_true = sum(len(v) for v in gt_val.values())
    
    def eval_res(cands, name):
        rec = sum(len(gt_val[sid] & cands.get(sid, set())) for sid in val_s1_ids)
        tot_c = sum(len(cands.get(sid, set())) for sid in val_s1_ids)
        print(f"{name:30s}: Recovered {rec}/{total_true} ({100*rec/total_true:.2f}%) | Total Cands: {tot_c} | Avg: {tot_c/len(val_s1_ids):.1f}")
        
    print("\n" + "=" * 70)
    print("BLOCKING CAPPING COMPARISON RESULTS")
    print("=" * 70)
    eval_res(uncapped, "Uncapped Union")
    eval_res(arbitrary_40, "Arbitrary Cap @ 40 (V1 Baseline)")
    eval_res(prioritized_40, "Prioritized Cap @ 40 (OPT-1)")
    eval_res(arbitrary_50, "Arbitrary Cap @ 50")
    eval_res(prioritized_50, "Prioritized Cap @ 50 (OPT-1)")

if __name__ == '__main__':
    main()
