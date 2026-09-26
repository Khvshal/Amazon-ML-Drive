"""
Phase 11 — Error analysis module.
Logs, categorizes, and diagnoses False Positives and False Negatives
to guide iterative model and blocking improvements.
"""
from typing import Dict, Set, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


def analyze_errors(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    blocked_candidates: Dict[str, Set[str]],
    scored_pairs: Dict[Tuple[str, str], float],
    s1_dict: Dict[str, Dict[str, Any]],
    s2s3_dict: Dict[str, Dict[str, Any]],
    max_examples_to_log: int = 20,
) -> Dict[str, Any]:
    """Run systematic error analysis over validation split.
    
    Returns:
        Dict containing error statistics and categorized diagnostic examples.
    """
    total_true_pairs = sum(len(v) for v in ground_truth.values())
    total_pred_pairs = sum(len(v) for v in predictions.values())
    
    tp_pairs = []
    fp_pairs = []
    fn_pairs = []
    
    # Categorization buckets for False Negatives
    fn_missed_in_blocking = []
    fn_blocked_low_score = []
    fn_blocked_margin_filtered = []
    
    # False Positives
    fp_details = []
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        blocked_set = blocked_candidates.get(s1_id, set())
        
        # True Positives
        tp = true_set & pred_set
        for cid in tp:
            tp_pairs.append((s1_id, cid))
            
        # False Positives
        fp = pred_set - true_set
        for cid in fp:
            fp_pairs.append((s1_id, cid))
            score = scored_pairs.get((s1_id, cid), 0.0)
            e1 = s1_dict.get(s1_id, {})
            e2 = s2s3_dict.get(cid, {})
            fp_details.append({
                's1_id': s1_id,
                'candidate_id': cid,
                'score': score,
                's1_name': e1.get('name_raw', ''),
                'cand_name': e2.get('name_raw', ''),
                's1_addr': e1.get('addr_raw', ''),
                'cand_addr': e2.get('addr_raw', ''),
                'country': e1.get('country', ''),
            })
            
        # False Negatives
        fn = true_set - pred_set
        for cid in fn:
            fn_pairs.append((s1_id, cid))
            e1 = s1_dict.get(s1_id, {})
            e2 = s2s3_dict.get(cid, {})
            in_blocking = (cid in blocked_set)
            score = scored_pairs.get((s1_id, cid), None)
            
            fn_item = {
                's1_id': s1_id,
                'candidate_id': cid,
                'in_blocking': in_blocking,
                'score': score,
                's1_name': e1.get('name_raw', ''),
                'cand_name': e2.get('name_raw', ''),
                's1_addr': e1.get('addr_raw', ''),
                'cand_addr': e2.get('addr_raw', ''),
                'country': e1.get('country', ''),
            }
            
            if not in_blocking:
                fn_missed_in_blocking.append(fn_item)
            elif score is not None and score < 0.5:
                fn_blocked_low_score.append(fn_item)
            else:
                fn_blocked_margin_filtered.append(fn_item)

    report = {
        'total_true_pairs': total_true_pairs,
        'total_pred_pairs': total_pred_pairs,
        'tp_count': len(tp_pairs),
        'fp_count': len(fp_pairs),
        'fn_count': len(fn_pairs),
        'fn_breakdown': {
            'missed_in_blocking': len(fn_missed_in_blocking),
            'missed_by_model_low_score': len(fn_blocked_low_score),
            'filtered_by_decision_margin': len(fn_blocked_margin_filtered),
        },
        'sample_false_positives': fp_details[:max_examples_to_log],
        'sample_fn_missed_in_blocking': fn_missed_in_blocking[:max_examples_to_log],
        'sample_fn_low_score': fn_blocked_low_score[:max_examples_to_log],
    }
    return report


def format_error_summary(report: Dict[str, Any]) -> str:
    """Format the error analysis report as human-readable string."""
    lines = [
        "=" * 60,
        "VALIDATION ERROR ANALYSIS SUMMARY",
        "=" * 60,
        f"True Pairs: {report['total_true_pairs']} | Predicted Pairs: {report['total_pred_pairs']}",
        f"TP: {report['tp_count']} | FP: {report['fp_count']} | FN: {report['fn_count']}",
        "-" * 60,
        "False Negative Breakdown:",
        f"  - Missed at Blocking Stage:       {report['fn_breakdown']['missed_in_blocking']} ({100*report['fn_breakdown']['missed_in_blocking']/max(1, report['fn_count']):.1f}%)",
        f"  - Model Scored Below Threshold:   {report['fn_breakdown']['missed_by_model_low_score']} ({100*report['fn_breakdown']['missed_by_model_low_score']/max(1, report['fn_count']):.1f}%)",
        f"  - Filtered by Decision Engine:    {report['fn_breakdown']['filtered_by_decision_margin']} ({100*report['fn_breakdown']['filtered_by_decision_margin']/max(1, report['fn_count']):.1f}%)",
        "=" * 60,
    ]
    return "\n".join(lines)
