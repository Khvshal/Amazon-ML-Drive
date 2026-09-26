"""
Candidate generator — union + cap across all blocking passes.

Per instructions (Section 6):
  - Each block is independent and evaluated separately first
  - Final candidate set is the union of all blocks
  - Cap per S1 entity to control candidate set size
  - output/candidate_pairs.tsv must be the EXACT final set fed to the classifier

This module provides:
  1. union_candidates() — merge multiple block outputs
  2. cap_candidates() — limit candidates per S1 entity
  3. evaluate_blocking() — measure recall, candidate counts vs ground truth
"""
import numpy as np
from collections import Counter


def union_candidates(*block_outputs: dict) -> dict:
    """Union multiple blocking outputs into a single candidate set.
    
    Args:
        *block_outputs: dicts mapping s1_entity_id -> set of candidate_ids
    
    Returns:
        dict mapping s1_entity_id -> set of candidate_ids (union)
    """
    all_s1_ids = set()
    for block in block_outputs:
        all_s1_ids.update(block.keys())
    
    merged = {}
    for s1_id in all_s1_ids:
        merged[s1_id] = set()
        for block in block_outputs:
            merged[s1_id] |= block.get(s1_id, set())
    
    return merged


def cap_candidates(candidates: dict, max_per_s1: int = 50) -> dict:
    """Cap the number of candidates per S1 entity.
    
    When an S1 entity has more than max_per_s1 candidates,
    keeps only the first max_per_s1 (arbitrary — for a smarter cap,
    score candidates first and keep top-scoring ones).
    
    Args:
        candidates: dict mapping s1_entity_id -> set of candidate_ids
        max_per_s1: maximum candidates per S1 entity
    
    Returns:
        dict mapping s1_entity_id -> set of candidate_ids (capped)
    """
    capped = {}
    for s1_id, cands in candidates.items():
        if len(cands) > max_per_s1:
            capped[s1_id] = set(list(cands)[:max_per_s1])
        else:
            capped[s1_id] = cands
    return capped


def evaluate_blocking(candidates: dict, ground_truth: dict,
                      block_name: str = 'unknown') -> dict:
    """Evaluate blocking quality against ground truth.
    
    Per instructions (Section 6), reports:
      - blocking recall = true positive pairs recovered / total true positive pairs
      - average / median / 95th-percentile / max candidates per S1
      - total candidate pairs
      - candidate reduction ratio
    
    Args:
        candidates: dict mapping s1_entity_id -> set of candidate_ids
        ground_truth: dict mapping s1_entity_id -> set of true match entity_ids
        block_name: name for logging
    
    Returns:
        dict with evaluation metrics
    """
    total_true_pairs = 0
    recovered_pairs = 0
    candidate_counts = []
    
    for s1_id, true_matches in ground_truth.items():
        true_set = set(true_matches) if isinstance(true_matches, list) else true_matches
        total_true_pairs += len(true_set)
        
        cand_set = candidates.get(s1_id, set())
        recovered_pairs += len(true_set & cand_set)
        candidate_counts.append(len(cand_set))
    
    # S1 entities not in candidates (if any)
    for s1_id in ground_truth:
        if s1_id not in candidates:
            candidate_counts.append(0)
    
    cc = np.array(candidate_counts) if candidate_counts else np.array([0])
    total_candidates = int(cc.sum())
    
    # Reduction ratio: 1 - (candidates / total possible pairs)
    # Total possible = |S1| * |S2+S3| (approximated from candidate set)
    
    blocking_recall = recovered_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    
    metrics = {
        'block_name': block_name,
        'blocking_recall': round(blocking_recall, 6),
        'recovered_pairs': recovered_pairs,
        'total_true_pairs': total_true_pairs,
        'total_candidate_pairs': total_candidates,
        'avg_candidates': round(float(cc.mean()), 2),
        'median_candidates': round(float(np.median(cc)), 2),
        'p95_candidates': round(float(np.percentile(cc, 95)), 2),
        'max_candidates': int(cc.max()),
        'min_candidates': int(cc.min()),
        's1_entities_with_candidates': int((cc > 0).sum()),
        's1_entities_total': len(cc),
    }
    
    return metrics


def format_blocking_report(metrics: dict) -> str:
    """Format blocking evaluation metrics as a readable string."""
    lines = [
        f"=== Blocking: {metrics['block_name']} ===",
        f"  Recall: {metrics['blocking_recall']:.4f} ({metrics['recovered_pairs']}/{metrics['total_true_pairs']})",
        f"  Total candidate pairs: {metrics['total_candidate_pairs']:,}",
        f"  Candidates per S1: avg={metrics['avg_candidates']:.1f}  med={metrics['median_candidates']:.1f}  "
        f"p95={metrics['p95_candidates']:.1f}  max={metrics['max_candidates']}",
        f"  S1 with candidates: {metrics['s1_entities_with_candidates']}/{metrics['s1_entities_total']}",
    ]
    return '\n'.join(lines)
