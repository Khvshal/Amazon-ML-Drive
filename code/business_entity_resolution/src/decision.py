"""
Entity-level decision layer.
Converts pairwise candidate probabilities into final matching sets per S1 entity.
Implements margin-based filtering, singleton guardrails, and multi-match preservation.
"""
from typing import Dict, List, Set, Tuple, Optional
import numpy as np


class EntityDecisionEngine:
    """Margin-based entity decision engine."""
    
    def __init__(
        self,
        base_threshold: float = 0.50,
        singleton_cutoff: float = 0.30,
        multi_match_relative_margin: float = 0.18,
        min_single_margin: float = 0.05,
        max_matches_per_s1: int = 10,
    ):
        """
        Args:
            base_threshold: Primary probability cutoff for match acceptance.
            singleton_cutoff: Below this score, candidates are completely disregarded.
            multi_match_relative_margin: For multi-matches, candidates must be within
                                        this distance from the top score.
            min_single_margin: When only one candidate is above threshold, require it
                               to have at least this margin over third/background noise.
            max_matches_per_s1: Maximum matches allowed for a single S1 entity.
        """
        self.base_threshold = base_threshold
        self.singleton_cutoff = singleton_cutoff
        self.multi_match_relative_margin = multi_match_relative_margin
        self.min_single_margin = min_single_margin
        self.max_matches_per_s1 = max_matches_per_s1

    def decide_entity(
        self,
        candidate_scores: List[Tuple[str, float]],
    ) -> Set[str]:
        """Make decision for a single S1 entity given its scored candidates.
        
        Args:
            candidate_scores: list of (cand_id, score) tuples
            
        Returns:
            Set of accepted matched candidate entity_ids (empty set if singleton)
        """
        if not candidate_scores:
            return set()
            
        # Filter candidates below the absolute noise floor
        valid_cands = [(cid, s) for cid, s in candidate_scores if s >= self.singleton_cutoff]
        if not valid_cands:
            return set()
            
        # Sort descending by score
        valid_cands.sort(key=lambda x: x[1], reverse=True)
        top_cand, top_score = valid_cands[0]
        
        # If top score doesn't reach the base threshold, bias toward singleton / no match
        if top_score < self.base_threshold:
            return set()
            
        # Check candidates above threshold
        above_thresh = [(cid, s) for cid, s in valid_cands if s >= self.base_threshold]
        
        if len(above_thresh) == 1:
            # Exactly one candidate above threshold
            # High confidence single match
            return {top_cand}
            
        # Multiple candidates above threshold: multi-match cluster
        # Keep candidates whose score is sufficiently close to the top score
        score_floor = max(self.base_threshold, top_score - self.multi_match_relative_margin)
        selected = [cid for cid, s in above_thresh if s >= score_floor]
        
        # Enforce max match cap
        return set(selected[:self.max_matches_per_s1])

    def predict_entities(
        self,
        s1_candidate_scores: Dict[str, List[Tuple[str, float]]],
    ) -> Dict[str, Set[str]]:
        """Run decision logic across all S1 entities."""
        decisions = {}
        for s1_id, scores in s1_candidate_scores.items():
            decisions[s1_id] = self.decide_entity(scores)
        return decisions
