"""
Pairwise feature extraction combining name, address, metadata, and cross-field interaction signals.
"""
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.features.name_features import extract_name_features
from src.features.address_features import extract_address_features


def extract_pair_features(e1: Dict[str, Any], e2: Dict[str, Any]) -> Dict[str, float]:
    """Extract complete pairwise feature dictionary between e1 (S1) and e2 (candidate).
    
    Args:
        e1: S1 normalized entity dict
        e2: S2/S3 candidate normalized entity dict
        
    Returns:
        Flat dictionary of feature name -> float value
    """
    feat = {}
    
    # Name features
    feat.update(extract_name_features(e1, e2))
    
    # Address features
    feat.update(extract_address_features(e1, e2))
    
    # Metadata features
    country1 = e1.get('country', '') or ''
    country2 = e2.get('country', '') or ''
    feat['country_match'] = float(country1 == country2 and len(country1) > 0)
    
    eid2 = e2.get('entity_id', '') or ''
    feat['is_s2'] = float(eid2.startswith('S2-'))
    feat['is_s3'] = float(eid2.startswith('S3-'))
    
    # Interaction / composite features
    feat['name_x_addr_token_jaccard'] = feat['name_token_jaccard'] * feat['addr_token_jaccard']
    feat['name_x_addr_lev'] = feat['name_lev_sim'] * feat['addr_lev_sim']
    feat['max_name_sim'] = max(feat['name_lev_sim'], feat['name_token_sort_ratio'], feat['name_jw_sim'])
    feat['max_addr_sim'] = max(feat['addr_lev_sim'], feat['addr_token_sort_ratio'], feat['addr_jw_sim'])
    feat['min_field_max_sim'] = min(feat['max_name_sim'], feat['max_addr_sim'])
    feat['mean_field_max_sim'] = (feat['max_name_sim'] + feat['max_addr_sim']) / 2.0
    feat['name_exact_and_shared_num'] = float(feat['name_exact_core'] > 0 and feat['addr_has_shared_num'] > 0)
    
    return feat


def build_pair_feature_matrix(
    candidate_pairs: Dict[str, set],
    s1_dict: Dict[str, Dict[str, Any]],
    s2s3_dict: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, set] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[str, str]]]:
    """Build feature matrix for all (s1_id, cand_id) candidate pairs.
    
    Args:
        candidate_pairs: dict mapping s1_id -> set of candidate entity_ids
        s1_dict: dict mapping s1_id -> normalized entity dict
        s2s3_dict: dict mapping cand_id -> normalized entity dict
        ground_truth: optional dict mapping s1_id -> set of true matched entity_ids
        
    Returns:
        df_features: DataFrame of features
        y: numpy array of binary labels (if ground_truth provided, else None)
        pair_ids: list of (s1_id, cand_id) tuples corresponding to rows
    """
    rows = []
    pair_ids = []
    labels = [] if ground_truth is not None else None
    
    for s1_id, cands in candidate_pairs.items():
        e1 = s1_dict.get(s1_id)
        if e1 is None:
            continue
        true_matches = ground_truth.get(s1_id, set()) if ground_truth is not None else None
        
        for cand_id in cands:
            e2 = s2s3_dict.get(cand_id)
            if e2 is None:
                continue
            
            f = extract_pair_features(e1, e2)
            rows.append(f)
            pair_ids.append((s1_id, cand_id))
            
            if labels is not None:
                is_match = int(cand_id in true_matches)
                labels.append(is_match)
                
    df_features = pd.DataFrame(rows)
    y_arr = np.array(labels, dtype=np.int32) if labels is not None else None
    
    return df_features, y_arr, pair_ids
