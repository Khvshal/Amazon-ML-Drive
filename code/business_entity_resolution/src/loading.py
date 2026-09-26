"""
Reusable TSV loader for the entity resolution pipeline.
Always enforces sep="\t" to prevent silent single-column parse errors.
"""
import pandas as pd
import os

DATASET_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'Dataset'
))


def load_tsv(path: str) -> pd.DataFrame:
    """Load a TSV file with explicit tab separator.
    
    Args:
        path: Absolute or relative path to a .tsv file.
        
    Returns:
        DataFrame with correctly parsed columns.
        
    Raises:
        ValueError: If the result has only 1 column (likely wrong separator).
    """
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
    if df.shape[1] == 1:
        raise ValueError(
            f"Only 1 column parsed from {path}. "
            f"This usually means the file isn't tab-separated or the path is wrong. "
            f"Got column: {list(df.columns)}"
        )
    return df


def load_source(split: str, source: int) -> pd.DataFrame:
    """Load a source file by split and source number.
    
    Args:
        split: 'train' or 'test'
        source: 1, 2, or 3
        
    Returns:
        DataFrame with columns: entity_id, business_name, business_address, country
    """
    filename = f"{split}_source{source}.tsv"
    path = os.path.join(DATASET_DIR, split, filename)
    return load_tsv(path)


def load_ground_truth() -> pd.DataFrame:
    """Load the training ground truth file.
    
    Returns:
        DataFrame with columns: source1_entity_id, matched_entity_ids
        The matched_entity_ids column is kept as a raw string (may be empty/NaN).
    """
    path = os.path.join(DATASET_DIR, 'train', 'train_ground_truth.tsv')
    df = pd.read_csv(path, sep='\t', dtype=str)
    return df


def parse_matched_ids(matched_str) -> list:
    """Parse a matched_entity_ids string into a list of entity IDs.
    
    Args:
        matched_str: Comma-separated string of entity IDs, or NaN/empty.
        
    Returns:
        List of entity ID strings (empty list if no matches).
    """
    if pd.isna(matched_str) or str(matched_str).strip() == '':
        return []
    return [eid.strip() for eid in str(matched_str).split(',') if eid.strip()]
