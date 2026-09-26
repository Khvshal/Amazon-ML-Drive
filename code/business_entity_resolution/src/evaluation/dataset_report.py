"""
Phase 1 — Dataset Forensics (memory-efficient version)
Loads files one at a time, builds a lightweight lookup, samples for similarity.
"""
import pandas as pd
import numpy as np
import os
import sys
import unicodedata
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE = r'c:\Users\Admin\Desktop\Amazon-ML-Drive\Dataset'
OUT_DIR = r'c:\Users\Admin\Desktop\Amazon-ML-Drive\experiments\results'
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, 'phase1_forensics.txt')

buf = []
def p(s=''):
    buf.append(str(s))

def flush():
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(buf))

def load(name):
    return pd.read_csv(os.path.join(BASE, name), sep='\t', dtype=str)

# ========== SECTION 1: Row counts, schemas, nulls ==========
p("=" * 80)
p("PHASE 1 — DATASET FORENSICS REPORT")
p("=" * 80)
p("\nSECTION 1: ROW COUNTS & SCHEMAS & NULLS")
p("-" * 60)

files_info = [
    ('train/train_source1.tsv', 'train_source1'),
    ('train/train_source2.tsv', 'train_source2'),
    ('train/train_source3.tsv', 'train_source3'),
    ('train/train_ground_truth.tsv', 'train_ground_truth'),
    ('test/test_source1.tsv', 'test_source1'),
    ('test/test_source2.tsv', 'test_source2'),
    ('test/test_source3.tsv', 'test_source3'),
]

row_counts = {}
for fpath, label in files_info:
    print(f"Loading {label}...", flush=True)
    df = load(fpath)
    row_counts[label] = len(df)
    p(f"\n--- {label} ---")
    p(f"  Shape: {df.shape}")
    p(f"  Columns: {list(df.columns)}")
    for col in df.columns:
        # For dtype=str with keep_default_na=True, NaN stays as NaN
        null_ct = df[col].isna().sum()
        empty_ct = (df[col] == '').sum() if col in df.columns else 0
        p(f"  {col}: {null_ct} null, {empty_ct} empty string ({len(df)} total)")
    
    # Country distribution for source files
    if 'country' in df.columns:
        p(f"  Country distribution:")
        for country, count in df['country'].value_counts().items():
            p(f"    {country}: {count} ({100*count/len(df):.2f}%)")
    
    del df  # free memory

flush()
print("Section 1 done.", flush=True)

# ========== SECTION 2: Ground truth match cardinality ==========
p("\n" + "=" * 60)
p("SECTION 2: MATCH CARDINALITY (ground truth)")
p("=" * 60)

gt = pd.read_csv(os.path.join(BASE, 'train', 'train_ground_truth.tsv'), sep='\t', dtype=str)

def parse_ids(s):
    if pd.isna(s) or str(s).strip() == '' or s == 'nan':
        return []
    return [e.strip() for e in str(s).split(',') if e.strip()]

gt['matched_list'] = gt['matched_entity_ids'].apply(parse_ids)
gt['total_matches'] = gt['matched_list'].apply(len)
gt['s2_count'] = gt['matched_list'].apply(lambda lst: sum(1 for e in lst if e.startswith('S2-')))
gt['s3_count'] = gt['matched_list'].apply(lambda lst: sum(1 for e in lst if e.startswith('S3-')))

p(f"\nTotal S1 entities in ground truth: {len(gt)}")
for n in range(6):
    ct = (gt['total_matches'] == n).sum()
    p(f"  {n} matches: {ct} entities ({100*ct/len(gt):.2f}%)")
ct_6plus = (gt['total_matches'] >= 6).sum()
p(f"  6+ matches: {ct_6plus} entities ({100*ct_6plus/len(gt):.2f}%)")
p(f"  Max matches: {gt['total_matches'].max()}")

p(f"\nS2 match count distribution:")
for n in range(6):
    ct = (gt['s2_count'] == n).sum()
    if ct > 0:
        p(f"  {n}: {ct}")
ct_s2_6plus = (gt['s2_count'] >= 6).sum()
if ct_s2_6plus > 0:
    p(f"  6+: {ct_s2_6plus}")

p(f"\nS3 match count distribution:")
for n in range(6):
    ct = (gt['s3_count'] == n).sum()
    if ct > 0:
        p(f"  {n}: {ct}")
ct_s3_6plus = (gt['s3_count'] >= 6).sum()
if ct_s3_6plus > 0:
    p(f"  6+: {ct_s3_6plus}")

p(f"\nS1 entities with at least one S2 match: {(gt['s2_count'] > 0).sum()}")
p(f"S1 entities with at least one S3 match: {(gt['s3_count'] > 0).sum()}")
p(f"S1 entities with both S2 and S3 matches: {((gt['s2_count'] > 0) & (gt['s3_count'] > 0)).sum()}")

# Collect positive pairs — but SAMPLE FIRST to avoid OOM on the lookup step.
# 7M+ pairs would require looking up ~10M entity IDs; we only need 50K pairs.
print("Collecting positive pairs (sampling to 50K)...", flush=True)

# Build pairs lazily from gt rows, sample S1 entities first for entity-level sampling
rng = np.random.RandomState(42)
all_s1_ids = gt['source1_entity_id'].values
has_matches = gt['total_matches'] > 0
s1_with_matches = gt[has_matches]['source1_entity_id'].values

# Sample S1 entities (not pairs) to keep entity-level distribution honest
n_sample_entities = min(30000, len(s1_with_matches))
sampled_s1 = set(rng.choice(s1_with_matches, n_sample_entities, replace=False))

# Collect all pairs for sampled S1 entities
sample_pairs = []
total_positive_pairs = 0
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    matches = row['matched_list']
    total_positive_pairs += len(matches)
    if s1_id in sampled_s1:
        for mid in matches:
            sample_pairs.append((s1_id, mid))

p(f"\nTotal positive pairs (full ground truth): {total_positive_pairs}")
p(f"Sampled {len(sampled_s1)} S1 entities -> {len(sample_pairs)} pairs for similarity analysis")

# Free ground truth
del gt

flush()
print(f"Section 2 done. {total_positive_pairs} total positive pairs, {len(sample_pairs)} sampled.", flush=True)

# ========== SECTION 3: Positive-pair similarity ==========
p("\n" + "=" * 60)
p("SECTION 3: POSITIVE-PAIR SIMILARITY ANALYSIS")
p("=" * 60)

# Build a lookup dict — only for entities in the SAMPLED pairs (~60K-100K IDs, not 10M)
needed_ids = set()
for s1_id, mid in sample_pairs:
    needed_ids.add(s1_id)
    needed_ids.add(mid)

print(f"Need to look up {len(needed_ids)} unique entity IDs (sampled subset)", flush=True)

entity_lookup = {}

# Load source files one at a time and extract only needed rows
for fpath in ['train/train_source1.tsv', 'train/train_source2.tsv', 'train/train_source3.tsv']:
    print(f"  Scanning {fpath} for needed entities...", flush=True)
    # Read in chunks to manage memory
    for chunk in pd.read_csv(os.path.join(BASE, fpath), sep='\t', dtype=str,
                              keep_default_na=False, chunksize=500000):
        mask = chunk['entity_id'].isin(needed_ids)
        for _, row in chunk[mask].iterrows():
            entity_lookup[row['entity_id']] = {
                'business_name': row['business_name'],
                'business_address': row['business_address'],
                'country': row['country'],
            }
    print(f"    Lookup now has {len(entity_lookup)} entities", flush=True)

p(f"Loaded {len(entity_lookup)} entities for similarity analysis")
p(f"Missing from lookup: {len(needed_ids) - len(entity_lookup)}")


# sample_pairs is already prepared above (entity-level sampling)
p(f"Using {len(sample_pairs)} sampled pairs for similarity computation")

def normalize_simple(s):
    if not s or s == 'nan':
        return ''
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFC', s)
    s = ' '.join(s.split())
    return s

def char_jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)

def token_jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb: return 1.0
    if not ta or not tb: return 0.0
    return len(ta & tb) / len(ta | tb)

print("Computing similarity stats...", flush=True)

stats = {k: [] for k in [
    'name_exact_raw', 'name_exact_norm', 'name_char_sim', 'name_token_sim',
    'addr_exact_raw', 'addr_exact_norm', 'addr_char_sim', 'addr_token_sim',
    'country_equal', 'source_pair',
    'name1_empty', 'name2_empty', 'addr1_empty', 'addr2_empty',
]}

skipped = 0
for i, (s1_id, m_id) in enumerate(sample_pairs):
    if i % 10000 == 0:
        print(f"  {i}/{len(sample_pairs)}...", flush=True)
    
    if s1_id not in entity_lookup or m_id not in entity_lookup:
        skipped += 1
        continue
    
    s1 = entity_lookup[s1_id]
    m = entity_lookup[m_id]
    
    n1r = s1['business_name']
    n2r = m['business_name']
    a1r = s1['business_address']
    a2r = m['business_address']
    
    n1n = normalize_simple(n1r)
    n2n = normalize_simple(n2r)
    a1n = normalize_simple(a1r)
    a2n = normalize_simple(a2r)
    
    stats['name_exact_raw'].append(n1r == n2r)
    stats['name_exact_norm'].append(n1n == n2n)
    stats['name_char_sim'].append(char_jaccard(n1n, n2n))
    stats['name_token_sim'].append(token_jaccard(n1n, n2n))
    
    stats['addr_exact_raw'].append(a1r == a2r)
    stats['addr_exact_norm'].append(a1n == a2n)
    stats['addr_char_sim'].append(char_jaccard(a1n, a2n))
    stats['addr_token_sim'].append(token_jaccard(a1n, a2n))
    
    stats['country_equal'].append(s1['country'] == m['country'])
    stats['source_pair'].append('S1xS2' if m_id.startswith('S2-') else 'S1xS3')
    
    stats['name1_empty'].append(n1n == '')
    stats['name2_empty'].append(n2n == '')
    stats['addr1_empty'].append(a1n == '')
    stats['addr2_empty'].append(a2n == '')

n = len(stats['name_exact_raw'])
p(f"\nAnalyzed {n} pairs (skipped {skipped} due to missing lookup)")

p(f"\n--- Name similarity (across all pairs) ---")
p(f"  Exact raw:        {sum(stats['name_exact_raw'])}/{n} ({100*np.mean(stats['name_exact_raw']):.2f}%)")
p(f"  Exact normalized: {sum(stats['name_exact_norm'])}/{n} ({100*np.mean(stats['name_exact_norm']):.2f}%)")
ncs = np.array(stats['name_char_sim'])
nts = np.array(stats['name_token_sim'])
p(f"  Char Jaccard:  mean={ncs.mean():.4f}  med={np.median(ncs):.4f}  p5={np.percentile(ncs,5):.4f}  p25={np.percentile(ncs,25):.4f}  p75={np.percentile(ncs,75):.4f}  p95={np.percentile(ncs,95):.4f}")
p(f"  Token Jaccard: mean={nts.mean():.4f}  med={np.median(nts):.4f}  p5={np.percentile(nts,5):.4f}  p25={np.percentile(nts,25):.4f}  p75={np.percentile(nts,75):.4f}  p95={np.percentile(nts,95):.4f}")

p(f"\n--- Address similarity (across all pairs) ---")
p(f"  Exact raw:        {sum(stats['addr_exact_raw'])}/{n} ({100*np.mean(stats['addr_exact_raw']):.2f}%)")
p(f"  Exact normalized: {sum(stats['addr_exact_norm'])}/{n} ({100*np.mean(stats['addr_exact_norm']):.2f}%)")
acs = np.array(stats['addr_char_sim'])
ats = np.array(stats['addr_token_sim'])
p(f"  Char Jaccard:  mean={acs.mean():.4f}  med={np.median(acs):.4f}  p5={np.percentile(acs,5):.4f}  p25={np.percentile(acs,25):.4f}  p75={np.percentile(acs,75):.4f}  p95={np.percentile(acs,95):.4f}")
p(f"  Token Jaccard: mean={ats.mean():.4f}  med={np.median(ats):.4f}  p5={np.percentile(ats,5):.4f}  p25={np.percentile(ats,25):.4f}  p75={np.percentile(ats,75):.4f}  p95={np.percentile(ats,95):.4f}")

p(f"\n--- Missingness in positive pairs ---")
p(f"  S1 name empty:    {sum(stats['name1_empty'])}")
p(f"  Match name empty: {sum(stats['name2_empty'])}")
p(f"  S1 addr empty:    {sum(stats['addr1_empty'])}")
p(f"  Match addr empty: {sum(stats['addr2_empty'])}")

p(f"\n--- Country equality ---")
p(f"  Same country: {sum(stats['country_equal'])}/{n} ({100*np.mean(stats['country_equal']):.2f}%)")

# Per source pair
p(f"\n--- Breakdown by source pair ---")
for sp in ['S1xS2', 'S1xS3']:
    mask = np.array([s == sp for s in stats['source_pair']])
    ct = mask.sum()
    if ct == 0:
        continue
    p(f"\n  {sp} ({ct} pairs, {100*ct/n:.1f}%):")
    p(f"    Name exact norm: {100*np.mean(np.array(stats['name_exact_norm'])[mask]):.2f}%")
    p(f"    Name char Jaccard: mean={ncs[mask].mean():.4f}  med={np.median(ncs[mask]):.4f}")
    p(f"    Name token Jaccard: mean={nts[mask].mean():.4f}  med={np.median(nts[mask]):.4f}")
    p(f"    Addr exact norm: {100*np.mean(np.array(stats['addr_exact_norm'])[mask]):.2f}%")
    p(f"    Addr char Jaccard: mean={acs[mask].mean():.4f}  med={np.median(acs[mask]):.4f}")
    p(f"    Addr token Jaccard: mean={ats[mask].mean():.4f}  med={np.median(ats[mask]):.4f}")
    p(f"    Addr empty (match side): {np.array(stats['addr2_empty'])[mask].sum()}")

flush()
print("Section 3 done.", flush=True)

# ========== SECTION 4: Representative examples ==========
p("\n" + "=" * 60)
p("SECTION 4: REPRESENTATIVE EXAMPLES")
p("=" * 60)

# Sort by name char similarity to find hardest pairs
indexed_name_sim = list(enumerate(stats['name_char_sim']))
indexed_name_sim.sort(key=lambda x: x[1])

analyzed_pairs = [(sample_pairs[i], i) for i in range(len(sample_pairs)) if i < n + skipped]

p("\n--- 15 HARDEST positive pairs (lowest name char Jaccard) ---")
shown = 0
for idx, sim in indexed_name_sim[:15]:
    s1_id, m_id = sample_pairs[idx]
    if s1_id not in entity_lookup or m_id not in entity_lookup:
        continue
    s1 = entity_lookup[s1_id]
    m = entity_lookup[m_id]
    p(f"\n  #{shown+1} name_char={sim:.4f} name_token={stats['name_token_sim'][idx]:.4f} addr_char={stats['addr_char_sim'][idx]:.4f}")
    p(f"    S1  [{s1_id}]: '{s1['business_name']}' | '{s1['business_address']}' | {s1['country']}")
    p(f"    Match [{m_id}]: '{m['business_name']}' | '{m['business_address']}' | {m['country']}")
    shown += 1

# Hardest by address
indexed_addr_sim = list(enumerate(stats['addr_char_sim']))
indexed_addr_sim.sort(key=lambda x: x[1])

p("\n--- 15 HARDEST positive pairs (lowest address char Jaccard) ---")
shown = 0
for idx, sim in indexed_addr_sim[:15]:
    s1_id, m_id = sample_pairs[idx]
    if s1_id not in entity_lookup or m_id not in entity_lookup:
        continue
    s1 = entity_lookup[s1_id]
    m = entity_lookup[m_id]
    p(f"\n  #{shown+1} addr_char={sim:.4f} addr_token={stats['addr_token_sim'][idx]:.4f} name_char={stats['name_char_sim'][idx]:.4f}")
    p(f"    S1  [{s1_id}]: '{s1['business_name']}' | '{s1['business_address']}' | {s1['country']}")
    p(f"    Match [{m_id}]: '{m['business_name']}' | '{m['business_address']}' | {m['country']}")
    shown += 1

# High similarity (sanity check)
p("\n--- 10 EASY positive pairs (highest name char Jaccard, sanity check) ---")
shown = 0
for idx, sim in indexed_name_sim[-10:]:
    s1_id, m_id = sample_pairs[idx]
    if s1_id not in entity_lookup or m_id not in entity_lookup:
        continue
    s1 = entity_lookup[s1_id]
    m = entity_lookup[m_id]
    p(f"\n  #{shown+1} name_char={sim:.4f} name_token={stats['name_token_sim'][idx]:.4f}")
    p(f"    S1  [{s1_id}]: '{s1['business_name']}' | '{s1['business_address']}' | {s1['country']}")
    p(f"    Match [{m_id}]: '{m['business_name']}' | '{m['business_address']}' | {m['country']}")
    shown += 1

p("\n" + "=" * 80)
p("END OF PHASE 1 FORENSICS REPORT")
p("=" * 80)

flush()
print(f"\nDone! Report saved to: {out_path}")
