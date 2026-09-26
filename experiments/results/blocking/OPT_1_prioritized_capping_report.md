# Experiment OPT-1: Prioritized Score-Based Candidate Capping

## 1. Executive Summary & Decision

- **Experiment ID:** `OPT-1-PRIORITIZED-CAPPING`
- **Component Changed:** Blocking candidate capping (`src/blocking/candidate_generator.py`)
- **Status:** **PROMOTED / ADOPTED AS DEFAULT PIPELINE COMPONENT**
- **Core Metric Impact:**
  - Macro $F_{0.5}$: **$0.9539 \rightarrow 0.9851$ (+0.0312 / +3.12 percentage points gain)**
  - Precision: **$0.9763 \rightarrow 0.9952$ (+1.90 percentage points)**
  - Recall: **$0.9054 \rightarrow 0.9629$ (+5.75 percentage points)**
  - False Negatives: **$181 \rightarrow 79$ (102 False Negatives eliminated, -56.4% reduction in misses)**
  - Singleton $F_{0.5}$: **$1.0000$ (0 false merges on singletons)**
  - Candidate Load: **Identical (27.6 candidates / S1 entity)**

---

## 2. Hypothesis & Root Cause

### Root Cause Discovered in Baseline Audit:
In Phase B of the comprehensive baseline audit (`Baseline_Audit_Report.md`), candidate generation without capping achieved **98.98% pair recall**. However, the V1 baseline capped candidate lists using arbitrary Python set slicing:
```python
capped[s1_id] = set(list(cands)[:max_per_s1])
```
Because Python set iteration order is non-deterministic and arbitrary, when an S1 entity had >40 candidate pairs (19.3% of all entities), this arbitrary slice truncated true matches (even exact name matches and high-confidence TF-IDF candidates), accounting for **62.3% of all validation false negatives**.

### Hypothesis:
Replacing arbitrary slicing with a **multi-block confidence & consensus scoring function**:
1. Exact name match (Pass A): weight 100.0
2. Core name match (Pass B): weight 80.0
3. Sorted name tokens match (Pass B2): weight 60.0
4. Character TF-IDF match (Pass E): weight 45.0
5. Word TF-IDF match (Pass F): weight 40.0
6. Numeric address match (Pass D): weight 35.0
7. Address char TF-IDF match (Pass G): weight 30.0
8. Rare token match (Pass C): weight 20.0

Candidates appearing in multiple blocks receive additive consensus scores. By sorting candidates descending by score and keeping the top 40, high-confidence and consensus pairs are guaranteed retention, recovering lost true matches without increasing candidate file size.

---

## 3. Experimental Protocol

- **Evaluation Split:** 1,500 S1 entities sampled from `train_ground_truth.tsv`, split entity-level (976 Train S1, 524 Validation S1) with exact singleton stratification (26 singletons, 498 non-singletons).
- **Candidate Pool:** 65,316 candidate pool entities from `train_source2.tsv` and `train_source3.tsv` across US and India.
- **Model:** LightGBM matcher with 51 pairwise features trained identically on the train split.
- **Decision Engine:** Evaluated across identical threshold search grids and entity decision thresholds.

---

## 4. Controlled Comparison Results

| Configuration | Macro F0.5 | Precision | Recall | Singleton F0.5 | FP Count | FN Count | Candidates / S1 | Optimal Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **V1 Baseline (Arbitrary Cap @ 40)** | 0.9539 | 0.9763 | 0.9054 | 1.0000 | 6 | 181 | 27.6 | 0.860 |
| **OPT-1 (Prioritized Cap @ 40)** | **0.9851** | **0.9952** | **0.9629** | **1.0000** | 11 | **79** | **27.6** | 0.820 |
| **OPT-1 (Prioritized Cap @ 50)** | 0.9850 | 0.9942 | 0.9658 | 1.0000 | 13 | 74 | 29.4 | 0.760 |

---

## 5. Blocking Stage Recovery Breakdown

At the blocking stage alone (prior to pairwise feature extraction and model scoring):

| Candidate Selection Strategy | Recovered True Pairs | Blocking Pair Recall | Total Candidate Pairs | Avg Candidates / S1 |
|---|---:|---:|---:|---:|
| **Uncapped Union** | 1,837 / 1,857 | 98.92% | 16,321 | 31.1 |
| **Arbitrary Cap @ 40 (V1 Baseline)** | 1,726 / 1,857 | 92.95% | 14,476 | 27.6 |
| **Prioritized Cap @ 40 (OPT-1)** | **1,837 / 1,857** | **98.92%** | **14,476** | **27.6** |

> [!IMPORTANT]
> **Key Finding:** Prioritized Cap @ 40 recovered **100% of all true positive matches retrieved by the uncapped union (1,837/1,857)**, eliminating all 111 lost pairs with **zero candidate overhead**!

---

## 6. Implementation Changes

1. **`src/blocking/candidate_generator.py`:**
   - Implemented `prioritized_cap_candidates(blocks_dict, max_per_s1, block_weights)`.
   - Updated docstrings and retained backward compatibility with `cap_candidates()`.
2. **`src/predict.py`:**
   - Updated `run_country_blocking` to use `prioritized_cap_candidates`.
   - Verified end-to-end inference on test set with `utils/validate_submission.py` -> **PASS**.

---

## 7. Next Research Target (Following Plan Section 31)

Now that blocking/capping loss has been resolved (reducing FNs from 181 to 79), the remaining errors are:
1. **Decision Layer Multi-Match Margins:** True matches on high-cardinality entities (4+ matches) scoring slightly below `top_score - 0.20`.
2. **Heavy Address Corruption / Transliterated Hindi/Tamil Names:** Pairs scoring between 0.35 and 0.70 due to absence of Latin phonetic features.
3. **Target Experiment OPT-2:** Adaptive Decision Margin & Phonetic / Script-Invariant Token Matching.
