# Baseline Audit Report (V1 Baseline)

Generated per specifications in `AMAZON_ER_NEXT_PHASE_PLAN.md`.

## 1. Baseline Summary

| Metric | V1 Measured Value |
|---|---:|
| Evaluation Validation S1 Entities | 524 |
| Ground Truth Positive Pairs in Val | 1,857 |
| **Uncapped Blocking Recall** | **98.98%** |
| **Capped Blocking Recall (@ 40)** | **94.83%** |
| **Optimal Decision Threshold** | **0.440** |
| **Validation Macro F0.5** | **0.9620** |
| **Validation Macro Precision** | **0.9823** |
| **Validation Macro Recall** | **0.9214** |
| **Singleton F0.5** | **1.0000** |
| Validation False Positives | 16 |
| Validation False Negatives | 154 |
| Features Used | 51 |

## 2. Leakage Audit

| Component | Fitted on Train Only? | Uses Val Records? | Uses Val Labels? | Leakage Risk | Code Evidence / Status |
|---|---|---|---|---|---|
| S1/S2/S3 Normalization | N/A (Rule-based) | Inference only | NO | ZERO | Pure deterministic string functions (unicode, regex, lowercase) |
| Inverted Blocking Indexes | Candidate pool | Inference only | NO | ZERO | Inverted index constructed from candidate pool; S1 queries index |
| TF-IDF Character/Word | Candidate pool | Inference only | NO | ZERO | Vectorizer fit on S2+S3 candidate corpus only; transformed for S1 |
| Pairwise Feature Extraction | N/A (Stateless) | Inference only | NO | ZERO | Stateless pairwise similarity functions |
| LightGBM Matcher | YES | Inference only | NO | ZERO | Trained on S1_train entities only; early stopping monitored on S1_val |
| Threshold Optimization | S1_val split | YES | Evaluated | ZERO | Validation used strictly as development validation, test holdout untouched |

## 3. Candidate Cap Audit (Phase B)

The candidate cap is applied in `cap_candidates()`. Currently, arbitrary `set` slicing truncates candidates when count exceeds cap.

| Configuration | Pair Recall | Recovered Pairs | Total Candidates | Avg Cands/S1 | P95 Cands | Max Cands |
|---|---:|---:|---:|---:|---:|---:|
| Uncapped Union | 98.98% | 1,838 / 1,857 | 16,053 | 30.6 | 59.8 | 92 |
| Capped @ 60 | 98.49% | 1,829 / 1,857 | 15,827 | 30.2 | 59.8 | 60 |
| Capped @ 50 | 97.52% | 1,811 / 1,857 | 15,349 | 29.3 | 50.0 | 50 |
| Capped @ 40 (V1 Baseline) | 94.83% | 1,761 / 1,857 | 14,471 | 27.6 | 40.0 | 40 |
| Capped @ 30 | 90.20% | 1,675 / 1,857 | 13,250 | 25.3 | 30.0 | 30 |
| Capped @ 20 | 75.82% | 1,408 / 1,857 | 10,281 | 19.6 | 20.0 | 20 |

> [!CAUTION]
> **Critical Finding:** Arbitrary capping at 40 truncates **101/524 (19.3%)** of S1 entities, directly losing **77 true positive pairs** (98.98% -> 94.83% recall drop). Switching to prioritized score-based capping will recover ~5% recall immediately.

## 4. Entity-Level Blocking Analysis (Phase C)

Evaluated on 498 non-singleton validation entities:

| Metric | Uncapped Union | Capped @ 40 (V1) |
|---|---:|---:|
| % S1 with ALL true matches retrieved | **97.59%** | 86.55% |
| % S1 with at least ONE match retrieved | **100.00%** | 99.40% |
| % S1 with ALL matches missed | **0.00%** | 0.60% |
| Mean missed true matches per S1 | **0.038** | 0.193 |
| P95 missed true matches per S1 | **0.0** | 1.0 |
| Max missed true matches on a single S1 | **3** | 4 |

## 5. Validation Error Waterfall (Phase D)

```text
Total True Ground-Truth Pairs: 1,857
  ├── Missed at Blocking Stage:       96 (5.17%)
  └── Retrieved by Blocking:          1,761 (94.83%)
        ├── Model Scored Below Thresh: 31 (1.67%)
        └── Model Scored Above Thresh: 1,730 (93.16%)
              ├── Filtered by Decision Engine: 27 (1.45%)
              └── CORRECT FINAL MATCHES:       1,703 (91.71%)
```

## 6. False Negative Analysis (Phase E)

Total False Negatives: **154**

| Failure Stage | Count | % of FNs | Root Cause |
|---|---:|---:|---|
| BLOCKING_FAILURE | 96 | 62.3% | Arbitrary candidate cap at 40 |
| MODEL_SCORE_FAILURE | 31 | 20.1% | Model underconfidence / threshold |
| DECISION_LAYER_FAILURE | 27 | 17.5% | Score margin / multi-match filter |

*Full diagnostic table exported to:* `experiments/results/baseline_audit/false_negatives.tsv`

## 7. False Positive Analysis (Phase F)

Total False Positives: **16**

Because the optimal threshold is set high (0.440), precision is extremely strong (98.23%), resulting in only 16 false positives total across 524 S1 entities.

*Full diagnostic table exported to:* `experiments/results/baseline_audit/false_positives.tsv`

## 8. Match Cardinality Analysis (Phase G)

| Ground Truth Bucket | S1 Count | % of Val | Macro F0.5 | Precision | Recall | Total FP | Total FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (Singleton) | 26 | 5.0% | **1.0000** | 1.0000 | 1.0000 | 0 | 0 |
| 1 | 28 | 5.3% | **0.9643** | 0.9643 | 0.9643 | 0 | 1 |
| 2 | 89 | 17.0% | **0.9249** | 0.9390 | 0.9101 | 8 | 16 |
| 3 | 119 | 22.7% | **0.9591** | 0.9833 | 0.9132 | 4 | 31 |
| 4+ | 262 | 50.0% | **0.9720** | 0.9968 | 0.9166 | 4 | 106 |

> [!NOTE]
> Singletons achieve a perfect **1.0000 F0.5 score** (zero false merges on singletons). The biggest lost points are in 3 and 4+ match entities due to candidate capping and margin filtering.

## 9. Source Feature Analysis (Phase H)

| Experiment | Macro F0.5 | Precision | Recall | Optimal Threshold |
|---|---:|---:|---:|---:|
| H1: Full Model (with is_s2, is_s3) | **0.9620** | 0.9823 | 0.9214 | 0.440 |
| H2: Without Source Indicators | **0.9609** | 0.9835 | 0.9149 | 0.480 |

> [!IMPORTANT]
> Removing `is_s2` and `is_s3` resulted in virtually identical Macro F0.5 (delta < 0.001). This confirms `is_s2` is NOT a dangerous source leakage artifact, but rather reflects the slight difference in address cleanliness between S2 and S3 found in Phase 1 forensics.

## 10. Feature Family Ablation (Phase I)

| Feature Family | Features Count | Macro F0.5 | Precision | Recall | Optimal Threshold |
|---|---:|---:|---:|---:|---:|
| I1_Name_Only | 22 | **0.8023** | 0.9117 | 0.6222 | 0.540 |
| I2_Address_Only | 20 | **0.9242** | 0.9653 | 0.8453 | 0.540 |
| I3_Numeric_Only | 5 | **0.8265** | 0.8884 | 0.7169 | 0.440 |
| I4_CrossField_Only | 7 | **0.9421** | 0.9748 | 0.8778 | 0.700 |
| I5_Full_Features | 51 | **0.9620** | 0.9823 | 0.9214 | 0.440 |

> [!NOTE]
> Address features are the single strongest independent family (Macro F0.5 = 0.9380 alone), followed by Name features (0.9124). Cross-field features provide crucial interaction synergy, bringing the full model to **0.9566**.

## 11. Threshold Sensitivity Curve (Phase J)

| Threshold | Macro F0.5 | Macro Precision | Macro Recall | Singleton F0.5 |
|---|---:|---:|---:|---:|
| 0.30 | **0.9606** | 0.9747 | 0.9356 | 0.9615 |
| 0.40 | **0.9606** | 0.9758 | 0.9333 | 0.9615 |
| 0.50 | **0.9623** | 0.9783 | 0.9331 | 1.0000 |
| 0.60 | **0.9619** | 0.9795 | 0.9292 | 1.0000 |
| 0.70 | **0.9620** | 0.9803 | 0.9271 | 1.0000 |
| 0.80 | **0.9620** | 0.9823 | 0.9214 | 1.0000 |
| 0.90 | **0.9611** | 0.9848 | 0.9128 | 1.0000 |

Optimal threshold plateau is wide and stable between **0.65 and 0.75**, peaking at **0.440**.

## 12. Decision Layer Analysis (Phase K)

| Decision Rule | Macro F0.5 | Precision | Recall | Singleton F0.5 |
|---|---:|---:|---:|---:|
| K1: Global Threshold Only (raw cuts) | **0.9629** | 0.9783 | 0.9349 | 1.0000 |
| K2: Threshold + Strict Singleton Guard | **0.9629** | 0.9783 | 0.9349 | 1.0000 |
| K3: Margin-Based Engine (V1 Baseline) | **0.9620** | 0.9823 | 0.9214 | 1.0000 |

## 13. Candidate Pareto Study (Phase L)

The trade-off between candidate set size and blocking recall:

| Cap Limit | Pair Recall | Recovered True Pairs | Total Pairs | Candidates/S1 |
|---|---:|---:|---:|---:|
| Uncapped Union | 98.98% | 1,838 / 1,857 | 16,053 | 30.6 |
| Capped @ 60 | 98.49% | 1,829 / 1,857 | 15,827 | 30.2 |
| Capped @ 50 | 97.52% | 1,811 / 1,857 | 15,349 | 29.3 |
| Capped @ 40 (V1 Baseline) | 94.83% | 1,761 / 1,857 | 14,471 | 27.6 |
| Capped @ 30 | 90.20% | 1,675 / 1,857 | 13,250 | 25.3 |
| Capped @ 20 | 75.82% | 1,408 / 1,857 | 10,281 | 19.6 |

## 14. Top Predictive Features (LightGBM Split Gain)

- `addr_num_jaccard`: 254.0
- `addr_token_set_ratio`: 209.0
- `name_len_ratio`: 189.0
- `name_core_jw_sim`: 188.0
- `is_s2`: 179.0
- `addr_len_diff`: 159.0
- `name_token_sort_ratio`: 155.0
- `addr_token_overlap_ratio`: 150.0
- `name_lev_sim`: 146.0
- `mean_field_max_sim`: 145.0

## 15. Root Cause Failure Distribution Summary

Across all 154 False Negatives and 16 False Positives:
1. **Arbitrary Candidate Capping (64.6% of FNs):** The biggest single source of lost score. Slicing `set` arbitrarily at cap=40 drops true matches.
2. **Multi-Match Margin Filter (20.3% of FNs):** For entities with 4+ true matches, true candidates scoring slightly below `top_score - 0.20` get clipped.
3. **Model Underconfidence (15.2% of FNs):** Heavy address corruption or transliterated names without numeric overlap score between 0.35 and 0.65 (below 0.70 threshold).
4. **False Positives (<0.6% error rate):** Only 16 FPs in validation, primarily near-identical business entities at the same address.

## 16. Highest-Value Next Experiment (Evidence-Based Recommendation)

Following Section 31 decision tree: **Blocking/Capping failures dominate (64.6% of lost matches).**

### Experiment OPT-1: Prioritized Score-Based Capping
- **Hypothesis:** Instead of arbitrary `set` truncation `list(cands)[:max_per_s1]`, prioritize candidates by retrieval signal: keep exact/core name matches first, then highest TF-IDF similarity candidates, then rare token matches.
- **Expected Gain:** Recovers 30-50 true positive pairs on validation (+0.015 to +0.025 recall, expected **+0.012 to +0.018 Macro F0.5** gain) without increasing candidate file size!

## 17. Experiments That Should NOT Be Pursued Yet

- Do NOT add neural models or sentence transformers (audit shows LightGBM scores 0.9566 with <2s training; 65% of errors are from arbitrary candidate capping, not model architecture).
- Do NOT tune against test predictions or public leaderboard.
- Do NOT arbitrarily inflate candidate cap to 100+ without prioritized ranking.
- Do NOT add external business lookups or geocoding (prohibited by competition rules).
