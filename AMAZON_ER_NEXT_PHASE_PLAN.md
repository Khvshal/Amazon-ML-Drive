# Amazon ML Challenge — Entity Resolution: Next-Step Research & Optimization Plan

**Purpose:** This document is the agent-ready work order for what to do after the current V1 baseline.

**Current baseline (record this as V1; do not overwrite it):**

| Metric | V1 |
|---|---:|
| Blocking pair recall | 99.29% |
| Average candidates / S1 | 50.89 |
| Candidate reduction | 98.34% |
| Validation Macro F0.5 | 0.9566 |
| Precision | 0.9776 |
| Recall | 0.9112 |
| Validation false positives | 5 |
| Threshold | 0.700 |
| Pairwise features | 51 |
| Model | LightGBM |
| Test S1 entities | 1,732,544 |
| Submission validator | PASS |

## PRIMARY OBJECTIVE — MAXIMIZE THE COMPETITION SCORE

The system is being built to maximize the competition's **Macro F0.5 on unseen test entities**.

Everything in this document must ultimately serve that objective.

The current validation score is only a proxy for the hidden competition score. Therefore:

1. Optimize for **Macro F0.5**, not accuracy, AUC, F1, recall, or model sophistication.
2. Treat precision as especially important because the competition uses **F0.5**.
3. Improve the validation score only when the improvement is likely to generalize.
4. Do not chase tiny validation gains that appear only on one split.
5. Prefer improvements that are stable across multiple entity-level validation splits / folds.
6. Penalize unnecessary complexity and unstable behavior.
7. Never use the hidden test set or external business information to tune the system.
8. The final objective is **generalization to the competition's unseen test entities**, not winning a single validation split.

The current validation result is:

```text
Macro F0.5 = 0.9566
Precision   = 0.9776
Recall      = 0.9112
FP          = 5
```

The job now is to find where the remaining errors occur and then make **measured changes that increase expected competition Macro F0.5 while reducing overfitting risk**.

---

# 1. Non-negotiable challenge constraints

The agent MUST preserve these constraints:

- No external business/entity data.
- No business search APIs.
- No geocoding.
- No government/company databases.
- No commercial entity-resolution APIs.
- No internet-based business-data augmentation.
- Do not use the test labels for any tuning or model-selection decision.
- Do not hard-code the known training countries.
- The pipeline must support open-set countries.
- Candidate generation must materially reduce the search space.
- `candidate_pairs.tsv` must contain the **final candidate set actually passed to the final matching model**.
- Every predicted match must appear in `candidate_pairs.tsv`.
- No duplicate candidate IDs.
- No duplicate output matches.
- No self matches.
- Every test S1 entity must have exactly one row in both required outputs.
- Keep the final model within the challenge's parameter/license restrictions.
- Preserve a runnable CPU-capable pipeline.
- Do not introduce account rotation, quota bypassing, or other attempts to circumvent platform limits.

---

# 2. GENERALIZATION / ANTI-OVERFITTING REQUIREMENTS

This is a competition system, so a high validation score is useful only if it predicts performance on unseen test entities.

## 2.1 Never optimize the hidden test set

The test set may be used only for:

- final inference
- runtime/resource checks
- submission-format validation

It MUST NOT be used to choose:

- features
- thresholds
- candidate K
- blocking methods
- model hyperparameters
- source-specific rules
- decision rules
- preprocessing variants
- model architecture

Do not inspect the hidden test predictions and then modify the pipeline because they "look wrong."

---

## 2.2 Do not overfit one validation split

The current V1 score of `0.9566` must be treated as a baseline, not as ground truth for generalization.

For important experiments, use **multiple entity-level validation splits/folds** where computationally practical.

The split unit must be the S1 entity.

Never split individual candidate pairs randomly.

For each serious experiment report:

| Experiment | Fold 1 F0.5 | Fold 2 F0.5 | Fold 3 F0.5 | Mean | Std | Worst fold |
|---|---:|---:|---:|---:|---:|---:|

Also report precision and recall.

A model that gains:

```text
+0.003 on one split
-0.004 on other splits
```

should NOT automatically replace the baseline.

Prefer improvements that are:

- positive across most folds,
- reasonably stable,
- supported by a plausible failure mechanism,
- not dependent on a tiny number of examples.

---

## 2.3 Keep a completely untouched validation holdout

Where computationally possible, establish:

```text
TRAIN
  |
  +---- development folds
  |
  +---- final untouched validation holdout
```

Use development folds for experimentation.

Use the untouched holdout to confirm the final candidate before test inference.

Do not repeatedly tune against the untouched holdout.

If the dataset size makes this impractical, document the limitation rather than pretending the validation estimate is independent.

---

## 2.4 Track experiment history

Maintain an experiment ledger:

```text
experiment_id
hypothesis
code/config version
features changed
blocking changed
model changed
threshold changed
fold scores
mean F0.5
std F0.5
precision
recall
FP
FN
runtime
decision
reason
```

This prevents unconscious cherry-picking.

---

## 2.5 Statistical / practical significance

Do not celebrate a tiny gain automatically.

For example:

```text
0.9566 -> 0.9568
```

is not necessarily a meaningful improvement.

For every proposed replacement, report:

- absolute F0.5 gain
- relative gain
- fold-to-fold variance
- number of entities affected
- FP change
- FN change
- runtime change
- memory change

If a change affects only a handful of validation entities, explicitly flag it as potentially unstable.

---

## 2.6 Avoid validation overfitting through repeated threshold tuning

Threshold optimization is itself a form of model selection.

Do not repeatedly inspect the same holdout and tune the threshold until the score peaks.

Use:

```text
development data -> choose threshold
held-out data    -> evaluate threshold
```

For the final selected system, document exactly where the threshold came from.

---

## 2.7 Avoid feature-selection overfitting

Do not add a feature merely because it ranks highly on one validation run.

A feature should survive:

- multiple validation splits where possible,
- ablation testing,
- leakage inspection,
- stability analysis.

Particular caution is required for:

- source indicators,
- rare-token statistics,
- highly specific IDs,
- candidate rank,
- retrieval scores,
- features derived from global frequency,
- features derived from the full dataset.

These can be legitimate, but they can also encode accidental dataset structure.

---

## 2.8 Prefer simple models when they perform equally well

If:

```text
Model A = 0.9565
Model B = 0.9566
```

and Model B requires substantially more complexity, prefer the simpler system unless additional evidence shows better generalization.

Do not equate:

```text
more models + more features + embeddings
```

with:

```text
better competition score
```

---

# 3. COMPETITION-SCORE OPTIMIZATION LOOP

The agent must use this loop for optimization:

```text
1. Identify a measured source of validation error
            |
            v
2. Form one specific hypothesis
            |
            v
3. Make ONE controlled change
            |
            v
4. Evaluate Macro F0.5 on development folds
            |
            v
5. Check precision / recall / FP / FN
            |
            v
6. Check stability across folds
            |
            v
7. Check whether the mechanism makes sense
            |
            v
8. Confirm on untouched validation holdout
            |
            v
9. Keep only if improvement generalizes
```

Do not perform ten changes at once and then claim that the combined system is better. That makes it impossible to know what caused the gain and makes overfitting much easier.

---

# 4. What "better" means

A candidate system is better when it improves the **competition-relevant objective** while maintaining generalization.

Primary metric:

```text
Macro F0.5
```

Secondary diagnostics:

```text
Precision
Recall
FP count
FN count
Entity-level blocking recall
Candidate count
Runtime
Memory
```

The primary ranking of experiments should therefore be:

1. Generalized Macro F0.5
2. Precision / false-positive behavior
3. Stability across validation splits
4. Recall / false-negative behavior
5. Blocking efficiency
6. Runtime/memory
7. Model complexity

Do NOT rank experiments by:

- AUC alone
- accuracy
- F1
- recall alone
- candidate reduction alone
- model size
- novelty
- neural-vs-non-neural sophistication

---

# 5. Competition-score bottleneck analysis

The current baseline is:

```text
Blocking recall = 99.29%
F0.5            = 0.9566
Precision       = 0.9776
Recall          = 0.9112
FP              = 5
Threshold       = 0.700
```

This immediately suggests an important diagnostic:

> Candidate generation is already retrieving almost all true pairs, so the next score gain may come primarily from matching/decision quality rather than simply adding more candidates.

However, this MUST be verified with the entity-level blocking analysis before acting on it.

The agent must quantify the number of FNs caused by:

```text
blocking
vs
pairwise model
vs
threshold
vs
entity-level decision
```

The largest measured source of lost Macro F0.5 becomes the first optimization target.

---

# 6. Score-oriented experiment priority

After the baseline audit, prioritize experiments using:

```text
Priority =
expected F0.5 improvement
× confidence that the improvement generalizes
÷ complexity / instability
```

This is a planning heuristic, not a literal competition metric.

High-priority experiments are those that:

- directly target many current FNs/FPs,
- have a plausible mechanism,
- can be evaluated cleanly,
- are stable across folds.

Low-priority experiments are those that:

- add complexity without a measured failure target,
- improve only one split,
- improve a secondary metric while reducing F0.5,
- rely on test-set observations,
- depend on external information.

---

# 7. Validation protocol for every optimization

Before any optimization is accepted:

### Step 1 — Baseline

Run the unchanged V1 pipeline on the exact same split(s).

### Step 2 — Candidate experiment

Change only the intended component.

### Step 3 — Compare

Report:

```text
baseline mean F0.5
experiment mean F0.5
delta
standard deviation
precision delta
recall delta
FP delta
FN delta
```

### Step 4 — Stability

Check whether the improvement survives other entity-level splits/folds.

### Step 5 — Holdout confirmation

Run the selected configuration on the untouched holdout.

### Step 6 — Decision

Classify:

```text
KEEP
REJECT
INVESTIGATE
```

Do not replace V1 simply because one experiment produced a higher number.

---

# 8. Preventing leaderboard overfitting

If competition submissions are allowed during development, treat the public/private leaderboard as an **external generalization signal**, not as a tuning oracle.

Do NOT:

```text
submit
inspect score
change threshold
submit
change feature
submit
repeat indefinitely
```

This can overfit to leaderboard feedback.

Instead:

- freeze meaningful candidate versions,
- submit only major candidates,
- keep a record of each submission,
- make the next experiment based primarily on validation evidence,
- use leaderboard feedback only to assess whether validation predictions are generalizing.

If the leaderboard score disagrees strongly with validation:

```text
DO NOT immediately tune to the leaderboard.
```

First investigate:

- validation distribution mismatch,
- country distribution,
- source distribution,
- cardinality distribution,
- candidate-generation behavior,
- preprocessing assumptions,
- hidden data artifacts,
- overfitting.

---

# 9. Generalization stress tests

For promising models, evaluate performance on validation slices that mimic possible test distribution changes:

- countries
- S2 vs S3
- short vs long business names
- missing/weak addresses
- numeric-heavy addresses
- low/high candidate counts
- rare/common names
- exact vs corrupted names
- exact vs corrupted addresses
- singleton vs multi-match entities

A model that performs well overall but collapses on one important slice should not be blindly promoted.

---

---

# 10. Why the next phase starts with an audit

The current result is already strong. The biggest mistake now would be to start adding embeddings, neural networks, or random features before understanding the 4.34 percentage points between the current score and a better system.

Entity resolution normally separates the problem into:

1. **Blocking / candidate generation**
2. **Pairwise matching / scoring**
3. **Entity-level decision making**

Blocking exists because comparing every possible pair is computationally expensive; it should reduce comparisons while retaining true matches. This is well established in entity-resolution literature.

Research references:
- Li et al., *A Survey on Blocking Technology of Entity Resolution*, Journal of Computer Science and Technology, 2020.
- Papadakis et al., *Blocking and Filtering Techniques for Entity Resolution: A Survey*, ACM Computing Surveys, 2020.

Do not treat those papers as evidence that a particular method will work on this dataset. Use them only as methodological background.

---

# 11. PHASE A — Complete baseline audit

## A1. Trace the complete validation pipeline

For every component, determine exactly what data it is fitted on.

Audit:

- name normalization
- address normalization
- tokenization
- rare-token frequency calculation
- TF-IDF vocabulary
- TF-IDF IDF statistics
- nearest-neighbor / retrieval indexes
- blocking dictionaries
- character n-gram indexes
- word n-gram indexes
- feature preprocessors
- numeric/address extraction
- LightGBM training data
- threshold optimization
- any calibration
- any candidate caps
- any cached artifacts

Create a table:

| Component | Fitted on train only? | Uses validation records? | Uses validation labels? | Uses all records? | Leakage risk | Evidence |
|---|---|---|---|---|---|---|

The answer must be based on code inspection, not assumptions.

### Critical rule

A validation record may be used for **inference/evaluation**, but validation labels and validation relationships must not influence:

- vocabulary construction
- rare-token statistics
- blocking indexes
- model training
- threshold tuning before the final evaluation point
- feature selection
- experiment selection

If a component intentionally uses unlabeled validation/test-side text, document it explicitly and explain why it does not leak labels.

---

# 12. PHASE B — Candidate capping audit

Find every place where candidates are capped.

For every cap report:

- candidate count before cap
- candidate count after cap
- cap value
- selection rule
- whether selection is deterministic
- whether selection is based on retrieval score
- whether selection is arbitrary
- blocking recall before cap
- blocking recall after cap
- entity-level blocking recall before cap
- entity-level blocking recall after cap

The key question:

> Are we losing true matches because the candidate cap throws away candidates?

If yes, quantify the exact number and entities affected.

### Important

Never silently replace a broad candidate set with a smaller one and call it an optimization unless the final candidate set has been evaluated.

---

# 13. PHASE C — Entity-level blocking analysis

The existing **99.29% blocking recall is not enough**.

Calculate all of the following on validation:

### Pair-level

- total true pairs
- true pairs retrieved
- true pairs missed
- pair recall

### Entity-level

- number of S1 entities
- % S1 entities with **all** true matches retrieved
- % S1 entities with **at least one** true match retrieved
- % S1 entities with at least one true match missed
- % S1 entities where all true matches were missed
- number of missed true matches per S1 distribution

Report:

- mean
- median
- p90
- p95
- p99
- maximum

Also create:

```text
missed_true_matches_per_S1
0
1
2
3
4+
```

This matters because the competition metric is calculated at the S1-entity level rather than simply treating every pair as an independent example.

---

# 14. PHASE D — Build the validation error waterfall

For every validation ground-truth pair, assign it to exactly one final stage.

Desired waterfall:

```text
ALL TRUE MATCH PAIRS
        |
        v
+-----------------------+
| Retrieved by blocking |
+-----------------------+
        |
        +---- NO ----> BLOCKING FAILURE
        |
       YES
        |
        v
+-----------------------+
| Model score accepted? |
+-----------------------+
        |
        +---- NO ----> MODEL / THRESHOLD FAILURE
        |
       YES
        |
        v
+-----------------------+
| Decision layer keeps? |
+-----------------------+
        |
        +---- NO ----> DECISION-LAYER FAILURE
        |
       YES
        |
        v
      CORRECT
```

Produce exact counts and percentages.

Also calculate the same waterfall at the S1-entity level where meaningful.

---

# 15. PHASE E — Complete false-negative analysis

Produce a machine-readable CSV/TSV containing **every validation false negative**.

Required fields:

- `source1_entity_id`
- `true_entity_id`
- `source`
- `blocked`
- `retrieval_rank`
- `retrieval_score`
- `model_score`
- `threshold`
- `name_exact`
- `name_core_exact`
- `name_jaccard`
- `name_dice`
- `name_jw`
- `name_edit_similarity`
- `name_char_tfidf`
- `name_word_tfidf`
- `address_exact`
- `address_jaccard`
- `address_dice`
- `address_jw`
- `address_edit_similarity`
- `address_char_tfidf`
- `address_word_tfidf`
- `addr_num_jaccard`
- house_number_overlap
- postal_code_overlap
- country_equal
- `is_s2`
- `is_s3`
- candidate_count_for_s1
- candidate rank
- ground-truth cardinality for S1
- failure category

Every FN must be classified as one primary category:

### A. Blocking failure
The true pair never reached the matcher.

### B. Feature failure
The pair reached the matcher, but the feature representation does not adequately distinguish it.

### C. Model failure
The features contain useful evidence but the model ranks/scores the pair incorrectly.

### D. Threshold failure
The pair receives a useful/high score but the global threshold rejects it.

### E. Decision-layer failure
The pair is individually acceptable but is removed by multi-match/singleton/entity-level logic.

If a case belongs to multiple categories, record:

- `primary_failure`
- `secondary_failure`

Do not guess the category. Define objective rules for the classification.

---

# 16. PHASE F — Complete false-positive analysis

Produce the same kind of evidence table for every validation false positive.

For each FP report:

- S1 ID
- predicted entity ID
- source
- model score
- threshold
- candidate rank
- name similarities
- address similarities
- numeric overlap
- country equality
- source features
- candidate count
- true cardinality of the S1 entity
- nearest competing true/incorrect candidates if available
- decision-layer evidence
- failure category

Classify each FP as:

- ambiguous duplicate
- name collision
- address collision
- numeric-address collision
- source-specific artifact
- normalization artifact
- retrieval artifact
- model error
- threshold error
- decision-layer error

Again, categories must be supported by evidence.

---

# 17. PHASE G — Match-cardinality analysis

For validation S1 entities, calculate:

- 0 matches
- 1 match
- 2 matches
- 3 matches
- 4+ matches

Do this separately for:

- total matches
- S2 matches
- S3 matches

Also report:

- number of S1 entities in each bucket
- average candidates
- precision
- recall
- F0.5
- FP count
- FN count

The objective is to discover whether the current decision rule behaves differently for:

- singleton entities
- multi-match entities
- S2-only entities
- S3-only entities
- entities with matches in both sources.

---

# 18. PHASE H — Investigate source dependence

`is_s2` is currently among the strongest LightGBM features.

This MUST be investigated before adding new model classes.

Run:

### Experiment H1 — Full model

Current 51-feature model.

### Experiment H2 — Remove source indicators

Remove:

- `is_s2`
- `is_s3`
- any equivalent source-identifying features

Keep everything else unchanged.

### Experiment H3 — Source-specific models

Only if the audit shows a meaningful source-specific difference:

- one model for S1→S2
- one model for S1→S3

Compare against the shared model.

For every experiment report:

| Experiment | F0.5 | Precision | Recall | FP | FN | Threshold | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|

Do not conclude that a source feature is "bad" simply because it is important. Feature importance means the model uses it; it does not by itself prove leakage or overfitting.

---

# 19. PHASE I — Feature-family ablation

Run controlled ablations.

Every experiment must use:

- same validation split
- same candidate pairs
- same random seed
- same evaluation code
- same threshold-search procedure
- same entity-level metric

Experiments:

### I1 — Name only

Name-related features.

### I2 — Address only

Address-related features.

### I3 — Numeric only

Numeric/address-number/postal features.

### I4 — Cross-field only

Features that combine evidence across fields.

### I5 — Source/metadata only

Only metadata/source indicators.

### I6 — Full

All current features.

Report both performance and feature count.

Do not use ablation results to cherry-pick a metric that is not comparable to the baseline.

---

# 20. PHASE J — Threshold analysis

The current threshold is `0.700`.

Do not assume 0.700 is optimal.

Generate a validation sweep, for example:

```text
0.30
0.35
0.40
...
0.95
```

Use a sufficiently fine grid around the best region.

For every threshold report:

- Macro F0.5
- precision
- recall
- FP
- FN
- empty prediction rate
- average predicted matches/S1

Plot:

1. F0.5 vs threshold
2. precision vs threshold
3. recall vs threshold
4. FP/FN vs threshold

The challenge metric is precision-heavy. The F-beta definition used by scikit-learn is:

`F_beta = ((1 + beta^2) * TP) / ((1 + beta^2) * TP + FP + beta^2 * FN)`

For beta = 0.5, false positives receive greater relative penalty than false negatives.

Reference:
https://scikit-learn.org/stable/modules/model_evaluation.html

Do not optimize ordinary accuracy.

---

# 21. PHASE K — Entity-level decision analysis

The current system may be losing points after pairwise scoring.

Inspect the decision layer.

Test, where supported by validation evidence:

### K1 — Global threshold only

Accept every candidate above threshold.

### K2 — Threshold + singleton protection

For entities whose ground truth is empty, be conservative about weak matches.

### K3 — Threshold + score margin

Require the top candidate to beat the next candidate by a measured margin.

### K4 — Adaptive threshold

Only if validation demonstrates that different entity/cardinality regimes require it.

### K5 — Multi-match logic

Study whether:

- top-1 only
- top-k
- all scores above threshold
- threshold + margin
- threshold + source-specific rules

changes Macro F0.5.

Do not impose a hard top-1 rule unless the data proves that each S1 has at most one match. The challenge ground truth explicitly allows multiple S2/S3 matches.

---

# 22. PHASE L — Candidate-generation Pareto study

Current combined blocking:

- 99.29% pair recall
- 50.89 average candidates/S1

That is strong, but it may not be the best efficiency/recall point.

Study candidate-generation configurations such as:

### L1
Current combined blocking.

### L2
Exact/core/rare-token + name retrieval + address retrieval.

### L3
Address-heavy retrieval.

### L4
Name-heavy retrieval.

### L5
Different top-K values:

```text
name K = 3, 5, 10, 20
address K = 3, 5, 10, 20
```

### L6
Dynamic K based on retrieval confidence.

### L7
Union retrieval followed by a cheap deterministic reranker.

For each configuration report:

| Blocking config | Pair recall | Entity all-match recall | Entity any-match recall | Avg candidates | P95 candidates | Max candidates | Reduction | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

Do not select a configuration solely because it has fewer candidates.

The primary constraint is:

> Do not lose true matches merely to make the candidate file smaller.

---

# 23. PHASE M — Three-stage retrieval experiment

Only perform this if candidate analysis shows that 50.89 candidates/S1 is a meaningful computational bottleneck.

Potential architecture:

```text
Stage 1
Cheap broad blocking
        |
        v
~50 candidates
        |
        v
Stage 2
Cheap retrieval/reranking
        |
        v
~20–30 candidates
        |
        v
Stage 3
LightGBM final matcher
        |
        v
entity-level decision
```

Possible Stage-2 signals:

- retrieval similarity
- normalized name similarity
- normalized address similarity
- numeric overlap
- token overlap
- source
- exact/core matches

Do NOT use the final LightGBM score to construct the candidate file unless that is explicitly the intended architecture and the candidate file still represents the model input set correctly.

Evaluate Stage 2 for:

- candidate reduction
- true-match retention
- entity-level recall
- final Macro F0.5

---

# 24. PHASE N — Hard-negative analysis

The current model should be evaluated on difficult negatives, not just random negatives.

Construct hard negatives from:

- same rare token
- similar name
- similar address
- same postal/house number
- same source
- high retrieval score but not ground truth
- high pairwise model score but not ground truth

Measure:

- FP rate
- score distribution
- margin to true match
- which feature groups create confusion

The objective is to understand whether the model is failing because:

1. the positive evidence is weak,
2. the negative evidence is missing,
3. two entities genuinely look similar,
4. normalization destroys useful distinctions.

---

# 25. PHASE O — Feature engineering only after failure analysis

Do not add dozens of random features.

Prioritize features suggested by measured errors.

Potential candidates:

## Name

- prefix/suffix agreement
- token-position overlap
- token length similarity
- acronym similarity
- character n-gram overlap
- transliteration-independent normalized variants if already available
- repeated-token handling
- business suffix removal indicator

## Address

- house-number exact match
- house-number conflict
- postal-code exact match
- postal-code partial match
- street-token overlap
- locality/city token overlap
- numeric token alignment
- numeric conflict count
- address-token rarity

## Cross-field

- name strong + address weak
- address strong + name weak
- numeric agreement + name similarity
- name exact + address partial
- address exact + name partial

Do not add a feature unless it can be justified by a measured failure pattern.

---

# 26. PHASE P — Model optimization

Only after the audit and feature-family analysis.

Current model: LightGBM.

First investigate:

- class imbalance
- hard-negative ratio
- tree depth
- number of leaves
- minimum child samples
- learning rate
- number of boosting rounds
- regularization
- feature subsampling
- row subsampling
- early stopping

Use validation only for model selection and threshold tuning.

LightGBM supports validation sets and early stopping; use that functionality where appropriate rather than training an arbitrary number of trees.

Reference:
https://lightgbm.readthedocs.io/

Do not optimize AUC and assume F0.5 will improve.

---

# 27. PHASE Q — Validation split integrity

Confirm that the validation split is done at the **S1 entity level**.

Do NOT create a random pair-level train/validation split where the same S1 entity appears on both sides.

Check for:

- S1 entity overlap
- accidental duplicated records
- duplicate normalized records
- source leakage
- preprocessing leakage
- threshold-selection leakage

Save a reproducible split seed and document it.

---

# 28. PHASE R — Robustness checks

Run the selected pipeline on several validation slices:

### Country

- each country represented in validation

### Source

- S2
- S3

### Name quality

- short names
- long names
- exact normalized names
- heavily corrupted names

### Address quality

- missing addresses
- short addresses
- numeric-heavy addresses
- long addresses

### Candidate difficulty

- candidate count 0
- candidate count 1
- candidate count 2–5
- candidate count 6–20
- candidate count 21–50
- candidate count 51–100
- candidate count 100+

This reveals whether the average score hides catastrophic behavior on specific subsets.

---

# 29. PHASE S — Experiment tracking

Every experiment must produce a machine-readable record.

Recommended location:

```text
experiments/results/
```

Suggested structure:

```text
experiments/results/
├── baseline_v1/
├── baseline_audit/
├── blocking/
├── feature_ablation/
├── model_ablation/
├── threshold/
├── decision_layer/
├── hard_negatives/
└── final_selection/
```

Every experiment should save:

```text
config.json
metrics.json
feature_list.json
notes.md
```

For major experiments also save:

```text
predictions.tsv
false_positives.tsv
false_negatives.tsv
```

---

# 30. Required audit report

Create:

```text
experiments/results/baseline_audit/Baseline_Audit_Report.md
```

It MUST contain:

1. Baseline summary
2. Leakage audit
3. Candidate cap audit
4. Entity-level blocking metrics
5. Error waterfall
6. False-negative analysis
7. False-positive analysis
8. Match-cardinality analysis
9. Source-feature analysis
10. Feature-family ablation
11. Threshold curve
12. Decision-layer analysis
13. Candidate Pareto study
14. Hard-negative analysis
15. Failure distribution
16. Highest-value next experiment
17. Experiments that should NOT be pursued yet

The final recommendation must be evidence-based.

---

# 31. Decision tree for what to do after the audit

Use this exact logic.

## If blocking failures dominate

Work on:

- blocking recall
- retrieval ranking
- candidate cap
- multi-pass blocking
- dynamic top-K

Do NOT start with a neural matcher.

## If blocking recall is already near-perfect but FNs remain

Work on:

- pairwise features
- hard negatives
- model calibration
- threshold
- decision layer

## If FPs dominate

Work on:

- hard-negative training
- numeric conflict features
- address/name contradiction features
- threshold
- score margins
- source-specific behavior

## If threshold/decision failures dominate

Work on:

- entity-level thresholding
- score margins
- multi-match handling
- singleton behavior
- source-specific thresholds only if validation supports them

## If source-specific behavior dominates

Test:

- source-specific models
- source-specific thresholds
- source-specific features

## If one feature family dominates

Investigate whether:

- it genuinely contains strong signal,
- it creates leakage,
- it overfits source artifacts,
- or it is simply the strongest legitimate evidence.

---

# 32. What NOT to do yet

Until the audit is complete:

- Do NOT add BERT.
- Do NOT add sentence transformers.
- Do NOT add CLIP.
- Do NOT add LLM-based matching.
- Do NOT add external embeddings from business datasets.
- Do NOT add internet business lookup.
- Do NOT add a graph neural network.
- Do NOT replace LightGBM just because a neural model sounds more advanced.
- Do NOT increase candidate K blindly.
- Do NOT tune on the test set.
- Do NOT submit the current test outputs as the final "best" system.
- Do NOT create multiple Colab accounts or automate account rotation to bypass quotas.

A stronger competition system is the one that produces the highest **generalizing Macro F0.5**, not the one with the most model components.

The target is:

```text
HIGH VALIDATION F0.5
        +
STABLE ACROSS ENTITY-LEVEL SPLITS
        +
SURVIVES UNTOUCHED HOLDOUT
        +
NO LEAKAGE
        +
LOW UNNECESSARY COMPLEXITY
        =
HIGHER EXPECTED COMPETITION SCORE
```

---

# 33. Final model-selection protocol

When the audit is complete:

1. Freeze the validation split.
2. Freeze the evaluation implementation.
3. Freeze the candidate-generation protocol.
4. Run controlled experiments.
5. Record all results.
6. Select the configuration using validation Macro F0.5 and supporting diagnostics.
7. Refit the selected pipeline on the permitted training data.
8. Generate test candidates.
9. Generate test matches.
10. Validate the submission with:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

11. Confirm every predicted match exists in the final candidate file.
12. Confirm every test S1 entity appears exactly once.
13. Confirm no duplicate matches.
14. Confirm no self matches.
15. Confirm no invalid IDs.
16. Save the final configuration and model metadata.

---

# 34. Final submission reproducibility

The final repository should make it possible for another person to understand:

```text
raw TSV
  |
  v
normalization
  |
  v
blocking
  |
  v
candidate pairs
  |
  v
pairwise feature generation
  |
  v
LightGBM
  |
  v
entity-level decision
  |
  v
matching_results.tsv
```

The exact candidate set used for final matching must be reproducible.

The README must document:

- environment
- Python version
- dependencies
- training command
- validation command
- test inference command
- output paths
- model configuration
- blocking configuration
- threshold
- known limitations

---

# 35. Recommended immediate execution order

The agent should execute in this order:

```text
STEP 1
Baseline pipeline/code audit
        |
        v
STEP 2
Leakage audit
        |
        v
STEP 3
Candidate-cap audit
        |
        v
STEP 4
Entity-level blocking analysis
        |
        v
STEP 5
Error waterfall
        |
        v
STEP 6
Complete FN + FP analysis
        |
        v
STEP 7
Cardinality analysis
        |
        v
STEP 8
Source-feature ablation
        |
        v
STEP 9
Feature-family ablation
        |
        v
STEP 10
Threshold + decision analysis
        |
        v
STEP 11
Candidate Pareto study
        |
        v
STEP 12
Hard-negative analysis
        |
        v
STEP 13
Choose ONE highest-value next experiment
        |
        v
STEP 14
Run controlled optimization
        |
        v
STEP 15
Re-evaluate
        |
        v
STEP 16
Only then consider more advanced models/features
```

Do not skip directly from Step 1 to neural/embedding approaches.

---

# 36. Research notes / external methodological references

These references were checked while preparing this work order.

### Entity-resolution blocking

Li et al. describe blocking as a way to reduce the quadratic comparison space by restricting pairwise comparisons to plausible candidate blocks.

- Li, B.-H. et al. (2020). *A Survey on Blocking Technology of Entity Resolution*. Journal of Computer Science and Technology.
- DOI: `10.1007/s11390-020-0350-4`

https://doi.org/10.1007/s11390-020-0350-4

### Blocking and filtering

Papadakis et al. survey blocking and filtering as complementary efficiency techniques for large-scale entity resolution.

- Papadakis, G. et al. (2020). *Blocking and Filtering Techniques for Entity Resolution: A Survey*. ACM Computing Surveys.
- DOI: `10.1145/3377455`

### F-beta / precision-heavy evaluation

Scikit-learn documents F-beta as a weighted harmonic mean of precision and recall. For beta < 1, precision receives greater relative weight.

https://scikit-learn.org/stable/modules/model_evaluation.html

For this challenge:

```text
beta = 0.5
```

Therefore:

```text
F0.5 = 1.25 TP / (1.25 TP + FP + 0.25 FN)
```

This is why false-positive analysis and conservative entity-level decisions matter.

### LightGBM

LightGBM supports validation sets, early stopping, categorical features, multiple evaluation metrics, and model serialization.

https://lightgbm.readthedocs.io/

Use these capabilities where they improve reproducibility and controlled experimentation.

---

# 37. Deliverable checklist for the agent

Before reporting that this phase is complete, verify:

- [ ] Baseline V1 preserved
- [ ] Leakage audit completed
- [ ] Candidate cap locations identified
- [ ] Before/after cap recall measured
- [ ] Entity-level blocking metrics calculated
- [ ] Error waterfall generated
- [ ] Complete FN table generated
- [ ] Complete FP table generated
- [ ] Cardinality analysis generated
- [ ] `is_s2` / `is_s3` ablation completed
- [ ] Feature-family ablation completed
- [ ] Threshold sweep completed
- [ ] Decision-layer analysis completed
- [ ] Candidate Pareto study completed
- [ ] Hard-negative analysis completed
- [ ] Results saved under `experiments/results/`
- [ ] No test labels used
- [ ] No external business data used
- [ ] No arbitrary candidate truncation introduced
- [ ] Final candidate set remains the model input set
- [ ] Submission validator still passes
- [ ] Highest-value next experiment selected using measured evidence

---

# 38. Final instruction to the coding agent

**Do not try to impress me with model complexity.**

First prove:

1. where the current 4.34% F0.5 gap is coming from,
2. whether blocking is actually the bottleneck,
3. whether the model is making the mistakes,
4. whether threshold/decision logic is making the mistakes,
5. whether source-specific behavior is real,
6. and which single experiment has the highest expected value according to the validation evidence.

Only after that should the pipeline be changed.

Every optimization must answer:

> **What measured failure does this change target, and did that failure actually decrease after the change?**

If you cannot answer that from the validation experiments, do not make the change.
