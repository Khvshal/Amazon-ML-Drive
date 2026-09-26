# Amazon ML Hackathon — Business Entity Resolution
## Agent Instructions

## 0. Scoring reminders (do not lose sight of these)
- Leaderboard metric: macro-averaged F_0.5 per S1 entity, precision-weighted 2:1. `candidate_pairs.tsv` and blocking code are separately reviewed — a **smaller** candidate set per S1 entity at fixed recall ranks higher.
- Singletons: correct empty prediction = 1.0 for that row; any false merge on a true singleton = 0.0. Bias decisions toward "no match" under uncertainty.
- `country` is an open-label field. France appears only in test. No hardcoded `{US, India}` branching, no one-hot encoding, no silent-drop on unseen labels.
- No external lookups (registries, geocoding, entity-resolution APIs). This is a compute constraint on *data*, not on your own infrastructure.
- Final matching model: MIT/Apache-2.0, ≤8B params. Applies to any pretrained component (e.g. an embedding model), not to a from-scratch LightGBM classifier.

## 1. Agent behavior rules — hard constraints, not suggestions
**MUST NOT:**
- Implement the full pipeline before dataset forensics is done and reported.
- Assume one model or one blocking pass is sufficient.
- Optimize for plain accuracy instead of macro F_0.5.
- Split validation at the pair level (leaks S1 entities across train/val) — split at the S1-entity level only.
- Use external business information of any kind.
- Hard-code US/India/France or force one-to-one matching.
- Bury load-bearing logic inside notebooks — notebooks import from `src/`, never the reverse.
- Overwrite a previous experiment's results without being told to.
- Claim an approach is better without a measured validation number attached.
- Tune anything against hidden test labels.
- Fabricate performance numbers.
- Build any form of Colab account rotation, automated account switching, proxy rotation, or distributed workers intended to bypass Colab's resource/session limits — including manually splitting one job across multiple logins for that purpose. Use one account's resources legitimately; if that's too slow, reduce the job, don't multiply accounts.

**MUST:**
- Inspect the repo and the actual data before writing pipeline code.
- Explain non-obvious design decisions inline (comments or a short note in the experiment log).
- Build incrementally, run a check after each step, benchmark alternatives before committing to one.
- Record every serious experiment to `experiments/results/`.
- Run error analysis after every validation run.
- Keep CPU execution always possible; use GPU only where it's demonstrably useful (embedding inference), never as a hard requirement.

## 2. Repo structure
```
Amazon-ML-Drive/
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── loading.py                # reusable TSV loader, sep="\t" enforced
│   │   ├── evaluation/
│   │   │   └── dataset_report.py     # Phase 1 forensics
│   │   ├── preprocessing/
│   │   │   ├── normalization.py
│   │   │   ├── names.py
│   │   │   └── addresses.py
│   │   ├── blocking/
│   │   │   ├── exact.py              # Block A/B
│   │   │   ├── rare_tokens.py        # Block C
│   │   │   ├── numeric.py            # Block D
│   │   │   ├── tfidf.py              # Blocks E/F/G
│   │   │   └── candidate_generator.py  # union + cap + Block H combos
│   │   ├── features/
│   │   │   ├── name_features.py
│   │   │   ├── address_features.py
│   │   │   └── pair_features.py
│   │   ├── models/
│   │   │   ├── base.py
│   │   │   ├── lightgbm_model.py
│   │   │   └── threshold.py
│   │   ├── decision.py               # entity-level margin-based decision layer
│   │   ├── error_analysis.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── README.md
│   └── requirements.txt
├── Dataset/{train,test}/
├── notebooks/
│   ├── 01_dataset_forensics.ipynb
│   └── colab_runner.ipynb            # calls src/, no duplicated logic
├── experiments/
│   ├── configs/
│   ├── results/
│   └── logs/
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
└── utils/validate_submission.py
```

## 3. Environment
```
pandas
numpy
rapidfuzz
scikit-learn
lightgbm
sentence-transformers   # optional, GPU-friendly, only for semantic blocking
faiss-cpu               # optional, only if using embeddings
tqdm
```
Everything except the optional embedding step runs on CPU by design (Section 27 below is non-negotiable: never make the pipeline require a GPU).

## 4. Phase 1 — Dataset forensics (do this before any modeling, report back before moving on)
Build `notebooks/01_dataset_forensics.ipynb` and `src/evaluation/dataset_report.py`. Report, don't assume:
- Row counts per source; missingness per field per source.
- Country distribution (report frequencies as found — do not assume the set is closed).
- Match cardinality: for each S1 entity, count of true S2 matches, true S3 matches, total — histogram of 0/1/2/3+.
- For every ground-truth positive pair: exact raw name equality, normalized name equality, character similarity, token similarity; same four for address; country equality. Distributions + representative examples.
- Goal: find the actual corruption patterns in *this* dataset, not the ones listed in the problem statement as illustrative examples.

Do not write blocking code until this report exists and you've looked at it.

## 5. Phase 2 — Normalization
Never destroy the raw value; keep multiple representations per field:
- Names: `name_raw`, `name_lower`, `name_unicode_normalized`, `name_alphanumeric`, `name_core` (legal-suffix stripped), `name_tokens`, `name_sorted_tokens`, `name_compact`.
- Addresses: punctuation/whitespace/Unicode normalization, tokenization, numeric extraction, postal-code extraction, house-number extraction, rare-token extraction.
- Keep both the plain-normalized and suffix-stripped name representations — some blocks want one, some want the other.
- Don't assume one country's address format for all rows — no format-specific parsing gated on the `country` field.

## 6. Phase 3 — Multi-pass blocking (the graded component — build this carefully)
Implement each block as an independent, swappable retrieval strategy, and evaluate each one's recall/candidate-count on its own before combining:
- **A — Exact normalized name**
- **B — Exact core (suffix-stripped) name**
- **C — Rare name tokens** — weight by inverse document frequency, prioritize rare-token matches
- **D — Postal/numeric components** — generic numeric/postal-code/address-number overlap, no country-specific hardcoding unless the Phase-1 report justifies it
- **E — Character TF-IDF name retrieval** (`analyzer="char_wb"`, `ngram_range=(2,5)`, tune `min_df` from data)
- **F — Token/word TF-IDF name retrieval**
- **G — Character TF-IDF address retrieval**
- **H — Combined name+address representation** — only if experiments show it beats the individual passes

Efficiency: no full Cartesian product except on a tiny diagnostic subset. Use inverted indexes, sparse TF-IDF nearest-neighbor retrieval, and batched lookups so this scales to the full dataset.

**Blocking evaluation**, per block and for the final union, computed against training ground truth:
- blocking recall = true positive pairs recovered / total true positive pairs
- average / median / 95th-percentile / max candidates per S1
- total candidate pairs, candidate reduction ratio

Target: very high recall **and** a small set — this is an explicit trade-off to search, not a single number to maximize. `output/candidate_pairs.tsv` must be the exact final set fed to the classifier — if a later stage filters candidates further, that filtered set is what gets written, not the raw union.

## 7. Phase 4 — Hard negative mining
Negatives for training come from candidates the blocking stage produced that are **not** ground-truth matches — these are hard negatives and far more informative than random negatives. Check and report the resulting class balance before training; don't let it silently become extreme.

## 8. Phase 5 — Pairwise feature engineering
Name: exact/normalized/core equality, Jaccard, Dice, edit similarity, character TF-IDF cosine, token TF-IDF cosine, token overlap, length ratio, token-count difference.
Address: same family, plus numeric overlap, postal-code equality, house-number equality, rare-token overlap, length ratio.
Metadata: country equality, source pair (S1×S2 vs S1×S3), both-missing flags.
No single feature should be assumed decisive — this gets checked in error analysis (Phase 11), not asserted up front.

## 9. Phase 6 — Model layer
Start with LightGBM on pairwise features as the baseline before anything fancier. Keep `src/models/` modular (`base.py` interface) so additional models can be swapped in and compared, not bolted on.

## 10. Phase 7 — Validation split
Split at the **S1-entity level**, then run the entire pipeline (normalization → blocking → features → model → decision → metric) independently on the validation entities. Never `train_test_split` on raw pair rows — that leaks entities across the split.

## 11. Phase 8 — Threshold optimization
Do not assume 0.5. Sweep thresholds, optimize for macro F_0.5 on the validation split only — never against hidden test labels. Also check whether a per-source-pair or singleton-specific threshold does better than one global cutoff, and record the comparison.

## 12. Phase 9 — Entity-level decision layer (this is where singleton/precision performance is actually won or lost)
The model outputs P(match | S1, candidate) per pair. Convert to a final ID set per S1 using, at minimum:
- top score, second-highest score, score margin between them
- count of candidates above threshold
Use these to separate: clear singleton (nothing near threshold), one clear high-confidence match, legitimate multi-match, and ambiguous cluster needing a different rule. Do not force one-to-one — multi-match is valid and expected.

## 13. Phase 10 — Source-pair analysis (optional, data-driven)
Compare S1↔S2 vs S1↔S3 on name/address similarity, missingness, positive-pair distributions, and blocking recall. Only introduce `source_pair` as a feature or separate models if this comparison shows it matters — don't do it speculatively.

## 14. Phase 11 — Error analysis (after every serious validation run)
Log false positives with: S1 ID, candidate ID, score, name/address similarities, country, numeric/postal features.
Log false negatives with: S1 ID, true candidate, whether it survived blocking at all, its retrieval rank, model score, feature values.
Classify failures into categories: name variation, address variation, typo, abbreviation, legal suffix, transliteration, missing field, landmark address, numeric mismatch, over-aggressive blocking, threshold miscalibration. The next experiment is driven by which category dominates — not by intuition.

## 15. Experiment tracking
Every experiment writes a machine-readable record to `experiments/results/`, e.g.:
```json
{
  "experiment": "block_CEF_lgbm_v3",
  "blocking": "C+E+F union, cap=15",
  "model": "lightgbm",
  "avg_candidates": 8.4,
  "median_candidates": 6,
  "p95_candidates": 22,
  "max_candidates": 41,
  "blocking_recall": 0.981,
  "candidate_reduction_ratio": 0.9993,
  "pairwise_precision": 0.71,
  "pairwise_recall": 0.94,
  "macro_precision": 0.85,
  "macro_recall": 0.88,
  "macro_f05": 0.856,
  "singleton_f05": 0.91,
  "runtime_sec": 142
}
```
Never keep results only in your head or in a chat transcript. Don't delete a previous experiment's record just because a newer one scored higher.

## 16. Notebook policy
Notebooks (`notebooks/`) are for exploration, visualization, and one-off analysis only. They must import from `src/`, never contain the only implementation of anything load-bearing. If you find yourself writing a function in a notebook that the pipeline depends on, move it to `src/` immediately.

## 17. CPU/GPU separation
The full pipeline must run on CPU: pandas, normalization, inverted indexes, sparse TF-IDF, similarity features, LightGBM, evaluation. GPU is optional and limited to: neural/transformer embeddings, large-scale embedding retrieval, and other clearly-scoped experiments. Never make the final, submitted pipeline require a GPU.

## 18. Colab usage — legitimate scope only
`notebooks/colab_runner.ipynb` should clone the repo, install pinned dependencies, locate the dataset, and call the existing `src/` pipeline — no duplicated logic. Use it to get GPU time for the optional embedding-blocking experiment or to run one config at a time. **Do not** build account rotation, automated account switching, or distributed workers across multiple accounts to get around a single account's limits — that's explicitly out of scope regardless of intent, and it's not a bottleneck you actually have: nothing in this pipeline requires more than one account's worth of compute to run correctly, just to run instantly.

## 19. Output & submission validation
- `output/matching_results.tsv`: exactly one row per test S1 entity, empty string for singletons, S2/S3 IDs only, no duplicates.
- `output/candidate_pairs.tsv`: exactly one row per test S1 entity, the literal final set passed to the model — not an earlier, later-filtered blocking stage. Every predicted match must appear here.
- Always run before declaring anything finished:
```
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir Dataset/test
```

## 20. Final pipeline shape
```
raw data → load → normalization → multi-pass blocking → candidate union/cap
→ candidate_pairs.tsv → pairwise features → trained matcher → probabilities
→ entity-level decision → matching_results.tsv → submission validation
```

## 21. Compliance checklist (fill into the methodology doc, verbatim)
- [ ] Blocking candidate count per S1 entity: min/median/p95/max, and blocking recall on the training validation split.
- [ ] Any pretrained model used: name, license, parameter count — must be MIT/Apache-2.0, ≤8B params.
- [ ] No external API/database calls anywhere in `src/`.
- [ ] France (and any unseen country) flows through normalization/blocking/features without special-casing or crashing.
- [ ] No account-rotation or multi-account compute-limit workarounds anywhere in the repo or run history.

## 22. First assignment — do not skip ahead
1. Inspect the repo as it currently exists.
2. Inspect all dataset files and schemas.
3. Build `notebooks/01_dataset_forensics.ipynb` + `src/loading.py`.
4. Produce the Phase 1 report (Section 4) in full.
5. **Stop and report findings before writing any blocking code.** The blocking design in Section 6 should be adjusted based on what Phase 1 actually shows, not implemented generically before you've looked.

## 23. Definition of success
Not "we trained LightGBM." Success is: high blocking recall + a small candidate set, strong pairwise discrimination, correct singleton handling, correct multi-match handling, high macro F_0.5, reproducible code, auditable experiment records, and a submission that passes validation on the first try. Optimize the whole system — a strong model behind weak blocking, or tight blocking feeding a sloppy threshold, both cap your score the same way.