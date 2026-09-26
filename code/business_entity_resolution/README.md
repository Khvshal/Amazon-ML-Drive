# Business Entity Resolution Pipeline

## Overview
High-performance Machine Learning pipeline for resolving business entities across 3 independent, noisy data sources with open-label countries, multi-script transliteration, and missing attributes.

Built for the **Amazon ML Challenge 2026** following strict precision-weighted evaluation rules ($F_{0.5}$ metric, 2:1 precision-to-recall weighting, strict singleton protection).

---

## Architecture & Workflow

```
Raw TSV Data
     │
     ▼
[Phase 1] Dataset Forensics & Missingness Analysis (`src/evaluation/dataset_report.py`)
     │
     ▼
[Phase 2] Multi-Representation Normalization (`src/preprocessing/`)
     ├── names.py (raw, lower, unicode, alphanumeric, core suffix-stripped, tokens, sorted)
     └── addresses.py (punctuation, numeric components, postal codes, tokenization)
     │
     ▼
[Phase 3] Multi-Pass Blocking & Retrieval (`src/blocking/`)
     ├── exact.py (Block A: exact name_lower, Block B: exact name_core, Block B2: sorted tokens)
     ├── rare_tokens.py (Block C: IDF-weighted rare name token inverted index)
     ├── numeric.py (Block D: address numeric & postal code overlap)
     ├── tfidf.py (Block E: char n-gram name, Block F: word TF-IDF name, Block G: char n-gram address)
     └── candidate_generator.py (Country-scoped Union + Candidate Cap)
     │   [Evaluated at 99.29% recall on validation subsample via `src/blocking_eval.py`]
     ▼
[Phase 4 & 5] Hard Negative Mining & Feature Engineering (`src/features/`)
     ├── name_features.py (exact matches, Levenshtein, Jaro-Winkler, token sort/set, char/token Jaccard & Dice)
     ├── address_features.py (edit distances, numeric overlap, postal match, missingness flags)
     └── pair_features.py (country match, source indicators, cross-field interaction features - 51 total features)
     │
     ▼
[Phase 6 & 7] Entity-Level Split & GBDT Matcher (`src/models/`, `src/train.py`)
     ├── base.py (BaseMatcher abstract interface)
     └── lightgbm_model.py (LightGBM binary classifier with validation early stopping)
     │
     ▼
[Phase 8 & 9] Optimization & Decision Layer (`src/models/threshold.py`, `src/decision.py`)
     ├── threshold.py (Entity-level Macro F_0.5 threshold sweep)
     └── decision.py (Margin-based decision engine: singleton guardrails, multi-match preservation)
     │
     ▼
[Phase 11] Systematic Error Analysis (`src/error_analysis.py`)
     └── False Positive / False Negative diagnostic breakdown
     │
     ▼
[Output & Validation] (`src/predict.py`, `utils/validate_submission.py`)
     ├── output/candidate_pairs.tsv
     ├── output/matching_results.tsv
     └── Submission schema & format verification
```

---

## Directory Structure

```
Amazon-ML-Drive/
├── Dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── loading.py                # Reusable TSV loader with strict separator checks
│   │   ├── evaluation/
│   │   │   └── dataset_report.py     # Phase 1 dataset forensics
│   │   ├── preprocessing/
│   │   │   ├── normalization.py      # Unicode, lowercase, whitespace utilities
│   │   │   ├── names.py              # Name variations and legal suffix stripping
│   │   │   └── addresses.py          # Address normalization and numeric extraction
│   │   ├── blocking/
│   │   │   ├── exact.py              # Exact and sorted token blocking
│   │   │   ├── rare_tokens.py        # IDF-weighted rare token blocking
│   │   │   ├── numeric.py            # Address numeric / postal code blocking
│   │   │   ├── tfidf.py              # Character and token TF-IDF sparse retrieval
│   │   │   └── candidate_generator.py# Multi-pass union and candidate capping
│   │   ├── blocking_eval.py          # Standalone blocking benchmark script
│   │   ├── features/
│   │   │   ├── name_features.py      # Pairwise name similarity features
│   │   │   ├── address_features.py   # Pairwise address similarity features
│   │   │   └── pair_features.py      # Unified 51-feature extractor and matrix builder
│   │   ├── models/
│   │   │   ├── base.py               # Abstract BaseMatcher interface
│   │   │   ├── lightgbm_model.py     # LightGBM pairwise matcher
│   │   │   └── threshold.py          # Macro F_0.5 evaluation and threshold sweeping
│   │   ├── decision.py               # Entity-level margin decision engine
│   │   ├── error_analysis.py         # Error analysis and diagnostic reporting
│   │   ├── train.py                  # End-to-end training and evaluation pipeline
│   │   └── predict.py                # Submission generator and validator runner
│   ├── README.md
│   └── requirements.txt
├── notebooks/
│   ├── 01_dataset_forensics.ipynb    # Interactive Phase 1 EDA & forensics
│   └── colab_runner.ipynb            # Cloud runner for src/ pipeline
├── experiments/
│   ├── models/                       # Saved trained model artifacts (.joblib)
│   ├── results/                      # Machine-readable experiment logs (.json)
│   └── logs/
├── output/
│   ├── candidate_pairs.tsv           # Blocking candidate sets
│   └── matching_results.tsv          # Scored submission matches
└── utils/
    └── validate_submission.py        # Official challenge format validator
```

---

## Setup & Installation

```bash
# Recommended: Python 3.10+
pip install -r code/business_entity_resolution/requirements.txt
```

Core dependencies:
- `pandas>=2.0.0`, `numpy>=1.24.0`, `scipy>=1.10.0`
- `rapidfuzz>=3.0.0` (C++ optimized Levenshtein and token similarity)
- `scikit-learn>=1.3.0` (TF-IDF vectorizers and sparse linear algebra)
- `lightgbm>=4.0.0` (Gradient boosted decision trees)

---

## Reproduction Guide

### 1. Run Phase 1 Forensics
```bash
python code/business_entity_resolution/src/evaluation/dataset_report.py
```
Outputs: `experiments/results/phase1_forensics.txt`.

### 2. Benchmark Multi-Pass Blocking
```bash
python code/business_entity_resolution/src/blocking_eval.py
```
Evaluates Blocks A through G individually and in union. Achieves **99.29% recall** on validation split. Outputs: `experiments/results/blocking_eval_sample.json`.

### 3. Train Matcher & Optimize Decision Threshold
```bash
python code/business_entity_resolution/src/train.py --sample-s1 6000 --candidate-cap 40 --exp-name lgbm_v1
```
Executes S1-entity-level split, blocking, 51-feature extraction, LightGBM training with early stopping, threshold sweep for Macro $F_{0.5}$, entity margin decision engine, and error analysis.
Outputs:
- Model: `experiments/models/lgbm_v1.joblib`
- Metrics & Error Log: `experiments/results/lgbm_v1.json`

### 4. Generate Predictions & Validate Submission
```bash
python code/business_entity_resolution/src/predict.py --model-path experiments/models/lgbm_v1.joblib --threshold 0.70
```
Generates:
- `output/candidate_pairs.tsv`
- `output/matching_results.tsv`
Automatically invokes `utils/validate_submission.py` to confirm 100% compliance with submission format.

---

## Validation Benchmark Results

| Stage / Component | Metric | Value |
|-------------------|--------|-------|
| Multi-Pass Blocking Union | Blocking Recall | **99.29%** |
| Multi-Pass Blocking (capped @ 40) | Blocking Recall | **94.11%** |
| Pairwise Matcher (LightGBM) | Validation Accuracy | High |
| Optimal Decision Threshold | Cutoff ($\theta$) | **0.700** |
| Post-Decision Performance | **Macro $F_{0.5}$** | **0.9566** |
| Post-Decision Performance | **Macro Precision** | **0.9776** |
| Post-Decision Performance | **Macro Recall** | **0.9112** |
| Singleton Protection | **Singleton $F_{0.5}$** | **1.0000** (0 false merges) |
| Submission Validation | Schema & Format Check | **PASS — Safe to submit** |
