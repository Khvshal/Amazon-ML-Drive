# Business Entity Resolution Pipeline

## Overview
ML solution for resolving business entities across 3 independent data sources with noisy and inconsistent fields.

## Challenge
Given business records from 3 independent sources (Source 1, Source 2, Source 3), determine which records refer to the same real-world business entity. Source 1 is the deduplicated reference — find all matching records from Source 2 and Source 3 for each Source 1 entity.

## Evaluation
- **Metric:** F_β Score (β = 0.5) — precision-heavy
- **Formula:** F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

## Project Structure

```
Amazon-ML-Drive/
├── dataset/                              # Training & test data (TSV files)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── matching_results.tsv              # Final matches (leaderboard submission)
│   └── candidate_pairs.tsv              # Blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                          # All source code
│       ├── README.md                     # This file
│       └── requirements.txt             # Pinned dependencies
├── utils/
│   └── validate_submission.py           # Submission format validator
├── notebooks/                           # EDA & experiment notebooks
├── Documentation_template.md            # Methodology write-up
├── .gitignore
└── instruction.txt                      # Challenge instructions
```

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## How to Reproduce (End-to-End)

```bash
# 1. Data preprocessing & cleaning
python src/preprocessing.py

# 2. Blocking / candidate generation
python src/blocking.py

# 3. Feature engineering
python src/feature_engineering.py

# 4. Matching model training & inference
python src/matching.py

# 5. Generate output files
python src/pipeline.py
```

## Validate Submission

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Constraints
- Model: MIT/Apache 2.0 License, up to 8B parameters
- No external data lookups (APIs, databases, geocoding)
- All Source 1 test entities must appear in submission
- Output format: tab-separated (.tsv)
