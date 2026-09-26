# Business Entity Resolution — Source Code

This directory contains all source code for the entity resolution pipeline.

## Modules

Place your Python source files here. Suggested organization:

```
src/
├── __init__.py
├── preprocessing.py      # Data loading, cleaning, normalization
├── blocking.py           # Candidate generation / blocking strategy
├── feature_engineering.py # Feature extraction (string similarity, TF-IDF, etc.)
├── matching.py           # ML model for final match prediction
├── pipeline.py           # End-to-end orchestration
└── utils.py              # Shared helpers
```
