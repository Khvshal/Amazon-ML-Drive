# Documentation Template — Business Entity Resolution

## Team Information
- **Team Name:** [Your Team Name]
- **Members:** [Names]

## 1. Methodology Overview

_Provide a high-level summary of your end-to-end approach._

## 2. Data Exploration & Preprocessing

_Describe any data cleaning, normalization, or feature extraction you applied to the raw business records._

### 2.1 Name Normalization
_How did you handle abbreviations, legal suffixes, DBA names, punctuation differences, etc.?_

### 2.2 Address Normalization
_How did you handle address abbreviations, transliteration variants, missing components, landmark references?_

### 2.3 Country Handling
_How did you handle the open-set country label (US, India, France)?_

## 3. Candidate Generation / Blocking Strategy

_Describe your blocking approach in detail. How did you reduce the O(n²) comparison space?_

- **Blocking keys used:**
- **Reduction ratio achieved:**
- **Recall ceiling on validation set:**

## 4. Feature Engineering

_List and describe all features fed into your matching model._

| Feature | Description | Source |
|---------|-------------|--------|
| | | |

## 5. Model Architecture

_Describe the ML model(s) used for matching._

- **Model type:**
- **Model license:** MIT / Apache 2.0
- **Parameter count:** ≤ 8B
- **Training details:**
- **Hyperparameters:**

## 6. Matching & Decision Logic

_How did you convert model scores into final match/no-match decisions? What threshold(s) did you use?_

## 7. Singleton Handling

_How did you handle Source 1 entities with no matches?_

## 8. Validation Strategy

_How did you validate performance on training data?_

- **Validation split strategy:**
- **Validation F_0.5 score:**

## 9. Results

| Metric | Validation | Public LB | Private LB |
|--------|-----------|-----------|------------|
| F_0.5  |           |           |            |
| Precision |        |           |            |
| Recall |           |           |            |

## 10. Key Insights & Lessons Learned

_What worked well? What didn't? What would you do differently?_

## 11. Reproducibility

_Exact steps to reproduce your results from scratch._

```bash
# Step-by-step commands
```

## 12. References

_Any papers, blog posts, or resources that informed your approach._
