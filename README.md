# Semantic Duplicate & Near-Plagiarism Detection Engine

> **A transparent duplicate and near-plagiarism detection engine for English and Persian text. MinHash, LSH, SimHash, TF-IDF scoring, and the calibrated ensemble are implemented inside the project; no ready-made MinHash/LSH/SimHash implementation is used.**

---

## Architecture Overview

The system features two parallel pipeline paths designed to capture different dimensions of document similarity:

```
                  +--------------------------------+
                  |      Raw Document Pair         |
                  +--------------------------------+
                                  |
                                  v
                  +--------------------------------+
                  |  Preprocessing & Normalization |
                  +--------------------------------+
                                  |
                  +---------------+---------------+
                  |                               |
                  v                               v
         [Path 1: Lexical]               [Path 2: Semantic-Vector]
   +---------------------------+   +---------------------------+
   |   3-5 Word Shingling      |   |   TF-IDF Weight Fitting   |
   +---------------------------+   +---------------------------+
                  |                               |
                  v                               v
   +---------------------------+   +---------------------------+
   |  MinHash Signature (128)  |   |    64-bit SimHash Sign    |
   +---------------------------+   +---------------------------+
                  |                               |
                  v                               v
   +---------------------------+   +---------------------------+
   |  LSH Banding Buckets (64) |   | Hamming Distance Boundary |
   +---------------------------+   +---------------------------+
                  |                               |
                  v                               v
   +---------------------------+   +---------------------------+
   |   Exact Jaccard Filter    |   |     Duplicate Verdict     |
   +---------------------------+   +---------------------------+
```

### 1. Lexical Path (MinHash + LSH)
* **Goal**: Capture exact or near-exact word overlap.
* **Mechanism**: Custom 3-5 word shingling, universal hashing using Mersenne Prime coefficients ($p = 2^{61}-1$), Locality Sensitive Hashing (LSH) candidate generation, and exact Jaccard similarity filtering.
* **Complexity**: Reduces the search space from $O(n^2)$ to sub-quadratic candidate pairs.

### 2. Vector Path (SimHash)
* **Goal**: Measure global semantic/paraphrase similarity.
* **Mechanism**: Sublinear term-frequency weights ($\text{TF} = 1 + \ln(\text{tf})$), smoothed inverse document frequency ($\text{IDF}$), 64-bit stable fingerprint accumulation via BLAKE2b, and Hamming distance classification.

---

## Directory Structure compliant with Repository Spec

```text
semantic-plagiarism-engine/
├── README.md                           # Comprehensive user manual
├── requirements.txt                    # Project dependencies
├── pyproject.toml                      # Package configuration
├── .gitignore                          # Git exclusions
├── .github/
│   └── workflows/
│       └── tests.yml                   # CI pipeline for automated testing
├── docs/                               # Academic and technical reports
│   ├── project_spec.tex                # XeLaTeX source code (XB Niloofar font)
│   └── project_spec.pdf                # Compiled technical report PDF
├── data/                               # Dataset directory
│   ├── sample_corpus/                  # Sample documents for folder-wide LSH tests
│   │   ├── doc_01.txt
│   │   ├── doc_02.txt
│   │   └── doc_03.txt
│   ├── raw/                            # Large datasets (e.g. Quora subset; Git-ignored)
│   └── processed/
│       └── sample_pairs.csv            # 30-pair validation set
├── src/plagiarism_engine/              # Source code of the engine
│   ├── __init__.py
│   ├── preprocessing.py                # Text normalization & stopword handling
│   ├── minhash.py                      # MinHash signature generation
│   ├── lsh.py                          # LSH banding and candidate selection
│   ├── simhash.py                      # SimHash fingerprinter with TF-IDF weighting
│   ├── dataset.py                      # Streamers and CSV data loaders
│   ├── evaluation.py                   # Precision, Recall, F1 metrics
│   ├── tfidf_cosine.py                 # Sparse TF-IDF cosine baseline
│   ├── ensemble.py                     # Calibrated 39-feature lexical ensemble
│   ├── bonus.py                        # Bonus modules (lemmatizer, AdaptiveLSH, etc.)
│   └── cli.py                          # argparse command-line interface
├── notebooks/
│   └── exploration.ipynb               # Persian RTL presentation notebook
├── tests/                              # Unit testing suite
│   ├── test_engine.py                  # Core engine tests
│   ├── test_bonus.py                   # Bonus feature verification
│   └── test_ensemble.py                # Ensemble calibration and serialization tests
└── outputs/                            # Target directory for execution outputs
    ├── metrics.csv                     # Evaluation metrics output
    ├── candidates.csv                  # Candidate pairs from LSH
    ├── pair_predictions.csv            # Individual pair similarity verdicts
    ├── bonus_metrics.csv               # Comparison results for bonus modules
    ├── two_file_compare.json           # Output of single-pair comparison
    ├── calibrated_metrics.csv          # Held-out validation/test results
    ├── calibrated_predictions.csv      # Held-out test predictions
    └── calibrated_ensemble.json        # Saved calibrated model
```

---

## Installation Guide (Isolated Environment)

### Windows (PowerShell)
```powershell
# Create and activate virtual environment
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# Install in editable mode
python -m pip install --upgrade pip
python -m pip install -e .
```

### Linux / macOS (Bash)
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
python -m pip install --upgrade pip
python -m pip install -e .
```

---

## Command Line Interface (CLI) Execution Guide

The engine exposes five verified commands for end-to-end operation.

### 1. Document-to-Document Comparison (`compare`)
Compares two documents and saves a structured JSON file detailing preprocessing token sizes, LSH candidacy status, exact Jaccard similarity, and SimHash Hamming distance.

```powershell
python -m plagiarism_engine.cli compare `
  --file-a data/sample_corpus/doc_01.txt `
  --file-b data/sample_corpus/doc_02.txt `
  --output outputs/two_file_compare.json
```

**Expected JSON Output Format (`outputs/two_file_compare.json`):**
```json
{
  "file_a": "data/sample_corpus/doc_01.txt",
  "file_b": "data/sample_corpus/doc_02.txt",
  "parameters": {
    "shingle_size": 3,
    "num_perm": 128,
    "bands": 64,
    "jaccard_threshold": 0.25,
    "simhash_max_distance": 25,
    "simhash_ngram_size": 1
  },
  "preprocessing": {
    "tokens_a": 5,
    "tokens_b": 6,
    "shingles_a": 3,
    "shingles_b": 4,
    "language_a": "en",
    "language_b": "en"
  },
  "minhash_lsh": {
    "lsh_candidate": true,
    "exact_jaccard": 0.166667,
    "minhash_similarity": 0.242188,
    "is_similar": false
  },
  "simhash": {
    "fingerprint_a_hex": "bc4a87d769d6bc63",
    "fingerprint_b_hex": "acda97d609f4bc7b",
    "hamming_distance": 11,
    "similarity": 0.828125,
    "is_similar": true
  }
}
```

---

### 2. Folder-Wide Plagiarism Scanning (`corpus`)
Indexes all text files inside a folder using LSH and extracts candidate plagiarism pairs.

```powershell
python -m plagiarism_engine.cli corpus `
  --data data/sample_corpus `
  --threshold 0.25 `
  --shingle-size 3 `
  --output outputs/candidates.csv
```

---

### 3. Evaluation on Labeled Pairs Dataset (`pairs`)
Computes accuracy, precision, recall, and F1-score on a labeled dataset. 

* **On the 30-pair validation set:**
```powershell
python -m plagiarism_engine.cli pairs `
  --pairs data/processed/sample_pairs.csv `
  --text-col-a text_a `
  --text-col-b text_b `
  --label-col label `
  --id-col pair_id `
  --output outputs/metrics.csv `
  --predictions-output outputs/pair_predictions.csv
```

* **On the 5,000 Quora Question Pairs subset:**
```powershell
python -m plagiarism_engine.cli pairs `
  --pairs data/raw/quora/train.csv `
  --text-col-a question1 `
  --text-col-b question2 `
  --label-col is_duplicate `
  --limit 5000 `
  --output outputs/metrics.csv `
  --predictions-output outputs/pair_predictions.csv
```

---

### 4. Evaluation of Bonus Architectural Upgrades (`bonus-eval`)
Compares the performance metrics (Precision, Recall, F1) of the standard paths versus the newly designed bonus components (Adaptive LSH parameter fitting, rule-based Persian lemmatizer, and hybrid char-level SimHash).

```powershell
python -m plagiarism_engine.cli bonus-eval `
  --pairs data/raw/quora/train.csv `
  --text-col-a question1 `
  --text-col-b question2 `
  --label-col is_duplicate `
  --limit 5000 `
  --output outputs/bonus_metrics.csv
```

---

## Verification of Unit Tests

Run the test suite using `unittest` to verify code correctness and consistency:
```powershell
# Set PYTHONPATH and execute test runner
$env:PYTHONPATH="src"; python -m unittest discover -s tests -v
```

All 20 tests covering preprocessing, shingling, universal hashing, SimHash weights, Adaptive LSH, Persian lemmatization, TF-IDF cosine, calibration, serialization, and the ensemble CLI are verified to pass successfully.

---

## Improved Short-Text and Evaluation Options

The `pairs`, `compare`, and `corpus` commands now support the following optional improvements:

- `--adaptive-short-text`: preserves stopwords in short texts and uses unigram/bigram fallbacks.
- `--short-text-token-limit`: controls the maximum length treated as short text (default: `10`).
- `--simhash-mode hybrid`: uses word features together with character 3-grams.
- `--cosine-threshold`: enables the optional sparse TF-IDF cosine baseline.
- `candidate_recall`: is reported for the LSH candidate-generation stage.

Example:

```powershell
python -m plagiarism_engine.cli pairs `
  --pairs data/processed/sample_pairs.csv `
  --text-col-a text_a `
  --text-col-b text_b `
  --label-col label `
  --id-col pair_id `
  --adaptive-short-text `
  --simhash-mode hybrid `
  --simhash-max-distance 22 `
  --jaccard-threshold 0.05 `
  --cosine-threshold 0.32 `
  --output outputs/metrics_v2.csv `
  --predictions-output outputs/predictions_v2.csv
```

The updated test suite contains 20 tests.


---

## 5. Calibrated Ensemble (`calibrate`)

The new bonus model combines 39 transparent lexical features, including word and character TF-IDF cosine similarity, token overlap, word-order similarity, bigram overlap, length ratio, question-word agreement, and negation mismatch. A small logistic-regression layer is trained with Newton/IRLS updates implemented in this repository using NumPy. The labeled data is split deterministically into train, validation, and held-out test partitions. IDF values and classifier weights are fitted only on the training partition, while the decision threshold is selected only on validation data.

```powershell
python -m plagiarism_engine.cli calibrate `
  --pairs data/raw/quora/train.csv `
  --text-col-a question1 `
  --text-col-b question2 `
  --label-col is_duplicate `
  --id-col id `
  --adaptive-short-text `
  --output outputs/calibrated_metrics.csv `
  --predictions-output outputs/calibrated_predictions.csv `
  --model-output outputs/calibrated_ensemble.json
```

The saved model can be used when comparing two files:

```powershell
python -m plagiarism_engine.cli compare `
  --file-a data/sample_corpus/doc_01.txt `
  --file-b data/sample_corpus/doc_02.txt `
  --ensemble-model outputs/calibrated_ensemble.json `
  --output outputs/two_file_compare_ensemble.json
```

It can also score only the candidate pairs generated by LSH in folder mode:

```powershell
python -m plagiarism_engine.cli corpus `
  --data data/sample_corpus `
  --ensemble-model outputs/calibrated_ensemble.json `
  --output outputs/candidates_ensemble.csv
```

### Held-out results on the included 15,000-pair Quora subset

The table below uses one deterministic stratified 60/20/20 train/validation/test split. These results are not measured on the same rows used to fit the model.

| Method | Test Precision | Test Recall | Test F1 | Test Accuracy |
|---|---:|---:|---:|---:|
| Character TF-IDF cosine | 0.509 | 0.821 | 0.628 | 0.638 |
| Token Jaccard | 0.503 | 0.877 | 0.639 | 0.631 |
| **Calibrated lexical ensemble** | **0.543** | **0.848** | **0.662** | **0.677** |

The calibrated ensemble improves held-out F1 by roughly 2.3 to 3.4 percentage points over the two transparent baselines on the same split. The exact values are stored in `outputs/calibrated_metrics.csv`.

> The Quora dataset measures duplicate-question and paraphrase similarity. MinHash/LSH remains especially useful for near-copy and plagiarism candidate generation, while the calibrated ensemble is most useful when labeled pairs are available for the target domain.
