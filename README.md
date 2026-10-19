# E-Commerce Product Trust Assessor

An explainable AI system that evaluates e-commerce product reliability and generates a Product Trust Score (0–100) using customer reviews, ratings, listing metadata, and machine learning.

---

## Overview

This project analyzes product reviews and listing information from the Amazon Reviews 2023 dataset to help users assess product reliability.

Review sentiment, rating quality, review consistency, verified purchases, suspicious review patterns, and listing completeness are combined into a weighted trust score. Each assessment includes a component breakdown, evidence coverage, confidence level, strengths, warnings, and a step-by-step explanation of how the final score was calculated.

The system also uses machine learning to identify anomalous review patterns and predict review helpfulness, with SHAP explanations to make model predictions more interpretable.

The trust score is intended for screening and exploration, not as definitive proof of product quality or review authenticity.

---

## Tech Stack

- **Machine Learning** — XGBoost, Isolation Forest, scikit-learn
- **Natural Language Processing** — DistilBERT for sentiment analysis
- **Explainable AI** — SHAP
- **Backend** — Python
- **Frontend** — NiceGUI
- **Database** — SQLite, SQLAlchemy
- **Data Storage** — Parquet, Pandas
- **Dataset** — Amazon Reviews 2023 (Hugging Face)

---

## Project Structure

```text
e-commerce-product-trust-assessor/
├── main.py
├── requirements.txt
├── .env.example
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_models.py
│   ├── assess_products.py
│   └── run_pipeline.py
├── trust_assessor/
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── nlp/
│   ├── ml/
│   ├── scoring/
│   ├── explain/
│   ├── db/
│   ├── services/
│   └── ui/
│       └── pages/
├── data/
│   ├── raw/
│   ├── processed/
│   └── trust_assessor.sqlite
└── models/
```

---

## Running the Project

### Backend & Dashboard

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source .venv/bin/activate

# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt
```

### Prepare the Dataset

```bash
python scripts/prepare_dataset.py
```

Downloads a configurable sample of Amazon Reviews 2023, cleans the data, engineers features, and generates review sentiment scores.

### Train the Models

```bash
python scripts/train_models.py
```

Trains the anomaly detection and review helpfulness models and generates explainability artifacts where supported.

### Assess Products

```bash
python scripts/assess_products.py
```

Calculates trust scores for eligible products and stores the results in SQLite.

### Launch the Application

```bash
python main.py
```

Open `http://127.0.0.1:8080` in your browser.

To run the complete data preparation, training, and assessment pipeline:

```bash
python scripts/run_pipeline.py
```

---

## How It Works

- Reviews and product metadata are sampled from Amazon Reviews 2023.
- The data is cleaned, processed, and transformed into product and review-level features.
- DistilBERT estimates review sentiment, while configurable rules identify suspicious review patterns.
- Isolation Forest detects anomalous review behavior without requiring fake-review labels.
- XGBoost predicts review helpfulness using observed helpful-vote labels.
- SHAP explains model predictions, while rule-based explanations describe suspicious signals.
- Eight weighted components are combined to calculate the final Product Trust Score.
- Assessments, scoring configurations, and version information are stored for historical comparison.

---

## Features

- Product Trust Score ranging from 0 to 100
- Review sentiment analysis using DistilBERT
- Detection of repeated reviews, suspicious bursts, and anomalous reviewer behavior
- Rating quality and consistency analysis
- Verified-purchase and listing completeness analysis
- Machine learning-based review helpfulness prediction
- SHAP-based model explanations
- Component-wise score breakdown and evidence coverage
- Product search and detailed product assessments
- Side-by-side product comparison
- Assessment history and model insights
- Configurable scoring weights and thresholds
- Persistent storage using SQLite and Parquet

---

## Trust Score Components

The default scoring configuration uses eight components:

| Component | Weight |
|---|---:|
| Rating Quality | 22% |
| Review Authenticity Signals | 18% |
| Rating Consistency | 12% |
| Review Volume | 12% |
| Verified Purchases | 12% |
| Review Sentiment | 12% |
| Listing Completeness | 8% |
| Review Informativeness | 4% |

Missing components are excluded rather than imputed, and the remaining weights are normalized. The system also reports evidence coverage and confidence to indicate how much information supports an assessment.

Products with insufficient reviews or evidence do not receive a trust score.

---

## Machine Learning Models

| Model | Purpose |
|---|---|
| DistilBERT | Estimates positive and negative review sentiment |
| Isolation Forest | Identifies anomalous review patterns |
| XGBoost | Predicts whether a review receives at least one helpful vote |
| SHAP | Explains model predictions and feature contributions |

The helpfulness model is evaluated using product-grouped train, validation, and test splits. Votes are excluded from its input features to prevent target leakage.

Isolation Forest and suspicious-review rules do not establish whether a review is fake because the dataset does not provide ground-truth fake-review labels.

---

## Evaluation

The review helpfulness model is evaluated using:

- ROC-AUC
- PR-AUC
- F1-score
- Brier score
- Comparison against a logistic regression baseline and the no-skill rate

The system also provides diagnostic insights into anomaly scores and suspicious-signal distributions.

Sentiment agreement with star ratings is treated as a proxy check, not as a measure of model accuracy.

---

## Data Source & Limitations

The project uses the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) from Hugging Face.

- The dataset is sampled by category rather than downloaded in full.
- Suspicious review signals indicate patterns worth investigating, not confirmed fraudulent activity.
- Sentiment predictions can be inaccurate for sarcasm, short reviews, or mixed opinions.
- Anomaly detection results depend on the sampled data and configured thresholds.
- Return risk, seller reliability, counterfeit detection, and price fairness are not assessed because the required evidence is unavailable.
- The trust score is a configurable heuristic and has not been validated against ground-truth product trustworthiness labels.

