# Professional Masking of Burnout in Digital Communities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Official code repository for the paper **"Professional Masking of Burnout in Digital Communities"**.

## 📄 Abstract

Burnout in the technology sector is often termed a "silent crisis." This study investigates the hypothesis of **"Professional Masking"**: the phenomenon where employees in high-performance environments (like Stack Overflow) suppress linguistic markers of distress that are otherwise visible in anonymous support communities (like Reddit).

Using a comparative analysis between **100,000 technical interactions** and **1,200 personal narratives**, we demonstrate that traditional sentiment analysis fails in professional settings due to emotional suppression. We propose a longitudinal **LSTM (Long Short-Term Memory)** architecture capable of detecting the "Pragmatic Gap"—a shift from functional agency to somatic vocabulary—achieving **74.5% accuracy** compared to the random baseline of static models.

## 📂 Repository Structure

The project is organized into two distinct pipelines corresponding to the environments analyzed:

### 1. The "Backstage" Pipeline (Reddit)
Located in `src/reddit/`. Implements the Deep Learning approach for early detection.
- `01_data_acquisition.py`: Scrapes anonymous data using PRAW and performs initial anonymization.
- `02_feature_extraction.py`: Generates a multi-level feature set including:
  - **Linguistic**: LIWC-like categories and TextStat complexity metrics.
  - **Semantic**: BERT embeddings (bert-base-uncased).
  - **Temporal**: Time-series aggregation for longitudinal analysis.
- `03_lstm_modeling.py`: Trains the sequential LSTM neural network with Masking layers.
- `04_evaluation_shap.py`: Evaluates performance (F1-Score, AUC-ROC) and generates SHAP interpretability plots to visualize the "Somatic Digital" markers.

### 2. The "Frontstage" Pipeline (Stack Overflow)
Located in `src/stackoverflow/`. Implements the statistical baseline and topic modeling analysis.

- `00_diagnose_dates.py`: Utility to verify the temporal coverage of the XML Data Dump.
- `01_cohort_extraction.py`: Processes the raw Stack Exchange XML dump to identify "Burnout" (abrupt cessation) vs. "Control" (active) cohorts based on reputation filters (>1000).
- `02_cleaning_sampling.py`: Performs deep cleaning (removing HTML tags and code blocks) and stratified sampling (50k per class) to create a balanced dataset.
- `03_feature_engineering.py`: Extracts psycholinguistic features including Sentiment Polarity (TextBlob) and structural complexity metrics (word count, average word length).
- `04_logistic_baseline.py`: Trains the Logistic Regression baseline using TF-IDF vectors and calculates performance metrics (Accuracy, AUC-ROC).
- `05_topic_modeling_lda.py`: Applies Latent Dirichlet Allocation (LDA) to ensure burnout risk is homogeneous across different technical domains (e.g., Web vs. Data Science).
- `06_lstm_bert_experiment`: This script implements a **Deep Learning Counter-Validation** experiment on the Stack Overflow ("Frontstage") dataset. It replicates the high-capacity architecture used for the Reddit dataset (BERT Embeddings + LSTM) but applies it to the technical/professional corpus.
- `07_validation_proxy.py`: To empirically validate the reliability of using "Abrupt Cessation + High Reputation" as a proxy for burnout, distinguishing it from benign attrition (casual churn)..

## 🚀 Usage

### Prerequisites
- Python 3.10 or higher.
- A Reddit API Client ID and Secret (for data acquisition).
- Access to the Stack Exchange Data Dump (for the Frontstage pipeline).

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/aguirrelam/burnout-professional-masking.git](https://github.com/aguirrelam/burnout-professional-masking.git)
   cd burnout-professional-masking
   
## 📊 Data Availability

To ensure reproducibility and transparency, the datasets and trained models used in this study are hosted on Zenodo.

- **Datasets & Models**: *(DOI: 10.5281/zenodo.18099992)*

This includes:
1. **Reddit Snapshot**: The static dataset used to train the LSTM model (N=1,200).
2. **Stack Overflow Cohorts**: The processed parquet files for the "Frontstage" analysis (N=100,000).
3. **Trained Models**: The final `.keras` model weights.

> **Note:** Due to privacy regulations and platform terms of service, raw data containing PII has been anonymized.
