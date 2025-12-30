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
Located in `src/stackoverflow/`. Implements the statistical baseline.
- `01_xml_processing.py`: Processes the massive Stack Exchange XML Data Dump to identify burnout cohorts based on abrupt activity cessation and "reputation" filtering.

## 🚀 Usage

### Prerequisites
- Python 3.10 or higher.
- A Reddit API Client ID and Secret (for data acquisition).
- Access to the Stack Exchange Data Dump (for the Frontstage pipeline).

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/burnout-professional-masking.git](https://github.com/YOUR_USERNAME/burnout-professional-masking.git)
   cd burnout-professional-masking
