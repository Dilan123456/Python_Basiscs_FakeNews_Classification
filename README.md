# Fake News Detection with Machine Learning

## Project Overview
This project focuses on building a machine learning pipeline to automatically detect fake news based on the article content.  
The goal is to combine textual features and stylistic cues using Natural Language Processing (NLP) and classical ML algorithms to distinguish between real and fake news articles.

## Project Team Members
Athika Pasupathipillai 
Dilan Joseph
Samuel Jucker
Nina Habegger

## Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and contains:

- `Fake.csv` – fake news articles
- `True.csv` – real news articles

Each row includes the article's title, text body, and publication date.

## Methodology
The pipeline includes the following steps:

1. Data Import & Cleaning
2. Text Preprocessing & Feature Engineering
   - Word count
   - Capital letter ratio
   - Number of exclamation marks
   - Sentiment score (VADER)
3. Exploratory Data Analysis (EDA)
4. Model Training
   - Logistic Regression
   - Random Forest
   - Calibrated Support Vector Machine (SVM)
5. Hyperparameter Tuning using GridSearchCV**
6. Evaluation with metrics: Accuracy, Precision, Recall, F1, ROC-AUC**
7. Threshold Optimization for best F1-score**
8. Model Explainability with SHAP (global feature importance)**

## 🤖 Model Comparison
| Model                  | Key Characteristics                      |
|------------------------|-------------------------------------------|
| **Logistic Regression** | Simple, fast, interpretable               |
| **Random Forest**       | Robust, non-linear, SHAP-compatible       |
| **Calibrated SVM**      | High performance, calibrated probabilities |

All models are wrapped in pipelines using **TF-IDF** for text vectorization and **scikit-learn transformers**.

## Sample Results
- F1-scores: **above 0.99** across all models
- ROC-AUC: high for all models, especially Logistic Regression and Random Forest
- SHAP analysis revealed meaningful patterns:
  - Key terms: `reuters`, `trump`, `said`
  - Style cues: number of exclamation marks, sentiment score
