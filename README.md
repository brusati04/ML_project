# Smoking Prediction from Blood Signals

## Project Overview

This project is a comprehensive guide to building a **Machine Learning model** capable of predicting whether a patient is a **smoker or non-smoker** based solely on **blood signal data**.

It is designed to offer both **technical insights** and a **step-by-step process** for medical or data science practitioners who wish to apply ML techniques to biomedical signals.

## Objective

The main goal is to create a **binary classifier** that can distinguish between smokers and non-smokers using features extracted from blood samples or related biomarkers.

## Dataset

* The dataset is downloaded from kaggle at this link: https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking
* It contains blood signal measurements for a number of individuals.
* Each row corresponds to a patient, and each column is a specific biomarker or signal.
* The target label indicates **smoker (1)** or **non-smoker (0)**.

## Methodology

The project includes:

1. **Data Preprocessing**

   * Cleaning, handling missing values, normalization/scaling
2. **Exploratory Data Analysis (EDA)**

   * Visualizations and statistics to understand correlations and class balance
3. **Model Selection**

   * Trying out multiple algorithms (e.g., Logistic Regression, Random Forest, XGBoost)
4. **Training & Validation**

   * Using train/test split or cross-validation
5. **Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
6. **Model Tuning**

   * Hyperparameter optimization via grid search or random search

## Dependencies

Python packages are provided inside the notebook


## Future Work

* Deploy the model in a clinical dashboard
* Use deep learning models on raw signal time-series data
* Test generalization on external datasets
* Combine with other lifestyle features for broader health risk prediction

## License

This project is open-source under the [MIT License](LICENSE).
