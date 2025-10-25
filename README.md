# BiasLens: GenAI Fairness Dashboard

## Overview
BiasLens is a Streamlit-based dashboard designed to **analyze fairness and performance of machine learning models** on datasets with sensitive attributes. It allows users to compare multiple models, visualize fairness and accuracy metrics, and generate a GenAI-powered fairness report to identify bias and recommend mitigation strategies. This project is particularly useful for datasets similar to the UCI Adult dataset, where income classification may exhibit gender or racial biases.

## Motivation & Objectives
The main goals of BiasLens are:
* To **compare multiple machine learning models** (Logistic Regression, Random Forest, SVM, XGBoost) on a given dataset.
* To **analyze fairness metrics** such as Disparate Impact, Statistical Parity Difference, Equal Opportunity Difference, Average Odds Difference, and Theil Index.
* To **leverage generative AI** to produce a professional, human-readable fairness report summarizing model performance and bias insights.
* To **provide actionable recommendations** for mitigating bias and improving fairness in predictive models.

## Methodology & Approach
The workflow of BiasLens includes the following steps:

1. **Data Upload & Preprocessing:**
    * Users upload a CSV dataset. Columns are assigned to match standard datasets like UCI Adult if necessary.
    * Users select a target column and a sensitive attribute (e.g., sex, race) for fairness evaluation.
    * Categorical features are encoded, and numerical features are standardized for model training.

2. **Model Training & Comparison:**
    * Four models are trained: Logistic Regression, Random Forest, SVM, and XGBoost.
    * Accuracy and fairness metrics are computed on the test set:
        * **Accuracy** – how often the model predicts correctly.
        * **Disparate Impact (DI)** – ratio of positive outcomes for unprivileged vs privileged groups.
        * **Statistical Parity Difference (SPD)** – difference in positive outcome rates between groups.
        * **Equal Opportunity Difference (EOD)** – difference in true positive rates.
        * **Average Odds Difference (AOD)** – considers both false positives and true positives.
        * **Theil Index** – measures inequality in predicted outcomes.

3. **Visualizations:**
    * Bar plots for accuracy and fairness metrics across models.
    * Scatter plot showing **Accuracy vs Statistical Parity Difference** to visualize the trade-off between performance and fairness.

4. **GenAI Fairness Report:**
    * A prompt is sent to Google’s Gemini model, which generates a **concise, professional report** summarizing dataset insights, model bias, best-performing model, and mitigation strategies.
    * Users can view and download the report as a `.txt` file.

## Dataset
* Example datasets include the **UCI Adult dataset** or any CSV with categorical and numerical features.
* The dataset must include a **target column** for prediction and a **sensitive attribute** to evaluate fairness.

## Results
* BiasLens produces a **model comparison table** with both accuracy and fairness metrics.
* The **GenAI report** summarizes:
    * Dataset overview
    * Bias insights across models
    * Best model balancing fairness and accuracy
    * Recommendations for mitigating bias

## How to Run
1. **Clone the repository:**
```bash
git clone https://github.com/Sreelakshmi-rv/Biaslens
cd BiasLens
