"""
Adult Dataset Multi-Model Comparison
------------------------------------
- Trains multiple classifiers on Adult dataset
- Uses reweighed instance weights if available
- Computes fairness metrics for each model using AIF360
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

print("Starting Multi-Model Comparison on Adult dataset...")

# ==============================
# 1. LOAD PROCESSED DATA
# ==============================
df = pd.read_csv("data/processed/adult_processed.csv")
print(f"✅ Loaded dataset with shape: {df.shape}")

# Fix target variable
if 'income_target' not in df.columns:
    if 'income' in df.columns:
        df['income_target'] = df['income'].astype(str).apply(lambda x: 1 if x.strip() in ['>50K', '>50K.'] else 0)
        df.drop(columns=['income'], inplace=True)
    else:
        raise ValueError("No 'income' or 'income_target' column found!")

df = df.dropna(subset=['income_target'])
df['income_target'] = df['income_target'].astype(int)

# Check label balance
label_counts = df['income_target'].value_counts()
print("\nLabel distribution in full dataset:")
print(label_counts)
if len(label_counts) < 2:
    raise ValueError("Dataset has only one class! Please re-check target encoding.")

# ==============================
# 2. PROTECTED ATTRIBUTE
# ==============================
if 'sex_binary' in df.columns:
    protected_col = 'sex_binary'
else:
    raise ValueError("Protected attribute 'sex_binary' not found!")

print(f"✅ Protected attribute selected: {protected_col}")

# ==============================
# 3. SPLIT DATA
# ==============================
X = df.drop(columns=['income_target'])
y = df['income_target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==============================
# 4. LOAD REWEIGHED WEIGHTS
# ==============================
try:
    weights_df = pd.read_csv("data/processed/adult_reweighing_weights.csv")
    train_weights = weights_df['instance_weight'].values
    print("✅ Loaded instance weights from reweighing.")
except FileNotFoundError:
    print("⚠️ No weights file found — proceeding without reweighing.")
    train_weights = np.ones(len(y_train))

# ==============================
# 5. DEFINE MODELS
# ==============================
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, solver='liblinear'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ==============================
# 6. TRAIN, PREDICT, AND EVALUATE
# ==============================
all_metrics = []  # To store results for all models

for model_name, clf in models.items():
    print(f"\n--- Training {model_name} ---")
    
    # Ensure training labels have both classes
    if len(np.unique(y_train)) < 2:
        print(f"Skipping {model_name}: Training labels contain only one class!")
        continue
    
    # Train
    clf.fit(X_train, y_train, sample_weight=train_weights)
    y_pred = clf.predict(X_test)
    
    # Accuracy and classification report
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # ==============================
    # Prepare AIF360 datasets
    # ==============================
    test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).dropna()
    aif_test = BinaryLabelDataset(
        df=test_df,
        label_names=['income_target'],
        protected_attribute_names=[protected_col]
    )
    
    pred_df = pd.concat([X_test.reset_index(drop=True), pd.Series(y_pred, name='income_target')], axis=1).dropna()
    aif_pred = BinaryLabelDataset(
        df=pred_df,
        label_names=['income_target'],
        protected_attribute_names=[protected_col]
    )
    
    # ==============================
    # Compute fairness metrics
    # ==============================
    metric = ClassificationMetric(
        aif_test,
        aif_pred,
        unprivileged_groups=[{protected_col: 0}],
        privileged_groups=[{protected_col: 1}]
    )
    
    eod = metric.equal_opportunity_difference()
    aod = metric.average_odds_difference()
    di = metric.disparate_impact()
    spd = metric.statistical_parity_difference()
    theil = metric.theil_index()
    
    print("\nFairness Metrics:")
    print(f"Equal Opportunity Difference: {eod:.4f}")
    print(f"Average Odds Difference: {aod:.4f}")
    print(f"Disparate Impact: {di:.4f}")
    print(f"Statistical Parity Difference: {spd:.4f}")
    print(f"Theil Index: {theil:.4f}")
    
    # Store metrics
    all_metrics.append({
        "Model": model_name,
        "Accuracy": acc,
        "Equal Opportunity Diff": eod,
        "Average Odds Diff": aod,
        "Disparate Impact": di,
        "Statistical Parity Diff": spd,
        "Theil Index": theil
    })

# ==============================
# 7. SAVE RESULTS
# ==============================
results_df = pd.DataFrame(all_metrics)
results_df.to_csv("data/processed/adult_model_comparison_results.csv", index=False)
print("\n✅ Model comparison results saved to data/processed/adult_model_comparison_results.csv")

print("\n===== MODEL COMPARISON SUMMARY =====")
print(results_df)
print("\n✅ Multi-model comparison completed successfully.")
