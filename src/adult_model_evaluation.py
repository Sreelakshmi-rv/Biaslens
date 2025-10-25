"""
Adult Dataset Model Evaluation
------------------------------
- Evaluates logistic regression on Adult dataset
- Uses reweighed instance weights if available
- Computes fairness metrics using AIF360
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

print("Starting Fairness Evaluation on Adult dataset...")

# ==============================
# 1. LOAD PROCESSED DATA
# ==============================
df = pd.read_csv("data/processed/adult_processed.csv")
print(f"✅ Loaded dataset with shape: {df.shape}")

# ------------------------------
# Fix target variable
# ------------------------------
if 'income' in df.columns:
    df['income_target'] = df['income'].astype(str).apply(lambda x: 1 if x.strip() in ['>50K', '>50K.'] else 0)
    df.drop(columns=['income'], inplace=True)
elif 'income_target' not in df.columns:
    raise ValueError("No 'income' or 'income_target' column found in dataset!")

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
    raise ValueError("Protected attribute 'sex_binary' not found after encoding!")

print(f"✅ Protected attribute selected: {protected_col}")

# ==============================
# 3. SPLIT DATA
# ==============================
X = df.drop(columns=['income_target'])
y = df['income_target']

# Stratify to preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nTrain label distribution:")
print(y_train.value_counts())
print("\nTest label distribution:")
print(y_test.value_counts())

# ==============================
# 4. LOAD REWEIGHED WEIGHTS (if available)
# ==============================
try:
    weights_df = pd.read_csv("data/processed/adult_reweighing_weights.csv")
    train_weights = weights_df['instance_weight'].values
    print("\n✅ Loaded instance weights from reweighing.")
except FileNotFoundError:
    print("\n⚠️ No weights file found — proceeding without reweighing.")
    train_weights = np.ones(len(y_train))

# ==============================
# 5. TRAIN MODEL
# ==============================
clf = LogisticRegression(max_iter=1000, solver='liblinear')

# Check again that training labels have both classes
if len(np.unique(y_train)) < 2:
    raise ValueError("Training labels contain only one class after split!")

clf.fit(X_train, y_train, sample_weight=train_weights)
y_pred = clf.predict(X_test)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# 6. CONVERT TO AIF360 FORMAT
# ==============================
train_df = X_train.copy()
train_df['income_target'] = y_train

test_df = X_test.copy()
test_df['income_target'] = y_test
test_df['predictions'] = y_pred

aif_train = BinaryLabelDataset(
    df=train_df,
    label_names=['income_target'],
    protected_attribute_names=[protected_col]
)

aif_test = BinaryLabelDataset(
    df=test_df,
    label_names=['income_target'],
    protected_attribute_names=[protected_col]
)

# Fix: predictions dataset should differ only in labels
aif_pred = aif_test.copy()
aif_pred.labels = test_df['predictions'].values.reshape(-1, 1)

# ==============================
# 7. COMPUTE FAIRNESS METRICS
# ==============================
metric = ClassificationMetric(
    aif_test,
    aif_pred,
    unprivileged_groups=[{protected_col: 0}],
    privileged_groups=[{protected_col: 1}]
)

print("\n===== FAIRNESS METRICS AFTER MODEL TRAINING =====")
print(f"Equal Opportunity Difference: {metric.equal_opportunity_difference():.4f}")
print(f"Average Odds Difference: {metric.average_odds_difference():.4f}")
print(f"Disparate Impact: {metric.disparate_impact():.4f}")
print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")
print(f"Theil Index (measure of inequality): {metric.theil_index():.4f}")

print("\n✅ Fairness evaluation completed successfully for Adult dataset.")
