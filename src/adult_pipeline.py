"""
Adult Dataset Preprocessing Pipeline
------------------------------------
Generates a clean, encoded dataset ready for BiasLens:
- Loads raw Adult dataset
- Cleans missing values
- Encodes target and sensitive attributes
- One-hot encodes categorical features
- Saves processed CSV for downstream fairness evaluation
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ==========================
# 1. LOAD RAW DATA
# ==========================
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

df = pd.read_csv("data/raw/adult.csv", header=None, names=columns)

# Strip whitespace in string columns
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Replace '?' with NaN and fill missing categorical values
df.replace('?', pd.NA, inplace=True)
df.fillna('Unknown', inplace=True)

# ==========================
# 2. ENCODE TARGET VARIABLE
# ==========================
# Strip again and map
df['income'] = df['income'].astype(str).str.strip()

income_map = {
    '<=50K': 0,
    '<=50K.': 0,
    '>50K': 1,
    '>50K.': 1
}

df['income_target'] = df['income'].map(income_map)

# Check for unexpected values
if df['income_target'].isnull().any():
    print("⚠️ Unexpected income values:", df['income'].unique())

# Confirm both classes exist
print("✅ Target distribution after encoding:")
print(df['income_target'].value_counts(normalize=True))

# ==========================
# 3. ENCODE SENSITIVE ATTRIBUTE (SEX)
# ==========================
df['sex_binary'] = df['sex'].apply(lambda x: 1 if x.strip().lower() == 'male' else 0)
print("✅ Sensitive attribute distribution:")
print(df['sex_binary'].value_counts(normalize=True))

# ==========================
# 4. DROP ORIGINAL COLUMNS
# ==========================
df = df.drop(columns=['income', 'sex'])

# ==========================
# 5. ONE-HOT ENCODE CATEGORICAL FEATURES
# ==========================
cat_cols = df.select_dtypes(include='object').columns.tolist()
enc = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_array = enc.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded_array, columns=enc.get_feature_names_out(cat_cols))

# Merge with numeric + target + sensitive attributes
num_df = df.drop(columns=cat_cols)
processed_df = pd.concat([num_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

print("✅ Processed dataset shape:", processed_df.shape)

# ==========================
# 6. SAVE PROCESSED DATA
# ==========================
processed_df.to_csv("data/processed/adult_processed.csv", index=False)
print("✅ Saved processed dataset to data/processed/adult_processed.csv")
