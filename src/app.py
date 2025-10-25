from dotenv import load_dotenv
load_dotenv()


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import google.generativeai as genai
import numpy as np
import os

# ========================
# CONFIG
# ========================
st.set_page_config(page_title="BiasLens GenAI Dashboard", layout="wide")
st.title("BiasLens: GenAI Fairness Dashboard")

# ✅ Keep your API key private
GENAI_API_KEY = os.getenv("GENAI_API_KEY")

# Initialize session state
for key in ["df", "comparison_df", "label_col", "sensitive_attr", "comparison_done"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.comparison_done = st.session_state.comparison_done or False

# ======================================================
# STEP 1: Upload Dataset
# ======================================================
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type="csv")

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=None)

        # Assign standard Adult dataset columns
        df.columns = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ]
        st.session_state.df = df

        st.success("✅ Dataset loaded successfully!")
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # ----------------------------
        # Select Target Column
        # ----------------------------
        label_col = st.selectbox("🎯 Select Target Column", df.columns)
        st.session_state.label_col = label_col

        # Select Sensitive Attribute for Fairness Analysis
        sensitive_attr = st.selectbox("⚖️ Select Sensitive Attribute for Fairness (e.g., sex or race)", df.columns)
        st.session_state.sensitive_attr = sensitive_attr

        # ======================================================
        # STEP 2: Run Model Comparisons
        # ======================================================
        if st.button("🚀 Run Model Comparison"):
            st.info("Running model comparisons... please wait.")

            df = st.session_state.df.copy()
            label_col = st.session_state.label_col
            sensitive = st.session_state.sensitive_attr

            X = df.drop(columns=[label_col])
            y = df[label_col].astype(str).str.strip()
            sens = df[sensitive].astype(str).str.strip()

            # Encode target
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)

            # Encode sensitive attribute
            sens_encoded = LabelEncoder().fit_transform(sens)

            # Encode categorical features
            cat_cols = X.select_dtypes(include=['object']).columns
            for col in cat_cols:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str).str.strip())

            # Train-test split
            X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
                X, y, sens_encoded, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define models
            models = {
                "LogisticRegression": LogisticRegression(max_iter=500),
                "RandomForest": RandomForestClassifier(n_estimators=100),
                "SVM": SVC(probability=True),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            }

            results = []
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                acc = accuracy_score(y_test, y_pred)

                # Compute fairness metrics
                privileged = sens_test == 1
                unprivileged = sens_test == 0

                if len(set(y_pred)) > 1:
                    spd = abs(y_pred[privileged].mean() - y_pred[unprivileged].mean())
                    di = y_pred[unprivileged].mean() / (y_pred[privileged].mean() + 1e-6)
                    mask_priv = (y_test == 1) & privileged
                    mask_unpriv = (y_test == 1) & unprivileged
                    eod = abs(y_pred[mask_priv].mean() - y_pred[mask_unpriv].mean())
                    mask_priv_neg = (y_test == 0) & privileged
                    mask_unpriv_neg = (y_test == 0) & unprivileged
                    fpr_priv = y_pred[mask_priv_neg].mean()
                    fpr_unpriv = y_pred[mask_unpriv_neg].mean()
                    aod = 0.5 * ((eod) + abs(fpr_priv - fpr_unpriv))
                else:
                    spd = di = eod = aod = np.nan

                theil_index = np.mean(y_pred * np.log((y_pred + 1e-6) / np.mean(y_pred + 1e-6)))

                results.append({
                    "Model": name,
                    "Accuracy": round(acc, 3),
                    "Disparate Impact": round(di, 3),
                    "Statistical Parity Diff": round(spd, 3),
                    "Equal Opportunity Diff": round(eod, 3),
                    "Average Odds Diff": round(aod, 3),
                    "Theil Index": round(theil_index, 3)
                })

            comparison_df = pd.DataFrame(results)
            st.session_state.comparison_df = comparison_df
            st.session_state.comparison_done = True

            st.subheader("📊 Model Comparison Table")
            st.dataframe(comparison_df)

            # ======================================================
            # STEP 3: Visualizations
            # ======================================================
            st.subheader("📈 Fairness and Accuracy Visualizations")

            metrics = ["Accuracy", "Disparate Impact", "Statistical Parity Diff", "Equal Opportunity Diff"]
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x="Model", y=metric, data=comparison_df, ax=ax, palette="viridis")
                ax.set_title(metric)
                st.pyplot(fig)

            st.subheader("⚖️ Accuracy vs Fairness Trade-off")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.scatterplot(
                x="Statistical Parity Diff",
                y="Accuracy",
                data=comparison_df,
                hue="Model",
                s=150,
                palette="deep",
                ax=ax
            )
            ax.axvline(0, color='grey', linestyle='--')
            ax.set_xlabel("Statistical Parity Difference (0 = fair)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy vs Fairness Trade-off")
            st.pyplot(fig)

        # ======================================================
        # STEP 4: GenAI Fairness Report (persistent)
        # ======================================================
        if st.session_state.comparison_done and st.session_state.comparison_df is not None:
            if st.button("🧠 Generate GenAI Fairness Report"):
                st.info("Generating report using Gemini 2.5 Flash... please wait.")
                try:
                    genai.configure(api_key=GENAI_API_KEY)
                    model_gemini = genai.GenerativeModel("gemini-2.5-flash")

                    prompt = f"""
                    You are BiasLens, an AI fairness auditor explaining results in clear, simple language.

                    The dataset is similar to the UCI Adult dataset used for income prediction.
                    The target variable is '{st.session_state.label_col}' and the sensitive attribute is '{st.session_state.sensitive_attr}'.
                    Here is the model comparison summary:
                    {st.session_state.comparison_df.to_string(index=False)}

                    Write a narrative report for someone without a data science background.
                    Include:
                    - Overview of the dataset and analysis purpose
                    - Fairness insights in plain language
                    - Which model is best considering both accuracy and fairness
                    - Suggestions to improve fairness
                    Keep paragraphs readable, no bullet points or markdown symbols, max 350 words.
                    """

                    response = model_gemini.generate_content(prompt)
                    summary = response.text.strip()

                    st.subheader("📝 GenAI Fairness Report")
                    st.text(summary)

                    output_path = "data/processed/genai_fairness_report.txt"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(summary)
                    st.success("✅ Report saved successfully!")
                    st.download_button("⬇️ Download Report", summary, file_name="genai_fairness_report.txt")

                except Exception as e:
                    st.error(f"Error generating GenAI report: {e}")

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file appears empty.")
    except Exception as e:
        st.error(f"⚠️ Error reading CSV: {e}")

else:
    st.warning("📂 Please upload a CSV file to proceed.")
