# app.py
# =============================================================
# Diabetes: Home + EDA Dashboard (Plotly) + Prediction (Streamlit)
# - UI: English only
# - EDA: fixed file "diabetes_prediction_project.csv" + SHAP images (single-column)
# - Plots: Plotly (compact) with brief conclusions
# - Prediction: loads an artifact dict from best_model.pkl
#   {model (XGBoost), threshold, feature_order, numeric_feats, scaler, ...}
# - Python 3.9+ compatible.
# Run: streamlit run app.py
# =============================================================

import os
import json
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Diabetes App", layout="wide")

# ---------------------------
# Paths & constants
# ---------------------------
RAW_DATA_PATH = "diabetes_prediction_project.csv"
MODEL_PATH    = "best_model.pkl"
SCALER_PATH   = "preprocessor.pkl"     # optional: overrides artifact scaler if present
FEATURES_PATH = "feature_list.json"    # optional: feature order from training

# SHAP images generated offline by your training notebook
SHAP_BAR_PATH   = "shap_feature_importance.png"
SHAP_SWARM_PATH = "shap_summary_beeswarm.png"

# Categories used by the MODEL (must match training)
SMOKING_CATS_MODEL = ["never", "former", "current", "ever", "not current", "No Info"]
GENDER_CATS_MODEL  = ["Female", "Male"]  # "Other" removed at training time

# UI-friendly options (mapped back to model vocab)
SMOKING_CATS_UI = ["never", "former", "current", "ever", "not current", "Prefer not to say"]
UI_TO_MODEL_SMOKE = {**{x: x for x in SMOKING_CATS_MODEL if x != "No Info"},
                     "Prefer not to say": "No Info"}

# Default plot size for Plotly charts
PLOT_W, PLOT_H = 680, 380

# ---------------------------
# Utilities
# ---------------------------
def exists(path: str) -> bool:
    return os.path.exists(path)

@st.cache_resource(show_spinner=False)
def load_artifact(path: str):
    return joblib.load(path) if exists(path) else None

@st.cache_resource(show_spinner=False)
def load_scaler(path: str) -> Optional[StandardScaler]:
    return joblib.load(path) if exists(path) else None

@st.cache_data(show_spinner=False)
def load_feature_list(path: str) -> Optional[List[str]]:
    if exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            if isinstance(data, list):
                return list(data)
    return None

@st.cache_data(show_spinner=False)
def load_raw_df() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH) if exists(RAW_DATA_PATH) else pd.DataFrame()

def preprocess_like_training(
    df_raw: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    numeric_feats_override: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reproduce training preprocessing:
      1) drop_duplicates; remove gender=='Other'
      2) one-hot encode gender & smoking_history (drop_first=True) with fixed categories
      3) feature engineering: comorbidity_count, age_bmi_interaction
      4) standardize numeric features with provided (fitted) scaler
    Additionally keep _raw_gender/_raw_smoking_history for EDA readability.
    """
    df = df_raw.copy()

    if "gender" in df.columns:
        df = df[df["gender"] != "Other"]
    df = df.drop_duplicates()

    # keep raw categorical columns for EDA
    df["_raw_gender"] = df.get("gender", pd.Series(index=df.index, dtype=object))
    df["_raw_smoking_history"] = df.get("smoking_history", pd.Series(index=df.index, dtype=object))

    # lock categories for stable dummies
    if "gender" in df.columns:
        df["gender"] = pd.Categorical(df["gender"], categories=GENDER_CATS_MODEL + ["Other"], ordered=False)
    if "smoking_history" in df.columns:
        df["smoking_history"] = df["smoking_history"].map(lambda v: UI_TO_MODEL_SMOKE.get(v, v))
        df["smoking_history"] = pd.Categorical(df["smoking_history"], categories=SMOKING_CATS_MODEL, ordered=False)

    # dummies (drop_first=True as in training)
    dmy_cols = [c for c in ["gender", "smoking_history"] if c in df.columns]
    if dmy_cols:
        df = pd.get_dummies(df, columns=dmy_cols, drop_first=True)

    # engineered features
    if set(["hypertension", "heart_disease"]).issubset(df.columns):
        df["comorbidity_count"] = df["hypertension"].astype(float) + df["heart_disease"].astype(float)
    if set(["age", "bmi"]).issubset(df.columns):
        df["age_bmi_interaction"] = df["age"].astype(float) * df["bmi"].astype(float)

    # numeric features to scale
    if numeric_feats_override is not None:
        num_feats = [c for c in numeric_feats_override if c in df.columns]
    else:
        num_feats = [c for c in ["age","bmi","HbA1c_level","blood_glucose_level","age_bmi_interaction"] if c in df.columns]

    if scaler is not None and num_feats:
        df.loc[:, num_feats] = scaler.transform(df[num_feats])

    return df

def align_to_model(X: pd.DataFrame, model, feature_list: Optional[List[str]]) -> pd.DataFrame:
    expected = None
    if feature_list:
        expected = list(feature_list)
    elif hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    return X.reindex(columns=expected, fill_value=0) if expected is not None else X

def px_show(fig):
    fig.update_layout(width=PLOT_W, height=PLOT_H, margin=dict(l=30, r=20, t=50, b=35))
    st.plotly_chart(fig, use_container_width=False)

# ---------------------------
# Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EDA Dashboard", "Prediction"], index=0)

# ===========================
# Home
# ===========================
if page == "Home":
    st.title("Diabetes Risk — Interactive App")
    st.markdown(
        """
**Welcome!**

This app includes:
1. **EDA Dashboard** — interactive Plotly charts (distributions, risk-factor rates, correlation heatmap, two-feature scatter).
2. **Prediction** — enter personal health information to get a diabetes risk estimate (probability + class).

---

### 🔹 Data
This app expects a dataset named **`diabetes_prediction_project.csv`** located **in the same folder as `app.py`**.
- **Format:** CSV with patient attributes (e.g., `age`, `gender`, `bmi`, `HbA1c_level`, `blood_glucose_level`, `hypertension`, `heart_disease`, `smoking_history`, and the target `diabetes`).
- **Usage:** Used for EDA in this app; during training the same schema was used to build the model.

### 🔹 Model
Predictions are powered by **`best_model.pkl`**, an artifact that contains:
- A trained **XGBoost** classifier optimized for diabetes risk.
- The **preprocessing details** identical to training (feature list, optional scaler, and category encoding).
- Helpful metadata such as the **decision threshold** and the expected **feature order**.

**Tip:** Keep preprocessing **identical** to training so the model receives features in the exact expected format.
"""
    )

# ===========================
# EDA Dashboard (Plotly)
# ===========================
elif page == "EDA Dashboard":
    st.title("EDA Dashboard")

    df_raw = load_raw_df()
    if df_raw.empty:
        st.error(f"Data file not found: `{RAW_DATA_PATH}`. Place it next to app.py.")
        st.stop()

    # For interpretability we do NOT scale numerics for EDA
    df_eda = preprocess_like_training(df_raw, scaler=None, numeric_feats_override=None)

    st.subheader("Preview")
    st.dataframe(df_raw.head())

    # ---------- Distributions: violin ----------
    st.subheader("Feature distributions by diabetes (violin)")
    if "diabetes" in df_eda.columns:
        for col in [c for c in ["bmi","HbA1c_level","blood_glucose_level"] if c in df_eda.columns]:
            fig = px.violin(df_eda, x="diabetes", y=col, box=True, points=False)
            fig.update_layout(title=f"{col} vs diabetes")
            px_show(fig)
            g = df_eda.groupby("diabetes")[col].mean().to_dict()
            g0 = g.get(0, np.nan); g1 = g.get(1, np.nan)
            st.caption(f"Conclusion: mean {col} ≈ {g1:.2f} (diabetes=1) vs {g0:.2f} (diabetes=0).")

    # ---------- Age histogram ----------
    if "diabetes" in df_eda.columns and "age" in df_eda.columns:
        st.subheader("Age distribution by diabetes")
        fig = px.histogram(df_eda, x="age", color="diabetes", barmode="overlay", nbins=30, opacity=0.65)
        fig.update_layout(title="Age distribution (overlay)")
        px_show(fig)
        g = df_eda.groupby("diabetes")["age"].mean().to_dict()
        st.caption(f"Conclusion: average age tends to be higher for diabetes=1 "
                   f"({g.get(1, np.nan):.1f}) than for diabetes=0 ({g.get(0, np.nan):.1f}).")

    # ---------- Risk-factor bars ----------
    st.subheader("Diabetes rate by risk factors")
    for col in ["hypertension", "heart_disease", "_raw_smoking_history"]:
        if col in df_eda.columns and "diabetes" in df_eda.columns:
            temp = df_eda[[col,"diabetes"]].copy()
            if col == "_raw_smoking_history":
                temp[col] = temp[col].astype(str).replace({"No Info": "Prefer not to say"})
            rate = temp.groupby(col, dropna=False)["diabetes"].mean().reset_index()
            rate["diabetes"] = rate["diabetes"].astype(float)
            fig = px.bar(rate, x=col, y="diabetes", text="diabetes", range_y=[0,1])
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(title=f"Diabetes rate by {col.replace('_raw_', '')}")
            px_show(fig)

    # ---------- Variable importance (SHAP) — single column ----------
    st.subheader("Variable importance (SHAP)")
    if not exists(SHAP_BAR_PATH) and not exists(SHAP_SWARM_PATH):
        st.info(
            "SHAP images not found. Generate them in your training notebook and save as "
            f"{SHAP_BAR_PATH} and {SHAP_SWARM_PATH} in the same folder as app.py."
        )
    else:
        if exists(SHAP_BAR_PATH):
            st.markdown("**Feature importance (mean |SHAP|)**")
            st.image(SHAP_BAR_PATH, use_container_width=True)
            st.caption("Conclusion: HbA1c and blood glucose dominate overall importance; "
                       "interaction and comorbidities also contribute.")
        if exists(SHAP_SWARM_PATH):
            st.markdown("**SHAP summary (beeswarm)**")
            st.image(SHAP_SWARM_PATH, use_container_width=True)
            st.caption("Conclusion: the spread shows how feature values push predictions up or down per sample.")

    # ---------- Correlation heatmap ----------
    st.subheader("Correlation heatmap")
    numeric_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
    default_pick = [c for c in ["age","bmi","HbA1c_level","blood_glucose_level"] if c in numeric_cols]
    left, right = st.columns([1, 3])
    with left:
        sel_num = st.multiselect(
            "Numeric features for heatmap / scatter",
            options=[c for c in numeric_cols if c != "diabetes"],
            default=default_pick
        )
    with right:
        if len(sel_num) >= 2:
            corr = df_eda[sel_num].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale="RdBu_r", origin="lower")
            fig.update_layout(title="Correlation heatmap")
            px_show(fig)

            corr_vals = corr.abs().stack()
            corr_vals = corr_vals[corr_vals < 0.999]
            if not corr_vals.empty:
                pair = corr_vals.idxmax()
                val  = corr_vals.max()
                st.caption(f"Conclusion: strongest absolute correlation is between **{pair[0]}** and **{pair[1]}** (|r|≈{val:.2f}).")
        else:
            st.info("Select at least two numeric features to render the heatmap.")

    # ---------- Scatter uses the SAME sel_num ----------
    st.subheader("Two-feature scatter (colored by diabetes)")
    if len(sel_num) >= 2 and "diabetes" in df_eda.columns:
        c1, c2 = st.columns(2)
        with c1:
            x_feat = st.selectbox("X axis", options=sel_num, index=0, key="xfeat")
        with c2:
            y_feat = st.selectbox("Y axis", options=[c for c in sel_num if c != x_feat], index=0, key="yfeat")
        fig = px.scatter(df_eda, x=x_feat, y=y_feat, color="diabetes", opacity=0.7)
        fig.update_layout(title=f"{x_feat} vs {y_feat}")
        px_show(fig)
        st.caption("Conclusion: look for separation between diabetes classes along the two features.")
    else:
        st.info("Pick two numeric features and ensure `diabetes` exists.")

# ===========================
# Prediction
# ===========================
else:
    st.title("Prediction")

    loaded = load_artifact(MODEL_PATH)
    if loaded is None:
        st.error(f"Model artifact not found: `{MODEL_PATH}`.")
        st.stop()

    model = loaded
    threshold = None
    feature_order = load_feature_list(FEATURES_PATH)   # optional external list
    art_scaler = None
    art_numeric = None

    if isinstance(loaded, dict):
        model         = loaded.get("model", None)
        threshold     = loaded.get("threshold", None)
        feature_order = loaded.get("feature_order", feature_order)
        art_scaler    = loaded.get("scaler", None)
        art_numeric   = loaded.get("numeric_feats", None)

    scaler = load_scaler(SCALER_PATH) or art_scaler

    if model is None:
        st.error("`model` not found inside artifact. Please check best_model.pkl.")
        st.stop()

    # Inputs (friendly labels; no underscores; no 'Other' for gender)
    st.sidebar.header("Input")
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=40, step=1)
    gender = st.sidebar.selectbox("Gender", options=GENDER_CATS_MODEL, index=1)
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=70.0, value=27.5, step=0.1, format="%.1f")
    hba1c = st.sidebar.number_input("HbA1c (%)", min_value=3.0, max_value=20.0, value=6.5, step=0.1, format="%.1f")
    glucose = st.sidebar.number_input("Blood Glucose (mg/dL)", min_value=40.0, max_value=500.0, value=120.0, step=1.0, format="%.1f")
    hypertension = st.sidebar.selectbox("Hypertension", options=["No", "Yes"], index=0)
    heart_disease = st.sidebar.selectbox("Heart Disease", options=["No", "Yes"], index=0)
    smoking_history_ui = st.sidebar.selectbox("Smoking History", options=SMOKING_CATS_UI, index=0)
    smoking_history_model = UI_TO_MODEL_SMOKE.get(smoking_history_ui, smoking_history_ui)

    input_raw = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": smoking_history_model
    }])

    st.markdown("**Raw input (training column names)**")
    st.dataframe(input_raw)

    if st.button("Predict", type="primary"):
        try:
            pipeline_like = hasattr(model, "predict") and (hasattr(model, "named_steps") or hasattr(model, "transform"))
            if pipeline_like and scaler is None:
                X_in = input_raw.copy()
            else:
                X_tmp = preprocess_like_training(input_raw, scaler=scaler, numeric_feats_override=art_numeric)
                for h in ["_raw_gender","_raw_smoking_history"]:
                    if h in X_tmp.columns:
                        X_tmp = X_tmp.drop(columns=[h])
                X_in = align_to_model(X_tmp, model, feature_order)

            proba = model.predict_proba(X_in)[:, 1] if hasattr(model, "predict_proba") else None
            if proba is not None and threshold is not None:
                pred = (proba >= float(threshold)).astype(int)
            else:
                pred = model.predict(X_in)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Prediction (diabetes)", "Yes" if int(pred[0]) == 1 else "No")
            with c2:
                if proba is not None:
                    st.metric("Probability (positive class)", f"{float(proba[0])*100:.1f}%")
                else:
                    st.info("Current model has no predict_proba; showing class only.")

            if threshold is not None and proba is not None:
                st.caption(f"Conclusion: classification used your saved threshold = {float(threshold):.3f}.")

        except Exception as e:
            st.exception(e)
            st.error("Prediction failed. Please verify model, preprocessing and feature names.")
