# app.py
# =============================================================
# Diabetes Dashboard (EDA) + Prediction (Streamlit)
# - EDA: Matplotlib + Seaborn；固定数据源：diabetes_prediction_project.csv
# - Prediction: 读取你打包保存的 best_model.pkl（artifact 字典），
#               与训练一致的预处理（去重、去除 gender=Other、独热、构造项、数值标准化）。
# 兼容 Python 3.9
# 运行：streamlit run app.py
# 需要：best_model.pkl（必需）；可选 preprocessor.pkl、feature_list.json
# =============================================================

import os
import json
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Diabetes: Dashboard & Prediction", layout="wide")
sns.set(style="whitegrid")

# ---------------------------
# 常量
# ---------------------------
RAW_DATA_PATH = "diabetes_prediction_project.csv"
MODEL_PATH = "best_model.pkl"
SCALER_PATH = "preprocessor.pkl"        # 训练时用的 StandardScaler（可选，如提供则覆盖artifact内的scaler）
FEATURES_PATH = "feature_list.json"     # 训练时特征顺序（可选）

# 训练中使用的类别（保持一致）
SMOKING_CATEGORIES = ["never", "former", "current", "ever", "not current", "No Info"]
GENDER_CATEGORIES = ["Female", "Male"]  # 训练中已去掉 'Other'

# ---------------------------
# 工具函数
# ---------------------------
def ensure_exists(path: str) -> bool:
    return os.path.exists(path)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path) if ensure_exists(path) else None

@st.cache_resource(show_spinner=False)
def load_scaler(path: str) -> Optional[StandardScaler]:
    return joblib.load(path) if ensure_exists(path) else None

@st.cache_data(show_spinner=False)
def load_feature_list(path: str) -> Optional[List[str]]:
    if ensure_exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "features" in data:
                return list(data["features"])
            if isinstance(data, list):
                return list(data)
    return None

@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH) if ensure_exists(RAW_DATA_PATH) else pd.DataFrame()

def training_like_preprocess(
    df_raw: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    numeric_feats_override: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    复现训练时预处理（用于建模/预测）：
      1) drop_duplicates；去掉 gender=='Other'
      2) 对 gender、smoking_history 进行独热编码（drop_first=True）
      3) 构造 comorbidity_count, age_bmi_interaction
      4) 标准化数值列（使用已fit的 scaler）
    说明：为 EDA 展示保留 _raw_gender/_raw_smoking_history（仅用于图表分组/着色）
    """
    df = df_raw.copy()

    # 去掉 Other 并去重
    if "gender" in df.columns:
        df = df[df["gender"] != "Other"]
    df = df.drop_duplicates()

    # 保留原始分类列供 EDA 使用
    df["_raw_gender"] = df.get("gender", pd.Series(index=df.index, dtype=object))
    df["_raw_smoking_history"] = df.get("smoking_history", pd.Series(index=df.index, dtype=object))

    # 固定类别，保证 dummy 列稳定
    if "gender" in df.columns:
        df["gender"] = pd.Categorical(df["gender"], categories=GENDER_CATEGORIES + ["Other"], ordered=False)
    if "smoking_history" in df.columns:
        df["smoking_history"] = pd.Categorical(df["smoking_history"], categories=SMOKING_CATEGORIES, ordered=False)

    # 独热编码（与训练一致 drop_first=True）
    dmy_cols = [c for c in ["gender", "smoking_history"] if c in df.columns]
    if dmy_cols:
        df = pd.get_dummies(df, columns=dmy_cols, drop_first=True)

    # 构造特征
    if set(["hypertension", "heart_disease"]).issubset(df.columns):
        df["comorbidity_count"] = df["hypertension"].astype(float) + df["heart_disease"].astype(float)
    if set(["age", "bmi"]).issubset(df.columns):
        df["age_bmi_interaction"] = df["age"].astype(float) * df["bmi"].astype(float)

    # —— 数值列：优先用 artifact 里保存的 numeric_feats —— #
    if numeric_feats_override is not None:
        numeric_feats = [c for c in numeric_feats_override if c in df.columns]
    else:
        numeric_feats = [c for c in ["age", "bmi", "HbA1c_level", "blood_glucose_level", "age_bmi_interaction"] if c in df.columns]

    # 标准化（若提供了训练时已 fit 的 scaler）
    if scaler is not None and numeric_feats:
        df.loc[:, numeric_feats] = scaler.transform(df[numeric_feats])

    return df

def align_features_for_model(X_df: pd.DataFrame, model, feature_list: Optional[List[str]]) -> pd.DataFrame:
    """将列对齐到模型期望的顺序与集合。缺失列补 0，多余列丢弃。"""
    expected: Optional[List[str]] = None
    if feature_list:
        expected = list(feature_list)
    elif hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    return X_df.reindex(columns=expected, fill_value=0) if expected is not None else X_df

def seaborn_to_streamlit(fig):
    st.pyplot(fig, clear_figure=True)

# ---------------------------
# 页面导航
# ---------------------------
st.sidebar.title("导航")
page = st.sidebar.radio("选择模块", ["Dashboard (EDA)", "Prediction"], index=0)

# ===========================
# Dashboard (EDA)
# ===========================
if page == "Dashboard (EDA)":
    st.title("📊 Dashboard：EDA（Matplotlib & Seaborn）")

    df_raw = load_raw_data()
    if df_raw.empty:
        st.error(f"未找到固定数据文件：{RAW_DATA_PATH}。请将其与 app.py 放在同一目录。")
        st.stop()

    # 仅用于展示：不做数值标准化，保证可解释性
    df_eda = training_like_preprocess(df_raw, scaler=None, numeric_feats_override=None)

    st.subheader("数据预览")
    st.dataframe(df_raw.head())

    # 变量选择
    numeric_cols_all = df_eda.select_dtypes(include=[np.number]).columns.tolist()
    # 用原始分类列做着色/分组，读起来更直观
    cat_candidates = []
    if "_raw_gender" in df_eda.columns: cat_candidates.append("_raw_gender")
    if "_raw_smoking_history" in df_eda.columns: cat_candidates.append("_raw_smoking_history")
    for c in ["hypertension", "heart_disease"]:
        if c in df_eda.columns: cat_candidates.append(c)

    st.sidebar.markdown("### 变量选择")
    sel_num = st.sidebar.multiselect(
        "数值变量（用于热力图/散点）",
        options=sorted([c for c in numeric_cols_all if c != "diabetes"]),
        default=[c for c in ["age", "bmi", "HbA1c_level", "blood_glucose_level"] if c in numeric_cols_all]
    )
    color_key = st.sidebar.selectbox(
        "分类变量（用于上色，可选）",
        options=["(无)"] + cat_candidates,
        index=0
    )
    color_key = None if color_key == "(无)" else color_key  # 目前主要在散点图用 hue=diabetes，上色更清晰

    # ---- 各特征与糖尿病的关系：bmi/HbA1c/glucose vs diabetes（小提琴图） ----
    st.subheader("各特征与糖尿病的关系")
    if "diabetes" in df_eda.columns:
        for col in [c for c in ["bmi", "HbA1c_level", "blood_glucose_level"] if c in df_eda.columns]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sns.violinplot(data=df_eda, x="diabetes", y=col, inner="quartile", ax=ax)
            ax.set_title(f"{col} vs diabetes")
            ax.set_xlabel("diabetes（0=否, 1=是）")
            ax.set_ylabel(col)
            seaborn_to_streamlit(fig)
    else:
        st.info("缺少 `diabetes` 列，无法绘制小提琴图。")

    # ---- age vs diabetes：直方 + KDE ----
    if "diabetes" in df_eda.columns and "age" in df_eda.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(data=df_eda, x="age", hue="diabetes", multiple="stack", bins=30, kde=True, ax=ax)
        ax.set_title("Age Distribution by diabetes")
        seaborn_to_streamlit(fig)

    # ---- 风险因素柱状图：患病率 ----
    st.subheader("风险因素与糖尿病患病率")
    for col in ["hypertension", "heart_disease", "_raw_smoking_history"]:
        if col in df_eda.columns and "diabetes" in df_eda.columns:
            tmp = df_eda[[col, "diabetes"]].copy()
            if col == "_raw_smoking_history":
                tmp[col] = tmp[col].astype(str)
            rate_df = tmp.groupby(col, dropna=False)["diabetes"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            sns.barplot(data=rate_df, x=col, y="diabetes", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel("糖尿病患病率")
            ax.set_title(f"Diabetes Rate by {col.replace('_raw_', '')}")
            plt.xticks(rotation=20, ha="right")
            seaborn_to_streamlit(fig)

    # ---- 数值特征相关性热力图 ----
    st.subheader("数值特征相关性热力图")
    if len(sel_num) >= 2:
        corr = df_eda[sel_num].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(1.2 * len(sel_num) + 3, 0.9 * len(sel_num) + 3))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap")
        seaborn_to_streamlit(fig)
    else:
        st.info("请选择至少两个数值变量以绘制热力图。")

    # ---- 两个数值特征散点（按 diabetes 上色）----
    st.subheader("两个数值特征散点图（按 diabetes 上色）")
    if len(sel_num) >= 2 and "diabetes" in df_eda.columns:
        c1, c2 = st.columns(2)
        with c1:
            x_feat = st.selectbox("X 轴特征", options=sel_num, index=0, key="xfeat")
        with c2:
            y_feat = st.selectbox("Y 轴特征", options=[c for c in sel_num if c != x_feat], index=0, key="yfeat")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=df_eda, x=x_feat, y=y_feat, hue="diabetes", alpha=0.6, ax=ax)
        ax.set_title(f"{x_feat} vs {y_feat}")
        seaborn_to_streamlit(fig)
    else:
        st.info("需要选择至少两个数值变量，且数据包含 `diabetes` 列。")

# ===========================
# Prediction
# ===========================
else:
    st.title("🧮 Prediction：糖尿病风险预测")

    # 载入 artifact（可能是 dict，也可能直接是模型/Pipeline）
    loaded = load_model(MODEL_PATH)
    if loaded is None:
        st.error(f"未找到模型文件：{MODEL_PATH}。")
        st.stop()

    # 默认值
    model = loaded
    threshold = None
    feature_list = load_feature_list(FEATURES_PATH)  # 可选的外部 json
    artifact_scaler = None
    artifact_numeric = None

    # 如果是保存的 dict，就解包
    if isinstance(loaded, dict):
        model = loaded.get("model", None)
        threshold = loaded.get("threshold", None)
        feature_list = loaded.get("feature_order", feature_list)
        artifact_scaler = loaded.get("scaler", None)
        artifact_numeric = loaded.get("numeric_feats", None)

    # 外部 preprocessor.pkl 优先级更高：如果提供了，就覆盖 artifact 里的 scaler
    scaler = load_scaler(SCALER_PATH) or artifact_scaler

    if model is None:
        st.error("artifact 中未找到 'model'。请检查 best_model.pkl 的内容。")
        st.stop()

    # —— 侧边栏输入（性别仅 Female/Male）—— #
    st.sidebar.header("输入个人健康信息")
    age = st.sidebar.number_input("age（年龄）", min_value=0, max_value=120, value=40, step=1)
    gender = st.sidebar.selectbox("gender（性别）", options=GENDER_CATEGORIES, index=1)  # 仅 Female/Male
    bmi = st.sidebar.number_input("bmi（体重指数）", min_value=10.0, max_value=70.0, value=27.5, step=0.1, format="%.1f")
    hba1c = st.sidebar.number_input("HbA1c_level（%）", min_value=3.0, max_value=20.0, value=6.5, step=0.1, format="%.1f")
    glucose = st.sidebar.number_input("blood_glucose_level（mg/dL）", min_value=40.0, max_value=500.0, value=120.0, step=1.0, format="%.1f")
    hypertension = st.sidebar.selectbox("hypertension（高血压史）", options=["No", "Yes"], index=0)
    heart_disease = st.sidebar.selectbox("heart_disease（心脏病史）", options=["No", "Yes"], index=0)
    smoking_history = st.sidebar.selectbox("smoking_history（吸烟史）", options=SMOKING_CATEGORIES, index=0)

    # 原始输入
    input_df_raw = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": smoking_history
    }])

    st.markdown("**输入数据（原始）**")
    st.dataframe(input_df_raw)

    if st.button("开始预测", type="primary"):
        try:
            # 如果模型是完整 Pipeline，让它自己处理；否则按训练流程预处理
            pipeline_like = hasattr(model, "predict") and (hasattr(model, "named_steps") or hasattr(model, "transform"))
            if pipeline_like and scaler is None:
                X_in = input_df_raw.copy()
            else:
                X_tmp = training_like_preprocess(
                    input_df_raw,
                    scaler=scaler,
                    numeric_feats_override=artifact_numeric
                )
                # 去掉仅用于 EDA 的保留列
                for h in ["_raw_gender", "_raw_smoking_history"]:
                    if h in X_tmp.columns:
                        X_tmp = X_tmp.drop(columns=[h])
                # 对齐模型特征
                X_in = align_features_for_model(X_tmp, model, feature_list)

            # 概率 & 阈值
            proba = model.predict_proba(X_in)[:, 1] if hasattr(model, "predict_proba") else None

            # 若有 threshold，用它出标签；否则退回 model.predict
            if proba is not None and threshold is not None:
                pred = (proba >= float(threshold)).astype(int)
            else:
                pred = model.predict(X_in)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("预测类别（是否糖尿病）", "是" if int(pred[0]) == 1 else "否")
            with col2:
                if proba is not None:
                    st.metric("预测概率（阳性类）", f"{float(proba[0]) * 100:.1f}%")
                else:
                    st.info("当前模型不支持 predict_proba，已显示类别预测。")

            with st.expander("调试信息（可折叠）", expanded=False):
                st.write("模型类型：", type(model))
                st.write("使用的阈值（如有）：", threshold)
                if feature_list:
                    st.write("特征顺序：", feature_list)
                elif hasattr(model, "feature_names_in_"):
                    st.write("特征顺序（model.feature_names_in_）：", list(model.feature_names_in_))
                st.write("用于预测的特征列：", list(X_in.columns))

        except Exception as e:
            st.exception(e)
            st.error("预测过程中出现错误。请检查模型/预处理与特征名是否一致。")

# Footer
st.caption("© 2025 Diabetes App — Streamlit")
