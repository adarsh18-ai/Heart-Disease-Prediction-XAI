import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="Heart Disease Prediction with XAI", page_icon="❤️", layout="wide")

@st.cache_resource
def load_files():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    X_train = joblib.load("X_train.pkl")
    return model, scaler, X_train

model, scaler, X_train = load_files()

feature_names = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

st.title("❤️ Heart Disease Prediction with Explainable AI")

st.sidebar.header("Patient Information")

age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", [0,1])
cp = st.sidebar.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 250, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar >120", [0,1])
restecg = st.sidebar.selectbox("Rest ECG", [0,1,2])
thalach = st.sidebar.number_input("Maximum Heart Rate", 50, 250, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.sidebar.selectbox("Slope", [0,1,2])
ca = st.sidebar.selectbox("Major Vessels", [0,1,2,3,4])
thal = st.sidebar.selectbox("Thal", [0,1,2,3])

input_df = pd.DataFrame({
    "age":[age],"sex":[sex],"cp":[cp],"trestbps":[trestbps],
    "chol":[chol],"fbs":[fbs],"restecg":[restecg],
    "thalach":[thalach],"exang":[exang],"oldpeak":[oldpeak],
    "slope":[slope],"ca":[ca],"thal":[thal]
})

if st.button("Predict"):
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.header("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Heart Disease Risk Detected\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk\n\nProbability: {(1-probability):.2%}")

    st.metric("Heart Disease Probability", f"{probability:.2%}")

    st.header("SHAP Explainability")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_input)

        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = np.array(shap_values).reshape(-1)

        st.subheader("Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": values[:len(feature_names)]
        })

        importance_df["AbsImpact"] = importance_df["Impact"].abs()
        importance_df = importance_df.sort_values("AbsImpact", ascending=False)

        st.bar_chart(importance_df.set_index("Feature")["Impact"])

        st.subheader("Top Risk Factors")

        for _, row in importance_df.head(5).iterrows():
            if row["Impact"] > 0:
                st.write(f"🔴 {row['Feature']} increases risk")
            else:
                st.write(f"🟢 {row['Feature']} decreases risk")

    except Exception as e:
        st.warning(f"SHAP Error: {e}")

    st.header("LIME Explanation")

    try:
        lime_explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=["No Disease", "Disease"],
            mode="classification"
        )

        exp = lime_explainer.explain_instance(
            scaled_input[0],
            model.predict_proba,
            num_features=10
        )

        fig = exp.as_pyplot_figure()
        st.pyplot(fig)

        st.subheader("Clinician-Friendly Explanation")

        for feature, weight in exp.as_list():
            if weight > 0:
                st.write(f"🔴 {feature} increases the likelihood of heart disease.")
            else:
                st.write(f"🟢 {feature} reduces the likelihood of heart disease.")

    except Exception as e:
        st.warning(f"LIME Error: {e}")

st.markdown("---")
st.warning("Educational use only. Not a medical diagnosis tool.")
