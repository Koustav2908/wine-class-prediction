import joblib
import pandas as pd
import streamlit as st

from descriptions import features, sample_data

pipeline = joblib.load("pipeline.pkl")

svm_model = pipeline["svm"]
knn_model = pipeline["knn"]
scaler = pipeline["scaler"]

st.title("🍷 Wine Class Prediction")

st.markdown("""
This app allows users to predict the class of wine based on 13 chemical features using two machine learning models:
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
""")

st.subheader("📊 Model Evaluation")

# model comparision
st.markdown("**Model Accuracy Comparison**")
st.image("visuals/accuracy_comparision.png", caption="SVM vs KNN Accuracy")

# confusion matrices
st.markdown("**Confusion Matrix**")
st.image("visuals/cm_svm.png", caption="SVM Confusion Matrix")
st.image("visuals/cm_knn.png", caption="KNN Confusion Matrix")

# correlation heatmap
st.subheader("📈 Feature Correlation Heatmap")
st.image("visuals/corr.png", caption="Feature Correlation Matrix")

# predict
st.subheader("🔮 Predict Wine Class")

model_choice = st.selectbox(
    "Choose model", ["Support Vector Machines (SVM)", "K-Nearest Neighbors (KNN)"]
)

feature_values = {}

for feature in features:
    feature_values[feature[0]] = st.number_input(
        label=f"**{feature[1]}**",
        value=float(sample_data[feature[0]]),
        format="%.2f",
    )

if st.button("Predict"):
    X = pd.DataFrame([feature_values])
    X_scaled = scaler.transform(X)

    if model_choice == "Support Vector Machines (SVM)":
        pred = svm_model.predict(X_scaled)
    else:
        pred = knn_model.predict(X_scaled)

    st.success(f"Predicted Class: {pred[0]}")
