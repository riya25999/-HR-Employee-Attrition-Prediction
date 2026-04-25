import streamlit as st
import pandas as pd
import pickle

# ================= TITLE =================
st.title("👨‍💼 HR Employee Attrition Prediction")

# ================= LOAD MODEL =================
model = None
try:
    with open("lr_model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Model load error: {e}")

# ================= INPUT =================
Age = st.number_input("Age", 18, 60, 30)
MonthlyIncome = st.number_input("Monthly Income", 1000, 20000, 5000)
YearsAtCompany = st.number_input("Years at Company", 0, 40, 5)
JobSatisfaction = st.radio("⭐ Job Satisfaction", [1, 2, 3, 4], horizontal=True)
WorkLifeBalance = st.radio("⚖️ Work Life Balance", [1, 2, 3, 4], horizontal=True)
# ================= BUTTON =================
if st.button("🔍 Predict"):

    if model is None:
        st.error("❌ Model not loaded")
    else:
        # Create DataFrame inside button
        input_df = pd.DataFrame({
            'Age':[Age],
            'MonthlyIncome':[MonthlyIncome],
            'YearsAtCompany':[YearsAtCompany],
            'JobSatisfaction':[JobSatisfaction],
            'WorkLifeBalance':[WorkLifeBalance]
        })

        # Match columns with model
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Prediction
        pred = model.predict(input_df)

        if pred[0] == 1:
            st.error("⚠️ Employee likely to leave")
        else:
            st.success("✅ Employee likely to stay")

        # Optional probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)
            st.write(f"📊 Attrition Probability: {prob[0][1]*100:.2f}%")