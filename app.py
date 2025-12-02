
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart Waste Prediction", layout="centered")

st.title("🗑️ Smart Waste Management System")
st.write("Predict future waste generation")

model = joblib.load("models/prophet_model.pkl")

future_date = st.date_input("Select Future Date")

if st.button("Predict Waste"):
    future_df = pd.DataFrame({"ds": [pd.to_datetime(future_date)]})
    forecast = model.predict(future_df)
    predicted_value = forecast["yhat"].values[0]

    st.success(f"✅ Predicted Waste on {future_date}: **{predicted_value:.2f} Tons**")
