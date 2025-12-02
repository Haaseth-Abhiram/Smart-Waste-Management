
import pandas as pd
from prophet import Prophet
import joblib
import os

df = pd.read_csv("data/waste_data.csv")

df.rename(columns={'Date': 'ds', 'Total_Waste_Tons': 'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

model = Prophet()
model.fit(df)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/prophet_model.pkl")

print("Model trained successfully")
