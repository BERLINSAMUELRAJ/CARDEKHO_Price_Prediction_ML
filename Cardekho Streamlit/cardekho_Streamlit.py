import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================
# 1. Load trained model
# =====================
model_path = r"D:\CERTIFICATIONS\GUVI\Capestone Project-3 (Carwale ML)\CarDekho Code\best_random_forest_model.pkl"
model = joblib.load(model_path)

# =====================
# 2. Load training data for medians/modes
# =====================
train_data_path = r"D:\CERTIFICATIONS\GUVI\Capestone Project-3 (Carwale ML)\Carwale Datasets\Cleaned\all_cities_cleaned.xlsx"
df = pd.read_excel(train_data_path)

# Drop target column
target = "price"
numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.drop(target)
categorical_cols = df.select_dtypes(include=["object", "category"]).columns

# Compute defaults
numeric_medians = df[numeric_cols].median()
categorical_modes = df[categorical_cols].mode().iloc[0]

# =====================
# 3. Streamlit UI
# =====================
st.title("🚗 Car Price Prediction")
st.write("Enter your car details to predict the price.")

# User inputs for important features
user_km = st.number_input("Kilometers Driven", min_value=0, value=50000)
user_year = st.number_input("Year of Manufacture", 1990, 2025, 2018)
user_transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
user_fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"])

# =====================
# 4. Prepare input DataFrame
# =====================
features = model.feature_names_in_
input_dict = {}

for col in features:
    if col == 'km':
        input_dict[col] = [user_km]
    elif col == 'modelYear':
        input_dict[col] = [user_year]
    elif col == 'overview_Transmission':
        input_dict[col] = [user_transmission]
    elif col == 'overview_Fuel Type':
        input_dict[col] = [user_fuel]
    elif col in numeric_medians.index:
        input_dict[col] = [numeric_medians[col]]  # numeric default
    elif col in categorical_modes.index:
        input_dict[col] = [categorical_modes[col]]  # categorical default
    else:
        # fallback
        input_dict[col] = [0]

input_df = pd.DataFrame(input_dict)

# =====================
# 5. Predict Price
# =====================
if st.button("Predict Price"):
    try:
        predicted_price = model.predict(input_df)[0]
        # Convert to lakhs for display if needed
        predicted_price_lakhs = predicted_price  # assuming model trained on lakh values
        st.success(f"Estimated Price: ₹{predicted_price_lakhs:,.2f} Lakhs")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# =====================
# 6. Instructions
# =====================
st.write("""
**Instructions:**  
- Fill in all fields with accurate car information.  
- Only key features are required; other values are filled automatically.  
- The predicted price is an estimate based on historical data.  
""")






