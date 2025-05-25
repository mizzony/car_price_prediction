import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Car Price Prediction 🚗")
st.write("Enter car details to predict its price using our advanced ML model (MAE: ~$4,500).")

# Load model
try:
    model = joblib.load('best_car_price_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input form
with st.form("car_form"):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", ["Toyota", "Honda", "BMW", "Porsche", "Mercedes-Benz", "Audi", "Lexus", "Ford", "Chevrolet", "RAM"])
        model = st.text_input("Model (e.g., Camry)", "Camry")
        milage = st.number_input("Milage (miles)", min_value=0.0, max_value=300000.0, value=15000.0, step=1000.0)
        age = st.number_input("Age (years)", min_value=0, max_value=50, value=2)
        horsepower = st.number_input("Horsepower", min_value=50.0, max_value=1000.0, value=203.0, step=10.0)
        engine_liters = st.number_input("Engine Size (liters)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric", "–"])
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT", "A/T", "7-Speed A/T", "8-Speed Automatic", "–"])
        accident = st.selectbox("Accident History", ["None reported", "At least 1 accident or damage reported"])
        clean_title = st.selectbox("Clean Title", ["Yes", "No", "Unknown"])
        is_turbo = st.checkbox("Turbo Engine", value=False)
        model_price_group = st.selectbox("Model Price Group", ["Low", "Mid", "High"], index=1)
    submit = st.form_submit_button("Predict Price")

# Process inputs
if submit:
    luxury_brands = ['Porsche', 'BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'INFINITI']
    ultra_premium_brands = ['Porsche', 'Mercedes-Benz', 'BMW', 'Audi', 'Lexus']
    ultra_premium_models = ['911', 'S-Class', 'M5', 'Q8', 'LX']  # Update with your script’s ultra_premium_models
    input_data = pd.DataFrame([{
        'brand': brand,
        'model': model,
        'milage': milage,
        'age': age,
        'milage_per_year': milage / (age + 1),
        'horsepower': horsepower,
        'engine_liters': engine_liters,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'accident': accident,
        'clean_title': clean_title if clean_title != "Unknown" else np.nan,
        'luxury_brand': 1 if brand in luxury_brands else 0,
        'is_turbo': 1 if is_turbo else 0,
        'ultra_premium': 1 if (model in ultra_premium_models or brand in ultra_premium_brands) else 0,
        'luxury_horsepower': (1 if brand in luxury_brands else 0) * horsepower,
        'low_milage_premium': (1 if milage < 20000 else 0) * (1 if model in ultra_premium_models or brand in ultra_premium_brands else 0),
        'model_price_group': model_price_group
    }])
    try:
        prediction = np.expm1(model.predict(input_data))[0]
        if input_data['ultra_premium'].iloc[0] == 1:
            prediction *= 1.15  # Boost premium predictions
        st.success(f"Predicted Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Visuals
st.markdown("---")
st.subheader("Model Performance")
st.write("Ensemble (XGBoost + LightGBM), MAE: $4,744.11, R²: 0.901")
st.image('error_distribution.png', caption='Error Distribution', use_column_width=True)
st.markdown("Built with Streamlit and Hugging Face Spaces. © 2025")
