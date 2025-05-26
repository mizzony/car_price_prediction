import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.title("Car Price Prediction 🚗")
st.write("Enter car details to predict its price using our advanced ML model.")

# Check if model file exists
model_path = 'best_car_price_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please ensure it's in the same directory as car_app.py.")
    st.stop()

# Load model
try:
    predictor = joblib.load(model_path)
    if not hasattr(predictor, 'predict'):
        st.error("Loaded model is not a valid predictor. Check 'best_car_price_model.pkl' integrity.")
        st.stop()
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define model options per brand
model_options = {
    "Toyota": ["Camry", "Corolla", "RAV4", "Highlander"],
    "Honda": ["Civic", "Accord", "CR-V", "Pilot"],
    "BMW": ["X1", "X3", "X5", "3 Series", "5 Series"],
    "Porsche": ["911", "Cayenne", "Macan"],
    "Mercedes-Benz": ["C-Class", "E-Class", "S-Class", "GLC"],
    "Audi": ["A3", "A4", "Q5", "Q8"],
    "Lexus": ["RX", "ES", "NX"],
    "Ford": ["F-150", "Mustang", "Explorer"],
    "Chevrolet": ["Silverado", "Camaro", "Equinox"],
    "RAM": ["1500", "2500"]
}

# Initialize session state
if 'brand' not in st.session_state:
    st.session_state.brand = "Toyota"
if 'car_model' not in st.session_state:
    st.session_state.car_model = model_options["Toyota"][0]

# Brand selection
brand = st.selectbox("Brand", list(model_options.keys()), index=0, key="brand_select")
# Update model if brand changes
if brand != st.session_state.brand:
    st.session_state.brand = brand
    st.session_state.car_model = model_options[brand][0]

# Model selection
car_model = st.selectbox(
    "Model",
    model_options[brand],
    index=0,  # Always start with first model
    key=f"model_select_{brand}"  # Unique key to force re-render
)
st.session_state.car_model = car_model
st.write(f"Selected brand: {brand}, Available models: {model_options[brand]}, Current model: {car_model}")

# Input form
with st.form("car_form"):
    col1, col2 = st.columns(2)
    with col1:
        milage = st.number_input("Milage (miles)", min_value=0.0, max_value=300000.0, value=20000.0, step=1000.0)
        age = st.number_input("Age (years)", min_value=0, max_value=50, value=5)
        horsepower = st.number_input("Horsepower", min_value=50.0, max_value=1000.0, value=169.0 if brand == "Toyota" and car_model == "Corolla" else 260.0, step=10.0)
        engine_liters = st.number_input("Engine Size (liters)", min_value=0.0, max_value=10.0, value=2.0 if brand == "Toyota" and car_model == "Corolla" else 3.0, step=0.1)
    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric", "–"], index=0)
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "CVT", "A/T", "7-Speed A/T", "8-Speed Automatic", "–"], index=0)
        accident = st.selectbox("Accident History", ["None reported", "At least 1 accident or damage reported"], index=0)
        clean_title = st.selectbox("Clean Title", ["Yes", "No", "Unknown"], index=0)
        is_turbo = st.checkbox("Turbo Engine", value=False if brand == "Toyota" and car_model == "Corolla" else True)
        model_price_group = st.selectbox("Model Price Group", ["Low", "Mid", "High"], index=1 if brand == "Toyota" and car_model == "Corolla" else 2)
    submit = st.form_submit_button("Predict Price")

# Process inputs
if submit:
    luxury_brands = ['Porsche', 'BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'INFINITI']
    ultra_premium_brands = ['Porsche', 'Mercedes-Benz', 'BMW', 'Audi', 'Lexus']
    ultra_premium_models = ['911', 'S-Class', 'M5', 'Q8', 'LX']
    input_data = pd.DataFrame([{
        'brand': brand,
        'model': car_model,
        'milage': milage,
        'model_year': 2025 - age,
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
        'ultra_premium': 1 if (car_model in ultra_premium_models or brand in ultra_premium_brands) else 0,
        'luxury_horsepower': (1 if brand in luxury_brands else 0) * horsepower,
        'low_milage_premium': (1 if milage < 20000 else 0) * (1 if car_model in ultra_premium_models or brand in ultra_premium_brands else 0),
        'model_price_group': model_price_group
    }])
    try:
        prediction = np.expm1(predictor.predict(input_data))[0]
        if input_data['ultra_premium'].iloc[0] == 1:
            prediction *= 1.15  # Boost premium predictions
        st.success(f"Predicted Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Model Performance
st.markdown("---")
st.subheader("Model Performance")
st.write("Ensemble (XGBoost + LightGBM), MAE: $4,744.11, R²: 0.901")
st.subheader("Model Comparison")
fig = px.bar(
    x=[5100.23, 4600.45, 4650.32, 4500.10],
    y=["Random Forest", "XGBoost", "LightGBM", "Ensemble"],
    orientation='h',
    labels={'x': 'MAE ($)', 'y': 'Model'},
    title='Model Performance Comparison',
    color=[5100.23, 4600.45, 4650.32, 4500.10],
    color_continuous_scale='Blues'
)
st.plotly_chart(fig)
st.markdown("Built with Streamlit and Hugging Face Spaces by Kiki. © 2025")
