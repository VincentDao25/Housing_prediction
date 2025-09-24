import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add models directory to path
sys.path.append('models')
from predict import predict_price, load_model

st.set_page_config(
    page_title="Melbourne Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model info for dropdown options
try:
    _, _, model_info = load_model()
    
    st.title("🏠 Melbourne Housing Price Predictor")
    st.markdown("---")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 2)
        car_spaces = st.slider("Car Spaces", 0, 4, 1)
        land_size = st.number_input("Land Size (sqm)", 0, 2000, 500)
        
        st.subheader("Location")
        property_type = st.selectbox("Property Type", model_info['property_types'])
        suburb = st.selectbox("Suburb", model_info['suburbs'])
        agent = st.selectbox("Real Estate Agent", model_info['agents'])
    
    with col2:
        st.subheader("Nearby Amenities (within 500m)")
        supermarket = st.slider("Supermarkets", 0, 20, 6)
        school = st.slider("Schools", 0, 20, 9)
        hospital = st.slider("Hospitals", 0, 20, 9)
        gym = st.slider("Gyms", 0, 20, 7)
        restaurant = st.slider("Restaurants", 0, 20, 15)
        
        st.subheader("Prediction")
        if st.button("🔮 Predict Price", type="primary"):
            property_data = {
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'car_spaces': car_spaces,
                'land_size_sqm': land_size,
                'supermarket': supermarket,
                'school': school,
                'hospital': hospital,
                'gym': gym,
                'restaurant': restaurant,
                'property_type': property_type,
                'suburb': suburb,
                'agent': agent
            }
            
            predicted_price = predict_price(property_data)
            
            st.success(f"Predicted Price: **${predicted_price:,}**")
            
            # Show comparison to market
            market_avg = model_info['price_range']['mean']
            difference = predicted_price - market_avg
            if difference > 0:
                st.info(f"${difference:,} above market average")
            else:
                st.info(f"${abs(difference):,} below market average")
    
    # Model info sidebar
    st.sidebar.subheader("Model Information")
    st.sidebar.write(f"**Model:** Random Forest")
    st.sidebar.write(f"**Accuracy (R²):** {model_info['model_performance']['R2']:.1%}")
    st.sidebar.write(f"**Average Error:** ${model_info['model_performance']['MAE']:,}")
    
except Exception as e:
    st.error("Error loading model. Please ensure model files exist.")
    st.write(f"Error details: {e}")