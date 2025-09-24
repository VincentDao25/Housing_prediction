import pandas as pd
import numpy as np
import joblib
import pickle

def load_model():
    """Load Random Forest model and preprocessing objects"""
    model = joblib.load('models/random_forest_model.pkl')
    encoder = joblib.load('models/encoder.pkl')
    
    with open('models/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    return model, encoder, model_info

def predict_price(property_data):
    """Predict house price using Random Forest model"""
    model, encoder, model_info = load_model()
    
    # Create DataFrame from input
    input_df = pd.DataFrame([property_data])
    
    # Apply categorical encoding
    categorical_features = model_info['categorical_features']
    encoded_features = encoder.transform(input_df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    
    # Combine with numerical features
    numerical_data = input_df[model_info['numerical_features']]
    processed_data = pd.concat([numerical_data, encoded_df], axis=1)
    
    # Ensure all expected columns are present
    for col in model_info['all_feature_names']:
        if col not in processed_data.columns:
            processed_data[col] = 0
    
    # Reorder columns to match training data
    processed_data = processed_data[model_info['all_feature_names']]
    
    # Make prediction
    prediction = model.predict(processed_data)[0]
    return max(0, int(prediction))