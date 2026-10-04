import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page configuration
st.set_page_config(
    page_title="Agroforestry Benefit Predictor",
    page_icon="🌱",
    layout="centered"
)

# Load pre-trained models and saved metadata
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_agroforestry_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    features = joblib.load("top_features.pkl")
    return model, preprocessor, features

try:
    model, preprocessor, top_features = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading models or preprocessor files: {e}")
    artifacts_loaded = False

st.title("🌱 Costs & Benefits of Agroforestry in Africa")
st.write("""
This interactive app uses a tuned **Random Forest Regressor** to estimate the 
economic benefits of agroforestry practices based on project costs and location traits.
""")

if artifacts_loaded:
    st.header("⚙️ Enter Parameters for Prediction")
    
    # Setup user inputs for features
    col1, col2 = st.columns(2)
    
    with col1:
        totalusd2020 = st.number_input("Total Cost (USD 2020)", min_value=0.0, value=152.0, step=10.0)
        EcC_total_max = st.number_input("Total Maximum Cost (Local Currency)", min_value=0.0, value=0.0, step=10.0)
        EcB_income_max = st.number_input("Maximum Tree Income (Local Currency)", min_value=0.0, value=0.0, step=10.0)

    with col2:
        EcB_tree = st.number_input("Specific Tree Benefits", min_value=0.0, value=0.0, step=10.0)
        curr_EcC_Transplanting = st.selectbox("Transplanting Currency Unit", ["FCFA ha-1", "Unspecified", "US$ ha-1", "N/A"])
        curr_EcC_total_max = st.selectbox("Total Max Cost Currency Unit", ["FCFA ha-1", "Unspecified", "US$ ha-1", "GH? ha-1", "N/A"])

    # Generate base representation of all features matching original X schema
    # Streamlit passes inputs to our preprocessor pipeline
    if st.button("🔮 Predict Benefits"):
        # Synthesize a complete observation matching the training schema
        # We initiate a single-row DataFrame with 'Unspecified' or median values for unprovided inputs
        dummy_row = {}
        
        # Initialize standard inputs with placeholder values
        dummy_row['totalusd2020'] = totalusd2020
        dummy_row['EcC_total_max'] = EcC_total_max
        dummy_row['EcB_income_max'] = EcB_income_max
        dummy_row['EcB_tree'] = EcB_tree
        dummy_row['curr_EcC_Transplanting'] = curr_EcC_Transplanting
        dummy_row['curr_EcC_total_max'] = curr_EcC_total_max
        
        # Create full dataframe representation
        feature_df = pd.DataFrame([dummy_row])
        
        # Align dataframe with dummy columns matching training columns of X
        # Filling non-selected variables with zeros/Unspecified values
        # Load a default empty dataframe matching your preprocessor configuration
        full_X_template = pd.DataFrame(columns=preprocessor.feature_names_in_)
        input_df = pd.concat([full_X_template, feature_df], ignore_index=True)
        
        # Backfill remaining empty values
        for col in input_df.columns:
            if input_df[col].isnull().any():
                if input_df[col].dtype in ['float64', 'int64']:
                    input_df[col] = 0.0
                else:
                    input_df[col] = 'Unspecified'
                    
        try:
            # Preprocess the input data
            input_encoded = preprocessor.transform(input_df)
            
            # Make predictions
            prediction = model.predict(input_encoded)[0]
            
            st.balloons()
            st.success(f"### 💰 Estimated Benefits: **${prediction:,.2f} USD** (2020 adjusted)")
            
        except Exception as pred_error:
            st.error(f"Error processing prediction: {pred_error}")
