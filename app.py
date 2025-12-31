# Week 4 - Day 24-25: Streamlit Dashboard
# Save as 'streamlit_app.py'

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests

# Page config
st.set_page_config(
    page_title="IoT Predictive Maintenance",
    page_icon="🤖",
    layout="wide"
)


# Load model
@st.cache_resource
def load_model():
    model = joblib.load('xgboost_final_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler


model, scaler = load_model()

# Title
st.title("🤖 IoT Predictive Maintenance Dashboard")
st.markdown("### Real-Time Equipment Health Monitoring System")

# Sidebar
st.sidebar.header("🎛️ Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Real-Time Monitor", "Historical Analysis", "Model Performance", "API Testing"]
)

# --------------- PAGE 1: Real-Time Monitoring ---------------
if page == "Real-Time Monitor":
    st.header("📊 Real-Time Sensor Monitoring")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sensor Inputs")
        arm = st.slider("🔧 Arm Position (degrees)", 0.0, 360.0, 180.0, 1.0)
        temperature = st.slider("🌡️ Temperature (°C)", 20.0, 80.0, 50.0, 0.1)

    with col2:
        st.markdown("##")  # Spacing
        vibration = st.slider("📳 Vibration (Hz)", 0.0, 10.0, 5.0, 0.1)
        pressure = st.slider("💨 Pressure (kPa)", 90.0, 110.0, 100.0, 0.1)

    # Create features (simplified for demo)
    input_data = pd.DataFrame({
        'arm': [arm],
        'temperature': [temperature],
        'vibration': [vibration],
        'pressure': [pressure],
        'temp_vib_product': [temperature * vibration],
        'temp_pressure_ratio': [temperature / pressure],
        'vib_pressure_product': [vibration * pressure],
        'combined_stress': [(temperature / 80 * 0.3 + vibration / 10 * 0.3 + pressure / 110 * 0.2 + arm / 360 * 0.2)]
    })

    # Add dummy features
    with open('feature_names.txt', 'r') as f:
        all_features = [line.strip() for line in f.readlines()]

    for col in all_features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[all_features]

    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]

    # Display results
    st.markdown("---")
    st.subheader("🔍 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        status = "⚠️ MAINTENANCE NEEDED" if prediction == 1 else "✅ NORMAL"
        st.metric("Equipment Status", status)

    with col2:
        st.metric("Failure Risk", f"{prediction_proba[1] * 100:.1f}%")

    with col3:
        health = (1 - prediction_proba[1]) * 100
        st.metric("Health Score", f"{health:.1f}/100")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba[1] * 100,
        title={'text': "Failure Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Sensor status table
    st.markdown("---")
    st.subheader("📡 Current Sensor Readings")

    status_data = pd.DataFrame({
        'Sensor': ['Arm Position', 'Temperature', 'Vibration', 'Pressure'],
        'Value': [f"{arm}°", f"{temperature}°C", f"{vibration} Hz", f"{pressure} kPa"],
        'Status': [
            '✅ Normal' if 0 <= arm <= 360 else '⚠️ Alert',
            '✅ Normal' if 20 <= temperature <= 60 else '⚠️ Alert',
            '✅ Normal' if 0 <= vibration <= 7 else '⚠️ Alert',
            '✅ Normal' if 95 <= pressure <= 105 else '⚠️ Alert'
        ]
    })

    st.table(status_data)

# --------------- PAGE 2: Historical Analysis ---------------
elif page == "Historical Analysis":
    st.header("📈 Historical Data Analysis")

    uploaded_file = st.file_uploader("Upload Historical Data (CSV)", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ Loaded {len(df)} records")

        st.subheader("Data Preview")
        st.dataframe(df.head(10))

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        # Time series plot
        sensor_cols = ['arm', 'temperature', 'vibration', 'pressure']
        available_sensors = [col for col in sensor_cols if col in df.columns]

        if available_sensors:
            selected_sensor = st.selectbox("Select Sensor to Visualize", available_sensors)

            fig = px.line(df, y=selected_sensor, title=f'{selected_sensor.capitalize()} Over Time')
            st.plotly_chart(fig, use_container_width=True)

# --------------- PAGE 3: Model Performance ---------------
elif page == "Model Performance":
    st.header("📊 Model Performance Metrics")

    st.subheader("Performance Summary")

    # Load results
    try:
        # Display SHAP images
        st.subheader("SHAP Feature Importance")
        st.image('shap_global_importance.png', caption='Global Feature Importance')

        st.subheader("Precision-Recall Curve")
        st.image('pr_curve_comparison.png', caption='Model Comparison')

        st.subheader("Feature Importance")
        st.image('feature_importance_xgb.png', caption='Top 20 Features')

    except:
        st.warning("Performance visualizations not found. Run training first.")

    st.subheader("Model Information")
    st.markdown("""
    **Model Type:** XGBoost Classifier  
    **Primary Metric:** PR-AUC (Precision-Recall Area Under Curve)  
    **Class Imbalance Handling:** scale_pos_weight parameter  
    **Hyperparameter Tuning:** Optuna (50 trials)  
    **Response Time Target:** < 50ms
    """)

# --------------- PAGE 4: API Testing ---------------
elif page == "API Testing":
    st.header("🔌 API Testing Interface")

    st.markdown("""
    Test the Flask API endpoint. Make sure the API is running at `http://localhost:5000`

    Start API with: `python main.py`
    """)

    api_url = st.text_input("API URL", "http://localhost:5000")

    col1, col2 = st.columns(2)

    with col1:
        test_arm = st.number_input("Arm", 0.0, 360.0, 180.0)
        test_temp = st.number_input("Temperature", 20.0, 80.0, 65.0)

    with col2:
        test_vib = st.number_input("Vibration", 0.0, 10.0, 8.0)
        test_press = st.number_input("Pressure", 90.0, 110.0, 98.0)

    if st.button("🚀 Test API"):
        try:
            import time

            start = time.time()

            response = requests.post(
                f"{api_url}/predict",
                json={
                    "arm": test_arm,
                    "temperature": test_temp,
                    "vibration": test_vib,
                    "pressure": test_press
                }
            )

            elapsed = (time.time() - start) * 1000

            if response.status_code == 200:
                result = response.json()

                st.success(f"✅ API Response received in {elapsed:.2f}ms")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Prediction", result['prediction'])

                with col2:
                    st.metric("Failure Probability", f"{result['failure_probability'] * 100:.1f}%")

                with col3:
                    st.metric("Health Score", f"{result['health_score']:.1f}/100")

                st.json(result)

                if elapsed < 50:
                    st.success("✅ Response time < 50ms")
                else:
                    st.warning(f"⚠️ Response time ({elapsed:.2f}ms) > 50ms target")
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.json(response.json())

        except Exception as e:
            st.error(f"❌ Connection Error: {str(e)}")
            st.info("Make sure the API is running: python main.py`")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | © 2025 IoT Predictive Maintenance")