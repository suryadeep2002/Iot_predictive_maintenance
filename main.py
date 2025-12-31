# Week 4 - Day 22-23: Flask API
# Save as 'app.py'

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import time

app = Flask(__name__)

# Load model and preprocessing pipeline
print("Loading model and scaler...")
model = joblib.load('xgboost_final_model.pkl')
scaler = joblib.load('scaler.pkl')
print("✅ Model loaded successfully!")

# Load feature names
with open('feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]


def create_features(sensor_data):
    """
    Create engineered features from raw sensor data
    """
    df = pd.DataFrame([sensor_data])

    # Add interaction features (must match training)
    df['temp_vib_product'] = df['temperature'] * df['vibration']
    df['temp_pressure_ratio'] = df['temperature'] / (df['pressure'] + 1e-5)
    df['vib_pressure_product'] = df['vibration'] * df['pressure']

    df['combined_stress'] = (
            df['temperature'] / 80 * 0.3 +
            df['vibration'] / 10 * 0.3 +
            df['pressure'] / 110 * 0.2 +
            df['arm'] / 360 * 0.2
    )

    # Add other required features with dummy values for real-time prediction
    # (In production, you'd track historical values)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Placeholder for missing features

    return df[feature_names]


@app.route('/')
def home():
    return jsonify({
        'message': 'IoT Predictive Maintenance API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'POST - Single prediction',
            '/batch_predict': 'POST - Batch predictions'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': time.time()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Real-time prediction endpoint

    Expected JSON format:
    {
        "arm": 180.5,
        "temperature": 65.2,
        "vibration": 8.3,
        "pressure": 98.5
    }
    """
    start_time = time.time()

    try:
        # Get sensor data from request
        sensor_data = request.json

        # Validate input
        required_fields = ['arm', 'temperature', 'vibration', 'pressure']
        if not all(field in sensor_data for field in required_fields):
            return jsonify({
                'error': 'Missing required fields',
                'required': required_fields
            }), 400

        # Create features
        features_df = create_features(sensor_data)

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Predict
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        # Return results
        result = {
            'prediction': 'MAINTENANCE_REQUIRED' if prediction == 1 else 'NORMAL',
            'failure_probability': float(prediction_proba[1]),
            'confidence': float(max(prediction_proba)),
            'health_score': float((1 - prediction_proba[1]) * 100),
            'response_time_ms': round(response_time_ms, 2),
            'timestamp': time.time()
        }

        # Add warning if response time > 50ms
        if response_time_ms > 50:
            result['warning'] = f'Response time ({response_time_ms:.2f}ms) exceeds 50ms target'

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': str(e),
            'timestamp': time.time()
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint

    Expected JSON format:
    {
        "data": [
            {"arm": 180, "temperature": 65, "vibration": 8, "pressure": 98},
            {"arm": 200, "temperature": 55, "vibration": 5, "pressure": 100}
        ]
    }
    """
    start_time = time.time()

    try:
        # Get batch data
        batch_data = request.json.get('data', [])

        if not batch_data:
            return jsonify({'error': 'No data provided'}), 400

        results = []

        for sensor_data in batch_data:
            # Create features
            features_df = create_features(sensor_data)
            features_scaled = scaler.transform(features_df)

            # Predict
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]

            results.append({
                'input': sensor_data,
                'prediction': 'MAINTENANCE_REQUIRED' if prediction == 1 else 'NORMAL',
                'failure_probability': float(prediction_proba[1]),
                'health_score': float((1 - prediction_proba[1]) * 100)
            })

        response_time_ms = (time.time() - start_time) * 1000

        return jsonify({
            'results': results,
            'count': len(results),
            'total_response_time_ms': round(response_time_ms, 2),
            'avg_response_time_ms': round(response_time_ms / len(results), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting Flask API Server")
    print("=" * 60)
    print("API will be available at: http://localhost:5000")
    print("Test with: curl http://localhost:5000/health")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)