# Test script: test_api.py
import requests
import json
import time

# API endpoint
API_URL = "http://localhost:5000"

# 1. Health check
print("Testing /health endpoint...")
response = requests.get(f"{API_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# 2. Single prediction
print("Testing /predict endpoint...")
test_data = {
    "arm": 180.5,
    "temperature": 65.2,
    "vibration": 8.3,
    "pressure": 98.5
}

response = requests.post(
    f"{API_URL}/predict",
    json=test_data,
    headers={'Content-Type': 'application/json'}
)

print(f"Status: {response.status_code}")
result = response.json()
print(json.dumps(result, indent=2))
print(f"\n⏱️  Response time: {result.get('response_time_ms', 'N/A')}ms")

# Check if response time < 50ms
if result.get('response_time_ms', 100) < 50:
    print("✅ Response time < 50ms - PASSED")
else:
    print("⚠️  Response time > 50ms - Optimization needed")

# 3. Batch prediction
print("\n\nTesting /batch_predict endpoint...")
batch_data = {
    "data": [
        {"arm": 180, "temperature": 65, "vibration": 8, "pressure": 98},
        {"arm": 200, "temperature": 55, "vibration": 5, "pressure": 100},
        {"arm": 150, "temperature": 75, "vibration": 9, "pressure": 95}
    ]
}

response = requests.post(
    f"{API_URL}/batch_predict",
    json=batch_data,
    headers={'Content-Type': 'application/json'}
)

print(f"Status: {response.status_code}")
result = response.json()
print(json.dumps(result, indent=2))