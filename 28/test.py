import requests

# Test data (4 features for Iris dataset)
test_data = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

# Make prediction request
response = requests.post("http://localhost:8000/predict", json=test_data)
print(response.json())