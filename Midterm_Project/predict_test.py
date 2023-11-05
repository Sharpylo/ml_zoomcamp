import requests
from data.data_test import data_for_predict, expected_answers

# The URL of the local Flask server
url = 'http://127.0.0.1:9696/predict'

# Calling data via API
for i, data in enumerate(data_for_predict):
    response = requests.post(url, json=data)
    print(f"Sending data {i+1} - {data}:")
    print("Server Response:", response.json())
    print("Expected:", expected_answers[i])
    print("\n")
