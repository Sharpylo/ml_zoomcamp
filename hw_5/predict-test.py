import requests

# Define the client data
client_data_1 = {"job": "retired", "duration": 445, "poutcome": "success"}
client_data_2 = {"job": "unknown", "duration": 270, "poutcome": "failure"}
client_data_3 = {"job": "retired", "duration": 445, "poutcome": "success"}

# URL of your Flask app
url = "http://localhost:9696/predict"  # Replace with the actual host and port

# Send a POST request to the /predict endpoint
response = requests.post(url, json=client_data_3)

# Check if the request was successful
if response.status_code == 200:
    result = response.json()
    probability = result.get('probability')
    if probability is not None:
        print(f"Probability: {round(probability, 3)}")
    else:
        print("Response does not contain a 'probability' key.")
else:
    print(f"Request failed with status code {response.status_code}")
