
import pickle
import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask('Heart attack risk')

# Load the model and StandardScaler
model_file_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_file_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

with open(model_file_path, "rb") as model_file, open(scaler_file_path, "rb") as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
        
    # Create a DataFrame with consistent column names
    X = pd.DataFrame(data, index=[0])
    
    # Scale the input data
    X_scaled = scaler.transform(X)
    
    # Make predictions
    probability = model.predict_proba(X_scaled)[:, 1]
    probability_rounded = (probability >= 0.5).astype(int)
    
    return jsonify({'probability': probability_rounded.tolist()})



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=9696, use_reloader=False)