import pickle
import os

from flask import Flask, request, jsonify


app = Flask('Credit scoring') 


model_file_path = os.path.join(os.path.dirname(__file__), 'model1.bin')
dv_file_path = os.path.join(os.path.dirname(__file__), 'dv.bin')

with open(model_file_path, "rb") as model_file, open(dv_file_path, "rb") as dv_file:
    model = pickle.load(model_file)
    dv = pickle.load(dv_file)
    
    
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X = dv.transform([data])
    probability = model.predict_proba(X)[0][1]
    return jsonify({'probability': probability})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696, use_reloader=False)
