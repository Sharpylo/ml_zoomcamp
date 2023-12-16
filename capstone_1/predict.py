import numpy as np
from flask import Flask, request, jsonify
from PIL import UnidentifiedImageError

from logic.constants import NAME, classes, FOLDER_PATH
from logic.train_logic import img_preprocessing
from logic.predict_logic import load_and_create_model, download_image


app = Flask('Plants Classifier')
# Load the recreated model
model = load_and_create_model(NAME)


@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        # Read image from the request
        img_data = request.files['image'].read()
        img, _ = img_preprocessing(img_data, label=0)
        img_array = np.expand_dims(img, axis=0)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        result = {'predicted_class': classes[predicted_class]}
        return jsonify(result), 200
    except UnidentifiedImageError:
        return jsonify({'error': 'Invalid image format'}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9697, use_reloader=False)
