import numpy as np
from flask import Flask, request, jsonify
from PIL import UnidentifiedImageError

from logic.constants import IMG_SIZE, NAME, classes
from logic.train_logic import img_preprocessing, create_custom_model

app = Flask('Plants Classifier')

def load_and_create_model(model_file_path):
    # Recreate the model architecture
    recreated_model = create_custom_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(classes))

    # Load the saved weights for the recreated model
    recreated_model.load_weights(model_file_path)

    return recreated_model

# Load the recreated model
recreated_model = load_and_create_model(NAME)
model = recreated_model

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
