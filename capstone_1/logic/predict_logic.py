import os
import requests
from urllib.parse import urlparse

from logic.train_logic import create_custom_model
from logic.constants import IMG_SIZE, classes


def load_and_create_model(model_file_path):
    # Recreate the model architecture
    recreated_model = create_custom_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(classes))

    # Load the saved weights for the recreated model
    recreated_model.load_weights(model_file_path)

    return recreated_model

def download_image(url, folder_path):
    # Get file name from URL
    parsed_url = urlparse(url)
    file_name = os.path.join(folder_path, os.path.basename(parsed_url.path))

    # Send a request to download the image
    response = requests.get(url)
    
    # Create a folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file for writing in binary mode
        with open(file_name, 'wb') as file:
            # Write the contents of the file
            file.write(response.content)
        print(f"The picture has been successfully downloaded: {file_name}")
    else:
        print(f"Error when downloading an image. Response code: {response.status_code}")
    
    return file_name
