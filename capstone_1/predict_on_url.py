import requests
import os

from logic.constants import FOLDER_PATH
from logic.predict_logic import download_image

def get_user_choice():
    user_input = input('Enter a URL to continue, or choose a predefined URL (1/2/3), or press Enter to stop: ')
    return user_input.strip()

while True:
    # Get user choice
    user_choice = get_user_choice()

    if not user_choice:
        print('Stopping the program.')
        exit(0)

    # If the user enters a predefined choice (1, 2, or 3), use the corresponding URL
    predefined_urls = {
        '1': 'https://fruit-time.ua/images/cache/blog/ru-e67711af-462e-4265-bbc4-d7513a4d0d5b-1110x740r.jpeg',
        '2': 'https://img.fozzyshop.com.ua/57145-thickbox_default/ogurec-ekstra.jpg',
        '3': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Aloe_vera386868234.jpg/1200px-Aloe_vera386868234.jpg'
    }

    url = predefined_urls.get(user_choice)

    # If the user didn't choose a predefined URL, use the entered URL
    if not url:
        url = user_choice

    # Download image if URL is provided
    img = download_image(url, FOLDER_PATH) if url else None

    # The URL of the local Flask server
    url_flask = 'http://127.0.0.1:9697/predict_image'

    files = {'image': img}

    response = requests.post(url_flask, files=files)

    # Check the response
    if response.status_code == 200:
        result = response.json()
        predicted_class = result['predicted_class']
        print(f'Predicted Class: {predicted_class}')
    else:
        print(f'Error: {response.status_code} - {response.text}')
