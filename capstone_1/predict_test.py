import requests

# Path to the image file you want to test
image_path_1 = 'img_test/cucumber.png'
image_path_2 = 'img_test/melon.jpg'
image_path_3 = 'img_test/441123_1.jpg'
image_path_4 = 'img_test/banan8.jpg'
image_path_5 = 'img_test/coconut-exoticfruitscouk-565414.jpg'

# The URL of the local Flask server
url = 'http://127.0.0.1:9697/predict_image'

files = {'image': image_path_1}

response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    result = response.json()
    predicted_class = result['predicted_class']
    print(f'Predicted Class: {predicted_class}')
else:
    print(f'Error: {response.status_code} - {response.text}')
