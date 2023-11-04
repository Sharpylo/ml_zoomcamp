import requests

# URL вашего локального сервера Flask
url = 'http://127.0.0.1:9696/predict'

# Данные для POST-запроса
data_for_predict = [
    {
        "bmi": 31.251233,
        "age": 67,
        "systolic": 158,
        "triglycerides": 286,
        "diastolic": 88,
        "heart_rate": 72,
        "sleep_hours_per_day": 6,
        "exercise_hours_per_week": 4.168189,
        "income": 261404,
        "sedentary_hours_per_day": 6.615001,
        "cholesterol": 208
    },
    {
        "bmi": 27.194973,
        "age": 21,
        "systolic": 165,
        "triglycerides": 235,
        "diastolic": 93,
        "heart_rate": 98,
        "sleep_hours_per_day": 7,
        "exercise_hours_per_week": 1.813242,
        "income": 285768,
        "sedentary_hours_per_day": 4.963459,
        "cholesterol": 389
    },
    {
        "bmi": 35.406146,
        "age": 47,
        "systolic": 161,
        "triglycerides": 527,
        "diastolic": 75,
        "heart_rate": 105,
        "sleep_hours_per_day": 4,
        "exercise_hours_per_week": 3.148438,
        "income": 36998,
        "sedentary_hours_per_day": 2.375214,
        "cholesterol": 250
    },
    {
        "bmi": 32.914151,
        "age": 25,
        "systolic": 138,
        "triglycerides": 180,
        "diastolic": 67,
        "heart_rate": 75,
        "sleep_hours_per_day": 4,
        "exercise_hours_per_week": 18.081748,
        "income": 247338,
        "sedentary_hours_per_day": 9.005234,
        "cholesterol": 356
    }
]

# Ожидаемые ответы
expected_answers = [
    {"probability": 0},
    {"probability": 0},
    {"probability": 1},
    {"probability": 1}
]

# Вызов данных через API
for i, data in enumerate(data_for_predict):
    response = requests.post(url, json=data)
    print(f"Sending data {i+1} - {data}:")
    print("Server Response:", response.json())
    print("Expected:", expected_answers[i])
    print("\n")
