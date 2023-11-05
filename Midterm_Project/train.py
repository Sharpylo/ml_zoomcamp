import time
import pickle
from sklearn.ensemble import GradientBoostingClassifier

from logic.train_logic import train_evaluate_model_with_cv
from logic.preprocess_logic import preprocess_data, preprocess_dataset, select_top_features_df


start_time = time.time()
file_path = "data/heart_attack_prediction_dataset.csv"
model_file_path = "data/model.pkl"

processed_data = preprocess_dataset(file_path)

# Features that have the greatest impact on prediction
top_features = [
    'bmi',
    'age',
    'systolic',
    'triglycerides',
    'diastolic',
    'heart_rate',
    'sleep_hours_per_day',
    'exercise_hours_per_week',
    'income',
    'sedentary_hours_per_day',
    'cholesterol',
    "heart_attack_risk"
]

df = select_top_features_df(processed_data, top_features)

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, "heart_attack_risk")

# Create GradientBoostingClassifier and set its parameters
gb_model = GradientBoostingClassifier(random_state=1)

gb_parameters = {
    "n_estimators": 500,
    "learning_rate": 0.2,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 1
}

best_gb_model = train_evaluate_model_with_cv(gb_model, gb_parameters, X_train, y_train, X_val, y_val, X_test, y_test)


end_time = time.time()
execution_time = end_time - start_time

with open(model_file_path, 'wb') as file:
    pickle.dump(best_gb_model, file)
    print(f"The model has been saved to a file '{model_file_path}'")
print(f"Execution time: {round(execution_time, 2)} seconds")