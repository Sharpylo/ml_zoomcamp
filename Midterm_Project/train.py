import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from train_logic import evaluate_model, train_and_evaluate_model, train_and_evaluate_with_cross_validation
from preprocess_logic import preprocess_data, preprocess_dataset, select_top_features_df


file_path = "heart_attack_prediction_dataset.csv"

processed_data = preprocess_dataset(file_path)

top_features = [
    'diet', 'income', 'bmi', 'exercise_hours_per_week', 'triglycerides', 'sedentary_hours_per_day',
    'cholesterol', 'age', 'systolic', 'diastolic', 'heart_rate', 'sleep_hours_per_day', "heart_attack_risk"
]

df = select_top_features_df(processed_data, top_features)

X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df, "heart_attack_risk")


# Create GradientBoostingClassifier and set its parameters
gb_model = GradientBoostingClassifier(random_state=1)
gb_parameters = {
    "n_estimators": 200,
    "learning_rate": 0.2,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2
}
best_gb_model = train_and_evaluate_model(gb_model, gb_parameters, X_train, y_train, X_val, y_val, X_test, y_test)


# Create RandomForestClassifier and set its parameters
rf_model = RandomForestClassifier(random_state=1)
rf_parameters = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 'log2'
}
best_rf_model = train_and_evaluate_model(rf_model, rf_parameters, X_train, y_train, X_val, y_val, X_test, y_test)


# Combine the best models into a VotingClassifier ensemble
voting_classifier = VotingClassifier(estimators=[
    ('best_gb', best_gb_model),
    ('best_rf', best_rf_model)
], voting='soft')  # Use soft voting for probability-based predictions
print("______VotingClassifier result_____")
# Объединение тренировочного и валидационного набора данных
X_train_combined = pd.concat([X_train, X_val], axis=0)
y_train_combined = pd.concat([y_train, y_val], axis=0)

# Fit the VotingClassifier on the training data
voting_classifier.fit(X_train_combined, y_train_combined)

# Evaluate the VotingClassifier on the test set
evaluate_model(voting_classifier, X_test, y_test, "Ensemble (VotingClassifier)")


# Models to be stacked
estimators = [
    ('best_gb', best_gb_model),
    ('best_rf', best_rf_model)
]

# StackingClassifier with Logistic Regression as the meta-classifier
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)

# Fit the stacking classifier on the training data
stacking_classifier.fit(X_train, y_train)

# Evaluate the stacking classifier on the test set
evaluate_model(stacking_classifier, X_test, y_test, "Stacking Ensemble")


# GradientBoostingClassifier
print("_____GradientBoostingClassifier result with Cross-Validation_____")
train_and_evaluate_with_cross_validation(best_gb_model, X_train_combined, y_train_combined)

# RandomForestClassifier
print("_____RandomForestClassifier result with Cross-Validation_____")
train_and_evaluate_with_cross_validation(best_rf_model, X_train_combined, y_train_combined)

# VotingClassifier
print("_____VotingClassifier result with Cross-Validation_____")
train_and_evaluate_with_cross_validation(voting_classifier, X_train_combined, y_train_combined)

# StackingClassifier
print("_____StackingClassifier result with Cross-Validation_____")
train_and_evaluate_with_cross_validation(stacking_classifier, X_train_combined, y_train_combined)