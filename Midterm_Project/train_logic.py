import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score


def train_model(model, X, y, parameters):
    model.set_params(**parameters)
    model.fit(X, y, sample_weight=calculate_sample_weight(y))
    return model


def calculate_sample_weight(y):
    class_counts = np.bincount(y)
    n_samples = len(y)
    weights = n_samples / (len(class_counts) * class_counts)
    sample_weight = np.array([weights[label] for label in y])
    return sample_weight


def retrain_with_errors(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    incorrect_indices = np.where(y_pred_val != y_val)[0]
    X_train_with_errors = pd.concat([X_train, X_val.iloc[incorrect_indices]])
    y_train_with_errors = pd.concat([y_train, y_val.iloc[incorrect_indices]])
    return X_train_with_errors, y_train_with_errors


def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    ac = accuracy_score(y, y_pred)
    cr = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred_proba)

    print(f"{dataset_name} Evaluation")
    print(f"Accuracy on {dataset_name}: {ac}")
    print("Classification Report:")
    print(cr)
    print(f"ROC AUC on {dataset_name}: {roc_auc}")
    print("Confusion Matrix:")
    print(cm)


def train_evaluate_model_with_cv(model, model_parameters, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"_____ {model.__class__.__name__} result_____")

    # Train the model to find the best parameters
    class_weight = {0: 1, 1: 3}
    best_model = train_model(model, X_train, y_train, model_parameters)

    # Evaluate on the validation set
    evaluate_model(best_model, X_val, y_val, "Validation Set")

    # Retrain with errors
    X_train_with_errors, y_train_with_errors = retrain_with_errors(best_model, X_train, y_train, X_val, y_val)
    best_model.fit(X_train_with_errors, y_train_with_errors)

    # Evaluate on the test set after retraining
    evaluate_model(best_model, X_test, y_test, "Test Set")

    # Cross-Validation Evaluation
    cross_validate_model(model, X_train, y_train)

    return best_model


def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f"Cross-Validation ROC AUC Scores: {scores}")
    print(f"Mean ROC AUC: {scores.mean()}")