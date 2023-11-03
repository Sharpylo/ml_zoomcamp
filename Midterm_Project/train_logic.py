import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score


def train_model(model, X, y, parameters):
    model.set_params(**parameters)
    model.fit(X, y)
    return model


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


def train_and_evaluate_model(model, model_parameters, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"_____ {model.__class__.__name__} result_____")

    # Train the model to find the best parameters
    best_model = train_model(model, X_train, y_train, model_parameters)

    # Evaluate on the validation set
    evaluate_model(best_model, X_val, y_val, "Validation Set")

    # Retrain with errors
    X_train_with_errors, y_train_with_errors = retrain_with_errors(best_model, X_train, y_train, X_val, y_val)
    best_model.fit(X_train_with_errors, y_train_with_errors)

    # Evaluate on the test set after retraining
    evaluate_model(best_model, X_test, y_test, "Test Set")

    return best_model


def train_and_evaluate_with_cross_validation(model, X, y):
    scoring = ['accuracy', 'roc_auc', 'f1']
    scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
    for metric, score in zip(scoring, scores.T):
        print(f"Cross-Validation {metric.capitalize()} Scores: {score}")
        print(f"Mean {metric.capitalize()}: {score.mean()}")