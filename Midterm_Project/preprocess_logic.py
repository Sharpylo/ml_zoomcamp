import pickle
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest


def preprocess_dataset(file_path):
    data = pd.read_csv(file_path)
    data = data.rename(columns=lambda col: col.replace(" ", "_").lower())
    data[["systolic", "diastolic"]] = data["blood_pressure"].str.split("/", expand=True)
    data = data.drop(["blood_pressure"], axis=1)
    return data


def select_top_features_df(df, top_features):
    selected_features_df = df[top_features]
    return selected_features_df


def preprocess_data(df, target_column, test_size=0.4, val_size=0.5, random_state=1, contamination=0.1):
    cleaned_df = remove_outliers(df, contamination, target_column)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(cleaned_df, target_column, test_size, val_size, random_state)
    return scale_and_balance_data(X_train, X_val, X_test, y_train, y_val, y_test)


def remove_outliers(df, contamination, target_column):
    iso_forest = IsolationForest(contamination=contamination)
    iso_forest.fit(df)  
    outlier_labels = iso_forest.predict(df)
    cleaned_df = df[outlier_labels == 1]
    return cleaned_df


def split_data(df, target_column, test_size, val_size, random_state):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_and_balance_data(X_train, X_val, X_test, y_train, y_val, y_test):
    columns = X_train.columns
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns=columns)
    X_val = pd.DataFrame(sc.transform(X_val), columns=columns)
    
    # Save the StandardScaler using pickle
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(sc, file)
    print(f"StandardScaler save 'scaler.pkl'")

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

