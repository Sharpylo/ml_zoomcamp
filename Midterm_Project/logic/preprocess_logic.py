import pickle
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest


def preprocess_dataset(file_path):
    """
    Reads a CSV file, preprocesses the data, and prepares it for further analysis.

    Parameters:
        file_path (str): The file path to the CSV file containing the dataset.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    data = pd.read_csv(file_path)
    data = data.rename(columns=lambda col: col.replace(" ", "_").lower())
    data[["systolic", "diastolic"]] = data["blood_pressure"].str.split("/", expand=True)
    data = data.drop(["blood_pressure"], axis=1)
    return data


def select_top_features_df(df, top_features):
    """
    Selects specific columns (features) from the DataFrame.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        top_features (list): A list of column names to be selected.

    Returns:
        pandas.DataFrame: DataFrame containing only the selected columns.
    """
    selected_features_df = df[top_features]
    return selected_features_df


def preprocess_data(df, target_column, test_size=0.4, val_size=0.5, random_state=1, contamination=0.1):
    """
    Prepares the data for modeling by removing outliers, splitting it into train/validation/test sets,
    and scaling/balancing the data.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): Controls the randomness in splitting the data.
        contamination (float): The proportion of outliers in the dataset.

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series, pandas.Series)
    """
    cleaned_df = remove_outliers(df, contamination, target_column)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(cleaned_df, target_column, test_size, val_size, random_state)
    return scale_and_balance_data(X_train, X_val, X_test, y_train, y_val, y_test)


def remove_outliers(df, contamination, target_column):
    """
    Removes outliers from the DataFrame using Isolation Forest.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        contamination (float): The proportion of outliers in the dataset.
        target_column (str): The name of the target column.

    Returns:
        pandas.DataFrame: DataFrame without the detected outliers.
    """
    iso_forest = IsolationForest(contamination=contamination)
    iso_forest.fit(df)  
    outlier_labels = iso_forest.predict(df)
    cleaned_df = df[outlier_labels == 1]
    return cleaned_df


def split_data(df, target_column, test_size, val_size, random_state):
    """
    Splits the DataFrame into train, validation, and test sets.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): Controls the randomness in splitting the data.

    Returns:
        train, X_val, X_test, y_train, y_val, y_test (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series, pandas.Series)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_and_balance_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Scales the data and balances the training data using SMOTE.

    Parameters:
        X_train (pandas.DataFrame): Training features.
        X_val (pandas.DataFrame): Validation features.
        X_test (pandas.DataFrame): Test features.
        y_train (pandas.Series): Training target values.
        y_val (pandas.Series): Validation target values.
        y_test (pandas.Series): Test target values.

    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series, pandas.Series)
    """
    columns = X_train.columns
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns=columns)
    X_val = pd.DataFrame(sc.transform(X_val), columns=columns)
    
    # Save the StandardScaler using pickle
    with open('data/scaler.pkl', 'wb') as file:
        pickle.dump(sc, file)
    print(f"StandardScaler save 'scaler.pkl'")

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test

