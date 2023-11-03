import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest


def preprocess_dataset(file_path):
    data = pd.read_csv(file_path)

    data = data.drop(["Patient ID"], axis=1)
    data = data.rename(columns=lambda col: col.replace(" ", "_").lower())
    data[["systolic", "diastolic"]] = data["blood_pressure"].str.split("/", expand=True)
    data = data.drop(["blood_pressure"], axis=1)

    columns_to_encode = list(data.select_dtypes(include=['object']).columns)

    label_encoder = LabelEncoder()
    for col in columns_to_encode:
        data[col] = label_encoder.fit_transform(data[col])

    return data


def select_top_features_df(df, top_features):
    selected_features_df = df[top_features]
    return selected_features_df


def preprocess_data(df, target_column, test_size=0.4, val_size=0.5, random_state=1, contamination=0.1):
    # Создание модели Isolation Forest
    iso_forest = IsolationForest(contamination=contamination)  # Указание уровня contamination
    iso_forest.fit(df)  

    # Получение предсказаний для определения аномальных точек (outliers)
    outlier_labels = iso_forest.predict(df)

    # Создание нового DataFrame без аномальных точек
    cleaned_df = df[outlier_labels == 1]  # 1 обозначает отсутствие аномалии

    X = cleaned_df.drop(columns=[target_column])
    y = cleaned_df[target_column]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    # Сохранение названий столбцов
    columns = X_train.columns

    # Создание объекта StandardScaler
    sc = StandardScaler()

    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns=columns)
    X_val = pd.DataFrame(sc.transform(X_val), columns=columns)

    print("Train set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
    
    # Balancing classes using SMOTE (oversampling the minority class)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


