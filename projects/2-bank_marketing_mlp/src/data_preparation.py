import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import TARGET_COLUMN


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("Первые 5 строк:")
    print(df.head())
    print("\n Количество строк и столбцов:")
    print(df.shape)
    print("\n Типы данных:")
    print(df.dtypes)
    print("\n Наличие пропущенных значений по столбцам:")
    print(df.isnull().sum())
    print("\n Распределение целевой переменной:")
    print(df[TARGET_COLUMN].value_counts())


def prepare_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()
    df[target_column] = df[target_column].map({"yes": 1, "no": 0})
    return df


def split_features_target(
    df: pd.DataFrame, target_column: str, drop_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(columns=[target_column] + drop_columns)
    y = df[target_column]
    return x, y


def split_data(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    valid_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    valid_relative_size = valid_size / (1 - test_size)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_temp, y_temp, test_size=valid_relative_size, random_state=random_state
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


def get_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_cols = x.select_dtypes(include=["str"]).columns.tolist()
    numerical_cols = x.select_dtypes(include=["int64"]).columns.tolist()
    return categorical_cols, numerical_cols


def build_preprocessor(categorical_cols: list[str], numerical_cols: list[str]):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_cols,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ]
    )
    return preprocessor


def preprocess_data(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
):
    categorical_cols, numerical_cols = get_feature_types(x_train)

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    x_train_processed = preprocessor.fit_transform(x_train)
    x_valid_processed = preprocessor.transform(x_valid)
    x_test_processed = preprocessor.transform(x_test)

    x_train_processed = np.asarray(x_train_processed, dtype=np.float32)
    x_valid_processed = np.asarray(x_valid_processed, dtype=np.float32)
    x_test_processed = np.asarray(x_test_processed, dtype=np.float32)

    return x_train_processed, x_valid_processed, x_test_processed, preprocessor


def prepare_targets_for_torch(
    y_train: pd.Series, y_valid: pd.Series, y_test: pd.Series
):
    y_train = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_valid = y_valid.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_test = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

    return y_train, y_valid, y_test
