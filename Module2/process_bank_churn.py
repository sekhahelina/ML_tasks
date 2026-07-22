import pandas as pd
import numpy as np

from typing import List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def split_data(
    raw_df: pd.DataFrame,
    target_col: str = "Exited",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split raw dataframe into training and validation sets.

    Args:
        raw_df: Raw dataframe.
        target_col: Target column.
        test_size: Validation set size.
        random_state: Random seed.

    Returns:
        Training and validation dataframes.
    """
    train_df, val_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[target_col]
    )

    return train_df, val_df


def create_inputs_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    input_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Create train and validation inputs and targets.

    Args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        input_cols: Input columns.
        target_col: Target column.

    Returns:
        X_train, train_targets, X_val, val_targets.
    """

    X_train = train_df[input_cols].copy()
    train_targets = train_df[target_col].copy()

    X_val = val_df[input_cols].copy()
    val_targets = val_df[target_col].copy()

    return X_train, train_targets, X_val, val_targets


def impute_missing_values(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    SimpleImputer,
    SimpleImputer
]:
    """
    Fill missing values.

    Numeric columns are filled with the median.
    Categorical columns are filled with the most frequent value.

    Args:
        X_train: Training inputs.
        X_val: Validation inputs.
        numeric_cols: Numeric columns.
        categorical_cols: Categorical columns.

    Returns:
        Processed train data, validation data and fitted imputers.
    """

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    if numeric_cols:
        X_train[numeric_cols] = numeric_imputer.fit_transform(
            X_train[numeric_cols]
        )
        X_val[numeric_cols] = numeric_imputer.transform(
            X_val[numeric_cols]
        )

    if categorical_cols:
        X_train[categorical_cols] = categorical_imputer.fit_transform(
            X_train[categorical_cols]
        )
        X_val[categorical_cols] = categorical_imputer.transform(
            X_val[categorical_cols]
        )

    return (
        X_train,
        X_val,
        numeric_imputer,
        categorical_imputer
    )


def scale_numeric_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler
]:
    """
    Scale numeric features using StandardScaler.

    Args:
        X_train: Training inputs.
        X_val: Validation inputs.
        numeric_cols: Numeric columns.

    Returns:
        Processed train data, validation data and fitted scaler.
    """

    scaler = StandardScaler()

    if numeric_cols:
        X_train[numeric_cols] = scaler.fit_transform(
            X_train[numeric_cols]
        )
        X_val[numeric_cols] = scaler.transform(
            X_val[numeric_cols]
        )

    return X_train, X_val, scaler


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    categorical_cols: List[str]
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    OneHotEncoder
]:
    """
    One-hot encode categorical features.

    Args:
        X_train: Training inputs.
        X_val: Validation inputs.
        categorical_cols: Categorical columns.

    Returns:
        Processed train data, validation data and fitted encoder.
    """

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoder.fit(X_train[categorical_cols])

    train_encoded = encoder.transform(X_train[categorical_cols])
    val_encoded = encoder.transform(X_val[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded = pd.DataFrame(
        train_encoded,
        columns=encoded_cols,
        index=X_train.index
    )

    val_encoded = pd.DataFrame(
        val_encoded,
        columns=encoded_cols,
        index=X_val.index
    )

    X_train = pd.concat(
        [X_train.drop(columns=categorical_cols), train_encoded],
        axis=1
    )

    X_val = pd.concat(
        [X_val.drop(columns=categorical_cols), val_encoded],
        axis=1
    )

    return X_train, X_val, encoder


def preprocess_data(
    raw_df: pd.DataFrame,
    scale_numeric: bool = True
) -> Tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    List[str],
    Optional[StandardScaler],
    OneHotEncoder,
    SimpleImputer,
    SimpleImputer
]:
    """
    Complete preprocessing pipeline.

    Args:
        raw_df: Raw dataframe.
        scale_numeric: Whether to scale numeric features.

    Returns:
        X_train,
        train_targets,
        X_val,
        val_targets,
        input_cols,
        scaler,
        encoder,
        numeric_imputer,
        categorical_imputer
    """

    target_col = "Exited"

    input_cols = raw_df.columns.drop(
        ["Exited", "CustomerId", "Surname"]
    ).tolist()

    train_df, val_df = split_data(raw_df)

    X_train, train_targets, X_val, val_targets = create_inputs_targets(
        train_df,
        val_df,
        input_cols,
        target_col
    )

    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    (
        X_train,
        X_val,
        numeric_imputer,
        categorical_imputer
    ) = impute_missing_values(
        X_train,
        X_val,
        numeric_cols,
        categorical_cols
    )

    scaler = None

    if scale_numeric:
        X_train, X_val, scaler = scale_numeric_features(
            X_train,
            X_val,
            numeric_cols
        )

    X_train, X_val, encoder = encode_categorical_features(
        X_train,
        X_val,
        categorical_cols
    )

    return (
        X_train,
        train_targets,
        X_val,
        val_targets,
        input_cols,
        scaler,
        encoder,
        numeric_imputer,
        categorical_imputer
    )


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    scaler: Optional[StandardScaler],
    encoder: OneHotEncoder,
    numeric_imputer: SimpleImputer,
    categorical_imputer: SimpleImputer,
    scale_numeric: bool = True
) -> pd.DataFrame:
    """
    Preprocess new data using fitted transformers.

    Args:
        new_df: New dataframe.
        input_cols: Columns used during training.
        scaler: Fitted scaler.
        encoder: Fitted encoder.
        numeric_imputer: Fitted numeric imputer.
        categorical_imputer: Fitted categorical imputer.
        scale_numeric: Whether scaling is enabled.

    Returns:
        Processed dataframe.
    """

    X = new_df[input_cols].copy()

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    if numeric_cols:
        X[numeric_cols] = numeric_imputer.transform(
            X[numeric_cols]
        )

    if categorical_cols:
        X[categorical_cols] = categorical_imputer.transform(
            X[categorical_cols]
        )

    if scale_numeric and scaler is not None:
        X[numeric_cols] = scaler.transform(
            X[numeric_cols]
        )

    encoded = encoder.transform(X[categorical_cols])

    encoded = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=X.index
    )

    X = pd.concat(
        [X.drop(columns=categorical_cols), encoded],
        axis=1
    )

    return X