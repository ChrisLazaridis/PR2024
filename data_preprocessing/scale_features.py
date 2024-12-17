import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from data_preprocessing.handle_missing_values import handle_missing_values


def scale_features(data_path: str, scaler: str) -> pd.DataFrame:
    """
    Scale the features of a dataset using a specified scaler

    Args:
        data_path (str): The file path to the CSV dataset.
        scaler (str): The type of scaler to use. Options: "robust", "min_max", "standard"

    Returns:
        pd.DataFrame: The scaled DataFrame.
    """
    # Handle missing values
    df, _ = handle_missing_values(data_path)

    # Initialize the scaler
    if scaler == "robust":
        scaler = RobustScaler()
    elif scaler == "min_max":
        scaler = MinMaxScaler()
    elif scaler == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaler. Choose from 'robust', 'min_max', or 'standard'.")

    # Exclude ocean_proximity and median_house_value columns
    exclude_columns = [col for col in df.columns if col.startswith("ocean_proximity")]

    # Ensure no NaN values exist
    if df.isnull().sum().any():
        raise ValueError("DataFrame contains NaN values after handling missing values.")

    # Scale the remaining columns
    columns_to_scale = df.columns.difference(exclude_columns)
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df
