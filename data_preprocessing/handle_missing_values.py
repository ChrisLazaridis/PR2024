import pandas as pd
from typing import Tuple


def handle_missing_values(data_path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Handles missing values in a dataset

    Args:
        data_path (str): The file path to the CSV dataset.

    Returns:
        Tuple[pd.DataFrame, dict]: The processed DataFrame and a dictionary with missing value counts.
    """
    # Read the data from the CSV file
    df = pd.read_csv(data_path)

    # Initialize a counter for missing values
    counter = {
        "longitude": 0,
        "latitude": 0,
        "housing_median_age": 0,
        "total_rooms": 0,
        "total_bedrooms": 0,
        "population": 0,
        "households": 0,
        "median_income": 0,
        "median_house_value": 0,
        "ocean_proximity": 0
    }

    # Replace missing values with the median of the respective column
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            counter[column] = df[column].isnull().sum()
            df[column] = df[column].fillna(df[column].median())

    # Ensure at least one 'ocean_proximity' column is true
    ocean_proximity_cols = [col for col in df.columns if col.startswith("ocean_proximity")]

    if ocean_proximity_cols:  # Check if any ocean_proximity columns exist
        row_sums = df[ocean_proximity_cols].sum(axis=1)
        if row_sums.eq(0).any():
            # Initialize 'ocean_proximity_<1H OCEAN' if it doesn't exist
            if "ocean_proximity_<1H OCEAN" not in df.columns:
                df["ocean_proximity_<1H OCEAN"] = 0
            # Set the new column to 1 where all ocean_proximity columns are 0
            counter["ocean_proximity"] = row_sums.eq(0).sum()
            df.loc[row_sums.eq(0), "ocean_proximity_<1H OCEAN"] = 1

    # Return the processed DataFrame and the counter
    return df, counter
