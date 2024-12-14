# find the median value of the 'median_house_value' column and use it as the threshold to classify the binary targets.

import pandas as pd

data_path = "housing/housing.csv"
df = pd.read_csv(data_path)
# Find the median value of the 'median_house_value' column

median_house_value_median = df["median_house_value"].median()
print(f"Median of 'median_house_value': {median_house_value_median}")