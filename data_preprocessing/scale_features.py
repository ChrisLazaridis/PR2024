# scale the data using robust scaling

import pandas as pd
from sklearn.preprocessing import RobustScaler

df = pd.read_csv("C:/Users/claza/PycharmProjects/PR2024/housing/housing_one_hot_encoded.csv")

# Scale only the numerical columns
scaler = RobustScaler()
df[["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households",
    "median_income"]] = scaler.fit_transform(df[["longitude", "latitude", "housing_median_age", "total_rooms",
                                                 "total_bedrooms", "population", "households", "median_income"]])

# save the scaled data to a new CSV file

df.to_csv("C:/Users/claza/PycharmProjects/PR2024/housing/housing_scaled.csv", index=False)
