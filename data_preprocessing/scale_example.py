import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Path to your CSV file (replace with your actual path)
data_path = "C:/Users/claza/PycharmProjects/PR2024/housing/housing.csv"

# Read the data from the CSV file, excluding the unwanted columns
df = pd.read_csv(data_path,
                 usecols=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
                          "households", "median_income"])

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
scaled_min_max = min_max_scaler.fit_transform(df)
scaled_min_max_df = pd.DataFrame(scaled_min_max, columns=df.columns)

# Standard Scaling
standard_scaler = StandardScaler()
scaled_standard = standard_scaler.fit_transform(df)
scaled_standard_df = pd.DataFrame(scaled_standard, columns=df.columns)

# Υπολογισμός MSE μεταξύ Min-Max και Standard Scaling
mse_min_max = mean_squared_error(df.mean(axis=0), scaled_min_max_df.mean(axis=0))
mse_standard = mean_squared_error(df.mean(axis=0), scaled_standard_df.mean(axis=0))

# Σύγκριση της κλίμακας
min_max_range = scaled_min_max_df.max(axis=0) - scaled_min_max_df.min(axis=0)
standard_range = scaled_standard_df.max(axis=0) - scaled_standard_df.min(axis=0)

# Plot αρχικών δεδομένων
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
df.boxplot(ax=axes[0])
axes[0].set_title("Αρχικά δεδομένα (Boxplot)")
axes[0].tick_params(axis="x", rotation=45)

# Plot Min-Max Scaling
scaled_min_max_df.boxplot(ax=axes[1])
axes[1].set_title("Min-Max Scaled δεδομένα (Boxplot)")
axes[1].tick_params(axis="x", rotation=45)

# Plot Standard Scaling
scaled_standard_df.boxplot(ax=axes[2])
axes[2].set_title("Standard Scaled δεδομένα (Boxplot)")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

print("MSE Min_Max", mse_min_max, "\n", "MSE Standard:", mse_standard, min_max_range, standard_range)
