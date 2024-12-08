import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Φορτώνουμε τα δεδομένα (αντικαταστήστε με το δικό σας αρχείο CSV)
data_path = "C:/Users/claza/PycharmProjects/PR2024/housing/housing.csv"

# Read the data from the CSV file, excluding the unwanted columns
df = pd.read_csv(data_path,
                 usecols=["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
                          "households", "median_income"])

# Robust Scaling
scaler = RobustScaler()
scaled_robust = scaler.fit_transform(df)
scaled_robust_df = pd.DataFrame(scaled_robust, columns=df.columns)

# Standard Scaling
standard_scaler = StandardScaler()
scaled_standard = standard_scaler.fit_transform(df)
scaled_standard_df = pd.DataFrame(scaled_standard, columns=df.columns)

# Υπολογισμός MSE μεταξύ Robust και Standard Scaling
mse_robust = mean_squared_error(df.mean(axis=0), scaled_robust_df.mean(axis=0))
mse_standard = mean_squared_error(df.mean(axis=0), scaled_standard_df.mean(axis=0))

# Σύγκριση της κλίμακας
robust_range = scaled_robust_df.max(axis=0) - scaled_robust_df.min(axis=0)
standard_range = scaled_standard_df.max(axis=0) - scaled_standard_df.min(axis=0)
original_range = df.max(axis=0) - df.min(axis=0)

# Plot αρχικών δεδομένων
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
df.boxplot(ax=axes[0])
axes[0].set_title("Αρχικά δεδομένα (Boxplot)")
axes[0].tick_params(axis="x", rotation=45)

# Plot Robust Scaling
scaled_robust_df.boxplot(ax=axes[1])
axes[1].set_title("Robust Scaled δεδομένα (Boxplot)")
axes[1].tick_params(axis="x", rotation=45)

# Plot Standard Scaling
scaled_standard_df.boxplot(ax=axes[2])
axes[2].set_title("Standard Scaled δεδομένα (Boxplot)")
axes[2].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

print("MSE Robust", mse_robust, "\n", "MSE Standard:", mse_standard, "\n Robust Range: \n", robust_range,
      "\n Standard Range: \n", standard_range, "\n Original Range: \n", original_range)
