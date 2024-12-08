import pandas as pd

data_path = "C:/Users/claza/PycharmProjects/PR2024/housing/housing.csv"

# encode the data of the column "ocean_proximity" based on

df = pd.read_csv(data_path)

# Encode the "ocean_proximity" column
encoded_df = pd.get_dummies(df, columns=["ocean_proximity"], prefix="ocean_proximity")

# Display the encoded data
print(encoded_df.head())

# Save the encoded data to a new CSV file
encoded_df.to_csv("C:/Users/claza/PycharmProjects/PR2024/housing/housing_one_hot_encoded.csv", index=False)
