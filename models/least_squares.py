from typing import Tuple, List, Union
from numpy import ndarray

from perceptron import calculate_metrics
import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from time import time
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from data_preprocessing.handle_missing_values import handle_missing_values


# Least Squares Regression Class
# noinspection PyShadowingNames
class LeastSquares:
    def __init__(self) -> None:
        """
        Initialize the Least Squares Regression model.
        """
        self.weights = None  # Weights vector
        self.bias = None  # Bias term

    def train(self, X: ndarray, y: ndarray) -> None:
        """
        Train the model using Least Squares.

        :param X: Feature matrix. Shape: [n_samples, n_features]
        :param y: Target vector. Shape: [n_samples]
        """
        n_samples, n_features = X.shape
        X_bias = np.hstack((X, np.ones((n_samples, 1))))  # Add bias term column

        # Use Moore-Penrose pseudoinverse directly
        params = np.linalg.pinv(X_bias) @ y
        self.weights = params[:-1]  # Weights excluding bias
        self.bias = params[-1]  # Bias term

    def predict(self, X: ndarray) -> ndarray:
        """
        Predict the target values.

        :param X: Feature matrix. Shape: [n_samples, n_features]
        :return: Predicted values. Shape: [n_samples]
        """
        return X @ self.weights + self.bias  # Shape: [n_samples]


# noinspection PyShadowingNames
def split_dataset(X: ndarray, y: ndarray, k: int = 10, bins: int = 10) -> List[Tuple[ndarray, ndarray, ndarray, ndarray]]:
    """
    Split the dataset into k stratified folds for cross-validation, incorporating the continuous distribution
    of the target variable by binning it into classes.

    :param X: Feature matrix. Shape: [n_samples, n_features]
    :param y: Target vector (continuous values). Shape: [n_samples]
    :param k: Number of folds.
    :param bins: Number of bins to split the continuous target variable into (default: 10).
    :return: List of tuples containing (X_train, y_train, X_test, y_test) for each fold.
    """
    # Ensure y is float and create bins using quantiles
    y = y.astype(float)
    y = y.ravel()  # Flatten y to 1D array
    y_binned = pd.qcut(y, q=bins, labels=False, duplicates="drop")

    # Shuffle data to avoid order bias
    X, y_binned, y = shuffle(X, y_binned, y, random_state=0)

    # Create stratified folds based on the binned target variable
    n_samples = X.shape[0]
    folds = []

    for fold_idx in range(k):
        test_indices = np.array([], dtype=int)

        # Collect test indices for each bin
        for bin_class in np.unique(y_binned):
            bin_indices = np.where(y_binned == bin_class)[0]
            bin_splits = np.array_split(bin_indices, k)
            test_indices = np.concatenate([test_indices, bin_splits[fold_idx]])

        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        folds.append((X_train, y_train, X_test, y_test))

    return folds



# noinspection PyShadowingNames
def prepare_dataset(data_path: str, scaler: str) -> Tuple[
    np.ndarray, np.ndarray, Union[StandardScaler, MinMaxScaler, RobustScaler]]:
    """
    Prepare and scale the dataset.

    :param data_path: Path to the dataset file.
    :param scaler: Scaler type ('standard', 'min_max', 'robust').
    :return: Feature matrix X, target vector y, and the target scaler used.
    """
    if scaler not in ["standard", "min_max", "robust"]:
        raise ValueError("Invalid scaler. Choose from 'standard', 'min_max', or 'robust'.")

    # Load the dataset
    df, _ = handle_missing_values(data_path)
    feature_scaler = None
    target_scaler = None
    # Initialize scalers for features and target
    if scaler == "standard":
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    elif scaler == "min_max":
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    elif scaler == "robust":
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
    # Exclude ocean_proximity and median_house_value columns
    exclude_columns = [col for col in df.columns if col.startswith("ocean_proximity")] + ["median_house_value"]

    feature_columns = df.columns.difference(exclude_columns)
    df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])
    df["median_house_value"] = target_scaler.fit_transform(df["median_house_value"].values.reshape(-1, 1))

    X = df[feature_columns].values
    y = df["median_house_value"].values.reshape(-1, 1)
    return X, y, target_scaler


if __name__ == "__main__":
    data_path = r"../housing/housing_one_hot_encoded.csv"
    scalers = ["standard", "min_max", "robust"]
    k = 10  # Number of folds for cross-validation

    for scaler in scalers:
        print(f"\nRunning Least Squares with {scaler} scaler:")
        X, y, target_scaler = prepare_dataset(data_path, scaler)  # Get the target scaler
        folds = split_dataset(X, y, k=k)

        # Lists to store scaled and descaled metrics
        train_mse_scaled, train_mae_scaled = [], []
        test_mse_scaled, test_mae_scaled = [], []

        train_mse_descaled, train_mae_descaled = [], []
        test_mse_descaled, test_mae_descaled = [], []

        # Cross-validation loop
        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
            model = LeastSquares()
            model.train(X_train, y_train)

            # Predictions on training and testing data (scaled)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate scaled metrics
            train_mse_s, train_mae_s = calculate_metrics(y_train, y_pred_train)
            test_mse_s, test_mae_s = calculate_metrics(y_test, y_pred_test)

            train_mse_scaled.append(train_mse_s)
            train_mae_scaled.append(train_mae_s)
            test_mse_scaled.append(test_mse_s)
            test_mae_scaled.append(test_mae_s)

            # Reverse scale the predictions and targets to get the original values
            y_pred_train_original = target_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
            y_pred_test_original = target_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
            y_train_original = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

            # Calculate descaled (original) metrics
            train_mse_d, train_mae_d = calculate_metrics(y_train_original, y_pred_train_original)
            test_mse_d, test_mae_d = calculate_metrics(y_test_original, y_pred_test_original)

            train_mse_descaled.append(train_mse_d)
            train_mae_descaled.append(train_mae_d)
            test_mse_descaled.append(test_mse_d)
            test_mae_descaled.append(test_mae_d)

        # Print scaled and descaled metrics
        print("\n--- Scaled Metrics ---")
        print(f"Average Training MSE: {np.mean(train_mse_scaled):.4f}, MAE: {np.mean(train_mae_scaled):.4f}")
        print(f"Average Testing MSE: {np.mean(test_mse_scaled):.4f}, MAE: {np.mean(test_mae_scaled):.4f}")

        print("\n--- Descaled Metrics (Original Scale) ---")
        print(f"Average Training MSE: {np.mean(train_mse_descaled):.2f}, MAE: {np.mean(train_mae_descaled):.2f}")
        print(f"Average Testing MSE: {np.mean(test_mse_descaled):.2f}, MAE: {np.mean(test_mae_descaled):.2f}")
