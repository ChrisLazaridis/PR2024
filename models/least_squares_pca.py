from typing import Tuple, List
from numpy import ndarray
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler
from time import time


# Least Squares Regression Class
# noinspection PyShadowingNames,DuplicatedCode
class LeastSquares:
    def __init__(self) -> None:
        self.weights = None
        self.bias = None

    def train(self, X: ndarray, y: ndarray) -> None:
        n_samples, n_features = X.shape
        X_bias = np.hstack((X, np.ones((n_samples, 1))))  # Add bias term
        params = np.linalg.pinv(X_bias) @ y
        self.weights = params[:-1]
        self.bias = params[-1]

    def predict(self, X: ndarray) -> ndarray:
        return X @ self.weights + self.bias


# noinspection PyShadowingNames,DuplicatedCode
def split_dataset(X: ndarray, y: ndarray, k: int = 10, bins: int = 10) -> List[Tuple[ndarray, ndarray, ndarray, ndarray]]:
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
def prepare_dataset_with_gaussian_kernel(data_path: str, n_components: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    df = pd.read_csv(data_path)
    df.fillna(df.median(), inplace=True)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_columns = df.columns.difference(["median_house_value"])
    X = scaler.fit_transform(df[feature_columns].values)
    y = target_scaler.fit_transform(df["median_house_value"].values.reshape(-1, 1))

    # Apply Gaussian Kernel (RBF) using KernelPCA
    print(f"Applying KernelPCA with Gaussian Kernel (n_components = {n_components})...")
    kpca = KernelPCA(n_components=n_components, kernel="rbf")
    X_expanded = kpca.fit_transform(X)

    return X_expanded, y, target_scaler


def calculate_metrics(y_true: ndarray, y_pred: ndarray) -> Tuple[float, float]:
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae


if __name__ == "__main__":
    data_path = r"../housing/housing_one_hot_encoded.csv"
    k = 10  # Number of folds for cross-validation
    n_components = 100  # Increase dimensions using Gaussian Kernel

    print("\nRunning Least Squares with Gaussian Kernel Expansion:")
    X, y, target_scaler = prepare_dataset_with_gaussian_kernel(data_path, n_components=n_components)
    folds = split_dataset(X, y, k=k)

    # Metrics storage
    train_mse_scaled, train_mae_scaled = [], []
    test_mse_scaled, test_mae_scaled = [], []
    train_mse_original, train_mae_original = [], []
    test_mse_original, test_mae_original = [], []

    # Cross-validation loop
    start_time = time()
    # noinspection DuplicatedCode
    for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
        model = LeastSquares()
        model.train(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Scaled metrics
        mse_train_scaled, mae_train_scaled = calculate_metrics(y_train, y_pred_train)
        mse_test_scaled, mae_test_scaled = calculate_metrics(y_test, y_pred_test)
        train_mse_scaled.append(mse_train_scaled)
        train_mae_scaled.append(mae_train_scaled)
        test_mse_scaled.append(mse_test_scaled)
        test_mae_scaled.append(mae_test_scaled)

        # Inverse scale for original metrics
        y_pred_train_original = target_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
        y_pred_test_original = target_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
        y_train_original = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mse_train_original, mae_train_original = calculate_metrics(y_train_original, y_pred_train_original)
        mse_test_original, mae_test_original = calculate_metrics(y_test_original, y_pred_test_original)
        train_mse_original.append(mse_train_original)
        train_mae_original.append(mae_train_original)
        test_mse_original.append(mse_test_original)
        test_mae_original.append(mae_test_original)

    elapsed_time = time() - start_time

    # Display metrics
    print("\n--- Scaled Metrics ---")
    print(f"Average Training MSE: {np.mean(train_mse_scaled):.4f}, MAE: {np.mean(train_mae_scaled):.4f}")
    print(f"Average Testing MSE: {np.mean(test_mse_scaled):.4f}, MAE: {np.mean(test_mae_scaled):.4f}")

    print("\n--- Original Scale Metrics ---")
    print(f"Average Training MSE: {np.mean(train_mse_original):.2f}, MAE: {np.mean(train_mae_original):.2f}")
    print(f"Average Testing MSE: {np.mean(test_mse_original):.2f}, MAE: {np.mean(test_mae_original):.2f}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
