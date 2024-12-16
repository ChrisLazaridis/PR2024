from typing import Tuple, List
from numpy import ndarray

from data_preprocessing.scale_features import scale_features
from perceptron import calculate_metrics
import numpy as np
from sklearn.utils import shuffle
from time import time


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
def split_dataset(X: ndarray, y: ndarray, k: int = 10, bins: int = 100) -> List[Tuple[ndarray, ndarray, ndarray, ndarray]]:
    """
    Split the dataset into k stratified folds for cross-validation, incorporating the continuous distribution
    of the target variable by binning it into classes.

    :param X: Feature matrix. Shape: [n_samples, n_features]
    :param y: Target vector (continuous values). Shape: [n_samples]
    :param k: Number of folds.
    :param bins: Number of bins to split the continuous target variable into (default: 30).
    :return: List of tuples containing (X_train, y_train, X_test, y_test) for each fold.
    """
    # Bin the continuous target variable into discrete classes
    y = y.astype(float)  # Ensure y is float
    bin_edges = np.linspace(float(y.min()), float(y.max()), num=bins + 1)  # Generate bin edges
    y_binned = np.digitize(y, bins=bin_edges, right=True)  # Bin target values

    # Shuffle data to avoid order bias
    X, y_binned, y = shuffle(X, y_binned, y, random_state=0)  # Shuffle data (random_state for reproducibility)

    # Create stratified folds based on the binned target variable
    n_samples = X.shape[0]
    folds = []

    for fold_idx in range(k):
        test_indices = np.array([], dtype=int)

        # Collect test indices for each bin
        for bin_class in np.unique(y_binned):
            bin_indices = np.where(y_binned == bin_class)[0]  # Get indices for the bin
            bin_fold_size = len(bin_indices) // k  # Number of samples per fold for the bin
            start_idx = fold_idx * bin_fold_size  # Start index for the fold
            end_idx = start_idx + bin_fold_size if fold_idx < k - 1 else len(bin_indices)  # End index for the fold
            test_indices = np.concatenate([test_indices, bin_indices[start_idx:end_idx]])  # Add test indices

        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        folds.append((X_train, y_train, X_test, y_test))

    return folds


# noinspection PyShadowingNames
def prepare_dataset(data_path: str, scaler: str) -> Tuple[ndarray, ndarray]:
    """
    Prepare and scale the dataset.

    :param data_path: Path to the dataset file.
    :param scaler: Scaler type ('standard', 'min_max', 'robust').
    :return: Feature matrix X and target vector y.
    """

    # Load the dataset
    df = scale_features(data_path, scaler)
    X = df.drop(["median_house_value"], axis=1).values  # Shape: [20640, 14]
    y = df["median_house_value"].astype(int).values  # Shape: [20640]
    return np.asarray(X, dtype=np.float64), y


if __name__ == "__main__":
    data_path = r"../housing/housing_one_hot_encoded.csv"
    scalers = ["standard", "min_max", "robust"]
    k = 10  # Number of folds for cross-validation

    for scaler in scalers:
        print(f"\nRunning Least Squares with {scaler} scaler:")
        X, y = prepare_dataset(data_path, scaler)
        folds = split_dataset(X, y, k=k)

        train_mse_list, train_mae_list = [], []
        test_mse_list, test_mae_list = [], []

        # Calculate average and median of target for context
        print(f"Median of target: {np.median(y):.2f}")
        print(f"Mean of target: {np.mean(y):.2f}")

        # Cross-validation loop
        start_time = time()
        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
            model = LeastSquares()
            model.train(X_train, y_train)

            # Predictions and metrics
            y_pred_train = model.predict(X_train)
            train_mse, train_mae = calculate_metrics(y_train, y_pred_train)

            y_pred_test = model.predict(X_test)
            test_mse, test_mae = calculate_metrics(y_test, y_pred_test)

            train_mse_list.append(train_mse)
            train_mae_list.append(train_mae)
            test_mse_list.append(test_mse)
            test_mae_list.append(test_mae)
        elapsed_time_ms = (time() - start_time) * 1000

        print("\n--- Average Metrics ---")
        print(f"Average Training MSE: {np.mean(train_mse_list):.4f}, MAE: {np.mean(train_mae_list):.4f}")
        print(f"Average Testing MSE: {np.mean(test_mse_list):.4f}, MAE: {np.mean(test_mae_list):.4f}")
        print(f"Elapsed time: {elapsed_time_ms:.2f} ms")
