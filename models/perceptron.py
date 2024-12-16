from typing import Tuple, List
from numpy import ndarray
from data_preprocessing.scale_features import scale_features
import numpy as np
from tqdm import tqdm
import math
from collections import defaultdict


# noinspection PyShadowingNames
class Perceptron:
    def __init__(self, learning_rate: float = 0.1, decay_rate: float = 0.01, n_iters: int = 1000) -> None:
        """
        Initialize the perceptron model with exponential decay for learning rate.

        :param learning_rate: Initial learning rate for the perceptron model (η₀).
        :param decay_rate: Decay rate for exponential decay (λ).
        :param n_iters: The number of iterations (epochs) to train the model.
        """
        self.learning_rate = learning_rate  # Starting learning rate
        self.decay_rate = decay_rate  # Exponential decay rate
        self.n_iters = n_iters  # Total training iterations
        self.weights = None  # Weight vector (Shape: [n_features])
        self.bias = None  # Bias term (Scalar)

    def train(self, X: ndarray, y: ndarray) -> None:
        """
        Train the perceptron model on the input data.

        :param X: Feature matrix (scaled). Shape: [n_samples, n_features]
        :param y: Target labels (-1 or 1). Shape: [n_samples]
        """
        n_samples, n_features = X.shape  # Dimensions of input data
        self.weights = np.zeros(n_features)  # Initialize weights: [n_features]
        self.bias = 0  # Initialize bias as scalar

        # Training loop with tqdm for visual progress
        for t in range(self.n_iters):
            all_correct = True  # Flag to check for early stopping
            current_learning_rate = self.learning_rate * math.exp(-self.decay_rate * t)  # Exponential decay formula

            for idx, x_i in enumerate(X):  # Loop through each sample
                linear_output = np.dot(x_i, self.weights) + self.bias  # Linear combination: Scalar
                y_predicted = self.activation(linear_output)  # Activation function output: Scalar

                if y_predicted != y[idx]:  # Misclassification case
                    x_i = np.asarray(x_i, dtype=np.float64)  # Explicit conversion to avoid overflow
                    self.update(x_i, int(y[idx]), current_learning_rate)  # Update weights and bias
                    all_correct = False

            if all_correct:  # Early stopping if all points are classified correctly
                print("\nEarly stopping: All points classified correctly.")
                break

    def predict(self, X: ndarray) -> ndarray:
        """
        Predict the target labels.

        :param X: Feature matrix (scaled). Shape: [n_samples, n_features]
        :return: Predicted labels (-1 or 1). Shape: [n_samples]
        """
        linear_output = np.dot(X, self.weights) + self.bias  # Linear combination
        return np.where(linear_output >= 0, 1, -1)  # Activation thresholding

    @staticmethod
    def activation(x: float) -> int:
        """
        Activation function: Binary step.

        :param x: Linear output.
        :return: Returns 1 if x >= 0, else -1.
        """
        return 1 if x >= 0 else -1

    def update(self, x: ndarray, y: int, learning_rate: float) -> None:
        """
        Update weights and bias.

        :param x: Feature vector. Shape: [n_features]
        :param y: Target label (-1 or 1).
        :param learning_rate: Decayed learning rate.
        """
        self.weights += learning_rate * y * x  # Update weights
        self.bias += learning_rate * y  # Update bias


# noinspection PyShadowingNames

def split_dataset(X: ndarray, y: ndarray, k: int = 10) -> List[Tuple[ndarray, ndarray, ndarray, ndarray]]:
    """
    Split the dataset into k stratified folds for cross-validation, preserving the distribution of target values.

    :param X: Feature matrix. Shape: [n_samples, n_features]
    :param y: Target labels. Shape: [n_samples]
    :param k: Number of folds (default: 10).
    :return: List of train-test splits [(X_train, y_train, X_test, y_test)].
    """
    # Group indices by target values to preserve distribution
    class_indices = defaultdict(list)  # Dictionary to store indices for each target value
    for idx, target in enumerate(y):
        class_indices[target].append(idx)
    print(f"Class distribution: {dict((k, len(v)) for k, v in class_indices.items())}")

    # Split indices for each class into approximately k (10) folds
    folds_indices = [[] for _ in range(k)]  # Initialize k(10) empty folds
    for indices in class_indices.values():  # Iterate through each class
        np.random.shuffle(indices)  # Shuffle within class to randomize
        for fold_idx, idx in enumerate(indices):
            folds_indices[fold_idx % k].append(idx)  # Distribute indices into folds round-robin

    # Create train-test splits
    folds = []
    for i in range(k):
        test_indices = np.array(folds_indices[i])  # Test fold indices
        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)  # Remaining indices for training
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
    df = scale_features(data_path, scaler)
    threshold = df["median_house_value"].median()  # Threshold for binary classification
    df["binary_target"] = np.where(df["median_house_value"] > threshold, 1, -1)
    X = df.drop(["median_house_value", "binary_target"], axis=1).values  # Shape: [20640, 14]
    y = df["binary_target"].astype(int).values  # Shape: [20640]
    return np.asarray(X, dtype=np.float64), y


def calculate_metrics(y_true: ndarray, y_pred: ndarray) -> Tuple[float, float]:
    """
    Calculate MSE and MAE metrics.

    :param y_true: True labels. Shape: [n_samples]
    :param y_pred: Predicted labels. Shape: [n_samples]
    :return: Tuple of MSE and MAE.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae


# Main Script
if __name__ == "__main__":
    data_path = r"../housing/housing_one_hot_encoded.csv"
    scalers = ["standard", "min_max", "robust"]  # Scalers to evaluate
    n_iters = 1000  # Number of iterations (epochs)
    learning_rate = 0.1  # Initial learning rate
    decay_rate = 0.01  # Decay rate for learning rate
    k = 10  # Number of folds for cross-validation

    for scaler in scalers:
        print(f"\nRunning Perceptron with {scaler} scaler:")
        X, y = prepare_dataset(data_path, scaler)
        folds = split_dataset(X, y, k=k)

        train_mse_list, train_mae_list = [], []
        test_mse_list, test_mae_list = [], []

        # Cross-validation loop with tqdm
        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(
                tqdm(folds, desc="Cross-validation", unit="fold", position=0, leave=True)):
            perceptron = Perceptron(learning_rate=learning_rate, decay_rate=decay_rate, n_iters=n_iters)
            perceptron.train(X_train, y_train)

            y_pred_train = perceptron.predict(X_train)
            train_mse, train_mae = calculate_metrics(y_train, y_pred_train)

            y_pred_test = perceptron.predict(X_test)
            test_mse, test_mae = calculate_metrics(y_test, y_pred_test)

            train_mse_list.append(train_mse)
            train_mae_list.append(train_mae)
            test_mse_list.append(test_mse)
            test_mae_list.append(test_mae)

        print("\n--- Average Metrics ---")
        print(f"Average Training MSE: {np.mean(train_mse_list):.4f}, MAE: {np.mean(train_mae_list):.4f}")
        print(f"Average Testing MSE: {np.mean(test_mse_list):.4f}, MAE: {np.mean(test_mae_list):.4f}")
