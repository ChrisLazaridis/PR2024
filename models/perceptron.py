from typing import Tuple
from numpy import ndarray
from data_preprocessing.scale_features import scale_features
import numpy as np
from tqdm import tqdm  # For the progress bar


# noinspection PyShadowingNames
class Perceptron:
    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
        """
        Initialize the perceptron model.

        :param learning_rate: The learning rate for the perceptron model.
        :param n_iters: The number of iterations to train the model.
        """
        self.learning_rate = learning_rate  # Scalar
        self.n_iters = n_iters  # Scalar
        self.weights = None  # Weights vector (shape: [n_features])
        self.bias = None  # Bias term (scalar)

    def train(self, X: ndarray, y: ndarray) -> None:
        """
        Train the perceptron model on the input data.

        :param X: Feature matrix (scaled). Shape: [n_samples, n_features]
        :param y: Target labels (-1 or 1). Shape: [n_samples]
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Initialize weights as zeros. Shape: [n_features]
        self.bias = 0  # Initialize bias as zero. Scalar

        # Progress bar for the number of iterations
        for _ in tqdm(range(self.n_iters), desc="Training", unit="iter"):
            all_correct = True  # Flag to track if all points are correctly classified

            for idx, x_i in enumerate(X):
                x_i = np.asarray(x_i)  # Ensure x_i is explicitly cast to ndarray
                linear_output = np.dot(x_i, self.weights) + self.bias  # Linear combination. Scalar
                y_predicted = self.activation(linear_output)  # Predicted label. Scalar

                # Update weights and bias if prediction is incorrect
                if y_predicted != y[idx]:
                    self.update(x_i, int(y[idx]))  # int re-casting cause IDE is crying
                    all_correct = False  # If one point is misclassified, set the flag to False

            # Early stopping if all points are correctly classified
            if all_correct:
                print("Early stopping: all points classified correctly.")
                break

    def predict(self, X: ndarray) -> ndarray:
        """
        Predict the target labels.

        :param X: Feature matrix (scaled). Shape: [n_samples, n_features]
        :return: Predicted labels (-1 or 1). Shape: [n_samples]
        """
        linear_output = np.dot(X, self.weights) + self.bias  # Linear combination for all samples. Shape: [n_samples]
        return np.where(linear_output >= 0, 1, -1)  # Predicted labels (-1 or 1). Shape: [n_samples]

    @staticmethod
    def activation(x: float) -> int:
        """
        Activation function.

        :param x: Linear output. Scalar
        :return: Binary output (-1 or 1). Scalar
        """
        return 1 if x >= 0 else -1

    def update(self, x: ndarray, y: int) -> None:
        """
        Update the weights and bias.

        :param x: Feature vector. Shape: [n_features]
        :param y: Target label (-1 or 1). Scalar
        """
        self.weights += self.learning_rate * y * x  # Update weights. Shape: [n_features]
        self.bias += self.learning_rate * y  # Update bias. Scalar


# noinspection PyShadowingNames
def prepare_dataset(data_path: str, scaler: str) -> Tuple[ndarray, ndarray]:
    """
    Prepare the dataset for the perceptron model.

    :param data_path: Path to the dataset.
    :param scaler: The scaler to be used ('standard', 'min_max', or 'robust').
    :return: Tuple of feature matrix (X) and target vector (y).
             X: Feature matrix. Shape: [n_samples, n_features]
             y: Binary target vector (-1 or 1). Shape: [n_samples]
    """
    # Scale the dataset
    df = scale_features(data_path, scaler)

    # Calculate the median threshold for binary classification
    threshold = df["median_house_value"].median()

    # Create binary target labels based on the threshold
    df["binary_target"] = np.where(df["median_house_value"] > threshold, 1, -1)

    # Drop the 'median_house_value' and 'binary_target' columns to create the feature matrix
    X = df.drop(["median_house_value", "binary_target"], axis=1).values  # Shape: [n_samples, n_features]
    y = df["binary_target"].astype(int).values  # Ensure y is a 1D integer array. Shape: [n_samples]

    # Ensure X is explicitly a NumPy ndarray
    X = np.asarray(X, dtype=np.float64)  # Cast to float64 for consistency

    return X, y


# Example usage
if __name__ == "__main__":
    # Path to the dataset
    data_path = r"C:\Users\claza\PycharmProjects\PR2024\housing\housing_one_hot_encoded.csv"

    # Prepare the dataset using different scalers
    X, y = prepare_dataset(data_path, scaler="standard")
    X_min_max, y_min_max = prepare_dataset(data_path, scaler="min_max")
    X_robust, y_robust = prepare_dataset(data_path, scaler="robust")

    # Initialize and train perceptrons for each scaler
    perceptron_standard = Perceptron(learning_rate=0.01, n_iters=10000)
    perceptron_standard.train(X, y)

    perceptron_min_max = Perceptron(learning_rate=0.01, n_iters=10000)
    perceptron_min_max.train(X_min_max, y_min_max)

    perceptron_robust = Perceptron(learning_rate=0.01, n_iters=10000)
    perceptron_robust.train(X_robust, y_robust)

    # Predict and calculate accuracy
    accuracy_standard = np.mean(perceptron_standard.predict(X) == y)
    accuracy_min_max = np.mean(perceptron_min_max.predict(X_min_max) == y_min_max)
    accuracy_robust = np.mean(perceptron_robust.predict(X_robust) == y_robust)

    # Print accuracies
    print(f"Training Accuracy (Standard Scaler): {accuracy_standard * 100:.2f}%")
    print(f"Training Accuracy (Min-Max Scaler): {accuracy_min_max * 100:.2f}%")
    print(f"Training Accuracy (Robust Scaler): {accuracy_robust * 100:.2f}%")
