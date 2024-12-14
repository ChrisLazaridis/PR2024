from typing import Tuple
from numpy import ndarray
from data_preprocessing.scale_features import scale_features
import numpy as np


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
        n_samples, n_features = X.shape  # [n_samples, n_features]
        self.weights = np.zeros(n_features)  # Initialize weights as zeros. Shape: [n_features]
        self.bias = 0  # Initialize bias as zero. Scalar

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):  # x_i: Feature vector for a single sample. Shape: [n_features]
                x_i = np.asarray(x_i, dtype=np.float64)  # Ensure x_i is explicitly cast to ndarray
                linear_output = np.dot(x_i, self.weights) + self.bias  # Linear combination. Scalar
                y_predicted = self.activation(linear_output)  # Predicted label. Scalar

                # Update weights and bias if prediction is incorrect
                if y_predicted != y[idx]:  # y[idx]: Scalar target label for the sample
                    self.update(x_i, int(y[idx]))

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

        :param x: Feature vector for a single sample. Shape: [n_features]
        :param y: Target label (-1 or 1). Scalar
        """
        self.weights += self.learning_rate * y * x  # Update weights. Shape: [n_features]
        self.bias += self.learning_rate * y  # Update bias. Scalar


# noinspection PyShadowingNames
def prepare_dataset(data_path: str, scaler: str, threshold: int = 250000) -> Tuple[ndarray, ndarray]:
    """
    Prepare the dataset for the perceptron model.

    :param data_path: Path to the dataset.
    :param scaler: The scaler to be used ('standard', 'min_max', or 'robust').
    :param threshold: The threshold for the median_house_value to classify binary targets.
    :return: Tuple of feature matrix (X) and target vector (y).
             X: Feature matrix. Shape: [n_samples, n_features]
             y: Binary target vector (-1 or 1). Shape: [n_samples]
    """
    # Scale the dataset
    df = scale_features(data_path, scaler)

    # Create binary target column
    df["binary_target"] = np.where(df["median_house_value"] > threshold, 1, -1)

    # Drop the median_house_value and binary_target columns to create the feature matrix
    X = df.drop(["median_house_value", "binary_target"], axis=1).values  # Shape: [n_samples, n_features]
    y = df["binary_target"].astype(int).values  # Shape: [n_samples]

    # Ensure X is explicitly a NumPy ndarray
    X = np.asarray(X, dtype=np.float64)  # Cast to float64 for consistency

    return X, y


# Example usage
if __name__ == "__main__":
    # Path to the dataset
    data_path = "C:/Users/claza/PycharmProjects/PR2024/housing/housing.csv"

    # Prepare the dataset using a standard scaler
    X, y = prepare_dataset(data_path, scaler="standard", threshold=250000)

    # Initialize and train the perceptron
    perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
    perceptron.train(X, y)

    # Predict the training data and calculate accuracy
    predictions = perceptron.predict(X)  # Shape: [n_samples]
    accuracy = np.mean(predictions == y)  # Calculate accuracy
    print(f"Training Accuracy: {accuracy * 100:.2f}%")
