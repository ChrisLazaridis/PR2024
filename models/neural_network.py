import torch
import torch.nn as nn
import torch_directml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from least_squares import prepare_dataset, split_dataset
from tqdm import tqdm
from torchviz import make_dot
from typing import Tuple, List


# Set a global random seed for reproducibility
SEED: int = 42

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# noinspection PyShadowingNames
class NeuralNetwork:
    """
    A class to encapsulate the Neural Network model, training, and evaluation processes.
    """

    def __init__(self, input_dim: int, initial_lr: float = 0.1, decay_rate: float = 0.01, batch_size: int = 2048, epochs: int = 100) -> None:
        """
        Initializes the Neural Network model with the given architecture and training parameters.

        Args:
            input_dim (int): Number of input features.
            initial_lr (float): Initial learning rate for optimization.
            decay_rate (float): Decay rate for exponential decay of learning rate.
            batch_size (int): Number of samples per batch during training.
            epochs (int): Number of training epochs.
        """
        # noinspection PyTypeChecker
        self.device: torch.device = torch_directml.device()
        self.initial_lr: float = initial_lr
        self.decay_rate: float = decay_rate
        self.batch_size: int = batch_size
        self.epochs: int = epochs

        # Define the neural network architecture
        self.model: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for regression
        ).to(self.device)

        # Define loss function and optimizer
        self.criterion: nn.MSELoss = nn.MSELoss()
        self.optimizer: torch.optim.SGD = torch.optim.SGD(self.model.parameters(), lr=initial_lr, momentum=0.9)

        # Scheduler for exponential learning rate decay
        self.scheduler: torch.optim.lr_scheduler.ExponentialLR = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1 - decay_rate)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the neural network on the given training data.

        Args:
            X_train (np.ndarray): Feature matrix for training.
            y_train (np.ndarray): Target values for training.
        """
        train_dataset: TensorDataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader: DataLoader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()

        for _ in tqdm(range(self.epochs), desc="Training Epochs", leave=False):
            for features, targets in train_loader:
                # Move data to the GPU
                features, targets = features.to(self.device), targets.to(self.device)

                # Forward pass
                outputs: torch.Tensor = self.model(features).squeeze()
                loss: torch.Tensor = self.criterion(outputs, targets.squeeze())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update learning rate
            self.scheduler.step()

    def predict(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates predictions for the given data.

        Args:
            X (np.ndarray): Feature matrix for prediction.
            y (np.ndarray): Ground truth target values (used for evaluation).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted values and true values.
        """
        dataset: TensorDataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        data_loader: DataLoader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()

        predictions: List[float] = []
        true_values: List[float] = []
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
                outputs: np.ndarray = self.model(features).squeeze().cpu().numpy()
                predictions.extend(outputs)
                true_values.extend(targets.cpu().numpy())

        return np.array(predictions), np.array(true_values)

    def visualize_architecture(self, sample_input: np.ndarray) -> make_dot:
        """
        Visualizes the architecture of the neural network.

        Args:
            sample_input (np.ndarray): Example input to pass through the network.

        Returns:
            make_dot: Graph object visualizing the network.
        """
        sample_tensor: torch.Tensor = torch.tensor(sample_input, dtype=torch.float32).to(self.device)
        y: torch.Tensor = self.model(sample_tensor)
        return make_dot(y, params=dict(self.model.named_parameters()))


if __name__ == "__main__":
    # Configuration
    data_path: str = "../housing/housing_one_hot_encoded.csv"
    scalers: List[str] = ["standard", "min_max", "robust"]
    k: int = 10  # Number of folds for cross-validation

    for scaler in scalers:
        print(f"\nUsing {scaler} scaler")
        X, y, target_scaler = prepare_dataset(data_path, scaler)
        folds: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = split_dataset(X, y, k=k)

        train_results, test_results = [], []

        # Inside the for-loop iterating over folds
        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
            nn_model = NeuralNetwork(input_dim=X.shape[1], initial_lr=0.1, decay_rate=0.01)
            nn_model.train(X_train, y_train)

            y_train_pred, y_train_true = nn_model.predict(X_train, y_train)
            y_test_pred, y_test_true = nn_model.predict(X_test, y_test)

            # Scaled metrics
            train_mse_scaled = mean_squared_error(y_train_true, y_train_pred)
            train_mae_scaled = mean_absolute_error(y_train_true, y_train_pred)
            test_mse_scaled = mean_squared_error(y_test_true, y_test_pred)
            test_mae_scaled = mean_absolute_error(y_test_true, y_test_pred)

            # Descaled metrics
            y_train_pred_original = target_scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
            y_test_pred_original = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
            y_train_true_original = target_scaler.inverse_transform(y_train_true.reshape(-1, 1)).flatten()
            y_test_true_original = target_scaler.inverse_transform(y_test_true.reshape(-1, 1)).flatten()

            train_mse_descaled = mean_squared_error(y_train_true_original, y_train_pred_original)
            train_mae_descaled = mean_absolute_error(y_train_true_original, y_train_pred_original)
            test_mse_descaled = mean_squared_error(y_test_true_original, y_test_pred_original)
            test_mae_descaled = mean_absolute_error(y_test_true_original, y_test_pred_original)

            train_results.append((train_mse_scaled, train_mae_scaled, train_mse_descaled, train_mae_descaled))
            test_results.append((test_mse_scaled, test_mae_scaled, test_mse_descaled, test_mae_descaled))
