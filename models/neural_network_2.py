import torch
import torch.nn as nn
import torch_directml
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from least_squares import prepare_dataset, split_dataset
from tqdm import tqdm
import json
import os

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        """
        Initializes the Neural Network model with the given architecture.
        :param input_dim: Number of input features.
        :param hidden_dims:  List of hidden layer dimensions.
        """
        super(NeuralNetwork, self).__init__()
        layers = []
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))  # Single output for regression
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        :param x:  Input tensor. Shape: [n_samples, n_features]
        :return:  Output tensor. Shape: [n_samples, 1]
        """
        return self.model(x)


def train_model(model: NeuralNetwork, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, epochs: int, scheduler=None) -> None:
    """
    Train the neural network model.
    :param model: Neural network model.
    :param loader: DataLoader for training data.
    :param optimizer:  Optimizer for training.
    :param criterion:  Loss function.
    :param device:  Device to run the model on.
    :param epochs:  Number of training epochs.
    :param scheduler:  Learning rate scheduler.
    :return:
    """
    model.train()
    for _ in tqdm(range(epochs), desc="Training Epochs", leave=False):
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()


def evaluate_model(model: NeuralNetwork, loader: DataLoader, device: torch.device,
                   target_scaler) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the neural network model.
    :param model: The trained neural network model.
    :param loader: DataLoader for evaluation data.
    :param device:  Device to run the model on.
    :param target_scaler:  Scaler for target variable.
    :return:
    """
    model.eval()
    predictions, true_values = [], []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            outputs = model(features).squeeze().cpu().numpy()
            predictions.extend(outputs)
            true_values.extend(targets.cpu().numpy())
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    descaled_predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    descaled_true_values = target_scaler.inverse_transform(true_values.reshape(-1, 1)).flatten()
    return predictions, true_values, descaled_predictions, descaled_true_values


# noinspection PyTypeChecker,PyShadowingNames
def save_checkpoint(output_file: str, state: dict) -> None:
    """
    Save checkpoint state to file.
    :param output_file: Path to save the checkpoint file.
    :param state: Checkpoint state to save.
    :return:
    """
    with open(output_file, "w") as f:
        json.dump(state, f)


# noinspection PyTypeChecker,PyShadowingNames,PyUnboundLocalVariable
def grid_search(data_path: str, output_file: str) -> None:
    """
    Perform grid search for hyperparameter tuning.
    :param data_path: Path to the dataset file.
    :param output_file:  Path to save the checkpoint file.
    :return:
    """
    # Check for existing checkpoint
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            state = json.load(f)
        continue_search = input("Checkpoint found. Continue from last checkpoint? (yes/no): ").lower()
        if continue_search != "yes":
            os.remove(output_file)
            state = None
    else:
        state = None

    # Initialize grid search state
    if state:
        print("Resuming grid search from checkpoint.")
        scalers = state["scalers"]
        learning_rates = state["learning_rates"]
        decay_rates = state["decay_rates"]
        momentums = state["momentums"]
        architectures = state["architectures"]
        epochs_range = state["epochs_range"]
        current_idx = state["current_idx"]
        best_metrics = state["best_metrics"]
        best_hyperparams = state["best_hyperparams"]
    else:
        scalers = ["standard"]
        k = 10
        learning_rate_min, learning_rate_max, learning_rate_step = 0.01, 0.2, 0.01
        decay_min, decay_max, decay_step = 0.01, 0.1, 0.01
        momentum_min, momentum_max, momentum_step = 0.5, 0.99, 0.05
        epoch_min, epoch_max, epoch_step = 50, 250, 25
        architectures = [[128, 64], [256, 128, 64], [512, 256, 128, 64], [1024, 512, 256, 128, 64]]

        learning_rates = np.arange(learning_rate_min, learning_rate_max + learning_rate_step, learning_rate_step)
        decay_rates = np.arange(decay_min, decay_max + decay_step, decay_step)
        momentums = np.arange(momentum_min, momentum_max + momentum_step, momentum_step)
        epochs_range = range(epoch_min, epoch_max + epoch_step, epoch_step)

        current_idx = 0
        best_metrics = {"test_mse": float("inf"), "test_mae": float("inf")}
        best_hyperparams = None

    device = torch_directml.device()
    parameter_grid = [
        (scaler, lr, decay, momentum, arch, epochs)
        for scaler in scalers
        for lr in learning_rates
        for decay in decay_rates
        for momentum in momentums
        for arch in architectures
        for epochs in epochs_range
    ]

    # Resume grid search from last saved point
    parameter_grid = parameter_grid[current_idx:]

    for idx, (scaler, lr, decay, momentum, arch, epochs) in enumerate(parameter_grid, start=current_idx):
        print(f"\n[{idx + 1}/{len(parameter_grid)}] Testing architecture: {arch}, LR: {lr}, Decay: {decay}, "
              f"Momentum: {momentum}, Epochs: {epochs}")
        X, y, target_scaler = prepare_dataset(data_path, scaler)
        folds = split_dataset(X, y, k=k)

        fold_metrics = []
        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(folds):
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.float32))
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                         torch.tensor(y_test, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            model = NeuralNetwork(input_dim=X.shape[1], hidden_dims=arch).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - decay)

            train_model(model, train_loader, optimizer, criterion, device, epochs, scheduler)
            _, _, descaled_predictions, descaled_true_values = evaluate_model(
                model, test_loader, device, target_scaler
            )

            mse = mean_squared_error(descaled_true_values, descaled_predictions)
            mae = mean_absolute_error(descaled_true_values, descaled_predictions)
            fold_metrics.append((mse, mae))

        avg_test_mse = np.mean([m[0] for m in fold_metrics])
        avg_test_mae = np.mean([m[1] for m in fold_metrics])

        if avg_test_mse < best_metrics["test_mse"]:
            best_metrics.update({"test_mse": avg_test_mse, "test_mae": avg_test_mae})
            best_hyperparams = {"scaler": scaler, "lr": lr, "decay": decay, "momentum": momentum, "arch": arch,
                                "epochs": epochs}

        # Save checkpoint every 10 iterations
        if (idx + 1) % 10 == 0 or (idx + 1) == len(parameter_grid):
            state = {
                "scalers": scalers,
                "learning_rates": learning_rates.tolist(),
                "decay_rates": decay_rates.tolist(),
                "momentums": momentums.tolist(),
                "architectures": architectures,
                "epochs_range": list(epochs_range),
                "current_idx": idx + 1,
                "best_metrics": best_metrics,
                "best_hyperparams": best_hyperparams,
            }
            save_checkpoint(output_file, state)

    print("\nBest Hyperparameters:")
    print(best_hyperparams)
    print("Best Metrics:")
    print(best_metrics)


if __name__ == "__main__":
    data_path = "../housing/housing_one_hot_encoded.csv"
    output_file = "grid_search_checkpoint.json"
    grid_search(data_path, output_file)
