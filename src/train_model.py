import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.loader import DataLoader

sns.set_style("whitegrid")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"


def train_model_vgae(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    model_name: str = "vgae_encoder.pth",
) -> list:
    """
    Train a Variational Graph Autoencoder (VGAE) model.
    Args:
        dataloader (DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The VGAE model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int, optional): Number of epochs to train. Defaults to 100.
        model_name (str, optional): Name of the file to save the model. Defaults to "vgae_encoder.pth".
    Returns:
        list: A list containing the loss history for each epoch.
    """
    loss_history = []
    model.train()
    os.makedirs("./models", exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for data in dataloader:
            data = data.to(device)
            num_nodes = data.num_nodes
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)
            loss = model.recon_loss(z, data.edge_index)
            loss += (1 / num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)

    model_save_path = os.path.join("./models", model_name)
    torch.save(model.state_dict(), model_save_path)
    os.makedirs("./results/loss", exist_ok=True)
    picture_save_path = os.path.join("./results/loss", f"{model_name}_loss_plot.png")
    plt.figure(figsize=(12, 5))
    plt.plot(loss_history)
    model_name = model_name.replace("_", " ")
    plt.title(f"Wartość funkcji straty w trakcie treningu dla modelu VGAE")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość funkcji straty")
    plt.tight_layout()
    plt.savefig(picture_save_path)
    plt.close()
    return loss_history

def train_model_gae(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    model_name: str = "gae_encoder.pth",
) -> list:
    """
    Train a Graph Autoencoder model.
    Args:
        dataloader (DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The Graph Autoencoder model to train.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int, optional): Number of epochs to train. Defaults to 100.
    Returns:
        list: A list containing the loss history for each epoch.
    """
    loss_history = []
    model.train()
    os.makedirs("./models", exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)
            loss = model.recon_loss(z, data.edge_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        loss_history.append(avg_loss)
    model_save_path = os.path.join("./models", model_name)
    torch.save(model.state_dict(), model_save_path)

    os.makedirs("./results/loss", exist_ok=True)
    picture_save_path = os.path.join("./results/loss", f"{model_name}_loss_plot.png")
    plt.figure(figsize=(12, 5))
    plt.plot(loss_history)
    model_name = model_name.replace("_", " ")
    plt.title(f"Wartość funkcji straty w trakcie treningu dla modelu GAE")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość funkcji straty")
    plt.tight_layout()
    plt.savefig(picture_save_path)
    plt.close()
    return loss_history
