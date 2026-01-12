import os

import torch
import torch.nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool


def extract_embeddings(
    model: torch.nn.Module, dataloader: DataLoader, name, device
) -> tuple:
    """Extract embeddings from a trained (Variational) Graph Autoencoder model.
    Args:
        model (torch.nn.Module): The trained Graph Autoencoder model.
        dataloader (DataLoader): DataLoader for the dataset.
        name (str): Name of the dataset, used for saving embeddings.
    Returns:
        tuple: A tuple containing the embeddings and their corresponding labels.
    """
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            pooled_z = global_mean_pool(z, data.batch)
            embeddings.append(pooled_z.cpu())
            labels.append(data.y.cpu())
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    os.makedirs("./data/embeddings", exist_ok=True)
    torch.save(embeddings, f"./data/embeddings/{name}_embeddings.pt")
    torch.save(labels, f"./data/embeddings/{name}_labels.pt")

    return embeddings, labels
