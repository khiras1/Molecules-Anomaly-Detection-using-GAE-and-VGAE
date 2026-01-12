import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class VGAEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int, dropout: float = 0.3):
        """
        Variational Graph Autoencoder (VGAE) encoder using GCNConv layers.
        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Number of hidden units in the first GCN layer.
            latent_dim (int): Dimensionality of the latent space.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logstd = GCNConv(hidden_channels, latent_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        self.conv_mu.reset_parameters()
        self.conv_logstd.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd