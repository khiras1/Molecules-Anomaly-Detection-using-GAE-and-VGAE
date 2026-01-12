import torch.nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GAEENcoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        dropout: float = 0.3,
    ):
        """Graph Autoencoder Encoder using GCNConv layers.
        Args:
            in_channels (int): Number of input features per node.
            hidden_channels (int): Number of hidden channels in the first GCN layer.
            latent_dim (int): Dimension of the latent space.
            dropout (float, optional): Dropout rate. Defaults to 0.3.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.bn1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
