from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import BaseTransform
import torch


class PadFeatures(BaseTransform):
    """
    A transform that pads node feature vectors to a fixed dimension.
    Args:
        target_dim (int): Desired feature dimension after padding.
    """

    def __init__(self, target_dim):
        """
        Initialize the PadFeatures transform.
        Args:
            target_dim (int): The target number of features per node.
        """
        self.target_dim = target_dim

    def __call__(self, data):
        """
        Apply padding to the node feature matrix in a graph data object.
        If the current number of features per node is less than target_dim,
        pads with zeros. If it exceeds target_dim, raises a ValueError.
        Args:
            data (torch_geometric.data.Data): A graph data object with attribute x of shape [num_nodes, num_features].
        Returns:
            torch_geometric.data.Data: The same data object with padded or unchanged data.x.
        Raises:
            ValueError: If data.x has more features than target_dim.
        """
        diff = self.target_dim - data.x.size(1)
        if diff > 0:
            pad = data.x.new_zeros((data.num_nodes, diff))
            data.x = torch.cat([data.x, pad], dim=1)
        elif diff < 0:
            raise ValueError(
                f"Graph has more features ({data.x.size(1)}) than target dimension ({self.target_dim})."
            )
        return data


def load_data(
    name: str = "ENZYMES",
    use_node_attr: bool = False,
    use_edge_attr: bool = False,
) -> TUDataset:
    """
    Load a dataset from the TUDataset collection.
    Args:
        name (str): The name of the dataset to load. Default is "ENZYMES".
        use_node_attr (bool): Whether to use node attributes. Default is False.
        use_edge_attr (bool): Whether to use edge attributes. Default is False.
    Returns:
        TUDataset: The loaded dataset.
    """
    dataset = TUDataset(
        root="./data/ENZYMES",
        name="ENZYMES",
        use_node_attr=use_node_attr,
        use_edge_attr=use_edge_attr,
    )
    max_features = dataset.num_node_features
    if name == "ENZYMES":
        return dataset
    elif name == "MUTAG":
        dataset = TUDataset(
            root="./data/MUTAG",
            name="MUTAG",
            use_node_attr=use_node_attr,
            use_edge_attr=use_edge_attr,
            transform=PadFeatures(max_features),
        )
    else:
        raise ValueError(
            f"Dataset {name} is not supported. Supported datasets are: ENZYMES, MUTAG."
        )

    return dataset
