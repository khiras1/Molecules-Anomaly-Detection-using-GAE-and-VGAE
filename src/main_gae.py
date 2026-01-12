import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from load_data import load_data
from gae_encoder import GAEENcoder
from train_model import train_model_gae
from recon_loss import GraphAnomalyEvaluator
from extract_embeddings import extract_embeddings
from classify import classify_embeddings
from gmm import EmbeddingAnomalyEvaluator
from torch_geometric.nn import GAE
from torch_geometric.loader import DataLoader
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    Args:
        seed (int): Seed value to initialize random number generators. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_model(
    in_channels: int, hidden_channels: int, latent_dim: int, dropout: float, lr: float
):
    """
    Create and initialize a Graph Autoencoder (GAE) model and its optimizer.
    Args:
        in_channels (int): Number of input feature dimensions.
        hidden_channels (int): Number of hidden layer units in the encoder.
        latent_dim (int): Dimensionality of the latent representation.
        dropout (float): Dropout probability for encoder layers.
        lr (float): Learning rate for the Adam optimizer.
    Returns:
        tuple: (model, optimizer) where model is a GAE instance and optimizer is Adam.
    """
    set_seed()
    encoder = GAEENcoder(in_channels, hidden_channels, latent_dim, dropout).to(device)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def pretrain_model_on_enzymes(
    hidden_channels, latent_dim, dropout, lr, batch_size, epochs, name
):
    """
    Pretrain the GAE model on the ENZYMES dataset.
    Args:
        hidden_channels (int): Hidden layer size for the encoder.
        latent_dim (int): Size of the latent embedding.
        dropout (float): Dropout rate for encoder.
        lr (float): Learning rate for optimizer.
        batch_size (int): Batch size for DataLoader.
        epochs (int): Number of training epochs.
        name (str): Identifier used for saving training artifacts.
    Returns:
        GAE: The pretrained model.
    """
    set_seed()
    dataset = load_data("ENZYMES", use_node_attr=True)
    model, optimizer = create_model(
        dataset.num_features, hidden_channels, latent_dim, dropout, lr
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    train_model_gae(loader, model, optimizer, epochs, name)
    return model


def finetune_on_mutag_anomaly(
    model,
    X_train,
    X_test,
    batch_size,
    lr,
    pretrained=True,
    hidden_channels=32,
    latent_dim=16,
    dropout=0.1,
    name="GAE_MUTAG",
):
    """
    Fine-tune or train from scratch the GAE model on MUTAG for anomaly detection.
    Trains on normals, then evaluates reconstruction and embedding-based anomaly detection using GMM.
    Args:
        model (GAE or None): Pretrained model to fine-tune. If None, trains from scratch.
        X_train (list): Training dataset.
        X_test (list): Test dataset.
        batch_size (int): DataLoader batch size.
        lr (float): Learning rate.
        pretrained (bool): Whether to fine-tune a pretrained model.
        hidden_channels (int): Hidden size if training from scratch.
        latent_dim (int): Latent size if training from scratch.
        dropout (float): Dropout rate if training from scratch.
        name (str): Base name for saving results.
    Returns:
        tuple: (
            f1_recon (float), auc_recon (float), prc_recon (float),
            f1_gmm (float), auc_gmm (float), prc_gmm (float)
        )
    """
    set_seed()
    
    # X_train to u nas 70% klasa 0, X_test to 50%(30% klasy 0 + klasy 1)
    classes = [d.y.item() for d in X_train]

    if len(set(classes)) == 2: # Dla najlepszego modelu - nie chcemy mieć w zbiorze treningowym anomalii, a łączymy zbiór treningowy i walidacyjny
        X_train = [d for d, c in zip(X_train, classes) if c != 1]

    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)

    if not pretrained:
        model, optimizer = create_model(
            in_channels=X_train[0].num_features,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            dropout=dropout,
            lr=lr,
        )
        epochs = 100
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epochs = 50

    train_model_gae(
        train_loader,
        model,
        optimizer,
        epochs,
        name + ("mutag_pretrained" if pretrained else "mutag_nopretrain"),
    )

    recon_evaluator = GraphAnomalyEvaluator(
        model,
        name + ("_recon_pretrained" if pretrained else "_recon_nopretrain"),
        device,
    )
    f1_recon, auc_score_recon, prc_auc_score_recon = recon_evaluator.detect_anomalies(
        X_train, X_test
    )

    X_train, _ = extract_embeddings(model, train_loader, "mutag_train_embed", device)
    X_test, y_test = extract_embeddings(model, test_loader, "mutag_test_embed", device)

    gmm_eval = EmbeddingAnomalyEvaluator(
        X_train,
        X_test,
        y_test,
        name + ("_gmm_pretrained" if pretrained else "_gmm_nopretrain"),
    )
    gmm_eval.fit_gmm(n_components=3)
    f1_gmm, auc_score_gmm, prc_auc_score_gmm = gmm_eval.evaluate_gmm()

    return (
        f1_recon,
        auc_score_recon,
        prc_auc_score_recon,
        f1_gmm,
        auc_score_gmm,
        prc_auc_score_gmm,
    )


def classification_task(
    X_train,
    X_val,
    pretrained_model=None,
    hidden_channels=32,
    latent_dim=16,
    dropout=0.1,
    lr=0.001,
    batch_size=32,
    epochs=100,
    pretrained=True,
    name="mutag_classification",
):
    """
    Perform embedding classification task on MUTAG dataset.
    Trains GAE if needed, extracts embeddings, and runs a downstream classifier.
    Args:
        X_train (list): Training dataset containing normal anomalous graphs.
        X_val (list): Validation dataset containing normal and anomalous graphs.
        pretrained_model (GAE, optional): Pretrained GAE encoder for embeddings.
        hidden_channels (int): Hidden channels if training GAE from scratch.
        latent_dim (int): Latent dimension if training from scratch.
        dropout (float): Dropout rate if training from scratch.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Training epochs for GAE.
        pretrained (bool): Whether to use pretrained_model.
        name (str): Base name for experiment artifacts.
    Returns:
        float: F1 score of downstream classification.
    """
    set_seed()

    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(X_val, batch_size=batch_size, shuffle=False)

    if pretrained:
        model = pretrained_model
    else:
        model, _ = create_model(
            X_train[0].num_features, hidden_channels, latent_dim, dropout, lr
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model_gae(
        train_loader,
        model,
        optimizer,
        epochs,
        name + ("_pretrained_classif" if pretrained else "_nopretrain_classif"),
    )

    X_train, y_train = extract_embeddings(
        model, train_loader, "mutag_train_classif_embed", device
    )
    X_test, y_test = extract_embeddings(
        model, test_loader, "mutag_test_classif_embed", device
    )

    f1 = classify_embeddings(
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=42,
        name=name + ("_pretrained" if pretrained else "_nopretrain"),
    )
    return f1
