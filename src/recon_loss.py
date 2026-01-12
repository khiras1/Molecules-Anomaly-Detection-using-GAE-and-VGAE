import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_curve)
from torch_geometric.loader import DataLoader


class GraphAnomalyEvaluator:
    """
    Evaluator for graph anomaly detection based on reconstruction loss using a GAE model.
    Attributes:
        model (torch.nn.Module): Graph Autoencoder model for computing reconstructions.
        name (str): Identifier for saving result plots.
        device (str): Computation device ('cpu', 'cuda', or 'mps').
    """
    def __init__(self, model: torch.nn.Module, name: str, device: str = "cpu"):
        """
        Initialize the GraphAnomalyEvaluator.
        Args:
            model (torch.nn.Module): A trained or partially trained GAE model.
            name (str): Base name used for saving evaluation artifacts.
            device (str): Device to run computations on. Defaults to 'cpu'.
        """
        self.model = model.to(device)
        self.device = device
        self.name = name

    def compute_recon_scores(self, loader):
        """
        Compute per-graph reconstruction scores for each batch in a DataLoader.
        Uses the GAE/VGAE's encoder and reconstruction loss normalized by edge count.
        Args:
            loader (DataLoader): DataLoader yielding graph data objects with x, edge_index, and y attributes.
        Returns:
            tuple:
                scores (list of float): Reconstruction loss per graph divided by number of edges.
                labels (list of int): True binary labels corresponding to each graph (0=normal, 1=anomaly).
        """
        self.model.eval()
        scores = []
        labels = []

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                z = self.model.encode(data.x, data.edge_index)
                loss = self.model.recon_loss(z, data.edge_index)
                score = loss.item() / data.num_edges
                scores.append(score)
                labels.append(int(data.y.item()))

        return scores, labels

    def calculate_threshold(self, train_scores):
        """
        Determine an anomaly detection threshold using the IQR method.
        Threshold = Q3 + 1.5 * (Q3 - Q1) computed on training reconstruction scores.
        Args:
            train_scores (list of float): Reconstruction scores from training (normal) graphs.
        Returns:
            float: Computed anomaly threshold.
        """
        scores_tensor = torch.tensor(train_scores)
        q1 = torch.quantile(scores_tensor, 0.25)
        q3 = torch.quantile(scores_tensor, 0.75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        return threshold

    def plot_confusion_matrix(self, cm, save_path):
        """
        Plot and save a confusion matrix heatmap.
        Args:
            cm (np.ndarray): Confusion matrix array of shape [2, 2].
            save_path (str): Filesystem path to save the PNG image.
        """
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
        )
        plt.xlabel("Wartość przewidziana")
        plt.ylabel("Wartość rzeczywista")
        plt.title(f"Macierz pomyłek (recon loss)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curve(self, fpr, tpr, auc_score, save_path):
        """
        Plot and save the Receiver Operating Characteristic (ROC) curve.
        Args:
            fpr (array-like): False positive rates at varying thresholds.
            tpr (array-like): True positive rates at varying thresholds.
            auc_score (float): Area under the ROC curve.
            save_path (str): Filesystem path to save the PNG image.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Wskaźnik False Positive')
        plt.ylabel('Wskaźnik True Positive')
        plt.title(f'Krzywa ROC - (recon loss)')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_precision_recall_curve(self, precision, recall, prc_auc, save_path):
        """
        Plot and save the Precision-Recall curve.
        Args:
            precision (array-like): Precision values at varying thresholds.
            recall (array-like): Recall values at varying thresholds.
            prc_auc (float): Average precision score (area under PR curve).
            save_path (str): Filesystem path to save the PNG image.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2, label=f'PRC (AUC = {prc_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Krzywa Precision-Recall - (recon loss)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def detect_anomalies(self, train_set, test_set, save_results: bool = True):
        """
        Perform end-to-end anomaly detection pipeline and optionally save plots.
        Steps:
            1. Compute reconstruction scores on training (normal) graphs.
            2. Compute threshold via IQR method.
            3. Compute reconstruction scores and labels on test graphs.
            4. Classify anomalies using threshold, compute F1, ROC AUC, and PRC AUC.
            5. Save confusion matrix, ROC curve, and PRC curve if save_results=True.
        Args:
            train_set (list): List of normal graph data objects for threshold calibration.
            test_set (list): List of graph data objects including anomalies for evaluation.
            save_results (bool): Whether to save evaluation plots to disk. Defaults to True.
        Returns:
            tuple:
                f1 (float): Binary F1 score of anomaly classification.
                roc_auc (float): AUC of ROC curve.
                prc_auc (float): AUC of Precision-Recall curve.
        """
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        train_scores, _ = self.compute_recon_scores(train_loader)

        threshold = self.calculate_threshold(train_scores)

        test_scores, test_labels = self.compute_recon_scores(test_loader)

        predictions = [1 if score > threshold else 0 for score in test_scores]
        f1 = f1_score(test_labels, predictions, average='binary', zero_division=0)
        cm = confusion_matrix(test_labels, predictions)

        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        auc_score = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(test_labels, test_scores)
        prc_auc_score = average_precision_score(test_labels, test_scores)

        if save_results:
            os.makedirs("./results/cm", exist_ok=True)
            cm_path = os.path.join("./results/cm", f"{self.name}_confusion_matrix_recon_loss.png")
            self.plot_confusion_matrix(cm, cm_path)

            os.makedirs("./results/roc", exist_ok=True)
            roc_path = os.path.join("./results/roc", f"{self.name}_roc_curve_recon_loss.png")
            self.plot_roc_curve(fpr, tpr, auc_score, roc_path)

            os.makedirs("./results/prc", exist_ok=True)
            prc_path = os.path.join("./results/prc", f"{self.name}_prc_curve_recon_loss.png")
            self.plot_precision_recall_curve(precision, recall, prc_auc_score, prc_path)

        return f1, auc_score, prc_auc_score