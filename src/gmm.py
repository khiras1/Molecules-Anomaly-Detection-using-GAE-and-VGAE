import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


class EmbeddingAnomalyEvaluator:
    """
    Evaluator for anomaly detection using Gaussian Mixture Models on embedding data.
    Attributes:
        X_train (np.ndarray): Embeddings of normal training samples.
        X_test (np.ndarray): Embeddings of test samples.
        labels_test (np.ndarray): True binary labels for test data (0=normal, 1=anomaly).
        name (str): Identifier used for saving result files.
    """

    def __init__(
        self,
        X_train,
        X_test,
        labels_test,
        name="model_eval",
    ):
        """
        Initialize the evaluator with embeddings and labels.
        Args:
            X_train (np.ndarray): Embeddings from normal training data.
            X_test (np.ndarray): Embeddings for test data.
            labels_test (list): True labels (0 = normal, 1 = anomaly).
            name (str): Identifier for saving results.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.labels_test = np.array(labels_test)
        self.name = name
        os.makedirs("./results", exist_ok=True)

    def _compute_threshold(self, scores):
        """
        Compute anomaly detection threshold based on the interquartile range (IQR) of training scores.
        Uses the formula: threshold = Q3 + 1.5 * IQR.
        Args:
            scores (array-like): Array of anomaly scores from the training set.
        Returns:
            float: Calculated threshold value above which samples are considered anomalies.
        """
        q1 = np.quantile(scores, 0.25)
        q3 = np.quantile(scores, 0.75)
        iqr = q3 - q1
        return q3 + 1.5 * iqr

    def _save_confusion_matrix(self, cm, model_name):
        """
        Save the confusion matrix as a heatmap image.
        Args:
            cm (np.ndarray): Confusion matrix array (2x2).
            model_name (str): Name of the model to include in the title and filename.
        """
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normalne", "Anomalie"],
            yticklabels=["Normalne", "Anomalie"],
        )
        plt.xlabel("Wartość przewidziana")
        plt.ylabel("Wartość rzeczywista")
        plt.title(f"Macierz pomyłek - {model_name}")
        plt.tight_layout()
        os.makedirs("results/cm", exist_ok=True)
        plt.savefig(f"results/cm/{self.name}_confusion_matrix.png")
        plt.close()

    def _save_roc_curve(self, fpr, tpr, auc_score):
        """
        Plot and save the Receiver Operating Characteristic (ROC) curve.
        Args:
            fpr (array-like): False Positive Rates at various thresholds.
            tpr (array-like): True Positive Rates at various thresholds.
            auc_score (float): Area Under the Curve (AUC) value.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {auc_score:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Wskaźnik False Positive")
        plt.ylabel("Wskaźnik True Positive")
        plt.title(f"Krzywa ROC - GMM")
        plt.legend(loc="lower right")
        plt.tight_layout()
        os.makedirs("results/roc", exist_ok=True)
        plt.savefig(f"results/roc/{self.name}_roc_curve.png")
        plt.close()

    def _save_precision_recall_curve(self, precision, recall, prc_auc):
        """
        Plot and save the Precision-Recall Curve.
        Args:
            precision (array-like): Precision values at different thresholds.
            recall (array-like): Recall values at different thresholds.
            prc_auc (float): Area under the Precision-Recall Curve.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(
            recall, precision, color="green", lw=2, label=f"PRC (AUC = {prc_auc:.2f})"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Krzywa Precision-Recall - GMM")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower left")
        plt.tight_layout()
        os.makedirs("results/prc", exist_ok=True)
        plt.savefig(f"results/prc/{self.name}_prc_curve.png")
        plt.close()

    def fit_gmm(self, n_components=3):
        """
        Fit a Gaussian Mixture Model on the training embeddings.
        Args:
            n_components (int): Number of mixture components (clusters).
        """
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.gmm.fit(self.X_train)

    def evaluate_gmm(self):
        """
        Evaluate the fitted GMM on test data and save performance plots.
        Steps:
            1. Compute negative log-likelihood scores for test embeddings.
            2. Determine anomaly threshold based on training scores.
            3. Generate binary predictions and compute the confusion matrix.
            4. Calculate F1 score, ROC AUC, and Precision-Recall AUC.
            5. Save confusion matrix, ROC curve, and Precision-Recall curve plots.
        Returns:
            tuple:
                f1 (float): F1 score of binary anomaly classification.
                roc_auc (float): Area Under the ROC Curve.
                prc_auc (float): Average Precision (AUC of Precision-Recall Curve).
        """
        scores = -self.gmm.score_samples(self.X_test)
        threshold = self._compute_threshold(-self.gmm.score_samples(self.X_train))
        preds = (scores > threshold).astype(int)

        cm = confusion_matrix(self.labels_test, preds)
        f1 = f1_score(self.labels_test, preds, average="binary", zero_division=0)
        self._save_confusion_matrix(cm, "GMM")

        fpr, tpr, _ = roc_curve(self.labels_test, scores)
        auc_score = auc(fpr, tpr)
        self._save_roc_curve(fpr, tpr, auc_score)

        precision, recall, _ = precision_recall_curve(self.labels_test, scores)
        prc_auc_score = average_precision_score(self.labels_test, scores)
        self._save_precision_recall_curve(precision, recall, prc_auc_score)

        return f1, auc_score, prc_auc_score
