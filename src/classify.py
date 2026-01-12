from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")


def classify_embeddings(
    X_train, X_test, y_train, y_test, random_state=42, name="logistic_regression"
):
    """
    Classify embeddings using Logistic Regression and plot confusion matrix.
    Args:
        X_train (np.ndarray): Training embeddings.
        X_test (np.ndarray): Test embeddings.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Test labels.
        random_state (int): Random state for reproducibility.
        name (str): Name for saving the confusion matrix plot.
    Returns:
        float: F1 score of the classification.
    """
    os.makedirs("./results/cm", exist_ok=True)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.xlabel("Wartość przewidziana")
    plt.ylabel("Wartość rzeczywista")
    plt.title(f"Macierz pomyłek - (regresja logistyczna)")
    plt.tight_layout()
    plt.savefig(f"./results/cm/{name}_confusion_matrix_classifier.png")
    plt.close()
    return f1
