import csv
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from itertools import product
from main_vgae import pretrain_model_on_enzymes, finetune_on_mutag_anomaly, classification_task
from load_data import load_data
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def run_experiments(name):
    """
    Run grid search experiments over hyperparameters for VGAE pretraining,
    fine-tuning on MUTAG anomaly detection, and classification tasks.
    Iterates over combinations of learning rates, batch sizes, hidden channel
    sizes, latent dimensions, and dropout rates. For each configuration:
      1. Pretrain on ENZYMES dataset.
      2. Fine-tune and evaluate on MUTAG for anomaly detection with and without pretraining.
      3. Perform downstream classification with and without pretraining.
    Results are saved to a CSV file under ./results.
    Args:
        name (str): Base experiment name, used for naming saved models and results.
    """
    params = {
        "lr": [1e-4, 1e-3, 1e-2],
        "batch_size": [8, 16],
        "hidden_channels": [64],
        "latent_dim": [16, 32, 64],
        "dropout": [0.1, 0.3, 0.5],
    }
    random.seed(42)

    results = []
    mutag_dataset = load_data("MUTAG", use_node_attr=True)
    param_combinations = list(product(*params.values()))

    # Najpierw robimy podział dla detekcji anomalii - train to 70% klasy 0, val 50% pozostałych danych i test to 50% pozostałych danych
    # Przy trenowaniu finalnego modelu po hyperparameter tuning trenujemy na train+val z usuniętymi przykładami z klasy 1 i testujemy na test
    normal = [d for d in mutag_dataset if d.y.item() == 0]
    anomalies = [d for d in mutag_dataset if d.y.item() == 1]

    random.shuffle(normal)

    train_set = normal[: int(0.7 * len(normal))]
    test_set = normal[int(0.7 * len(normal)):] + anomalies

    random.shuffle(test_set)

    val_set = test_set[: int(0.5 * len(test_set))]
    test_set = test_set[int(0.5 * len(test_set)):]

    train_and_val_set = train_set + val_set

    # Podział dla zadania klasyfikacji - 35% całego zbioru to train, 35% to val i 30% to test
    indices = np.arange(len(mutag_dataset))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=[d.y.item() for d in mutag_dataset])

    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.5, random_state=42, stratify=[mutag_dataset[i].y.item() for i in train_idx]
    )

    train_set_classif = [mutag_dataset[i] for i in train_idx]
    val_set_classif = [mutag_dataset[i] for i in val_idx]
    test_set_classif = [mutag_dataset[i] for i in test_idx]

    train_and_val_set_classif = train_set_classif + val_set_classif

    for lr, batch_size, hidden_channels, latent_dim, dropout in tqdm(param_combinations, desc="Eksperymenty"):
        print(
            f"Running with lr={lr}, batch_size={batch_size}, hidden={hidden_channels}, latent={latent_dim}, dropout={dropout}"
        )
        name = f"VGAE_lr_{lr}_batch_{batch_size}_hidden_{hidden_channels}_latent_{latent_dim}_dropout_{dropout}"

        model = pretrain_model_on_enzymes(
            hidden_channels, latent_dim, dropout, lr, batch_size, epochs=100, name=name
        )

        recon_f1, auc_score_recon, prc_auc_score_recon, gmm_f1, auc_score_gmm, prc_auc_score_gmm = finetune_on_mutag_anomaly(
            model,
            X_train=train_set,
            X_test=val_set,
            batch_size=batch_size,
            lr=lr,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            dropout=dropout,
            pretrained=True,
            name=name
        )

        recon_f1_no_pre, auc_score_recon_no_pre, prc_auc_score_recon_no_pre, gmm_f1_no_pre, auc_score_gmm_no_pre, prc_auc_score_gmm_no_pre = finetune_on_mutag_anomaly(
            model=None,
            X_train=train_set,
            X_test=val_set,
            batch_size=batch_size,
            lr=lr,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            dropout=dropout,
            pretrained=False,
            name=name
        )

        classif_f1 = classification_task(
            X_train=train_set_classif,
            X_val=val_set_classif,
            pretrained_model=model,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            pretrained=True,
            name=name
        )

        classif_f1_no_pre = classification_task(
            X_train=train_set_classif,
            X_val=val_set_classif,
            pretrained_model=None,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            pretrained=False,
            name=name
        )

        result = {
            "lr": lr,
            "batch_size": batch_size,
            "hidden_channels": hidden_channels,
            "latent_dim": latent_dim,
            "dropout": dropout,
            "recon_f1": recon_f1,
            "gmm_f1": gmm_f1,
            "recon_auc_score": auc_score_recon,
            "gmm_auc_score": auc_score_gmm,
            "recon_prc_auc_score": prc_auc_score_recon,
            "gmm_prc_auc_score": prc_auc_score_gmm,
            "recon_f1_no_pre": recon_f1_no_pre,
            "gmm_f1_no_pre": gmm_f1_no_pre,
            "recon_auc_score_no_pre": auc_score_recon_no_pre,
            "gmm_auc_score_no_pre": auc_score_gmm_no_pre,
            "recon_prc_auc_score_no_pre": prc_auc_score_recon_no_pre,
            "gmm_prc_auc_score_no_pre": prc_auc_score_gmm_no_pre,
            "classif_f1_pre": classif_f1,
            "classif_f1_no_pre": classif_f1_no_pre,
        }
        results.append(result)

    os.makedirs("./results", exist_ok=True)
    csv_file = f"./results/results_VGAE.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Wyniki zapisane do {csv_file}")

    metrics = [
        "gmm_auc_score", "gmm_auc_score_no_pre",
        "recon_auc_score", "recon_auc_score_no_pre",
        "classif_f1_pre", "classif_f1_no_pre"
    ]
    df = pd.read_csv(csv_file)

    results_best = []
    for metric in metrics:
        best = df.loc[df[metric].idxmax()]
        lr = best["lr"]
        batch_size = int(best["batch_size"])
        hidden_channels = int(best["hidden_channels"])
        latent_dim = int(best["latent_dim"])
        dropout = float(best["dropout"])

        # Build name
        name = f"BEST_VGAE_{metric}_lr_{lr}_batch_{batch_size}_hidden_{hidden_channels}_latent_{latent_dim}_dropout_{dropout}"
        print(f"Running best model for {metric}: {name}")

        # Pretrain if needed
        pretrained_model = None
        if "_no_pre" not in metric:
            pretrained_model = pretrain_model_on_enzymes(
                hidden_channels, latent_dim, dropout, lr, batch_size,
                epochs=100, name=name
            )

        if metric.startswith("gmm_") or metric.startswith("recon_"):
            classif_f1 = None
            recon_f1, auc_score_recon, prc_auc_score_recon, gmm_f1, auc_score_gmm, prc_auc_score_gmm = finetune_on_mutag_anomaly(
                model=pretrained_model,
                X_train=train_and_val_set,
                X_test=test_set,
                batch_size=batch_size,
                lr=lr,
                hidden_channels=hidden_channels,
                latent_dim=latent_dim,
                dropout=dropout,
                pretrained=("_no_pre" not in metric),
                name=name
            )
        else:
            recon_f1, auc_score_recon, prc_auc_score_recon = None, None, None
            gmm_f1, auc_score_gmm, prc_auc_score_gmm = None, None, None
            classif_f1 = classification_task(
                X_train=train_and_val_set_classif,
                X_val=test_set_classif,
                pretrained_model=pretrained_model,
                hidden_channels=hidden_channels,
                latent_dim=latent_dim,
                dropout=dropout,
                lr=lr,
                batch_size=batch_size,
                pretrained=("_no_pre" not in metric),
                name=name
            )
        result = {
            "metric": metric,
            "recon_f1": recon_f1,
            "gmm_f1": gmm_f1,
            "recon_auc_score": auc_score_recon,
            "gmm_auc_score": auc_score_gmm,
            "recon_prc_auc_score": prc_auc_score_recon,
            "gmm_prc_auc_score": prc_auc_score_gmm,
            "classif_f1": classif_f1,
            "lr": lr,
            "batch_size": batch_size,
            "hidden_channels": hidden_channels,
            "latent_dim": latent_dim,
            "dropout": dropout,
        }
        results_best.append(result)

    best_csv_file = f"./results/best_results_VGAE.csv"
    with open(best_csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results_best[0].keys())
        writer.writeheader()
        writer.writerows(results_best)
    print(f"Najlepsze wyniki zapisane do {best_csv_file}")


if __name__ == "__main__":
    run_experiments("VGAE")
    print("Eksperymenty zakończone.")
