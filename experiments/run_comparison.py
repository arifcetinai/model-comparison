import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data.load_data import load_and_preprocess
from models import logistic_regression, knn, decision_tree, neural_network


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_confusion_matrix(y_test, preds, model_name, target_names):
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close()


def plot_accuracy_bar(results):
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison — Wine Dataset")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_comparison.png"), dpi=120)
    plt.close()


def main():
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess()

    summary = {}

    # --- Logistic Regression ---
    print("\n[1/4] Training Logistic Regression...")
    lr_model = logistic_regression.train(X_train, y_train)
    lr_acc, lr_report, lr_preds = logistic_regression.evaluate(
        lr_model, X_test, y_test, target_names
    )
    print(f"  Accuracy: {lr_acc:.4f}")
    plot_confusion_matrix(y_test, lr_preds, "Logistic Regression", target_names)
    summary["Logistic Regression"] = {"accuracy": round(lr_acc, 4)}

    # --- k-NN ---
    print("\n[2/4] Training k-NN (tuning k)...")
    best_k, k_scores = knn.tune_k(X_train, y_train, X_test, y_test)
    print(f"  Best k: {best_k}")
    knn_model = knn.train(X_train, y_train, n_neighbors=best_k)
    knn_acc, knn_report, knn_preds = knn.evaluate(
        knn_model, X_test, y_test, target_names
    )
    print(f"  Accuracy: {knn_acc:.4f}")
    plot_confusion_matrix(y_test, knn_preds, "k-NN", target_names)
    summary["k-NN"] = {"accuracy": round(knn_acc, 4), "best_k": best_k}

    # --- Decision Tree ---
    print("\n[3/4] Training Decision Tree...")
    dt_model = decision_tree.train(X_train, y_train, max_depth=5)
    dt_acc, dt_report, dt_preds = decision_tree.evaluate(
        dt_model, X_test, y_test, target_names
    )
    print(f"  Accuracy: {dt_acc:.4f}")
    plot_confusion_matrix(y_test, dt_preds, "Decision Tree", target_names)
    summary["Decision Tree"] = {"accuracy": round(dt_acc, 4)}

    # --- Neural Network ---
    print("\n[4/4] Training Neural Network (MLP)...")
    nn_model = neural_network.train(X_train, y_train)
    nn_acc, nn_report, nn_preds = neural_network.evaluate(
        nn_model, X_test, y_test, target_names
    )
    print(f"  Accuracy: {nn_acc:.4f}")
    plot_confusion_matrix(y_test, nn_preds, "Neural Network", target_names)
    summary["Neural Network"] = {"accuracy": round(nn_acc, 4)}

    # Save summary
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_accuracy_bar(summary)

    print("\n=== Results ===")
    for model_name, metrics in summary.items():
        print(f"  {model_name:25s}: {metrics['accuracy']:.4f}")

    best = max(summary, key=lambda k: summary[k]["accuracy"])
    print(f"\nBest model: {best} ({summary[best]['accuracy']:.4f})")


if __name__ == "__main__":
    main()
