import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_wine


def build_pipelines():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=500, random_state=42)),
            ]
        ),
        "k-NN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        ),
        "Neural Network": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=500,
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }


def run_cv(n_splits=5):
    wine = load_wine()
    X, y = wine.data, wine.target

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipelines = build_pipelines()

    print(f"Cross-validation ({n_splits}-fold) on Wine dataset\n")
    print(f"{'Model':<25} {'Mean Acc':>10} {'Std':>8}")
    print("-" * 46)

    cv_results = {}
    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}
        print(f"{name:<25} {scores.mean():>10.4f} {scores.std():>8.4f}")

    return cv_results


if __name__ == "__main__":
    run_cv()
