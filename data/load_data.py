import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(test_size=0.2, random_state=42):
    wine = load_wine()

    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Series(wine.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Classes: {wine.target_names}")
    print(f"Features: {list(wine.feature_names)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, wine.target_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, target_names = load_and_preprocess()
    print("\nSample of training data:")
    print(X_train.head())
