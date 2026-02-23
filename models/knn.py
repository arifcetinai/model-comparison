from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def train(X_train, y_train, n_neighbors=5, metric="euclidean"):
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, target_names=None):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=target_names)
    return acc, report, preds


def tune_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    """Try different k values and return accuracy for each."""
    results = {}
    for k in k_range:
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        results[k] = acc
    best_k = max(results, key=results.get)
    return best_k, results
