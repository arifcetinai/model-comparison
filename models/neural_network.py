from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def train(
    X_train,
    y_train,
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42,
):
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, target_names=None):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=target_names)
    return acc, report, preds
