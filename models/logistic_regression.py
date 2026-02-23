from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train(X_train, y_train, max_iter=500, C=1.0, random_state=42):
    model = LogisticRegression(max_iter=max_iter, C=C, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, target_names=None):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=target_names)
    return acc, report, preds
