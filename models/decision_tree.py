from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def train(X_train, y_train, max_depth=None, criterion="gini", random_state=42):
    model = DecisionTreeClassifier(
        max_depth=max_depth, criterion=criterion, random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, target_names=None):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=target_names)
    return acc, report, preds
