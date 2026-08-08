"""
train_model.py
----------------
Loads the dataset, trains Logistic Regression, Decision Tree, Random
Forest and SVM classifiers, compares their performance, and saves the
best model (by accuracy) to model/model.pkl for use by app.py.

Run: python3 train_model.py
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

from features import FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "phishing_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        results[name] = {"accuracy": round(acc * 100, 2), "precision": round(prec * 100, 2), "recall": round(rec * 100, 2)}
        trained[name] = model
        print(f"{name:22s} | Accuracy: {acc*100:5.2f}%  Precision: {prec*100:5.2f}%  Recall: {rec*100:5.2f}%")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = trained[best_name]
    print(f"\nBest model: {best_name} ({results[best_name]['accuracy']}% accuracy) -- saving as model/model.pkl")

    joblib.dump({"model": best_model, "name": best_name, "features": FEATURE_NAMES}, os.path.join(MODEL_DIR, "model.pkl"))

    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Comparison table saved to model/results.json")


if __name__ == "__main__":
    main()
