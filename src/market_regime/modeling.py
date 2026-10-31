"""Chronological model training, evaluation, and prediction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

MODELS = {
    "Logistic Regression": lambda: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": lambda: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=350, min_samples_leaf=3,
             class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": lambda: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=.08,
             l2_regularization=1.0, random_state=42))
    ])
}

def prepare_dataset(features: pd.DataFrame, labels: pd.Series):
    data = features.join(labels.rename("target")).dropna(subset=["target"])
    X = data.drop(columns="target").replace([np.inf,-np.inf], np.nan)
    y = data["target"].astype(str)
    # Drop features that are entirely missing, retain missing values for train-only imputation.
    X = X.dropna(axis=1, how="all")
    return X, y

def chronological_split(X, y, test_size=0.25):
    if len(X) < 80:
        raise ValueError(f"Need at least 80 labeled rows; received {len(X)}.")
    split = int(len(X) * (1-test_size))
    if split < 40 or len(X)-split < 20:
        raise ValueError("Insufficient rows for chronological train/test split.")
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def train_and_evaluate(X, y, test_size=0.25):
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_size)
    if y_train.nunique() < 2:
        raise ValueError("Training period contains fewer than two regime classes. Increase lookback or change assets.")
    results, fitted = {}, {}
    labels_order = sorted(y.unique())
    for name, factory in MODELS.items():
        model = factory()
        if name == "Gradient Boosting":
            # Balanced sample weights are computed only from training labels.
            sample_weight = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, model__sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        p, r, f, _ = precision_recall_fscore_support(
            y_test, pred, average="weighted", zero_division=0)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_weighted": float(p),
            "recall_weighted": float(r),
            "f1_weighted": float(f),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=labels_order).tolist(),
            "labels": labels_order,
            "test_start": str(X_test.index.min()),
            "test_end": str(X_test.index.max()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test))
        }
        fitted[name] = model
    return results, fitted, X_train, X_test, y_train, y_test

def feature_importances(model, feature_names):
    estimator = model.named_steps.get("model")
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        return pd.DataFrame(columns=["Feature","Importance"])
    return pd.DataFrame({"Feature": list(feature_names), "Importance": values}).sort_values(
        "Importance", ascending=False).reset_index(drop=True)
