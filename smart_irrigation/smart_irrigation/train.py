"""Train the irrigation classifier, compare two candidates, register the best."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import registry
from generate_dataset import FEATURES

MODEL_NAME = "irrigation-classifier"


def main(data_path):
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df["irrigate"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
        "random_forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=7, n_jobs=-1),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = {
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
        }
        print(f"{name:20s}", results[name])

    best_name = max(results, key=lambda k: results[k]["f1"])
    best = candidates[best_name]
    print(f"\nBest model by F1: {best_name}")

    version = registry.register_model(
        best, MODEL_NAME,
        metrics=results[best_name],
        params={"algorithm": best_name, "train_rows": len(X_tr), "test_rows": len(X_te)},
        features=FEATURES,
    )
    registry.promote(MODEL_NAME, version, "Production")
    print(f"Registered {MODEL_NAME} v{version} -> Production")

    out = Path("screenshots")
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(best, X_te, y_te, display_labels=["No irrigation", "Irrigate"],
                                          ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/training_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/irrigation_data.csv")
    main(ap.parse_args().data)
