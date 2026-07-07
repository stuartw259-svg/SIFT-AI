"""
train.py
Trains the SIFT classifier and writes every artifact the console needs.
Run once before launching the app (or let the Dockerfile do it at build time):

    python train.py
"""

import json
from pathlib import Path

import skops.io as sio

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from sift_core import FEATURES, RANDOM_STATE, synthesize_corpus

ART = Path(__file__).parent / "artifacts"
ART.mkdir(exist_ok=True)


def main():
    print("Synthesizing corpus...")
    df = synthesize_corpus()
    X, y = df[FEATURES], df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    print("Training tuned Gradient Boosting (params from capstone GridSearchCV)...")
    model = GradientBoostingClassifier(
        learning_rate=0.05, max_depth=2, n_estimators=200,
        subsample=0.8, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    pred = (p_test >= 0.5).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, p_test), 4),
        "n_test": int(len(y_test)),
    }
    print("Test metrics:", metrics)

    print("Computing permutation importance...")
    imp = permutation_importance(model, X_test, y_test, n_repeats=10,
                                 random_state=RANDOM_STATE, scoring="f1", n_jobs=-1)
    importance = (pd.DataFrame({"feature": FEATURES,
                                "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False))

    # Test-set frame the console's queue and policy tabs run on
    test_frame = X_test.copy()
    test_frame["p_malicious"] = p_test
    test_frame["label"] = y_test.values
    test_frame["archetype"] = df.loc[X_test.index, "archetype"].values

    benign_p90 = df[df.label == 0][FEATURES].quantile(0.90).to_dict()

    print("Writing artifacts...")
    # skops instead of pickle: serializes sklearn estimators without the
    # arbitrary-code-execution risk that pickle.load carries. Fitting for a
    # supply-chain security tool not to ship a deserialize-and-execute artifact.
    sio.dump(model, ART / "model.skops")
    test_frame.to_csv(ART / "test_scores.csv", index=False)
    importance.to_csv(ART / "feature_importance.csv", index=False)
    with open(ART / "benign_p90.json", "w") as f:
        json.dump(benign_p90, f, indent=2)
    with open(ART / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Done. Artifacts in {ART}/")


if __name__ == "__main__":
    main()
