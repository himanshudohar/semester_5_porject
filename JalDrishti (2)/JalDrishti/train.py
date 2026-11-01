"""Train JalDrishti: baselines -> spatial cross-validation -> calibrated final model."""
import json
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, brier_score_loss)
from sklearn.inspection import permutation_importance

import config as C


def load_data():
    if C.REAL_DATA_PATH.exists():
        path, is_demo = C.REAL_DATA_PATH, False
    elif C.DEMO_DATA_PATH.exists():
        path, is_demo = C.DEMO_DATA_PATH, True
    else:
        sys.exit("No data found. Run: python generate_demo_data.py  (or add data/wells.csv)")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}" + ("  [SYNTHETIC DEMO DATA]" if is_demo else ""))
    df = df.drop_duplicates().dropna(subset=["lat", "lon", C.TARGET])
    missing = [c for c in C.FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"Missing required columns in data: {missing}")
    return df, is_demo


def spatial_groups(df):
    """Assign each well to a lat/lon block so whole areas are held out together."""
    gx = np.floor(df.lon / C.SPATIAL_BLOCK_DEG).astype(int)
    gy = np.floor(df.lat / C.SPATIAL_BLOCK_DEG).astype(int)
    return (gx.astype(str) + "_" + gy.astype(str)).values


def make_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, C.NUMERIC_FEATURES), ("cat", cat, C.CATEGORICAL_FEATURES)])


def candidates():
    s = C.RANDOM_STATE
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=s),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=s),
        "GradientBoosting": GradientBoostingClassifier(random_state=s),
    }


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
                brier=brier_score_loss(y, p), tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))


def main():
    C.MODEL_DIR.mkdir(exist_ok=True)
    C.REPORT_DIR.mkdir(exist_ok=True)
    df, is_demo = load_data()
    X, y, groups = df[C.FEATURES], df[C.TARGET].astype(int), spatial_groups(df)
    print("Class balance:", y.value_counts(normalize=True).round(2).to_dict())
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        sys.exit("Need wells in at least 2 spatial blocks for geographic validation.")

    # ---- Spatial (geographic) cross-validation of every candidate ----
    gkf = GroupKFold(n_splits=n_splits)
    results = {}
    for name, clf in candidates().items():
        oof = np.zeros(len(df))
        for tr, te in gkf.split(X, y, groups):
            pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        results[name] = metrics(y, oof)
        print(f"{name:20s} AUC={results[name]['roc_auc']:.3f}  "
              f"P={results[name]['precision']:.3f}  R={results[name]['recall']:.3f}  F1={results[name]['f1']:.3f}")

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print("Best by spatial-CV ROC-AUC:", best)

    # ---- Final model: calibrated probabilities (train + held-out spatial test) ----
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(C.RANDOM_STATE)
    test_blocks = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False))
    is_test = np.array([g in test_blocks for g in groups])
    Xtr, ytr, Xte, yte = X[~is_test], y[~is_test], X[is_test], y[is_test]

    base = Pipeline([("prep", make_preprocessor()), ("clf", candidates()[best])])
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(Xtr, ytr)
    p_te = calibrated.predict_proba(Xte)[:, 1]
    test_metrics = metrics(yte, p_te)
    print("Held-out REGION test:", {k: round(v, 3) for k, v in test_metrics.items()})

    # Refit on all data for deployment
    final = CalibratedClassifierCV(Pipeline([("prep", make_preprocessor()), ("clf", candidates()[best])]),
                                   method="isotonic", cv=3)
    final.fit(X, y)

    # ---- Explainability: permutation importance on held-out region ----
    imp = permutation_importance(calibrated, Xte, yte, scoring="roc_auc", n_repeats=10,
                                 random_state=C.RANDOM_STATE)
    imp_df = pd.DataFrame({"feature": C.FEATURES, "importance": imp.importances_mean}).sort_values(
        "importance", ascending=False)
    imp_df.to_csv(C.REPORT_DIR / "feature_importance.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.barh(imp_df.feature[::-1], imp_df.importance[::-1])
    plt.title("Permutation importance (held-out region)")
    plt.tight_layout()
    plt.savefig(C.REPORT_DIR / "feature_importance.png", dpi=130)
    plt.close()

    # ---- Save ----
    joblib.dump({"model": final, "features": C.FEATURES, "importance": imp_df.to_dict("records"),
                 "trained_on_demo_data": is_demo, "best_model": best,
                 "feature_medians": X[C.NUMERIC_FEATURES].median().to_dict(),
                 "feature_modes": X[C.CATEGORICAL_FEATURES].mode().iloc[0].to_dict()}, C.MODEL_PATH)
    with open(C.METRICS_PATH, "w") as f:
        json.dump({"spatial_cv": results, "best_model": best, "heldout_region_test": test_metrics,
                   "trained_on_demo_data": is_demo}, f, indent=2)
    print(f"Saved model -> {C.MODEL_PATH}\nSaved reports -> {C.REPORT_DIR}")
    if is_demo:
        print("\nWARNING: trained on SYNTHETIC data. Do not use for real drilling decisions.")


if __name__ == "__main__":
    main()
