from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from backend.config import MODELS_DIR, load_yaml_config


META_PATH = MODELS_DIR / "model_meta.json"


def _feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "report_date", "symbol", "as_of_date", "signal_tier",
        "long_momentum_10d", "long_momentum_5d", "early_onset",
        "acc_1D_power", "dist_1D_power",
    }
    exclude.update(c for c in df.columns if c.startswith("forward_return_"))
    exclude.update(c for c in df.columns if c.startswith("max_drawdown_"))
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in ("float64", "float32", "int64", "int32", "bool"):
            cols.append(c)
    return cols


def train_models(features: pd.DataFrame, labels: pd.DataFrame) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml_config("settings.yaml")["ml"]

    df = features.merge(
        labels[["symbol", "as_of_date", "long_momentum_10d"]],
        left_on=["symbol", "report_date"],
        right_on=["symbol", "as_of_date"],
        how="left",
    )
    df = df.dropna(subset=["long_momentum_10d"])
    df["long_momentum_10d"] = df["long_momentum_10d"].astype(int)

    feature_cols = _feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for training")

    df = df.sort_values("report_date")
    X = df[feature_cols].fillna(0)
    y = df["long_momentum_10d"]

    split_idx = max(1, len(df) - cfg.get("test_size_days", 30))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=cfg["random_state"],
        class_weight="balanced",
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(lgb_model, cv=min(3, len(X_train)), method="isotonic")
    if len(X_train) >= 10:
        calibrated.fit(X_train, y_train)
        prob_model = calibrated
    else:
        prob_model = lgb_model

    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=cfg["random_state"],
        objective="reg:squarederror",
    )
    ret_col = "forward_return_10d"
    if ret_col in df.columns:
        y_reg = df[ret_col].fillna(0)
        xgb_model.fit(X.iloc[:split_idx], y_reg.iloc[:split_idx])
    else:
        xgb_model.fit(X_train, y_train.astype(float))

    if_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=cfg["random_state"])
    if_model.fit(X_train)

    metrics = {}
    if len(X_test) > 0:
        proba = prob_model.predict_proba(X_test)[:, 1]
        try:
            metrics["auc"] = float(roc_auc_score(y_test, proba))
        except ValueError:
            metrics["auc"] = None
        metrics["brier"] = float(brier_score_loss(y_test, proba))

    # Walk-forward cross-validation metrics
    tscv = TimeSeriesSplit(n_splits=min(3, max(2, len(X) // 20)))
    wf_aucs = []
    for train_idx, val_idx in tscv.split(X):
        if len(val_idx) < 5:
            continue
        fold_clf = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=6,
            random_state=cfg["random_state"], class_weight="balanced", verbose=-1,
        )
        fold_clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        try:
            wf_proba = fold_clf.predict_proba(X.iloc[val_idx])[:, 1]
            wf_aucs.append(float(roc_auc_score(y.iloc[val_idx], wf_proba)))
        except ValueError:
            pass
    if wf_aucs:
        metrics["walk_forward_auc_mean"] = float(np.mean(wf_aucs))

    joblib.dump(prob_model, MODELS_DIR / "classifier.joblib")
    joblib.dump(xgb_model, MODELS_DIR / "regressor.joblib")
    joblib.dump(if_model, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")

    meta = {"feature_cols": feature_cols, "metrics": metrics, "trained_rows": len(df)}
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def predict(features: pd.DataFrame) -> pd.DataFrame:
    clf_path = MODELS_DIR / "classifier.joblib"
    reg_path = MODELS_DIR / "regressor.joblib"
    cols_path = MODELS_DIR / "feature_cols.joblib"

    if not clf_path.exists():
        return _heuristic_predictions(features)

    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path) if reg_path.exists() else None
    feature_cols = joblib.load(cols_path)

    df = features.copy()
    X = df.reindex(columns=feature_cols, fill_value=0).fillna(0)

    if hasattr(clf, "predict_proba"):
        df["p_long_momentum"] = clf.predict_proba(X)[:, 1]
    else:
        df["p_long_momentum"] = clf.predict(X)

    if reg is not None:
        df["expected_return_10d"] = reg.predict(X)
    else:
        df["expected_return_10d"] = df["p_long_momentum"] * 8

    if_path = MODELS_DIR / "isolation_forest.joblib"
    if if_path.exists():
        if_model = joblib.load(if_path)
        scores = if_model.decision_function(X)
        df["anomaly_score"] = scores
        df["anomaly_flag"] = if_model.predict(X) == -1

    df["confidence"] = pd.cut(
        df["p_long_momentum"],
        bins=[0, 0.4, 0.65, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    return df


def _heuristic_predictions(features: pd.DataFrame) -> pd.DataFrame:
    """Fallback when no trained model exists."""
    df = features.copy()
    ems = df.get("early_momentum_score", pd.Series(0, index=df.index)).fillna(0)
    sms = df.get("smart_money_score", pd.Series(0, index=df.index)).fillna(0)
    drs = df.get("distribution_risk_score", pd.Series(0, index=df.index)).fillna(0)
    # Distribution-only fallback: use inverse distribution risk + float z-score
    if "acc_horizon_score" not in df.columns or df["acc_horizon_score"].fillna(0).sum() == 0:
        ft_z = df.get("float_turnover_zscore", pd.Series(0, index=df.index)).fillna(0).clip(0, 3) / 3
        df["p_long_momentum"] = ((1 - drs / 100) * 0.3 + ft_z * 0.2).clip(0, 0.5)
    else:
        df["p_long_momentum"] = ((ems * 0.6 + sms * 0.4) / 100 * (1 - drs / 200)).clip(0, 1)
    df["expected_return_10d"] = df["p_long_momentum"] * 10
    df["confidence"] = pd.cut(
        df["p_long_momentum"],
        bins=[0, 0.4, 0.65, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )
    return df


def compute_shap_values(features: pd.DataFrame, symbol: str) -> dict[str, float]:
    clf_path = MODELS_DIR / "classifier.joblib"
    cols_path = MODELS_DIR / "feature_cols.joblib"
    if not clf_path.exists() or not cols_path.exists():
        return {}

    try:
        import shap
    except ImportError:
        return {}

    row = features[features["symbol"] == symbol].tail(1)
    if row.empty:
        return {}

    clf = joblib.load(clf_path)
    feature_cols = joblib.load(cols_path)
    X = row.reindex(columns=feature_cols, fill_value=0).fillna(0)

    base = clf.calibrated_classifiers_[0].estimator if hasattr(clf, "calibrated_classifiers_") else clf
    explainer = shap.TreeExplainer(base)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    return dict(zip(feature_cols, map(float, sv[0])))
