"""Piece classifier for ``clean_masks``: train on labels, predict P(channel).

Labels come from the labelling bundle (``rivmapper label``) as
``{river}/labels/{river}_labels.csv``: one row per labelled piece with
``river, ds_order, year, row, col, label`` (``channel`` / ``not_channel`` /
``unsure``; unsure rows are ignored). ``row, col`` is a pixel inside the
piece on the reach grid, so labels are matched to pieces by location and
survive renumbering when the piece code changes.

Model: gradient-boosted trees (scikit-learn ``HistGradientBoostingClassifier``).
Missing values (the persistence of a piece next to a no-data year, stage
without discharge) are coded −1. Features are the piece measurements in ``units.FEATURES``
plus context (stage percentile, braided reach, log channel width in pixels,
sparse/inferred year). Accuracy is estimated by grouped cross-validation:
whole reaches are held out, so the score reflects reaches the model has not
seen. The saved model file holds the fitted model, the feature list, the
cross-validated metrics and the label counts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import units

log = logging.getLogger(__name__)

LABELS = {"channel": 1, "not_channel": 0}
CONTEXT_FEATURES = ("q_scene_pctile", "braided", "log_w_px", "sparse_or_inferred")
FEATURES = tuple(units.FEATURES) + CONTEXT_FEATURES
MODEL_VERSION = 1
MISSING = -1.0


def design(table: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix (NaN allowed) from a piece table."""
    X = pd.DataFrame(index=table.index)
    for f in units.FEATURES:
        X[f] = pd.to_numeric(table.get(f), errors="coerce")
    X["q_scene_pctile"] = pd.to_numeric(table.get("q_scene_pctile"), errors="coerce")
    X["braided"] = table.get("braided", False).astype(str).str.lower().isin(["true", "1"]).astype(float)
    X["log_w_px"] = np.log(pd.to_numeric(table.get("w_px"), errors="coerce"))
    X["sparse_or_inferred"] = table.get("status", "").astype(str).isin(["sparse", "inferred"]).astype(float)
    # missing (no neighbouring year, no discharge) -> -1: all features are
    # >= 0, so the trees can still split on "missing"
    return X[list(FEATURES)].fillna(MISSING)


def match_labels(labels: pd.DataFrame, pieces: pd.DataFrame, lab_arrays) -> pd.DataFrame:
    """Attach labels to piece rows by location.

    ``lab_arrays``: {(ds_order, year): piece-id array} for the labelled
    reach-years. Labels whose pixel falls in no piece (the piece code
    changed so much that it is now corridor or gone) are dropped and counted.
    """
    out = []
    lost = 0
    for (o, y), g in labels.groupby(["ds_order", "year"]):
        lab = lab_arrays.get((int(o), int(y)))
        if lab is None:
            lost += len(g)
            continue
        ids = lab[g["row"].astype(int).clip(0, lab.shape[0] - 1),
                  g["col"].astype(int).clip(0, lab.shape[1] - 1)]
        for (_, r), pid in zip(g.iterrows(), ids, strict=True):
            if pid == 0:
                lost += 1
                continue
            out.append({"ds_order": int(o), "year": int(y), "piece": int(pid), "label": r["label"]})
    if lost:
        log.warning("  %d label(s) no longer fall in a piece; ignored", lost)
    m = pd.DataFrame(out)
    if m.empty:
        return m
    # one label per piece (last one wins)
    m = m.drop_duplicates(["ds_order", "year", "piece"], keep="last")
    return pieces.merge(m, on=["ds_order", "year", "piece"], how="inner")


def train(data: pd.DataFrame, out: Path, seed: int = 0, min_labels: int = 20) -> dict:
    """Fit on labelled pieces (``label`` column), cross-validate, save.

    Returns the metrics dict (also written next to the model as JSON).
    """
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    d = data[data["label"].isin(LABELS)].copy()
    y = d["label"].map(LABELS).to_numpy()
    X = design(d)
    groups = (d["river"].astype(str) + ":" + d["ds_order"].astype(str)).to_numpy()
    n_groups = len(set(groups))
    if len(d) < min_labels or len(set(y)) < 2:
        raise ValueError(
            f"need at least {min_labels} labels of both classes (have {len(d)}, classes {sorted(set(y))})"
        )

    def model():
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, class_weight="balanced", random_state=seed,
        )

    k = min(5, n_groups)
    p_cv = np.full(len(d), np.nan)
    if k >= 2:
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            m = model().fit(X.iloc[tr], y[tr])
            p_cv[te] = m.predict_proba(X.iloc[te])[:, 1]
    final = model().fit(X, y)
    ok = np.isfinite(p_cv)
    pred = (p_cv[ok] >= 0.5).astype(int)
    yt = y[ok]
    tn = int(((pred == 0) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    tp = int(((pred == 1) & (yt == 1)).sum())
    metrics = {
        "model_version": MODEL_VERSION,
        "n_labels": int(len(d)),
        "n_channel": int(y.sum()),
        "n_not_channel": int((y == 0).sum()),
        "rivers": sorted(d["river"].astype(str).unique().tolist()),
        "cv": f"GroupKFold by reach, k={k}" if k >= 2 else "none (one reach)",
        "cv_accuracy": float(accuracy_score(yt, pred)) if ok.any() else None,
        "cv_auc": float(roc_auc_score(yt, p_cv[ok])) if ok.any() and len(set(yt)) == 2 else None,
        "cv_confusion": {"true_channel_kept": tp, "true_channel_removed": fn,
                         "not_channel_removed": tn, "not_channel_kept": fp},
        "features": list(FEATURES),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": list(FEATURES), "metrics": metrics}, out)
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load(path: Path) -> dict:
    import joblib

    bundle = joblib.load(path)
    if list(bundle.get("features", [])) != list(FEATURES):
        raise ValueError(
            f"{path}: model was trained on different features; retrain with this version"
        )
    return bundle


def predict(bundle: dict, table: pd.DataFrame) -> np.ndarray:
    """P(channel) for each piece row."""
    return bundle["model"].predict_proba(design(table))[:, 1]
