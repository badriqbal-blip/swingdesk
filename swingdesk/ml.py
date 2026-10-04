"""Optional ML layer: walk-forward gradient boosting that predicts whether a name lands in the top 30% of the
cross-section over the next `horizon` bars. Training is purged/embargoed (no label overlaps the test date),
features are cross-sectional ranks (stationary across time), and only point-in-time-safe technical inputs are
used — fundamentals/news snapshots are excluded because we don't have their history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ML_FEATURES = ["g_trend", "g_momentum", "g_timing", "g_volatility", "g_volume", "rsi14", "dist21_atr", "atr_pct",
               "bb_bw_pct", "pct_52w_high", "mom_1m", "mom_3m", "rv20", "adx", "updown_vol", "cmf20", "clenow",
               "close_range3", "rvol", "squeeze_on"]


def _model(seed: int = 0):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for --ml  (pip install scikit-learn)") from e
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
                                          min_samples_leaf=40, random_state=seed)


def feature_frame(scored: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in ML_FEATURES if c in scored.columns]
    X = scored[feats].groupby(level=0).rank(pct=True)
    X["regime_code"] = scored["regime"].map({"CRISIS": 0, "RISK_OFF": 1, "NEUTRAL": 2, "RISK_ON": 3}).astype(float)
    return X


def walk_forward_ml(scored: pd.DataFrame, horizon: int, step: int = 20, min_train: int = 250, every: int = 5,
                    top_q: float = 0.7) -> dict:
    X = feature_frame(scored)
    rank = scored["fwd_ret"].groupby(level=0).rank(pct=True)
    y = pd.Series(np.where(rank.isna(), np.nan, (rank > top_q).astype(float)), index=rank.index)
    dates = X.index.get_level_values(0).unique().sort_values()
    train_dates = set(dates[::every])
    d_all = X.index.get_level_values(0)
    base_ok = X.notna().all(axis=1).to_numpy()

    preds = []
    for j in range(min_train, len(dates), step):
        test_date = dates[j]
        cutoff = dates[j - horizon - 1]
        tr = base_ok & y.notna().to_numpy() & (d_all <= cutoff) & d_all.isin(train_dates)
        if tr.sum() < 500:
            continue
        mdl = _model()
        mdl.fit(X[tr].to_numpy(), y[tr].to_numpy())
        te_mask = base_ok & (d_all == test_date)
        if te_mask.sum() == 0:
            continue
        p = mdl.predict_proba(X[te_mask].to_numpy())[:, 1]
        preds.append(pd.DataFrame({"prob": p}, index=X.index[te_mask]))

    metrics: dict = {"n_refits": len(preds)}
    if preds:
        P = pd.concat(preds).join(y.rename("label")).join(scored["fwd_ret"]).dropna()
        if len(P) > 100 and P["label"].nunique() == 2:
            from sklearn.metrics import roc_auc_score  # type: ignore

            metrics["auc"] = float(roc_auc_score(P["label"], P["prob"]))
            metrics["base_rate"] = float(P["label"].mean())
            top = P[P["prob"] >= P["prob"].quantile(0.9)]
            metrics["top_decile_precision"] = float(top["label"].mean())
            metrics["top_decile_fwd_ret_pct"] = 100 * float(top["fwd_ret"].mean())
            metrics["all_fwd_ret_pct"] = 100 * float(P["fwd_ret"].mean())
            metrics["n_test_rows"] = int(len(P))

    # final model on everything labelled → probability for today's cross-section
    latest_prob = pd.Series(dtype=float)
    fit_mask = base_ok & y.notna().to_numpy() & d_all.isin(train_dates)
    if fit_mask.sum() >= 500:
        mdl = _model()
        mdl.fit(X[fit_mask].to_numpy(), y[fit_mask].to_numpy())
        last_date = dates[-1]
        te = base_ok & (d_all == last_date)
        if te.sum():
            probs = mdl.predict_proba(X[te].to_numpy())[:, 1]
            latest_prob = pd.Series(probs, index=X.index[te].get_level_values(1))
    return {"metrics": metrics, "latest_prob": latest_prob}
