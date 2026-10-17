"""Tune the Tweedie LightGBM model. Keep a change only if WRMSSE improves.

Validation is unchanged from Stage 5: train on days up to d_1913, validate on
d_1914..d_1941 (time-based, no random split). Every trial is logged to
outputs/tuning_log.csv and its predictions saved to outputs/trials/.

Usage (from project root):
  python modeling/tune_tweedie.py --run baseline
  python modeling/tune_tweedie.py --run singles    # one change at a time
  python modeling/tune_tweedie.py --run combo      # combine the singles that helped
"""
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wrmsse import ROOT, HORIZON, TRAIN_END, WRMSSE, load_history, preds_to_matrix

OUT = ROOT / "outputs"
TRIALS = OUT / "trials"
LOG = OUT / "tuning_log.csv"

CAT = ["item_id_enc", "dept_id_enc", "cat_id_enc", "store_id_enc", "state_id_enc",
       "event_name_1_enc", "event_type_1_enc", "event_name_2_enc", "event_type_2_enc"]
NUM = ["wday", "month", "year", "is_weekend", "snap_active", "has_event",
       "lag_28", "lag_35", "lag_42", "lag_49", "lag_56",
       "rolling_mean_7_lag28", "rolling_mean_28_lag28",
       "sell_price", "price_change_vs_last_week", "price_rel_to_own_avg"]
FEATURES = CAT + NUM

# Stage 5 Tweedie settings = the baseline to beat
BASE = {"objective": "tweedie", "tweedie_variance_power": 1.2, "metric": "rmse",
        "num_leaves": 31, "learning_rate": 0.1, "min_data_in_leaf": 50, "verbose": -1}

# one change at a time, each compared with BASE
SINGLES = {
    "vp_1.1": {"tweedie_variance_power": 1.1},
    "vp_1.3": {"tweedie_variance_power": 1.3},
    "lr_0.05": {"learning_rate": 0.05},
    "leaves_63": {"num_leaves": 63},
    "leaves_127_mdl100": {"num_leaves": 127, "min_data_in_leaf": 100},
    "subsample_0.8": {"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1},
    "l2_10": {"lambda_l2": 10},
}


def load_data():
    """Train/valid split by time, same cutoff as Stage 5."""
    df = pq.read_table(ROOT / "data" / "processed" / "features.parquet",
                       columns=["id", "date", "sales"] + FEATURES).to_pandas()
    cutoff = df["date"].max() - pd.Timedelta(days=HORIZON)
    train, valid = df[df["date"] <= cutoff], df[df["date"] > cutoff]
    assert train["date"].max() < valid["date"].min()  # no overlap in time
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def read_log():
    return pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=["trial"])


def run_trial(name, changes, train_set, valid_set, valid, meta, metric_proj, metric_off):
    """Train one config, score it, save predictions, append to the log."""
    if name in set(read_log()["trial"]):
        print(f"skip {name} (already in log)")
        return
    params = {**BASE, **changes}
    t0 = time.time()
    model = lgb.train(params, train_set, num_boost_round=2000, valid_sets=[valid_set],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    pred = model.predict(valid[FEATURES], num_iteration=model.best_iteration)
    out = valid[["id", "date"]].copy()
    out["pred"] = pred
    out.to_parquet(TRIALS / f"{name}.parquet")
    mat = preds_to_matrix(meta, out)
    row = {"trial": name, "changes": str(changes), "best_iter": model.best_iteration,
           "rmse": model.best_score["valid_0"]["rmse"],
           "wrmsse": metric_proj.score(mat)["WRMSSE"],              # Stage 6 definition
           "wrmsse_official_style": metric_off.score(mat)["WRMSSE"],  # skips leading zeros
           "minutes": round((time.time() - t0) / 60, 1)}
    pd.concat([read_log(), pd.DataFrame([row])]).to_csv(LOG, index=False)
    print(row, flush=True)


def seasonal_naive(meta, S, metric_proj, metric_off):
    """Seasonal-naive WRMSSE: repeat the last 7 training days over the 28 days."""
    last7 = S[:, TRAIN_END - 7:TRAIN_END]
    mat = np.tile(last7, 4)
    return {"model": "seasonal_naive", "wrmsse": metric_proj.score(mat)["WRMSSE"],
            "wrmsse_official_style": metric_off.score(mat)["WRMSSE"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["baseline", "singles", "combo"], required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    TRIALS.mkdir(exist_ok=True)

    meta, S, P = load_history()
    metric_proj = WRMSSE(meta, S, P, skip_leading_zeros=False)
    metric_off = WRMSSE(meta, S, P, skip_leading_zeros=True)

    if args.run == "baseline":  # also record the seasonal-naive reference
        pd.DataFrame([seasonal_naive(meta, S, metric_proj, metric_off)]).to_csv(
            OUT / "seasonal_naive_wrmsse.csv", index=False)
        trials = {"baseline": {}}
    elif args.run == "singles":
        trials = SINGLES
    else:  # combine every single change that beat the baseline
        log = read_log().set_index("trial")
        base = log.loc["baseline", "wrmsse"]
        good = [t for t in SINGLES if t in log.index and log.loc[t, "wrmsse"] < base]
        print("improving singles:", good)
        if not good:
            return
        merged = {}
        for t in good:
            merged.update(SINGLES[t])
        trials = {"combo": merged}

    train, valid = load_data()
    train_set = lgb.Dataset(train[FEATURES], label=train["sales"], categorical_feature=CAT,
                            free_raw_data=False)
    valid_set = lgb.Dataset(valid[FEATURES], label=valid["sales"], categorical_feature=CAT,
                            reference=train_set, free_raw_data=False)
    for name, changes in trials.items():
        run_trial(name, changes, train_set, valid_set, valid, meta, metric_proj, metric_off)


if __name__ == "__main__":
    main()
