"""WRMSSE (the M5 accuracy metric) over 12 hierarchy levels.

The same metric as notebook 06, kept in a module so the tuning script can reuse it.
Recipe per level: sum sales up to the level -> scale = mean squared day-to-day
change in training -> RMSSE = sqrt(MSE / scale) -> weight by last-28-day dollar
sales -> sum. Final score = average over the 12 levels.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SALES_PATH = ROOT / "data" / "processed" / "sales_long.parquet"

HORIZON = 28
TRAIN_END = 1913  # last training day; validation = d_1914 .. d_1941

LEVELS = {
    "Total": [],
    "State": ["state_id"],
    "Store": ["store_id"],
    "Category": ["cat_id"],
    "Department": ["dept_id"],
    "State-Category": ["state_id", "cat_id"],
    "State-Department": ["state_id", "dept_id"],
    "Store-Category": ["store_id", "cat_id"],
    "Store-Department": ["store_id", "dept_id"],
    "Item": ["item_id"],
    "Item-State": ["item_id", "state_id"],
    "Item-Store": ["item_id", "store_id"],
}


def load_history():
    """Return (meta, S, P): series info, sales matrix and price matrix.

    S and P have one row per series (sorted by id) and one column per day d_1..d_1941.
    """
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "sales", "sell_price"]
    t = pq.read_table(SALES_PATH, columns=cols).to_pandas()
    for c in cols[:6]:
        t[c] = t[c].astype(str)
    t["d_num"] = t["d"].astype(str).str[2:].astype(int)
    meta = t.drop_duplicates("id")[cols[:6]].sort_values("id").reset_index(drop=True)
    row = t["id"].map(dict(zip(meta["id"], range(len(meta))))).to_numpy()
    col = t["d_num"].to_numpy() - 1
    n_days = col.max() + 1
    S = np.zeros((len(meta), n_days), dtype=np.float32)
    P = np.zeros((len(meta), n_days), dtype=np.float32)
    S[row, col] = t["sales"].to_numpy()
    P[row, col] = np.nan_to_num(t["sell_price"].to_numpy(dtype=np.float32))  # no price = 0
    return meta, S, P


class WRMSSE:
    # skip_leading_zeros=False reproduces the Stage 6 numbers exactly (0.7174 / 0.7036).
    # True follows the official M5 definition (ignore days before an item's first sale).
    def __init__(self, meta, S, P, skip_leading_zeros=False):
        train = S[:, :TRAIN_END]
        self.actual = S[:, TRAIN_END:TRAIN_END + HORIZON].astype(np.float64)
        # dollar sales over the last 28 training days (weights are fixed, model-independent)
        dollars = (S[:, TRAIN_END - HORIZON:TRAIN_END] * P[:, TRAIN_END - HORIZON:TRAIN_END]).sum(axis=1)
        self.levels = {}
        for name, cols in LEVELS.items():
            codes = (meta.groupby(cols).ngroup().to_numpy() if cols
                     else np.zeros(len(meta), dtype=int))
            n = codes.max() + 1
            agg_train = np.zeros((n, train.shape[1]))
            np.add.at(agg_train, codes, train)
            agg_dollars = np.bincount(codes, weights=dollars, minlength=n)
            agg_actual = np.zeros((n, HORIZON))
            np.add.at(agg_actual, codes, self.actual)
            scale = np.empty(n)
            for i in range(n):
                x = agg_train[i]
                if skip_leading_zeros:  # ignore days before the first sale
                    nz = np.flatnonzero(x)
                    x = x[nz[0]:] if len(nz) else x
                scale[i] = np.mean(np.diff(x) ** 2) if len(x) > 1 else np.nan
            self.levels[name] = (codes, n, scale, agg_dollars / agg_dollars.sum(), agg_actual)

    def score(self, pred):
        """pred: (n_series, 28) array in the same order as meta. Returns {level: score, 'WRMSSE': final}."""
        out = {}
        for name, (codes, n, scale, weight, agg_actual) in self.levels.items():
            agg_pred = np.zeros((n, HORIZON))
            np.add.at(agg_pred, codes, pred)
            mse = np.mean((agg_actual - agg_pred) ** 2, axis=1)
            rmsse = np.sqrt(mse / scale)
            out[name] = float(np.nansum(rmsse * weight))
        out["WRMSSE"] = float(np.mean(list(out.values())))
        return out


def preds_to_matrix(meta, df):
    """Turn a (id, date, pred) table into an (n_series, 28) matrix ordered like meta."""
    df = df.copy()
    df["id"] = df["id"].astype(str)
    wide = df.pivot(index="id", columns="date", values="pred").loc[meta["id"]]
    return wide.to_numpy()
