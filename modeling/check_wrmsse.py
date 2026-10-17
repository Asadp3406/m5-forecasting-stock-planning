"""Check that wrmsse.py reproduces the Stage 6 scores (regression 0.7174, Tweedie 0.7036)."""
import sys
import pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from wrmsse import load_history, WRMSSE, preds_to_matrix, ROOT

meta, S, P = load_history()
for skip in (True, False):
    m = WRMSSE(meta, S, P, skip_leading_zeros=skip)
    for f, expected in [("valid_predictions.parquet", 0.7174), ("valid_predictions_tweedie.parquet", 0.7036)]:
        pred = preds_to_matrix(meta, pd.read_parquet(ROOT / "data" / "processed" / f))
        r = m.score(pred)
        print(f"skip_leading_zeros={skip} {f}: WRMSSE={r['WRMSSE']:.4f} (summary says {expected})")
        print({k: round(v, 4) for k, v in r.items() if k != "WRMSSE"})
