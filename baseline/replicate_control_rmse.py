#import 
import sys
import os
from os.path import join

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error  # <-- RMSE via MSE

# Load the data
test  = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_test_control_cv2.csv')
train = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_train_control_cv2.csv')

control = pd.concat([test, train], ignore_index=True, sort=False)

# ---- 38-feature panel: all biomarker columns after index 11 ----
# (0-based: columns[0..10] are meta, columns[11:] are the 38 biomarkers)
FEATURE_38 = list(control.columns[11:])
print("[info] 38-feature panel (cols[11:]):")
print(FEATURE_38)

def _prepare_panel_features(df: pd.DataFrame, feature_cols):
    """Keep feature columns that exist; coerce to numeric; return list actually used."""
    present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[warn] Missing feature columns (skipped): {missing}")
    if not present:
        raise ValueError("None of the requested feature columns are present in the dataframe.")
    # coerce to numeric
    for c in present:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # keep numeric-only after coercion
    present = [c for c in present if pd.api.types.is_numeric_dtype(df[c])]
    if not present:
        raise ValueError("After coercion, none of the feature columns are numeric.")
    return present

def compute_replicate_control_rmse_panel(
    real: pd.DataFrame,
    feature_cols,
    filename: str,
    id_col: str = 'INDIVIDUAL_ID'
):
    """
    Replicate RMSE within each (COMPOUND_NAME, SACRIFICE_PERIOD) group,
    using all given feature columns together. One row per unique pair (i<j).
    """
    # 1) Features (all 38 biomarkers)
    work = real.copy()
    features = _prepare_panel_features(work, feature_cols)

    # 2) Keys
    group_keys = ['COMPOUND_NAME', 'SACRIFICE_PERIOD']
    for k in group_keys:
        if k not in work.columns:
            raise KeyError(f"Missing required column '{k}' in input dataframe.")

    results = []
    # 3) For each compound & time, compare replicates (unique pairs)
    for (cmpd, period), grp in work.groupby(group_keys, dropna=False):
        if len(grp) < 2:
            continue

        X = grp[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        ids = grp[id_col].values if id_col in grp.columns else np.arange(len(grp))

        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                y_true = X[i, :]
                y_pred = X[j, :]
                mse    = mean_squared_error(y_true, y_pred)  # squared=True by default
                rmse   = float(np.sqrt(mse))
                results.append({
                    'COMPOUND_NAME':    cmpd,
                    'SACRIFICE_PERIOD': period,
                    f'{id_col}_1':      ids[i],
                    f'{id_col}_2':      ids[j],
                    'rmse':             rmse
                })

    # 4) Write out
    out = pd.DataFrame(results)
    out.to_csv(filename, index=False)
    print(f"[info] Wrote {len(out)} rows to: {filename}")
    return out

# ---- Paths ----
base_path = '/account001/mansi.chandra/clin_path/positive_control'

# ---- Run for all 38 biomarkers together ----
all38_csv = join(base_path, 'pc_rmse_overall.csv')

all38_df = compute_replicate_control_rmse_panel(
    control,
    FEATURE_38,
    all38_csv,
    id_col='INDIVIDUAL_ID'
)
