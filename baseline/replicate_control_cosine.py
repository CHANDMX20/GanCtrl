"""
Replicate-level cosine similarity for control clinical pathology data.

Pipeline overview
-----------------
1. Load per-animal control clinical pathology data from:
   - A training split
   - A test split
2. Concatenate the splits into a single control dataframe.
3. Define a 38-feature biomarker panel as all columns after index 11
   (assuming the first 11 columns are ID/metadata).
4. Within each (COMPOUND_NAME, SACRIFICE_PERIOD) group:
   - Prepare numeric feature matrices.
   - Compute pairwise cosine similarity between all replicate pairs.
5. Write out one row per unique replicate pair with:
   COMPOUND_NAME, SACRIFICE_PERIOD, INDIVIDUAL_ID_1, INDIVIDUAL_ID_2, cosine.

Note:
-----
- Paths below are placeholders; update them to match your environment.
- This script assumes certain column names exist:
  COMPOUND_NAME, SACRIFICE_PERIOD, INDIVIDUAL_ID, etc.
"""

import sys
import os
from os.path import join

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 0. FILE PATHS (UPDATE THESE FOR YOUR ENVIRONMENT)
# =============================================================================

CONTROL_TEST_FILE = "/path/to/clin_path/repeat_test_control_cv2.csv"
CONTROL_TRAIN_FILE = "/path/to/clin_path/repeat_train_control_cv2.csv"
POS_CTRL_BASE_PATH = "/path/to/clin_path/positive_control"

# =============================================================================
# 1. LOAD CONTROL DATA
# =============================================================================

test = pd.read_csv(CONTROL_TEST_FILE)
train = pd.read_csv(CONTROL_TRAIN_FILE)

control = pd.concat([test, train], ignore_index=True, sort=False)

# =============================================================================
# 2. DEFINE 38-FEATURE BIOMARKER PANEL
# =============================================================================

# 38-feature panel: all biomarker columns after index 11
# (0-based: columns[0..10] are assumed meta, columns[11:] are the 38 biomarkers)
FEATURE_38 = list(control.columns[11:])
print("[info] 38-feature panel (cols[11:]):")
print(FEATURE_38)

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

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

def compute_replicate_control_cosine_panel(
    real: pd.DataFrame,
    feature_cols,
    filename: str,
    id_col: str = "INDIVIDUAL_ID",
):
    """
    Replicate cosine similarity within each (COMPOUND_NAME, SACRIFICE_PERIOD) group,
    using all given feature columns together. One row per unique pair (i < j).
    """
    # 1) Features (e.g. all 38 biomarkers)
    work = real.copy()
    features = _prepare_panel_features(work, feature_cols)

    # 2) Keys
    group_keys = ["COMPOUND_NAME", "SACRIFICE_PERIOD"]
    for k in group_keys:
        if k not in work.columns:
            raise KeyError(f"Missing required column '{k}' in input dataframe.")

    results = []
    # 3) For each compound & time, compare replicates (unique pairs)
    for (cmpd, period), grp in work.groupby(group_keys, dropna=False):
        if len(grp) < 2:
            continue
        X = (
            grp[features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        ids = grp[id_col].values if id_col in grp.columns else np.arange(len(grp))
        S = cosine_similarity(X, X)

        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                results.append({
                    "COMPOUND_NAME":    cmpd,
                    "SACRIFICE_PERIOD": period,
                    f"{id_col}_1":      ids[i],
                    f"{id_col}_2":      ids[j],
                    "cosine":           float(S[i, j]),
                })

    # 4) Write out
    out = pd.DataFrame(results)
    out.to_csv(filename, index=False)
    print(f"[info] Wrote {len(out)} rows to: {filename}")
    return out

# =============================================================================
# 4. RUN FOR ALL 38 BIOMARKERS TOGETHER
# =============================================================================

# Output file for replicate cosine similarity
all38_csv = join(POS_CTRL_BASE_PATH, "pc_cosine_overall.csv")

all38_df = compute_replicate_control_cosine_panel(
    control,
    FEATURE_38,
    all38_csv,
    id_col="INDIVIDUAL_ID",
)
