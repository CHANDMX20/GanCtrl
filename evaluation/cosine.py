import sys
import os
from os.path import join

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # <-- use package fn

# Load the data
real_control = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_train_control_cv2.csv')
syn_control  = pd.read_csv('/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/predictions_decoded/test/generated_predictions_merged_train.csv')

# ---- STRICT filter: DOSE_LEVEL must be exactly "High" ----
dose_col = 'DOSE_LEVEL'
if dose_col not in syn_control.columns:
    raise KeyError("syn_control is missing 'DOSE_LEVEL' column.")
before_n = len(syn_control)
syn_control = syn_control[syn_control[dose_col] == 'High'].copy()
after_n = len(syn_control)
print(f"[info] syn_control filtered to DOSE_LEVEL == 'High': {after_n}/{before_n} rows retained.")

# ---- 38-feature panel: all columns after index 11 in real_control ----
# (0-based indexing: columns[0]..columns[10] are meta, columns[11:] are the 38 biomarkers)
FEATURE_38 = list(real_control.columns[11:])
print(f"[info] 38-feature panel taken from real_control columns[11:]:")
print(FEATURE_38)

def _prepare_panel_features(real: pd.DataFrame, syn: pd.DataFrame, panel_cols):
    """Keep panel columns present in BOTH dataframes; coerce to numeric; return actual list used."""
    present_both = [c for c in panel_cols if (c in real.columns and c in syn.columns)]
    missing = [c for c in panel_cols if c not in present_both]
    if missing:
        print(f"[warn] Missing in one/both inputs (skipped): {missing}")
    if not present_both:
        raise ValueError("None of the requested panel columns are present in both dataframes.")
    for c in present_both:
        real[c] = pd.to_numeric(real[c], errors="coerce")
        syn[c]  = pd.to_numeric(syn[c],  errors="coerce")
    present_both = [
        c for c in present_both
        if pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(syn[c])
    ]
    if not present_both:
        raise ValueError("After coercion, none of the panel columns are numeric in both dataframes.")
    return present_both

def compute_control_cosine_panel(real: pd.DataFrame, syn: pd.DataFrame, panel_cols, outfile: str):
    """
    Real-vs-Synthetic cosine similarity using ONLY the given panel columns.
    Inner-joins on ['COMPOUND_NAME','SACRIFICE_PERIOD'] and computes rowwise cosine
    with sklearn.metrics.pairwise.cosine_similarity.
    """
    key_cols = ['COMPOUND_NAME', 'SACRIFICE_PERIOD']
    for k in key_cols:
        if k not in real.columns or k not in syn.columns:
            raise KeyError(f"Missing required key column '{k}' in inputs.")

    real = real.copy()
    syn  = syn.copy()

    feats = _prepare_panel_features(real, syn, panel_cols)
    print(f"[info] Using {len(feats)} feature columns for cosine calculation.")

    merged = pd.merge(
        real[key_cols + feats],
        syn[key_cols + feats],
        on=key_cols,
        how='inner',
        suffixes=('_real', '_syn')
    )

    if merged.empty:
        print(f"[warn] Merge produced 0 rows for panel; writing empty file: {outfile}")
        merged.assign(cosine=pd.Series(dtype=float))[key_cols + ['cosine']].to_csv(outfile, index=False)
        return merged

    # Clean NaN/inf ? 0.0 for the feature columns before cosine calc
    for c in feats:
        merged[f"{c}_real"] = merged[f"{c}_real"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        merged[f"{c}_syn"]  = merged[f"{c}_syn"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    real_cols = [f"{c}_real" for c in feats]
    syn_cols  = [f"{c}_syn"  for c in feats]

    # Use sklearn's cosine_similarity per row
    merged['cosine'] = [
        float(cosine_similarity(merged.loc[[i], real_cols], merged.loc[[i], syn_cols])[0, 0])
        for i in merged.index
    ]

    result = merged[key_cols + ['cosine']]
    result.to_csv(outfile, index=False)
    print(f"[info] Wrote {len(result)} rows to: {outfile}")
    return result

# ---- Output paths ----
base_path   = '/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2'
cosine_path = join(base_path, 'performance', 'cosine', 'test')
os.makedirs(cosine_path, exist_ok=True)

# ---- Run: single 38-feature cosine (High dose only) ----
all_out = join(cosine_path, 'generated_cosine_all_train.csv')
all_df  = compute_control_cosine_panel(real_control, syn_control, FEATURE_38, all_out)
