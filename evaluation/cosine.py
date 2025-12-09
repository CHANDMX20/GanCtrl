"""
Real vs Synthetic Control Cosine Similarity (Controls, High Dose Only)
===========================================================================

Purpose
-------
This script compares real test controls against synthetic controls generated
by a VAE/GAN-style model, using cosine similarity across a 38-biomarker panel.

It does the following:

1. Loads:
   - real_control:  repeat_test_control_cv2.csv
   - syn_control:   generated_predictions_merged_test.csv

2. Strictly filters synthetic rows to:
   DOSE_LEVEL == "High"
   (these are interpreted as “control-equivalent under High-dose conditions”).

3. Defines the 38-feature panel as all columns after index 11 in real_control
   (0-based indexing: columns[0..10] are metadata, columns[11:] are biomarkers).

4. For each row where real and synthetic data share the same
   (COMPOUND_NAME, SACRIFICE_PERIOD), computes a cosine similarity between the
   38-dimensional real and synthetic vectors using sklearn.metrics.pairwise.cosine_similarity.

5. Writes out a CSV containing:
   COMPOUND_NAME, SACRIFICE_PERIOD, cosine

Paths & Assumptions
-------------------
- Assumes both real and synthetic files share:
    * COMPOUND_NAME
    * SACRIFICE_PERIOD
    * DOSE_LEVEL
    * the same biomarker columns in FEATURE_38

- Input/output paths are defined as placeholders below; update them to match
  your local directory structure.

Outputs
-------
- <BASE_OUTPUT_DIR>/performance/cosine/test/generated_cosine_all_test.csv

"""

import sys
import os
from os.path import join

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # <-- package cosine

# =============================================================================
# 0. CONFIG: update these paths for your environment
# =============================================================================

REAL_CONTROL_FILE = "/path/to/repeat_test_control_cv2.csv"
SYN_CONTROL_FILE = (
    "/path/to/results_vae_corr_mod3_cv2/"
    "predictions_decoded/test/generated_predictions_merged_test.csv"
)
BASE_OUTPUT_DIR = "/path/to/results_vae_corr_mod3_cv2"

# =============================================================================
# 1. Load data
# =============================================================================

# Real train controls
real_control = pd.read_csv(REAL_CONTROL_FILE)

# Synthetic control-like predictions
syn_control = pd.read_csv(SYN_CONTROL_FILE)

# =============================================================================
# 2. STRICT filter: retain only DOSE_LEVEL == "High" in synthetic data
# =============================================================================

dose_col = "DOSE_LEVEL"
if dose_col not in syn_control.columns:
    raise KeyError("syn_control is missing 'DOSE_LEVEL' column.")

before_n = len(syn_control)
syn_control = syn_control[syn_control[dose_col] == "High"].copy()
after_n = len(syn_control)
print(
    f"[info] syn_control filtered to DOSE_LEVEL == 'High': "
    f"{after_n}/{before_n} rows retained."
)

# =============================================================================
# 3. Define 38-feature panel (all columns after index 11 in real_control)
# =============================================================================

# 0-based indexing: columns[0..10] are meta, columns[11:] are the 38 biomarkers
FEATURE_38 = list(real_control.columns[11:])
print("[info] 38-feature panel taken from real_control columns[11:]:")
print(FEATURE_38)

# =============================================================================
# 4. Helpers
# =============================================================================


def _prepare_panel_features(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    panel_cols: list[str],
) -> list[str]:
    """
    Restrict to panel columns present in BOTH dataframes, coerce to numeric,
    and return the list of feature columns actually used.

    Non-numeric or missing columns are dropped.

    Parameters
    ----------
    real : pd.DataFrame
        Real control dataframe.
    syn : pd.DataFrame
        Synthetic control-like dataframe.
    panel_cols : list of str
        Candidate biomarker column names (e.g. FEATURE_38).

    Returns
    -------
    list of str
        Feature columns usable in both real and syn after numeric coercion.

    Raises
    ------
    ValueError
        If no usable panel columns exist in both inputs.
    """
    # Columns present in both frames
    present_both = [c for c in panel_cols if (c in real.columns and c in syn.columns)]
    missing = [c for c in panel_cols if c not in present_both]
    if missing:
        print(f"[warn] Missing in one/both inputs (skipped): {missing}")
    if not present_both:
        raise ValueError(
            "None of the requested panel columns are present in both dataframes."
        )

    # Coerce to numeric
    for c in present_both:
        real[c] = pd.to_numeric(real[c], errors="coerce")
        syn[c] = pd.to_numeric(syn[c], errors="coerce")

    # Keep only numeric columns after coercion
    present_both = [
        c
        for c in present_both
        if pd.api.types.is_numeric_dtype(real[c])
        and pd.api.types.is_numeric_dtype(syn[c])
    ]
    if not present_both:
        raise ValueError(
            "After coercion, none of the panel columns are numeric in both dataframes."
        )
    return present_both


def compute_control_cosine_panel(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    panel_cols: list[str],
    outfile: str,
) -> pd.DataFrame:
    """
    Compute real-vs-synthetic cosine similarity for control-equivalent profiles.

    For each row with matching (COMPOUND_NAME, SACRIFICE_PERIOD), we:

      1. Take the panel feature vector from 'real' (38 biomarkers).
      2. Take the panel feature vector from 'syn'.
      3. Compute cosine similarity between these two vectors using
         sklearn.metrics.pairwise.cosine_similarity.
      4. Output COMPOUND_NAME, SACRIFICE_PERIOD, cosine.

    Parameters
    ----------
    real : pd.DataFrame
        Real control dataframe.
    syn : pd.DataFrame
        Synthetic control-like dataframe (already filtered to DOSE_LEVEL == "High").
    panel_cols : list of str
        Biomarker column names (e.g. FEATURE_38).
    outfile : str
        Full path to the CSV file to write.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          ['COMPOUND_NAME', 'SACRIFICE_PERIOD', 'cosine']
    """
    key_cols = ["COMPOUND_NAME", "SACRIFICE_PERIOD"]
    for k in key_cols:
        if k not in real.columns or k not in syn.columns:
            raise KeyError(f"Missing required key column '{k}' in inputs.")

    real = real.copy()
    syn = syn.copy()

    # Decide which panel columns can be used
    feats = _prepare_panel_features(real, syn, panel_cols)
    print(f"[info] Using {len(feats)} feature columns for cosine calculation.")

    # Inner-join on keys to align real & synthetic rows
    merged = pd.merge(
        real[key_cols + feats],
        syn[key_cols + feats],
        on=key_cols,
        how="inner",
        suffixes=("_real", "_syn"),
    )

    if merged.empty:
        print(
            f"[warn] Merge produced 0 rows for panel; "
            f"writing empty file: {outfile}"
        )
        merged.assign(cosine=pd.Series(dtype=float))[
            key_cols + ["cosine"]
        ].to_csv(outfile, index=False)
        return merged

    # Clean NaN/inf -> 0.0 in feature columns prior to cosine computation
    for c in feats:
        merged[f"{c}_real"] = (
            merged[f"{c}_real"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        merged[f"{c}_syn"] = (
            merged[f"{c}_syn"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    real_cols = [f"{c}_real" for c in feats]
    syn_cols = [f"{c}_syn" for c in feats]

    # Row-wise cosine similarity (real vs syn) using sklearn’s implementation
    merged["cosine"] = [
        float(
            cosine_similarity(
                merged.loc[[i], real_cols],
                merged.loc[[i], syn_cols],
            )[0, 0]
        )
        for i in merged.index
    ]

    result = merged[key_cols + ["cosine"]]
    result.to_csv(outfile, index=False)
    print(f"[info] Wrote {len(result)} rows to: {outfile}")
    return result


# =============================================================================
# 5. Paths & run (single 38-feature cosine, High dose synthetic only)
# =============================================================================

cosine_path = join(BASE_OUTPUT_DIR, "performance", "cosine", "test")
os.makedirs(cosine_path, exist_ok=True)

all_out = join(cosine_path, "generated_cosine_all_test.csv")

all_df = compute_control_cosine_panel(
    real_control,
    syn_control,
    FEATURE_38,
    all_out,
)
