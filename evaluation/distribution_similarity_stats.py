#!/usr/bin/env python3
"""
Similarity stats for Real vs GanCtrl synthetic controls (Liver & Kidney panels)

- Reads the same input CSVs as the R histogram script:
    * generated liver predictions
    * generated kidney predictions
    * real controls

- Applies:
    * numeric harmonization (rounding / integer casting)
    * High-dose filtering (generic DOSE / DOSE_LEVEL logic)
    * collapsing to per-group means:
        - Real: COMPOUND_NAME × SACRIFICE_PERIOD × INDIVIDUAL_ID
        - Syn : COMPOUND_NAME × targetTime × targetBioCopy

- Computes, per biomarker:
    * n_real, n_syn
    * mean
    * relative mean difference
    * Wasserstein distance (raw + normalized by IQR(real))
    * Jensen–Shannon divergence (0–1, log base 2)

Outputs:
    similarity_stats_liver.csv
    similarity_stats_kidney.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 0) User-configurable paths
# =============================================================================

# EDIT THESE to match the R script
generated_path_liver  = "/path/to/generated_predictions_liver.csv"
generated_path_kidney = "/path/to/generated_predictions_kidney.csv"
real_path             = "/path/to/test_control.csv"

out_dir = "/path/to/output/results_stats"
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 1) Helpers: numeric rounding, dose filtering, etc.
# =============================================================================

def num_round(df, col, digits=None, to_integer=False):
    """
    Mimic the R num_round function:
        - coerce to numeric
        - round (digits or nearest int)
        - optionally cast to integer
    If column doesn't exist, returns df unchanged.
    """
    if col not in df.columns:
        return df

    v = df[col]
    # Convert factors/strings to numeric
    x = pd.to_numeric(v, errors="coerce")

    if digits is None:
        x = x.round()
    else:
        x = x.round(digits)

    if to_integer:
        x = x.astype("Int64")  # nullable integer
    df[col] = x
    return df


def normalize_dose(x):
    return x.astype(str).str.strip().str.lower()


def is_high_label(series):
    labels = {"high", "hi", "top", "max", "highest", "hd", "h"}
    return normalize_dose(series).isin(labels)


def pick_dose_col(df):
    """
    Try to find a dose column similar to the R script:
    DOSE_LEVEL, Dose_Level, DoseLevel, DOSE_GROUP, DoseGroup, DOSE
    """
    candidates = ["DOSE_LEVEL", "Dose_Level", "DoseLevel",
                  "DOSE_GROUP", "DoseGroup", "DOSE"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def filter_high_only(df):
    """
    Filter generated data to High dose, mimicking the R logic:
      - If DOSE numeric: keep max DOSE within each group
      - If DOSE_LEVEL/DOSE_GROUP string-like: keep 'High' labels
      - If no dose column: return df unchanged (with warning).
    """
    dose_col = pick_dose_col(df)
    if dose_col is None:
        print("[WARN] No DOSE/DOSE_LEVEL column found; skipping High-dose filter.")
        return df

    # Try numeric or string based on dtype
    if dose_col == "DOSE":
        # Numeric DOSE: keep max dose within group
        # Group keys: COMPOUND_NAME, targetTime, INDIVIDUAL_ID (if present)
        group_keys = [c for c in ["COMPOUND_NAME", "targetTime", "INDIVIDUAL_ID"]
                      if c in df.columns]

        if len(group_keys) == 0:
            # No grouping keys: just keep rows with max dose
            max_dose = pd.to_numeric(df[dose_col], errors="coerce").max()
            return df[pd.to_numeric(df[dose_col], errors="coerce") == max_dose]

        tmp = df.copy()
        tmp[dose_col] = pd.to_numeric(tmp[dose_col], errors="coerce")

        # groupby and filter max within each group
        tmp["__dose_max__"] = tmp.groupby(group_keys)[dose_col].transform("max")
        out = tmp[tmp[dose_col] == tmp["__dose_max__"]].drop(columns="__dose_max__")
        return out.reset_index(drop=True)
    else:
        # Categorical DOSE_LEVEL / DOSE_GROUP / DoseGroup
        mask = is_high_label(df[dose_col])
        out = df[mask].copy()
        if out.empty:
            print(f"[WARN] High-dose filtering on {dose_col} returned 0 rows.")
        return out.reset_index(drop=True)


def to_num_series(s):
    """Helper: coerce to numeric, drop NAs later."""
    return pd.to_numeric(s, errors="coerce")


# =============================================================================
# 2) Load data & harmonize numeric formats (like R)
# =============================================================================

print("Reading input CSVs...")
generated_liver  = pd.read_csv(generated_path_liver)
generated_kidney = pd.read_csv(generated_path_kidney)
real             = pd.read_csv(real_path)


# ---- Liver numeric harmonization ----
for col, cfg in [
    ("TBIL(mg/dL)", dict(digits=2, to_integer=False)),
    ("RALB(g/dL)",  dict(digits=2, to_integer=False)),
    ("AST(IU/L)",   dict(digits=None, to_integer=True)),
    ("TP(g/dL)",    dict(digits=1, to_integer=False)),
    ("DBIL(mg/dL)", dict(digits=2, to_integer=False)),
    ("BUN(mg/dL)",  dict(digits=None, to_integer=True)),
    ("ALP(IU/L)",   dict(digits=None, to_integer=True)),
    ("ALT(IU/L)",   dict(digits=None, to_integer=True)),
    ("LDH(IU/L)",   dict(digits=None, to_integer=True)),
]:
    generated_liver = num_round(generated_liver, col, **cfg)

# final RALB at 1 decimal
generated_liver = num_round(generated_liver, "RALB(g/dL)", digits=1, to_integer=False)

# ---- Kidney numeric harmonization ----
for col, cfg in [
    ("TBIL(mg/dL)", dict(digits=2, to_integer=False)),
    ("RALB(g/dL)",  dict(digits=2, to_integer=False)),
    ("AST(IU/L)",   dict(digits=None, to_integer=True)),
    ("TP(g/dL)",    dict(digits=1, to_integer=False)),
    ("DBIL(mg/dL)", dict(digits=2, to_integer=False)),
    ("BUN(mg/dL)",  dict(digits=None, to_integer=True)),
    ("ALP(IU/L)",   dict(digits=None, to_integer=True)),
    ("ALT(IU/L)",   dict(digits=None, to_integer=True)),
    ("LDH(IU/L)",   dict(digits=None, to_integer=True)),
]:
    generated_kidney = num_round(generated_kidney, col, **cfg)

generated_kidney = num_round(generated_kidney, "RALB(g/dL)", digits=1, to_integer=False)


# Feature lists (matches your R code order)
liver_features = [
    "ALP(IU/L)", "ALT(IU/L)", "AST(IU/L)",
    "GTP(IU/L)", "LDH(IU/L)",
    "DBIL(mg/dL)", "TBIL(mg/dL)"
]

kidney_features = [
    "BUN(mg/dL)", "CRE(mg/dL)", "Cl(meq/L)", "Ca(mg/dL)",
    "K(meq/L)", "IP(mg/dL)", "Na(meq/L)"
]


# =============================================================================
# 3) Filter generated to High dose only
# =============================================================================

print("Applying High-dose filter to generated data...")
generated_liver  = filter_high_only(generated_liver)
generated_kidney = filter_high_only(generated_kidney)


# =============================================================================
# 4) Collapse to per-group means (mirrors R)
# =============================================================================

def collapse_real_liver(real_df):
    cols = ["COMPOUND_NAME", "SACRIFICE_PERIOD", "INDIVIDUAL_ID"] + liver_features
    cols_existing = [c for c in cols if c in real_df.columns]
    df = real_df[cols_existing].copy()

    group_cols = [c for c in ["COMPOUND_NAME", "SACRIFICE_PERIOD", "INDIVIDUAL_ID"]
                  if c in df.columns]
    if not group_cols:
        raise ValueError("Real liver: required grouping columns not found.")

    for feat in liver_features:
        if feat in df.columns:
            df[feat] = to_num_series(df[feat])

    agg_dict = {feat: "mean" for feat in liver_features if feat in df.columns}
    return df.groupby(group_cols, as_index=False).agg(agg_dict)


def collapse_real_kidney(real_df):
    cols = ["COMPOUND_NAME", "SACRIFICE_PERIOD", "INDIVIDUAL_ID"] + kidney_features
    cols_existing = [c for c in cols if c in real_df.columns]
    df = real_df[cols_existing].copy()

    group_cols = [c for c in ["COMPOUND_NAME", "SACRIFICE_PERIOD", "INDIVIDUAL_ID"]
                  if c in df.columns]
    if not group_cols:
        raise ValueError("Real kidney: required grouping columns not found.")

    for feat in kidney_features:
        if feat in df.columns:
            df[feat] = to_num_series(df[feat])

    agg_dict = {feat: "mean" for feat in kidney_features if feat in df.columns}
    return df.groupby(group_cols, as_index=False).agg(agg_dict)


def collapse_generated_liver(gen_df):
    cols = ["COMPOUND_NAME", "targetTime", "targetBioCopy"] + liver_features
    cols_existing = [c for c in cols if c in gen_df.columns]
    df = gen_df[cols_existing].copy()

    group_cols = [c for c in ["COMPOUND_NAME", "targetTime", "targetBioCopy"]
                  if c in df.columns]
    if not group_cols:
        raise ValueError("Generated liver: required grouping columns not found.")

    for feat in liver_features:
        if feat in df.columns:
            df[feat] = to_num_series(df[feat])

    agg_dict = {feat: "mean" for feat in liver_features if feat in df.columns}
    return df.groupby(group_cols, as_index=False).agg(agg_dict)


def collapse_generated_kidney(gen_df):
    cols = ["COMPOUND_NAME", "targetTime", "targetBioCopy"] + kidney_features
    cols_existing = [c for c in cols if c in gen_df.columns]
    df = gen_df[cols_existing].copy()

    group_cols = [c for c in ["COMPOUND_NAME", "targetTime", "targetBioCopy"]
                  if c in df.columns]
    if not group_cols:
        raise ValueError("Generated kidney: required grouping columns not found.")

    for feat in kidney_features:
        if feat in df.columns:
            df[feat] = to_num_series(df[feat])

    agg_dict = {feat: "mean" for feat in kidney_features if feat in df.columns}
    return df.groupby(group_cols, as_index=False).agg(agg_dict)


print("Collapsing to per-group means...")
real_liver  = collapse_real_liver(real)
real_kidney = collapse_real_kidney(real)
gen_liver_collapsed  = collapse_generated_liver(generated_liver)
gen_kidney_collapsed = collapse_generated_kidney(generated_kidney)


# =============================================================================
# 5) Jensen–Shannon divergence helper
# =============================================================================

def jensen_shannon_from_samples(x_real, x_syn, n_bins=25, base=2):
    """
    Estimate JSD between two 1D distributions using shared histogram bins.

    Robust to Pandas nullable dtypes by coercing to float first.
    Returns:
        jsd: Jensen–Shannon divergence (0 to 1 if base=2).
    """
    # Coerce to numeric 1D float arrays
    x_real = pd.to_numeric(pd.Series(x_real).ravel(), errors="coerce").astype(float)
    x_syn  = pd.to_numeric(pd.Series(x_syn).ravel(),  errors="coerce").astype(float)

    # Keep only finite values
    x_real = x_real[np.isfinite(x_real)]
    x_syn  = x_syn[np.isfinite(x_syn)]

    if x_real.size == 0 or x_syn.size == 0:
        return np.nan

    data_min = min(x_real.min(), x_syn.min())
    data_max = max(x_real.max(), x_syn.max())
    if data_min == data_max:
        # All values equal -> identical distributions
        return 0.0

    bins = np.linspace(data_min, data_max, n_bins + 1)

    p_counts, _ = np.histogram(x_real, bins=bins)
    q_counts, _ = np.histogram(x_syn,  bins=bins)

    eps = 1e-12
    p = p_counts.astype(float) + eps
    q = q_counts.astype(float) + eps
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    if base == 2:
        log_fn = np.log2
    else:
        log_fn = lambda x: np.log(x) / np.log(base)

    kl_pm = np.sum(p * (log_fn(p) - log_fn(m)))
    kl_qm = np.sum(q * (log_fn(q) - log_fn(m)))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(jsd)

# =============================================================================
# 6) Core comparison function (stats only, no plots)
# =============================================================================

def compare_distributions(real_df, syn_df, features,
                          label_real="Real", label_syn="GanCtrl",
                          jsd_bins=25):
    """
    For each feature in `features`, compute:

      - n_real, n_syn
      - mean_real, mean_syn
      - relative mean difference: (mean_syn - mean_real) / mean_real
      - Wasserstein distance (raw, and normalized by IQR(real))
      - Jensen–Shannon divergence (0–1, base-2 log)

    No SDs, medians, or KS stats.
    """
    rows = []

    for feat in features:
        if feat not in real_df.columns or feat not in syn_df.columns:
            print(f"[WARN] {feat} not found in both real & synthetic; skipping.")
            continue

        # Coerce to numeric
        xr = to_num_series(real_df[feat]).dropna()
        xs = to_num_series(syn_df[feat]).dropna()

        n_r = xr.size
        n_s = xs.size
        if n_r == 0 or n_s == 0:
            print(f"[WARN] {feat}: empty real or synthetic sample; skipping.")
            continue

        mean_r = xr.mean()
        mean_s = xs.mean()

        # Relative mean difference (syn - real)/real
        if mean_r != 0:
            rel_mean_diff = (mean_s - mean_r) / mean_r
        else:
            rel_mean_diff = np.nan

        # Wasserstein distance
        w_raw = stats.wasserstein_distance(xr, xs)
        q25, q75 = xr.quantile([0.25, 0.75])
        iqr = q75 - q25
        w_norm = w_raw / iqr if iqr > 0 else np.nan

        # Jensen–Shannon divergence (histogram-based)
        jsd = jensen_shannon_from_samples(xr, xs,
                                          n_bins=jsd_bins, base=2)

        rows.append({
            "feature": feat,
            "n_real": n_r,
            "n_syn": n_s,
            f"{label_real}_mean": mean_r,
            f"{label_syn}_mean": mean_s,
            "rel_mean_diff": rel_mean_diff,
            "wasserstein_raw": w_raw,
            "wasserstein_norm_IQR": w_norm,
            "js_divergence_log2": jsd
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("js_divergence_log2")
    return summary.reset_index(drop=True)

# =============================================================================
# 7) Run comparisons and write results
# =============================================================================

if __name__ == "__main__":
    print("Computing similarity stats for liver panel...")
    liver_stats = compare_distributions(real_liver, gen_liver_collapsed, liver_features)
    liver_out_path = os.path.join(out_dir, "similarity_stats_liver.csv")
    liver_stats.to_csv(liver_out_path, index=False)
    print(f"Liver stats saved to: {liver_out_path}")
    print(liver_stats)

    print("\nComputing similarity stats for kidney panel...")
    kidney_stats = compare_distributions(real_kidney, gen_kidney_collapsed, kidney_features)
    kidney_out_path = os.path.join(out_dir, "similarity_stats_kidney.csv")
    kidney_stats.to_csv(kidney_out_path, index=False)
    print(f"Kidney stats saved to: {kidney_out_path}")
    print(kidney_stats)

    print("\nDone.")

