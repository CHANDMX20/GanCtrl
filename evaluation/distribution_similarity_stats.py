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
    * mean, median, sd
    * relative mean difference
    * sd ratio (syn/real)
    * KS statistic & p-value
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
    Coerce column to numeric, round, and optionally cast to integer (like R num_round).
    If column doesn't exist, returns df unchanged.
    """
    if col not in df.columns:
        return df

    x = pd.to_numeric(df[col], errors="coerce")

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

    if dose_col == "DOSE":
        # Numeric DOSE: keep max dose within group
        group_keys = [c for c in ["COMPOUND_NAME", "targetTime", "INDIVIDUAL_ID"]
                      if c in df.columns]
        tmp = df.copy()
        tmp[dose_col] = pd.to_numeric(tmp[dose_col], errors="coerce")

        if len(group_keys) == 0:
            max_dose = tmp[dose_col].max()
            out = tmp[tmp[dose_col] == max_dose]
            return out.reset_index(drop=True)

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
    """Helper: coerce to numeric."""
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


# =============================================================================
# 3) Filter generated to High dose only
# =============================================================================

print("Applying High-dose filter to generated data...")
generated_liver  = filter_high_only(generated_liver)
generated_kidney = filter_high_only(generated_kidney)


# =============================================================================
# 4) Build collapsed real_all and gen_all for *all* features
# =============================================================================

# All measurement columns in REAL: columns from index 11 onward
raw_all_features = list(real.columns[11:])

# Only keep those that also appear in at least one synthetic file
synthetic_cols = set(generated_liver.columns) | set(generated_kidney.columns)
all_features = [c for c in raw_all_features if c in synthetic_cols]

if not all_features:
    raise ValueError("No overlapping measurement columns found between real and synthetic data.")

print(f"Using {len(all_features)} measurement features (from real.columns[11:]):")
print(all_features)

# ---- Collapse REAL ----
group_cols_real = [c for c in ["COMPOUND_NAME", "SACRIFICE_PERIOD", "INDIVIDUAL_ID"]
                   if c in real.columns]
if not group_cols_real:
    raise ValueError("Real data is missing required grouping columns "
                     "(COMPOUND_NAME/SACRIFICE_PERIOD/INDIVIDUAL_ID).")

real_all = real[group_cols_real + all_features].copy()
for feat in all_features:
    real_all[feat] = to_num_series(real_all[feat])

agg_real = {feat: "mean" for feat in all_features}
real_all = real_all.groupby(group_cols_real, as_index=False).agg(agg_real)

print(f"Collapsed real_all shape: {real_all.shape}")

# ---- Collapse SYNTHETIC (combine liver + kidney, then group) ----
gen_all_raw = pd.concat([generated_liver, generated_kidney],
                        ignore_index=True, sort=False)

group_cols_syn = [c for c in ["COMPOUND_NAME", "targetTime", "targetBioCopy"]
                  if c in gen_all_raw.columns]
if not group_cols_syn:
    raise ValueError("Synthetic data is missing required grouping columns "
                     "(COMPOUND_NAME/targetTime/targetBioCopy).")

gen_all = gen_all_raw[group_cols_syn + all_features].copy()
for feat in all_features:
    gen_all[feat] = to_num_series(gen_all[feat])

agg_syn = {feat: "mean" for feat in all_features}
gen_all = gen_all.groupby(group_cols_syn, as_index=False).agg(agg_syn)

print(f"Collapsed gen_all shape: {gen_all.shape}")


# =============================================================================
# 5) Jensen–Shannon divergence helper (robust)
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
# 6) Core comparison function (all features, no Wasserstein)
# =============================================================================

def compare_distributions(
    real_df,
    syn_df,
    all_features,
    label_real="Real",
    label_syn="GanCtrl",
    jsd_bins=25,
):
    """
    For each feature in `all_features`, compute:

      - n_real, n_syn
      - mean_real, mean_syn
      - relative mean difference: (mean_syn - mean_real) / mean_real
      - js_divergence_log2 (Jensen–Shannon, base 2)

    No Wasserstein, no SDs, no KS.
    """
    rows = []

    for feat in all_features:
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

        # Jensen–Shannon
        jsd = jensen_shannon_from_samples(xr, xs,
                                          n_bins=jsd_bins, base=2)

        rows.append({
            "feature": feat,
            "n_real": n_r,
            "n_syn": n_s,
            f"{label_real}_mean": mean_r,
            f"{label_syn}_mean": mean_s,
            "rel_mean_diff": rel_mean_diff,
            "js_divergence_log2": jsd,
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["js_divergence_log2", "feature"],
            na_position="last"
        )
    return summary.reset_index(drop=True)


# =============================================================================
# 7) Run comparisons and write results
# =============================================================================

if __name__ == "__main__":
    print("Computing similarity stats for all features (means + rel diff + JSD)...")
    all_stats = compare_distributions(
        real_df=real_all,
        syn_df=gen_all,
        all_features=all_features,
        label_real="Real",
        label_syn="GanCtrl",
        jsd_bins=25,
    )

    out_path = os.path.join(out_dir, "similarity_stats_all_features.csv")
    all_stats.to_csv(out_path, index=False)
    print(f"Saved stats to: {out_path}")
    print(all_stats)

