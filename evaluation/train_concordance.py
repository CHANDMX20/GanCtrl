"""
Train-Set Abnormality Concordance: Real vs Synthetic Controls
=============================================================

Purpose
-------
This script evaluates how well *synthetic control–derived* abnormality calls
match those from *real controls* on the **train** split, across a range of
per-feature z-score thresholds.

At a high level, it:

1. Loads:
   - gen:       synthetic control-equivalent profiles for train
   - control:   real train controls
   - treatment: real train treated animals
2. Rounds selected generated analytes to mimic real-world reporting precision
   (e.g., TBIL to 2 decimals, ALT as integer).
3. For each (COMPOUND_NAME, SACRIFICE_PERIOD) on the REAL side:
   - Uses real controls to compute mean and SD per analyte.
   - Converts High-dose real animals into z-scores.
   - Collapses to treatment-level flags: whether **any** animal is abnormal
     (|z| > 2) for each feature.
4. For each (COMPOUND_NAME, targetTime) on the SYNTHETIC side:
   - Uses synthetic controls to compute mean and SD per analyte.
   - Converts the matching High-dose real animals into z-scores.
   - For thresholds 1..20, collapses to treatment-level flags:
     whether **any** animal exceeds |z| > threshold for each feature.
   - Only groups with at least one valid sample on **both** real and synthetic
     sides for a feature contribute to counts.
5. Builds confusion matrices for each (feature, threshold) by comparing:
   - REAL:   abnormal if |z_real| > 2
   - SYNTH:  abnormal if |z_synth| > threshold
   treating synthetic NA flags as **False** (normal).
6. Outputs:
   - `train_concordance_threshold.csv`: per-feature accuracy for thresholds 1..20
   - `best_per_feature.csv`: best threshold and accuracy per feature.

Paths & assumptions
-------------------
- Update the CSV paths below (`/path/to/...`) to point to your own files.
- Expected columns:
    * real & gen:
        - COMPOUND_NAME
        - SACRIFICE_PERIOD (and 'targetTime' in gen)
        - DOSE_LEVEL (Control / High)
        - INDIVIDUAL_ID
        - 38+ clinical-pathology analytes starting at column index 11.
"""

import pandas as pd
import numpy as np
import math
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kstest, zscore

# =============================================================================
# 1. Load input data (UPDATE PATHS)
# =============================================================================

gen = pd.read_csv(
    "/path/to/generated_predictions_1161985_ControlGenerator_train.csv"
)
control = pd.read_csv(
    "/path/to/repeat_train_control_cv2.csv"
)
treatment = pd.read_csv(
    "/path/to/repeat_train_treatment_cv2.csv"
)

# Combine control + treatment for convenience
real = pd.concat([control, treatment], ignore_index=True, sort=False)

# =============================================================================
# 2. Harmonize generated analytes to match real reporting precision
# =============================================================================

#optimizing the prediction values to be consistent with real 

#gen["TBIL(mg/dL)"] = pd.to_numeric(gen["TBIL(mg/dL)"], errors="coerce").round(2)
gen["RALB(g/dL)"] = pd.to_numeric(gen["RALB(g/dL)"], errors="coerce").round(2)
gen["AST(IU/L)"] = pd.to_numeric(gen["AST(IU/L)"], errors="coerce").round().astype(int)
gen["TP(g/dL)"] = pd.to_numeric(gen["TP(g/dL)"], errors="coerce").round(1)
#gen["CRE(mg/dL)"] = pd.to_numeric(gen["CRE(mg/dL)"], errors="coerce").round(1)
#gen["DBIL(mg/dL)"] = pd.to_numeric(gen["DBIL(mg/dL)"], errors="coerce").round(2)
#gen["BUN(mg/dL)"] = pd.to_numeric(gen["BUN(mg/dL)"], errors="coerce").round().astype(int)
#gen["K(meq/L)"] = pd.to_numeric(gen["K(meq/L)"], errors="coerce").round(2)
#gen["GTP(IU/L)"] = pd.to_numeric(gen["GTP(IU/L)"], errors="coerce").round().astype(int)
gen["Ca(mg/dL)"] = pd.to_numeric(gen["Ca(mg/dL)"], errors="coerce").round(1)
gen["Cl(meq/L)"] = pd.to_numeric(gen["Cl(meq/L)"], errors="coerce").round(1)
#gen["Na(meq/L)"] = pd.to_numeric(gen["Na(meq/L)"], errors="coerce").round(1)
gen["IP(mg/dL)"] = pd.to_numeric(gen["IP(mg/dL)"], errors="coerce").round(1)
gen["ALP(IU/L)"] = pd.to_numeric(gen["ALP(IU/L)"], errors="coerce").round().astype(int)
gen["ALT(IU/L)"] = pd.to_numeric(gen["ALT(IU/L)"], errors="coerce").round().astype(int)
gen["LDH(IU/L)"] = pd.to_numeric(gen["LDH(IU/L)"], errors="coerce").round().astype(int)
gen["RALB(g/dL)"] = pd.to_numeric(gen["RALB(g/dL)"], errors="coerce").round(1)


# =============================================================================
# 3. REAL side: per-sample z-scores vs real controls, then per-treatment calls
# =============================================================================

# 1) Which features to score? (assumes biomarkers start at column index 11)
feature_cols = real.columns[11:]

results = []

# 2) Loop per (compound, time)
for (compound, time), grp in real.groupby(["COMPOUND_NAME", "SACRIFICE_PERIOD"]):
    # 2a) pull the real control rows
    ctrl = grp[grp["DOSE_LEVEL"] == "Control"]
    if ctrl.shape[0] < 2:
        continue

    # 2b) precompute control means & SDs per feature
    ctrl_means = ctrl[feature_cols].mean()
    ctrl_sds = ctrl[feature_cols].std(ddof=1)

    # 3) for each treatment dose (here we only care about High)
    for dose in ["High"]:
        trt = grp[grp["DOSE_LEVEL"] == dose]
        if trt.empty:
            continue

        # 4) compute z-scores for each individual treatment sample
        for _, row in trt.iterrows():
            rec = {
                "compound": compound,
                "time": time,
                "dose": dose,
                "INDIVIDUAL_ID": row["INDIVIDUAL_ID"],
            }
            for f in feature_cols:
                μ = ctrl_means[f]
                σ = ctrl_sds[f]
                x = row[f]

                # cannot compute if SD is zero or missing
                if pd.isna(σ) or σ == 0:
                    rec[f"{f}_z"] = np.nan
                else:
                    rec[f"{f}_z"] = (x - μ) / σ

            results.append(rec)

# 5) Build the final REAL z-score DataFrame
results_df = pd.DataFrame(results)

# -----------------------------------------------------------------------------
# REAL: build NA-aware abnormal flags (|z| > 2) and treatment-level calls
# -----------------------------------------------------------------------------

z_cols = [c for c in results_df.columns if c.endswith("_z")]

# 1) Cell-level abnormal flags, NA-aware
for z in z_cols:
    abn = results_df[z].abs().gt(2)  # True/False; NaN initially treated as False
    abn = abn.astype("boolean")      # nullable boolean dtype
    abn = abn.where(results_df[z].notna(), pd.NA)  # restore NaN as <unknown>
    results_df[f"{z}_abn"] = abn

# 2) Per-feature, per-(compound,time,dose) treatment calls
feature_counts = []
for z in z_cols:
    grp = (
        results_df
        .groupby(["compound", "time", "dose"])[f"{z}_abn"]
        .agg(
            valid_n=lambda s: s.notna().sum(),        # samples with a real z
            abn_any=lambda s: s.any(skipna=True),     # any True among non-NA
        )
        .reset_index()
    )

    # Drop treatments where this feature had no usable z at all (all NA)
    grp = grp[grp["valid_n"] > 0]

    feature_counts.append(
        {
            "feature": z[:-2],  # strip "_z"
            "abnormal": int((grp["abn_any"] == True).sum()),
            "normal": int((grp["abn_any"] == False).sum()),
        }
    )

feature_status_df = pd.DataFrame(feature_counts)
feature_status_df

# =============================================================================
# 4. GEN side: z-scores for real High using synthetic controls (train)
# =============================================================================

# 1) Features to score (reuse same set)
feature_cols = real.columns[11:]

results_gen_z = []

# 2) Loop over each synthetic-control block: (COMPOUND_NAME, targetTime)
for (gen_cmpd, gen_time), gen_ctrl_group in gen.groupby(
    ["COMPOUND_NAME", "targetTime"]
):
    # need at least two synthetic control samples to compute an SD
    if gen_ctrl_group.shape[0] < 2:
        continue

    # 2a) compute synthetic control means & SDs for this compound/time
    ctrl_means = gen_ctrl_group[feature_cols].mean()
    ctrl_sds = gen_ctrl_group[feature_cols].std(ddof=1)

    # 3) for each treatment dose (here High only to match REAL)
    for real_dose in ["High"]:
        real_treat = real.loc[
            (real["COMPOUND_NAME"] == gen_cmpd)
            & (real["SACRIFICE_PERIOD"] == gen_time)
            & (real["DOSE_LEVEL"] == real_dose)
        ]
        if real_treat.empty:
            continue

        # 4) compute z-scores for each individual sample vs synthetic controls
        for _, sample in real_treat.iterrows():
            rec = {
                "compound": gen_cmpd,
                "time": gen_time,
                "dose": real_dose,
                "INDIVIDUAL_ID": sample["INDIVIDUAL_ID"],
            }
            for feat in feature_cols:
                x = sample[feat]
                mu = ctrl_means[feat]
                sigma = ctrl_sds[feat]

                # guard against zero or missing sd
                if pd.isna(x) or pd.isna(mu) or pd.isna(sigma) or sigma == 0:
                    rec[f"{feat}_z"] = np.nan
                else:
                    rec[f"{feat}_z"] = (x - mu) / sigma

            results_gen_z.append(rec)

# 5) Final GEN z-score DataFrame (real High vs synthetic controls)
results_gen_z_df = pd.DataFrame(results_gen_z)

# =============================================================================
# 5. GEN side: threshold sweep 1..20, aligned to REAL validity
# =============================================================================

# Grouping keys shared by both datasets
keys = ["compound", "time", "dose"]

# ----- REAL: valid counts per group/feature -----
z_cols_real = [c for c in results_df.columns if c.endswith("_z")]
abs_real = results_df[z_cols_real].abs()

group_valid_real = abs_real.groupby(
    [results_df[k] for k in keys], observed=True
).count()  # index: group, columns: features

# ----- GEN: max(|z|) and valid counts per group/feature -----
z_cols_gen = [c for c in results_gen_z_df.columns if c.endswith("_z")]
abs_gen = results_gen_z_df[z_cols_gen].abs()

group_max_gen = abs_gen.groupby(
    [results_gen_z_df[k] for k in keys], observed=True
).max()
group_valid_gen = abs_gen.groupby(
    [results_gen_z_df[k] for k in keys], observed=True
).count()

# Restrict to features present on BOTH sides
common = [c for c in z_cols_gen if c in group_valid_real.columns]
group_max_gen = group_max_gen[common]
group_valid_gen = group_valid_gen[common]

# Align REAL valid counts to GEN's group index and common feature columns; fill missing with 0 valid
group_valid_real = (
    group_valid_real
    .reindex(index=group_max_gen.index, fill_value=0)
    .reindex(columns=common, fill_value=0)
)

# ----- Threshold sweep: only include groups valid on BOTH REAL and GEN -----
records = []
for thr in range(1, 21):
    # Eligible if REAL and GEN both have ≥1 non-NaN sample
    valid_mask = (group_valid_real > 0) & (group_valid_gen > 0)

    # Abnormal if max(|z|) > thr, but only where eligible
    abn_mask = (group_max_gen > thr) & valid_mask

    # Counts per feature
    abn_counts = abn_mask.sum(axis=0).astype(int)         # column-wise sums
    total_valid = valid_mask.sum(axis=0).astype(int)
    norm_counts = (total_valid - abn_counts).astype(int)

    records.append(
        pd.DataFrame(
            {
                "feature": [c[:-2] for c in common],  # strip "_z"
                "threshold": thr,
                "abnormal": abn_counts.values,
                "normal": norm_counts.values,
            }
        )
    )

feature_status_gen_df = pd.concat(records, ignore_index=True)
feature_status_gen_df

# =============================================================================
# 6. REAL vs GEN confusion matrices per (feature, threshold)
# =============================================================================

keys = ["compound", "time", "dose"]

# ===== Step 1: REAL flags at fixed threshold 2 (gold standard) =====
z_cols_real = [c for c in results_df.columns if c.endswith("_z")]
abs_real = results_df[z_cols_real].abs()

group_max_real = abs_real.groupby(
    [results_df[k] for k in keys], observed=True
).max()
group_valid_real = abs_real.groupby(
    [results_df[k] for k in keys], observed=True
).count()

# Treatment-level REAL abnormal calls (True/False/NaN) at |z| > 2
abn_real_df = (group_max_real > 2)
abn_real_df = abn_real_df.where(group_valid_real > 0, np.nan)

real_abn_long = (
    abn_real_df.stack()
    .rename("abn_real_any")
    .reset_index()
    .rename(columns={"level_3": "feature"})
)
real_valid_long = (
    group_valid_real.stack()
    .rename("valid_n_real")
    .reset_index()
    .rename(columns={"level_3": "feature"})
)
real_flags_df = real_abn_long.merge(
    real_valid_long, on=keys + ["feature"], how="left"
)
real_flags_df["feature"] = real_flags_df["feature"].str.replace(
    r"_z$", "", regex=True
)
real_flags_df["abn_real_any"] = real_flags_df["abn_real_any"].astype("boolean")

# ===== Step 2: GEN flags for thresholds 1..20 (using max(|z_gen|)) =====
z_cols_gen = [c for c in results_gen_z_df.columns if c.endswith("_z")]
abs_gen = results_gen_z_df[z_cols_gen].abs()

group_max_gen = abs_gen.groupby(
    [results_gen_z_df[k] for k in keys], observed=True
).max()
group_valid_gen = abs_gen.groupby(
    [results_gen_z_df[k] for k in keys], observed=True
).count()

# Restrict GEN to features present on REAL side
common = [c for c in z_cols_gen if c in group_valid_real.columns]
group_max_gen = group_max_gen[common]
group_valid_gen = group_valid_gen[common]

# Align REAL valid counts to GEN index/columns
gvr_aligned = (
    group_valid_real
    .reindex(index=group_max_gen.index, fill_value=0)
    .reindex(columns=common, fill_value=0)
)

# Effective GEN valid counts that respect REAL eligibility (zero where REAL had no data)
group_valid_gen_eff = group_valid_gen.where(gvr_aligned > 0, 0)
valid_gen_long = (
    group_valid_gen_eff.stack()
    .rename("valid_n_gen")
    .reset_index()
    .rename(columns={"level_3": "feature"})
)
valid_gen_long["feature"] = valid_gen_long["feature"].str.replace(
    r"_z$", "", regex=True
)

gen_frames = []
for thr in range(1, 21):
    # Eligible groups = REAL has ≥1 valid sample (only REAL validity enforced)
    elig_real = (gvr_aligned > 0)

    # GEN abnormal if max(|z_gen|) > thr; set <NA> where GEN has no valid AND/OR REAL not eligible
    abn_gen_df = (group_max_gen > thr)
    abn_gen_df = abn_gen_df.where(elig_real & (group_valid_gen > 0), np.nan)

    abn_gen_long = (
        abn_gen_df.stack()
        .rename("abn_gen_any")
        .reset_index()
        .rename(columns={"level_3": "feature"})
    )
    abn_gen_long["feature"] = abn_gen_long["feature"].str.replace(
        r"_z$", "", regex=True
    )
    abn_gen_long["abn_gen_any"] = abn_gen_long["abn_gen_any"].astype("boolean")
    abn_gen_long["threshold"] = thr

    gen_frames.append(
        abn_gen_long.merge(valid_gen_long, on=keys + ["feature"], how="left")
    )

gen_flags_df = pd.concat(gen_frames, ignore_index=True)

# ===== Step 3: Merge REAL vs GEN; keep rows where REAL had ≥1 valid sample =====
merged = (
    real_flags_df
    .merge(gen_flags_df, on=["compound", "time", "dose", "feature"], how="inner")
    .query("valid_n_real > 0")  # enforce REAL validity only
)

# ===== Step 4: Confusion per (feature, threshold), treating GEN NA as False =====
confusion = []
for (feat, thr), grp in merged.groupby(["feature", "threshold"]):
    r = grp["abn_real_any"].astype(bool)
    g = grp["abn_gen_any"].fillna(False).astype(bool)  # GEN NA -> False

    TP = int((r & g).sum())
    TN = int((~r & ~g).sum())
    FP = int((~r & g).sum())
    FN = int((r & ~g).sum())

    # How many GEN NAs were filled for this (feature, threshold)
    na_filled = int(grp["abn_gen_any"].isna().sum())

    confusion.append(
        {
            "feature": feat,
            "threshold": thr,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "na_filled_gen": na_filled,
            "N_pairs": int(len(grp)),
        }
    )

confusion_df = pd.DataFrame(confusion)

# =============================================================================
# 7. Accuracy per (feature, threshold) + best threshold per feature
# =============================================================================

# 1) Total usable pairs per (feature, threshold)
confusion_df["total"] = (
    confusion_df["TP"]
    + confusion_df["TN"]
    + confusion_df["FP"]
    + confusion_df["FN"]
)

# 2) Accuracy = (TP + TN) / total
confusion_df["accuracy"] = (
    (confusion_df["TP"] + confusion_df["TN"]) / confusion_df["total"]
)

# 3) Optionally drop raw 'total' if not needed downstream
confusion_df = confusion_df.drop(columns=["total"])

# Pivot to wide form: rows = features, columns = thresholds
accuracy_wide = confusion_df.pivot(
    index="feature",
    columns="threshold",
    values="accuracy",
)

# Rename columns to accuracy_1, accuracy_2, ..., accuracy_20
accuracy_wide.columns = [f"accuracy_{thr}" for thr in accuracy_wide.columns]

# Return 'feature' as a column instead of index
accuracy_wide = accuracy_wide.reset_index()

# Save threshold vs accuracy grid (UPDATE PATH)
accuracy_wide.to_csv(
    "/path/to/train_concordance_threshold.csv",
    index=False,
)

# Identify best threshold per feature (max accuracy)
idx = confusion_df.groupby("feature")["accuracy"].idxmax()

best_per_feature = (
    confusion_df.loc[idx, ["feature", "threshold", "accuracy"]]
    .rename(
        columns={
            "threshold": "best_threshold",
            "accuracy": "best_accuracy",
        }
    )
    .reset_index(drop=True)
)

# Save best threshold summary (UPDATE PATH)
best_per_feature.to_csv(
    "/path/to/best_per_feature.csv",
    index=False,
)
