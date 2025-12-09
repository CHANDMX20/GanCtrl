"""
Abnormal-Elevation Classification: Real vs Synthetic Controls
============================================================

Purpose
-------
This script compares **real** and **synthetic** elevation calling at the
per-(compound, time, dose, feature) level using z-scores and
feature-specific abnormality thresholds.

High-level workflow
-------------------
1. Load:
   - gen:       generated control-equivalent profiles (e.g., from a VAE/GAN)
   - control:   real control animals
   - treatment: real treated animals
   - threshold: per-feature best z-score threshold (column: 'best_threshold')

2. Harmonize selected generated analytes (rounding to match real assay
   reporting: e.g. TBIL to 2 decimals, ALT as integer, etc.).

3. **Real side** (control → High):
   - For each (COMPOUND_NAME, SACRIFICE_PERIOD):
       * Compute control means and SDs (using real controls only).
       * Convert High-dose real animals to z-scores.
       * Mark cell-level abnormalities where |z| > 2.
       * Collapse to treatment-level flags: for each
         (compound, time, dose, feature), whether **any** animal is abnormal.
   - Produce `feature_status_df` with treatment-level abnormal vs normal counts
     per feature on the **real** side.

4. **Synthetic side** (synthetic control → real High):
   - For each (COMPOUND_NAME, targetTime) in generated controls:
       * Compute means and SDs from **synthetic controls**.
       * Convert the corresponding High-dose real animals to z-scores
         using synthetic means/SDs.
   - Use `threshold` table to retrieve per-feature best z-score thresholds
     (defaulting to 2 when a feature is missing).
   - Mark cell-level abnormalities where |z| > best_threshold(feature).
   - Collapse to treatment-level flags aligned to real validity, yielding
     `feature_status_gen_df`.

5. Combine real vs synthetic flags:
   - For each feature, build a confusion matrix by comparing:
       * REAL: treatment-level abnormality (|z_real| > 2)
       * GEN:  treatment-level abnormality (|z_gen| > best_threshold(feature))
   - Treat generated NA flags as **False** (normal) when computing
     TP, TN, FP, FN.
   - Compute per-feature accuracy: (TP + TN) / (TP + TN + FP + FN).

6. Outputs:
   - `feature_status_df`: real-side abnormal/normal treatment counts
   - `feature_status_gen_df`: synthetic-side abnormal/normal treatment counts
   - `confusion_df`: per-feature TP, TN, FP, FN, na_filled_gen, accuracy

Paths & assumptions
-------------------
- Update the CSV paths below to match your environment (placeholders used here).
- Expected columns:
    * real/gen:
        - COMPOUND_NAME
        - SACRIFICE_PERIOD (and 'targetTime' in gen)
        - DOSE_LEVEL (Control / High)
        - INDIVIDUAL_ID
        - 38 clinical-pathology analytes starting at column index 11
    * threshold:
        - 'feature'
        - 'best_threshold'
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
    "/path/to/generated_predictions_1161985_ControlGenerator_test.csv"
)
control = pd.read_csv(
    "/path/to/repeat_test_control_cv2.csv"
)
treatment = pd.read_csv(
    "/path/to/repeat_test_treatment_cv2.csv"
)
real = pd.concat([control, treatment], ignore_index=True, sort=False)

threshold = pd.read_csv(
    "/path/to/best_per_feature.csv"
)

# =============================================================================
# 2. Harmonize / round selected generated analytes to mimic real reporting
# =============================================================================

gen["TBIL(mg/dL)"] = pd.to_numeric(gen["TBIL(mg/dL)"], errors="coerce").round(2)
gen["RALB(g/dL)"] = pd.to_numeric(gen["RALB(g/dL)"], errors="coerce").round(2)
gen["AST(IU/L)"]  = pd.to_numeric(gen["AST(IU/L)"],  errors="coerce").round().astype(int)
gen["TP(g/dL)"]   = pd.to_numeric(gen["TP(g/dL)"],   errors="coerce").round(1)
# gen["CRE(mg/dL)"] = pd.to_numeric(gen["CRE(mg/dL)"], errors="coerce").round(1)
gen["DBIL(mg/dL)"] = pd.to_numeric(gen["DBIL(mg/dL)"], errors="coerce").round(2)
gen["BUN(mg/dL)"]  = pd.to_numeric(gen["BUN(mg/dL)"],  errors="coerce").round().astype(int)
# gen["K(meq/L)"]   = pd.to_numeric(gen["K(meq/L)"],   errors="coerce").round(2)
# gen["GTP(IU/L)"]  = pd.to_numeric(gen["GTP(IU/L)"],  errors="coerce").round().astype(int)
# gen["Ca(mg/dL)"]  = pd.to_numeric(gen["Ca(mg/dL)"],  errors="coerce").round(1)
# gen["Cl(meq/L)"]  = pd.to_numeric(gen["Cl(meq/L)"],  errors="coerce").round(1)
# gen["Na(meq/L)"]  = pd.to_numeric(gen["Na(meq/L)"],  errors="coerce").round(1)
# gen["IP(mg/dL)"]  = pd.to_numeric(gen["IP(mg/dL)"],  errors="coerce").round(1)
gen["ALP(IU/L)"]   = pd.to_numeric(gen["ALP(IU/L)"],   errors="coerce").round().astype(int)
gen["ALT(IU/L)"]   = pd.to_numeric(gen["ALT(IU/L)"],   errors="coerce").round().astype(int)
gen["LDH(IU/L)"]   = pd.to_numeric(gen["LDH(IU/L)"],   errors="coerce").round().astype(int)
gen["RALB(g/dL)"]  = pd.to_numeric(gen["RALB(g/dL)"],  errors="coerce").round(1)

# =============================================================================
# 3. REAL side: per-treatment z-scores vs real controls and abnormal counts
# =============================================================================

# 1) Which features to score? (assumes biomarker columns start at index 11)
feature_cols = real.columns[11:]

results = []

# 2) Loop per (compound, time)
for (compound, time), grp in real.groupby(["COMPOUND_NAME", "SACRIFICE_PERIOD"]):
    # 2a) pull the control rows
    ctrl = grp[grp["DOSE_LEVEL"] == "Control"]
    if ctrl.shape[0] < 2:
        continue

    # 2b) precompute control means & SDs per feature
    ctrl_means = ctrl[feature_cols].mean()
    ctrl_sds = ctrl[feature_cols].std(ddof=1)

    # 3) for each treatment dose (here: High only)
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

# 5) Build the final DataFrame of real-side z-scores
results_df = pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Collapse real z-scores to treatment-level abnormal vs normal counts
# -----------------------------------------------------------------------------

# 1) Identify z-score columns
z_cols = [col for col in results_df.columns if col.endswith("_z")]
keys = ["compound", "time", "dose"]

# 2) Container
feature_counts = []

# 3) Per-feature (NA-aware) counting
for z in z_cols:
    s = results_df[z]

    # cell-level abnormal (strict |z| > 2), keep NaN as NA (nullable boolean)
    abn = s.abs().gt(2)
    abn = abn.where(s.notna(), pd.NA).astype("boolean")

    # group to treatment-level: any True among non-NA; also track usable count
    grp = (
        pd.concat([results_df[keys], abn.rename("abn")], axis=1)
        .groupby(keys)["abn"]
        .agg(
            valid_n=lambda x: x.notna().sum(),
            abn_any=lambda x: x.any(skipna=True),
        )
        .reset_index()
    )

    # exclude treatments where this feature had no usable value
    grp = grp[grp["valid_n"] > 0]

    # counts
    abn_count = int((grp["abn_any"] == True).sum())
    norm_count = int((grp["abn_any"] == False).sum())

    feature_counts.append(
        {
            "feature": z[:-2],  # strip '_z'
            "abnormal": abn_count,
            "normal": norm_count,
        }
    )

# 4) Result table: real-side feature-level abnormal vs normal treatments
feature_status_df = pd.DataFrame(feature_counts)

feature_status_df

# =============================================================================
# 4. GEN side: z-scores for real High using synthetic controls + thresholds
# =============================================================================

# 1) Which features to score? (re-using the same columns)
feature_cols = real.columns[11:]

results_gen_z = []

# 2) Loop over each synthetic-control block (compound, targetTime)
for (gen_cmpd, gen_time), gen_ctrl_group in gen.groupby(
    ["COMPOUND_NAME", "targetTime"]
):
    # need at least two control samples to compute an SD
    if gen_ctrl_group.shape[0] < 2:
        continue

    # 2a) compute control means & SDs for this compound/time (synthetic controls)
    ctrl_means = gen_ctrl_group[feature_cols].mean()
    ctrl_sds = gen_ctrl_group[feature_cols].std(ddof=1)

    # 3) for each treatment dose (here High only)
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

# 5) assemble into a DataFrame: real High vs synthetic controls (z-scores)
results_gen_z_df = pd.DataFrame(results_gen_z)

# =============================================================================
# 5. Thresholds and GEN abnormal/normal counts (aligned with REAL validity)
# =============================================================================

# Map: feature -> best_threshold (default to 2 if absent)
best_thresh = threshold.set_index("feature")["best_threshold"].to_dict()

z_cols_gen = [c for c in results_gen_z_df.columns if c.endswith("_z")]
keys = ["compound", "time", "dose"]

feature_counts_gen = []

for z in z_cols_gen:
    feat = z[:-2]
    thr = best_thresh.get(feat, 2)

    # Skip if this feature isn't present on the REAL side
    if z not in results_df.columns:
        continue

    # --- GEN: cell-level abnormal (NA-aware, using feature-specific threshold)
    s_gen = results_gen_z_df[z]
    abn = s_gen.abs().gt(thr)
    abn = abn.where(s_gen.notna(), pd.NA).astype("boolean")

    grp_gen = (
        pd.concat([results_gen_z_df[keys], abn.rename("abn_gen")], axis=1)
        .groupby(keys)["abn_gen"]
        .agg(
            valid_n_gen=lambda x: x.notna().sum(),
            abn_gen_any=lambda x: x.any(skipna=True),
        )
        .reset_index()
    )

    # --- REAL: per-group valid counts for the same feature
    grp_real_valid = (
        results_df[keys + [z]]
        .groupby(keys, as_index=False)[z]
        .count()
        .rename(columns={z: "valid_n_real"})
    )

    # --- Intersect validity: keep only groups valid on BOTH sides
    merged = grp_gen.merge(grp_real_valid, on=keys, how="left")
    merged["valid_n_real"] = merged["valid_n_real"].fillna(0).astype(int)
    eligible = merged[(merged["valid_n_real"] > 0) & (merged["valid_n_gen"] > 0)]

    # Counts among eligible groups
    abn_count = int((eligible["abn_gen_any"] == True).sum())
    norm_count = int((eligible["abn_gen_any"] == False).sum())

    feature_counts_gen.append(
        {
            "feature": feat,
            "threshold": thr,
            "abnormal": abn_count,
            "normal": norm_count,
        }
    )

feature_status_gen_df = pd.DataFrame(feature_counts_gen)

feature_status_gen_df

# =============================================================================
# 6. Build confusion matrices: REAL vs GEN abnormal flags
# =============================================================================

# keys shared by both frames
keys = ["compound", "time", "dose"]

# ── 1) REAL flags (NA-aware; |z| > 2, collapsed to treatment-level) ──
real_chunks = []
for z in [c for c in results_df.columns if c.endswith("_z")]:
    feat = z[:-2]
    s = results_df[z]

    abn = s.abs().gt(2)
    abn = abn.where(s.notna(), pd.NA).astype("boolean")

    grp = (
        pd.concat([results_df[keys], abn.rename("abn_real")], axis=1)
        .groupby(keys)["abn_real"]
        .agg(
            valid_n_real=lambda x: x.notna().sum(),
            abn_real_any=lambda x: x.any(skipna=True),
        )
        .reset_index()
    )
    # only keep groups where REAL had ≥1 usable value for this feature
    grp = grp[grp["valid_n_real"] > 0]
    grp["feature"] = feat
    real_chunks.append(grp)

real_flags_df = pd.concat(real_chunks, ignore_index=True)

# ── 2) GEN flags (NA-aware; feature-specific threshold; aligned to REAL validity) ──
gen_chunks = []
for z in [c for c in results_gen_z_df.columns if c.endswith("_z")]:
    feat = z[:-2]
    thr = best_thresh.get(feat, 2)

    s = results_gen_z_df[z]
    abn = s.abs().gt(thr)
    abn = abn.where(s.notna(), pd.NA).astype("boolean")

    grp = (
        pd.concat([results_gen_z_df[keys], abn.rename("abn_gen")], axis=1)
        .groupby(keys)["abn_gen"]
        .agg(
            valid_n_gen=lambda x: x.notna().sum(),
            abn_gen_any=lambda x: x.any(skipna=True),
        )
        .reset_index()
    )
    # DO NOT drop gen groups just because gen had all-NaN;
    # instead mark abn_gen_any as <NA> when valid_n_gen == 0
    grp.loc[grp["valid_n_gen"] == 0, "abn_gen_any"] = pd.NA
    grp["abn_gen_any"] = grp["abn_gen_any"].astype("boolean")

    grp["feature"] = feat
    grp["threshold"] = thr
    gen_chunks.append(grp)

gen_flags_df = pd.concat(gen_chunks, ignore_index=True)

# ── 3) Merge, keeping rows where REAL had ≥1 valid sample.
#     (GEN may be <NA>; those rows won’t contribute to TP/TN/FP/FN
#      unless you explicitly tally NA mismatches.)
merged_flags = (
    real_flags_df.merge(gen_flags_df, on=keys + ["feature"], how="inner")
    .query("valid_n_real > 0")  # <- ONLY real validity enforced
    # (optional) if you want to also require gen validity, add: and valid_n_gen > 0
)

# ── 4) Confusion per feature (treat gen NA as False/normal) ──
confusion = []
for feat, grp in merged_flags.groupby("feature"):
    grp2 = grp.copy()
    # Fill NA on generated side with False (normal)
    grp2["abn_gen_any_filled"] = grp2["abn_gen_any"].fillna(False)

    r = grp2["abn_real_any"].astype(bool)          # REAL abnormal flag
    g = grp2["abn_gen_any_filled"].astype(bool)    # GEN abnormal flag with NA -> False

    TP = int((r & g).sum())        # real abnormal and gen abnormal
    TN = int((~r & ~g).sum())      # real normal and gen normal
    FP = int((~r & g).sum())       # real normal, gen abnormal
    FN = int((r & ~g).sum())       # real abnormal, gen normal

    # how many gen NAs were filled for this feature
    na_filled = int(grp2["abn_gen_any"].isna().sum())

    confusion.append(
        {
            "feature": feat,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "na_filled_gen": na_filled,  # optional diagnostic column
        }
    )

confusion_df = pd.DataFrame(confusion)

# -----------------------------------------------------------------------------
# Compute per-feature accuracy
# -----------------------------------------------------------------------------

# 1) Compute total cases per feature
confusion_df["total"] = (
    confusion_df["TP"]
    + confusion_df["TN"]
    + confusion_df["FP"]
    + confusion_df["FN"]
)

# 2) Compute accuracy = (TP + TN) / total
confusion_df["accuracy"] = (
    (confusion_df["TP"] + confusion_df["TN"]) / confusion_df["total"]
)

# 3) (Optional) drop the 'total' column if you don’t need it downstream
confusion_df = confusion_df.drop(columns=["total"])

confusion_df
