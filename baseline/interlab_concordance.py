"""
Abnormality scoring and agreement analysis for clinical pathology features.

Pipeline overview
-----------------
1. Load per-animal clinical pathology data (with DOSE_LEVEL, COMPOUND_NAME, etc.)
2. Load baseline metadata table (treatment group info: Vehicle, Lab, etc.)
3. Clean and normalize compound names in the metadata table.
4. Merge metadata into control and treatment data, reorder columns for readability.
5. Compute per-sample z-scores using within-compound controls (Control vs High).
6. Summarize, per feature, how often treatments are "abnormal" (|z| > 2).
7. Compute matched-control z-scores using cross-compound controls
   (same time, same vehicle, different compound & lab).
8. Summarize "abnormal" rates in this matched-control (generated) setting.
9. Build per-treatment abnormality flags for real vs generated z-scores.
10. Compute TP/TN/FP/FN per feature and derive per-feature accuracy.

Note:
-----
- Paths below are placeholders; update them to match your environment.
- This script assumes certain column names exist:
  COMPOUND_NAME, DOSE_LEVEL, SACRIFICE_PERIOD, INDIVIDUAL_ID, etc.
"""

import pandas as pd
import numpy as np

# =============================================================================
# 0. FILE PATHS (UPDATE THESE FOR YOUR ENVIRONMENT)
# =============================================================================

CLIN_PATH_FILE = "/path/to/clin_path/repeat_merged.csv"
BASELINE_META_FILE = "/path/to/baseline/metadata.csv"

# =============================================================================
# 1. LOAD DATA
# =============================================================================

data = pd.read_csv(CLIN_PATH_FILE)
tgp = pd.read_csv(BASELINE_META_FILE)

# Select dose groups of interest
control = data[data["DOSE_LEVEL"] == "Control"]
treatment = data[data["DOSE_LEVEL"] == "High"]

# Keep only columns common to both control and treatment
common_cols = control.columns.intersection(treatment.columns)

control_labeled = control[common_cols].copy()
treatment_labeled = treatment[common_cols].copy()

# Combined dataset for within-compound control vs high-dose z-scores
combined = pd.concat([control_labeled, treatment_labeled], ignore_index=True)

# =============================================================================
# 2. CLEAN & NORMALIZE COMPOUND NAMES IN BASELINE TABLE
# =============================================================================

# Standardize naming for lab and compound columns
tgp = tgp.rename(columns={"Test Facility (in vivo animal treatment)": "Lab"})
tgp = tgp.rename(columns={"Compound name (E)": "COMPOUND_NAME"})

# Drop rows with missing metadata entirely
tgp = tgp.dropna()

# Lowercase the first non-space character in COMPOUND_NAME
s = tgp["COMPOUND_NAME"].astype("string")
mask = s.notna()
tgp.loc[mask, "COMPOUND_NAME"] = s[mask].str.replace(
    r"^(\s*)(\S)",
    lambda m: m.group(1) + m.group(2).lower(),
    regex=True,
)

# Compound name harmonization (manual mapping)
tgp["COMPOUND_NAME"] = tgp["COMPOUND_NAME"].replace(
    {
        "2-acetamidofluorene": "acetamidofluorene",
        "chlorpheniramine maleate": "chlorpheniramine",
        "clomipramine hydrochloride": "clomipramine",
        "danazol ": "danazol",
        "dantrolene sodium hemiheptahydrate": "dantrolene",
        "diclofenac sodium salt": "diclofenac",
        "erythromycin": "erythromycin ethylsuccinate",
        "fultamide": "flutamide",
        "glybenclamide": "glibenclamide",
        "hydroxyzine dihydrochloride": "hydroxyzine",
        "labetalol hydrochloride": "labetalol",
        "methapyrilene hydrochloride": "methapyrilene",
        "alpha-metyldopa": "methyldopa",
        "alpha-naphthylisothiocyanate": "naphthyl isothiocyanate",
        "n-nitrosodiethylamine": "nitrosodiethylamine",
        "n-phenylanthranilic acid": "phenylanthranilic acid",
        "tacrine hydrochloride": "tacrine",
        "chlormadinone acetate": "chlormadinone",
        "enalapril maleate": "enalapril",
        "iproniazid phosphate salt": "iproniazid",
    }
)

# One row per compound: (COMPOUND_NAME, Vehicle, Lab)
tgp_sub = (
    tgp[["COMPOUND_NAME", "Vehicle", "Lab"]]
    .drop_duplicates(subset=["COMPOUND_NAME"], keep="first")
)

# =============================================================================
# 3. MERGE METADATA INTO CONTROL & TREATMENT, REORDER COLUMNS
# =============================================================================

# --- CONTROL ---
control_new = control.merge(
    tgp_sub,
    on="COMPOUND_NAME",
    how="left",
    validate="m:1",
)

cols = control_new.columns.tolist()
for c in ["Vehicle", "Lab"]:
    if c in cols:
        cols.remove(c)
insert_at = min(11, len(cols))  # insert around the 12th column if possible
new_order = cols[:insert_at] + ["Vehicle", "Lab"] + cols[insert_at:]
control_new = control_new[new_order]

# --- TREATMENT ---
treatment_new = treatment.merge(
    tgp_sub,
    on="COMPOUND_NAME",
    how="left",
    validate="m:1",
)

cols2 = treatment_new.columns.tolist()
for c2 in ["Vehicle", "Lab"]:
    if c2 in cols2:
        cols2.remove(c2)
insert_at = min(11, len(cols2))
new_order2 = cols2[:insert_at] + ["Vehicle", "Lab"] + cols2[insert_at:]
treatment_new = treatment_new[new_order2]

# Drop rows with any missing values after merges
control_new = control_new.dropna()
treatment_new = treatment_new.dropna()

# =============================================================================
# 4. FILTER COMBINED DATA TO COMPOUNDS WITH VALID CONTROLS
# =============================================================================

combined_comp_col = (
    "COMPOUND_NAME" if "COMPOUND_NAME" in combined.columns else "Compound name"
)
control_comp_col = (
    "COMPOUND_NAME" if "COMPOUND_NAME" in control_new.columns else "Compound name"
)

valid_compounds = control_new[control_comp_col].dropna().unique()
combined = combined[combined[combined_comp_col].isin(valid_compounds)].copy()

# Features to score: columns 11 onward (assumes first 11 columns are ID/meta)
feature_cols = combined.columns[11:]

# =============================================================================
# 5. WITHIN-COMPOUND Z-SCORES (CONTROL VS HIGH DOSE)
# =============================================================================

results = []

for (compound, time), grp in combined.groupby(
    ["COMPOUND_NAME", "SACRIFICE_PERIOD"],
    dropna=False,
):
    # Control rows for this (compound, time)
    ctrl = grp[grp["DOSE_LEVEL"] == "Control"]
    if ctrl.shape[0] < 2:
        # Need >= 2 controls for a usable SD
        continue

    # Per-feature control means and SDs
    ctrl_means = ctrl[feature_cols].mean()
    ctrl_sds = ctrl[feature_cols].std(ddof=1)

    # Treatment rows (High dose)
    trt = grp[grp["DOSE_LEVEL"] == "High"]
    if trt.empty:
        continue

    # Per-sample z-scores
    for _, row in trt.iterrows():
        rec = {
            "compound": compound,
            "time": time,
            "dose": "High",
            "INDIVIDUAL_ID": row["INDIVIDUAL_ID"],
        }
        for f in feature_cols:
            mu = ctrl_means[f]
            sd = ctrl_sds[f]
            x = row[f]
            rec[f"{f}_z"] = np.nan if (pd.isna(sd) or sd == 0) else (x - mu) / sd

        results.append(rec)

results_df = pd.DataFrame(results)

# =============================================================================
# 6. ABNORMAL COUNTS PER FEATURE (REAL DATA, WITHIN-COMPOUND)
# =============================================================================

z_cols = [col for col in results_df.columns if col.endswith("_z")]
keys = ["compound", "time", "dose"]

feature_counts = []

for z in z_cols:
    s = results_df[z]

    # Cell-level abnormal flag: |z| > 2 (NaN preserved as NA)
    abn = s.abs().gt(2)
    abn = abn.where(s.notna(), pd.NA).astype("boolean")

    # Collapse to treatment level
    grp = (
        pd.concat([results_df[keys], abn.rename("abn")], axis=1)
        .groupby(keys)["abn"]
        .agg(
            valid_n=lambda x: x.notna().sum(),
            abn_any=lambda x: x.any(skipna=True),
        )
        .reset_index()
    )

    # Exclude treatments with no usable values for this feature
    grp = grp[grp["valid_n"] > 0]

    # Count abnormal vs normal treatments
    abn_count = int((grp["abn_any"] == True).sum())
    norm_count = int((grp["abn_any"] == False).sum())

    feature_counts.append(
        {"feature": z[:-2], "abnormal": abn_count, "normal": norm_count}
    )

feature_status_df = pd.DataFrame(feature_counts)

# =============================================================================
# 7. MATCHED-CONTROL Z-SCORES (CROSS-COMPOUND CONTROLS)
# =============================================================================

# Ensure features are numeric
control_new[feature_cols] = control_new[feature_cols].apply(
    pd.to_numeric, errors="coerce"
)
treatment_new[feature_cols] = treatment_new[feature_cols].apply(
    pd.to_numeric, errors="coerce"
)

comp_col = "COMPOUND_NAME"
time_col = "SACRIFICE_PERIOD"
veh_col = "Vehicle" if "Vehicle" in control_new.columns else "vehicle"
lab_col = "Lab" if "Lab" in control_new.columns else "lab"

rows = []

for (compound, time), grp in treatment_new.groupby(
    [comp_col, time_col],
    dropna=False,
):
    for _, row in grp.iterrows():
        v = row[veh_col] if veh_col in treatment_new.columns else np.nan
        l = row[lab_col] if lab_col in treatment_new.columns else np.nan

        # Matched controls:
        # same time, same vehicle, different compound, different lab
        ctrl_mask = (
            (control_new[time_col] == time)
            & (control_new[comp_col] != compound)
            & (control_new[veh_col] == v)
            & (control_new[lab_col] != l)
        )
        ctrl = control_new.loc[ctrl_mask, feature_cols]

        rec = {"compound": compound, "time": time}
        if "DOSE_LEVEL" in treatment_new.columns:
            rec["dose"] = row["DOSE_LEVEL"]
        if "INDIVIDUAL_ID" in treatment_new.columns:
            rec["INDIVIDUAL_ID"] = row["INDIVIDUAL_ID"]

        if ctrl.shape[0] < 2:
            # Not enough matched controls; all z-scores NaN
            for f in feature_cols:
                rec[f"{f}_z"] = np.nan
            rows.append(rec)
            continue

        means = ctrl.mean()
        stds = ctrl.std(ddof=1)

        for f in feature_cols:
            x = row[f]
            mu = means[f]
            sd = stds[f]
            rec[f"{f}_z"] = np.nan if (pd.isna(sd) or sd == 0) else (x - mu) / sd

        rows.append(rec)

results_inter_df = pd.DataFrame(rows)

# =============================================================================
# 8. ABNORMAL COUNTS PER FEATURE (GENERATED / MATCHED-CONTROL Z-SCORES)
# =============================================================================

z_cols_inter = [col for col in results_inter_df.columns if col.endswith("_z")]
keys = ["compound", "time", "dose"]

feature_counts_inter = []

for z in z_cols_inter:
    s = results_inter_df[z]

    abn = s.abs().gt(2)
    abn = abn.where(s.notna(), pd.NA).astype("boolean")

    grp = (
        pd.concat([results_inter_df[keys], abn.rename("abn")], axis=1)
        .groupby(keys)["abn"]
        .agg(
            valid_n=lambda x: x.notna().sum(),
            abn_any=lambda x: x.any(skipna=True),
        )
        .reset_index()
    )

    grp = grp[grp["valid_n"] > 0]

    abn_count = int((grp["abn_any"] == True).sum())
    norm_count = int((grp["abn_any"] == False).sum())

    feature_counts_inter.append(
        {"feature": z[:-2], "abnormal": abn_count, "normal": norm_count}
    )

feature_status_inter_df = pd.DataFrame(feature_counts_inter)

# =============================================================================
# 9. BUILD TREATMENT-LEVEL ABNORMAL FLAGS (REAL VS GENERATED)
# =============================================================================

# Real data flags
real_flags = []
for z in [c for c in results_df.columns if c.endswith("_z")]:
    feat = z[:-2]
    tmp = results_df[["compound", "time", "dose", z]].copy()
    tmp["abn_real"] = tmp[z].abs() > 2
    grp = (
        tmp.groupby(["compound", "time", "dose"])["abn_real"]
        .any()
        .reset_index()
    )
    grp["feature"] = feat
    real_flags.append(grp)

real_flags_df = pd.concat(real_flags, ignore_index=True)

# Generated / matched-control flags
inter_flags = []
for z in [c for c in results_inter_df.columns if c.endswith("_z")]:
    feat = z[:-2]
    tmp = results_inter_df[["compound", "time", "dose", z]].copy()
    tmp["abn_gen"] = tmp[z].abs() > 2
    grp = (
        tmp.groupby(["compound", "time", "dose"])["abn_gen"]
        .any()
        .reset_index()
    )
    grp["feature"] = feat
    inter_flags.append(grp)

inter_flags_df = pd.concat(inter_flags, ignore_index=True)

# Merge real vs generated flags
merged_flags = pd.merge(
    real_flags_df,
    inter_flags_df,
    on=["compound", "time", "dose", "feature"],
    how="inner",
)

# =============================================================================
# 10. CONFUSION COUNTS & ACCURACY PER FEATURE
# =============================================================================

confusion = []

for feat, grp in merged_flags.groupby("feature"):
    real_abn = grp["abn_real"]
    gen_abn = grp["abn_gen"]

    # NOTE: by naming here,
    # TP = both normal, TN = both abnormal
    # (keep as-is if this matches your original interpretation)
    TP = ((~real_abn) & (~gen_abn)).sum()  # both normal
    TN = (real_abn & gen_abn).sum()        # both abnormal
    FP = (real_abn & (~gen_abn)).sum()     # real abnormal, generated normal
    FN = ((~real_abn) & gen_abn).sum()     # real normal, generated abnormal

    confusion.append(
        {"feature": feat, "TP": TP, "TN": TN, "FP": FP, "FN": FN}
    )

confusion_df = pd.DataFrame(confusion)

# Total cases and accuracy per feature
confusion_df["total"] = (
    confusion_df["TP"]
    + confusion_df["TN"]
    + confusion_df["FP"]
    + confusion_df["FN"]
)

confusion_df["accuracy"] = (
    (confusion_df["TP"] + confusion_df["TN"]) / confusion_df["total"]
)

# Drop helper column if not needed
confusion_df = confusion_df.drop(columns=["total"])

# Final output
print(confusion_df)
