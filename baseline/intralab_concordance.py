"""
Abnormality scoring and agreement analysis for clinical pathology features --> calculation of intra-laboratory baseline

Pipeline overview
-----------------
1. Load per-animal clinical pathology data (with DOSE_LEVEL, COMPOUND_NAME, etc.).
2. Load baseline metadata table (treatment group info: Vehicle, Lab, etc.).
3. Clean and normalize compound names in the metadata table.
4. Merge metadata into control and treatment data, reorder columns for readability.
5. Compute per-sample z-scores using within-compound controls (Control vs High)
   for each (COMPOUND_NAME, SACRIFICE_PERIOD).
6. Summarize, per feature, how often treatments are "abnormal" (|z| > 2) in this
   within-compound comparison.
7. Compute matched-control z-scores using same-lab, same-vehicle, different-compound
   controls at each (SACRIFICE_PERIOD).
8. Summarize "abnormal" rates in this intra-lab matched-control setting.
9. Build per-treatment abnormality flags for real vs generated z-scores.
10. Compute TP/TN/FP/FN per feature and derive per-feature accuracy.

Note:
-----
- Paths below are placeholders; update them to match your environment.
- This script assumes certain column names exist:
  COMPOUND_NAME, DOSE_LEVEL, SACRIFICE_PERIOD, INDIVIDUAL_ID, Vehicle, Lab, etc.
"""

import pandas as pd
import numpy as np
import math 
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kstest, zscore

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

control = data[data['DOSE_LEVEL'] == 'Control']
treatment = data[data['DOSE_LEVEL'] == 'High']

# 1) Ensure we only concat common columns (avoids surprises if cols differ)
common_cols = control.columns.intersection(treatment.columns)

# 2) Optional helper label (since DOSE_LEVEL already encodes group, this is just alignment)
control_labeled   = control[common_cols].copy()
treatment_labeled = treatment[common_cols].copy()

# 3) Combined dataset for within-compound Control vs High comparisons
combined = pd.concat([control_labeled, treatment_labeled], ignore_index=True)

# =============================================================================
# 2. CLEAN & NORMALIZE COMPOUND NAMES IN BASELINE TABLE
# =============================================================================

# Standardize naming for lab and compound columns
tgp = tgp.rename(columns={"Test Facility (in vivo animal treatment)": "Lab"})
tgp = tgp.rename(columns={"Compound name (E)": "COMPOUND_NAME"})
tgp = tgp.dropna()

s = tgp['COMPOUND_NAME'].astype("string")
mask = s.notna()
tgp.loc[mask, 'COMPOUND_NAME'] = s[mask].str.replace(
    r"^(\s*)(\S)", lambda m: m.group(1) + m.group(2).lower(), regex=True
)

# Compound name harmonization (manual mapping)
tgp["COMPOUND_NAME"] = tgp["COMPOUND_NAME"].replace({
    "2-acetamidofluorene": "acetamidofluorene",
    "chlorpheniramine maleate": "chlorpheniramine",
    "clomipramine hydrochloride": "clomipramine",
    "danazol ": "danazol",
    "dantrolene sodium hemiheptahydrate": "dantrolene",
    "diclofenac sodium salt" : "diclofenac",
    "erythromycin" : "erythromycin ethylsuccinate",
    "fultamide" : "flutamide",
    "glybenclamide" : "glibenclamide",
    "hydroxyzine dihydrochloride" : "hydroxyzine",
    "labetalol hydrochloride": "labetalol",
    "methapyrilene hydrochloride" : "methapyrilene",
    "alpha-metyldopa":"methyldopa",
    "alpha-naphthylisothiocyanate": "naphthyl isothiocyanate",
    "n-nitrosodiethylamine" : "nitrosodiethylamine",
    "n-phenylanthranilic acid" : "phenylanthranilic acid",
    "tacrine hydrochloride" : "tacrine",
    "chlormadinone acetate": "chlormadinone",
    "enalapril maleate": "enalapril",
    "iproniazid phosphate salt": "iproniazid"
})

# --- 0) Make a de-duplicated lookup from tgp (one row per compound) ---
tgp_sub = (
    tgp[["COMPOUND_NAME", "Vehicle", "Lab"]]
    .drop_duplicates(subset=["COMPOUND_NAME"], keep="first")
)

# =============================================================================
# 3. MERGE METADATA INTO CONTROL & TREATMENT, REORDER COLUMNS
# =============================================================================

# =========================
# CONTROL: merge + reorder
# =========================

# 1) Left-join on Compound name (many control rows to one tgp row per compound)
control_new = control.merge(tgp_sub, on="COMPOUND_NAME", how="left", validate="m:1")

# 3) Move 'Vehicle' and 'Lab' to the 12th position (1-based -> index 11)
cols = control_new.columns.tolist()
for c in ["Vehicle", "Lab"]:
    if c in cols:
        cols.remove(c)
insert_at = min(11, len(cols))  # safe if there are <12 columns
new_order = cols[:insert_at] + ["Vehicle", "Lab"] + cols[insert_at:]
control_new = control_new[new_order]

# ===========================
# TREATMENT: merge + reorder
# ===========================

# 2) Left-join on Compound name
treatment_new = treatment.merge(tgp_sub, on="COMPOUND_NAME", how="left", validate="m:1")

# 3) Move 'Vehicle' and 'Lab' to the 12th position
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

# Pick the compound-name column in each DF
combined_comp_col = "COMPOUND_NAME" if "COMPOUND_NAME" in combined.columns else "Compound name"
control_comp_col  = "COMPOUND_NAME" if "COMPOUND_NAME" in control_new.columns else "Compound name"

# Exact-match filter (keeps only test rows whose compound exists in control_new)
valid_compounds = control_new[control_comp_col].dropna().unique()
combined = combined[combined[combined_comp_col].isin(valid_compounds)].copy()

# 1) Which features to score? (columns 11 onward; assumes first 11 are ID/meta)
feature_cols = combined.columns[11:]

# =============================================================================
# 5. WITHIN-COMPOUND Z-SCORES (CONTROL VS HIGH DOSE)
# =============================================================================

results = []

# 2) Loop per (compound, time)
for (compound, time), grp in combined.groupby(['COMPOUND_NAME','SACRIFICE_PERIOD'], dropna=False):
    # 2a) pull the control rows
    ctrl = grp[grp['DOSE_LEVEL'] == 'Control']
    if ctrl.shape[0] < 2:
        continue  # need >=2 to get a nonzero SD

    # 2b) precompute control means & SDs per feature
    ctrl_means = ctrl[feature_cols].mean()
    ctrl_sds   = ctrl[feature_cols].std(ddof=1)

    # 3) treatment dose (High)
    trt = grp[grp['DOSE_LEVEL'] == 'High']
    if trt.empty:
        continue

    # 4) compute z-scores for each individual treatment sample
    for _, row in trt.iterrows():
        rec = {
            'compound':      compound,
            'time':          time,
            'dose':          'High',
            'INDIVIDUAL_ID': row['INDIVIDUAL_ID']
        }
        for f in feature_cols:
            mu = ctrl_means[f]
            sd = ctrl_sds[f]
            x  = row[f]
            rec[f'{f}_z'] = np.nan if (pd.isna(sd) or sd == 0) else (x - mu) / sd

        results.append(rec)

# 5) Final DataFrame of within-compound z-scores
results_df = pd.DataFrame(results)

# =============================================================================
# 6. ABNORMAL COUNTS PER FEATURE (REAL DATA, WITHIN-COMPOUND)
# =============================================================================

# 1) Identify all your z-score columns (they end with '_z')
z_cols = [col for col in results_df.columns if col.endswith('_z')]
keys   = ['compound','time','dose']

# 2) Container for per-feature treatment counts
feature_counts = []

# 3) Loop over each feature’s z-scores (NA-aware)
for z in z_cols:
    s = results_df[z]

    # 3a) Cell-level abnormal flag: |z| > 2; keep NaN as NA (not False)
    abn = s.abs().gt(2)
    abn = abn.where(s.notna(), pd.NA).astype('boolean')

    # 3b) Collapse to treatment level with valid counts and any-abnormal over non-NA
    grp = (
        pd.concat([results_df[keys], abn.rename('abn')], axis=1)
          .groupby(keys)['abn']
          .agg(valid_n=lambda x: x.notna().sum(),
               abn_any=lambda x: x.any(skipna=True))
          .reset_index()
    )

    # 3c) Exclude treatments with zero usable cells for this feature
    grp = grp[grp['valid_n'] > 0]

    # 3d) Count abnormal vs. normal treatments
    abn_count  = int((grp['abn_any'] == True).sum())
    norm_count = int((grp['abn_any'] == False).sum())

    feature_counts.append({
        'feature':  z[:-2],   # strip '_z'
        'abnormal': abn_count,
        'normal':   norm_count
    })

# 4) Build a DataFrame for easy viewing
feature_status_df = pd.DataFrame(feature_counts)
feature_status_df

# =============================================================================
# 7. INTRA-LAB MATCHED-CONTROL Z-SCORES (SAME LAB, SAME VEHICLE, DIFF COMPOUND)
# =============================================================================

# Make sure features are numeric in both frames
control_new[feature_cols]   = control_new[feature_cols].apply(pd.to_numeric, errors="coerce")
treatment_new[feature_cols] = treatment_new[feature_cols].apply(pd.to_numeric, errors="coerce")

# Column names
comp_col = "COMPOUND_NAME"
time_col = "SACRIFICE_PERIOD"
veh_col  = "Vehicle" if "Vehicle" in control_new.columns else "vehicle"
lab_col  = "Lab"     if "Lab"     in control_new.columns else "lab"

# --- 1) Compute per-sample z-scores in treatment_new using matched controls ---
rows = []
for (compound, time), grp in treatment_new.groupby([comp_col, time_col], dropna=False):

    for _, row in grp.iterrows():
        v = row[veh_col] if veh_col in treatment_new.columns else np.nan
        l = row[lab_col] if lab_col in treatment_new.columns else np.nan

        # matched controls: same time, same vehicle, same lab, different compound
        ctrl_mask = (
            (control_new[time_col] == time) &
            (control_new[comp_col] != compound) &
            (control_new[veh_col]  == v) &
            (control_new[lab_col]  == l)
        )
        ctrl = control_new.loc[ctrl_mask, feature_cols]

        rec = {
            "compound": compound,
            "time": time,
        }
        if "DOSE_LEVEL" in treatment_new.columns:
            rec["dose"] = row["DOSE_LEVEL"]
        if "INDIVIDUAL_ID" in treatment_new.columns:
            rec["INDIVIDUAL_ID"] = row["INDIVIDUAL_ID"]

        if ctrl.shape[0] < 2:
            for f in feature_cols:
                rec[f"{f}_z"] = np.nan
            rows.append(rec)
            continue

        means = ctrl.mean()
        stds  = ctrl.std(ddof=1)  # <-- STANDARD DEVIATION

        for f in feature_cols:
            x   = row[f]
            mu  = means[f]
            sd  = stds[f]
            if pd.isna(sd) or sd == 0:
                rec[f"{f}_z"] = np.nan
            else:
                rec[f"{f}_z"] = (x - mu) / sd  # <-- divide by std

        rows.append(rec)

results_intra_df = pd.DataFrame(rows)

# =============================================================================
# 8. ABNORMAL COUNTS PER FEATURE (INTRA-LAB MATCHED CONTROLS)
# =============================================================================

# 1) Identify all z-score columns in your generated-z DataFrame
z_cols_intra = [col for col in results_intra_df.columns if col.endswith('_z')]
keys = ['compound','time','dose']

# 2) Container for counts per feature
feature_counts_intra = []

# 3) Loop over each feature’s z-scores (NA-aware)
for z in z_cols_intra:
    s = results_intra_df[z]

    # Cell-level abnormal: |z|>2; preserve NaN as NA (not False)
    abn = s.abs().gt(2)
    abn = abn.where(s.notna(), pd.NA).astype('boolean')

    # Collapse to treatment, tracking usable cells and "any abnormal" over non-NA
    grp = (
        pd.concat([results_intra_df[keys], abn.rename('abn')], axis=1)
          .groupby(keys)['abn']
          .agg(valid_n=lambda x: x.notna().sum(),
               abn_any=lambda x: x.any(skipna=True))
          .reset_index()
    )

    # Exclude treatments with zero usable cells for this feature
    grp = grp[grp['valid_n'] > 0]

    # Count abnormal vs normal treatments
    abn_count  = int((grp['abn_any'] == True).sum())
    norm_count = int((grp['abn_any'] == False).sum())

    feature_counts_intra.append({
        'feature':  z[:-2],   # strip '_z'
        'abnormal': abn_count,
        'normal':   norm_count
    })

# 4) Build a DataFrame for display
feature_status_intra_df = pd.DataFrame(feature_counts_intra)
feature_status_intra_df

# =============================================================================
# 9. BUILD TREATMENT-LEVEL ABNORMAL FLAGS (REAL VS INTRA-LAB MATCHED)
# =============================================================================

# ── 1) Build per-treatment “abnormal” flags for real data ──
real_flags = []
for z in [c for c in results_df.columns if c.endswith('_z')]:
    feat = z[:-2]   # strip off '_z'
    tmp = results_df[['compound','time','dose', z]].copy()
    tmp['abn_real'] = tmp[z].abs() > 2
    # one row per (compound,time,dose), True if any sample was abnormal
    grp = (
        tmp
        .groupby(['compound','time','dose'])['abn_real']
        .any()
        .reset_index()
    )
    grp['feature'] = feat
    real_flags.append(grp)
real_flags_df = pd.concat(real_flags, ignore_index=True)

# ── 2) Build per-treatment “abnormal” flags for generated data (intra-lab) ──
intra_flags = []
for z in [c for c in results_intra_df.columns if c.endswith('_z')]:
    feat = z[:-2]
    tmp = results_intra_df[['compound','time','dose', z]].copy()
    tmp['abn_gen'] = tmp[z].abs() > 2
    grp = (
        tmp
        .groupby(['compound','time','dose'])['abn_gen']
        .any()
        .reset_index()
    )
    grp['feature'] = feat
    intra_flags.append(grp)
intra_flags_df = pd.concat(intra_flags, ignore_index=True)

# ── 3) Merge real vs gen flags ──
merged_flags = pd.merge(
    real_flags_df,
    intra_flags_df,
    on=['compound','time','dose','feature'],
    how='inner'
)

# =============================================================================
# 10. CONFUSION COUNTS & ACCURACY PER FEATURE
# =============================================================================

# ── 4) Compute TP/TN/FP/FN per feature ──
confusion = []
for feat, grp in merged_flags.groupby('feature'):
    real_abn = grp['abn_real']
    gen_abn  = grp['abn_gen']

    TP = ((~real_abn) & (~gen_abn)).sum()   # both normal
    TN = ( real_abn  &  gen_abn ).sum()     # both abnormal
    FP = ( real_abn  & (~gen_abn)).sum()    # real abnormal, gen normal
    FN = ((~real_abn) &  gen_abn ).sum()    # real normal,   gen abnormal

    confusion.append({
        'feature': feat,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    })

confusion_df = pd.DataFrame(confusion)

# 1) Compute total cases per feature
confusion_df['total'] = (
    confusion_df['TP'] +
    confusion_df['TN'] +
    confusion_df['FP'] +
    confusion_df['FN']
)

# 2) Compute accuracy = (TP + TN) / total
confusion_df['accuracy'] = (
    (confusion_df['TP'] + confusion_df['TN']) /
    confusion_df['total']
)

# 3) (Optional) drop the 'total' column if you don’t need it
confusion_df = confusion_df.drop(columns=['total'])

# Final per-feature confusion matrix + accuracy
confusion_df
