import pandas as pd
import numpy as np
import math 
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kstest, zscore

data = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_merged.csv')
tgp = pd.read_csv('/account001/mansi.chandra/clin_path/baseline/Sample list_NCTR_09-06-2012.csv')

control = data[data['DOSE_LEVEL'] == 'Control']
treatment = data[data['DOSE_LEVEL'] == 'High']

# 1) ensure we only concat common columns (avoids surprises if cols differ)
common_cols = control.columns.intersection(treatment.columns)

# 2) optional helper label (since DOSE_LEVEL already tells this, you can skip if you want)
control_labeled   = control[common_cols].copy()
treatment_labeled = treatment[common_cols].copy()

# 3) concatenate
combined = pd.concat([control_labeled, treatment_labeled], ignore_index=True)

# change column named "old_name" to "lab"
tgp = tgp.rename(columns={"Test Facility (in vivo animal treatment)": "Lab"})
tgp = tgp.rename(columns={"Compound name (E)": "COMPOUND_NAME"})
tgp = tgp.dropna()
s = tgp['COMPOUND_NAME'].astype("string")
mask = s.notna()
tgp.loc[mask, 'COMPOUND_NAME'] = s[mask].str.replace(
    r"^(\s*)(\S)", lambda m: m.group(1) + m.group(2).lower(), regex=True
)
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


# =========================
# CONTROL: merge + reorder
# =========================

# 1) Left-join on Compound name (many control rows to one tgp row per compound)
control_new = control.merge(tgp_sub, on="COMPOUND_NAME", how="left", validate="m:1")

# 3) Move 'vehicle' and 'lab' to the 12th position (1-based -> index 11)
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

# 3) Move 'vehicle' and 'lab' to the 12th position
cols2 = treatment_new.columns.tolist()
for c2 in ["Vehicle", "Lab"]:
    if c2 in cols2:
        cols2.remove(c2)
insert_at = min(11, len(cols2))
new_order2 = cols2[:insert_at] + ["Vehicle", "Lab"] + cols2[insert_at:]
treatment_new = treatment_new[new_order2]

control_new = control_new.dropna()
treatment_new = treatment_new.dropna()


# Pick the compound-name column in each DF
combined_comp_col   = "COMPOUND_NAME" if "COMPOUND_NAME" in combined.columns else "Compound name"
control_comp_col= "COMPOUND_NAME" if "COMPOUND_NAME" in control_new.columns else "Compound name"

# Exact-match filter (keeps only test rows whose compound exists in control_new)
valid_compounds = control_new[control_comp_col].dropna().unique()
combined = combined[combined[combined_comp_col].isin(valid_compounds)].copy()

# 1) Which features to score? (columns 11 onward)
feature_cols = combined.columns[11:]

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

# 5) Final DataFrame
results_df = pd.DataFrame(results)


# 1) Identify all your z-score columns (they end with '_z')
z_cols = [col for col in results_df.columns if col.endswith('_z')]
keys   = ['compound','time','dose']

# 2) Container for per-feature treatment counts
feature_counts = []

# 3) Loop over each feature’s z-scores (NA-aware)
for z in z_cols:
    s = results_df[z]

    # 3a) sample-level abnormal (strict > 2), keep NaN as NA (nullable boolean)
    abn = s.abs().gt(2)
    abn = abn.where(s.notna(), pd.NA).astype('boolean')  # NA stays NA, not False

    # 3b) Collapse to treatment: track usable n and "any abnormal among non-NA"
    grp = (
        pd.concat([results_df[keys], abn.rename('abn')], axis=1)
          .groupby(keys)['abn']
          .agg(valid_n=lambda x: x.notna().sum(),
               abn_any=lambda x: x.any(skipna=True))
          .reset_index()
    )

    # 3c) Exclude treatments where this feature had no usable value at all
    grp = grp[grp['valid_n'] > 0]

    # 3d) Count abnormal vs normal treatments
    abn_count  = int((grp['abn_any'] == True).sum())
    norm_count = int((grp['abn_any'] == False).sum())

    feature_counts.append({
        'feature':  z[:-2],   # strip '_z'
        'abnormal': abn_count,
        'normal':   norm_count
    })

# 4) Build the result table
feature_status_df = pd.DataFrame(feature_counts)
feature_status_df

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

        # matched controls: same time, same vehicle, different lab, different compound
        ctrl_mask = (
            (control_new[time_col] == time) &
            (control_new[comp_col] != compound) &
            (control_new[veh_col]  == v) &
            (control_new[lab_col]  != l)
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

results_inter_df = pd.DataFrame(rows)

# 1) Identify all z-score columns in your generated-z DataFrame
z_cols_inter = [col for col in results_inter_df.columns if col.endswith('_z')]
keys = ['compound','time','dose']

# 2) Container for counts per feature
feature_counts_inter = []

# 3) Loop over each feature’s z-scores (NA-aware)
for z in z_cols_inter:
    s = results_inter_df[z]

    # cell-level abnormal flag (strict > 2), keep NaN as NA (not False)
    abn = s.abs().gt(2)
    abn = abn.where(s.notna(), pd.NA).astype('boolean')

    # collapse to treatment level: count usable cells, "any abnormal" over non-NA
    grp = (
        pd.concat([results_inter_df[keys], abn.rename('abn')], axis=1)
          .groupby(keys)['abn']
          .agg(valid_n=lambda x: x.notna().sum(),
               abn_any=lambda x: x.any(skipna=True))
          .reset_index()
    )

    # exclude treatments with zero usable cells for this feature
    grp = grp[grp['valid_n'] > 0]

    # count abnormal vs normal treatments
    abn_count  = int((grp['abn_any'] == True).sum())
    norm_count = int((grp['abn_any'] == False).sum())

    feature_counts_inter.append({
        'feature':  z[:-2],   # strip '_z'
        'abnormal': abn_count,
        'normal':   norm_count
    })

# 4) Build a DataFrame for display
feature_status_inter_df = pd.DataFrame(feature_counts_inter)
feature_status_inter_df

# ── 1) Build per‑treatment “abnormal” flags for real data ──
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


# ── 2) Build per‑treatment “abnormal” flags for generated data ──
inter_flags = []
for z in [c for c in results_inter_df.columns if c.endswith('_z')]:
    feat = z[:-2]
    tmp = results_inter_df[['compound','time','dose', z]].copy()
    tmp['abn_gen'] = tmp[z].abs() > 2
    grp = (
        tmp
        .groupby(['compound','time','dose'])['abn_gen']
        .any()
        .reset_index()
    )
    grp['feature'] = feat
    inter_flags.append(grp)
inter_flags_df = pd.concat(inter_flags, ignore_index=True)


# ── 3) Merge real vs gen flags ──
merged_flags = pd.merge(
    real_flags_df,
    inter_flags_df,
    on=['compound','time','dose','feature'],
    how='inner'
)


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

# assume confusion_df has columns ['feature','TP','TN','FP','FN']

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

confusion_df













