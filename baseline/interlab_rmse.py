"""
Inter-lab RMSE analysis for control-only clinical pathology data.

Pipeline overview
-----------------
1. Load per-animal control clinical pathology data from:
   - A training split
   - A test split
2. Load baseline metadata table (e.g., treatment group info: Vehicle, Lab, etc.).
3. Clean and normalize compound names in the metadata table.
4. Merge metadata into the control data and reorder columns for readability.
5. Prepare numeric feature matrices for the biomarker panel (e.g., 38 clinical-pathology features).
6. For each sample, compute RMSE to samples from other labs and compounds,
   matched by (SACRIFICE_PERIOD, Vehicle), using the full feature panel.
7. Output a sample-level table of pairwise RMSE values for downstream analysis.

Note:
-----
- Paths below are placeholders; update them to match your environment.
- This script assumes certain column names exist:
  COMPOUND_NAME, SACRIFICE_PERIOD, Vehicle/vehicle, Lab/lab, INDIVIDUAL_ID, etc.
"""

import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from scipy.stats import pearsonr
import math
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

import os
from os import listdir
from os.path import join, isfile

# =============================================================================
# 0. FILE PATHS (UPDATE THESE FOR YOUR ENVIRONMENT)
# =============================================================================

CONTROL_TEST_FILE = "/path/to/clin_path/repeat_test_control_cv2.csv"
CONTROL_TRAIN_FILE = "/path/to/clin_path/repeat_train_control_cv2.csv"
BASELINE_META_FILE = "/path/to/baseline/metadata.csv"  # sample-level lab/treatment metadata

# =============================================================================
# 1. LOAD CONTROL DATA & BASELINE METADATA
# =============================================================================

test = pd.read_csv(CONTROL_TEST_FILE)
train = pd.read_csv(CONTROL_TRAIN_FILE)
tgp = pd.read_csv(BASELINE_META_FILE)

control = pd.concat([test, train], ignore_index=True, sort=False)

# =============================================================================
# 2. CLEAN & NORMALIZE COMPOUND NAMES IN BASELINE METADATA (tgp)
# =============================================================================

# Standardize column names for lab and compound
tgp = tgp.rename(columns={"Test Facility (in vivo animal treatment)": "Lab"})
tgp = tgp.rename(columns={"Compound name (E)": "COMPOUND_NAME"})
tgp = tgp.dropna()

s = tgp["COMPOUND_NAME"].astype("string")
mask = s.notna()
tgp.loc[mask, "COMPOUND_NAME"] = s[mask].str.replace(
    r"^(\s*)(\S)", lambda m: m.group(1) + m.group(2).lower(), regex=True
)

tgp["COMPOUND_NAME"] = tgp["COMPOUND_NAME"].replace({
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
    "iproniazid phosphate salt": "iproniazid"
})

# --- 0) Make a de-duplicated lookup from tgp (one row per compound) ---
tgp_sub = (
    tgp[["COMPOUND_NAME", "Vehicle", "Lab"]]
    .drop_duplicates(subset=["COMPOUND_NAME"], keep="first")
)

# =============================================================================
# 3. CONTROL: MERGE METADATA & REORDER COLUMNS
# =============================================================================

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

control_new = control_new.dropna()

# =============================================================================
# 4. HELPERS FOR FEATURE PREP & RMSE CALCULATION
# =============================================================================

def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _prep_features(df, feature_cols):
    present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[warn] Missing feature columns (skipped): {missing}")
    if not present:
        raise ValueError("None of the requested feature columns are present in the dataframe.")
    out = df.copy()
    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    present = [c for c in present if pd.api.types.is_numeric_dtype(out[c])]
    if not present:
        raise ValueError("After coercion, none of the feature columns are numeric.")
    return out, present

def _numeric_matrix(df, cols):
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=float)

def _numeric_row(row, cols):
    v = row[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return v.to_numpy(dtype=float)


def _compute_rmse_grouped_by_cmpd_time_with_vehicle_lab(
    df, feature_cols, panel_name, outfile=None, id_col="INDIVIDUAL_ID"
):
    """
    For each anchor sample, compute RMSE to samples from other labs and compounds
    matched by (SACRIFICE_PERIOD, Vehicle), using the given feature_cols together.
    RMSE is computed via sklearn.metrics.mean_squared_error.
    """
    comp_col = "COMPOUND_NAME"
    time_col = "SACRIFICE_PERIOD"
    veh_col  = _pick_col(df, ["Vehicle", "vehicle"])
    lab_col  = _pick_col(df, ["Lab", "lab"])
    need = [comp_col, time_col, veh_col, lab_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")

    work, feats = _prep_features(df, feature_cols)
    work = work.dropna(subset=need).reset_index(drop=True)

    # ---- precompute per (time, vehicle) pools to reuse matrices
    tv_cache = {}
    for (period, vehicle), grp_tv in work.groupby([time_col, veh_col], dropna=False):
        if grp_tv[lab_col].dropna().nunique() < 2:
            # need at least 2 labs to compare across labs
            continue
        tv_cache[(period, vehicle)] = {
            "df": grp_tv,
            "X":  _numeric_matrix(grp_tv, feats),
        }

    rows = []

    # ---- OUTER GROUP: by (compound, time)
    for (cmpd, period), grp_ct in work.groupby([comp_col, time_col], dropna=False):
        if grp_ct.empty:
            continue

        for _, r in grp_ct.iterrows():
            vehicle = r[veh_col]
            lab     = r[lab_col]

            key = (period, vehicle)
            if key not in tv_cache:
                continue

            pool_df = tv_cache[key]["df"]
            pool_X  = tv_cache[key]["X"]

            xa = _numeric_row(r, feats)  # shape (d,)

            # ---- RMSE vs every row in pool using mean_squared_error ----
            mse_vec = np.array([
                mean_squared_error(xa, pool_X[j, :], squared=True)
                for j in range(pool_X.shape[0])
            ])
            rmse_vec = np.sqrt(mse_vec)

            # other lab, other compound
            valid_mask = (
                (pool_df[lab_col].notna()) &
                (pool_df[lab_col].values != lab) &
                (pool_df[comp_col].values != cmpd)
            )
            if not np.any(valid_mask):
                continue

            ids2 = pool_df[id_col].values if id_col in pool_df.columns else pool_df.index.values
            out_idx = np.where(valid_mask)[0]

            for j in out_idx:
                rows.append({
                    "panel":             panel_name,
                    comp_col:            cmpd,
                    "other_compound":    pool_df.iloc[j][comp_col],
                    time_col:            period,
                    veh_col:             vehicle,
                    "anchor_lab":        lab,
                    "other_lab":         pool_df.iloc[j][lab_col],
                    f"{id_col}_1":       r[id_col] if id_col in r.index else np.nan,
                    f"{id_col}_2":       ids2[j],
                    "rmse":              float(rmse_vec[j]),
                })

    out = pd.DataFrame(rows)
    if outfile:
        out.to_csv(outfile, index=False)
        print(f"[info] Wrote {len(out)} rows to: {outfile}")
    return out


def compute_rmse_all38_grouped_cmpd_time(
    df, outfile, id_col="INDIVIDUAL_ID"
):
    """
    Wrapper: use ALL 38 biomarkers together.

    Assumes biomarker columns start at index 13,
    so feature columns are df.columns[13:].
    """
    feature_cols = list(df.columns[13:])
    print("[info] 38-feature panel (cols[13:]):")
    print(feature_cols)

    return _compute_rmse_grouped_by_cmpd_time_with_vehicle_lab(
        df, feature_cols, "all38", outfile=outfile, id_col=id_col
    )

# =============================================================================
# 5. USAGE: COMPUTE & SAVE INTER-LAB RMSE
# =============================================================================

all38_rmse_csv = "interlab_rmse_overall.csv"

all38_rmse_df = compute_rmse_all38_grouped_cmpd_time(
    control_new,
    outfile=all38_rmse_csv,
    id_col="INDIVIDUAL_ID",
)
