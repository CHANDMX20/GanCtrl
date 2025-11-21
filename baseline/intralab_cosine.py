import sys

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer

import os
from os import listdir
from os.path import join, isfile


test = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_test_control_cv2.csv')
train = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_train_control_cv2.csv')
tgp = pd.read_csv('/account001/mansi.chandra/clin_path/baseline/Sample list_NCTR_09-06-2012.csv')

control = pd.concat([test, train], ignore_index=True, sort=False)

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
control_new = control_new.dropna()


# --- helpers ---

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

def _numeric_row(row, cols):
    v = (
        row[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return v.reshape(1, -1)

def _numeric_matrix(df, cols):
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=float)


def _compute_cosine_grouped_by_cmpd_time_with_vehicle_lab(
    df,
    feature_cols,
    panel_name,
    outfile=None,
    id_col="INDIVIDUAL_ID",
    lab_relation="same",
):
    """
    For each anchor sample in (COMPOUND_NAME, SACRIFICE_PERIOD), compute cosine to targets from:
      - same time (SACRIFICE_PERIOD),
      - same vehicle,
      - lab relation per `lab_relation`: "same" | "different" | "both",
      - different compound (other_compound != anchor compound).
    Uses all given feature_cols together.
    """
    comp_col = "COMPOUND_NAME"
    time_col = "SACRIFICE_PERIOD"
    veh_col  = _pick_col(df, ["Vehicle", "vehicle"])
    lab_col  = _pick_col(df, ["Lab", "lab"])
    need = [comp_col, time_col, veh_col, lab_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns: {miss}")

    # features
    work, feats = _prep_features(df, feature_cols)
    print(f"[info] Using {len(feats)} features for panel '{panel_name}'")
    # require keys (drop NaN in keys)
    work = work.dropna(subset=need).reset_index(drop=True)

    # Precompute (time, vehicle) pools (intra / inter lab handled by mask)
    tv_cache = {}
    for (period, vehicle), grp_tv in work.groupby([time_col, veh_col], dropna=False):
        tv_cache[(period, vehicle)] = {
            "df": grp_tv,
            "X":  _numeric_matrix(grp_tv, feats),
        }

    rows = []

    # OUTER GROUP: (compound, time)
    for (cmpd, period), grp_ct in work.groupby([comp_col, time_col], dropna=False):
        if grp_ct.empty:
            continue

        for _, r in grp_ct.iterrows():
            vehicle = r[veh_col]
            lab     = r[lab_col]
            key     = (period, vehicle)
            if key not in tv_cache:
                continue

            pool_df = tv_cache[key]["df"]
            pool_X  = tv_cache[key]["X"]

            # anchor vector and cosine vs entire pool
            xa = _numeric_row(r, feats)
            s  = cosine_similarity(xa, pool_X).ravel()

            # lab relation mask
            if lab_relation == "different":
                lab_mask = (pool_df[lab_col].values != lab)
            elif lab_relation == "same":
                lab_mask = (pool_df[lab_col].values == lab)
            elif lab_relation == "both":
                lab_mask = pool_df[lab_col].notna().values
            else:
                raise ValueError("lab_relation must be 'same', 'different', or 'both'.")

            valid_mask = (
                pool_df[lab_col].notna().values &
                lab_mask &
                (pool_df[comp_col].values != cmpd)  # other compound only
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
                    "lab_relation":      lab_relation,
                    "cosine":            float(s[j]),
                })

    out = pd.DataFrame(rows)
    if outfile:
        out.to_csv(outfile, index=False)
    return out


def compute_cosine_all38_grouped_cmpd_time(
    df,
    outfile,
    id_col="INDIVIDUAL_ID",
    lab_relation="same",
):
    """
    Wrapper: use ALL 38 biomarkers together.

    Assumes:
      - biomarker columns start at index 13
      - feature columns are df.columns[13:]
    """
    feature_cols = list(df.columns[13:])
    print("[info] 38-feature panel (cols[13:]):")
    print(feature_cols)

    return _compute_cosine_grouped_by_cmpd_time_with_vehicle_lab(
        df=df,
        feature_cols=feature_cols,
        panel_name="all38",
        outfile=outfile,
        id_col=id_col,
        lab_relation=lab_relation,
    )

# -------- usage example --------

intra_all38_csv = "intralab_cosine_overall.csv"

intra_all38_df = compute_cosine_all38_grouped_cmpd_time(
    control_new,
    outfile=intra_all38_csv,
    id_col="INDIVIDUAL_ID",
    lab_relation="same",   # "same" / "different" / "both"
)









