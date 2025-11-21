import os
import random
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# ==============================
# Global seeding (call once)
# ==============================
def set_all_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        random.seed(seed)
    except Exception:
        pass

    try:
        np.random.seed(seed)
    except Exception:
        pass

    # Optional: if you run TF or PyTorch upstream to produce "generated" data
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    except Exception:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    except Exception:
        pass

# Set seed immediately
set_all_seeds(42)


control = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_test_control_cv2.csv')
treatment = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_test_treatment_cv2.csv')
real = pd.concat([control, treatment], ignore_index=True, sort=False)
gen = pd.read_csv('/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/predictions_decoded/test/generated_predictions_1507935_ControlGenerator_test.csv')


#optimizing the prediction values to be consistent with real 

gen["TBIL(mg/dL)"] = pd.to_numeric(gen["TBIL(mg/dL)"], errors="coerce").round(2)
gen["RALB(g/dL)"] = pd.to_numeric(gen["RALB(g/dL)"], errors="coerce").round(2)
gen["AST(IU/L)"] = pd.to_numeric(gen["AST(IU/L)"], errors="coerce").round().astype(int)
gen["TP(g/dL)"] = pd.to_numeric(gen["TP(g/dL)"], errors="coerce").round(1)
#gen["CRE(mg/dL)"] = pd.to_numeric(gen["CRE(mg/dL)"], errors="coerce").round(1)
gen["DBIL(mg/dL)"] = pd.to_numeric(gen["DBIL(mg/dL)"], errors="coerce").round(2)
gen["BUN(mg/dL)"] = pd.to_numeric(gen["BUN(mg/dL)"], errors="coerce").round().astype(int)
#gen["K(meq/L)"] = pd.to_numeric(gen["K(meq/L)"], errors="coerce").round(2)
#gen["GTP(IU/L)"] = pd.to_numeric(gen["GTP(IU/L)"], errors="coerce").round().astype(int)
#gen["Ca(mg/dL)"] = pd.to_numeric(gen["Ca(mg/dL)"], errors="coerce").round(1)
#gen["Cl(meq/L)"] = pd.to_numeric(gen["Cl(meq/L)"], errors="coerce").round(1)
#gen["Na(meq/L)"] = pd.to_numeric(gen["Na(meq/L)"], errors="coerce").round(1)
#gen["IP(mg/dL)"] = pd.to_numeric(gen["IP(mg/dL)"], errors="coerce").round(1)
gen["ALP(IU/L)"] = pd.to_numeric(gen["ALP(IU/L)"], errors="coerce").round().astype(int)
gen["ALT(IU/L)"] = pd.to_numeric(gen["ALT(IU/L)"], errors="coerce").round().astype(int)
gen["LDH(IU/L)"] = pd.to_numeric(gen["LDH(IU/L)"], errors="coerce").round().astype(int)
gen["RALB(g/dL)"] = pd.to_numeric(gen["RALB(g/dL)"], errors="coerce").round(1)

control = real[real['DOSE_LEVEL'] == 'Control']
treat = real[real['DOSE_LEVEL'] == 'High']
generated = gen[gen['DOSE_LEVEL'] == 'High']


# ==============================
# Feature mapping & keys
# ==============================
FEATURES = {
    "AST":  ["AST(IU/L)"],
    "ALT":  ["ALT(IU/L)"],
    "TBIL": ["TBIL(mg/dL)"],
    "DBIL": ["DBIL(mg/dL)"],
    "ALP":  ["ALP(IU/L)"],
    "LDH":  ["LDH(IU/L)"],
    "RALB": ["RALB(g/dL)"],
    "TP":   ["TP(g/dL)"],
    "GTP":  ["GTP(IU/L)"],
    "BUN":  ["BUN(mg/dL)"],
    "CRE":  ["CRE(mg/dL)"],
    "Ca":   ["Ca(mg/dL)"],
    "Cl":   ["Cl(meq/L)"],
    "IP":   ["IP(mg/dL)"],
    "K":    ["K(meq/L)"],
    "Na":   ["Na(meq/L)"],
}
KEYS = ["COMPOUND_NAME", "SACRIFICE_PERIOD"]

FEATURE_ORDER = [
    "ALP",
    "ALT",
    "AST",
    "GTP",
    "LDH",
    "TBIL",
    "DBIL",
    "RALB",
    "TP",
    "BUN",
    "CRE",
    "Cl",
    "Ca",
    "IP",
    "K",
    "Na",
]


# ==============================
# Helpers (deterministic BH-FDR)
# ==============================
def _bh_fdr_stable(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=float)
    idx = np.arange(n)
    order = np.lexsort((idx, p))                     # primary: p asc, secondary: original index
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q_raw = p * n / ranks
    q_on_order = np.minimum.accumulate(q_raw[order][::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(q_on_order, 1.0)
    return q

def _add_fdr_by_group(
    df: pd.DataFrame,
    p_col: str = "p_one_sided",
    group_keys=KEYS,
    fdr_alpha: float = 0.05,
    flag_col: str = "significant_fdr"
) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby(group_keys, dropna=False, sort=False):
        g = g.copy()
        p = g[p_col].fillna(1.0).to_numpy()
        g["q_value"] = _bh_fdr_stable(p)
        g[flag_col] = g["q_value"] <= fdr_alpha
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df.assign(q_value=np.nan, **{flag_col: False})

def _ttest_right(x: np.ndarray, y: np.ndarray):
    try:
        res = ttest_ind(x, y, equal_var=False, nan_policy="omit", alternative="greater")
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        t_stat, p_two = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        if np.isnan(t_stat) or np.isnan(p_two):
            return float("nan"), float("nan")
        p_right = (p_two / 2.0) if (t_stat > 0) else (1.0 - p_two / 2.0)
        return float(t_stat), float(p_right)

# ==============================
# 1) Treat (real High) vs Control (real Control) — elevation only
# ==============================
def ttests_treat_vs_control(
    control_df: pd.DataFrame,
    treat_df: pd.DataFrame,
    features_map=FEATURES,
    alpha: float = 0.05,
    fdr_alpha: float = 0.05,
):
    keys_c = control_df[KEYS].drop_duplicates()
    keys_t = treat_df[KEYS].drop_duplicates()
    keys = keys_c.merge(keys_t, on=KEYS, how="inner").sort_values(KEYS, kind="mergesort").reset_index(drop=True)

    rows = []
    for _, grp in keys.iterrows():
        m_c = (control_df["COMPOUND_NAME"] == grp["COMPOUND_NAME"]) & (control_df["SACRIFICE_PERIOD"] == grp["SACRIFICE_PERIOD"])
        m_t = (treat_df["COMPOUND_NAME"] == grp["COMPOUND_NAME"]) & (treat_df["SACRIFICE_PERIOD"] == grp["SACRIFICE_PERIOD"])
        c_grp = control_df.loc[m_c]
        t_grp = treat_df.loc[m_t]

        for feat_name, cols in features_map.items():
            col = cols[0]
            if (col not in c_grp.columns) or (col not in t_grp.columns):
                continue

            x = pd.to_numeric(t_grp[col], errors="coerce").dropna().to_numpy()
            y = pd.to_numeric(c_grp[col], errors="coerce").dropna().to_numpy()
            n_t, n_c = len(x), len(y)
            mean_t = float(np.mean(x)) if n_t else np.nan
            mean_c = float(np.mean(y)) if n_c else np.nan
            effect = (mean_t - mean_c) if (n_t and n_c) else np.nan

            if n_t < 2 or n_c < 2:
                rows.append({**grp.to_dict(),
                             "FEATURE": feat_name,
                             "n_treat": n_t, "n_control": n_c,
                             "mean_treat": mean_t, "mean_control": mean_c,
                             "effect": effect,
                             "t_stat": np.nan, "p_one_sided": np.nan,
                             "significant_raw": False})
                continue

            t_stat, p_right = _ttest_right(x, y)
            rows.append({**grp.to_dict(),
                         "FEATURE": feat_name,
                         "n_treat": n_t, "n_control": n_c,
                         "mean_treat": mean_t, "mean_control": mean_c,
                         "effect": effect,
                         "t_stat": t_stat, "p_one_sided": p_right,
                         "significant_raw": bool(np.isfinite(p_right) and (p_right <= alpha))})

    detailed = pd.DataFrame(rows)
    if not detailed.empty:
        detailed = detailed.sort_values(KEYS + ["FEATURE"], kind="mergesort").reset_index(drop=True)
        detailed = _add_fdr_by_group(detailed, p_col="p_one_sided", group_keys=KEYS, fdr_alpha=fdr_alpha, flag_col="significant_fdr")
        detailed["significant_fdr_elev"] = detailed["significant_fdr"] & (detailed["effect"] >= 0.0)
    else:
        detailed = pd.DataFrame(columns=[*KEYS, "FEATURE", "n_treat", "n_control", "mean_treat", "mean_control",
                                         "effect", "t_stat", "p_one_sided", "significant_raw", "q_value",
                                         "significant_fdr", "significant_fdr_elev"])

    counts_by_ct = (
        detailed.groupby(KEYS, dropna=False)["significant_fdr_elev"]
        .sum()
        .reset_index(name="n_significant_features_fdr_elev")
        .sort_values(KEYS, kind="mergesort")
    )

    counts_by_feature = (
        detailed.groupby(["FEATURE"], dropna=False)["significant_fdr_elev"]
        .sum()
        .reset_index(name="n_significant_groups_fdr_elev")
        .sort_values("n_significant_groups_fdr_elev", ascending=False, kind="mergesort")
    )

    return detailed, counts_by_ct, counts_by_feature

# ==============================
# 2) Real High vs Generated High — elevation only
# ==============================
def ttests_realHigh_vs_genHigh(
    real_high_df: pd.DataFrame,
    gen_high_df: pd.DataFrame,
    features_map=FEATURES,
    alpha: float = 0.05,
    fdr_alpha: float = 0.05,
):
    keys_real = real_high_df[KEYS].drop_duplicates()
    keys_gen  = gen_high_df[KEYS].drop_duplicates()
    keys = keys_real.merge(keys_gen, on=KEYS, how="inner").sort_values(KEYS, kind="mergesort").reset_index(drop=True)

    rows = []
    for _, grp in keys.iterrows():
        m_r = (real_high_df["COMPOUND_NAME"] == grp["COMPOUND_NAME"]) & (real_high_df["SACRIFICE_PERIOD"] == grp["SACRIFICE_PERIOD"])
        m_g = (gen_high_df["COMPOUND_NAME"] == grp["COMPOUND_NAME"]) & (gen_high_df["SACRIFICE_PERIOD"] == grp["SACRIFICE_PERIOD"])
        r_grp = real_high_df.loc[m_r]
        g_grp = gen_high_df.loc[m_g]

        for feat_name, cols in features_map.items():
            col = cols[0]
            if (col not in r_grp.columns) or (col not in g_grp.columns):
                continue

            x = pd.to_numeric(r_grp[col], errors="coerce").dropna().to_numpy()
            y = pd.to_numeric(g_grp[col], errors="coerce").dropna().to_numpy()
            n_r, n_g = len(x), len(y)
            mean_r = float(np.mean(x)) if n_r else np.nan
            mean_g = float(np.mean(y)) if n_g else np.nan
            effect = (mean_r - mean_g) if (n_r and n_g) else np.nan

            if n_r < 2 or n_g < 2:
                rows.append({**grp.to_dict(),
                             "FEATURE": feat_name,
                             "n_realHigh": n_r, "n_genHigh": n_g,
                             "mean_realHigh": mean_r, "mean_genHigh": mean_g,
                             "effect": effect,
                             "t_stat": np.nan, "p_one_sided": np.nan,
                             "significant_raw": False})
                continue

            t_stat, p_right = _ttest_right(x, y)
            rows.append({**grp.to_dict(),
                         "FEATURE": feat_name,
                         "n_realHigh": n_r, "n_genHigh": n_g,
                         "mean_realHigh": mean_r, "mean_genHigh": mean_g,
                         "effect": effect,
                         "t_stat": t_stat, "p_one_sided": p_right,
                         "significant_raw": bool(np.isfinite(p_right) and (p_right <= alpha))})

    detailed = pd.DataFrame(rows)
    if not detailed.empty:
        detailed = detailed.sort_values(KEYS + ["FEATURE"], kind="mergesort").reset_index(drop=True)
        detailed = _add_fdr_by_group(detailed, p_col="p_one_sided", group_keys=KEYS, fdr_alpha=fdr_alpha, flag_col="significant_fdr")
        detailed["significant_fdr_elev"] = detailed["significant_fdr"] & (detailed["effect"] >= 0.0)
    else:
        detailed = pd.DataFrame(columns=[*KEYS, "FEATURE", "n_realHigh", "n_genHigh", "mean_realHigh", "mean_genHigh",
                                         "effect", "t_stat", "p_one_sided", "significant_raw", "q_value",
                                         "significant_fdr", "significant_fdr_elev"])

    counts_by_ct = (
        detailed.groupby(KEYS, dropna=False)["significant_fdr_elev"]
        .sum()
        .reset_index(name="n_significant_features_fdr_elev")
        .sort_values(KEYS, kind="mergesort")
    )

    counts_by_feature = (
        detailed.groupby(["FEATURE"], dropna=False)["significant_fdr_elev"]
        .sum()
        .reset_index(name="n_significant_groups_fdr_elev")
        .sort_values("n_significant_groups_fdr_elev", ascending=False, kind="mergesort")
    )

    return detailed, counts_by_ct, counts_by_feature

# ==============================
# 3) Overlap metrics using FDR+upward flags
# ==============================

def match_significant_groups_fdr(detailed_ct: pd.DataFrame, detailed_tg: pd.DataFrame):
    # Significant (FDR + elevation) for treat vs control
    sig_ct = detailed_ct.loc[
        detailed_ct["significant_fdr_elev"] == True,
        KEYS + ["FEATURE"]
    ].drop_duplicates()

    # Significant (FDR + elevation) for treat vs generated
    sig_tg = detailed_tg.loc[
        detailed_tg["significant_fdr_elev"] == True,
        KEYS + ["FEATURE"]
    ].drop_duplicates()

    # Overlap (TP)
    overlap = sig_ct.merge(sig_tg, on=KEYS + ["FEATURE"], how="inner")

    # Counts per FEATURE
    n_ct = sig_ct.groupby("FEATURE", dropna=False).size().rename("n_sig_treat_vs_control_fdr_elev")
    n_tg = sig_tg.groupby("FEATURE", dropna=False).size().rename("n_sig_treat_vs_generated_fdr_elev")
    n_ov = overlap.groupby("FEATURE", dropna=False).size().rename("n_overlap_both_fdr_elev")  # TP

    summary = pd.concat([n_ct, n_tg, n_ov], axis=1).fillna(0).astype(int).reset_index()

    # recall = TP / (TP + FN) with (TP+FN) = n_ct
    den_recall = summary["n_sig_treat_vs_control_fdr_elev"].astype(float)
    den_recall = den_recall.where(den_recall != 0, np.nan)
    summary["recall_ct→tg"] = summary["n_overlap_both_fdr_elev"] / den_recall

    # derive FP and FN counts explicitly
    summary["fp_only_syn"] = (
        summary["n_sig_treat_vs_generated_fdr_elev"] - summary["n_overlap_both_fdr_elev"]
    ).clip(lower=0)
    summary["fn_only_real"] = (
        summary["n_sig_treat_vs_control_fdr_elev"] - summary["n_overlap_both_fdr_elev"]
    ).clip(lower=0)

    # total universe of (compound,time) groups per FEATURE from detailed tables (includes negatives)
    universe = pd.concat(
        [
            detailed_ct.loc[:, [*KEYS, "FEATURE"]],
            detailed_tg.loc[:, [*KEYS, "FEATURE"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    n_total = universe.groupby("FEATURE", dropna=False).size().rename("N_total_groups")

    # attach N, compute TN and ACCURACY
    summary = summary.merge(n_total, on="FEATURE", how="left")
    summary["N_total_groups"] = summary["N_total_groups"].fillna(0).astype(int)

    summary["tn_neither"] = (
        summary["N_total_groups"]
        - (
            summary["n_overlap_both_fdr_elev"]
            + summary["fp_only_syn"]
            + summary["fn_only_real"]
        )
    ).clip(lower=0)

    den_acc = summary["N_total_groups"].astype(float)
    den_acc = den_acc.where(den_acc != 0, np.nan)
    summary["accuracy_ct→tg"] = (
        summary["n_overlap_both_fdr_elev"] + summary["tn_neither"]
    ) / den_acc

    # ---------- NEW: Specificity, MCC, Balanced Accuracy ----------
    tp = summary["n_overlap_both_fdr_elev"].astype(float)
    fp = summary["fp_only_syn"].astype(float)
    fn = summary["fn_only_real"].astype(float)
    tn = summary["tn_neither"].astype(float)

    # Specificity = TN / (TN + FP)
    den_spec = (tn + fp)
    summary["specificity_ct→tg"] = (tn / den_spec).where(den_spec != 0, np.nan)

    # MCC = (TP*TN − FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    summary["mcc_ct→tg"] = ((tp * tn) - (fp * fn)) / mcc_den
    summary.loc[mcc_den == 0, "mcc_ct→tg"] = np.nan

    # Balanced Accuracy = (Recall + Specificity) / 2
    summary["balanced_accuracy_ct→tg"] = (
        summary["recall_ct→tg"] + summary["specificity_ct→tg"]
    ) / 2
    # --------------------------------------------------------------

    # Per-(compound,time,feature) mark table (optional for auditing)
    all_rows = pd.concat(
        [sig_ct.assign(src="REAL"), sig_tg.assign(src="SYN")],
        ignore_index=True,
    )
    mark = (
        all_rows.assign(val=True)
        .pivot_table(
            index=KEYS + ["FEATURE"],
            columns="src",
            values="val",
            aggfunc="max",
            fill_value=False,
        )
        .reset_index()
        .rename(
            columns={
                "REAL": "in_treat_vs_control_fdr_elev",
                "SYN": "in_treat_vs_generated_fdr_elev",
            }
        )
    )
    mark["in_both_fdr_elev"] = (
        mark["in_treat_vs_control_fdr_elev"]
        & mark["in_treat_vs_generated_fdr_elev"]
    )

    per_feature_tables = {
        feat: df.drop(columns=["FEATURE"])
               .sort_values(KEYS, kind="mergesort")
               .reset_index(drop=True)
        for feat, df in mark.groupby("FEATURE", dropna=False, sort=False)
    }

    # ---------- Enforce desired biomarker order on summary ----------
    order_map = {feat: i for i, feat in enumerate(FEATURE_ORDER)}
    summary["__order"] = summary["FEATURE"].map(order_map)
    summary = (
        summary.sort_values("__order", kind="mergesort")
               .drop(columns=["__order"])
               .reset_index(drop=True)
    )
    # ---------------------------------------------------------------

    return summary, per_feature_tables



# ---- usage (example) ----
detailed_ct, counts_ct, counts_feat = ttests_treat_vs_control(control, treat, FEATURES, alpha=0.05, fdr_alpha=0.05)
detailed_tg, counts_ct_tg, counts_feat_tg = ttests_realHigh_vs_genHigh(treat, generated, FEATURES, alpha=0.05, fdr_alpha=0.05)
overlap_summary, per_feature_marks = match_significant_groups_fdr(detailed_ct, detailed_tg)
overlap_summary






