# ==========================================
# Canonical Co-Elevation Agreement (CT vs TG)
# Reproducible version (with selective rounding) + metrics
# ==========================================

# ---- Reproducibility (must come first) ----
import os
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import random
random.seed(42)

import numpy as np
np.random.seed(42)

import pandas as pd
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings("ignore", message="Precision loss.*", category=RuntimeWarning)

# -----------------------
# I/O (your paths)
# -----------------------
control_real = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_test_control_cv2.csv')
treatment_real = pd.read_csv('/account001/mansi.chandra/clin_path/repeat_test_treatment_cv2.csv')
real = pd.concat([control_real, treatment_real], ignore_index=True, sort=False)

gen = pd.read_csv('/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/predictions_decoded/test/generated_predictions_1507935_ControlGenerator_test.csv')

# -----------------------
# Harmonize GEN rounding (keep float64) — do NOT touch GTP/CRE/Ca/Cl/Na/K/IP
# -----------------------
def rn(x, nd=None):
    s = pd.to_numeric(x, errors="coerce")
    if nd is not None:
        s = s.round(nd)
    return s.astype("float64")

# Round selected analytes only
if "TBIL(mg/dL)" in gen.columns: gen["TBIL(mg/dL)"] = rn(gen["TBIL(mg/dL)"], 2)
if "DBIL(mg/dL)" in gen.columns: gen["DBIL(mg/dL)"] = rn(gen["DBIL(mg/dL)"], 2)
if "RALB(g/dL)" in gen.columns:  gen["RALB(g/dL)"]  = rn(gen["RALB(g/dL)"], 1)
if "AST(IU/L)" in gen.columns:   gen["AST(IU/L)"]   = rn(gen["AST(IU/L)"], 0)
if "ALT(IU/L)" in gen.columns:   gen["ALT(IU/L)"]   = rn(gen["ALT(IU/L)"], 0)
if "ALP(IU/L)" in gen.columns:   gen["ALP(IU/L)"]   = rn(gen["ALP(IU/L)"], 0)
if "LDH(IU/L)" in gen.columns:   gen["LDH(IU/L)"]   = rn(gen["LDH(IU/L)"], 0)
if "TP(g/dL)" in gen.columns:    gen["TP(g/dL)"]    = rn(gen["TP(g/dL)"], 1)
if "BUN(mg/dL)" in gen.columns:  gen["BUN(mg/dL)"]  = rn(gen["BUN(mg/dL)"], 0)

# (Intentionally no rounding for:)
# "GTP(IU/L)", "CRE(mg/dL)", "K(meq/L)", "Na(meq/L)", "Cl(meq/L)", "Ca(mg/dL)", "IP(mg/dL)"

# Generated rows are control-equivalent profiles conditioned on High
gen_gc = gen[gen["DOSE_LEVEL"] == "High"]

# -----------------------
# Cohorts & config
# -----------------------
KEYS = ["COMPOUND_NAME", "SACRIFICE_PERIOD"]
control   = real[real['DOSE_LEVEL'] == 'Control'].copy()
treat     = real[real['DOSE_LEVEL'] == 'High'].copy()
gen_ctrl  = gen_gc.copy()

FEATURES = {
    "AST":  ["AST(IU/L)"],
    "ALT":  ["ALT(IU/L)"],
    "TBIL": ["TBIL(mg/dL)"],
    "DBIL": ["DBIL(mg/dL)"],
    "ALP":  ["ALP(IU/L)"],
    "LDH":  ["LDH(IU/L)"],
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

CANONICAL_EDGES = [
    ("ALT","AST"),
    ("ALP","TBIL"),
    ("ALP","GTP"),
    #("ALP","LDH"),
    ("BUN","CRE"),
]
CANONICAL_TRIADS = [
    ("ALT","AST","LDH"),
]

# -----------------------
# Inference settings
# -----------------------
UNIVERSE = "all"        # "all" (primary) or "edge"
USE_FDR = True
ALPHA = 0.05
FDR_ALPHA = 0.05
EFFECT_MIN_UP = 0.0
ALPHA_STRICT = ALPHA / max(1, len(FEATURES))

out_dir = "/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/coelev_canonical_outputs/pair_scatter_kidney"
os.makedirs(out_dir, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def _to_floats(arr_like):
    a = pd.to_numeric(pd.Series(arr_like), errors="coerce").astype("float64").to_numpy()
    return a[np.isfinite(a)]

def _degenerate_equal(x, y, eps=1e-12):
    if x.size >= 2 and y.size >= 2:
        if (np.nanstd(x, ddof=1) < eps) and (np.nanstd(y, ddof=1) < eps):
            if abs(np.nanmean(x) - np.nanmean(y)) < eps:
                return True
    return False

def _ttest_right(x, y):
    x = _to_floats(x); y = _to_floats(y)
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan
    if _degenerate_equal(x, y):
        return 0.0, 1.0
    try:
        res = ttest_ind(x, y, equal_var=False, nan_policy="omit", alternative="greater")
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        t_stat, p_two = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        if not np.isfinite(t_stat) or not np.isfinite(p_two):
            return np.nan, np.nan
        p_right = (p_two / 2.0) if (t_stat > 0) else (1.0 - p_two / 2.0)
        return float(t_stat), float(p_right)

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float); n = p.size
    if n == 0: return np.array([], dtype=float)
    order = np.argsort(p); q = np.empty_like(p); min_so_far = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        q_i = p[idx] * n / (n - rank + 1)
        min_so_far = min(min_so_far, q_i); q[idx] = min_so_far
    return np.minimum(q, 1.0)

def _add_fdr_by_group(df: pd.DataFrame, p_col="p_one_sided",
                      group_keys=KEYS, fdr_alpha=FDR_ALPHA, flag_col="significant_fdr") -> pd.DataFrame:
    parts = []
    for _, g in df.groupby(group_keys, dropna=False, sort=True):  # sort=True for determinism
        g = g.copy()
        p = g[p_col].fillna(1.0).values
        g["q_value"] = _bh_fdr(p)
        g[flag_col] = g["q_value"] <= fdr_alpha
        parts.append(g)
    out = pd.concat(parts, ignore_index=True) if parts else df.assign(q_value=np.nan, **{flag_col: False})
    return out.sort_values(group_keys + ["FEATURE"]).reset_index(drop=True)

# -----------------------
# Elevation calls: High > Ref (CT and TG)
# -----------------------
def ttests_by_group(ref_df, hi_df, *, keys=KEYS, features=FEATURES,
                    alpha=ALPHA, fdr_alpha=FDR_ALPHA, effect_floor_up=EFFECT_MIN_UP):
    keys_ref = ref_df[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    keys_hi  = hi_df[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    both = keys_ref.merge(keys_hi, on=keys, how="inner")

    rows = []
    for _, grp in both.iterrows():
        m_r = (ref_df[keys[0]] == grp[keys[0]]) & (ref_df[keys[1]] == grp[keys[1]])
        m_h = (hi_df[keys[0]]  == grp[keys[0]]) & (hi_df[keys[1]]  == grp[keys[1]])
        r_grp, h_grp = ref_df.loc[m_r], hi_df.loc[m_h]
        for feat, cols in features.items():
            col = cols[0]
            if col not in r_grp.columns or col not in h_grp.columns:
                continue
            x = _to_floats(h_grp[col])  # High
            y = _to_floats(r_grp[col])  # Reference
            n_x, n_y = x.size, y.size
            mean_x = float(np.mean(x)) if n_x else np.nan
            mean_y = float(np.mean(y)) if n_y else np.nan
            if n_x < 2 or n_y < 2:
                rows.append({**grp.to_dict(), "FEATURE": feat,
                             "n_high": n_x, "n_ref": n_y,
                             "mean_high": mean_x, "mean_ref": mean_y,
                             "effect": (mean_x - mean_y) if (n_x and n_y) else np.nan,
                             "t_stat": np.nan, "p_one_sided": np.nan,
                             "significant_raw": False,
                             "tested": False})
                continue
            t_stat, p_right = _ttest_right(x, y)
            effect = mean_x - mean_y
            rows.append({**grp.to_dict(), "FEATURE": feat,
                         "n_high": n_x, "n_ref": n_y,
                         "mean_high": mean_x, "mean_ref": mean_y,
                         "effect": float(effect),
                         "t_stat": float(t_stat), "p_one_sided": float(p_right),
                         "significant_raw": bool(np.isfinite(p_right) and (p_right <= alpha)),
                         "tested": True})
    detailed = pd.DataFrame(rows).sort_values(keys + ["FEATURE"]).reset_index(drop=True)

    if not detailed.empty:
        if USE_FDR:
            detailed = _add_fdr_by_group(detailed, p_col="p_one_sided", group_keys=keys,
                                         fdr_alpha=fdr_alpha, flag_col="significant_fdr")
            thr_mask = detailed["significant_fdr"]
        else:
            detailed["q_value"] = detailed["p_one_sided"]
            thr_mask = detailed["p_one_sided"] <= ALPHA_STRICT
        detailed["significant_fdr"] = thr_mask
        detailed["significant_fdr_elev"] = thr_mask & (detailed["effect"] >= float(effect_floor_up))
    else:
        detailed = pd.DataFrame(columns=[*keys,"FEATURE","n_high","n_ref","mean_high","mean_ref","effect",
                                         "t_stat","p_one_sided","significant_raw","tested",
                                         "q_value","significant_fdr","significant_fdr_elev"])
    return detailed

# Run tests
detailed_ct = ttests_by_group(ref_df=control,  hi_df=treat)   # Treat vs real Control
detailed_tg = ttests_by_group(ref_df=gen_ctrl, hi_df=treat)   # Treat vs generated Control

# -----------------------
# Build elevated sets per stratum, restricted to canonical features
# Also build "tested eligibility" maps (n>=2 per side) for denominators.
# -----------------------
canonical_features = set({f for a,b in CANONICAL_EDGES for f in (a,b)} |
                         {f for tri in CANONICAL_TRIADS for f in tri})

def elevated_features_by_stratum_all(detailed: pd.DataFrame, canonical_feats:set, keys=KEYS):
    df = detailed[(detailed["significant_fdr_elev"] == True) & (detailed["FEATURE"].isin(canonical_feats))].copy()
    got = (df.groupby(keys, dropna=False, sort=True)["FEATURE"]
             .apply(lambda s: set(s.tolist()))
             .reset_index(name="elevated_set"))
    strata = detailed[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    out = strata.merge(got, on=keys, how="left")
    out["elevated_set"] = out["elevated_set"].apply(lambda x: x if isinstance(x, set) else set())
    return out.sort_values(keys).reset_index(drop=True)

def _sort_cols_after_pivot(df: pd.DataFrame, keys=KEYS) -> pd.DataFrame:
    key_cols = list(keys)
    feat_cols = sorted([c for c in df.columns if c not in key_cols])
    return df[key_cols + feat_cols]

def tested_map(detailed: pd.DataFrame, keys=KEYS):
    df = detailed.loc[detailed["tested"] == True, keys + ["FEATURE"]].drop_duplicates()
    pv = (df.assign(val=True)
            .pivot_table(index=keys, columns="FEATURE", values="val", aggfunc="max", fill_value=False)
            .reset_index())
    return _sort_cols_after_pivot(pv, keys)

ct_sets_all = elevated_features_by_stratum_all(detailed_ct, canonical_features, keys=KEYS)
tg_sets_all = elevated_features_by_stratum_all(detailed_tg, canonical_features, keys=KEYS)

elig_ct = tested_map(detailed_ct, keys=KEYS)
elig_tg = tested_map(detailed_tg, keys=KEYS)
elig_union = elig_ct.merge(elig_tg, on=KEYS, how="outer", suffixes=("_CT","_TG")).fillna(False)
elig_union = _sort_cols_after_pivot(elig_union, keys=KEYS)

# -----------------------
# Choose universe: "all" vs "edge"
# -----------------------
if UNIVERSE == "edge":
    merged_sets = (ct_sets_all.merge(tg_sets_all, on=KEYS, how="inner", suffixes=("_ct","_tg"))
                              .query("elevated_set_ct.apply(len) > 0 and elevated_set_tg.apply(len) > 0")
                              .sort_values(KEYS).reset_index(drop=True))
else:
    merged_sets = ct_sets_all.merge(tg_sets_all, on=KEYS, how="outer", suffixes=("_ct","_tg"))
    merged_sets["elevated_set_ct"] = merged_sets["elevated_set_ct"].apply(lambda x: x if isinstance(x, set) else set())
    merged_sets["elevated_set_tg"] = merged_sets["elevated_set_tg"].apply(lambda x: x if isinstance(x, set) else set())
    merged_sets = merged_sets.sort_values(KEYS).reset_index(drop=True)

# -----------------------
# Pair/triad presence + denominators inside chosen universe
# -----------------------
def edge_present(feat_set:set, edge:tuple): return set(edge).issubset(feat_set)
def triad_present(feat_set:set, tri:tuple): return set(tri).issubset(feat_set)

# Build eligibility tables restricted to UNIVERSE
univ_keys = merged_sets[KEYS].drop_duplicates()
elig_ct_u = _sort_cols_after_pivot(univ_keys.merge(elig_ct, on=KEYS, how="left").fillna(False), keys=KEYS)
elig_tg_u = _sort_cols_after_pivot(univ_keys.merge(elig_tg, on=KEYS, how="left").fillna(False), keys=KEYS)
elig_u    = _sort_cols_after_pivot(elig_ct_u.merge(elig_tg_u, on=KEYS, how="inner", suffixes=("_CT","_TG")), keys=KEYS)

# ----- EDGES -----
rows = []
for _, r in merged_sets.iterrows():
    ct_set = r["elevated_set_ct"]; tg_set = r["elevated_set_tg"]
    for a,b in CANONICAL_EDGES:
        rows.append({
            KEYS[0]: r[KEYS[0]],
            KEYS[1]: r[KEYS[1]],
            "FEATURE_A": a, "FEATURE_B": b,
            "present_ct": edge_present(ct_set, (a,b)),
            "present_tg": edge_present(tg_set, (a,b)),
        })
edge_presence = pd.DataFrame(rows).sort_values(KEYS + ["FEATURE_A","FEATURE_B"]).reset_index(drop=True)
edge_presence["present_ct"] = edge_presence["present_ct"].astype(bool)
edge_presence["present_tg"] = edge_presence["present_tg"].astype(bool)

# Denominators per pair within UNIVERSE
def pair_denominators(pair):
    a,b = pair
    n_ct = int((elig_ct_u.get(a, False) & elig_ct_u.get(b, False)).sum())
    n_tg = int((elig_tg_u.get(a, False) & elig_tg_u.get(b, False)).sum())
    n_union = int((
        (elig_u.get(f"{a}_CT", False) & elig_u.get(f"{b}_CT", False)) |
        (elig_u.get(f"{a}_TG", False) & elig_u.get(f"{b}_TG", False))
    ).sum())
    return n_ct, n_tg, n_union

edge_stats = (
    edge_presence
    .groupby(["FEATURE_A","FEATURE_B"], dropna=False, sort=True)
    .apply(lambda g: pd.Series({
        "n_strata": len(g),
        "n_ct": int(g["present_ct"].sum()),
        "n_tg": int(g["present_tg"].sum()),
        "n_overlap": int((g["present_ct"] & g["present_tg"]).sum())
    }))
    .reset_index()
    .sort_values(["FEATURE_A","FEATURE_B"])
    .reset_index(drop=True)
)

# Attach denominators
den_rows = []
for _, row in edge_stats.iterrows():
    pair = (row["FEATURE_A"], row["FEATURE_B"])
    d_ct, d_tg, d_union = pair_denominators(pair)
    den_rows.append((d_ct, d_tg, d_union))
den_df = pd.DataFrame(den_rows, columns=["n_total_CT","n_total_TG","n_total_union"])
edge_stats = pd.concat([edge_stats, den_df], axis=1)

# Core set similarity (for reference)
def jaccard_counts(n_ct, n_tg, n_ov):
    denom = (n_ct + n_tg - n_ov)
    return (n_ov / denom) if denom > 0 else np.nan

edge_stats["jaccard_edge"] = edge_stats.apply(
    lambda r: jaccard_counts(r["n_ct"], r["n_tg"], r["n_overlap"]), axis=1
)

# ---------- NEW: confusion-based metrics for edges (recall, specificity, BA, MCC) ----------
def _edge_confusion_for_pair(pair, edge_presence, elig_ct_u, elig_tg_u, keys=KEYS):
    a, b = pair
    both_elig_series = (
        elig_ct_u.get(a, False) & elig_ct_u.get(b, False) &
        elig_tg_u.get(a, False) & elig_tg_u.get(b, False)
    ).astype(bool)

    both_elig_df = elig_ct_u[keys].copy()
    both_elig_df["both_elig"] = both_elig_series.values

    ep = (edge_presence.merge(both_elig_df, on=keys, how="left")
                        .fillna({"both_elig": False}))
    mask = ep["both_elig"] & ep["FEATURE_A"].eq(a) & ep["FEATURE_B"].eq(b)
    e = ep.loc[mask, ["present_ct","present_tg"]].astype(bool)

    tp = int((e["present_ct"] &  e["present_tg"]).sum())
    fp = int((~e["present_ct"] & e["present_tg"]).sum())
    fn = int((e["present_ct"] & ~e["present_tg"]).sum())
    tn = int((~e["present_ct"] & ~e["present_tg"]).sum())
    n  = tp + fp + fn + tn

    rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ba   = np.nan if (np.isnan(rec) or np.isnan(spec)) else 0.5*(rec + spec)

    mcc_den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = ((tp*tn) - (fp*fn)) / mcc_den if mcc_den > 0 else np.nan

    jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    return tp, fp, fn, tn, n, rec, spec, ba, mcc, jacc

# Build edge metrics table
edge_metrics_rows = []
for _, r in edge_stats.iterrows():
    pair = (r["FEATURE_A"], r["FEATURE_B"])
    tp, fp, fn, tn, n, rec, spec, ba, mcc, jacc = _edge_confusion_for_pair(pair, edge_presence, elig_ct_u, elig_tg_u, keys=KEYS)
    edge_metrics_rows.append({
        "FEATURE_A": pair[0], "FEATURE_B": pair[1],
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "N": n,
        "recall_edge": rec,
        "specificity_edge": spec,
        "balanced_accuracy_edge": ba,
        "mcc_edge": mcc,
        "jaccard_edge_recomputed": jacc
    })
edge_metrics = pd.DataFrame(edge_metrics_rows).sort_values(["FEATURE_A","FEATURE_B"]).reset_index(drop=True)

# Merge metrics back into edge_stats for one consolidated table
edge_stats = (edge_stats
              .merge(edge_metrics, on=["FEATURE_A","FEATURE_B"], how="left")
              .sort_values(["FEATURE_A","FEATURE_B"])
              .reset_index(drop=True))

# ----- TRIADS -----
tri_rows = []
for _, r in merged_sets.iterrows():
    ct_set = r["elevated_set_ct"]; tg_set = r["elevated_set_tg"]
    for tri in CANONICAL_TRIADS:
        tri_rows.append({
            KEYS[0]: r[KEYS[0]],
            KEYS[1]: r[KEYS[1]],
            "TRIAD": "+".join(tri),
            "present_ct": triad_present(ct_set, tri),
            "present_tg": triad_present(tg_set, tri),
        })
tri_presence = pd.DataFrame(tri_rows) if len(CANONICAL_TRIADS)>0 else pd.DataFrame(columns=[*KEYS,"TRIAD","present_ct","present_tg"])

if not tri_presence.empty:
    tri_presence = tri_presence.sort_values(KEYS + ["TRIAD"]).reset_index(drop=True)
    tri_presence["present_ct"] = tri_presence["present_ct"].astype(bool)
    tri_presence["present_tg"] = tri_presence["present_tg"].astype(bool)

    tri_stats = (
        tri_presence
        .groupby(["TRIAD"], dropna=False, sort=True)
        .apply(lambda g: pd.Series({
            "n_strata": len(g),
            "n_ct": int(g["present_ct"].sum()),
            "n_tg": int(g["present_tg"].sum()),
            "n_overlap": int((g["present_ct"] & g["present_tg"]).sum())
        }))
        .reset_index()
        .sort_values(["TRIAD"])
        .reset_index(drop=True)
    )

    # Triad denominators
    for i, row in tri_stats.iterrows():
        ta, tb, tc = CANONICAL_TRIADS[0]
        n_total_CT = int((elig_ct_u.get(ta, False) & elig_ct_u.get(tb, False) & elig_ct_u.get(tc, False)).sum())
        n_total_TG = int((elig_tg_u.get(ta, False) & elig_tg_u.get(tb, False) & elig_tg_u.get(tc, False)).sum())
        n_total_union = int((
            (elig_u.get(f"{ta}_CT", False) & elig_u.get(f"{tb}_CT", False) & elig_u.get(f"{tc}_CT", False)) |
            (elig_u.get(f"{ta}_TG", False) & elig_u.get(f"{tb}_TG", False) & elig_u.get(f"{tc}_TG", False))
        ).sum())
        tri_stats.loc[i, "n_total_CT"] = n_total_CT
        tri_stats.loc[i, "n_total_TG"] = n_total_TG
        tri_stats.loc[i, "n_total_union"] = n_total_union

    # Jaccard from counts (reference)
    def jaccard_counts(n_ct, n_tg, n_ov):
        denom = (n_ct + n_tg - n_ov)
        return (n_ov / denom) if denom > 0 else np.nan

    tri_stats["jaccard_triad"] = tri_stats.apply(
        lambda r: jaccard_counts(r["n_ct"], r["n_tg"], r["n_overlap"]), axis=1
    )
    tri_stats = tri_stats.sort_values(
        ["jaccard_triad","n_overlap","n_ct","n_tg","TRIAD"],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

    # ---------- NEW: confusion-based metrics for triads ----------
    def _triad_confusion(tri, tri_presence, elig_ct_u, elig_tg_u, keys=KEYS):
        a, b, c = tri
        both_elig_series = (
            elig_ct_u.get(a, False) & elig_ct_u.get(b, False) & elig_ct_u.get(c, False) &
            elig_tg_u.get(a, False) & elig_tg_u.get(b, False) & elig_tg_u.get(c, False)
        ).astype(bool)

        both_elig_df = elig_ct_u[keys].copy()
        both_elig_df["both_elig"] = both_elig_series.values

        tpv = (tri_presence.merge(both_elig_df, on=keys, how="left")
                             .fillna({"both_elig": False}))

        tri_name = "+".join(tri)
        mask = tpv["TRIAD"].eq(tri_name) & tpv["both_elig"]
        e = tpv.loc[mask, ["present_ct","present_tg"]].astype(bool)

        tp = int((e["present_ct"] &  e["present_tg"]).sum())
        fp = int((~e["present_ct"] & e["present_tg"]).sum())
        fn = int((e["present_ct"] & ~e["present_tg"]).sum())
        tn = int((~e["present_ct"] & ~e["present_tg"]).sum())
        n  = tp + fp + fn + tn

        rec  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ba   = np.nan if (np.isnan(rec) or np.isnan(spec)) else 0.5*(rec + spec)

        mcc_den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc = ((tp*tn) - (fp*fn)) / mcc_den if mcc_den > 0 else np.nan

        jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
        return tp, fp, fn, tn, n, rec, spec, ba, mcc, jacc

    tri_metrics_rows = []
    for tri in CANONICAL_TRIADS:
        tp, fp, fn, tn, n, rec, spec, ba, mcc, jacc = _triad_confusion(tri, tri_presence, elig_ct_u, elig_tg_u, keys=KEYS)
        tri_metrics_rows.append({
            "TRIAD": "+".join(tri),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn, "N": n,
            "recall_triad": rec,
            "specificity_triad": spec,
            "balanced_accuracy_triad": ba,
            "mcc_triad": mcc,
            "jaccard_triad_recomputed": jacc
        })
    tri_metrics = pd.DataFrame(tri_metrics_rows)

    # Merge triad metrics into tri_stats
    tri_stats = (tri_stats
                 .merge(tri_metrics, on="TRIAD", how="left")
                 .sort_values(["TRIAD"])
                 .reset_index(drop=True))
else:
    tri_stats = pd.DataFrame(columns=[
        "TRIAD","n_strata","n_ct","n_tg","n_overlap","n_total_CT","n_total_TG",
        "n_total_union","jaccard_triad","TP","FP","FN","TN","N",
        "recall_triad","specificity_triad","balanced_accuracy_triad","mcc_triad","jaccard_triad_recomputed"
    ])

# -----------------------
# Summaries
# -----------------------
def _nanmean_safe(s):
    try:
        return float(np.nanmean(s)) if len(s) else np.nan
    except Exception:
        return np.nan

edge_summary = pd.DataFrame({
    "universe": [UNIVERSE],
    "n_edges_considered": [len(CANONICAL_EDGES)],
    "mean_jaccard_edge": [_nanmean_safe(edge_stats["jaccard_edge"])],
    "median_jaccard_edge": [np.nanmedian(edge_stats["jaccard_edge"]) if len(edge_stats) else np.nan],
    "mean_recall_edge": [_nanmean_safe(edge_stats["recall_edge"])],
    "mean_specificity_edge": [_nanmean_safe(edge_stats["specificity_edge"])],
    "mean_balanced_accuracy_edge": [_nanmean_safe(edge_stats["balanced_accuracy_edge"])],
    "mean_mcc_edge": [_nanmean_safe(edge_stats["mcc_edge"])],
    "prop_perfect_jaccard_1.0": [np.mean(edge_stats["jaccard_edge"]==1.0) if len(edge_stats) else np.nan],
    "prop_zero_jaccard_0.0": [np.mean(edge_stats["jaccard_edge"]==0.0) if len(edge_stats) else np.nan],
})

if len(CANONICAL_TRIADS) > 0 and not tri_stats.empty:
    tri_summary = pd.DataFrame({
        "universe": [UNIVERSE],
        "n_triads_considered": [len(CANONICAL_TRIADS)],
        "mean_jaccard_triad": [_nanmean_safe(tri_stats["jaccard_triad"])],
        "median_jaccard_triad": [np.nanmedian(tri_stats["jaccard_triad"]) if len(tri_stats) else np.nan],
        "mean_recall_triad": [_nanmean_safe(tri_stats["recall_triad"])],
        "mean_specificity_triad": [_nanmean_safe(tri_stats["specificity_triad"])],
        "mean_balanced_accuracy_triad": [_nanmean_safe(tri_stats["balanced_accuracy_triad"])],
        "mean_mcc_triad": [_nanmean_safe(tri_stats["mcc_triad"])],
        "prop_perfect_jaccard_1.0": [np.mean(tri_stats["jaccard_triad"]==1.0) if len(tri_stats) else np.nan],
        "prop_zero_jaccard_0.0": [np.mean(tri_stats["jaccard_triad"]==0.0) if len(tri_stats) else np.nan],
    })
else:
    tri_summary = pd.DataFrame({
        "universe":[UNIVERSE],
        "n_triads_considered":[len(CANONICAL_TRIADS)],
        "mean_jaccard_triad":[np.nan],
        "median_jaccard_triad":[np.nan],
        "mean_recall_triad":[np.nan],
        "mean_specificity_triad":[np.nan],
        "mean_balanced_accuracy_triad":[np.nan],
        "mean_mcc_triad":[np.nan],
        "prop_perfect_jaccard_1.0":[np.nan],
        "prop_zero_jaccard_0.0":[np.nan],
    })

# -----------------------
# SAVE (stable float formatting)
# -----------------------
ffmt = "%.10g"
os.makedirs(out_dir, exist_ok=True)

edge_stats.sort_values(["FEATURE_A","FEATURE_B"]).to_csv(
    os.path.join(out_dir, f"canonical_edge_agreement_{UNIVERSE}.csv"),
    index=False, float_format=ffmt
)
edge_summary.to_csv(os.path.join(out_dir, f"canonical_edge_summary_{UNIVERSE}.csv"),
                    index=False, float_format=ffmt)

tri_stats.sort_values(["TRIAD"] if "TRIAD" in tri_stats.columns else []).to_csv(
    os.path.join(out_dir, f"canonical_triad_agreement_{UNIVERSE}.csv"),
    index=False, float_format=ffmt
)
tri_summary.to_csv(os.path.join(out_dir, f"canonical_triad_summary_{UNIVERSE}.csv"),
                   index=False, float_format=ffmt)

# Console peek
print("\n=== Canonical co-elevation — edge summary ===")
print(edge_summary.to_string(index=False))
print("\nSpecified canonical edges (with denominators & metrics):")
print(edge_stats.to_string(index=False))

if len(CANONICAL_TRIADS) > 0:
    print("\n=== Canonical triad summary ===")
    print(tri_summary.to_string(index=False))
    print("\nTriad details (with denominators & metrics):")
    print(tri_stats.to_string(index=False))
