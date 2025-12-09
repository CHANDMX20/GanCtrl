"""
Canonical Co-Elevation Agreement Analysis (CT vs TG)
====================================================

Goal
----
Compare canonical co-elevation patterns between:

1. Real data:
   - Control animals (DOSE_LEVEL == "Control")
   - High-dose animals (DOSE_LEVEL == "High")

2. Generated data:
   - Control-equivalent profiles for High-dose conditions from a generative model

We quantify how often canonical co-elevation motifs (edges & triads of biomarkers)
appear in:

- Real high vs real control (CT)
- Real high vs generated control (TG)

and compute overlap metrics such as:

- Jaccard index
- Recall, specificity, accuracy, balanced accuracy
- Matthews correlation coefficient (MCC)

Key assumptions
---------------
- Input clinical pathology files (real & generated) share:
  * DOSE_LEVEL
  * COMPOUND_NAME
  * SACRIFICE_PERIOD
  * Analyte columns listed in FEATURES
- Generated data contains DOSE_LEVEL == "High" rows representing synthetic controls.
- Canonical edges/triads are defined in CANONICAL_EDGES / CANONICAL_TRIADS.

Paths below are placeholders; update them to match your environment before use.
"""

# =============================================================================
# 0. Reproducibility: make numerical operations as deterministic as possible
# =============================================================================

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

# =============================================================================
# 1. I/O CONFIGURATION (UPDATE THESE PATHS FOR YOUR ENVIRONMENT)
# =============================================================================

# Clinical pathology real data
CONTROL_FILE = "/path/to/clin_path/repeat_test_control_cv2.csv"
TREATMENT_FILE = "/path/to/clin_path/repeat_test_treatment_cv2.csv"

# Generated control-like predictions conditioned on High
GENERATED_FILE = (
    "/path/to/results_vae_corr_mod3_cv2/"
    "predictions_decoded/test/generated_predictions_control_like.csv"
)

# Output folder for canonical co-elevation metrics
OUTPUT_DIR = (
    "/path/to/results_vae_corr_mod3_cv2/"
    "coelev_canonical_outputs/pair_scatter_kidney"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. LOAD REAL & GENERATED DATA
# =============================================================================

control_real = pd.read_csv(CONTROL_FILE)
treatment_real = pd.read_csv(TREATMENT_FILE)
real = pd.concat([control_real, treatment_real], ignore_index=True, sort=False)

gen = pd.read_csv(GENERATED_FILE)

# =============================================================================
# 3. HARMONIZE GENERATED ROUNDING (KEEP float64; DO NOT TOUCH SOME ANALYTES)
# =============================================================================

def rn(x: pd.Series, nd: int | None = None) -> pd.Series:
    """
    Convert a series to numeric (coerce invalid to NaN), round to 'nd' decimals
    if provided, and cast to float64.

    Parameters
    ----------
    x : pd.Series
        Input series.
    nd : int or None
        Number of decimal places for rounding. If None, no rounding is applied.

    Returns
    -------
    pd.Series
        Float64 series with optional rounding applied.
    """
    s = pd.to_numeric(x, errors="coerce")
    if nd is not None:
        s = s.round(nd)
    return s.astype("float64")

# Round selected analytes only (others intentionally left untouched)
if "TBIL(mg/dL)" in gen.columns:
    gen["TBIL(mg/dL)"] = rn(gen["TBIL(mg/dL)"], 2)
if "DBIL(mg/dL)" in gen.columns:
    gen["DBIL(mg/dL)"] = rn(gen["DBIL(mg/dL)"], 2)
if "RALB(g/dL)" in gen.columns:
    gen["RALB(g/dL)"] = rn(gen["RALB(g/dL)"], 1)
if "AST(IU/L)" in gen.columns:
    gen["AST(IU/L)"] = rn(gen["AST(IU/L)"], 0)
if "ALT(IU/L)" in gen.columns:
    gen["ALT(IU/L)"] = rn(gen["ALT(IU/L)"], 0)
if "ALP(IU/L)" in gen.columns:
    gen["ALP(IU/L)"] = rn(gen["ALP(IU/L)"], 0)
if "LDH(IU/L)" in gen.columns:
    gen["LDH(IU/L)"] = rn(gen["LDH(IU/L)"], 0)
if "TP(g/dL)" in gen.columns:
    gen["TP(g/dL)"] = rn(gen["TP(g/dL)"], 1)
if "BUN(mg/dL)" in gen.columns:
    gen["BUN(mg/dL)"] = rn(gen["BUN(mg/dL)"], 0)

# NOTE: We intentionally do NOT round the following:
# "GTP(IU/L)", "CRE(mg/dL)", "K(meq/L)", "Na(meq/L)",
# "Cl(meq/L)", "Ca(mg/dL)", "IP(mg/dL)"

# Generated rows are control-equivalent profiles conditioned on High
gen_gc = gen[gen["DOSE_LEVEL"] == "High"]

# =============================================================================
# 4. COHORTS & FEATURE CONFIGURATION
# =============================================================================

KEYS = ["COMPOUND_NAME", "SACRIFICE_PERIOD"]

# Split real data into control and high-dose groups
control = real[real["DOSE_LEVEL"] == "Control"].copy()
treat = real[real["DOSE_LEVEL"] == "High"].copy()
gen_ctrl = gen_gc.copy()

# Mapping from short biomarker names to the corresponding column names
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

# Canonical co-elevation patterns to evaluate
CANONICAL_EDGES = [
    ("ALT", "AST"),
    ("ALP", "TBIL"),
    ("ALP", "GTP"),
    # ("ALP", "LDH"),  # optional edge, currently commented out
    ("BUN", "CRE"),
]

CANONICAL_TRIADS = [
    ("ALT", "AST", "LDH"),
]

# =============================================================================
# 5. INFERENCE CONFIGURATION
# =============================================================================

# Universe of strata:
#   "all"  - all (COMPOUND_NAME, SACRIFICE_PERIOD) strata
#   "edge" - only strata where both CT and TG have at least one canonical edge
UNIVERSE = "all"          # primary setting

USE_FDR = True            # True -> BH-FDR within each (compound, time)
ALPHA = 0.05              # per-test alpha level for raw p-values
FDR_ALPHA = 0.05          # FDR threshold for q-values
EFFECT_MIN_UP = 0.0       # minimum (mean_high - mean_ref) for "elevated"
ALPHA_STRICT = ALPHA / max(1, len(FEATURES))  # Bonferroni if USE_FDR=False

# =============================================================================
# 6. HELPER FUNCTIONS
# =============================================================================

def _to_floats(arr_like) -> np.ndarray:
    """
    Convert an array-like object to a 1D float64 NumPy array, dropping NaNs
    and non-finite values.
    """
    a = pd.to_numeric(pd.Series(arr_like), errors="coerce").astype("float64").to_numpy()
    return a[np.isfinite(a)]


def _degenerate_equal(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> bool:
    """
    Detect a degenerate case where both x and y have near-zero variance and
    essentially identical means (to avoid unstable t-tests).
    """
    if x.size >= 2 and y.size >= 2:
        if (np.nanstd(x, ddof=1) < eps) and (np.nanstd(y, ddof=1) < eps):
            if abs(np.nanmean(x) - np.nanmean(y)) < eps:
                return True
    return False


def _ttest_right(x, y):
    """
    One-sided Welch t-test H0: mean(x) <= mean(y), H1: mean(x) > mean(y).

    Parameters
    ----------
    x, y : array-like
        Sample vectors.

    Returns
    -------
    t_stat : float
        t-statistic.
    p_right : float
        One-sided p-value for H1: x > y.
    """
    x = _to_floats(x)
    y = _to_floats(y)
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan
    if _degenerate_equal(x, y):
        # If they are effectively identical, treat as no difference.
        return 0.0, 1.0

    try:
        res = ttest_ind(x, y, equal_var=False, nan_policy="omit", alternative="greater")
        return float(res.statistic), float(res.pvalue)
    except TypeError:
        # For older SciPy versions without 'alternative' parameter.
        t_stat, p_two = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        if not np.isfinite(t_stat) or not np.isfinite(p_two):
            return np.nan, np.nan
        p_right = (p_two / 2.0) if (t_stat > 0) else (1.0 - p_two / 2.0)
        return float(t_stat), float(p_right)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    pvals : np.ndarray
        Array of p-values.

    Returns
    -------
    np.ndarray
        Array of q-values (FDR-adjusted p-values).
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(p)
    q = np.empty_like(p)
    min_so_far = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        q_i = p[idx] * n / (n - rank + 1)
        min_so_far = min(min_so_far, q_i)
        q[idx] = min_so_far
    return np.minimum(q, 1.0)


def _add_fdr_by_group(
    df: pd.DataFrame,
    p_col: str = "p_one_sided",
    group_keys=KEYS,
    fdr_alpha: float = FDR_ALPHA,
    flag_col: str = "significant_fdr",
) -> pd.DataFrame:
    """
    Apply BH-FDR within each (COMPOUND_NAME, SACRIFICE_PERIOD) group.

    Adds:
      - q_value (group-wise FDR-adjusted p-value)
      - boolean column 'flag_col' marking q <= fdr_alpha.
    """
    parts = []
    for _, g in df.groupby(group_keys, dropna=False, sort=True):  # sort=True for determinism
        g = g.copy()
        p = g[p_col].fillna(1.0).values
        g["q_value"] = _bh_fdr(p)
        g[flag_col] = g["q_value"] <= fdr_alpha
        parts.append(g)

    out = (
        pd.concat(parts, ignore_index=True)
        if parts
        else df.assign(q_value=np.nan, **{flag_col: False})
    )
    return out.sort_values(group_keys + ["FEATURE"]).reset_index(drop=True)

# =============================================================================
# 7. ELEVATION CALLS: High > Reference (CT and TG)
# =============================================================================

def ttests_by_group(
    ref_df: pd.DataFrame,
    hi_df: pd.DataFrame,
    *,
    keys=KEYS,
    features=FEATURES,
    alpha: float = ALPHA,
    fdr_alpha: float = FDR_ALPHA,
    effect_floor_up: float = EFFECT_MIN_UP,
) -> pd.DataFrame:
    """
    For each (compound, time, feature), compare High vs Reference using a
    one-sided Welch t-test (High > Reference).

    Returns a detailed DataFrame with:
      - group keys (COMPOUND_NAME, SACRIFICE_PERIOD)
      - n_high, n_ref, means, effect (mean_high - mean_ref)
      - t_stat, p_one_sided
      - FDR-adjusted q_value (if USE_FDR=True)
      - significant_fdr_elev flag: FDR-significant AND effect >= effect_floor_up
    """
    keys_ref = ref_df[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    keys_hi = hi_df[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    both = keys_ref.merge(keys_hi, on=keys, how="inner")

    rows = []
    for _, grp in both.iterrows():
        m_r = (ref_df[keys[0]] == grp[keys[0]]) & (ref_df[keys[1]] == grp[keys[1]])
        m_h = (hi_df[keys[0]] == grp[keys[0]]) & (hi_df[keys[1]] == grp[keys[1]])
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
            effect = (mean_x - mean_y) if (n_x and n_y) else np.nan

            if n_x < 2 or n_y < 2:
                # Not enough replicates on either side for a t-test
                rows.append(
                    {
                        **grp.to_dict(),
                        "FEATURE": feat,
                        "n_high": n_x,
                        "n_ref": n_y,
                        "mean_high": mean_x,
                        "mean_ref": mean_y,
                        "effect": effect,
                        "t_stat": np.nan,
                        "p_one_sided": np.nan,
                        "significant_raw": False,
                        "tested": False,
                    }
                )
                continue

            t_stat, p_right = _ttest_right(x, y)
            rows.append(
                {
                    **grp.to_dict(),
                    "FEATURE": feat,
                    "n_high": n_x,
                    "n_ref": n_y,
                    "mean_high": mean_x,
                    "mean_ref": mean_y,
                    "effect": float(effect),
                    "t_stat": float(t_stat),
                    "p_one_sided": float(p_right),
                    "significant_raw": bool(np.isfinite(p_right) and (p_right <= alpha)),
                    "tested": True,
                }
            )

    detailed = (
        pd.DataFrame(rows)
        .sort_values(keys + ["FEATURE"])
        .reset_index(drop=True)
    )

    if not detailed.empty:
        if USE_FDR:
            detailed = _add_fdr_by_group(
                detailed,
                p_col="p_one_sided",
                group_keys=keys,
                fdr_alpha=fdr_alpha,
                flag_col="significant_fdr",
            )
            thr_mask = detailed["significant_fdr"]
        else:
            detailed["q_value"] = detailed["p_one_sided"]
            thr_mask = detailed["p_one_sided"] <= ALPHA_STRICT

        detailed["significant_fdr"] = thr_mask
        detailed["significant_fdr_elev"] = thr_mask & (
            detailed["effect"] >= float(effect_floor_up)
        )
    else:
        detailed = pd.DataFrame(
            columns=[
                *keys,
                "FEATURE",
                "n_high",
                "n_ref",
                "mean_high",
                "mean_ref",
                "effect",
                "t_stat",
                "p_one_sided",
                "significant_raw",
                "tested",
                "q_value",
                "significant_fdr",
                "significant_fdr_elev",
            ]
        )

    return detailed

# Run CT (High vs real Control) and TG (High vs generated Control)
detailed_ct = ttests_by_group(ref_df=control, hi_df=treat)    # Treat vs real Control
detailed_tg = ttests_by_group(ref_df=gen_ctrl, hi_df=treat)   # Treat vs generated Control

# =============================================================================
# 8. BUILD ELEVATED SETS PER STRATUM (RESTRICTED TO CANONICAL FEATURES)
#    + ELIGIBILITY MAPS FOR DENOMINATORS
# =============================================================================

canonical_features = set(
    {f for a, b in CANONICAL_EDGES for f in (a, b)}
    | {f for tri in CANONICAL_TRIADS for f in tri}
)

def elevated_features_by_stratum_all(
    detailed: pd.DataFrame,
    canonical_feats: set,
    keys=KEYS,
) -> pd.DataFrame:
    """
    Build, for each (compound, time) stratum, the set of canonical features
    that are FDR-significant and elevated.

    Returns a DataFrame with:
      keys + ["elevated_set"] where elevated_set is a Python set.
    """
    df = detailed[
        (detailed["significant_fdr_elev"] == True)
        & (detailed["FEATURE"].isin(canonical_feats))
    ].copy()

    got = (
        df.groupby(keys, dropna=False, sort=True)["FEATURE"]
        .apply(lambda s: set(s.tolist()))
        .reset_index(name="elevated_set")
    )

    strata = detailed[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    out = strata.merge(got, on=keys, how="left")
    out["elevated_set"] = out["elevated_set"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    return out.sort_values(keys).reset_index(drop=True)


def _sort_cols_after_pivot(df: pd.DataFrame, keys=KEYS) -> pd.DataFrame:
    """
    Ensure key columns are first, followed by other columns sorted alphabetically.
    """
    key_cols = list(keys)
    feat_cols = sorted([c for c in df.columns if c not in key_cols])
    return df[key_cols + feat_cols]


def tested_map(detailed: pd.DataFrame, keys=KEYS) -> pd.DataFrame:
    """
    Build a boolean map of which features were eligible for testing (n >= 2 per side)
    in each (compound, time) stratum.
    """
    df = detailed.loc[detailed["tested"] == True, keys + ["FEATURE"]].drop_duplicates()
    pv = (
        df.assign(val=True)
        .pivot_table(
            index=keys,
            columns="FEATURE",
            values="val",
            aggfunc="max",
            fill_value=False,
        )
        .reset_index()
    )
    return _sort_cols_after_pivot(pv, keys)


ct_sets_all = elevated_features_by_stratum_all(
    detailed_ct, canonical_features, keys=KEYS
)
tg_sets_all = elevated_features_by_stratum_all(
    detailed_tg, canonical_features, keys=KEYS
)

elig_ct = tested_map(detailed_ct, keys=KEYS)
elig_tg = tested_map(detailed_tg, keys=KEYS)
elig_union = elig_ct.merge(
    elig_tg, on=KEYS, how="outer", suffixes=("_CT", "_TG")
).fillna(False)
elig_union = _sort_cols_after_pivot(elig_union, keys=KEYS)

# =============================================================================
# 9. CHOOSE UNIVERSE: "all" vs "edge"
# =============================================================================

if UNIVERSE == "edge":
    # Only strata where both CT and TG have at least one canonical elevated feature
    merged_sets = (
        ct_sets_all.merge(
            tg_sets_all, on=KEYS, how="inner", suffixes=("_ct", "_tg")
        )
        .query(
            "elevated_set_ct.apply(len) > 0 and elevated_set_tg.apply(len) > 0"
        )
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
else:
    # Universe of all strata where we have CT and/or TG elevated sets
    merged_sets = ct_sets_all.merge(
        tg_sets_all, on=KEYS, how="outer", suffixes=("_ct", "_tg")
    )
    merged_sets["elevated_set_ct"] = merged_sets["elevated_set_ct"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    merged_sets["elevated_set_tg"] = merged_sets["elevated_set_tg"].apply(
        lambda x: x if isinstance(x, set) else set()
    )
    merged_sets = merged_sets.sort_values(KEYS).reset_index(drop=True)

# Eligibility restricted to UNIVERSE
univ_keys = merged_sets[KEYS].drop_duplicates()
elig_ct_u = _sort_cols_after_pivot(
    univ_keys.merge(elig_ct, on=KEYS, how="left").fillna(False), keys=KEYS
)
elig_tg_u = _sort_cols_after_pivot(
    univ_keys.merge(elig_tg, on=KEYS, how="left").fillna(False), keys=KEYS
)
elig_u = _sort_cols_after_pivot(
    elig_ct_u.merge(elig_tg_u, on=KEYS, how="inner", suffixes=("_CT", "_TG")),
    keys=KEYS,
)

# =============================================================================
# 10. EDGE-LEVEL PRESENCE & DENOMINATORS
# =============================================================================

def edge_present(feat_set: set, edge: tuple) -> bool:
    """Return True if both features in 'edge' are in feat_set."""
    return set(edge).issubset(feat_set)


def triad_present(feat_set: set, tri: tuple) -> bool:
    """Return True if all three features in 'tri' are in feat_set."""
    return set(tri).issubset(feat_set)


# Edge-level presence per stratum
rows = []
for _, r in merged_sets.iterrows():
    ct_set = r["elevated_set_ct"]
    tg_set = r["elevated_set_tg"]
    for a, b in CANONICAL_EDGES:
        rows.append(
            {
                KEYS[0]: r[KEYS[0]],
                KEYS[1]: r[KEYS[1]],
                "FEATURE_A": a,
                "FEATURE_B": b,
                "present_ct": edge_present(ct_set, (a, b)),
                "present_tg": edge_present(tg_set, (a, b)),
            }
        )

edge_presence = (
    pd.DataFrame(rows)
    .sort_values(KEYS + ["FEATURE_A", "FEATURE_B"])
    .reset_index(drop=True)
)
edge_presence["present_ct"] = edge_presence["present_ct"].astype(bool)
edge_presence["present_tg"] = edge_presence["present_tg"].astype(bool)


def pair_denominators(pair: tuple) -> tuple[int, int, int]:
    """
    Compute denominators for a given edge (A, B):
      - n_total_CT: strata where CT could have seen (A,B)
      - n_total_TG: strata where TG could have seen (A,B)
      - n_total_union: strata where at least one of CT or TG could have seen (A,B)
    """
    a, b = pair
    n_ct = int((elig_ct_u.get(a, False) & elig_ct_u.get(b, False)).sum())
    n_tg = int((elig_tg_u.get(a, False) & elig_tg_u.get(b, False)).sum())
    n_union = int(
        (
            (elig_u.get(f"{a}_CT", False) & elig_u.get(f"{b}_CT", False))
            | (elig_u.get(f"{a}_TG", False) & elig_u.get(f"{b}_TG", False))
        ).sum()
    )
    return n_ct, n_tg, n_union


# Aggregate edge-level counts across strata
edge_stats = (
    edge_presence.groupby(["FEATURE_A", "FEATURE_B"], dropna=False, sort=True)
    .apply(
        lambda g: pd.Series(
            {
                "n_strata": len(g),
                "n_ct": int(g["present_ct"].sum()),
                "n_tg": int(g["present_tg"].sum()),
                "n_overlap": int(
                    (g["present_ct"] & g["present_tg"]).sum()
                ),
            }
        )
    )
    .reset_index()
    .sort_values(["FEATURE_A", "FEATURE_B"])
    .reset_index(drop=True)
)

# Attach denominators to edge_stats
den_rows = []
for _, row in edge_stats.iterrows():
    pair = (row["FEATURE_A"], row["FEATURE_B"])
    d_ct, d_tg, d_union = pair_denominators(pair)
    den_rows.append((d_ct, d_tg, d_union))
den_df = pd.DataFrame(
    den_rows, columns=["n_total_CT", "n_total_TG", "n_total_union"]
)
edge_stats = pd.concat([edge_stats, den_df], axis=1)


def jaccard_counts(n_ct: int, n_tg: int, n_ov: int) -> float:
    """
    Jaccard index from counts: TP / (TP + FP + FN)
    where:
      - TP = n_ov
      - n_ct  = TP + FN
      - n_tg  = TP + FP
    """
    denom = (n_ct + n_tg - n_ov)
    return (n_ov / denom) if denom > 0 else np.nan


edge_stats["jaccard_edge"] = edge_stats.apply(
    lambda r: jaccard_counts(r["n_ct"], r["n_tg"], r["n_overlap"]),
    axis=1,
)

# =============================================================================
# 11. EDGE-LEVEL CONFUSION METRICS (RECALL, SPECIFICITY, BA, MCC)
# =============================================================================

def _edge_confusion_for_pair(
    pair: tuple,
    edge_presence: pd.DataFrame,
    elig_ct_u: pd.DataFrame,
    elig_tg_u: pd.DataFrame,
    keys=KEYS,
):
    """
    Build a confusion matrix for a given canonical edge (A,B) across strata
    where it was testable for BOTH CT and TG.
    """
    a, b = pair
    both_elig_series = (
        elig_ct_u.get(a, False)
        & elig_ct_u.get(b, False)
        & elig_tg_u.get(a, False)
        & elig_tg_u.get(b, False)
    ).astype(bool)

    both_elig_df = elig_ct_u[keys].copy()
    both_elig_df["both_elig"] = both_elig_series.values

    ep = (
        edge_presence.merge(both_elig_df, on=keys, how="left")
        .fillna({"both_elig": False})
    )
    mask = (
        ep["both_elig"]
        & ep["FEATURE_A"].eq(a)
        & ep["FEATURE_B"].eq(b)
    )
    e = ep.loc[mask, ["present_ct", "present_tg"]].astype(bool)

    tp = int((e["present_ct"] & e["present_tg"]).sum())
    fp = int((~e["present_ct"] & e["present_tg"]).sum())
    fn = int((e["present_ct"] & ~e["present_tg"]).sum())
    tn = int((~e["present_ct"] & ~e["present_tg"]).sum())
    n = tp + fp + fn + tn

    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ba = np.nan if (np.isnan(rec) or np.isnan(spec)) else 0.5 * (rec + spec)

    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den > 0 else np.nan

    jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    return tp, fp, fn, tn, n, rec, spec, ba, mcc, jacc


edge_metrics_rows = []
for _, r in edge_stats.iterrows():
    pair = (r["FEATURE_A"], r["FEATURE_B"])
    tp, fp, fn, tn, n, rec, spec, ba, mcc, jacc = _edge_confusion_for_pair(
        pair, edge_presence, elig_ct_u, elig_tg_u, keys=KEYS
    )
    edge_metrics_rows.append(
        {
            "FEATURE_A": pair[0],
            "FEATURE_B": pair[1],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "N": n,
            "recall_edge": rec,
            "specificity_edge": spec,
            "balanced_accuracy_edge": ba,
            "mcc_edge": mcc,
            "jaccard_edge_recomputed": jacc,
        }
    )

edge_metrics = (
    pd.DataFrame(edge_metrics_rows)
    .sort_values(["FEATURE_A", "FEATURE_B"])
    .reset_index(drop=True)
)

# Merge confusion metrics back into edge_stats for a consolidated table
edge_stats = (
    edge_stats.merge(edge_metrics, on=["FEATURE_A", "FEATURE_B"], how="left")
    .sort_values(["FEATURE_A", "FEATURE_B"])
    .reset_index(drop=True)
)

# =============================================================================
# 12. TRIAD-LEVEL PRESENCE & METRICS
# =============================================================================

tri_rows = []
for _, r in merged_sets.iterrows():
    ct_set = r["elevated_set_ct"]
    tg_set = r["elevated_set_tg"]
    for tri in CANONICAL_TRIADS:
        tri_rows.append(
            {
                KEYS[0]: r[KEYS[0]],
                KEYS[1]: r[KEYS[1]],
                "TRIAD": "+".join(tri),
                "present_ct": triad_present(ct_set, tri),
                "present_tg": triad_present(tg_set, tri),
            }
        )

if CANONICAL_TRIADS:
    tri_presence = pd.DataFrame(tri_rows)
else:
    tri_presence = pd.DataFrame(
        columns=[*KEYS, "TRIAD", "present_ct", "present_tg"]
    )

if not tri_presence.empty:
    tri_presence = (
        tri_presence.sort_values(KEYS + ["TRIAD"]).reset_index(drop=True)
    )
    tri_presence["present_ct"] = tri_presence["present_ct"].astype(bool)
    tri_presence["present_tg"] = tri_presence["present_tg"].astype(bool)

    tri_stats = (
        tri_presence.groupby(["TRIAD"], dropna=False, sort=True)
        .apply(
            lambda g: pd.Series(
                {
                    "n_strata": len(g),
                    "n_ct": int(g["present_ct"].sum()),
                    "n_tg": int(g["present_tg"].sum()),
                    "n_overlap": int(
                        (g["present_ct"] & g["present_tg"]).sum()
                    ),
                }
            )
        )
        .reset_index()
        .sort_values(["TRIAD"])
        .reset_index(drop=True)
    )

    # Triad denominators: how many strata where each triad could be observed
    for i, row in tri_stats.iterrows():
        ta, tb, tc = CANONICAL_TRIADS[0]  # assumes one triad; extend if needed
        n_total_CT = int(
            (
                elig_ct_u.get(ta, False)
                & elig_ct_u.get(tb, False)
                & elig_ct_u.get(tc, False)
            ).sum()
        )
        n_total_TG = int(
            (
                elig_tg_u.get(ta, False)
                & elig_tg_u.get(tb, False)
                & elig_tg_u.get(tc, False)
            ).sum()
        )
        n_total_union = int(
            (
                (
                    elig_u.get(f"{ta}_CT", False)
                    & elig_u.get(f"{tb}_CT", False)
                    & elig_u.get(f"{tc}_CT", False)
                )
                | (
                    elig_u.get(f"{ta}_TG", False)
                    & elig_u.get(f"{tb}_TG", False)
                    & elig_u.get(f"{tc}_TG", False)
                )
            ).sum()
        )
        tri_stats.loc[i, "n_total_CT"] = n_total_CT
        tri_stats.loc[i, "n_total_TG"] = n_total_TG
        tri_stats.loc[i, "n_total_union"] = n_total_union

    # Jaccard index from counts (as reference)
    tri_stats["jaccard_triad"] = tri_stats.apply(
        lambda r: jaccard_counts(r["n_ct"], r["n_tg"], r["n_overlap"]),
        axis=1,
    )
    tri_stats = tri_stats.sort_values(
        ["jaccard_triad", "n_overlap", "n_ct", "n_tg", "TRIAD"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    # Triad-level confusion metrics
    def _triad_confusion(
        tri: tuple,
        tri_presence: pd.DataFrame,
        elig_ct_u: pd.DataFrame,
        elig_tg_u: pd.DataFrame,
        keys=KEYS,
    ):
        a, b, c = tri
        both_elig_series = (
            elig_ct_u.get(a, False)
            & elig_ct_u.get(b, False)
            & elig_ct_u.get(c, False)
            & elig_tg_u.get(a, False)
            & elig_tg_u.get(b, False)
            & elig_tg_u.get(c, False)
        ).astype(bool)

        both_elig_df = elig_ct_u[keys].copy()
        both_elig_df["both_elig"] = both_elig_series.values

        tpv = (
            tri_presence.merge(both_elig_df, on=keys, how="left")
            .fillna({"both_elig": False})
        )

        tri_name = "+".join(tri)
        mask = tpv["TRIAD"].eq(tri_name) & tpv["both_elig"]
        e = tpv.loc[mask, ["present_ct", "present_tg"]].astype(bool)

        tp = int((e["present_ct"] & e["present_tg"]).sum())
        fp = int((~e["present_ct"] & e["present_tg"]).sum())
        fn = int((e["present_ct"] & ~e["present_tg"]).sum())
        tn = int((~e["present_ct"] & ~e["present_tg"]).sum())
        n = tp + fp + fn + tn

        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ba = np.nan if (np.isnan(rec) or np.isnan(spec)) else 0.5 * (rec + spec)

        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den > 0 else np.nan

        jacc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
        return tp, fp, fn, tn, n, rec, spec, ba, mcc, jacc

    tri_metrics_rows = []
    for tri in CANONICAL_TRIADS:
        tp, fp, fn, tn, n, rec, spec, ba, mcc, jacc = _triad_confusion(
            tri, tri_presence, elig_ct_u, elig_tg_u, keys=KEYS
        )
        tri_metrics_rows.append(
            {
                "TRIAD": "+".join(tri),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "N": n,
                "recall_triad": rec,
                "specificity_triad": spec,
                "balanced_accuracy_triad": ba,
                "mcc_triad": mcc,
                "jaccard_triad_recomputed": jacc,
            }
        )
    tri_metrics = pd.DataFrame(tri_metrics_rows)

    # Merge triad metrics into tri_stats
    tri_stats = (
        tri_stats.merge(tri_metrics, on="TRIAD", how="left")
        .sort_values(["TRIAD"])
        .reset_index(drop=True)
    )
else:
    tri_stats = pd.DataFrame(
        columns=[
            "TRIAD",
            "n_strata",
            "n_ct",
            "n_tg",
            "n_overlap",
            "n_total_CT",
            "n_total_TG",
            "n_total_union",
            "jaccard_triad",
            "TP",
            "FP",
            "FN",
            "TN",
            "N",
            "recall_triad",
            "specificity_triad",
            "balanced_accuracy_triad",
            "mcc_triad",
            "jaccard_triad_recomputed",
        ]
    )

# =============================================================================
# 13. EDGE & TRIAD SUMMARY TABLES (AGGREGATED METRICS)
# =============================================================================

def _nanmean_safe(s: pd.Series) -> float:
    """NaN-safe mean helper."""
    try:
        return float(np.nanmean(s)) if len(s) else np.nan
    except Exception:
        return np.nan

edge_summary = pd.DataFrame(
    {
        "universe": [UNIVERSE],
        "n_edges_considered": [len(CANONICAL_EDGES)],
        "mean_jaccard_edge": [_nanmean_safe(edge_stats["jaccard_edge"])],
        "median_jaccard_edge": [
            np.nanmedian(edge_stats["jaccard_edge"]) if len(edge_stats) else np.nan
        ],
        "mean_recall_edge": [_nanmean_safe(edge_stats["recall_edge"])],
        "mean_specificity_edge": [
            _nanmean_safe(edge_stats["specificity_edge"])
        ],
        "mean_balanced_accuracy_edge": [
            _nanmean_safe(edge_stats["balanced_accuracy_edge"])
        ],
        "mean_mcc_edge": [_nanmean_safe(edge_stats["mcc_edge"])],
        "prop_perfect_jaccard_1.0": [
            np.mean(edge_stats["jaccard_edge"] == 1.0)
            if len(edge_stats)
            else np.nan
        ],
        "prop_zero_jaccard_0.0": [
            np.mean(edge_stats["jaccard_edge"] == 0.0)
            if len(edge_stats)
            else np.nan
        ],
    }
)

if CANONICAL_TRIADS and not tri_stats.empty:
    tri_summary = pd.DataFrame(
        {
            "universe": [UNIVERSE],
            "n_triads_considered": [len(CANONICAL_TRIADS)],
            "mean_jaccard_triad": [_nanmean_safe(tri_stats["jaccard_triad"])],
            "median_jaccard_triad": [
                np.nanmedian(tri_stats["jaccard_triad"])
                if len(tri_stats)
                else np.nan
            ],
            "mean_recall_triad": [_nanmean_safe(tri_stats["recall_triad"])],
            "mean_specificity_triad": [
                _nanmean_safe(tri_stats["specificity_triad"])
            ],
            "mean_balanced_accuracy_triad": [
                _nanmean_safe(tri_stats["balanced_accuracy_triad"])
            ],
            "mean_mcc_triad": [_nanmean_safe(tri_stats["mcc_triad"])],
            "prop_perfect_jaccard_1.0": [
                np.mean(tri_stats["jaccard_triad"] == 1.0)
                if len(tri_stats)
                else np.nan
            ],
            "prop_zero_jaccard_0.0": [
                np.mean(tri_stats["jaccard_triad"] == 0.0)
                if len(tri_stats)
                else np.nan
            ],
        }
    )
else:
    tri_summary = pd.DataFrame(
        {
            "universe": [UNIVERSE],
            "n_triads_considered": [len(CANONICAL_TRIADS)],
            "mean_jaccard_triad": [np.nan],
            "median_jaccard_triad": [np.nan],
            "mean_recall_triad": [np.nan],
            "mean_specificity_triad": [np.nan],
            "mean_balanced_accuracy_triad": [np.nan],
            "mean_mcc_triad": [np.nan],
            "prop_perfect_jaccard_1.0": [np.nan],
            "prop_zero_jaccard_0.0": [np.nan],
        }
    )

# =============================================================================
# 14. SAVE RESULTS (STABLE FLOAT FORMATTING) + CONSOLE PEEK
# =============================================================================

FLOAT_FMT = "%.10g"
os.makedirs(OUTPUT_DIR, exist_ok=True)

edge_stats.sort_values(
    ["FEATURE_A", "FEATURE_B"]
).to_csv(
    os.path.join(OUTPUT_DIR, f"canonical_edge_agreement_{UNIVERSE}.csv"),
    index=False,
    float_format=FLOAT_FMT,
)

edge_summary.to_csv(
    os.path.join(OUTPUT_DIR, f"canonical_edge_summary_{UNIVERSE}.csv"),
    index=False,
    float_format=FLOAT_FMT,
)

tri_stats.sort_values(
    ["TRIAD"] if "TRIAD" in tri_stats.columns else []
).to_csv(
    os.path.join(OUTPUT_DIR, f"canonical_triad_agreement_{UNIVERSE}.csv"),
    index=False,
    float_format=FLOAT_FMT,
)

tri_summary.to_csv(
    os.path.join(OUTPUT_DIR, f"canonical_triad_summary_{UNIVERSE}.csv"),
    index=False,
    float_format=FLOAT_FMT,
)

# Console peek (optional; safe to remove in production)
print("\n=== Canonical co-elevation — edge summary ===")
print(edge_summary.to_string(index=False))
print("\nSpecified canonical edges (with denominators & metrics):")
print(edge_stats.to_string(index=False))

if CANONICAL_TRIADS:
    print("\n=== Canonical triad summary ===")
    print(tri_summary.to_string(index=False))
    print("\nTriad details (with denominators & metrics):")
    print(tri_stats.to_string(index=False))
