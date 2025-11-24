"""
One-Sided Conditional VAE–GAN Training Script (GanCtrl)

This script trains a one-sided treatment→control translation model with:

  - Strict reproducibility for TensorFlow / NumPy / Python.
  - Data loading for control/treatment train & test splits.
  - BODY_WEIGHT merge (keeping latest per PROGRESS_TIME).
  - MinMax scaling (per split) and per-sample peer variance maps.
  - KMeans-based clustering + cluster one-hot encoding.
  - Feature weighting for key biomarkers in the NLL loss.
  - Spatial neighbor mask for biologically-informed mixing.
  - TBIL range constraints (compound/time-specific bounds).
  - BIO MoE correlation loss (expert-based correlation matching).
  - Batch mean-matching add_loss (gentle control alignment).
  - CVAE-GAN with:
        * Encoder (q_phi)
        * Generator VAE (p_theta + MoE sub-decoders)
        * Conditional Discriminator
        * Composite model tying together KL, NLL, GAN, TBIL-range,
          BIO MoE corr, and mean-match losses.
  - Training loop with periodic checkpointing and prediction dumps.

Intended usage:
  - Run as a standalone training script.
  - Adjust `dataPath` and `resultPath` as needed for your environment.
"""

# ==============================
# Reproducibility (MUST be first)
# ==============================
import os

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

# ==============================
# Imports
# ==============================
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    LeakyReLU, Concatenate, Lambda, Add, Activation
)
from tensorflow.keras.losses import BinaryCrossentropy

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from math import sqrt
import re

# ==============================
# I/O helpers
# ==============================
def read_data(path: str) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def binarizer(data: pd.DataFrame):
    """
    Fit LabelBinarizers for dose, sacrifice period, and individual ID.

    Returns
    -------
    doseBinarizer, stageBinarizer, bioCopyBinarizer
    """
    dose_levels = data.loc[data['DOSE_LEVEL'] != 'Control', 'DOSE_LEVEL'].unique()
    doseBinarizer    = LabelBinarizer().fit(dose_levels)
    stageBinarizer   = LabelBinarizer().fit(data['SACRIFICE_PERIOD'])
    bioCopyBinarizer = LabelBinarizer().fit(data['INDIVIDUAL_ID'])
    return doseBinarizer, stageBinarizer, bioCopyBinarizer


# ==============================
# Load data
# ==============================
dataPath = '/account001/mansi.chandra/clin_path'  # <-- update if needed

controlTrain   = read_data(f"{dataPath}/repeat_train_control_cv2.csv")
controlTest    = read_data(f"{dataPath}/repeat_test_control_cv2.csv")
treatmentTrain = read_data(f"{dataPath}/repeat_train_treatment_cv2.csv")
treatmentTest  = read_data(f"{dataPath}/repeat_test_treatment_cv2.csv")

# Combine for consistent binarizers & EXP_ID ordering
dataset = pd.concat([controlTrain, controlTest, treatmentTrain, treatmentTest], axis=0)
dataset = dataset.sort_values('EXP_ID')

doseBinarizer, stageBinarizer, bioCopyBinarizer = binarizer(dataset)
cols = dataset.columns[11:]  # numeric feature columns (match training pipeline)


# ==============================
# BODY_WEIGHT merge: keep latest PROGRESS_TIME per composite key
# ==============================
bw_path = "/account001/mansi.chandra/clin_path/body_wt.csv"  # <-- update if needed
bw_df = pd.read_csv(bw_path, encoding="latin-1")
bw_df.columns = bw_df.columns.str.replace("\ufeff", "", regex=False).str.strip()

# Composite key for BODY_WEIGHT rows
_BW_KEYS = [
    "COMPOUND_NAME", "DOSE_LEVEL", "SACRIFICE_PERIOD",
    "EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"
]

_required_cols = _BW_KEYS + ["BODY_WEIGHT"]
_missing = [c for c in _required_cols if c not in bw_df.columns]
if _missing:
    raise RuntimeError(f"BODY_WEIGHT file missing columns: {_missing}")
if "PROGRESS_TIME" not in bw_df.columns:
    raise RuntimeError("BODY_WEIGHT file is missing 'PROGRESS_TIME' column.")

# Clean keys and coerce BODY_WEIGHT
for k in _BW_KEYS:
    bw_df[k] = bw_df[k].astype(str).str.strip()
bw_df["BODY_WEIGHT"] = pd.to_numeric(bw_df["BODY_WEIGHT"], errors="coerce")

# Build numeric PROGRESS_TIME surrogate for sorting
_prog_dt  = pd.to_datetime(bw_df["PROGRESS_TIME"], errors="coerce", infer_datetime_format=True)
_prog_num = pd.to_numeric(bw_df["PROGRESS_TIME"], errors="coerce")
_prog_val = np.where(~_prog_dt.isna(), _prog_dt.view("int64"), _prog_num)
bw_df["_PROG_VAL"] = pd.Series(_prog_val, index=bw_df.index).astype("float64").fillna(-1e300)

# Keep last (latest) row per composite key, tolerating legacy "_PROGVAL" naming if present
bw_df = (
    bw_df.sort_values("_PROGVAL" if "_PROGVAL" in bw_df.columns else "_PROG_VAL")
         .drop_duplicates(_BW_KEYS, keep="last")
         .drop(columns=["_PROGVAL" if "_PROGVAL" in bw_df.columns else "_PROG_VAL"])
)


def _merge_bw(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Merge latest BODY_WEIGHT into a given split (e.g., controlTrain).

    Validation is strict:
      - Requires all merge keys in df.
      - Ensures many-to-one merge.
      - Raises if BODY_WEIGHT is missing after merge.
    """
    left_missing = [k for k in _BW_KEYS if k not in df.columns]
    if left_missing:
        raise RuntimeError(f"{name} missing merge keys: {left_missing}")

    left = df.copy()
    for k in _BW_KEYS:
        left[k] = left[k].astype(str).str.strip()

    # Verify uniqueness of BODY_WEIGHT rows per composite key
    if bw_df.duplicated(_BW_KEYS).any():
        dups = bw_df[bw_df.duplicated(_BW_KEYS, keep=False)][_BW_KEYS].drop_duplicates()
        raise RuntimeError(
            f"Duplicate BW rows remain after latest-per-key selection: {len(dups)} examples"
        )

    out = left.merge(
        bw_df[_BW_KEYS + ["BODY_WEIGHT"]],
        on=_BW_KEYS,
        how="left",
        validate="many_to_one"
    )

    if out["BODY_WEIGHT"].isna().any():
        bad = out[out["BODY_WEIGHT"].isna()][_BW_KEYS].drop_duplicates()
        n_bad = len(bad)
        print("Examples of missing BW rows (up to 5):")
        print(bad.head(5).to_string(index=False))
        raise RuntimeError(
            f"Missing BODY_WEIGHT for {name}: {n_bad} composite-key rows. "
            f"Check join keys/values."
        )
    return out


# Attach BODY_WEIGHT to all four splits
controlTrain   = _merge_bw(controlTrain,   "controlTrain")
controlTest    = _merge_bw(controlTest,    "controlTest")
treatmentTrain = _merge_bw(treatmentTrain, "treatmentTrain")
treatmentTest  = _merge_bw(treatmentTest,  "treatmentTest")


# ==============================
# Scaling (per split, MinMax on train)
# ==============================
def scale(df_fit: pd.DataFrame,
          df_transform: pd.DataFrame,
          feature_cols):
    """
    Fit MinMaxScaler on df_fit[feature_cols] and transform df_transform[feature_cols].

    Returns
    -------
    scaled_df : DataFrame (meta columns preserved, features scaled)
    scaler    : MinMaxScaler
    """
    X_fit = df_fit[feature_cols]
    X_tr  = df_transform[feature_cols]

    scaler = MinMaxScaler()
    scaler.fit(X_fit)

    X_scaled = scaler.transform(X_tr)
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df_transform.index)

    meta_cols = [c for c in df_transform.columns if c not in feature_cols]
    result_df = pd.concat([df_transform[meta_cols], scaled_df], axis=1)
    return result_df, scaler


# NOTE: we scale each split with scaler fitted on its train counterpart
treatmentTest, _          = scale(treatmentTrain, treatmentTest, cols=cols)
treatmentTrain, tScaler   = scale(treatmentTrain, treatmentTrain, cols=cols)
controlTest, _            = scale(controlTrain, controlTest, cols=cols)
controlTrain, cScaler     = scale(controlTrain, controlTrain, cols=cols)

treatmentScaler = tScaler
controlScaler   = cScaler


# ==============================
# Real variance map (PER-SAMPLE peer variance, scaled space)
# ==============================
def _compute_per_sample_peer_variance(ctrl_df: pd.DataFrame, feature_cols):
    """
    For each control sample, compute the variance of all *other* control samples
    in the same (COMPOUND_NAME, SACRIFICE_PERIOD) group.

    Returns
    -------
    mapping     : index -> variance vector (float32)
    global_var  : global variance vector across all control samples (fallback)
    """
    global_var = ctrl_df[feature_cols].var(ddof=0).values.astype('float32')
    mapping = {}

    grouped = ctrl_df.groupby(['COMPOUND_NAME', 'SACRIFICE_PERIOD'], dropna=False)
    for (_, _), g in grouped:
        idxs = g.index.to_numpy()
        X = g[feature_cols].values.astype('float64')
        m = X.shape[0]

        if m <= 1:
            # Degenerate: use global variance
            for ridx in idxs:
                mapping[ridx] = global_var.copy()
            continue

        # Efficient computation of peer variance excluding self
        s1 = X.sum(axis=0)
        s2 = np.square(X).sum(axis=0)
        sum_sqdiff = s2[None, :] - 2.0 * X * s1[None, :] + m * np.square(X)
        n_other = m - 1
        var_peer = (sum_sqdiff / n_other).astype('float32')

        for ridx, v in zip(idxs, var_peer):
            mapping[ridx] = v

    return mapping, global_var


PER_SAMPLE_VAR_MAP, GLOBAL_VAR = _compute_per_sample_peer_variance(controlTrain, list(cols))


# ==============================
# Clustering (KMeans + cluster per compound)
# ==============================
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)

# Cluster on scaled control features
controlTrain['cluster'] = kmeans.fit_predict(controlTrain[cols].values)

# Majority cluster per compound
compound_cluster_map = (
    controlTrain.groupby('COMPOUND_NAME')['cluster']
                .agg(lambda x: x.value_counts().idxmax())
                .to_dict()
)

# Propagate clusters to treatment train
treatmentTrain['cluster'] = (
    treatmentTrain['COMPOUND_NAME']
      .map(compound_cluster_map)
      .fillna(-1)
      .astype(int)
)

clusterB = LabelBinarizer().fit(controlTrain['cluster'])

# For treatment test, assign cluster based on nearest cluster mean
treatment_cluster_means = (
    treatmentTrain.groupby('cluster')[cols]
                  .mean()
                  .to_dict(orient='index')
)


def assign_treatment_test_cluster(df: pd.DataFrame,
                                  cluster_means: dict,
                                  feature_cols):
    """Assign each row in df to nearest cluster mean."""
    assigned = []
    for _, row in df.iterrows():
        x = row[feature_cols].values.astype(float)
        best = min(
            cluster_means.keys(),
            key=lambda cl: sqrt(np.sum((x - np.array(list(cluster_means[cl].values()))) ** 2))
        )
        assigned.append(best)
    return np.array(assigned, dtype=int)


treatmentTest['cluster'] = assign_treatment_test_cluster(
    treatmentTest, treatment_cluster_means, cols
)

# Stability: majority cluster per compound for test
test_compound_cluster_map = (
    treatmentTest.groupby('COMPOUND_NAME')['cluster']
                 .agg(lambda x: x.value_counts().idxmax())
                 .to_dict()
)
treatmentTest['cluster'] = (
    treatmentTest['COMPOUND_NAME']
      .map(test_compound_cluster_map)
      .fillna(-1)
      .astype(int)
)


# ==============================
# Feature weighting (for NLL / variance-related terms)
# ==============================
feature_weights = {
    'ALP(IU/L)':       200.0,
    'TC(mg/dL)':       100.0,
    'TG(mg/dL)':       100.0,
    'PL(mg/dL)':       100.0,
    'LDH(IU/L)':       10.0,
    'WBC(x10_2/uL)':   200.0,
    'AST(IU/L)':       100.0,
    'ALT(IU/L)':       100.0,
    'TBIL(mg/dL)':     100.0,
    'TP(g/dL)':        200.0,
}

# Default weight = 1, override selected markers
weights_array = np.ones(len(cols), dtype='float32')
for feat, w in feature_weights.items():
    if feat in cols:
        idxw = list(cols).index(feat)
        weights_array[idxw] = w
weights_tensor = tf.constant(weights_array)


# ==============================
# Spatial groups mask (for mixing attention)
# ==============================
A = len(cols)

# Regex aliases to locate standard clinical markers
ALIASES = {
    "ALT":  [r"^ALT"], "AST":  [r"^AST"], "LDH":  [r"^LDH"], "ALP":  [r"^ALP"],
    "GTP":  [r"(GGT|GTP)"], "TBIL": [r"^TBIL|Total\s*Bilirubin"], "DBIL": [r"^DBIL|Direct\s*Bilirubin"],
    "TP":   [r"^TP|Total\s*Protein"], "RALB": [r"^(ALB|RALB|Albumin)"], "A/G":  [r"A/?G"],
    "BUN":  [r"^BUN"], "CRE":  [r"^CRE|Creat"], "Na":   [r"^Na"], "K":    [r"^K(?!g)"],
    "Cl":   [r"^Cl(\b|[^A-Za-z])|Chloride"], "Ca":   [r"^Ca(\b|[^A-Za-z])|Calcium"], "IP":   [r"^(IP|Phos|Phosphate)"],
    "GLC":  [r"^(GLC|Glucose)"], "TC":  [r"^TC|Cholesterol"], "TG":  [r"^TG|Triglycer"],
    "PL":   [r"^PL|Phospho.?lipid"], "RBC":  [r"^RBC"], "Hb":   [r"^(Hb|HGB)"], "Ht":   [r"^(Ht|HCT)"],
    "MCV":  [r"^MCV"], "MCH":  [r"^MCH(\b|[^A-Za-z])"], "MCHC":[r"^MCHC"], "Ret":  [r"^Ret"], "Plat":[r"^(Plat|PLT)"],
    "PT":   [r"^PT(?!H)"], "APTT":[r"^APTT|PTT"], "Fbg": [r"^(Fbg|Fibrinogen)"],
    "WBC":  [r"^WBC"], "Neu":  [r"^(Neu|Neutro)"], "Lym":  [r"^(Lym|Lympho)"], "Mono":[r"^Mono"],
    "Eos":  [r"^Eos"], "Bas":  [r"^Bas"],
}


def _idx_from_alias(key: str):
    """Return the index of the first column in `cols` matching alias regex patterns."""
    pats = ALIASES.get(key, [])
    for p in pats:
        for j, c in enumerate(cols):
            if re.search(p, c, flags=re.I):
                return j
    return None


idx = {k: _idx_from_alias(k) for k in ALIASES.keys()}

# Group biologically-related markers (for mixing mask)
GROUPS = {
    "hep_injury":   [idx["ALT"], idx["AST"], idx["LDH"]],
    "cholestasis":  [idx["ALP"], idx["GTP"], idx["TBIL"], idx["DBIL"]],
    "hep_synthetic":[idx["TP"], idx["RALB"], idx["A/G"]],
    "renal":        [idx["BUN"], idx["CRE"]],
    "electrolytes": [idx["Na"], idx["K"], idx["Cl"], idx["Ca"], idx["IP"]],
    "glucose":      [idx["GLC"]],
    "lipids":       [idx["TC"], idx["TG"], idx["PL"]],
    "rbc":          [idx["RBC"], idx["Hb"], idx["Ht"], idx["MCV"], idx["MCH"], idx["MCHC"], idx["Ret"]],
    "platelet":     [idx["Plat"]],
    "coag":         [idx["PT"], idx["APTT"], idx["Fbg"]],
    "wbc":          [idx["WBC"], idx["Neu"], idx["Lym"], idx["Mono"], idx["Eos"], idx["Bas"]],
}

# Build adjacency mask: 1 for within-group / self, 0 otherwise
_M = np.zeros((A, A), dtype="float32")
for _, glist in GROUPS.items():
    g = [g for g in glist if g is not None]
    for i in g:
        for j in g:
            _M[i, j] = 1.0
np.fill_diagonal(_M, 1.0)
NEIGHBOR_MASK = tf.constant(_M, dtype=tf.float32)


def masked_row_softmax(logits, mask):
    """
    Row-wise softmax, but only over positions where mask > 0.5.
    Used to keep mixing localized to biological neighbors.
    """
    very_neg = tf.constant(-1e9, dtype=logits.dtype)
    masked   = tf.where(mask > 0.5, logits, very_neg)
    return tf.nn.softmax(masked, axis=-1)


# ==============================
# TBIL index + per-group min/max bounds
# ==============================
TBIL_IDX = idx.get("TBIL", None)
if TBIL_IDX is None:
    cand = [j for j, c in enumerate(cols)
            if re.search(r"TBIL|Total\s*Bilirubin", c, flags=re.I)]
    if not cand:
        raise RuntimeError("Could not locate TBIL column for range loss.")
    TBIL_IDX = cand[0]


def _compute_group_minmax(ctrl_df: pd.DataFrame, feature_cols):
    """
    Compute per-(COMPOUND_NAME, SACRIFICE_PERIOD) min/max for each feature.
    """
    mm = {}
    grouped = ctrl_df.groupby(['COMPOUND_NAME', 'SACRIFICE_PERIOD'], dropna=False)
    for key, g in grouped:
        gX = g[feature_cols].astype('float32')
        rmin = gX.min(axis=0).values.astype('float32')
        rmax = gX.max(axis=0).values.astype('float32')
        mm[key] = (rmin, rmax)
    return mm


CTRL_MINMAX_MAP = _compute_group_minmax(controlTrain, list(cols))
TBIL_GLOBAL_MIN = float(controlTrain[cols].min().iloc[TBIL_IDX])
TBIL_GLOBAL_MAX = float(controlTrain[cols].max().iloc[TBIL_IDX])


def _build_tbil_bounds_for_rows(c_rows: pd.DataFrame):
    """
    For each control row, build [lo, hi] TBIL bounds from its compound/time group,
    or fall back to global TBIL bounds if group missing / non-finite.
    """
    bounds = np.zeros((len(c_rows), 2), dtype='float32')
    for i, (_, r) in enumerate(c_rows.iterrows()):
        key = (r['COMPOUND_NAME'], r['SACRIFICE_PERIOD'])
        m = CTRL_MINMAX_MAP.get(key, None)
        if m is None:
            lo, hi = TBIL_GLOBAL_MIN, TBIL_GLOBAL_MAX
        else:
            lo = float(m[0][TBIL_IDX]); hi = float(m[1][TBIL_IDX])
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = TBIL_GLOBAL_MIN, TBIL_GLOBAL_MAX
        bounds[i, 0] = lo
        bounds[i, 1] = hi
    return bounds


# ==============================
# BIO MoE GROUPS (feature-expert sets for correlation loss, H5-safe)
# ==============================
def _safe_idx(name: str):
    """Safe alias-to-index lookup: returns None if alias not found."""
    return idx.get(name, None)


# Only these five experts contribute to bio_corr_loss
EXPERT_SPECS = {
    "hep_injury":     ["ALT", "AST", "LDH"],
    "cholestasis":    ["ALP", "GTP", "TBIL", "DBIL"],
    "hep_synthetic":  ["TP", "RALB", "A/G"],
    "renal":          ["BUN", "CRE"],
    "electrolytes":   ["Na", "K", "Cl", "Ca", "IP"],
}

# Map experts to actual column indices; drop experts with <2 mapped columns
_EXPERT_IDXS = {}
for name, aliases in EXPERT_SPECS.items():
    feat_idxs = [_safe_idx(a) for a in aliases]
    feat_idxs = [j for j in feat_idxs if j is not None]
    if len(feat_idxs) >= 2:
        _EXPERT_IDXS[name] = sorted(set(feat_idxs))

# Expert weights: default 2.0, except hep_injury & renal = 1.2
EXPERT_WEIGHTS = {name: 2.0 for name in _EXPERT_IDXS.keys()}
if "hep_injury" in EXPERT_WEIGHTS:
    EXPERT_WEIGHTS["hep_injury"] = 1.2
if "renal" in EXPERT_WEIGHTS:
    EXPERT_WEIGHTS["renal"] = 1.2


def _pairs_from_indices(idxs):
    """All unique (i, j) index pairs with i < j."""
    pairs = []
    n = len(idxs)
    for a in range(n):
        for b in range(a + 1, n):
            pairs.append((idxs[a], idxs[b]))
    return pairs


_EXPERT_PAIRS = {name: _pairs_from_indices(v) for name, v in _EXPERT_IDXS.items()}

# H5-SAFE FLAT STRUCTURE (store Python lists, not tf.constants at the top level)
_EXPERT_FLAT = []
for name, pairs in _EXPERT_PAIRS.items():
    if not pairs:
        continue
    I = [i for (i, _) in pairs]                       # Python list of ints
    J = [j for (_, j) in pairs]                       # Python list of ints
    W = [EXPERT_WEIGHTS.get(name, 1.0)] * len(pairs) # Python list of floats
    _EXPERT_FLAT.append((name, I, J, W, len(pairs)))

# Keep graph valid with a tiny dummy if nothing mapped (should not happen normally)
if not _EXPERT_FLAT:
    _EXPERT_FLAT = [("dummy", [0], [0], [0.0], 1)]


# ==============================
# Label dims (dynamic; from data)
# ==============================
STAGE_DIM = len(stageBinarizer.classes_)
BIO_DIM   = len(bioCopyBinarizer.classes_)
DOSE_DIM  = len(doseBinarizer.classes_)
BW_DIM    = 1

LABEL_SHAPE  = (DOSE_DIM + STAGE_DIM + BIO_DIM + BW_DIM,)   # dose + time + bio + BW (src)
TARGET_SHAPE = (STAGE_DIM + BIO_DIM + BW_DIM,)              # time + bio + BW (tgt)
CLUSTER_DIM  = clusterB.classes_.size


# ==============================
# Discriminator (conditional on treatment/time/bio)
# ==============================
def define_discriminator(feature_dim: int,
                         stage_dim: int,
                         bio_dim: int) -> Model:
    """
    Conditional discriminator:
      Inputs  : [x_gen_or_real, x_treat, time_onehot, bio_onehot]
      Outputs : probability of being real.
    """
    init    = RandomNormal(stddev=0.01)
    mean_in = Input((feature_dim,), name='disc_mean_input')
    cond_in = Input((feature_dim,), name='disc_treatment_input')
    time_in = Input((stage_dim,),  name='disc_time_input')
    bio_in  = Input((bio_dim,),    name='disc_bio_input')

    x = Concatenate(name='disc_concat')([mean_in, cond_in, time_in, bio_in])
    x = Dense(128, activation='relu', kernel_initializer=init)(x); x = Dropout(0.4)(x)
    x = Dense(64,  activation='relu', kernel_initializer=init)(x); x = Dropout(0.3)(x)
    x = Dense(32,  activation='relu', kernel_initializer=init)(x); x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', kernel_initializer=init)(x)

    model = Model([mean_in, cond_in, time_in, bio_in], out, name='Discriminator_condTB')
    model.compile(
        loss=BinaryCrossentropy(label_smoothing=0.1),
        optimizer=Adam(5e-5, beta_1=0.5, beta_2=0.999),
        metrics=['accuracy']
    )
    return model


# ==============================
# Variance-focused Gaussian NLL (uses real control variance)
# ==============================
def make_gaussian_nll(lambda_var: float = 6.0,
                      batch_var_weight: float = 0.6,
                      floor: float = -3.0,
                      ceil: float = -1.5,
                      var_cons_in_logspace: bool = True):
    """
    Build a Gaussian NLL loss that:
      - Uses predicted mu/logvar per feature.
      - Aligns predicted variance with real per-sample variance.
      - Aligns predicted total variance with empirical batch variance.
    All terms are feature-weighted via `weights_tensor`.
    """
    var_floor = tf.exp(tf.constant(floor, dtype=tf.float32))

    def loss(y_true_concat, y_pred_concat):
        # y_true_concat = [y_true, real_var]
        y_true = y_true_concat[:, :A]
        real_v = y_true_concat[:, A:]
        real_v = tf.maximum(real_v, var_floor)

        mu     = y_pred_concat[:, :A]
        logvar = tf.clip_by_value(y_pred_concat[:, A:], floor, ceil)

        invvar = tf.exp(-logvar)
        err2   = tf.square(y_true - mu)
        nll    = 0.5 * (logvar + err2 * invvar)
        nll    = nll * weights_tensor
        base   = tf.reduce_mean(tf.reduce_sum(nll, axis=-1))

        # Per-sample variance consistency
        if var_cons_in_logspace:
            real_logv = tf.math.log(real_v)
            var_cons  = tf.square(logvar - tf.stop_gradient(real_logv))
        else:
            pred_v    = tf.exp(logvar)
            var_cons  = tf.square(pred_v - tf.stop_gradient(real_v))
        var_cons = var_cons * weights_tensor
        var_cons = tf.reduce_mean(tf.reduce_sum(var_cons, axis=-1))

        # Batch-level variance matching (total variance)
        y_var   = tf.math.reduce_variance(y_true, axis=0)
        mu_var  = tf.math.reduce_variance(mu,     axis=0)
        exp_var = tf.reduce_mean(tf.exp(logvar), axis=0)
        pred_total_var = exp_var + mu_var

        var_match = tf.reduce_sum(
            tf.square(pred_total_var - tf.stop_gradient(y_var)) * weights_tensor
        )
        var_match = var_match / tf.cast(tf.shape(y_true)[1], tf.float32)

        return base + lambda_var * var_cons + batch_var_weight * var_match

    return loss


# ==============================
# Encoder (q_phi), cluster-conditional
# ==============================
def define_encoder(input_shape,
                   label_shape,
                   target_shape,
                   cluster_shape,
                   z_dim: int = 64) -> Model:
    """
    Encoder q_phi(z | x_treat, label_src, label_tgt, cluster).
    Outputs mean/logvar + reparameterized z.
    """
    init = RandomNormal(stddev=0.01)
    f_in = Input(shape=input_shape,   name='enc_feat_in')
    l_in = Input(shape=label_shape,   name='enc_label_in')
    t_in = Input(shape=target_shape,  name='enc_tgt_in')
    c_in = Input(shape=cluster_shape, name='enc_cluster_in')

    x = Concatenate(name='enc_concat')([f_in, l_in, t_in, c_in])
    x = Dense(256, activation='relu', kernel_initializer=init)(x)
    x = Dense(128, activation='relu', kernel_initializer=init)(x)

    z_mean   = Dense(z_dim, kernel_initializer=init, name='z_mean')(x)
    z_logvar = Dense(z_dim, kernel_initializer=init, name='z_logvar')(x)

    def _sample(args):
        m, lv = args
        eps = K.random_normal(shape=K.shape(m))
        return m + K.exp(0.5 * lv) * eps

    z = Lambda(_sample, name='z_sample')([z_mean, z_logvar])
    return Model([f_in, l_in, t_in, c_in], [z_mean, z_logvar, z], name='Encoder')


# ==============================
# Generator VAE (p_theta) with spatial mixing + MoE sub-decoders
# ==============================
def define_generator_vae(
    input_shape,
    label_shape,
    target_shape,
    cluster_shape,
    output_shape: int,
    z_dim: int,
    n_clusters: int = 6
) -> Model:
    """
    Generator p_theta(x_control | x_treat, label_src, label_tgt, z, cluster).

    Includes:
      - Spatial mixing attention constrained by NEIGHBOR_MASK.
      - Shared base trunk.
      - Cluster-specific MoE sub-decoders (residuals).
      - Outputs concatenated [mu, logvar].
    """
    init = RandomNormal(stddev=0.01)

    f_in = Input(shape=input_shape,   name='feat_in')
    l_in = Input(shape=label_shape,   name='label_in')
    t_in = Input(shape=target_shape,  name='target_in')
    z_in = Input(shape=(z_dim,),      name='z_in')
    c_in = Input(shape=cluster_shape, name='cluster_in')

    # ---- Spatial mixing (context) ----
    ctx = Concatenate(name='mix_ctx_concat')([f_in, l_in, t_in])
    h   = Dense(128, activation='relu', kernel_initializer=init, name='mix_ctx_d1')(ctx)
    h   = Dense(64,  activation='relu', kernel_initializer=init, name='mix_ctx_d2')(h)

    attn_logits = Dense(output_shape * output_shape,
                        kernel_initializer=init,
                        name='mix_logits')(h)
    attn = Lambda(
        lambda u: tf.reshape(u, (-1, output_shape, output_shape)),
        name='mix_reshape'
    )(attn_logits)
    attn = Lambda(
        lambda u: masked_row_softmax(u, NEIGHBOR_MASK),
        name='mix_masked_softmax'
    )(attn)

    mixed = Lambda(
        lambda xs: tf.matmul(xs[0], tf.expand_dims(xs[1], -1)),
        name='mix_matmul'
    )([attn, f_in])
    mixed = Lambda(lambda v: tf.squeeze(v, axis=-1), name='mix_squeeze')(mixed)

    # ---- Trunk + MoE sub-decoders ----
    x = Concatenate()([mixed, l_in, t_in, z_in])
    x = Dense(256, kernel_initializer=init)(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x); x = Dropout(0.3)(x)
    x = Dense(128, kernel_initializer=init)(Concatenate()([x, t_in])); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x); x = Dropout(0.3)(x)
    x = Dense(128, kernel_initializer=init)(Concatenate()([x, t_in])); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x); x = Dropout(0.2)(x)
    x = Dense(64,  kernel_initializer=init)(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x); x = Dropout(0.2)(x)
    x = Dense(64,  kernel_initializer=init)(x); x = BatchNormalization()(x); x = LeakyReLU(0.2)(x)

    base = Dense(128, kernel_initializer=init, name='base_pre')(x)

    subs = []
    for i in range(n_clusters):
        s = Dense(128, kernel_initializer=init, name=f'sub{i}_d1')(x)
        s = BatchNormalization()(s); s = LeakyReLU(0.2)(s); s = Dropout(0.2)(s)
        s = Dense(64,  kernel_initializer=init, name=f'sub{i}_d2')(s)
        s = BatchNormalization()(s); s = LeakyReLU(0.2)(s); s = Dropout(0.2)(s)
        s = Dense(output_shape, kernel_initializer=init, name=f'sub{i}_out')(s)
        subs.append(s)

    stack = Lambda(lambda args: tf.stack(args, axis=1), name='sub_stack')(subs)
    c_exp = Lambda(lambda c: tf.expand_dims(c, axis=-1), name='cluster_expand')(c_in)
    resid = Lambda(
        lambda a: tf.reduce_sum(a[0] * a[1], axis=1),
        name='cluster_residual'
    )([stack, c_exp])

    core = Add(name='add_residual')([
        Dense(output_shape, kernel_initializer=init, name='base_out')(base),
        resid
    ])

    mu_head     = Activation('sigmoid', name='mu')(core)
    logvar_head = Dense(output_shape, kernel_initializer=init, name='logvar')(x)

    out = Concatenate(name='mu_logvar_concat')([mu_head, logvar_head])
    return Model([f_in, l_in, t_in, z_in, c_in], out, name='Generator_VAE')


# ==============================
# Composite CVAE-GAN (+ KL + TBIL range + BIO MoE corr + mean-match)
# ==============================
def make_range_loss_tbil(tbil_index: int):
    """
    Build a TBIL range loss that penalizes predictions outside [lo, hi] per sample.
    y_true_bounds: [lo, hi]
    """
    def loss(y_true_bounds, y_pred):
        lo = y_true_bounds[:, 0:1]
        hi = y_true_bounds[:, 1:2]
        y  = y_pred[:, tbil_index:tbil_index+1]
        below = tf.nn.relu(lo - y)
        above = tf.nn.relu(y - hi)
        viol  = below + above
        return tf.reduce_mean(viol)
    return loss


def _batch_pairwise_corr(xi, xj, eps: float = 1e-6):
    """Batch-wise Pearson correlation across columns."""
    xi_c = xi - tf.reduce_mean(xi, axis=0, keepdims=True)
    xj_c = xj - tf.reduce_mean(xj, axis=0, keepdims=True)
    cov  = tf.reduce_mean(xi_c * xj_c, axis=0)
    vi   = tf.reduce_mean(tf.square(xi_c), axis=0) + eps
    vj   = tf.reduce_mean(tf.square(xj_c), axis=0) + eps
    return cov / tf.sqrt(vi * vj)


def define_composite_cvae_gan(
    e_model: Model,
    g_model: Model,
    d_model: Model,
    input_shape,
    label_shape,
    target_shape,
    cluster_shape,
    beta_kl: float = 0.05,
    range_mu_weight: float = 1.0,
    range_x_weight: float = 2.0,
    bio_corr_weight: float = 5.0,
    report_corr_metric: bool = True,
    mean_match_weight: float = 0.05,
) -> Model:
    """
    Composite model tying together:
      - Encoder + Generator (VAE)
      - Discriminator
      - Losses:
          * KL divergence
          * Gaussian NLL with variance alignment
          * TBIL range loss on mu and samples
          * BIO MoE correlation loss
          * Batch mean-match (gen vs real control)
    """
    # Freeze D inside composite
    d_model.trainable = False

    # ----- Inputs -----
    f_in   = Input(input_shape,   name='feat_in')        # treated features (source)
    l_in   = Input(label_shape,   name='label_in')       # src: dose+time+bio+BW
    t_in   = Input(target_shape,  name='target_in')      # tgt: time+bio+BW
    c_in   = Input(cluster_shape, name='cluster_in')     # cluster one-hot
    rng_tb = Input((2,),          name='tbil_range_in')  # [lo, hi] bounds for TBIL
    corr_ref_in = Input(input_shape, name='corr_ref_in') # real control batch (corr + mean-match ref)

    # ----- Encoder -----
    z_mean, z_logvar, z = e_model([f_in, l_in, t_in, c_in])

    # ----- Generator -----
    mu_logvar = g_model([f_in, l_in, t_in, z, c_in])
    mu     = Lambda(lambda y: y[:, :A],  name='mu_slice')(mu_logvar)
    logvar = Lambda(lambda y: y[:, A:],  name='logvar_slice')(mu_logvar)

    # ---- Variance monitoring: average logvar & fraction at bounds ----
    avg_logvar = Lambda(lambda lv: tf.reduce_mean(lv), name='avg_logvar_value')(logvar)
    FLOOR = -3.0
    CEIL  = -1.5
    frac_at_floor = Lambda(
        lambda lv, f=FLOOR: tf.reduce_mean(
            tf.cast(tf.less_equal(lv, f + 1e-6), tf.float32)
        ),
        name='frac_logvar_floor_value'
    )(logvar)
    frac_at_ceil = Lambda(
        lambda lv, c=CEIL: tf.reduce_mean(
            tf.cast(tf.greater_equal(lv, c - 1e-6), tf.float32)
        ),
        name='frac_logvar_ceil_value'
    )(logvar)

    # Raw (unclipped) sample for TBIL-range loss on samples
    def _sample_raw(args):
        mu01, lv = args
        eps = K.random_normal(shape=K.shape(mu01))
        return mu01 + K.exp(0.5 * lv) * eps
    x_raw = Lambda(_sample_raw, name='x_raw')([mu, logvar])

    # Sample for discriminator (clipped to [0,1])
    def _sample_x(args):
        mu_, lv_ = args
        eps = K.abs(K.random_normal(shape=K.shape(mu_)))
        x = mu_ + K.exp(0.5 * lv_) * eps
        return tf.clip_by_value(x, 0.0, 1.0)
    x_hat = Lambda(_sample_x, name='x_sample')([mu, logvar])

    # ----- Discriminator path -----
    t_time = Lambda(lambda x: x[:, :STAGE_DIM],                  name='disc_time_slice')(t_in)
    t_bio  = Lambda(lambda x: x[:, STAGE_DIM:STAGE_DIM+BIO_DIM], name='disc_bio_slice')(t_in)
    d_out  = d_model([x_hat, f_in, t_time, t_bio])

    # ----- Composite model (multiple heads) -----
    comp = Model(
        [f_in, l_in, t_in, c_in, rng_tb, corr_ref_in],
        [mu_logvar, d_out, mu, x_raw],
        name='Composite_CVAE_GAN'
    )

    # Log variance monitoring tensors as metrics
    comp.add_metric(avg_logvar,     name='avg_logvar',        aggregation='mean')
    comp.add_metric(frac_at_floor,  name='frac_logvar_floor', aggregation='mean')
    comp.add_metric(frac_at_ceil,   name='frac_logvar_ceil',  aggregation='mean')

    # ----- KL loss -----
    kl = -0.5 * tf.reduce_sum(
        1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar),
        axis=-1
    )
    comp.add_loss(beta_kl * tf.reduce_mean(kl))

    # ----- Head losses -----
    nll_loss_fn     = make_gaussian_nll(
        lambda_var=6.0,
        batch_var_weight=0.6,
        floor=-3.0,
        ceil=-1.5,
        var_cons_in_logspace=True
    )
    tbil_range_loss = make_range_loss_tbil(TBIL_IDX)

    # ==============================
    # BIO MoE correlation loss (H5-safe)
    # ==============================
    bio_moe_losses = []
    for (exp_name, I, J, W, n_pairs) in _EXPERT_FLAT:
        mu_i  = Lambda(
            lambda x, I=I: tf.gather(x, tf.constant(I, dtype=tf.int32), axis=1),
            name=f'{exp_name}_mu_i'
        )(mu)
        mu_j  = Lambda(
            lambda x, J=J: tf.gather(x, tf.constant(J, dtype=tf.int32), axis=1),
            name=f'{exp_name}_mu_j'
        )(mu)
        rf_i  = Lambda(
            lambda x, I=I: tf.gather(x, tf.constant(I, dtype=tf.int32), axis=1),
            name=f'{exp_name}_rf_i'
        )(corr_ref_in)
        rf_j  = Lambda(
            lambda x, J=J: tf.gather(x, tf.constant(J, dtype=tf.int32), axis=1),
            name=f'{exp_name}_rf_j'
        )(corr_ref_in)

        corr_mu  = Lambda(
            lambda xy: _batch_pairwise_corr(xy[0], xy[1]),
            name=f'{exp_name}_corr_mu'
        )([mu_i, mu_j])
        corr_ref = Lambda(
            lambda xy: _batch_pairwise_corr(xy[0], xy[1]),
            name=f'{exp_name}_corr_ref'
        )([rf_i, rf_j])

        corr_err = Lambda(
            lambda ab: tf.square(ab[0] - tf.stop_gradient(ab[1])),
            name=f'{exp_name}_corr_err'
        )([corr_mu, corr_ref])

        corr_w = Lambda(
            lambda e, W=W: tf.reduce_mean(e * tf.constant(W, dtype=tf.float32)),
            name=f'{exp_name}_corr_weighted'
        )(corr_err)
        bio_moe_losses.append(corr_w)

    if bio_moe_losses and bio_corr_weight > 0.0:
        bio_moe_total = (
            Add(name='bio_moe_total')(bio_moe_losses)
            if len(bio_moe_losses) > 1 else bio_moe_losses[0]
        )
        comp.add_loss(bio_corr_weight * bio_moe_total)
        if report_corr_metric:
            comp.add_metric(bio_moe_total, name='bio_corr_moe', aggregation='mean')

    # ==============================
    # Batch mean-matching add_loss: E[x_hat] vs E[x_real_control]
    # ==============================
    if mean_match_weight and mean_match_weight > 0.0:
        gen_mean  = Lambda(lambda x: tf.reduce_mean(x, axis=0), name='gen_batch_mean')(x_hat)
        real_mean = Lambda(lambda x: tf.reduce_mean(x, axis=0), name='real_batch_mean')(corr_ref_in)
        mean_err  = Lambda(
            lambda ab: tf.reduce_mean(tf.square(ab[0] - tf.stop_gradient(ab[1]))),
            name='mean_match_err'
        )([gen_mean, real_mean])
        comp.add_loss(mean_match_weight * mean_err)
        comp.add_metric(mean_err, name='mean_match', aggregation='mean')

    # ----- Compile composite (multi-head) -----
    comp.compile(
        loss=[
            nll_loss_fn,
            BinaryCrossentropy(label_smoothing=0.1),
            tbil_range_loss,
            tbil_range_loss,
        ],
        loss_weights=[20, 2, range_mu_weight, range_x_weight],
        optimizer=Adam(1e-4, 0.5, 0.999)
    )
    return comp


# ==============================
# Convenience models (not saved): Mean predictor & sampler
# ==============================
def build_mean_predictor(e_model: Model,
                         g_model: Model,
                         input_shape,
                         label_shape,
                         target_shape,
                         cluster_shape) -> Model:
    """Wrapper model: [x_treat, label_src, label_tgt, cluster] -> mu."""
    f_in = Input(input_shape)
    l_in = Input(label_shape)
    t_in = Input(target_shape)
    c_in = Input(cluster_shape)
    _, _, z = e_model([f_in, l_in, t_in, c_in])
    mu_logvar = g_model([f_in, l_in, t_in, z, c_in])
    mu = Lambda(lambda y: y[:, :A], name='mu_out')(mu_logvar)
    return Model([f_in, l_in, t_in, c_in], mu, name='MeanPredictor')


def build_sampler(e_model: Model,
                  g_model: Model,
                  input_shape,
                  label_shape,
                  target_shape,
                  cluster_shape) -> Model:
    """Wrapper model: [x_treat, label_src, label_tgt, cluster] -> sample x."""
    f_in = Input(input_shape)
    l_in = Input(label_shape)
    t_in = Input(target_shape)
    c_in = Input(cluster_shape)
    _, _, z = e_model([f_in, l_in, t_in, c_in])
    mu_logvar = g_model([f_in, l_in, t_in, z, c_in])
    mu     = Lambda(lambda y: y[:, :A])(mu_logvar)
    logvar = Lambda(lambda y: y[:, A:])(mu_logvar)
    eps = Lambda(lambda m: K.abs(K.random_normal(shape=K.shape(m))))(mu)
    x_sample = Lambda(
        lambda args: tf.clip_by_value(args[0] + K.exp(0.5*args[1]) * args[2], 0.0, 1.0),
        name='x_sample'
    )([mu, logvar, eps])
    return Model([f_in, l_in, t_in, c_in], x_sample, name='Sampler')


# ==============================
# Pairing helpers (time-matched sampling)
# ==============================
def build_time_index(treat_df: pd.DataFrame,
                     ctrl_df: pd.DataFrame):
    """
    Precompute, per compound, pools of (timepoint, treat_indices, control_indices)
    for time-matched sampling during training.
    """
    index = {}

    tdf = treat_df[treat_df['DOSE_LEVEL'] != 'Control']
    cdf = ctrl_df[ctrl_df['DOSE_LEVEL'] == 'Control']

    for cmpd in np.intersect1d(
        tdf['COMPOUND_NAME'].unique(),
        cdf['COMPOUND_NAME'].unique()
    ):
        t_sub = tdf[tdf['COMPOUND_NAME'] == cmpd]
        c_sub = cdf[cdf['COMPOUND_NAME'] == cmpd]
        if t_sub.empty or c_sub.empty:
            continue

        common_times = np.intersect1d(
            t_sub['SACRIFICE_PERIOD'].unique(),
            c_sub['SACRIFICE_PERIOD'].unique()
        )
        pools = []
        for tp in common_times:
            t_pool = t_sub.index[t_sub['SACRIFICE_PERIOD'] == tp].to_numpy()
            c_pool = c_sub.index[c_sub['SACRIFICE_PERIOD'] == tp].to_numpy()
            if len(t_pool) and len(c_pool):
                pools.append((tp, t_pool, c_pool))

        if pools:
            index[cmpd] = pools

    return index


def sample_time_matched_batch_from_index(
    treatment_df: pd.DataFrame,
    control_df: pd.DataFrame,
    time_index: dict,
    cmpd: str,
    n: int,
    feature_cols,
    var_map: dict,
    global_var: np.ndarray,
    mix_times: bool = False
):
    """
    Sample a batch of time-matched (treatment, control) pairs for a given compound.

    Returns X_treat, X_ctrl, real_var, L_src, L_tgt, C, y_real, TBIL_BOUNDS.
    """
    # Choose a single timepoint (or mix across timepoints if mix_times=True)
    if not mix_times:
        tp, t_pool, c_pool = random.choice(time_index[cmpd])
        t_idx = np.random.choice(t_pool, size=n, replace=True)
        c_idx = np.random.choice(c_pool, size=n, replace=True)
    else:
        choices = [random.choice(time_index[cmpd]) for _ in range(n)]
        t_idx = [np.random.choice(tp_pool) for (_, tp_pool, _) in choices]
        c_idx = [np.random.choice(cp_pool) for (_, _, cp_pool) in choices]

    t_rows = treatment_df.loc[t_idx]
    c_rows = control_df.loc[c_idx]

    X_treat = t_rows[feature_cols].astype('float32').values
    X_ctrl  = c_rows[feature_cols].astype('float32').values

    # Real per-sample variance for control rows
    real_var = np.empty_like(X_ctrl, dtype='float32')
    for i, (rid, _) in enumerate(c_rows.iterrows()):
        v = var_map.get(rid, global_var)
        real_var[i, :] = v

    TBIL_BOUNDS = _build_tbil_bounds_for_rows(c_rows)

    # Source labels (treatment): dose + time + bio + BW_src
    d_src = doseBinarizer.transform(t_rows['DOSE_LEVEL'])
    t_src = stageBinarizer.transform(t_rows['SACRIFICE_PERIOD'])
    b_src = bioCopyBinarizer.transform(
        t_rows['INDIVIDUAL_ID'].astype(bioCopyBinarizer.classes_.dtype)
    )
    bw_s  = t_rows[['BODY_WEIGHT']].values.astype('float32')
    L_src = np.hstack([d_src, t_src, b_src, bw_s]).astype('float32')

    # Target labels (control): time + bio + BW_tgt
    t_tgt = stageBinarizer.transform(c_rows['SACRIFICE_PERIOD'])
    b_tgt = bioCopyBinarizer.transform(
        c_rows['INDIVIDUAL_ID'].astype(bioCopyBinarizer.classes_.dtype)
    )
    bw_t  = c_rows[['BODY_WEIGHT']].values.astype('float32')
    L_tgt = np.hstack([t_tgt, b_tgt, bw_t]).astype('float32')

    # Cluster one-hot (cluster assigned to treatment rows)
    C = clusterB.transform(t_rows['cluster'].astype(int))

    # Real labels for D (1 = real)
    y_real = np.ones((n, 1), dtype='float32')
    return X_treat, X_ctrl, real_var, L_src, L_tgt, C, y_real, TBIL_BOUNDS


def generate_test_real_samples(treatmentTest: pd.DataFrame,
                               controlTest: pd.DataFrame):
    """
    Build all treatment–control pairs for evaluation on test data.

    Returns
    -------
    data     : meta DataFrame with pairing info
    feat     : features for treatment rows
    src_lbl  : src labels (dose + time + bio + BW_src)
    tgt_lbl  : tgt labels (time + bio + BW_tgt)
    C        : cluster one-hot
    """
    df = (
        pd.concat([treatmentTest, controlTest], axis=0)
          .sort_values(['COMPOUND_NAME', 'SACRIFICE_PERIOD'])
          .reset_index(drop=True)
    )

    def pairs(src: pd.DataFrame, tgt: pd.DataFrame):
        left = pd.DataFrame(
            np.repeat(src.values, len(tgt), axis=0),
            columns=src.columns
        )
        right = pd.concat(
            [tgt[['ID', 'SACRIFICE_PERIOD', 'DOSE_LEVEL', 'INDIVIDUAL_ID']]] * len(src),
            ignore_index=True
        )
        right.columns = ['targetId', 'targetTime', 'targetDose', 'targetBioCopy']
        return pd.concat(
            [left.reset_index(drop=True), right.reset_index(drop=True)],
            axis=1
        )

    all_pairs = []
    for cmpd in df.COMPOUND_NAME.unique():
        sub = df[df.COMPOUND_NAME == cmpd]
        for tp in sub.SACRIFICE_PERIOD.unique():
            tr = sub[(sub.DOSE_LEVEL != 'Control') & (sub.SACRIFICE_PERIOD == tp)]
            ct = sub[(sub.DOSE_LEVEL == 'Control')   & (sub.SACRIFICE_PERIOD == tp)]
            if len(tr) and len(ct):
                all_pairs.append(pairs(tr, ct))

    data = pd.concat(all_pairs, ignore_index=True)

    _target_bw = controlTest[['ID', 'BODY_WEIGHT']].rename(
        columns={'ID': 'targetId', 'BODY_WEIGHT': 'BODY_WEIGHT_TGT'}
    )
    data = data.merge(_target_bw, on='targetId', how='left', validate='many_to_one')
    if data['BODY_WEIGHT_TGT'].isna().any():
        raise RuntimeError("Missing target BODY_WEIGHT in generated test pairs; check targetId mapping.")

    data['cluster'] = data['cluster'].astype(int)
    feat = data[cols].astype('float32').values

    tvals = stageBinarizer.transform(data['SACRIFICE_PERIOD'])
    dvals = doseBinarizer.transform(data['DOSE_LEVEL'])
    bvals = bioCopyBinarizer.transform(
        data['INDIVIDUAL_ID'].astype(bioCopyBinarizer.classes_.dtype)
    )
    tt = stageBinarizer.transform(data['targetTime'])
    tb = bioCopyBinarizer.transform(
        data['targetBioCopy'].astype(bioCopyBinarizer.classes_.dtype)
    )

    bw_src = data[['BODY_WEIGHT']].values.astype('float32')
    bw_tgt = data[['BODY_WEIGHT_TGT']].values.astype('float32')

    src_lbl = np.hstack([dvals, tvals, bvals, bw_src])
    tgt_lbl = np.hstack([tt, tb, bw_tgt])

    C = clusterB.transform(data['cluster'])
    return data, feat, src_lbl, tgt_lbl, C


def generate_fake_samples(sampler_model: Model,
                          feats: np.ndarray,
                          lbl_s: np.ndarray,
                          lbl_t: np.ndarray,
                          clusters: np.ndarray):
    """
    Generate fake samples x_hat from sampler_model for discriminator training.
    Returns xs, y_fake (0-labels).
    """
    xs = sampler_model.predict([feats, lbl_s, lbl_t, clusters], verbose=0)
    y_fake = np.zeros((xs.shape[0], 1), dtype='float32')
    return xs, y_fake


# ==============================
# Save helpers (only G, D, Composite)
# ==============================
def save_models(step: int,
                g_model: Model,
                d_model: Model,
                c_model: Model,
                resultPath: str = '/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2'):
    """Save generator, discriminator, and composite model checkpoints."""
    tf.io.gfile.makedirs(f"{resultPath}/model1")
    tf.io.gfile.makedirs(f"{resultPath}/d_model1")
    tf.io.gfile.makedirs(f"{resultPath}/composite_model1")

    g_model.save(f"{resultPath}/model1/g_model1_{step+1:06d}.h5")
    d_model.save(f"{resultPath}/d_model1/d_model1_{step+1:06d}.h5")
    c_model.save(f"{resultPath}/composite_model1/composite_model1_{step+1:06d}.h5")


def summarize_performance(
    step: int,
    mean_predictor: Model,
    in_f: np.ndarray,
    in_l: np.ndarray,
    in_t: np.ndarray,
    in_c: np.ndarray,
    meta_df: pd.DataFrame,
    name: str,
    scaler: MinMaxScaler,
    features=cols,
    resultPath: str = '/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2',
):
    """
    Run mean predictor on a fixed test set, inverse-scale predictions,
    and write them to CSV for monitoring.
    """
    feature_list = list(features)
    pm = mean_predictor.predict([in_f, in_l, in_t, in_c], verbose=0)
    mean_rescaled = scaler.inverse_transform(pm)
    mean_df = pd.DataFrame(mean_rescaled, columns=feature_list)

    # Drop feature columns from meta (if present) and recombine
    meta = meta_df.drop(columns=feature_list, errors='ignore').reset_index(drop=True)
    out = pd.concat([meta, mean_df], axis=1)
    ordered_cols = list(meta.columns) + feature_list
    out = out[ordered_cols]

    tf.io.gfile.makedirs(f"{resultPath}/predictions_encoded")
    out.to_csv(
        f"{resultPath}/predictions_encoded/generated_predictions_{step+1:06d}_{name}.csv",
        index=False
    )


# ==============================
# Training loop (CVAE-GAN + TBIL range + BIO MoE corr)
# ==============================
def train_one_sided_translation(
    d_model: Model,
    e_model: Model,
    g_model: Model,
    c_model: Model,
    mean_predictor: Model,
    sampler_model: Model,
    treatmentTrain: pd.DataFrame,
    controlTrain: pd.DataFrame,
    treatmentTest: pd.DataFrame,
    controlTest: pd.DataFrame,
    controlScaler: MinMaxScaler,
    resultPath: str = '/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2',
    n_epochs: int = 10000,
    n_batch: int = 16,
    print_every: int = 100,
):
    """
    Main training loop for one-sided treatment→control translation:

      - Samples time-matched High-dose batches per compound.
      - Updates composite (E+G) via:
          * NLL, KL, GAN (BCE), TBIL-range, BIO MoE, mean-match.
      - Updates D on real and fake batches.
      - Periodically saves checkpoints and test predictions.
    """
    # Ensure directories exist
    tf.io.gfile.makedirs(resultPath)
    tf.io.gfile.makedirs(f"{resultPath}/loss")
    tf.io.gfile.makedirs(f"{resultPath}/model1")
    tf.io.gfile.makedirs(f"{resultPath}/d_model1")
    tf.io.gfile.makedirs(f"{resultPath}/composite_model1")

    progress_file = f"{resultPath}/progress.txt"
    loss_file     = f"{resultPath}/loss/loss.csv"

    steps_per_epoch = max(1, len(treatmentTrain) // n_batch)
    total_steps     = steps_per_epoch * n_epochs
    start_step      = 0
    print(f"Starting from step 0/{total_steps}")

    logs = pd.read_csv(loss_file).values.tolist() if tf.io.gfile.exists(loss_file) else []

    # Fixed test evaluation pairs
    mt, Xt, Lt, Tt, Ct = generate_test_real_samples(treatmentTest, controlTest)

    # High-dose subset (source of training batches)
    treatmentTrain_high = treatmentTrain[
        treatmentTrain['DOSE_LEVEL'].astype(str).str.strip() == 'High'
    ].copy()
    if treatmentTrain_high.empty:
        raise RuntimeError("No High-dose rows found in treatmentTrain.")

    # Time-matching index
    time_index = build_time_index(treatmentTrain_high, controlTrain)
    valid_cmpds = list(time_index.keys())
    if not valid_cmpds:
        raise RuntimeError("No compounds have High-dose treatment & matched control timepoints.")

    for i in range(start_step, total_steps):
        # Pick a compound for this step
        cmpd = random.choice(valid_cmpds)

        (
            XrB, XrA, RVAR,
            LrB, LrA, ClrB,
            yrA_tf, TBIL_BOUNDS
        ) = sample_time_matched_batch_from_index(
            treatmentTrain_high,
            controlTrain,
            time_index,
            cmpd,
            n_batch,
            list(cols),
            PER_SAMPLE_VAR_MAP,
            GLOBAL_VAR,
            mix_times=True,
        )

        # Target for NLL head: concat [real features, REAL per-sample variance]
        XrA_ext = np.hstack([XrA.astype('float32'), RVAR.astype('float32')])

        # Update G+E via composite (NLL+BCE+KL+TBIL-range+BIO-MoE+mean-match)
        g_res = c_model.train_on_batch(
            [XrB, LrB, LrA, ClrB, TBIL_BOUNDS, XrA],  # corr_ref_in = XrA
            [XrA_ext, yrA_tf, TBIL_BOUNDS, TBIL_BOUNDS]
        )
        total_loss, nll_loss, bce_loss, range_mu_loss, range_x_loss = [
            float(x) for x in g_res[:5]
        ]

        # Discriminator: real batch
        LrA_time = LrA[:, :STAGE_DIM].astype('float32')
        LrA_bio  = LrA[:, STAGE_DIM:STAGE_DIM+BIO_DIM].astype('float32')
        d_real   = d_model.train_on_batch([XrA, XrB, LrA_time, LrA_bio], yrA_tf)

        # Discriminator: fake batch
        Xf_s, y_fake = generate_fake_samples(sampler_model, XrB, LrB, LrA, ClrB)
        d_fake       = d_model.train_on_batch([Xf_s, XrB, LrA_time, LrA_bio], y_fake)

        logs.append([
            i,
            total_loss,
            nll_loss,
            bce_loss,
            range_mu_loss,
            range_x_loss,
            float(d_real[0]),
            float(d_fake[0]),
        ])

        # Progress print
        if (i + 1) % print_every == 0:
            print(
                f"Step {i+1}/{total_steps}  "
                f"Total={total_loss:.4f}, NLL={nll_loss:.4f}, BCE={bce_loss:.4f}, "
                f"TBILmeanRange={range_mu_loss:.4f}, TBILxRange={range_x_loss:.4f}, "
                f"D_real={float(d_real[0]):.4f}, D_fake={float(d_fake[0]):.4f}"
            )

        # Persist progress step
        with tf.io.gfile.GFile(progress_file, 'w') as pf:
            pf.write(str(i))

        # Periodic: predictions + checkpoints
        if (i + 1) % (steps_per_epoch * 5) == 0:
            summarize_performance(
                i, mean_predictor, Xt, Lt, Tt, Ct,
                mt, 'ControlGenerator', controlScaler,
                resultPath=resultPath
            )
            save_models(i, g_model, d_model, c_model, resultPath=resultPath)

        # Periodic: write loss log
        if (i + 1) % (steps_per_epoch * 50) == 0:
            pd.DataFrame(
                logs,
                columns=[
                    'step', 'total_loss', 'nll_loss', 'bce_loss',
                    'tbil_mu_range', 'tbil_x_range', 'd_real_loss', 'd_fake_loss'
                ]
            ).to_csv(loss_file, index=False)

    # Final loss log
    pd.DataFrame(
        logs,
        columns=[
            'step', 'total_loss', 'nll_loss', 'bce_loss',
            'tbil_mu_range', 'tbil_x_range', 'd_real_loss', 'd_fake_loss'
        ]
    ).to_csv(loss_file, index=False)


# ==============================
# Build & Train
# ==============================
input_shape   = (len(cols),)
label_shape   = LABEL_SHAPE
target_shape  = TARGET_SHAPE
cluster_shape = (CLUSTER_DIM,)
output_shape  = len(cols)

z_dim   = 64
beta_kl = 0.05

# Build models
d_model = define_discriminator(len(cols), STAGE_DIM, BIO_DIM)

e_model = define_encoder(input_shape, label_shape, target_shape, cluster_shape, z_dim=z_dim)
g_model = define_generator_vae(
    input_shape,
    label_shape,
    target_shape,
    cluster_shape,
    output_shape,
    z_dim,
    n_clusters
)

c_model = define_composite_cvae_gan(
    e_model, g_model, d_model,
    input_shape, label_shape, target_shape, cluster_shape,
    beta_kl=beta_kl,
    range_mu_weight=1.0,
    range_x_weight=2.0,
    bio_corr_weight=5.0,      # tune within [0.5, 2.0] if needed
    report_corr_metric=True,
    mean_match_weight=0.05    # tiny weight so NLL still dominates
)

mean_predictor = build_mean_predictor(
    e_model, g_model,
    input_shape, label_shape, target_shape, cluster_shape
)
sampler_model  = build_sampler(
    e_model, g_model,
    input_shape, label_shape, target_shape, cluster_shape
)

# Launch training
train_one_sided_translation(
    d_model, e_model, g_model, c_model,
    mean_predictor, sampler_model,
    treatmentTrain, controlTrain,
    treatmentTest, controlTest,
    controlScaler
)
