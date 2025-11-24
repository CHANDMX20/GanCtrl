"""
Control Generator Inference Script (GanCtrl / VAE–GAN)

This script:
  - Enforces strict reproducibility for TensorFlow / NumPy / Python.
  - Loads pre-split control/treatment clinical pathology CSVs.
  - Merges the latest BODY_WEIGHT measurement per animal (per compound/dose/time).
  - Applies MinMax scaling consistent with training.
  - Reconstructs the clustering and spatial group mask used at training time.
  - Loads the trained composite model (Encoder + Generator) or fallback Generator.
  - Builds two inference models:
        1) mean_predictor: generates mean control predictions
        2) sampler_model: generates sampled control profiles (via VAE sampling)
  - Creates all treatment–control pairings for train and test sets.
  - Writes out decoded (inverse-scaled) predictions and samples to CSV.

Intended usage:
  - Run as a standalone script after training has completed.
  - Adjust `dataPath` and `BASE_RESULTS` paths as needed for your environment.
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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    LeakyReLU, Concatenate, Lambda, Add, Activation
)
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from math import sqrt
import re

# =============================================================================
# Helper functions: I/O and label binarizers
# =============================================================================
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


# =============================================================================
# Load train / test data (control and treatment)
# =============================================================================
dataPath = '/account001/mansi.chandra/clin_path'  # <-- update if needed

controlTrain   = read_data(f"{dataPath}/repeat_train_control_cv5.csv")
controlTest    = read_data(f"{dataPath}/repeat_test_control_cv5.csv")
treatmentTrain = read_data(f"{dataPath}/repeat_train_treatment_cv5.csv")
treatmentTest  = read_data(f"{dataPath}/repeat_test_treatment_cv5.csv")

# Combine for consistent binarizers & EXP_ID ordering
dataset = pd.concat([controlTrain, controlTest, treatmentTrain, treatmentTest], axis=0)
dataset = dataset.sort_values('EXP_ID')

doseBinarizer, stageBinarizer, bioCopyBinarizer = binarizer(dataset)
cols = dataset.columns[11:]  # numeric feature columns (match training)


# =============================================================================
# BODY_WEIGHT merge: keep latest measurement by PROGRESS_TIME per key
# =============================================================================
bw_path = "/account001/mansi.chandra/clin_path/body_wt.csv"  # <-- update if needed
bw_df = pd.read_csv(bw_path, encoding="latin-1")
bw_df.columns = bw_df.columns.str.replace("\ufeff", "", regex=False).str.strip()

# Key columns that define an individual timepoint for BODY_WEIGHT
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

# Clean keys and coerce BODY_WEIGHT to numeric
for k in _BW_KEYS:
    bw_df[k] = bw_df[k].astype(str).str.strip()
bw_df["BODY_WEIGHT"] = pd.to_numeric(bw_df["BODY_WEIGHT"], errors="coerce")

# Convert PROGRESS_TIME to a numeric value for ordering
_prog_dt  = pd.to_datetime(bw_df["PROGRESS_TIME"], errors="coerce", infer_datetime_format=True)
_prog_num = pd.to_numeric(bw_df["PROGRESS_TIME"], errors="coerce")
_prog_val = np.where(~_prog_dt.isna(), _prog_dt.view("int64"), _prog_num)
bw_df["_PROG_VAL"] = pd.Series(_prog_val, index=bw_df.index).astype("float64").fillna(-1e300)

# Keep only the last (latest) BODY_WEIGHT per composite key
bw_df = (
    bw_df.sort_values("_PROG_VAL")
         .drop_duplicates(_BW_KEYS, keep="last")
         .drop(columns=["_PROG_VAL"])
)


def _merge_bw(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Merge latest BODY_WEIGHT into a given split (e.g., controlTrain).

    Validation is strict:
      - Requires all merge keys.
      - Ensures many-to-one merge.
      - Raises if BODY_WEIGHT is missing after merge.
    """
    left_missing = [k for k in _BW_KEYS if k not in df.columns]
    if left_missing:
        raise RuntimeError(f"{name} missing merge keys: {left_missing}")

    left = df.copy()
    for k in _BW_KEYS:
        left[k] = left[k].astype(str).str.strip()

    # Double-check that BODY_WEIGHT is unique per composite key
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
        print("Examples of missing BW rows (up to 5):")
        print(bad.head(5).to_string(index=False))
        raise RuntimeError(f"Missing BODY_WEIGHT for {name}: {len(bad)} composite-key rows.")
    return out


# Attach BODY_WEIGHT to all four splits
controlTrain   = _merge_bw(controlTrain,   "controlTrain")
controlTest    = _merge_bw(controlTest,    "controlTest")
treatmentTrain = _merge_bw(treatmentTrain, "treatmentTrain")
treatmentTest  = _merge_bw(treatmentTest,  "treatmentTest")


# =============================================================================
# Scaling: MinMax scaling consistent with training pipeline
# =============================================================================
def scale(df_fit: pd.DataFrame,
          df_transform: pd.DataFrame,
          feature_cols) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Fit MinMaxScaler on df_fit[feature_cols] and transform df_transform[feature_cols].

    Returns
    -------
    scaled_df : pd.DataFrame (meta columns preserved, features scaled)
    scaler    : MinMaxScaler (fit on df_fit)
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


# NOTE: Scaling is structured to match training-time behavior exactly
treatmentTest, _        = scale(treatmentTrain, treatmentTest, cols)
treatmentTrain, tScaler = scale(treatmentTrain, treatmentTrain, cols)
controlTest, _          = scale(controlTrain, controlTest, cols)
controlTrain, cScaler   = scale(controlTrain, controlTrain, cols)

treatmentScaler = tScaler
controlScaler   = cScaler


# =============================================================================
# Clustering (KMeans on control, propagate clusters to treatment/test)
# =============================================================================
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)

# Cluster control animals on scaled features
controlTrain['cluster'] = kmeans.fit_predict(controlTrain[cols].values)

# Majority cluster per compound (for treatment)
compound_cluster_map = (
    controlTrain.groupby('COMPOUND_NAME')['cluster']
                .agg(lambda x: x.value_counts().idxmax())
                .to_dict()
)

treatmentTrain['cluster'] = (
    treatmentTrain['COMPOUND_NAME']
      .map(compound_cluster_map)
      .fillna(-1)
      .astype(int)
)

clusterB = LabelBinarizer().fit(controlTrain['cluster'])

# For treatment test, assign cluster based on nearest mean (Euclidean in feature space)
treatment_cluster_means = (
    treatmentTrain.groupby('cluster')[cols]
                  .mean()
                  .to_dict(orient='index')
)


def assign_treatment_test_cluster(df: pd.DataFrame,
                                  cluster_means: dict,
                                  feature_cols) -> np.ndarray:
    """Assign each row in df to the nearest cluster mean in cluster_means."""
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

# Override with majority cluster per compound (for stability)
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


# =============================================================================
# Spatial groups / NEIGHBOR_MASK (biologically-informed adjacency)
# =============================================================================
A = len(cols)

# Patterns to locate classical clinical pathology markers within `cols`
ALIASES = {
    "ALT":  [r"^ALT"],
    "AST":  [r"^AST"],
    "LDH":  [r"^LDH"],
    "ALP":  [r"^ALP"],
    "GTP":  [r"(GGT|GTP)"],
    "TBIL": [r"^TBIL|Total\s*Bilirubin"],
    "DBIL": [r"^DBIL|Direct\s*Bilirubin"],
    "TP":   [r"^TP|Total\s*Protein"],
    "RALB": [r"^(ALB|RALB|Albumin)"],
    "A/G":  [r"A/?G"],
    "BUN":  [r"^BUN"],
    "CRE":  [r"^CRE|Creat"],
    "Na":   [r"^Na"],
    "K":    [r"^K(?!g)"],
    "Cl":   [r"^Cl(\b|[^A-Za-z])|Chloride"],
    "Ca":   [r"^Ca(\b|[^A-Za-z])|Calcium"],
    "IP":   [r"^(IP|Phos|Phosphate)"],
    "GLC":  [r"^(GLC|Glucose)"],
    "TC":   [r"^TC|Cholesterol"],
    "TG":   [r"^TG|Triglycer"],
    "PL":   [r"^PL|Phospho.?lipid"],
    "RBC":  [r"^RBC"],
    "Hb":   [r"^(Hb|HGB)"],
    "Ht":   [r"^(Ht|HCT)"],
    "MCV":  [r"^MCV"],
    "MCH":  [r"^MCH(\b|[^A-Za-z])"],
    "MCHC": [r"^MCHC"],
    "Ret":  [r"^Ret"],
    "Plat": [r"^(Plat|PLT)"],
    "PT":   [r"^PT(?!H)"],
    "APTT": [r"^APTT|PTT"],
    "Fbg":  [r"^(Fbg|Fibrinogen)"],
    "WBC":  [r"^WBC"],
    "Neu":  [r"^(Neu|Neutro)"],
    "Lym":  [r"^(Lym|Lympho)"],
    "Mono": [r"^Mono"],
    "Eos":  [r"^Eos"],
    "Bas":  [r"^Bas"],
}


def _idx_from_alias(key: str):
    """Return the index of the first column in `cols` matching any alias pattern."""
    pats = ALIASES.get(key, [])
    for p in pats:
        for j, c in enumerate(cols):
            if re.search(p, c, flags=re.I):
                return j
    return None


idx = {k: _idx_from_alias(k) for k in ALIASES.keys()}

# Group biologically-related markers
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

# Build adjacency mask: 1 for within-group pairs, 0 otherwise
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
    Softmax along last dimension, but only over positions where mask > 0.5.
    Used to keep attention localized to biological neighbors.
    """
    very_neg = tf.constant(-1e9, dtype=logits.dtype)
    masked   = tf.where(mask > 0.5, logits, very_neg)
    return tf.nn.softmax(masked, axis=-1)


# Needed so composite H5 can deserialize MoE corr Lambdas
def _batch_pairwise_corr(xi, xj, eps=1e-6):
    """
    Batch-wise Pearson correlation between xi and xj used by MoE correlation loss.
    """
    xi_c = xi - tf.reduce_mean(xi, axis=0, keepdims=True)
    xj_c = xj - tf.reduce_mean(xj, axis=0, keepdims=True)
    cov  = tf.reduce_mean(xi_c * xj_c, axis=0)
    vi   = tf.reduce_mean(tf.square(xi_c), axis=0) + eps
    vj   = tf.reduce_mean(tf.square(xj_c), axis=0) + eps
    return cov / tf.sqrt(vi * vj)


# =============================================================================
# Model loading: composite (Encoder + Generator) or generator-only
# =============================================================================
STAGE_DIM = len(stageBinarizer.classes_)
BIO_DIM   = len(bioCopyBinarizer.classes_)
DOSE_DIM  = len(doseBinarizer.classes_)
BW_DIM    = 1

LABEL_DIM   = DOSE_DIM + STAGE_DIM + BIO_DIM + BW_DIM     # dose+time+bio+BW (src)
TARGET_DIM  = STAGE_DIM + BIO_DIM + BW_DIM                # time+bio+BW (tgt)
CLUSTER_DIM = clusterB.classes_.size

# ---- IMPORTANT: must match the saved filename suffix (training saves i+1) ----
MODEL_STEP = 895560

BASE_RESULTS = '/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv5'  # <-- match training

enc = None
gen = None

# Prefer loading the composite model (preserves wiring and custom layers)
try:
    comp = keras.models.load_model(
        f'{BASE_RESULTS}/composite_model1/composite_model1_{MODEL_STEP:06d}.h5',
        custom_objects={
            'masked_row_softmax': masked_row_softmax,
            '_batch_pairwise_corr': _batch_pairwise_corr,
            'BinaryCrossentropy': BinaryCrossentropy
        },
        compile=False
    )
    enc = comp.get_layer('Encoder')
    gen = comp.get_layer('Generator_VAE')
except Exception as e:
    print(f"[WARN] Failed to load composite or extract submodels: {e}")
    print("[INFO] Falling back to loading the generator directly.")
    try:
        gen = keras.models.load_model(
            f'{BASE_RESULTS}/model1/g_model1_{MODEL_STEP:06d}.h5',
            custom_objects={'masked_row_softmax': masked_row_softmax},
            compile=False
        )
    except Exception as e2:
        raise RuntimeError(f"Failed to load generator checkpoint: {e2}")


# =============================================================================
# Inference models: mean predictor and sampler
# =============================================================================
def build_mean_predictor(enc_model, gen_model, z_dim: int = 64) -> Model:
    """
    Mean predictor:
      Inputs  : [features, src_label, tgt_label, cluster_onehot]
      Outputs : mean control prediction (mu)
    """
    fi = Input((A,), name='mp_feat_in')
    li = Input((LABEL_DIM,), name='mp_label_in')
    ti = Input((TARGET_DIM,), name='mp_target_in')
    ci = Input((CLUSTER_DIM,), name='mp_cluster_in')

    if enc_model is not None:
        _, _, z = enc_model([fi, li, ti, ci])
    else:
        # If encoder is unavailable, fall back to z = 0
        z = Lambda(
            lambda x: tf.zeros(
                tf.stack([tf.shape(x)[0], tf.constant(z_dim, tf.int32)]),
                dtype=tf.float32
            ),
            name='mp_zero_z'
        )(fi)

    mu_logvar = gen_model([fi, li, ti, z, ci])
    mu = Lambda(lambda y: y[:, :A], name='mp_mu_out')(mu_logvar)
    return Model([fi, li, ti, ci], mu, name='MeanPredictor_usesZ')


def build_sampler(enc_model, gen_model, z_dim: int = 64) -> Model:
    """
    Sampler:
      Inputs  : [features, src_label, tgt_label, cluster_onehot]
      Outputs : VAE sample x ~ N(mu, sigma^2) clipped to [0, 1]
    """
    fi = Input((A,), name='sp_feat_in')
    li = Input((LABEL_DIM,), name='sp_label_in')
    ti = Input((TARGET_DIM,), name='sp_target_in')
    ci = Input((CLUSTER_DIM,), name='sp_cluster_in')

    if enc_model is not None:
        _, _, z_ = enc_model([fi, li, ti, ci])
    else:
        # If encoder is unavailable, fall back to z = 0
        z_ = Lambda(
            lambda x: tf.zeros(
                tf.stack([tf.shape(x)[0], tf.constant(z_dim, tf.int32)]),
                dtype=tf.float32
            ),
            name='sp_zero_z'
        )(fi)

    mu_lv = gen_model([fi, li, ti, z_, ci])
    mu_s  = Lambda(lambda y: y[:, :A], name='sp_mu')(mu_lv)
    lv_s  = Lambda(lambda y: y[:, A:], name='sp_logvar')(mu_lv)
    eps   = Lambda(lambda m: K.abs(K.random_normal(shape=K.shape(m))), name='sp_eps')(mu_s)

    x_s = Lambda(
        lambda args: tf.clip_by_value(
            args[0] + K.exp(0.5 * args[1]) * args[2],
            0.0, 1.0
        ),
        name='sp_sample_clip'
    )([mu_s, lv_s, eps])

    return Model([fi, li, ti, ci], x_s, name='Sampler')


mean_predictor = build_mean_predictor(enc, gen)
sampler_model  = build_sampler(enc, gen)


# =============================================================================
# Pair creation: generate all treatment–control pairs for evaluation
# =============================================================================
def generate_test_real_samples(treatment_df: pd.DataFrame,
                               control_df: pd.DataFrame):
    """
    Construct all treatment–control pairs for a given split.

    - Pairs are within compound and sacrifice period.
    - For each treatment row, we pair with all matching control animals.
    - Adds target BODY_WEIGHT as BODY_WEIGHT_TGT.
    - Returns:
        data  : meta DataFrame (with cluster, labels and pairing info)
        feat  : feature matrix for treatment rows
        src_lbl : source label matrix (dose + time + bio + BW_src)
        tgt_lbl : target label matrix (time + bio + BW_tgt)
        C       : one-hot cluster matrix
    """
    df = (
        pd.concat([treatment_df, control_df], axis=0)
          .sort_values(['COMPOUND_NAME', 'SACRIFICE_PERIOD'])
          .reset_index(drop=True)
    )

    def pairs(src: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
        # Cartesian pairing: all src x all tgt for the same compound/time
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

    # Attach target BODY_WEIGHT
    _target_bw = (
        control_df[['ID', 'BODY_WEIGHT']]
        .rename(columns={'ID': 'targetId', 'BODY_WEIGHT': 'BODY_WEIGHT_TGT'})
    )
    data = data.merge(_target_bw, on='targetId', how='left', validate='many_to_one')
    if data['BODY_WEIGHT_TGT'].isna().any():
        raise RuntimeError(
            "Missing target BODY_WEIGHT in generated test pairs; check targetId mapping."
        )

    # Cluster labels
    data['cluster'] = data['cluster'].astype(int)

    # Features (treatment)
    feat = data[cols].astype('float32').values

    # Source labels: dose + time + bio + BW_src
    tvals = stageBinarizer.transform(data['SACRIFICE_PERIOD'])
    dvals = doseBinarizer.transform(data['DOSE_LEVEL'])
    bvals = bioCopyBinarizer.transform(
        data['INDIVIDUAL_ID'].astype(bioCopyBinarizer.classes_.dtype)
    )

    # Target labels: time + bio + BW_tgt
    tt = stageBinarizer.transform(data['targetTime'])
    tb = bioCopyBinarizer.transform(
        data['targetBioCopy'].astype(bioCopyBinarizer.classes_.dtype)
    )

    bw_src = data[['BODY_WEIGHT']].values.astype('float32')
    bw_tgt = data[['BODY_WEIGHT_TGT']].values.astype('float32')

    src_lbl = np.hstack([dvals, tvals, bvals, bw_src]).astype('float32')
    tgt_lbl = np.hstack([tt, tb, bw_tgt]).astype('float32')

    # Cluster one-hot
    C = clusterB.transform(data['cluster'])

    return data, feat, src_lbl, tgt_lbl, C


# =============================================================================
# Prediction helpers: decode and write means and samples
# =============================================================================
def summarize_means(
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
    resultPath: str = BASE_RESULTS,
):
    """
    Run mean predictor, inverse scale, and write decoded means to CSV.
    """
    feature_list = list(features)
    pm = mean_predictor.predict([in_f, in_l, in_t, in_c], verbose=0)
    mean_rescaled = scaler.inverse_transform(pm)
    mean_df = pd.DataFrame(mean_rescaled, columns=feature_list)

    # Drop feature columns from meta (if present) and recombine
    meta = meta_df.drop(columns=feature_list, errors='ignore').reset_index(drop=True)
    out = pd.concat([meta, mean_df], axis=1)
    out = out[list(meta.columns) + feature_list]

    out_dir = f"{resultPath}/predictions_decoded/test"
    tf.io.gfile.makedirs(out_dir)
    out.to_csv(
        f"{out_dir}/generated_predictions_{step:06d}_{name}.csv",
        index=False
    )


def summarize_samples(
    step: int,
    sampler_model: Model,
    num_samples: int,
    in_f: np.ndarray,
    in_l: np.ndarray,
    in_t: np.ndarray,
    in_c: np.ndarray,
    meta_df: pd.DataFrame,
    name: str,
    scaler: MinMaxScaler,
    features=cols,
    resultPath: str = BASE_RESULTS,
):
    """
    Run sampler `num_samples` times, inverse scale, and write decoded samples to CSV.
    Each sample uses a fresh epsilon (eps) draw from the VAE.
    """
    feature_list = list(features)
    out_dir = f"{resultPath}/predictions_decoded/test/samples"
    tf.io.gfile.makedirs(out_dir)

    meta = meta_df.drop(columns=feature_list, errors='ignore').reset_index(drop=True)

    for s in range(1, num_samples + 1):
        xs = sampler_model.predict([in_f, in_l, in_t, in_c], verbose=0)  # fresh eps each call
        xs = np.clip(xs, 0.0, 1.0)
        xs_rescaled = scaler.inverse_transform(xs)
        xs_df = pd.DataFrame(xs_rescaled, columns=feature_list)

        out = pd.concat([meta, xs_df], axis=1)
        out = out[list(meta.columns) + feature_list]

        out.to_csv(
            f"{out_dir}/generated_samples_s{s}_{step:06d}_{name}.csv",
            index=False
        )


# =============================================================================
# Main execution: run for train and test (means + samples)
# =============================================================================
if __name__ == "__main__":
    NUM_SAMPLES = 5  # number of VAE draws per paired example

    # ------------------------------
    # Train split
    # ------------------------------
    mt_train, Xt_train, Lt_train, Tt_train, Ct_train = generate_test_real_samples(
        treatmentTrain, controlTrain
    )
    summarize_means(
        MODEL_STEP, mean_predictor,
        Xt_train, Lt_train, Tt_train, Ct_train,
        mt_train, 'ControlGenerator_train', controlScaler
    )
    summarize_samples(
        MODEL_STEP, sampler_model, NUM_SAMPLES,
        Xt_train, Lt_train, Tt_train, Ct_train,
        mt_train, 'ControlGenerator_train', controlScaler
    )

    # ------------------------------
    # Test split
    # ------------------------------
    mt_test, Xt_test, Lt_test, Tt_test, Ct_test = generate_test_real_samples(
        treatmentTest, controlTest
    )
    summarize_means(
        MODEL_STEP, mean_predictor,
        Xt_test, Lt_test, Tt_test, Ct_test,
        mt_test, 'ControlGenerator_test', controlScaler
    )
    summarize_samples(
        MODEL_STEP, sampler_model, NUM_SAMPLES,
        Xt_test, Lt_test, Tt_test, Ct_test,
        mt_test, 'ControlGenerator_test', controlScaler
    )
