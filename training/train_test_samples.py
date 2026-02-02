"""
Control Generator Inference Script (GanCtrl / VAE–GAN)

This script:
  - Enforces strict reproducibility for TensorFlow / NumPy / Python.
  - Loads pre-split control/treatment clinical pathology CSVs (2D layout).
  - Merges the latest BODY_WEIGHT measurement per animal (by composite key + PROGRESS_TIME).
  - Applies MinMax scaling consistent with training (ONLY clinical parameter columns).
  - Reconstructs clustering and the spatial neighbor mask used at training time.
  - Loads a trained composite H5 model (Encoder + Generator) and extracts submodels.
  - Builds two inference models:
        1) mean_predictor: returns predicted control means (mu)
        2) sampler_model: returns sampled control profiles (VAE sampling with clipped logvar)
  - Creates all treatment–control pairings (time-matched, per compound) for train/test.
  - Writes decoded (inverse-scaled) predictions (and optional samples) to CSV.

Intended usage:
  - Run as a standalone script after training has completed.
  - Place expected CSVs under ./data (or set env var GANCTRL_DATA_DIR).
  - Point BASE_RESULTS to your trained model directory (or set GANCTRL_RESULTS_DIR).
"""

# =============================================================================
# Reproducibility (MUST be first)
# =============================================================================
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

# =============================================================================
# Imports
# =============================================================================
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, StandardScaler
from math import sqrt
import re

# =============================================================================
# Paths (GitHub-friendly defaults)
#   - dataPath: directory containing the 2D CSV splits + body_wt.csv
#   - BASE_RESULTS: directory containing trained model checkpoints + output folders
# =============================================================================
dataPath = os.environ.get("GANCTRL_DATA_DIR", "./data")
BASE_RESULTS = os.environ.get("GANCTRL_RESULTS_DIR", "./results")
bw_path = os.environ.get("GANCTRL_BODY_WEIGHT_CSV", f"{dataPath}/body_wt.csv")

# ============================== I/O helpers ==============================
def read_data(path):
    return pd.read_csv(path)

def binarizer(data):
    """
    Fit binarizers for:
      - dose (excluding Control)
      - sacrifice period (timepoint)
      - individual id (bio replicate)
    """
    dose_levels = data.loc[data['DOSE_LEVEL'] != 'Control', 'DOSE_LEVEL'].find
    dose_levels = data.loc[data['DOSE_LEVEL'] != 'Control', 'DOSE_LEVEL'].unique()
    doseBinarizer    = LabelBinarizer().fit(dose_levels)
    stageBinarizer   = LabelBinarizer().fit(data['SACRIFICE_PERIOD'])
    bioCopyBinarizer = LabelBinarizer().fit(data['INDIVIDUAL_ID'])
    return doseBinarizer, stageBinarizer, bioCopyBinarizer

# =============================================================================
# Load data (2D CSV splits)
#   Expected files under dataPath:
#     repeat_train_control_2d.csv
#     repeat_test_control_2d.csv
#     repeat_train_treatment_2d.csv
#     repeat_test_treatment_2d.csv
# =============================================================================
controlTrain   = read_data(f"{dataPath}/repeat_train_control_2d.csv")
controlTest    = read_data(f"{dataPath}/repeat_test_control_2d.csv")
treatmentTrain = read_data(f"{dataPath}/repeat_train_treatment_2d.csv")
treatmentTest  = read_data(f"{dataPath}/repeat_test_treatment_2d.csv")

dataset = pd.concat([controlTrain, controlTest, treatmentTrain, treatmentTest], axis=0)
dataset = dataset.sort_values('EXP_ID')

doseBinarizer, stageBinarizer, bioCopyBinarizer = binarizer(dataset)

# =============================================================================
# Column slices (MATCH TRAINING)
#   - mol descriptors: columns [11:PARAMS_START)
#   - clinical parameters (predicted outputs): [PARAMS_START:]
# =============================================================================
PARAMS_START = 1368
if len(dataset.columns) <= PARAMS_START:
    raise RuntimeError(
        f"Expected parameters to start at column index {PARAMS_START}, "
        f"but dataset has only {len(dataset.columns)} columns."
    )

MOL_COLS = list(dataset.columns[11:PARAMS_START])
cols     = list(dataset.columns[PARAMS_START:])
A        = len(cols)

# =============================================================================
# BODY_WEIGHT merge
#   - keep latest per composite key using PROGRESS_TIME
# =============================================================================
bw_df = pd.read_csv(bw_path, encoding="latin-1")
bw_df.columns = bw_df.columns.str.replace("\ufeff", "", regex=False).str.strip()

_BW_KEYS = ["COMPOUND_NAME","DOSE_LEVEL","SACRIFICE_PERIOD","EXP_ID","GROUP_ID","INDIVIDUAL_ID"]
_required_cols = _BW_KEYS + ["BODY_WEIGHT"]
_missing = [c for c in _required_cols if c not in bw_df.columns]
if _missing:
    raise RuntimeError(f"BODY_WEIGHT file missing columns: {_missing}")
if "PROGRESS_TIME" not in bw_df.columns:
    raise RuntimeError("BODY_WEIGHT file is missing 'PROGRESS_TIME' column.")

for k in _BW_KEYS:
    bw_df[k] = bw_df[k].astype(str).str.strip()
bw_df["BODY_WEIGHT"] = pd.to_numeric(bw_df["BODY_WEIGHT"], errors="coerce")

_prog_dt  = pd.to_datetime(bw_df["PROGRESS_TIME"], errors="coerce", infer_datetime_format=True)
_prog_num = pd.to_numeric(bw_df["PROGRESS_TIME"], errors="coerce")
_prog_val = np.where(~_prog_dt.isna(), _prog_dt.view("int64"), _prog_num)
bw_df["_PROG_VAL"] = pd.Series(_prog_val, index=bw_df.index).astype("float64").fillna(-1e300)

bw_df = (
    bw_df.sort_values("_PROG_VAL")
         .drop_duplicates(_BW_KEYS, keep="last")
         .drop(columns=["_PROG_VAL"])
)

def _merge_bw(df, name):
    """
    Left-join BODY_WEIGHT onto each split and fail loudly if any rows are missing BW.
    """
    left_missing = [k for k in _BW_KEYS if k not in df.columns]
    if left_missing:
        raise RuntimeError(f"{name} missing merge keys: {left_missing}")

    left = df.copy()
    for k in _BW_KEYS:
        left[k] = left[k].astype(str).str.strip()

    if bw_df.duplicated(_BW_KEYS).any():
        dups = bw_df[bw_df.duplicated(_BW_KEYS, keep=False)][_BW_KEYS].drop_duplicates()
        raise RuntimeError(f"Duplicate BW rows remain after latest-per-key selection: {len(dups)} examples")

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

controlTrain   = _merge_bw(controlTrain,   "controlTrain")
controlTest    = _merge_bw(controlTest,    "controlTest")
treatmentTrain = _merge_bw(treatmentTrain, "treatmentTrain")
treatmentTest  = _merge_bw(treatmentTest,  "treatmentTest")

# =============================================================================
# Scaling (MATCH TRAINING: scale ONLY `cols`)
#   - Fit scaler on train split
#   - Transform the paired split(s)
# =============================================================================
def scale(df1, df2, cols):
    X1 = df1[cols]
    X2 = df2[cols]
    scaler = MinMaxScaler()
    scaler.fit(X1)
    X2 = scaler.transform(X2)
    scaled_df = pd.DataFrame(X2, columns=cols, index=df2.index)
    meta_cols = [c for c in df2.columns if c not in cols]
    result_df = pd.concat([df2[meta_cols], scaled_df], axis=1)
    return result_df, scaler

treatmentTest, _  = scale(treatmentTrain, treatmentTest, cols=cols)
treatmentTrain, treatmentScaler = scale(treatmentTrain, treatmentTrain, cols=cols)
controlTest, _  = scale(controlTrain, controlTest, cols=cols)
controlTrain, controlScaler = scale(controlTrain, controlTrain, cols=cols)

# =============================================================================
# Molecular descriptor scaling (MATCH TRAINING)
#   - StandardScaler for encoder stability
#   - Fit on treatmentTrain mols; apply everywhere
# =============================================================================
def _fit_mol_scaler(train_df, mol_cols):
    X = train_df[mol_cols].astype('float32').values
    if np.isnan(X).any() or np.isinf(X).any():
        raise RuntimeError("NaN/Inf found in MOL_COLS before scaling. Clean mol descriptors first.")
    sc = StandardScaler(with_mean=True, with_std=True)
    sc.fit(X)
    return sc

def _apply_mol_scaler(df, mol_cols, scaler):
    X = df[mol_cols].astype('float32').values
    X = np.where(np.isfinite(X), X, np.nan).astype('float32')
    Xs = scaler.transform(np.nan_to_num(X, nan=0.0)).astype('float32')
    out = df.copy()
    out[mol_cols] = Xs
    return out

mol_scaler = _fit_mol_scaler(treatmentTrain, MOL_COLS)
controlTrain   = _apply_mol_scaler(controlTrain,   MOL_COLS, mol_scaler)
controlTest    = _apply_mol_scaler(controlTest,    MOL_COLS, mol_scaler)
treatmentTrain = _apply_mol_scaler(treatmentTrain, MOL_COLS, mol_scaler)
treatmentTest  = _apply_mol_scaler(treatmentTest,  MOL_COLS, mol_scaler)

# =============================================================================
# Clustering (MATCH TRAINING)
#   - KMeans on controlTrain
#   - Map treatment clusters by compound-majority
#   - Assign treatmentTest by nearest cluster mean then per-compound majority
# =============================================================================
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)

controlTrain['cluster'] = kmeans.fit_predict(controlTrain[cols].values)

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

treatment_cluster_means = (
    treatmentTrain.groupby('cluster')[cols]
                  .mean()
                  .to_dict(orient='index')
)

def assign_treatment_test_cluster(df, cluster_means, cols):
    assigned = []
    for _, row in df.iterrows():
        x = row[cols].values.astype(float)
        best = min(
            cluster_means.keys(),
            key=lambda cl: sqrt(np.sum((x - np.array(list(cluster_means[cl].values())))**2))
        )
        assigned.append(best)
    return np.array(assigned, dtype=int)

treatmentTest['cluster'] = assign_treatment_test_cluster(treatmentTest, treatment_cluster_means, cols)

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
# Spatial groups mask (MATCH TRAINING)
#   - required for generator's mixing/attention mask
# =============================================================================
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

def _idx_from_alias(key):
    pats = ALIASES.get(key, [])
    for p in pats:
        for j, c in enumerate(cols):
            if re.search(p, c, flags=re.I):
                return j
    return None

idx = {k: _idx_from_alias(k) for k in ALIASES.keys()}

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
_M = np.zeros((A, A), dtype="float32")
for _, glist in GROUPS.items():
    g = [g for g in glist if g is not None]
    for i in g:
        for j in g:
            _M[i, j] = 1.0
np.fill_diagonal(_M, 1.0)
NEIGHBOR_MASK = tf.constant(_M, dtype=tf.float32)

def masked_row_softmax(logits, mask):
    very_neg = tf.constant(-1e9, dtype=logits.dtype)
    masked   = tf.where(mask > 0.5, logits, very_neg)
    return tf.nn.softmax(masked, axis=-1)

# Needed so H5 can deserialize Lambda layers saved in the composite model
def _batch_pairwise_corr(xi, xj, eps=1e-6):
    xi_c = xi - tf.reduce_mean(xi, axis=0, keepdims=True)
    xj_c = xj - tf.reduce_mean(xj, axis=0, keepdims=True)
    cov  = tf.reduce_mean(xi_c * xj_c, axis=0)
    vi   = tf.reduce_mean(tf.square(xi_c), axis=0) + eps
    vj   = tf.reduce_mean(tf.square(xj_c), axis=0) + eps
    return cov / tf.sqrt(vi * vj)

# =============================================================================
# Label dims (derived from binarizers)
# =============================================================================
STAGE_DIM = len(stageBinarizer.classes_)
BIO_DIM   = len(bioCopyBinarizer.classes_)
DOSE_DIM  = len(doseBinarizer.classes_)
BW_DIM    = 1

LABEL_DIM   = DOSE_DIM + STAGE_DIM + BIO_DIM + BW_DIM     # dose+time+bio+BW (src)
TARGET_DIM  = STAGE_DIM + BIO_DIM + BW_DIM                # time+bio+BW (tgt)
CLUSTER_DIM = clusterB.classes_.size
MOL_DIM     = len(MOL_COLS)

# =============================================================================
# Load composite model and extract Encoder + Generator_VAE
# =============================================================================
MODEL_STEP = int(os.environ.get("GANCTRL_MODEL_STEP", "900705"))

# Enable unsafe deserialization for Lambda layers saved in H5 (TensorFlow/Keras version dependent)
try:
    keras.config.enable_unsafe_deserialization()
except Exception:
    try:
        tf.keras.saving.enable_unsafe_deserialization()
    except Exception:
        pass

CUSTOM_OBJECTS = {
    'masked_row_softmax': masked_row_softmax,
    '_batch_pairwise_corr': _batch_pairwise_corr,
    'BinaryCrossentropy': BinaryCrossentropy,
}

comp = keras.models.load_model(
    f'{BASE_RESULTS}/composite_model1/composite_model1_{MODEL_STEP:06d}.h5',
    custom_objects=CUSTOM_OBJECTS,
    compile=False
)

enc = comp.get_layer('Encoder')
gen = comp.get_layer('Generator_VAE')

def infer_z_dim(enc_model, default=64):
    """Infer z_dim from encoder output shapes (fallback to default)."""
    try:
        z_shape = enc_model.output_shape[2]   # outputs: [z_mean, z_logvar, z]
        return int(z_shape[-1])
    except Exception:
        return int(default)

Z_DIM = infer_z_dim(enc, default=64)
print(f"[INFO] Using z_dim = {Z_DIM}")

# =============================================================================
# Predictors (ENC takes mol; GEN does NOT)
# =============================================================================
def build_mean_predictor(enc_model, gen_model):
    fi = Input((A,),           name='mp_feat_in')
    mi = Input((MOL_DIM,),     name='mp_mol_in')
    li = Input((LABEL_DIM,),   name='mp_label_in')
    ti = Input((TARGET_DIM,),  name='mp_target_in')
    ci = Input((CLUSTER_DIM,), name='mp_cluster_in')

    _, _, z = enc_model([fi, mi, li, ti, ci])
    mu_logvar = gen_model([fi, li, ti, z, ci])
    mu = Lambda(lambda y: y[:, :A], name='mp_mu_out')(mu_logvar)
    return Model([fi, mi, li, ti, ci], mu, name='MeanPredictor')

def build_sampler(enc_model, gen_model, floor=-3.0, ceil=-1.5):
    fi = Input((A,),           name='sp_feat_in')
    mi = Input((MOL_DIM,),     name='sp_mol_in')
    li = Input((LABEL_DIM,),   name='sp_label_in')
    ti = Input((TARGET_DIM,),  name='sp_target_in')
    ci = Input((CLUSTER_DIM,), name='sp_cluster_in')

    _, _, z = enc_model([fi, mi, li, ti, ci])
    mu_lv = gen_model([fi, li, ti, z, ci])
    mu    = Lambda(lambda y: y[:, :A], name='sp_mu')(mu_lv)
    lv    = Lambda(lambda y: y[:, A:], name='sp_logvar')(mu_lv)

    lv_clip = Lambda(lambda v, f=floor, c=ceil: tf.clip_by_value(v, f, c), name='sp_logvar_clip')(lv)

    eps = Lambda(lambda m: K.abs(K.random_normal(shape=K.shape(m))), name='sp_eps')(mu)
    x_s = Lambda(
        lambda args: tf.clip_by_value(args[0] + K.exp(0.5*args[1]) * args[2], 0.0, 1.0),
        name='sp_sample_clip'
    )([mu, lv_clip, eps])

    return Model([fi, mi, li, ti, ci], x_s, name='Sampler')

mean_predictor = build_mean_predictor(enc, gen)
sampler_model  = build_sampler(enc, gen, floor=-3.0, ceil=-1.5)

# =============================================================================
# Pairing & inputs (returns mol batch too)
#   - builds all time-matched treatment→control pairs per compound
# =============================================================================
def generate_test_real_samples(treatment_df, control_df):
    df = pd.concat([treatment_df, control_df], axis=0) \
           .sort_values(['COMPOUND_NAME','SACRIFICE_PERIOD']) \
           .reset_index(drop=True)

    def pairs(src, tgt):
        left  = pd.DataFrame(np.repeat(src.values, len(tgt), axis=0), columns=src.columns)
        right = pd.concat([tgt[['ID','SACRIFICE_PERIOD','DOSE_LEVEL','INDIVIDUAL_ID']]] * len(src),
                          ignore_index=True)
        right.columns = ['targetId','targetTime','targetDose','targetBioCopy']
        return pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)

    all_pairs = []
    for cmpd in df.COMPOUND_NAME.unique():
        sub = df[df.COMPOUND_NAME == cmpd]
        for tp in sub.SACRIFICE_PERIOD.unique():
            tr = sub[(sub.DOSE_LEVEL != 'Control') & (sub.SACRIFICE_PERIOD == tp)]
            ct = sub[(sub.DOSE_LEVEL == 'Control') & (sub.SACRIFICE_PERIOD == tp)]
            if len(tr) and len(ct):
                all_pairs.append(pairs(tr, ct))

    if not all_pairs:
        raise RuntimeError("No treatment-control time-matched pairs found. Check compound/time overlap.")

    data = pd.concat(all_pairs, ignore_index=True)

    _target_bw = control_df[['ID', 'BODY_WEIGHT']].rename(columns={'ID': 'targetId', 'BODY_WEIGHT': 'BODY_WEIGHT_TGT'})
    data = data.merge(_target_bw, on='targetId', how='left', validate='many_to_one')
    if data['BODY_WEIGHT_TGT'].isna().any():
        raise RuntimeError("Missing target BODY_WEIGHT in generated pairs; check targetId mapping.")

    data['cluster'] = data['cluster'].astype(int)

    feat = data[cols].astype('float32').values
    mol  = data[MOL_COLS].astype('float32').values  # StandardScaled above

    tvals = stageBinarizer.transform(data['SACRIFICE_PERIOD'])
    dvals = doseBinarizer.transform(data['DOSE_LEVEL'])
    bvals = bioCopyBinarizer.transform(data['INDIVIDUAL_ID'].astype(bioCopyBinarizer.classes_.dtype))

    tt = stageBinarizer.transform(data['targetTime'])
    tb = bioCopyBinarizer.transform(data['targetBioCopy'].astype(bioCopyBinarizer.classes_.dtype))

    bw_src = data[['BODY_WEIGHT']].values.astype('float32')
    bw_tgt = data[['BODY_WEIGHT_TGT']].values.astype('float32')

    src_lbl = np.hstack([dvals, tvals, bvals, bw_src]).astype('float32')
    tgt_lbl = np.hstack([tt, tb, bw_tgt]).astype('float32')

    C = clusterB.transform(data['cluster'])
    return data, feat, mol, src_lbl, tgt_lbl, C

# =============================================================================
# Prediction & write-out (decoded / inverse-scaled)
# =============================================================================
def summarize_means(
    step, mean_predictor,
    in_f, in_m, in_l, in_t, in_c,
    meta_df, name, scaler,
    features=cols,
    mol_cols=MOL_COLS,
    resultPath=BASE_RESULTS
):
    feature_list = list(features)
    pm = mean_predictor.predict([in_f, in_m, in_l, in_t, in_c], verbose=0)
    mean_rescaled = scaler.inverse_transform(pm)
    mean_df = pd.DataFrame(mean_rescaled, columns=feature_list)

    meta = meta_df.drop(columns=(feature_list + list(mol_cols)), errors='ignore').reset_index(drop=True)
    out = pd.concat([meta, mean_df], axis=1)
    out = out[list(meta.columns) + feature_list]

    out_dir = f"{resultPath}/predictions_decoded/test"
    tf.io.gfile.makedirs(out_dir)
    out.to_csv(
        f"{out_dir}/generated_predictions_{step:06d}_{name}.csv",
        index=False
    )

def summarize_samples(
    step, sampler_model, num_samples,
    in_f, in_m, in_l, in_t, in_c,
    meta_df, name, scaler,
    features=cols,
    mol_cols=MOL_COLS,
    resultPath=BASE_RESULTS
):
    feature_list = list(features)
    out_dir = f"{resultPath}/predictions_decoded/test/samples"
    tf.io.gfile.makedirs(out_dir)

    meta = meta_df.drop(columns=(feature_list + list(mol_cols)), errors='ignore').reset_index(drop=True)

    for s in range(1, num_samples + 1):
        xs = sampler_model.predict([in_f, in_m, in_l, in_t, in_c], verbose=0)
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
# Run: train + test (means + optional samples)
# =============================================================================
NUM_SAMPLES = int(os.environ.get("GANCTRL_NUM_SAMPLES", "5"))

# ---- train pairs ----
mt, Xt, Mt, Lt, Tt, Ct = generate_test_real_samples(treatmentTrain, controlTrain)
summarize_means(MODEL_STEP, mean_predictor, Xt, Mt, Lt, Tt, Ct, mt, 'ControlGenerator_train', controlScaler)
# summarize_samples(MODEL_STEP, sampler_model, NUM_SAMPLES, Xt, Mt, Lt, Tt, Ct, mt, 'ControlGenerator_train', controlScaler)

# ---- test pairs ----
mt, Xt, Mt, Lt, Tt, Ct = generate_test_real_samples(treatmentTest, controlTest)
summarize_means(MODEL_STEP, mean_predictor, Xt, Mt, Lt, Tt, Ct, mt, 'ControlGenerator_test', controlScaler)
# summarize_samples(MODEL_STEP, sampler_model, NUM_SAMPLES, Xt, Mt, Lt, Tt, Ct, mt, 'ControlGenerator_test', controlScaler)
