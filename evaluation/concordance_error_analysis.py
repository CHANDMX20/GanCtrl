"""Evaluate endpoint-level concordance using GanCtrl synthetic controls.

The script compares high-dose treatment abnormality calls obtained using real
concurrent controls with calls obtained using GanCtrl synthetic controls.

Workflow
--------
1. Load test-set real controls, test-set high-dose treatments, generated
   synthetic controls, and feature-specific thresholds.
2. Harmonize generated clinical-pathology values with the precision used in
   the real TG-GATEs data.
3. Calculate animal-level z-scores using either real or synthetic controls.
4. Collapse animal-level z-scores to treatment-level abnormal/normal calls.
5. Compare real-control and synthetic-control calls using TP, TN, FP, FN,
   accuracy, sensitivity, specificity, and balanced accuracy.
6. Report the focused hepatotoxicity/nephrotoxicity endpoints.

Input paths are intentionally generic so the script can be run from a
repository directory containing the required CSV files.
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------
TEST_CONTROL_FILE = "repeat_test_control_2d.csv"
TEST_TREATMENT_FILE = "repeat_test_treatment_2d.csv"
GENERATED_CONTROL_FILE = "generated_predictions_merged_test.csv"
THRESHOLD_FILE = "train_threshold.csv"


# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------
METADATA_COLS = 11
FEATURE_START_COL = 1368
MIN_CONTROL_N = 2
REAL_Z_THRESHOLD = 2.0
DEFAULT_SYNTHETIC_Z_THRESHOLD = 2.0

COMPOUND_COL = "COMPOUND_NAME"
TIME_COL = "SACRIFICE_PERIOD"
GENERATED_TIME_COL = "targetTime"
DOSE_COL = "DOSE_LEVEL"
ID_COL = "INDIVIDUAL_ID"

# The original analysis treats a synthetic call with no usable z-score as
# normal when constructing the confusion matrix. This flag makes that choice
# explicit and easy to change if a different missing-data policy is desired.
FILL_MISSING_SYNTHETIC_AS_NORMAL = True


# Generated endpoints rounded to the measurement precision used in the real
# dataset before z-score calculation.
GENERATED_ROUNDING = {
    "RALB(g/dL)": 1,
    "AST(IU/L)": 0,
    "TP(g/dL)": 1,
    "Ca(mg/dL)": 1,
    "Cl(meq/L)": 1,
    "IP(mg/dL)": 1,
    "ALP(IU/L)": 0,
    "ALT(IU/L)": 0,
    "LDH(IU/L)": 0,
}


FOCUSED_ENDPOINTS = [
    "ALP(IU/L)",
    "ALT(IU/L)",
    "AST(IU/L)",
    "GTP(IU/L)",
    "LDH(IU/L)",
    "DBIL(mg/dL)",
    "TBIL(mg/dL)",
    "BUN(mg/dL)",
    "CRE(mg/dL)",
    "Ca(mg/dL)",
    "Cl(meq/L)",
    "IP(mg/dL)",
    "K(meq/L)",
    "Na(meq/L)",
]


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
def select_analysis_columns(df):
    """Keep the first metadata columns and clinical-pathology features."""
    return pd.concat(
        [df.iloc[:, :METADATA_COLS], df.iloc[:, FEATURE_START_COL:]],
        axis=1,
    )


def round_generated_values(generated):
    """Match selected generated endpoints to the precision of real values."""
    generated = generated.copy()

    for feature, decimals in GENERATED_ROUNDING.items():
        if feature not in generated.columns:
            continue

        generated[feature] = pd.to_numeric(
            generated[feature], errors="coerce"
        ).round(decimals)

    return generated


def validate_required_columns(df, required, dataframe_name):
    """Raise a clear error when required columns are missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def prepare_feature_columns(real, generated):
    """Return clinical features shared by the real and generated datasets."""
    real_features = list(real.columns[METADATA_COLS:])
    shared_features = [
        feature for feature in real_features if feature in generated.columns
    ]

    if not shared_features:
        raise ValueError(
            "No shared clinical-pathology feature columns were found between "
            "the real and generated datasets."
        )

    return shared_features


def make_features_numeric(df, feature_cols):
    """Convert clinical feature columns to numeric values."""
    df = df.copy()
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return df


# -----------------------------------------------------------------------------
# Z-score calculation
# -----------------------------------------------------------------------------
def calculate_real_control_zscores(real, feature_cols):
    """Calculate treatment z-scores using real concurrent controls."""
    rows = []

    for (compound, time), group in real.groupby(
        [COMPOUND_COL, TIME_COL],
        dropna=False,
        sort=False,
    ):
        controls = group[group[DOSE_COL] == "Control"]
        treatments = group[group[DOSE_COL] == "High"]

        if controls.shape[0] < MIN_CONTROL_N or treatments.empty:
            continue

        control_means = controls[feature_cols].mean()
        control_sds = controls[feature_cols].std(ddof=1)

        for _, row in treatments.iterrows():
            record = {
                "compound": compound,
                "time": time,
                "dose": "High",
                ID_COL: row[ID_COL],
            }

            for feature in feature_cols:
                value = row[feature]
                mean = control_means[feature]
                sd = control_sds[feature]

                record[f"{feature}_z"] = (
                    np.nan
                    if pd.isna(value) or pd.isna(mean) or pd.isna(sd) or sd == 0
                    else (value - mean) / sd
                )

            rows.append(record)

    return pd.DataFrame(rows)


def calculate_synthetic_control_zscores(real, generated, feature_cols):
    """Calculate treatment z-scores using GanCtrl synthetic controls."""
    rows = []

    for (compound, time), synthetic_controls in generated.groupby(
        [COMPOUND_COL, GENERATED_TIME_COL],
        dropna=False,
        sort=False,
    ):
        if synthetic_controls.shape[0] < MIN_CONTROL_N:
            continue

        treatments = real.loc[
            (real[COMPOUND_COL] == compound)
            & (real[TIME_COL] == time)
            & (real[DOSE_COL] == "High")
        ]

        if treatments.empty:
            continue

        control_means = synthetic_controls[feature_cols].mean()
        control_sds = synthetic_controls[feature_cols].std(ddof=1)

        for _, row in treatments.iterrows():
            record = {
                "compound": compound,
                "time": time,
                "dose": "High",
                ID_COL: row[ID_COL],
            }

            for feature in feature_cols:
                value = row[feature]
                mean = control_means[feature]
                sd = control_sds[feature]

                record[f"{feature}_z"] = (
                    np.nan
                    if pd.isna(value) or pd.isna(mean) or pd.isna(sd) or sd == 0
                    else (value - mean) / sd
                )

            rows.append(record)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Treatment-level calls
# -----------------------------------------------------------------------------
def load_thresholds(threshold_df):
    """Convert the feature-threshold table to a lookup dictionary."""
    validate_required_columns(
        threshold_df,
        ["feature", "best_threshold"],
        "Threshold table",
    )

    threshold_df = threshold_df.copy()
    threshold_df["best_threshold"] = pd.to_numeric(
        threshold_df["best_threshold"], errors="coerce"
    )

    return threshold_df.set_index("feature")["best_threshold"].to_dict()


def build_feature_calls(results_df, threshold_lookup=None, call_col="call"):
    """Collapse animal-level z-scores to treatment-level endpoint calls.

    A treatment is called abnormal for an endpoint when at least one animal has
    |z| greater than the endpoint threshold. Missing z-scores remain missing.
    """
    keys = ["compound", "time", "dose"]
    z_cols = [column for column in results_df.columns if column.endswith("_z")]
    call_rows = []

    for z_col in z_cols:
        feature = z_col.removesuffix("_z")

        if threshold_lookup is None:
            threshold = REAL_Z_THRESHOLD
        else:
            threshold = threshold_lookup.get(
                feature, DEFAULT_SYNTHETIC_Z_THRESHOLD
            )
            if pd.isna(threshold):
                threshold = DEFAULT_SYNTHETIC_Z_THRESHOLD

        z_scores = results_df[z_col]
        abnormal = z_scores.abs().gt(threshold)
        abnormal = abnormal.where(z_scores.notna(), pd.NA).astype("boolean")

        grouped = (
            pd.concat(
                [results_df[keys], abnormal.rename("abnormal")],
                axis=1,
            )
            .groupby(keys, dropna=False)["abnormal"]
            .agg(
                valid_n=lambda values: values.notna().sum(),
                abnormal_any=lambda values: values.any(skipna=True),
            )
            .reset_index()
        )

        # For all-missing endpoint groups, keep the call as NA rather than
        # silently assigning a normal call.
        grouped.loc[grouped["valid_n"] == 0, "abnormal_any"] = pd.NA
        grouped["abnormal_any"] = grouped["abnormal_any"].astype("boolean")

        grouped["feature"] = feature
        grouped["threshold"] = float(threshold)
        grouped[call_col] = grouped["abnormal_any"].map(
            {True: "abnormal", False: "normal"}
        )

        call_rows.append(grouped)

    if not call_rows:
        return pd.DataFrame(
            columns=keys
            + ["valid_n", "abnormal_any", "feature", "threshold", call_col]
        )

    return pd.concat(call_rows, ignore_index=True)


def summarize_calls(calls_df, call_col):
    """Count abnormal and normal treatment-level calls by feature."""
    valid = calls_df[calls_df["valid_n"] > 0].copy()

    if valid.empty:
        return pd.DataFrame(columns=["feature", "abnormal", "normal"])

    counts = (
        valid.groupby(["feature", call_col], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for label in ("abnormal", "normal"):
        if label not in counts.columns:
            counts[label] = 0

    return counts[["feature", "abnormal", "normal"]]


# -----------------------------------------------------------------------------
# Concordance metrics
# -----------------------------------------------------------------------------
def compare_real_and_synthetic_calls(real_calls, synthetic_calls):
    """Align real and GanCtrl calls at the compound/time/dose/feature level."""
    keys = ["compound", "time", "dose", "feature"]

    real = real_calls[
        real_calls["valid_n"] > 0
    ][keys + ["valid_n", "abnormal_any"]].rename(
        columns={
            "valid_n": "valid_n_real",
            "abnormal_any": "abnormal_real",
        }
    )

    synthetic = synthetic_calls[
        keys + ["valid_n", "abnormal_any", "threshold"]
    ].rename(
        columns={
            "valid_n": "valid_n_synthetic",
            "abnormal_any": "abnormal_synthetic",
        }
    )

    # Match the original workflow: evaluate groups that have a real-control
    # call and a corresponding synthetic-control group.
    merged = real.merge(synthetic, on=keys, how="inner")

    if FILL_MISSING_SYNTHETIC_AS_NORMAL:
        merged["synthetic_call_was_missing"] = merged[
            "abnormal_synthetic"
        ].isna()
        merged["abnormal_synthetic_eval"] = merged[
            "abnormal_synthetic"
        ].fillna(False)
    else:
        merged = merged[merged["valid_n_synthetic"] > 0].copy()
        merged["synthetic_call_was_missing"] = False
        merged["abnormal_synthetic_eval"] = merged[
            "abnormal_synthetic"
        ]

    merged["abnormal_real"] = merged["abnormal_real"].astype(bool)
    merged["abnormal_synthetic_eval"] = merged[
        "abnormal_synthetic_eval"
    ].astype(bool)
    merged["concordant"] = (
        merged["abnormal_real"] == merged["abnormal_synthetic_eval"]
    )

    return merged


def safe_divide(numerator, denominator):
    """Return NaN when a metric denominator is zero."""
    return np.nan if denominator == 0 else numerator / denominator


def calculate_confusion_metrics(merged_calls):
    """Calculate endpoint-level confusion counts and agreement metrics."""
    rows = []

    for feature, group in merged_calls.groupby("feature", dropna=False):
        real_abnormal = group["abnormal_real"]
        synthetic_abnormal = group["abnormal_synthetic_eval"]

        tp = int((real_abnormal & synthetic_abnormal).sum())
        tn = int((~real_abnormal & ~synthetic_abnormal).sum())
        fp = int((~real_abnormal & synthetic_abnormal).sum())
        fn = int((real_abnormal & ~synthetic_abnormal).sum())

        total = tp + tn + fp + fn
        accuracy = safe_divide(tp + tn, total)
        sensitivity = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)

        if pd.isna(sensitivity) or pd.isna(specificity):
            balanced_accuracy = np.nan
        else:
            balanced_accuracy = (sensitivity + specificity) / 2

        rows.append(
            {
                "feature": feature,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "N": total,
                "missing_synthetic_calls_filled": int(
                    group["synthetic_call_was_missing"].sum()
                ),
                "accuracy": accuracy,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
            }
        )

    return pd.DataFrame(rows)


def build_focused_summary(confusion_df):
    """Return concordance metrics for the 14 focused clinical endpoints."""
    focused = confusion_df[
        confusion_df["feature"].isin(FOCUSED_ENDPOINTS)
    ].copy()

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
    ]

    focused[metric_cols] = focused[metric_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    focused[metric_cols] = (focused[metric_cols] * 100).round(1)

    # Preserve the predefined endpoint order for easier comparison across runs.
    focused["feature"] = pd.Categorical(
        focused["feature"], categories=FOCUSED_ENDPOINTS, ordered=True
    )
    focused = focused.sort_values("feature").reset_index(drop=True)
    focused["feature"] = focused["feature"].astype(str)

    return focused[
        [
            "feature",
            "TP",
            "TN",
            "FP",
            "FN",
            "N",
            "missing_synthetic_calls_filled",
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
        ]
    ]


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------
def main():
    """Run the GanCtrl endpoint-level concordance analysis."""
    control = pd.read_csv(TEST_CONTROL_FILE, low_memory=False)
    treatment = pd.read_csv(TEST_TREATMENT_FILE, low_memory=False)
    generated = pd.read_csv(GENERATED_CONTROL_FILE, low_memory=False)
    threshold_df = pd.read_csv(THRESHOLD_FILE)

    validate_required_columns(
        control,
        [COMPOUND_COL, TIME_COL, DOSE_COL, ID_COL],
        "Test-control data",
    )
    validate_required_columns(
        treatment,
        [COMPOUND_COL, TIME_COL, DOSE_COL, ID_COL],
        "Test-treatment data",
    )
    validate_required_columns(
        generated,
        [COMPOUND_COL, GENERATED_TIME_COL],
        "Generated-control data",
    )

    # Restrict real data to metadata plus clinical-pathology features.
    real = pd.concat([control, treatment], ignore_index=True, sort=False)
    real = select_analysis_columns(real)

    generated = round_generated_values(generated)
    feature_cols = prepare_feature_columns(real, generated)

    real = make_features_numeric(real, feature_cols)
    generated = make_features_numeric(generated, feature_cols)

    threshold_lookup = load_thresholds(threshold_df)

    # Animal-level z-scores using real and synthetic controls.
    real_zscores = calculate_real_control_zscores(real, feature_cols)
    synthetic_zscores = calculate_synthetic_control_zscores(
        real, generated, feature_cols
    )

    # Treatment-level endpoint calls.
    real_calls = build_feature_calls(
        real_zscores,
        threshold_lookup=None,
        call_col="real_call",
    )
    synthetic_calls = build_feature_calls(
        synthetic_zscores,
        threshold_lookup=threshold_lookup,
        call_col="synthetic_call",
    )

    real_feature_status = summarize_calls(real_calls, "real_call")
    synthetic_feature_status = summarize_calls(
        synthetic_calls, "synthetic_call"
    )

    # Endpoint-level agreement between real- and synthetic-control calls.
    merged_calls = compare_real_and_synthetic_calls(
        real_calls, synthetic_calls
    )
    concordance = calculate_confusion_metrics(merged_calls)
    focused_summary = build_focused_summary(concordance)

    # Save analysis outputs.
    real_zscores.to_csv("real_control_zscores.csv", index=False)
    synthetic_zscores.to_csv("ganctrl_control_zscores.csv", index=False)
    real_calls.to_csv("real_feature_calls.csv", index=False)
    synthetic_calls.to_csv("ganctrl_feature_calls.csv", index=False)
    real_feature_status.to_csv("real_feature_status.csv", index=False)
    synthetic_feature_status.to_csv("ganctrl_feature_status.csv", index=False)
    merged_calls.to_csv("ganctrl_merged_calls.csv", index=False)
    concordance.to_csv("ganctrl_concordance.csv", index=False)
    focused_summary.to_csv("ganctrl_focused_concordance.csv", index=False)

    print("GanCtrl concordance analysis complete.")
    print(
        f"Evaluated {concordance['feature'].nunique()} endpoints; "
        f"{focused_summary.shape[0]} focused endpoints were available."
    )
    print("\nFocused endpoint concordance (%):")
    print(focused_summary.to_string(index=False))


if __name__ == "__main__":
    main()
