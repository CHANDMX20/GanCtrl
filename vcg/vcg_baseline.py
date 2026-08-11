"""Construct and evaluate virtual control groups (VCGs).

VCGs are sampled from training-set historical controls matched to each test
compound/time group by sacrifice time, vehicle, and laboratory. Controls from
the target compound are excluded to prevent information leakage.

Each VCG contains the same number of animals as the corresponding real
concurrent control group and is independently sampled 100 times.
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------
TEST_CONTROL_FILE = "repeat_test_control_2d.csv"
TRAIN_CONTROL_FILE = "repeat_train_control_2d.csv"
TEST_TREATMENT_FILE = "repeat_test_treatment_2d.csv"
METADATA_FILE = "tgp_data.csv"


# -----------------------------------------------------------------------------
# Analysis settings
# -----------------------------------------------------------------------------
METADATA_COLS = 11
FEATURE_START_COL = 1368

N_VCG_REPS = 100
SEED = 123
MIN_CTRL_N = 2
REPLACE_IF_NEEDED = False
Z_THRESHOLD = 2.0

COMPOUND_COL = "COMPOUND_NAME"
TIME_COL = "SACRIFICE_PERIOD"
DOSE_COL = "DOSE_LEVEL"
ID_COL = "INDIVIDUAL_ID"
VEHICLE_COL = "Vehicle"
LAB_COL = "Lab"


# Compound-name harmonization between the metadata file and analysis tables.
COMPOUND_NAME_MAP = {
    "2-acetamidofluorene": "acetamidofluorene",
    "chlorpheniramine maleate": "chlorpheniramine",
    "clomipramine hydrochloride": "clomipramine",
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
    "iproniazid phosphate salt": "iproniazid",
}


def select_analysis_columns(df):
    """Keep the first metadata columns and the clinical-pathology features."""
    return pd.concat(
        [df.iloc[:, :METADATA_COLS], df.iloc[:, FEATURE_START_COL:]],
        axis=1,
    )


def prepare_metadata(metadata):
    """Standardize metadata column names and compound names."""
    metadata = metadata.rename(
        columns={
            "Test Facility (in vivo animal treatment)": LAB_COL,
            "Compound name (E)": COMPOUND_COL,
        }
    ).copy()

    # Match the original workflow by removing incomplete metadata rows.
    metadata = metadata.dropna().copy()

    compound_names = metadata[COMPOUND_COL].astype("string").str.strip()
    metadata[COMPOUND_COL] = compound_names.str.replace(
        r"^(\S)",
        lambda match: match.group(1).lower(),
        regex=True,
    )
    metadata[COMPOUND_COL] = metadata[COMPOUND_COL].replace(COMPOUND_NAME_MAP)

    metadata_lookup = metadata[[COMPOUND_COL, VEHICLE_COL, LAB_COL]].drop_duplicates()

    # The downstream merge expects one vehicle/lab combination per compound.
    duplicated = metadata_lookup.duplicated(COMPOUND_COL, keep=False)
    if duplicated.any():
        duplicate_compounds = sorted(
            metadata_lookup.loc[duplicated, COMPOUND_COL].astype(str).unique()
        )
        raise ValueError(
            "Metadata contains multiple vehicle/lab combinations for the same "
            f"compound: {duplicate_compounds}"
        )

    return metadata_lookup


def add_study_metadata(df, metadata_lookup):
    """Add vehicle and laboratory metadata and place them after column 11."""
    merged = df.merge(
        metadata_lookup,
        on=COMPOUND_COL,
        how="left",
        validate="m:1",
    )

    cols = merged.columns.tolist()
    for col in (VEHICLE_COL, LAB_COL):
        if col in cols:
            cols.remove(col)

    insert_at = min(METADATA_COLS, len(cols))
    ordered_cols = (
        cols[:insert_at]
        + [VEHICLE_COL, LAB_COL]
        + cols[insert_at:]
    )

    # Match the original workflow by dropping rows with missing values.
    return merged[ordered_cols].dropna().copy()


def same_value(series, value):
    """NA-aware equality comparison."""
    if pd.isna(value):
        return series.isna()
    return series.eq(value)


def different_value(series, value):
    """NA-aware inequality comparison."""
    if pd.isna(value):
        return series.notna()
    return series.ne(value)


def first_non_na(series):
    """Return the first non-missing value, or NaN if none is available."""
    values = series.dropna()
    return values.iloc[0] if not values.empty else np.nan


def calculate_real_control_zscores(combined, feature_cols):
    """Calculate treatment-animal z-scores using real concurrent controls."""
    rows = []

    for (compound, time), group in combined.groupby(
        [COMPOUND_COL, TIME_COL],
        dropna=False,
        sort=False,
    ):
        controls = group[group[DOSE_COL] == "Control"]
        if controls.shape[0] < MIN_CTRL_N:
            continue

        treatments = group[group[DOSE_COL] == "High"]
        if treatments.empty:
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
                x = row[feature]
                mean = control_means[feature]
                sd = control_sds[feature]

                record[f"{feature}_z"] = (
                    np.nan
                    if pd.isna(x) or pd.isna(sd) or sd == 0
                    else (x - mean) / sd
                )

            rows.append(record)

    return pd.DataFrame(rows)


def build_feature_calls(results_df, keys, call_col):
    """Collapse animal-level z-scores into treatment-level abnormal/normal calls."""
    z_cols = [col for col in results_df.columns if col.endswith("_z")]
    call_rows = []

    for z_col in z_cols:
        z_scores = results_df[z_col]

        # Preserve missing z-scores as missing rather than treating them as normal.
        abnormal = z_scores.abs().gt(Z_THRESHOLD)
        abnormal = abnormal.where(z_scores.notna(), pd.NA).astype("boolean")

        grouped = (
            pd.concat(
                [results_df[keys], abnormal.rename("abnormal")],
                axis=1,
            )
            .groupby(keys, dropna=False)["abnormal"]
            .agg(
                valid_n=lambda x: x.notna().sum(),
                abnormal_any=lambda x: x.any(skipna=True),
            )
            .reset_index()
        )

        # Exclude feature/group combinations with no usable z-scores.
        grouped = grouped[grouped["valid_n"] > 0].copy()
        grouped["feature"] = z_col.removesuffix("_z")
        grouped[call_col] = grouped["abnormal_any"].map(
            {True: "abnormal", False: "normal"}
        )
        call_rows.append(grouped)

    if not call_rows:
        return pd.DataFrame(
            columns=keys + ["valid_n", "abnormal_any", "feature", call_col]
        )

    return pd.concat(call_rows, ignore_index=True)


def summarize_feature_calls(calls_df, call_col, group_cols):
    """Count abnormal and normal calls within the requested groups."""
    if calls_df.empty:
        return pd.DataFrame(columns=group_cols + ["abnormal", "normal"])

    counts = (
        calls_df.groupby(group_cols + [call_col], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for label in ("abnormal", "normal"):
        if label not in counts.columns:
            counts[label] = 0

    return counts[group_cols + ["abnormal", "normal"]]


def generate_vcg_zscores(
    historical_controls,
    real_controls,
    treatments,
    feature_cols,
):
    """Generate same-time, same-vehicle, same-lab VCGs and treatment z-scores."""
    rng = np.random.default_rng(SEED)

    historical_controls = historical_controls.copy()
    real_controls = real_controls.copy()
    treatments = treatments.copy()

    historical_controls = historical_controls[
        historical_controls[DOSE_COL] == "Control"
    ].copy()
    real_controls = real_controls[real_controls[DOSE_COL] == "Control"].copy()
    treatments = treatments[treatments[DOSE_COL] == "High"].copy()

    for df in (historical_controls, real_controls, treatments):
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    vcg_rows = []
    group_log_rows = []
    sample_log_rows = []

    for iteration in range(1, N_VCG_REPS + 1):
        for (compound, time), treatment_group in treatments.groupby(
            [COMPOUND_COL, TIME_COL],
            dropna=False,
            sort=False,
        ):
            # Real test controls define the target VCG sample size only.
            real_control_mask = (
                same_value(real_controls[COMPOUND_COL], compound)
                & same_value(real_controls[TIME_COL], time)
            )
            real_control_group = real_controls.loc[real_control_mask].copy()
            n_real_controls = real_control_group.shape[0]

            # Prefer treatment metadata; use the real control metadata as fallback.
            target_vehicle = first_non_na(treatment_group[VEHICLE_COL])
            if pd.isna(target_vehicle):
                target_vehicle = first_non_na(real_control_group[VEHICLE_COL])

            target_lab = first_non_na(treatment_group[LAB_COL])
            if pd.isna(target_lab):
                target_lab = first_non_na(real_control_group[LAB_COL])

            # Historical-control pool: same time, vehicle, and lab; different compound.
            pool_mask = (
                same_value(historical_controls[TIME_COL], time)
                & same_value(historical_controls[VEHICLE_COL], target_vehicle)
                & same_value(historical_controls[LAB_COL], target_lab)
                & different_value(historical_controls[COMPOUND_COL], compound)
            )
            pool = historical_controls.loc[pool_mask].copy()

            if not pd.isna(compound) and not pool.empty:
                assert not pool[COMPOUND_COL].eq(compound).any(), (
                    f"Leakage detected: VCG pool contains controls from {compound}."
                )

            # Sample a VCG with the same number of animals as the real control group.
            if n_real_controls < MIN_CTRL_N:
                status = "real_ccg_lt_2"
                vcg_sample = pool.iloc[0:0].copy()
                replace = False
            elif pool.shape[0] < MIN_CTRL_N:
                status = "hcd_pool_lt_2"
                vcg_sample = pool.iloc[0:0].copy()
                replace = False
            elif pool.shape[0] < n_real_controls and not REPLACE_IF_NEEDED:
                status = "hcd_pool_lt_real_ccg"
                vcg_sample = pool.iloc[0:0].copy()
                replace = False
            else:
                replace = pool.shape[0] < n_real_controls
                vcg_sample = pool.sample(
                    n=n_real_controls,
                    replace=replace,
                    random_state=int(rng.integers(0, 2**32 - 1)),
                )
                status = "ok_with_replacement" if replace else "ok"

                if not pd.isna(compound):
                    assert not vcg_sample[COMPOUND_COL].eq(compound).any(), (
                        "Leakage detected: sampled VCG contains controls from "
                        f"{compound}."
                    )

            group_log_rows.append(
                {
                    "vcg_iter": iteration,
                    "compound": compound,
                    "time": time,
                    "target_vehicle": target_vehicle,
                    "target_lab": target_lab,
                    "n_real_ccg_test": n_real_controls,
                    "n_hcd_pool": pool.shape[0],
                    "n_vcg_sampled": vcg_sample.shape[0],
                    "replace": replace,
                    "vcg_status": status,
                }
            )

            if not vcg_sample.empty:
                sample_log = pd.DataFrame(
                    {
                        "vcg_iter": iteration,
                        "target_compound": compound,
                        "target_time": time,
                        "target_vehicle": target_vehicle,
                        "target_lab": target_lab,
                        "sampled_compound": vcg_sample[COMPOUND_COL].to_numpy(),
                        "sampled_time": vcg_sample[TIME_COL].to_numpy(),
                        "sampled_vehicle": vcg_sample[VEHICLE_COL].to_numpy(),
                        "sampled_lab": vcg_sample[LAB_COL].to_numpy(),
                    }
                )

                if ID_COL in vcg_sample.columns:
                    sample_log["sampled_INDIVIDUAL_ID"] = vcg_sample[
                        ID_COL
                    ].to_numpy()

                sample_log["source_is_target_compound"] = sample_log[
                    "sampled_compound"
                ].eq(compound)
                sample_log_rows.append(sample_log)

            if vcg_sample.shape[0] >= MIN_CTRL_N:
                vcg_means = vcg_sample[feature_cols].mean()
                vcg_sds = vcg_sample[feature_cols].std(ddof=1)
            else:
                vcg_means = None
                vcg_sds = None

            # Score each high-dose animal relative to the sampled VCG.
            for _, row in treatment_group.iterrows():
                record = {
                    "vcg_iter": iteration,
                    "compound": compound,
                    "time": time,
                    "dose": row[DOSE_COL],
                    ID_COL: row[ID_COL] if ID_COL in treatments.columns else np.nan,
                    "n_real_ccg_test": n_real_controls,
                    "n_hcd_pool": pool.shape[0],
                    "n_vcg_sampled": vcg_sample.shape[0],
                    "vcg_status": status,
                }

                for feature in feature_cols:
                    if vcg_means is None or vcg_sds is None:
                        record[f"{feature}_z"] = np.nan
                        continue

                    x = row[feature]
                    mean = vcg_means[feature]
                    sd = vcg_sds[feature]

                    record[f"{feature}_z"] = (
                        np.nan
                        if pd.isna(x) or pd.isna(sd) or sd == 0
                        else (x - mean) / sd
                    )

                vcg_rows.append(record)

    results_vcg_df = pd.DataFrame(vcg_rows)
    vcg_group_log_df = pd.DataFrame(group_log_rows)
    vcg_sample_log_df = (
        pd.concat(sample_log_rows, ignore_index=True)
        if sample_log_rows
        else pd.DataFrame()
    )

    return results_vcg_df, vcg_group_log_df, vcg_sample_log_df


def summarize_concordance(merged_calls, group_cols):
    """Calculate confusion counts and concordance/accuracy."""
    rows = []

    for group_values, group in merged_calls.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        real_abnormal = group["real_abnormal"]
        vcg_abnormal = group["vcg_abnormal"]

        tp = int((real_abnormal & vcg_abnormal).sum())
        tn = int((~real_abnormal & ~vcg_abnormal).sum())
        fp = int((~real_abnormal & vcg_abnormal).sum())
        fn = int((real_abnormal & ~vcg_abnormal).sum())

        total = tp + tn + fp + fn
        concordant = tp + tn
        accuracy = np.nan if total == 0 else concordant / total

        record = dict(zip(group_cols, group_values))
        record.update(
            {
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "total": total,
                "n_concordant": concordant,
                "accuracy": accuracy,
                "concordance": accuracy,
            }
        )
        rows.append(record)

    return pd.DataFrame(rows)


def evaluate_vcg_concordance(real_calls_df, vcg_calls_df):
    """Compare each VCG iteration-level call with the corresponding real call."""
    merge_keys = ["compound", "time", "dose", "feature"]

    real_calls = real_calls_df[
        merge_keys + ["real_call"]
    ].copy()
    vcg_calls = vcg_calls_df[
        ["vcg_iter"] + merge_keys + ["vcg_call"]
    ].copy()

    merged = vcg_calls.merge(real_calls, on=merge_keys, how="inner")
    merged["real_abnormal"] = merged["real_call"].eq("abnormal")
    merged["vcg_abnormal"] = merged["vcg_call"].eq("abnormal")
    merged["concordant"] = merged["real_abnormal"] == merged["vcg_abnormal"]

    pooled = summarize_concordance(merged, ["feature"])
    by_iteration = summarize_concordance(merged, ["vcg_iter", "feature"])

    if by_iteration.empty:
        summary = pd.DataFrame(
            columns=[
                "feature",
                "mean_accuracy",
                "sd_accuracy",
                "min_accuracy",
                "max_accuracy",
                "n_vcg_iters",
                "mean_total_compared",
            ]
        )
    else:
        summary = (
            by_iteration.groupby("feature", dropna=False)
            .agg(
                mean_accuracy=("accuracy", "mean"),
                sd_accuracy=("accuracy", "std"),
                min_accuracy=("accuracy", "min"),
                max_accuracy=("accuracy", "max"),
                n_vcg_iters=("vcg_iter", "nunique"),
                mean_total_compared=("total", "mean"),
            )
            .reset_index()
        )

    return merged, pooled, by_iteration, summary


def main():
    """Run the complete VCG construction and concordance workflow."""
    control_test = pd.read_csv(TEST_CONTROL_FILE)
    control_train = pd.read_csv(TRAIN_CONTROL_FILE)
    treatment_test = pd.read_csv(TEST_TREATMENT_FILE)
    metadata = pd.read_csv(METADATA_FILE)

    # Keep the analysis columns used in the original workflow.
    test = pd.concat(
        [control_test, treatment_test],
        ignore_index=True,
        sort=False,
    )
    test = select_analysis_columns(test)
    control_train = select_analysis_columns(control_train)

    control = test[test[DOSE_COL] == "Control"].copy()
    treatment = test[test[DOSE_COL] == "High"].copy()

    common_cols = control.columns.intersection(treatment.columns, sort=False)
    combined = pd.concat(
        [control[common_cols], treatment[common_cols]],
        ignore_index=True,
    )

    metadata_lookup = prepare_metadata(metadata)
    control_new = add_study_metadata(control, metadata_lookup)
    treatment_new = add_study_metadata(treatment, metadata_lookup)
    control_train_new = add_study_metadata(control_train, metadata_lookup)

    # Keep only test compounds represented in the metadata-matched control data.
    valid_compounds = control_new[COMPOUND_COL].dropna().unique()
    combined = combined[combined[COMPOUND_COL].isin(valid_compounds)].copy()

    feature_cols = combined.columns[METADATA_COLS:].tolist()
    combined[feature_cols] = combined[feature_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Real concurrent-control calls
    # -------------------------------------------------------------------------
    results_df = calculate_real_control_zscores(combined, feature_cols)
    real_calls_df = build_feature_calls(
        results_df,
        keys=["compound", "time", "dose"],
        call_col="real_call",
    )
    feature_status_df = summarize_feature_calls(
        real_calls_df,
        call_col="real_call",
        group_cols=["feature"],
    )

    # -------------------------------------------------------------------------
    # VCG construction and calls
    # -------------------------------------------------------------------------
    results_vcg_df, vcg_group_log_df, vcg_sample_log_df = generate_vcg_zscores(
        historical_controls=control_train_new,
        real_controls=control_new,
        treatments=treatment_new,
        feature_cols=feature_cols,
    )

    vcg_calls_df = build_feature_calls(
        results_vcg_df,
        keys=["vcg_iter", "compound", "time", "dose"],
        call_col="vcg_call",
    )
    feature_status_vcg_df = summarize_feature_calls(
        vcg_calls_df,
        call_col="vcg_call",
        group_cols=["vcg_iter", "feature"],
    )

    # -------------------------------------------------------------------------
    # Concordance between VCG-based and real-control-based calls
    # -------------------------------------------------------------------------
    (
        merged_vcg_calls_df,
        confusion_vcg_df,
        iteration_accuracy_vcg_df,
        accuracy_summary_vcg_df,
    ) = evaluate_vcg_concordance(real_calls_df, vcg_calls_df)

    # Save outputs with generic filenames in the current working directory.
    results_df.to_csv("real_control_zscores.csv", index=False)
    real_calls_df.to_csv("real_control_feature_calls.csv", index=False)
    feature_status_df.to_csv("real_control_feature_status.csv", index=False)

    results_vcg_df.to_csv("vcg_zscores.csv", index=False)
    vcg_calls_df.to_csv("vcg_feature_calls.csv", index=False)
    feature_status_vcg_df.to_csv("vcg_feature_status_by_iteration.csv", index=False)
    vcg_group_log_df.to_csv("vcg_group_log.csv", index=False)
    vcg_sample_log_df.to_csv("vcg_sample_log.csv", index=False)

    merged_vcg_calls_df.to_csv("vcg_merged_calls.csv", index=False)
    confusion_vcg_df.to_csv("vcg_concordance.csv", index=False)
    iteration_accuracy_vcg_df.to_csv("vcg_accuracy_by_iteration.csv", index=False)
    accuracy_summary_vcg_df.to_csv("vcg_accuracy_summary.csv", index=False)

    print("VCG analysis complete.")
    print("\nPooled concordance by feature:")
    print(confusion_vcg_df)
    print("\nMean accuracy across VCG iterations:")
    print(accuracy_summary_vcg_df)


if __name__ == "__main__":
    main()
