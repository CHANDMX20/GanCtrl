# =============================================================================
# GanCtrl vs. Real-Control Biomarker Distributions
# =============================================================================
#
# This script compares the distributions of selected liver and kidney clinical
# pathology biomarkers between real test-set controls and GanCtrl-generated
# controls. High-dose synthetic controls are retained, replicate rows are
# collapsed to per-animal/per-generated-copy means, and overlaid histograms are
# written as a 5 x 3 TIFF figure.
#
# Required input files (expected in the working directory):
#   - generated_predictions_892645_ControlGenerator_test.csv
#   - generated_predictions_251875_ControlGenerator_test.csv
#   - repeat_test_control_2d.csv
#
# Output:
#   - hist_counts_liver_kidney_combined_test.tif
# =============================================================================

library(dplyr)


# =============================================================================
# Configuration
# =============================================================================

GENERATED_LIVER_FILE <- "generated_predictions_892645_ControlGenerator_test.csv"
GENERATED_KIDNEY_FILE <- "generated_predictions_251875_ControlGenerator_test.csv"
REAL_CONTROL_FILE <- "repeat_test_control_2d.csv"
OUTPUT_TIFF <- "hist_counts_liver_kidney_combined_test.tif"

N_BINS <- 25
N_COL <- 5
N_ROW <- 3
TIFF_WIDTH_PX <- 15000
TIFF_HEIGHT_PX <- 9000
TIFF_RESOLUTION <- 600

# TG-GATEs metadata occupy the first 11 columns. Clinical pathology features
# begin at column 1369 in the original wide tables.
N_METADATA_COLS <- 11
FEATURE_START_COL <- 1369

LIVER_FEATURES <- c(
  "ALP(IU/L)",
  "ALT(IU/L)",
  "AST(IU/L)",
  "GTP(IU/L)",
  "LDH(IU/L)",
  "TBIL(mg/dL)",
  "DBIL(mg/dL)"
)

KIDNEY_FEATURES <- c(
  "BUN(mg/dL)",
  "CRE(mg/dL)",
  "Ca(mg/dL)",
  "Cl(meq/L)",
  "IP(mg/dL)",
  "K(meq/L)",
  "Na(meq/L)"
)

ALL_FEATURES <- c(LIVER_FEATURES, KIDNEY_FEATURES)


# =============================================================================
# Utility functions
# =============================================================================

as_numeric_safely <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

validate_columns <- function(df, required, data_name) {
  missing <- setdiff(required, names(df))

  if (length(missing) > 0) {
    stop(
      sprintf(
        "%s is missing required column(s): %s",
        data_name,
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }
}

normalize_dose <- function(x) {
  tolower(trimws(as.character(x)))
}

is_high_label <- function(x) {
  normalize_dose(x) %in% c(
    "high", "hi", "top", "max", "highest", "hd", "h"
  )
}

find_dose_column <- function(df) {
  candidates <- c(
    "DOSE_LEVEL",
    "Dose_Level",
    "DoseLevel",
    "DOSE_GROUP",
    "DoseGroup",
    "DOSE"
  )

  matches <- intersect(candidates, names(df))

  if (length(matches) == 0) {
    return(NA_character_)
  }

  matches[1]
}

filter_high_dose <- function(df) {
  dose_col <- find_dose_column(df)

  if (is.na(dose_col)) {
    warning(
      "No dose column found; high-dose filtering was skipped for this data frame.",
      call. = FALSE
    )
    return(df)
  }

  # Numeric DOSE fields are handled by retaining the maximum dose within the
  # available compound/time/individual grouping. Categorical dose fields are
  # matched against common labels for the high-dose group.
  if (dose_col == "DOSE") {
    group_keys <- intersect(
      c("COMPOUND_NAME", "targetTime", "INDIVIDUAL_ID"),
      names(df)
    )

    if (length(group_keys) == 0) {
      max_dose <- suppressWarnings(max(df[[dose_col]], na.rm = TRUE))

      if (!is.finite(max_dose)) {
        return(df[0, , drop = FALSE])
      }

      return(df %>% filter(.data[[dose_col]] == max_dose))
    }

    return(
      df %>%
        group_by(across(all_of(group_keys))) %>%
        filter(.data[[dose_col]] == max(.data[[dose_col]], na.rm = TRUE)) %>%
        ungroup()
    )
  }

  df %>% filter(is_high_label(.data[[dose_col]]))
}

mean_numeric_na <- function(x) {
  x <- as_numeric_safely(x)

  if (all(is.na(x))) {
    return(NA_real_)
  }

  mean(x, na.rm = TRUE)
}

finite_numeric <- function(x) {
  x <- as_numeric_safely(x)
  x[is.finite(x)]
}

is_integerish <- function(x, tolerance = 1e-9) {
  x <- x[is.finite(x)]
  length(x) > 0 && all(abs(x - round(x)) < tolerance)
}


# =============================================================================
# Data preparation
# =============================================================================

prepare_real_controls <- function(real_df, features) {
  required_metadata <- c(
    "COMPOUND_NAME",
    "SACRIFICE_PERIOD",
    "INDIVIDUAL_ID"
  )
  validate_columns(real_df, required_metadata, "Real-control data")

  available_features <- intersect(features, names(real_df))

  if (length(available_features) == 0) {
    stop("No requested biomarker features were found in the real-control data.")
  }

  real_df %>%
    select(any_of(c(required_metadata, available_features))) %>%
    group_by(COMPOUND_NAME, SACRIFICE_PERIOD, INDIVIDUAL_ID) %>%
    summarise(
      across(all_of(available_features), mean_numeric_na),
      .groups = "drop"
    )
}

prepare_generated_controls <- function(
  liver_df,
  kidney_df,
  liver_features,
  kidney_features
) {
  required_metadata <- c("COMPOUND_NAME", "targetTime", "targetBioCopy")

  validate_columns(liver_df, required_metadata, "Generated liver-control data")
  validate_columns(kidney_df, required_metadata, "Generated kidney-control data")

  liver_df <- filter_high_dose(liver_df)
  kidney_df <- filter_high_dose(kidney_df)

  liver_available <- intersect(liver_features, names(liver_df))
  kidney_available <- intersect(kidney_features, names(kidney_df))

  if (length(liver_available) == 0) {
    stop("No liver biomarker features were found in the generated liver data.")
  }

  if (length(kidney_available) == 0) {
    stop("No kidney biomarker features were found in the generated kidney data.")
  }

  generated_all <- bind_rows(
    liver_df %>%
      select(any_of(c(required_metadata, liver_available))),
    kidney_df %>%
      select(any_of(c(required_metadata, kidney_available)))
  )

  available_features <- intersect(
    c(liver_features, kidney_features),
    names(generated_all)
  )

  generated_all %>%
    group_by(COMPOUND_NAME, targetTime, targetBioCopy) %>%
    summarise(
      across(all_of(available_features), mean_numeric_na),
      .groups = "drop"
    )
}


# =============================================================================
# Histogram plotting
# =============================================================================

plot_histogram_panel <- function(
  real_df,
  generated_df,
  feature,
  bins,
  real_fill,
  real_line,
  generated_fill,
  generated_line
) {
  real_values <- if (feature %in% names(real_df)) {
    finite_numeric(real_df[[feature]])
  } else {
    numeric(0)
  }

  generated_values <- if (feature %in% names(generated_df)) {
    finite_numeric(generated_df[[feature]])
  } else {
    numeric(0)
  }

  if (length(real_values) + length(generated_values) == 0) {
    plot.new()
    box(lwd = 1.4)
    title(
      main = feature,
      cex.main = 1.65,
      font.main = 2,
      family = "Arial"
    )
    return(invisible(NULL))
  }

  all_values <- c(real_values, generated_values)

  if (is_integerish(all_values)) {
    breaks <- seq(
      floor(min(all_values)) - 0.5,
      ceiling(max(all_values)) + 0.5,
      by = 1
    )
  } else {
    value_range <- range(all_values, finite = TRUE)

    if (!all(is.finite(value_range)) || diff(value_range) == 0) {
      value_range <- value_range + c(-1, 1) * 1e-6
    }

    breaks <- seq(value_range[1], value_range[2], length.out = bins + 1)
  }

  # Build histogram objects on the same breaks. If one distribution has no
  # finite observations, use an all-zero histogram so the other distribution
  # can still be plotted normally.
  template_values <- if (length(real_values) > 0) real_values else generated_values
  template_hist <- hist(template_values, breaks = breaks, plot = FALSE)

  if (length(real_values) > 0) {
    real_hist <- hist(real_values, breaks = breaks, plot = FALSE)
  } else {
    real_hist <- template_hist
    real_hist$counts[] <- 0
    real_hist$density[] <- 0
  }

  if (length(generated_values) > 0) {
    generated_hist <- hist(generated_values, breaks = breaks, plot = FALSE)
  } else {
    generated_hist <- template_hist
    generated_hist$counts[] <- 0
    generated_hist$density[] <- 0
  }

  y_max <- max(c(real_hist$counts, generated_hist$counts), na.rm = TRUE)
  if (!is.finite(y_max) || y_max == 0) {
    y_max <- 1
  }

  plot(
    real_hist,
    freq = TRUE,
    xlim = range(breaks),
    ylim = c(0, y_max * 1.25),
    col = real_fill,
    border = real_line,
    lwd = 1.5,
    xlab = "",
    ylab = "Count",
    main = feature,
    family = "Arial",
    cex.axis = 1.40,
    cex.lab = 1.50,
    cex.main = 1.65,
    font.lab = 2,
    font.main = 2
  )

  plot(
    generated_hist,
    freq = TRUE,
    add = TRUE,
    col = generated_fill,
    border = generated_line,
    lwd = 1.5
  )

  box(lwd = 1.4)

  legend(
    "topright",
    legend = c(
      sprintf("Real Control (n=%d)", length(real_values)),
      sprintf("GanCtrl (n=%d)", length(generated_values))
    ),
    fill = c(real_fill, generated_fill),
    border = c(real_line, generated_line),
    bty = "n",
    cex = 1.20,
    text.font = 1,
    inset = 0.02,
    x.intersp = 0.7,
    y.intersp = 0.8
  )

  invisible(NULL)
}

plot_histogram_grid <- function(
  real_df,
  generated_df,
  features,
  output_file,
  bins = 25,
  n_col = 5,
  n_row = 3,
  width_px = 15000,
  height_px = 9000,
  resolution = 600
) {
  if (length(features) > n_col * n_row) {
    stop("Too many features for the requested histogram grid.")
  }

  # Colors retained from the original analysis figure.
  real_line <- "#1F78B4"
  real_fill <- rgb(31, 119, 180, alpha = 120, maxColorValue = 255)
  generated_line <- "#E31A1C"
  generated_fill <- rgb(227, 26, 28, alpha = 120, maxColorValue = 255)

  tiff(
    output_file,
    width = width_px,
    height = height_px,
    res = resolution,
    compression = "lzw"
  )

  old_par <- par(no.readonly = TRUE)

  on.exit({
    par(old_par)
    dev.off()
  }, add = TRUE)

  par(
    family = "Arial",
    mfrow = c(n_row, n_col),
    oma = c(0.5, 0.5, 0.5, 0.5),
    mar = c(4.5, 6.0, 3.2, 1.4),
    mgp = c(3.7, 0.9, 0),
    cex.axis = 1.40,
    cex.lab = 1.50,
    font.lab = 2,
    font.axis = 1,
    las = 1
  )

  for (feature in features) {
    plot_histogram_panel(
      real_df = real_df,
      generated_df = generated_df,
      feature = feature,
      bins = bins,
      real_fill = real_fill,
      real_line = real_line,
      generated_fill = generated_fill,
      generated_line = generated_line
    )
  }

  n_empty <- n_col * n_row - length(features)
  if (n_empty > 0) {
    for (i in seq_len(n_empty)) {
      plot.new()
    }
  }

  invisible(output_file)
}


# =============================================================================
# Main workflow
# =============================================================================

main <- function() {
  message("Reading input data...")

  generated_liver <- read.csv(
    GENERATED_LIVER_FILE,
    check.names = FALSE
  )

  generated_kidney <- read.csv(
    GENERATED_KIDNEY_FILE,
    check.names = FALSE
  )

  real <- read.csv(
    REAL_CONTROL_FILE,
    check.names = FALSE
  )

  if (ncol(real) < FEATURE_START_COL) {
    stop(
      sprintf(
        "Real-control data have %d columns; expected at least %d.",
        ncol(real),
        FEATURE_START_COL
      ),
      call. = FALSE
    )
  }

  # Retain the first 11 metadata columns and the clinical pathology block.
  real <- cbind(
    real[, seq_len(N_METADATA_COLS), drop = FALSE],
    real[, FEATURE_START_COL:ncol(real), drop = FALSE]
  )

  message("Preparing real-control biomarker data...")
  real_collapsed <- prepare_real_controls(real, ALL_FEATURES)

  message("Preparing GanCtrl biomarker data...")
  generated_collapsed <- prepare_generated_controls(
    generated_liver,
    generated_kidney,
    LIVER_FEATURES,
    KIDNEY_FEATURES
  )

  message("Creating histogram grid...")
  plot_histogram_grid(
    real_df = real_collapsed,
    generated_df = generated_collapsed,
    features = ALL_FEATURES,
    output_file = OUTPUT_TIFF,
    bins = N_BINS,
    n_col = N_COL,
    n_row = N_ROW,
    width_px = TIFF_WIDTH_PX,
    height_px = TIFF_HEIGHT_PX,
    resolution = TIFF_RESOLUTION
  )

  message(sprintf("Saved: %s", OUTPUT_TIFF))
}


if (sys.nframe() == 0) {
  main()
}
