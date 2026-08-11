# GanCtrl ratio-of-means comparison
#
# Compares treatment-to-control ratios for ALT, AST, BUN, and CRE using:
#   1. Real concurrent controls
#   2. GanCtrl synthetic controls
#
# The script generates a grouped bar plot for a selected compound and time point.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

# =============================================================================
# Configuration
# =============================================================================

CONTROL_FILE <- "repeat_test_control_2d.csv"
TREATMENT_FILE <- "repeat_test_treatment_2d.csv"
GENERATED_FILE <- "generated_predictions_merged_test.csv"

TARGET_COMPOUND <- "nitrosodiethylamine"
TARGET_TIME <- "8 day"

OUTPUT_FILE <- "nitrosodiethylamine_8day_ALT_AST_BUN_CRE_ratios.tif"

BAR_WIDTH <- 0.33
BOX_LWD <- 2.0
TITLE_CEX <- 1.25
AXIS_CEX <- 1.25
LABEL_CEX <- 1.25
LEGEND_CEX <- 1.10

# Set to NULL for automatic scaling, or use e.g. c(0, 8).
RATIO_YLIM <- NULL

# Colors for the two control comparisons.
COLORS <- c("#168AAD", "#F6C85F")
names(COLORS) <- c("Treatment / Real Control", "Treatment / GanCtrl")

# =============================================================================
# Helper functions
# =============================================================================

check_input_files <- function(paths) {
  missing <- paths[!file.exists(paths)]

  if (length(missing) > 0) {
    stop(
      "Missing input file(s):\n",
      paste0("  - ", missing, collapse = "\n")
    )
  }
}

keep_clinical_pathology_columns <- function(df) {
  if (ncol(df) < 1369) {
    stop(
      "Expected at least 1369 columns in the real-data input. ",
      "Check that the TG-GATEs file structure matches the original analysis."
    )
  }

  bind_cols(df[, 1:11], df[, 1369:ncol(df)])
}

round_if_present <- function(df, column, digits = 0, as_integer = FALSE) {
  if (!column %in% names(df)) {
    return(df)
  }

  values <- suppressWarnings(as.numeric(df[[column]]))
  values <- round(values, digits)

  if (as_integer) {
    values <- as.integer(values)
  }

  df[[column]] <- values
  df
}

harmonize_generated_values <- function(df) {
  df %>%
    round_if_present("TBIL(mg/dL)", 2) %>%
    round_if_present("RALB(g/dL)", 1) %>%
    round_if_present("AST(IU/L)", 0, as_integer = TRUE) %>%
    round_if_present("TP(g/dL)", 1) %>%
    round_if_present("DBIL(mg/dL)", 2) %>%
    round_if_present("BUN(mg/dL)", 0, as_integer = TRUE) %>%
    round_if_present("ALP(IU/L)", 0, as_integer = TRUE) %>%
    round_if_present("ALT(IU/L)", 0, as_integer = TRUE) %>%
    round_if_present("LDH(IU/L)", 0, as_integer = TRUE)
}

pick_column <- function(df, candidates) {
  actual_names <- names(df)
  lower_names <- tolower(actual_names)

  for (candidate in candidates) {
    idx <- which(lower_names == tolower(candidate))
    if (length(idx) > 0) {
      return(actual_names[idx[1]])
    }
  }

  stop("Could not find column: ", paste(candidates, collapse = " / "))
}

pick_feature <- function(df, preferred, fallback) {
  if (preferred %in% names(df)) {
    return(preferred)
  }

  if (fallback %in% names(df)) {
    return(fallback)
  }

  stop("Could not find feature column: ", preferred, " or ", fallback)
}

subset_compound_time <- function(df, compound_col, time_col, compound, time_value) {
  compound_values <- trimws(tolower(as.character(df[[compound_col]])))
  time_values <- trimws(tolower(as.character(df[[time_col]])))

  df[
    compound_values == trimws(tolower(compound)) &
      time_values == trimws(tolower(time_value)),
    ,
    drop = FALSE
  ]
}

numeric_values <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

mean_finite <- function(x) {
  x <- numeric_values(x)
  x <- x[is.finite(x)]

  if (length(x) == 0) {
    return(NA_real_)
  }

  mean(x)
}

ratio_of_means <- function(treatment_values, control_values) {
  treatment_mean <- mean_finite(treatment_values)
  control_mean <- mean_finite(control_values)

  if (
    !is.finite(treatment_mean) ||
      !is.finite(control_mean) ||
      control_mean == 0
  ) {
    return(NA_real_)
  }

  treatment_mean / control_mean
}

capitalize_first <- function(x) {
  if (!nzchar(x)) {
    return(x)
  }

  paste0(toupper(substr(x, 1, 1)), substr(x, 2, nchar(x)))
}

calculate_ratio_matrix <- function(treatment_df, control_df, generated_df) {
  treatment_compound_col <- pick_column(
    treatment_df,
    c("COMPOUND_NAME", "COMPOUND", "compound")
  )
  treatment_time_col <- pick_column(
    treatment_df,
    c("SACRIFICE_PERIOD", "TIME", "time", "PERIOD")
  )

  control_compound_col <- pick_column(
    control_df,
    c("COMPOUND_NAME", "COMPOUND", "compound")
  )
  control_time_col <- pick_column(
    control_df,
    c("SACRIFICE_PERIOD", "TIME", "time", "PERIOD")
  )

  generated_compound_col <- pick_column(
    generated_df,
    c("COMPOUND_NAME", "COMPOUND", "compound")
  )
  generated_time_col <- pick_column(
    generated_df,
    c("SACRIFICE_PERIOD", "targetTime", "TIME", "time", "PERIOD")
  )

  feature_specs <- list(
    ALT = c("ALT(IU/L)", "ALT"),
    AST = c("AST(IU/L)", "AST"),
    BUN = c("BUN(mg/dL)", "BUN"),
    CRE = c("CRE(mg/dL)", "CRE")
  )

  treatment_features <- lapply(
    feature_specs,
    function(x) pick_feature(treatment_df, x[1], x[2])
  )
  control_features <- lapply(
    feature_specs,
    function(x) pick_feature(control_df, x[1], x[2])
  )
  generated_features <- lapply(
    feature_specs,
    function(x) pick_feature(generated_df, x[1], x[2])
  )

  treatment_subset <- subset_compound_time(
    treatment_df,
    treatment_compound_col,
    treatment_time_col,
    TARGET_COMPOUND,
    TARGET_TIME
  )

  control_subset <- subset_compound_time(
    control_df,
    control_compound_col,
    control_time_col,
    TARGET_COMPOUND,
    TARGET_TIME
  )

  generated_subset <- subset_compound_time(
    generated_df,
    generated_compound_col,
    generated_time_col,
    TARGET_COMPOUND,
    TARGET_TIME
  )

  if (nrow(treatment_subset) == 0) {
    stop("No treatment rows found for ", TARGET_COMPOUND, " at ", TARGET_TIME, ".")
  }
  if (nrow(control_subset) == 0) {
    stop("No real-control rows found for ", TARGET_COMPOUND, " at ", TARGET_TIME, ".")
  }
  if (nrow(generated_subset) == 0) {
    stop("No GanCtrl rows found for ", TARGET_COMPOUND, " at ", TARGET_TIME, ".")
  }

  endpoints <- names(feature_specs)

  ratio_real <- vapply(
    endpoints,
    function(endpoint) {
      ratio_of_means(
        treatment_subset[[treatment_features[[endpoint]]]],
        control_subset[[control_features[[endpoint]]]]
      )
    },
    numeric(1)
  )

  ratio_ganctrl <- vapply(
    endpoints,
    function(endpoint) {
      ratio_of_means(
        treatment_subset[[treatment_features[[endpoint]]]],
        generated_subset[[generated_features[[endpoint]]]]
      )
    },
    numeric(1)
  )

  rbind(
    "Treatment / Real Control" = ratio_real,
    "Treatment / GanCtrl" = ratio_ganctrl
  )
}

plot_ratio_bars <- function(ratio_matrix, output_file) {
  y_lim <- RATIO_YLIM

  if (is.null(y_lim)) {
    ymax <- suppressWarnings(max(ratio_matrix, na.rm = TRUE))

    if (!is.finite(ymax) || ymax <= 0) {
      ymax <- 1
    }

    y_lim <- c(0, ymax * 1.25)
  }

  tiff(
    filename = output_file,
    width = 8.5,
    height = 7.0,
    units = "in",
    res = 600,
    compression = "lzw"
  )

  old_par <- par(no.readonly = TRUE)
  on.exit({
    par(old_par)
    dev.off()
  }, add = TRUE)

  par(
    mar = c(5.2, 5.0, 3.2, 1.2),
    oma = c(0, 0, 0, 0),
    family = "sans"
  )

  barplot(
    ratio_matrix,
    beside = TRUE,
    col = COLORS,
    border = NA,
    width = BAR_WIDTH,
    ylim = y_lim,
    ylab = "Ratio of means",
    xlab = "",
    names.arg = colnames(ratio_matrix),
    cex.names = LABEL_CEX,
    cex.axis = AXIS_CEX,
    cex.lab = LABEL_CEX,
    font.axis = 1,
    font.lab = 1,
    family = "sans",
    las = 1
  )

  box(lwd = BOX_LWD)

  legend(
    "topright",
    legend = rownames(ratio_matrix),
    fill = COLORS,
    border = NA,
    bty = "n",
    cex = LEGEND_CEX,
    text.font = 1
  )

  plot_title <- paste0(TARGET_COMPOUND, " - ", TARGET_TIME)
  title(
    main = capitalize_first(plot_title),
    cex.main = TITLE_CEX,
    font.main = 2,
    family = "sans",
    line = 1.0
  )
}

# =============================================================================
# Main analysis
# =============================================================================

main <- function() {
  check_input_files(c(CONTROL_FILE, TREATMENT_FILE, GENERATED_FILE))

  control <- suppressMessages(
    read_csv(CONTROL_FILE, show_col_types = FALSE, progress = FALSE)
  )
  treatment <- suppressMessages(
    read_csv(TREATMENT_FILE, show_col_types = FALSE, progress = FALSE)
  )
  generated <- suppressMessages(
    read_csv(GENERATED_FILE, show_col_types = FALSE, progress = FALSE)
  )

  real <- bind_rows(control, treatment)
  real <- keep_clinical_pathology_columns(real)

  generated <- harmonize_generated_values(generated)

  if (!"DOSE_LEVEL" %in% names(real)) {
    stop("DOSE_LEVEL is missing from the real-data inputs.")
  }
  if (!"DOSE_LEVEL" %in% names(generated)) {
    stop("DOSE_LEVEL is missing from the generated-control input.")
  }

  control_df <- real %>% filter(.data[["DOSE_LEVEL"]] == "Control")
  treatment_df <- real %>% filter(.data[["DOSE_LEVEL"]] == "High")
  generated_df <- generated %>% filter(.data[["DOSE_LEVEL"]] == "High")

  ratio_matrix <- calculate_ratio_matrix(
    treatment_df,
    control_df,
    generated_df
  )

  cat("Ratio of means:\n")
  print(round(ratio_matrix, 4))

  plot_ratio_bars(ratio_matrix, OUTPUT_FILE)

  cat("\nSaved plot:\n", normalizePath(OUTPUT_FILE, winslash = "/"), "\n")
}

main()
