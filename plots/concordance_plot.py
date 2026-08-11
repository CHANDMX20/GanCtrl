#!/usr/bin/env Rscript

# GanCtrl vs. VCG/VCG-LR concordance heatmap
#
# This script compares endpoint-level concordance for GanCtrl against two
# virtual control group benchmarks:
#   - VCG: laboratory-matched historical controls
#   - VCG-LR: laboratory-relaxed historical controls
#
# Relative error is calculated as:
#   GanCtrl vs. VCG    = (VCG - GanCtrl) / VCG
#   GanCtrl vs. VCG-LR = (VCG-LR - GanCtrl) / VCG-LR
#
# Interpretation:
#   <= 0.5% relative difference: GanCtrl is better or comparable
#   >  0.5% relative difference: VCG/VCG-LR has higher concordance
#
# Expected input files in the working directory:
#   intralab_VCG_train.csv
#   interlab_VCG_train.csv
#   test_liver.csv
#   test_kidney.csv
#
# Output:
#   concordance_heatmap.tif


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

VCG_FILE <- "intralab_VCG_train.csv"
VCG_LR_FILE <- "interlab_VCG_train.csv"
GANCTRL_LIVER_FILE <- "test_liver.csv"
GANCTRL_KIDNEY_FILE <- "test_kidney.csv"
OUTPUT_FILE <- "concordance_heatmap.tif"

ZERO_TOLERANCE <- 0.005

LIVER_FEATURES <- c(
  "ALP(IU/L)",
  "ALT(IU/L)",
  "AST(IU/L)",
  "TBIL(mg/dL)",
  "DBIL(mg/dL)",
  "LDH(IU/L)",
  "GTP(IU/L)"
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

SELECTED_FEATURES <- c(LIVER_FEATURES, KIDNEY_FEATURES)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

rename_accuracy_column <- function(df) {
  lower_names <- tolower(names(df))

  if ("concordance" %in% lower_names) {
    return(df)
  }

  candidates <- c("best_accuracy", "accuracy")

  for (candidate in candidates) {
    idx <- which(lower_names == candidate)

    if (length(idx) > 0L) {
      names(df)[idx[1]] <- "concordance"
      return(df)
    }
  }

  df
}

canonicalize_feature <- function(x) {
  # Remove units in parentheses, convert to lowercase, and remove punctuation.
  x <- gsub("\\([^)]*\\)", "", as.character(x))
  x <- tolower(x)
  gsub("[^a-z0-9]+", "", x)
}

clean_feature_label <- function(x) {
  # Remove units from labels displayed on the heatmap.
  trimws(gsub("\\([^)]*\\)", "", x))
}

detect_feature_column <- function(df) {
  lower_names <- tolower(names(df))
  candidates <- c("feature", "analyte", "marker", "parameter", "name")
  idx <- which(lower_names %in% candidates)

  if (length(idx) == 0L) {
    stop(
      paste0(
        "Could not find a feature column. Expected one of: ",
        paste(candidates, collapse = ", ")
      )
    )
  }

  names(df)[idx[1]]
}

detect_concordance_column <- function(df, source_name) {
  lower_names <- tolower(names(df))
  candidates <- c("concordance", "concordance_score", "conc")
  idx <- which(lower_names %in% candidates)

  if (length(idx) == 0L) {
    stop(
      paste0(
        "Could not find a concordance column in ", source_name,
        ". Expected one of: ", paste(candidates, collapse = ", ")
      )
    )
  }

  names(df)[idx[1]]
}

extract_concordance <- function(df, selected_features, source_name) {
  feature_col <- detect_feature_column(df)
  concordance_col <- detect_concordance_column(df, source_name)

  data_keys <- canonicalize_feature(df[[feature_col]])
  requested_keys <- canonicalize_feature(selected_features)

  values <- setNames(rep(NA_real_, length(selected_features)), selected_features)

  for (i in seq_along(selected_features)) {
    idx <- which(data_keys == requested_keys[i])

    if (length(idx) == 0L) {
      next
    }

    numeric_values <- suppressWarnings(as.numeric(df[[concordance_col]][idx]))
    numeric_values <- numeric_values[is.finite(numeric_values)]

    if (length(numeric_values) > 0L) {
      values[i] <- mean(numeric_values)
    }
  }

  values
}

validate_feature_coverage <- function(values, source_name) {
  missing_features <- names(values)[is.na(values)]

  if (length(missing_features) > 0L) {
    warning(
      paste0(
        source_name,
        " is missing concordance values for: ",
        paste(missing_features, collapse = ", ")
      )
    )
  }
}

compute_relative_error <- function(reference, ganctrl) {
  error <- (reference - ganctrl) / reference
  error[!is.finite(error)] <- NA_real_
  error
}


# -----------------------------------------------------------------------------
# Heatmap plotting
# -----------------------------------------------------------------------------

plot_error_heatmap <- function(heat_mat, zero_tolerance = ZERO_TOLERANCE) {
  par(
    mar = c(5.8, 5.8, 1.2, 1.2),
    mgp = c(2.2, 0.6, 0),
    tcl = -0.25,
    las = 1,
    family = "sans",
    cex.axis = 1.15
  )

  n_rows <- nrow(heat_mat)
  n_cols <- ncol(heat_mat)

  # Reverse rows so the largest sorted difference appears at the top.
  z <- heat_mat[n_rows:1, , drop = FALSE]

  # Blue: GanCtrl better or within the comparability tolerance.
  # Red: benchmark has higher concordance beyond the tolerance.
  z_category <- matrix(NA_real_, nrow = n_rows, ncol = n_cols)
  z_category[z <= zero_tolerance] <- -1
  z_category[z > zero_tolerance] <- 1

  rownames(z_category) <- rownames(z)
  colnames(z_category) <- colnames(z)

  colors <- c("#D8E7F3", "#F3D8D8")

  image(
    x = seq_len(n_cols),
    y = seq_len(n_rows),
    z = t(z_category),
    col = colors,
    breaks = c(-1.5, 0, 1.5),
    axes = FALSE,
    xlab = "",
    ylab = "",
    useRaster = TRUE
  )

  abline(
    h = seq(0.5, n_rows + 0.5, by = 1),
    col = "white",
    lwd = 1.2
  )

  abline(
    v = seq(0.5, n_cols + 0.5, by = 1),
    col = "white",
    lwd = 1.2
  )

  box(col = "#6F6F6F", lwd = 1.2)

  axis(
    side = 1,
    at = seq_len(n_cols),
    labels = colnames(heat_mat),
    tick = FALSE,
    cex.axis = 1.22,
    font = 2,
    line = 0.9
  )

  axis(
    side = 2,
    at = seq_len(n_rows),
    labels = rownames(z),
    tick = FALSE,
    cex.axis = 1.20,
    font = 2,
    line = 0.15
  )

  # Display absolute percentage differences inside cells.
  for (i in seq_len(n_rows)) {
    for (j in seq_len(n_cols)) {
      value <- z[i, j]

      if (is.na(value)) {
        label <- "NA"
        text_color <- "#555555"
      } else {
        label <- if (abs(value) <= zero_tolerance) {
          "0%"
        } else {
          sprintf("%.0f%%", abs(value) * 100)
        }
        text_color <- "#2F2F2F"
      }

      text(
        x = j,
        y = i,
        labels = label,
        cex = 1.0,
        font = 2,
        family = "sans",
        col = text_color
      )
    }
  }
}

draw_heatmap_legend <- function() {
  plot.new()
  plot.window(xlim = c(0, 1), ylim = c(0, 1))

  colors <- c("#D8E7F3", "#F3D8D8")

  x_box <- 0.22
  x_text <- 0.30

  rect(
    xleft = x_box,
    ybottom = 0.58,
    xright = x_box + 0.05,
    ytop = 0.74,
    col = colors[1],
    border = "white"
  )

  text(
    x = x_text,
    y = 0.66,
    labels = "GanCtrl better / comparable",
    adj = c(0, 0.5),
    cex = 1.25,
    font = 2,
    family = "sans",
    col = "#2F2F2F"
  )

  rect(
    xleft = x_box,
    ybottom = 0.26,
    xright = x_box + 0.05,
    ytop = 0.42,
    col = colors[2],
    border = "white"
  )

  text(
    x = x_text,
    y = 0.34,
    labels = "VCG / VCG-LR better",
    adj = c(0, 0.5),
    cex = 1.25,
    font = 2,
    family = "sans",
    col = "#2F2F2F"
  )
}

save_heatmap <- function(heat_mat, output_file) {
  tiff(
    filename = output_file,
    width = 3300,
    height = 5600,
    res = 600,
    compression = "lzw"
  )

  on.exit(dev.off(), add = TRUE)

  layout(
    matrix(c(1, 2), nrow = 2, byrow = TRUE),
    heights = c(1, 0.06)
  )

  plot_error_heatmap(heat_mat)

  par(
    mar = c(0, 0, 0.1, 0),
    family = "sans"
  )

  draw_heatmap_legend()
}


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

main <- function() {
  input_files <- c(
    VCG_FILE,
    VCG_LR_FILE,
    GANCTRL_LIVER_FILE,
    GANCTRL_KIDNEY_FILE
  )

  missing_files <- input_files[!file.exists(input_files)]

  if (length(missing_files) > 0L) {
    stop(
      paste0(
        "Missing required input file(s): ",
        paste(missing_files, collapse = ", ")
      )
    )
  }

  vcg_df <- rename_accuracy_column(
    read.csv(VCG_FILE, check.names = FALSE)
  )

  vcg_lr_df <- rename_accuracy_column(
    read.csv(VCG_LR_FILE, check.names = FALSE)
  )

  ganctrl_liver_df <- rename_accuracy_column(
    read.csv(GANCTRL_LIVER_FILE, check.names = FALSE)
  )

  ganctrl_kidney_df <- rename_accuracy_column(
    read.csv(GANCTRL_KIDNEY_FILE, check.names = FALSE)
  )

  vcg_values <- extract_concordance(
    vcg_df,
    SELECTED_FEATURES,
    "VCG"
  )

  vcg_lr_values <- extract_concordance(
    vcg_lr_df,
    SELECTED_FEATURES,
    "VCG-LR"
  )

  ganctrl_liver_values <- extract_concordance(
    ganctrl_liver_df,
    LIVER_FEATURES,
    "GanCtrl liver"
  )

  ganctrl_kidney_values <- extract_concordance(
    ganctrl_kidney_df,
    KIDNEY_FEATURES,
    "GanCtrl kidney"
  )

  ganctrl_values <- c(ganctrl_liver_values, ganctrl_kidney_values)

  validate_feature_coverage(vcg_values, "VCG")
  validate_feature_coverage(vcg_lr_values, "VCG-LR")
  validate_feature_coverage(ganctrl_values, "GanCtrl")

  feature_order <- SELECTED_FEATURES

  error_vcg <- compute_relative_error(
    vcg_values[feature_order],
    ganctrl_values[feature_order]
  )

  error_vcg_lr <- compute_relative_error(
    vcg_lr_values[feature_order],
    ganctrl_values[feature_order]
  )

  heat_mat <- cbind(
    "GanCtrl vs. VCG" = error_vcg,
    "GanCtrl vs. VCG-LR" = error_vcg_lr
  )

  rownames(heat_mat) <- clean_feature_label(feature_order)

  # Sort by the absolute GanCtrl-vs.-VCG-LR relative difference.
  sort_score <- abs(heat_mat[, "GanCtrl vs. VCG-LR"])
  sort_score[!is.finite(sort_score)] <- NA_real_

  ordered_rows <- names(
    sort(sort_score, decreasing = TRUE, na.last = TRUE)
  )

  heat_mat_sorted <- heat_mat[ordered_rows, , drop = FALSE]

  cat("Relative errors:\n")
  print(round(heat_mat, 6))

  cat("\nRelative errors (%):\n")
  print(round(heat_mat * 100, 2))

  cat("\nEndpoint order based on absolute GanCtrl vs. VCG-LR difference:\n")
  print(ordered_rows)

  save_heatmap(heat_mat_sorted, OUTPUT_FILE)

  cat("\nSaved concordance heatmap:\n", OUTPUT_FILE, "\n")
}


if (sys.nframe() == 0L) {
  main()
}
