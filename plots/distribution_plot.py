# =============================================================================
# Histograms of Real vs Synthetic (GanCtrl) Controls — Liver & Kidney Panels
# =============================================================================
# Purpose
# -------
# This script:
#   1. Loads:
#        - Synthetic “control-equivalent” predictions for liver and kidney
#        - Real control data (same study)
#   2. Harmonizes numeric formats (rounding / integer casting) so that
#      synthetic values align with the real reporting grids (e.g. AST as int).
#   3. Filters the synthetic data to High dose only, using generic DOSE/DOSE_LEVEL
#      logic that can handle either numeric doses or labelled dose levels.
#   4. Collapses both real and synthetic data to:
#        - Real: COMPOUND_NAME × SACRIFICE_PERIOD × INDIVIDUAL_ID
#        - Syn : COMPOUND_NAME × targetTime × targetBioCopy
#      and computes mean biomarker values per group.
#   5. Draws per-feature histograms comparing:
#        - Real controls vs GanCtrl synthetic controls
#      for:
#        - 7 liver biomarkers
#        - 7 kidney biomarkers (in a 3×3 layout with the last one centred)
#   6. Saves two high-resolution TIFFs for publication.
#
# Expected input columns
# ----------------------
# Real control file:
#   - COMPOUND_NAME
#   - SACRIFICE_PERIOD
#   - INDIVIDUAL_ID
#   - DOSE_LEVEL or DOSE_GROUP or DOSE (or similar)
#   - Liver features:
#       "ALP(IU/L)", "ALT(IU/L)", "AST(IU/L)",
#       "GTP(IU/L)", "LDH(IU/L)", "TBIL(mg/dL)", "DBIL(mg/dL)"
#   - Kidney features:
#       "BUN(mg/dL)", "CRE(mg/dL)", "Ca(mg/dL)",
#       "Cl(meq/L)", "IP(mg/dL)", "K(meq/L)", "Na(meq/L)"
#
# Generated (synthetic) files:
#   - COMPOUND_NAME
#   - targetTime
#   - targetBioCopy
#   - DOSE_LEVEL or DOSE (for filtering High dose)
#   - Same feature columns as above
#
# How to use
# ----------
# 1. Edit the three paths in the “inputs” section below.
# 2. Edit `out_dir` (at the bottom) to your desired output directory.
# 3. Source this script in R; it will produce two TIFFs:
#      - hist_counts_liver_test.tif
#      - hist_counts_kidney_test.tif
# =============================================================================

library(dplyr)
library(ggplot2)
library(ggpubr)
library(ggh4x)

## ---------- inputs (EDIT THESE FOR YOUR ENV) ----------
generated_path_liver  <- "path/to/generated_predictions_liver_test.csv"
generated_path_kidney <- "path/to/generated_predictions_kidney_test.csv"
real_path             <- "path/to/repeat_test_control.csv"

## ---------- read data ----------
generated_liver <- read.csv(generated_path_liver, check.names = FALSE)
generated_kidney <- read.csv(generated_path_kidney, check.names=FALSE)
real      <- read.csv(real_path,      check.names = FALSE)
real <- cbind(real[, 1:11], real[, 1369:ncol(real)])

## =============================================================================
## 1) Harmonize generated values (rounding / integer casting)
##    This mirrors your Python postprocessing to match reporting resolution.
## =============================================================================

## helper: coerce to numeric, round, optionally cast to integer
num_round <- function(df, col, digits = NULL, to_integer = FALSE) {
  if (!col %in% names(df)) return(df)
  v <- df[[col]]
  if (is.factor(v)) v <- as.character(v)
  x <- suppressWarnings(as.numeric(v))
  x <- if (is.null(digits)) round(x) else round(x, digits)
  if (to_integer) x <- as.integer(x)
  df[[col]] <- x
  df
}

## ---------- generated_liver ----------
#generated_liver <- num_round(generated_liver, "TBIL(mg/dL)", digits = 2)
generated_liver <- num_round(generated_liver, "RALB(g/dL)", digits = 2)
generated_liver <- num_round(generated_liver, "AST(IU/L)",  to_integer = TRUE)
generated_liver <- num_round(generated_liver, "TP(g/dL)",   digits = 1)
#generated_liver <- num_round(generated_liver, "CRE(mg/dL)", digits = 1)
#generated_liver <- num_round(generated_liver, "DBIL(mg/dL)", digits = 2)
#generated_liver <- num_round(generated_liver, "BUN(mg/dL)",  to_integer = TRUE)
#generated_liver <- num_round(generated_liver, "K(meq/L)",    digits = 2)
#generated_liver <- num_round(generated_liver, "GTP(IU/L)",   to_integer = TRUE)
generated_liver <- num_round(generated_liver, "Ca(mg/dL)",   digits = 1)
generated_liver <- num_round(generated_liver, "Cl(meq/L)",   digits = 1)
#generated_liver <- num_round(generated_liver, "Na(meq/L)",   digits = 1)
generated_liver <- num_round(generated_liver, "IP(mg/dL)",   digits = 1)
generated_liver <- num_round(generated_liver, "ALP(IU/L)",   to_integer = TRUE)
generated_liver <- num_round(generated_liver, "ALT(IU/L)",   to_integer = TRUE)
generated_liver <- num_round(generated_liver, "LDH(IU/L)",   to_integer = TRUE)
## final RALB at 1 decimal (matches your last line overriding to 1 dp)
generated_liver <- num_round(generated_liver, "RALB(g/dL)",  digits = 1)

## ---------- generated_kidney ----------
#generated_kidney <- num_round(generated_kidney, "TBIL(mg/dL)", digits = 2)
generated_kidney <- num_round(generated_kidney, "RALB(g/dL)",  digits = 2)
generated_kidney <- num_round(generated_kidney, "AST(IU/L)",   to_integer = TRUE)
generated_kidney <- num_round(generated_kidney, "TP(g/dL)",    digits = 1)
#generated_kidney <- num_round(generated_kidney, "CRE(mg/dL)",  digits = 1)
#generated_kidney <- num_round(generated_kidney, "DBIL(mg/dL)", digits = 2)
#generated_kidney <- num_round(generated_kidney, "BUN(mg/dL)",  to_integer = TRUE)
#generated_kidney <- num_round(generated_kidney, "K(meq/L)",    digits = 2)
#generated_kidney <- num_round(generated_kidney, "GTP(IU/L)",   to_integer = TRUE)
generated_kidney <- num_round(generated_kidney, "Ca(mg/dL)",   digits = 1)
generated_kidney <- num_round(generated_kidney, "Cl(meq/L)",   digits = 1)
#generated_kidney <- num_round(generated_kidney, "Na(meq/L)",   digits = 1)
generated_kidney <- num_round(generated_kidney, "IP(mg/dL)",   digits = 1)
generated_kidney <- num_round(generated_kidney, "ALP(IU/L)",   to_integer = TRUE)
generated_kidney <- num_round(generated_kidney, "ALT(IU/L)",   to_integer = TRUE)
generated_kidney <- num_round(generated_kidney, "LDH(IU/L)",   to_integer = TRUE)
## final RALB at 1 decimal (last override)
generated_kidney <- num_round(generated_kidney, "RALB(g/dL)",  digits = 1)


## =============================================================================
## 2) Feature lists (biomarker panels)
## =============================================================================

liver_features <- c("ALP(IU/L)", "ALT(IU/L)", "AST(IU/L)",
                    "GTP(IU/L)", "LDH(IU/L)",
                    "TBIL(mg/dL)", "DBIL(mg/dL)")

kidney_features <- c("BUN(mg/dL)", "CRE(mg/dL)", "Ca(mg/dL)",
                     "Cl(meq/L)", "IP(mg/dL)", "K(meq/L)", "Na(meq/L)")

.to_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

## =============================================================================
## 3) Dose filtering — keep only “High” dose synthetic controls
##    (generic logic: works with labelled or numeric dose columns)
## =============================================================================

normalize_dose <- function(x) tolower(trimws(as.character(x)))
is_high_label  <- function(x) normalize_dose(x) %in%
  c("high","hi","top","max","highest","hd","h")

pick_dose_col <- function(df) {
  # Try common variants; return the first that exists or NA
  cands <- c("DOSE_LEVEL","Dose_Level","DoseLevel","DOSE_GROUP","DoseGroup","DOSE")
  hit <- intersect(cands, names(df))
  if (length(hit)) hit[1] else NA_character_
}

filter_high_only <- function(df) {
  dose_col <- pick_dose_col(df)
  if (!is.na(dose_col)) {
    if (dose_col == "DOSE") {
      # Numeric DOSE: keep the highest dose within each group
      # Grouping keys: fall back to full-frame max if keys are missing
      group_keys <- intersect(c("COMPOUND_NAME","targetTime","INDIVIDUAL_ID"), names(df))
      if (length(group_keys) == 0) {
        df %>% filter(.data[[dose_col]] == max(.data[[dose_col]], na.rm = TRUE))
      } else {
        df %>%
          group_by(across(all_of(group_keys))) %>%
          filter(.data[[dose_col]] == max(.data[[dose_col]], na.rm = TRUE)) %>%
          ungroup()
      }
    } else {
      # Categorical DOSE_LEVEL/DOSE_GROUP: match High labels
      df %>% filter(is_high_label(.data[[dose_col]]))
    }
  } else {
    warning("No DOSE/DOSE_LEVEL column found; skipping high-dose filter for this frame.")
    df
  }
}

# ---------- FILTER: generated → only High dose ----------
generated_liver  <- filter_high_only(generated_liver)
generated_kidney <- filter_high_only(generated_kidney)

## =============================================================================
## 4) Collapse real & generated into per-group means
## =============================================================================

# Real: average per COMPOUND_NAME × SACRIFICE_PERIOD × INDIVIDUAL_ID
real_liver <- real %>%
  select(any_of(c("COMPOUND_NAME","SACRIFICE_PERIOD","INDIVIDUAL_ID", liver_features))) %>%
  group_by(COMPOUND_NAME, SACRIFICE_PERIOD, INDIVIDUAL_ID) %>%
  summarise(across(all_of(liver_features), ~ mean(.to_num(.x), na.rm = TRUE)), .groups = "drop")

real_kidney <- real %>%
  select(any_of(c("COMPOUND_NAME","SACRIFICE_PERIOD","INDIVIDUAL_ID", kidney_features))) %>%
  group_by(COMPOUND_NAME, SACRIFICE_PERIOD, INDIVIDUAL_ID) %>%
  summarise(across(all_of(kidney_features), ~ mean(.to_num(.x), na.rm = TRUE)), .groups = "drop")

# Generated: average per COMPOUND_NAME × targetTime × targetBioCopy
gen_liver_collapsed <- generated_liver %>%
  select(any_of(c("COMPOUND_NAME","targetTime","targetBioCopy", liver_features))) %>%
  group_by(COMPOUND_NAME, targetTime, targetBioCopy) %>%
  summarise(across(all_of(liver_features), ~ mean(.to_num(.x), na.rm = TRUE)), .groups = "drop")

gen_kidney_collapsed <- generated_kidney %>%
  select(any_of(c("COMPOUND_NAME","targetTime","targetBioCopy", kidney_features))) %>%
  group_by(COMPOUND_NAME, targetTime, targetBioCopy) %>%
  summarise(across(all_of(kidney_features), ~ mean(.to_num(.x), na.rm = TRUE)), .groups = "drop")

## =============================================================================
## 5) Plotting helpers — multi-panel histograms
## =============================================================================

## ---------- helpers for plotting ----------

# (kept simple; legend is fixed in top-right of each panel)
.hist_counts_panel <- function(real_df, gen_df, feat_name, bins = 25,
                               col_real_fill, col_real_line,
                               col_gen_fill,  col_gen_line) {
  .to_num <- function(x) { x <- suppressWarnings(as.numeric(x)); x[is.finite(x)] }

  xr <- .to_num(real_df[[feat_name]])
  xg <- .to_num(gen_df[[feat_name]])

  if (length(xr) + length(xg) == 0) {
    plot.new(); box(); title(main = feat_name, cex.main = 1.4, font.main = 2)
    return(invisible())
  }

  is_integerish <- function(x, tol = 1e-9) {
    x <- x[is.finite(x)]
    length(x) > 0 && all(abs(x - round(x)) < tol)
  }

  # Choose binning: integer bins if data look integer-like, otherwise numeric
  if (is_integerish(xr) && is_integerish(xg)) {
    br <- seq(floor(min(c(xr, xg), na.rm = TRUE)) - 0.5,
              ceiling(max(c(xr, xg), na.rm = TRUE)) + 0.5, by = 1)
  } else {
    rng <- range(c(xr, xg), finite = TRUE)
    if (!all(is.finite(rng)) || diff(rng) == 0) rng <- rng + c(-1, 1) * 1e-6
    br <- seq(rng[1], rng[2], length.out = bins + 1)
  }

  hr <- hist(xr, breaks = br, plot = FALSE)
  hg <- hist(xg, breaks = br, plot = FALSE)
  ymax <- max(c(hr$counts, hg$counts), na.rm = TRUE)
  if (!is.finite(ymax) || ymax == 0) ymax <- 1

  plot(hr, freq = TRUE,
       xlim = range(br), ylim = c(0, ymax * 1.25),
       col = col_real_fill, border = col_real_line, lwd = 1.5,
       xlab = "", ylab = "Count", main = feat_name,
       cex.axis = 1.35, cex.lab = 1.45, cex.main = 1.6, font.main = 2)
  plot(hg, freq = TRUE, add = TRUE,
       col = col_gen_fill, border = col_gen_line, lwd = 1.5)
  box(lwd = 1.4)

  legend("topright",
         legend = c(sprintf("Real Control (n=%d)", sum(is.finite(xr))),
                    sprintf("GanCtrl (n=%d)",      sum(is.finite(xg)))),
         fill   = c(col_real_fill, col_gen_fill),
         border = c(col_real_line, col_gen_line),
         bty = "n", cex = 1.4, inset = 0.02, x.intersp = 0.7, y.intersp = 0.8)
}

# Multi-panel plotter with:
#   - 3×3 grid for liver (7 panels + 2 empty if desired)
#   - 3×3 custom layout for kidney (last panel centred)
plot_hist_grid <- function(real_df, gen_df, features, tiff_path,
                           bins = 25, width_px = 9000, height_px = 7000) {

  col_real_line <- "#1F78B4"
  col_real_fill <- rgb(31,119,180, alpha = 120, maxColorValue = 255)
  col_gen_line  <- "#E31A1C"
  col_gen_fill  <- rgb(227,26,28,  alpha = 120, maxColorValue = 255)

  tiff(tiff_path, width = width_px, height = height_px, res = 600, compression = "lzw")
  op <- par(no.readonly = TRUE); on.exit({ par(op); dev.off() }, add = TRUE)
  par(family = "sans")

  if (length(features) == 7) {
    ## Kidney: 7 panels → centre last in bottom row
    layout_matrix <- matrix(c(1, 2, 3,
                              4, 5, 6,
                              0, 7, 0), nrow = 3, byrow = TRUE)
    layout(layout_matrix, heights = c(1, 1, 0.9))
    par(oma = c(0.5, 0.5, 0.5, 0.5), mar = c(4.3, 5, 3, 1.3))
    for (i in seq_len(max(layout_matrix))) {
      pos <- which(layout_matrix == i, arr.ind = TRUE)
      if (nrow(pos) == 0) next
      .hist_counts_panel(real_df, gen_df, features[i], bins,
                         col_real_fill, col_real_line, col_gen_fill, col_gen_line)
    }
  } else {
    ## Liver: full 3×3 grid
    par(mfrow = c(3, 3), oma = c(0.5, 0.5, 0.5, 0.5), mar = c(4.3, 5, 3, 1.3))
    for (feat in features) {
      .hist_counts_panel(real_df, gen_df, feat, bins,
                         col_real_fill, col_real_line, col_gen_fill, col_gen_line)
    }
  }
}

## =============================================================================
## 6) Final calls: draw and save liver & kidney histograms
## =============================================================================

# Liver order (as you specified)
liver_features <- c("ALP(IU/L)", "ALT(IU/L)", "AST(IU/L)",
                    "GTP(IU/L)", "LDH(IU/L)",
                    "DBIL(mg/dL)", "TBIL(mg/dL)")

# Kidney: 7 analytes
kidney_features <- c("BUN(mg/dL)", "CRE(mg/dL)", "Cl(meq/L)", "Ca(mg/dL)",
                     "K(meq/L)", "IP(mg/dL)", "Na(meq/L)")

# Output directory (EDIT AS NEEDED)
plot_hist_grid(
  real_liver, gen_liver_collapsed, liver_features,
  tiff_path = "hist_counts_liver_test.tif",
  bins = 25, width_px = 9000, height_px = 7000
)

plot_hist_grid(
  real_kidney, gen_kidney_collapsed, kidney_features,
  tiff_path = "hist_counts_kidney_test.tif",
  bins = 25, width_px = 9000, height_px = 7000
)

# optional: confirm where they were saved + list them
getwd()
list.files(getwd(), pattern = "\\.tif$", full.names = TRUE)

