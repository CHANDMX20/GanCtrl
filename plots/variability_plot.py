library(dplyr)
library(ggplot2)
library(ggpubr)
library(ggh4x)



## ---------- inputs ----------
generated_path_liver <- "/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/predictions_decoded/test/generated_predictions_1161985_ControlGenerator_test.csv"
generated_path_kidney <- "/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/predictions_decoded/test/generated_predictions_1507935_ControlGenerator_test.csv"
real_path      <- "/account001/mansi.chandra/clin_path/repeat_test_control_cv2.csv"


## ---------- read data ----------
generated_liver <- read.csv(generated_path_liver, check.names = FALSE)
generated_kidney <- read.csv(generated_path_kidney, check.names=FALSE)
real      <- read.csv(real_path,      check.names = FALSE)


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
generated_liver <- num_round(generated_liver, "TBIL(mg/dL)", digits = 2)
generated_liver <- num_round(generated_liver, "RALB(g/dL)", digits = 2)
generated_liver <- num_round(generated_liver, "AST(IU/L)",  to_integer = TRUE)
generated_liver <- num_round(generated_liver, "TP(g/dL)",   digits = 1)
#generated_liver <- num_round(generated_liver, "CRE(mg/dL)", digits = 1)
generated_liver <- num_round(generated_liver, "DBIL(mg/dL)", digits = 2)
generated_liver <- num_round(generated_liver, "BUN(mg/dL)",  to_integer = TRUE)
#generated_liver <- num_round(generated_liver, "K(meq/L)",    digits = 2)
#generated_liver <- num_round(generated_liver, "GTP(IU/L)",   to_integer = TRUE)
#generated_liver <- num_round(generated_liver, "Ca(mg/dL)",   digits = 1)
#generated_liver <- num_round(generated_liver, "Cl(meq/L)",   digits = 1)
#generated_liver <- num_round(generated_liver, "Na(meq/L)",   digits = 1)
#generated_liver <- num_round(generated_liver, "IP(mg/dL)",   digits = 1)
generated_liver <- num_round(generated_liver, "ALP(IU/L)",   to_integer = TRUE)
generated_liver <- num_round(generated_liver, "ALT(IU/L)",   to_integer = TRUE)
generated_liver <- num_round(generated_liver, "LDH(IU/L)",   to_integer = TRUE)
## final RALB at 1 decimal (matches your last line overriding to 1 dp)
generated_liver <- num_round(generated_liver, "RALB(g/dL)",  digits = 1)

## ---------- generated_kidney ----------
generated_kidney <- num_round(generated_kidney, "TBIL(mg/dL)", digits = 2)
generated_kidney <- num_round(generated_kidney, "RALB(g/dL)",  digits = 2)
generated_kidney <- num_round(generated_kidney, "AST(IU/L)",   to_integer = TRUE)
generated_kidney <- num_round(generated_kidney, "TP(g/dL)",    digits = 1)
#generated_kidney <- num_round(generated_kidney, "CRE(mg/dL)",  digits = 1)
generated_kidney <- num_round(generated_kidney, "DBIL(mg/dL)", digits = 2)
generated_kidney <- num_round(generated_kidney, "BUN(mg/dL)",  to_integer = TRUE)
#generated_kidney <- num_round(generated_kidney, "K(meq/L)",    digits = 2)
#generated_kidney <- num_round(generated_kidney, "GTP(IU/L)",   to_integer = TRUE)
#generated_kidney <- num_round(generated_kidney, "Ca(mg/dL)",   digits = 1)
#generated_kidney <- num_round(generated_kidney, "Cl(meq/L)",   digits = 1)
#generated_kidney <- num_round(generated_kidney, "Na(meq/L)",   digits = 1)
#generated_kidney <- num_round(generated_kidney, "IP(mg/dL)",   digits = 1)
generated_kidney <- num_round(generated_kidney, "ALP(IU/L)",   to_integer = TRUE)
generated_kidney <- num_round(generated_kidney, "ALT(IU/L)",   to_integer = TRUE)
generated_kidney <- num_round(generated_kidney, "LDH(IU/L)",   to_integer = TRUE)
## final RALB at 1 decimal (last override)
generated_kidney <- num_round(generated_kidney, "RALB(g/dL)",  digits = 1)


# ---------- helpers ----------
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
      # Numeric DOSE: keep the highest dose within each compound × time × individual
      # (falls back to simple max if INDIVIDUAL_ID/targetTime missing)
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
      # Categorical DOSE_LEVEL/DOSE_GROUP: keep labels matching "high"
      df %>% filter(is_high_label(.data[[dose_col]]))
    }
  } else {
    warning("No DOSE/DOSE_LEVEL column found; skipping high-dose filter for this frame.")
    df
  }
}
# ---------- feature lists (as you have) ----------
liver_features <- c("ALP(IU/L)", "ALT(IU/L)", "AST(IU/L)",
                    "GTP(IU/L)", "LDH(IU/L)",
                    "TBIL(mg/dL)", "DBIL(mg/dL)",
                    "RALB(g/dL)", "TP(g/dL)")

kidney_features <- c("BUN(mg/dL)", "CRE(mg/dL)", "Ca(mg/dL)",
                     "Cl(meq/L)", "IP(mg/dL)", "K(meq/L)", "Na(meq/L)")

.to_num <- function(x) suppressWarnings(as.numeric(as.character(x)))

# ---------- FILTER: generated → only High dose ----------
generated_liver  <- filter_high_only(generated_liver)
generated_kidney <- filter_high_only(generated_kidney)

# ---------- collapse real ----------
real_liver <- real %>%
  select(any_of(c("COMPOUND_NAME","SACRIFICE_PERIOD","INDIVIDUAL_ID", liver_features))) %>%
  group_by(COMPOUND_NAME, SACRIFICE_PERIOD, INDIVIDUAL_ID) %>%
  summarise(across(all_of(liver_features), ~ mean(.to_num(.x), na.rm = TRUE)), .groups = "drop")

real_kidney <- real %>%
  select(any_of(c("COMPOUND_NAME","SACRIFICE_PERIOD","INDIVIDUAL_ID", kidney_features))) %>%
  group_by(COMPOUND_NAME, SACRIFICE_PERIOD, INDIVIDUAL_ID) %>%
  summarise(across(all_of(kidney_features), ~ mean(.to_num(.x), na.rm = TRUE)), .groups = "drop")

# ---------- collapse generated (High-dose only now) ----------
gen_liver_collapsed <- generated_liver %>%
  select(any_of(c("COMPOUND_NAME","targetTime","INDIVIDUAL_ID", liver_features))) %>%
  group_by(COMPOUND_NAME, targetTime, INDIVIDUAL_ID) %>%
  summarise(across(all_of(liver_features), ~ mean(.to_num(.x), na.rm = TRUE)), .groups = "drop")

gen_kidney_collapsed <- generated_kidney %>%
  select(any_of(c("COMPOUND_NAME","targetTime","INDIVIDUAL_ID", kidney_features))) %>%
  group_by(COMPOUND_NAME, targetTime, INDIVIDUAL_ID) %>%
  summarise(across(all_of(kidney_features), ~ mean(.to_num(.x), na.rm = TRUE)), .groups = "drop")






## ---------- helpers for plotting size, legend placement, and layout ----------

# Pick a legend corner that’s least likely to overlap (simple, robust heuristic)
.pick_legend_corner <- function(xr, xg) {
  m_r <- median(xr, na.rm = TRUE); m_g <- median(xg, na.rm = TRUE)
  m   <- median(c(xr, xg), na.rm = TRUE)
  # If most of the mass is on the right, put legend on the left; else on right
  if (mean(c(xr, xg) <= m, na.rm = TRUE) < 0.5) "topleft" else "topright"
}

# Single histogram panel (with compact legend in a smart corner)
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

  is_integerish <- function(x, tol = 1e-9)
    { x <- x[is.finite(x)]; length(x) > 0 && all(abs(x - round(x)) < tol) }

  if (is_integerish(xr) && is_integerish(xg)) {
    br <- seq(floor(min(c(xr, xg), na.rm=TRUE)) - 0.5,
              ceiling(max(c(xr, xg), na.rm=TRUE)) + 0.5, by = 1)
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
                    sprintf("GanCtrl (n=%d)", sum(is.finite(xg)))),
         fill   = c(col_real_fill, col_gen_fill),
         border = c(col_real_line, col_gen_line),
         bty = "n", cex = 1.4, inset = 0.02, x.intersp = 0.7, y.intersp = 0.8)
}


# Multi-panel plotter with correct kidney layout and big readable legends
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
    ## Kidney: 7 panels → center last in bottom row, no NA or empty boxes
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
    ## Liver: full 3x3 grid
    par(mfrow = c(3, 3), oma = c(0.5, 0.5, 0.5, 0.5), mar = c(4.3, 5, 3, 1.3))
    for (feat in features) {
      .hist_counts_panel(real_df, gen_df, feat, bins,
                         col_real_fill, col_real_line, col_gen_fill, col_gen_line)
    }
  }
}

# Liver order (as you specified)
liver_features <- c("ALP(IU/L)", "ALT(IU/L)", "AST(IU/L)",
                    "GTP(IU/L)", "LDH(IU/L)",
                    "TBIL(mg/dL)", "DBIL(mg/dL)",
                    "RALB(g/dL)", "TP(g/dL)")

# Kidney: 7 analytes
kidney_features <- c("BUN(mg/dL)", "CRE(mg/dL)", "Ca(mg/dL)",
                     "Cl(meq/L)", "IP(mg/dL)", "K(meq/L)", "Na(meq/L)")

# Save larger canvases; legends per subplot; last kidney plot centered
out_dir <- "/account001/mansi.chandra/clin_path/results_plots_updated"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

plot_hist_grid(real_liver,  gen_liver_collapsed,  liver_features,
               tiff_path = file.path(out_dir, "hist_counts_liver_test.tif"),
               bins = 25, width_px = 9000, height_px = 7000)

plot_hist_grid(real_kidney, gen_kidney_collapsed, kidney_features,
               tiff_path = file.path(out_dir, "hist_counts_kidney_test.tif"),
               bins = 25, width_px = 9000, height_px = 7000)
