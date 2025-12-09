library(dplyr)
library(ggplot2)
library(ggpubr)
library(ggh4x)


interlab = read.csv('/account001/mansi.chandra/clin_path/baseline/interlab_concordance_overall.csv')
intralab = read.csv('/account001/mansi.chandra/clin_path/baseline/intralab_concordance_overall.csv')
generated = read.csv('/account001/mansi.chandra/clin_path/results_plots_updated/test_liver.csv')

## ========= 0) Inputs expected =========
## interlab, intralab, generated (data frames)

# --- helper: rename "best_accuracy" or "accuracy" -> "concordance" if needed
rename_accuracy <- function(df) {
  nms <- names(df)
  ln  <- tolower(nms)

  if ("concordance" %in% ln) return(df)
  idx_best <- which(ln == "best_accuracy")
  idx_acc  <- which(ln == "accuracy")
  if (length(idx_best) > 0L) {
    nms[idx_best[1]] <- "concordance"
  } else if (length(idx_acc) > 0L) {
    nms[idx_acc[1]] <- "concordance"
  }
  names(df) <- nms
  df
}

interlab  <- rename_accuracy(interlab)
intralab  <- rename_accuracy(intralab)
generated <- rename_accuracy(generated)   # data stays named 'generated'

## ========= 1) I/O and feature set =========
out_dir  <- "/account001/mansi.chandra/clin_path/results_plots_updated"
tif_file <- file.path(out_dir, "liver_concordance_test.tif")

selected_features <- c(
  "ALP(IU/L)", "ALT(IU/L)", "AST(IU/L)",
  "LDH(IU/L)", "TBIL(mg/dL)", "DBIL(mg/dL)",
  "GTP(IU/L)"
)

## ========= 2) Helpers =========
canon <- function(x) {
  x <- gsub("\\(.*?\\)", "", x)
  x <- tolower(x)
  gsub("[^a-z0-9]+", "", x)
}

detect_feature_col <- function(df) {
  nms <- tolower(names(df))
  hit <- which(nms %in% c("feature","analyte","marker","parameter","name"))
  if (length(hit) == 0L) stop("Could not find a feature/marker column.")
  names(df)[hit[1]]
}

extract_concordance <- function(df, sel_feats, source_name = "") {
  fcol <- detect_feature_col(df)
  ccol <- names(df)[tolower(names(df)) %in% c("concordance","concordance_score","conc")]
  if (length(ccol) == 0L) stop("Could not find a 'concordance' column.")
  ccol <- ccol[1]

  base_feats <- sel_feats[sel_feats != "Average"]
  key_df   <- canon(df[[fcol]])
  key_want <- canon(base_feats)

  out <- setNames(rep(NA_real_, length(base_feats)), base_feats)
  if (nrow(df) > 0) {
    for (i in seq_along(base_feats)) {
      idx <- which(key_df == key_want[i])
      if (length(idx) > 0L)
        out[i] <- mean(suppressWarnings(as.numeric(df[[ccol]][idx])), na.rm = TRUE)
    }
  }

  avg_val <- mean(out, na.rm = TRUE)
  if (!is.finite(avg_val)) avg_val <- NA_real_
  c(out, "Average" = avg_val)
}

## ========= 3) Build plotting matrix =========
vec_interlab <- extract_concordance(interlab,  selected_features, "Interlab")
vec_intralab <- extract_concordance(intralab,  selected_features, "Intralab")
vec_synctrl  <- extract_concordance(generated, selected_features, "GanCtrl")

feature_order <- selected_features
mat <- rbind(
  "GanCtrl"  = vec_synctrl[feature_order],    # 1st
  "Interlab" = vec_interlab[feature_order],   # 2nd
  "Intralab" = vec_intralab[feature_order]    # 3rd
)
if (anyNA(mat)) mat[is.na(mat)] <- 0

## ========= 4) Plotting =========
plot_concordance <- function(mat, feature_order) {
  op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)

  par(
    mar = c(8.5, 5.5, 2.5, 1.5),
    mgp = c(2.3, 0.6, 0),
    tcl = -0.3,
    las = 1,
    xpd = TRUE,
    font.lab = 2,
    family = "sans"
  )

  cols <- c("#1F78B4", "#A6CEE3", "#FFB74D")  # maps to row order of 'mat'

  lower_bound <- 0.2
  upper_bound <- 1.0

  # Clip values to [0.2, 1.0] and shift so bars start from 0 (which we relabel as 0.2)
  mat_clipped <- pmin(pmax(mat, lower_bound), upper_bound)
  mat_shifted <- mat_clipped - lower_bound

  ylim <- c(0, upper_bound - lower_bound)  # i.e., 0 to 0.8

  bp <- barplot(
    mat_shifted,
    beside    = TRUE,
    col       = cols,
    border    = "white",
    ylim      = ylim,
    axes      = FALSE,
    names.arg = rep("", length(feature_order)),
    ylab      = "Concordance",
    cex.lab   = 1.1,
    space     = c(0.1, 1.0),
    width     = 0.85
  )

  # Y-axis: labels from 0.2 to 1.0, but ticks placed at shifted positions
  axis_values_plot  <- seq(0, upper_bound - lower_bound, by = 0.2)       # 0, 0.2, ..., 0.8
  axis_values_real  <- seq(lower_bound, upper_bound, by = 0.2)           # 0.2, 0.4, ..., 1.0

  axis(2,
       at     = axis_values_plot,
       labels = sprintf("%.1f", axis_values_real),
       cex.axis = 0.90,
       font = 2)

  box(bty = "o", lwd = 1.2)

  axis(1,
       at = colMeans(bp),
       labels = gsub("\\(.*?\\)", "", feature_order),
       tick = FALSE,
       cex.axis = 1.1,
       font = 2,
       line = 0.2)

  title("Liver", line = 1.0, cex.main = 1.2, font.main = 2)

  par(xpd = NA)
  legend_labels <- c("GanCtrl", "Inter-lab", "Intra-lab")
  legend(
    "bottom",
    inset = c(0, -0.36),
    horiz = TRUE,
    bty   = "n",
    fill  = cols,
    border= NA,
    legend= legend_labels,
    cex   = 1.0,
    text.font = 2,
    pt.cex = 2,
    x.intersp = 0.8,
    seg.len = 2.0,
    text.width = max(strwidth(legend_labels)) * 1.4,
    xjust = 0.5,
    yjust = 0
  )
}


## --- Apply order and labels ---
ordered_features <- c("ALP", "ALT", "AST", "GTP", "LDH", "DBIL", "TBIL")
clean_names <- function(x) gsub("\\(.*?\\)", "", x)
colnames(mat) <- clean_names(colnames(mat))
mat <- mat[, ordered_features, drop = FALSE]

## --- Save final figure ---
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

tiff(tif_file, width = 7200, height = 3000, res = 600, compression = "lzw")
plot_concordance(mat, ordered_features)
dev.off()

cat("Saved final TIFF with order: GanCtrl, Inter-lab, Intra-lab\n", tif_file, "\n")

interlab = read.csv('/account001/mansi.chandra/clin_path/baseline/interlab_concordance_overall.csv')
intralab = read.csv('/account001/mansi.chandra/clin_path/baseline/intralab_concordance_overall.csv')
generated = read.csv('/account001/mansi.chandra/clin_path/results_plots_updated/test_kidney.csv')


## ========= Finalized Kidney Concordance Plot (SynCtrl first) =========

# --- 0) Inputs expected ---
# interlab, intralab, generated (data frames)

rename_accuracy <- function(df) {
  nms <- names(df)
  ln  <- tolower(nms)
  if ("concordance" %in% ln) return(df)
  idx_best <- which(ln == "best_accuracy")
  idx_acc  <- which(ln == "accuracy")
  if (length(idx_best) > 0L) {
    nms[idx_best[1]] <- "concordance"
  } else if (length(idx_acc) > 0L) {
    nms[idx_acc[1]] <- "concordance"
  }
  names(df) <- nms
  df
}

interlab  <- rename_accuracy(interlab)
intralab  <- rename_accuracy(intralab)
generated <- rename_accuracy(generated)

## ========= 1) I/O and feature set =========
out_dir  <- "/account001/mansi.chandra/clin_path/results_plots_updated"
tif_file <- file.path(out_dir, "kidney_concordance_test.tif")

selected_features <- c(
  "BUN(mg/dL)", "CRE(mg/dL)", "Ca(mg/dL)", "Cl(meq/L)",
  "IP(mg/dL)", "K(meq/L)", "Na(meq/L)"
)

## ========= 2) Helpers =========
canon <- function(x) {
  x <- gsub("\\(.*?\\)", "", x)
  x <- tolower(x)
  gsub("[^a-z0-9]+", "", x)
}

detect_feature_col <- function(df) {
  nms <- tolower(names(df))
  hit <- which(nms %in% c("feature","analyte","marker","parameter","name"))
  if (length(hit) == 0L) stop("Could not find a feature/marker column.")
  names(df)[hit[1]]
}

extract_concordance <- function(df, sel_feats, source_name = "") {
  fcol <- detect_feature_col(df)
  ccol <- names(df)[tolower(names(df)) %in% c("concordance","concordance_score","conc")]
  if (length(ccol) == 0L) stop("Could not find a 'concordance' column.")
  ccol <- ccol[1]

  base_feats <- sel_feats[sel_feats != "Average"]
  key_df   <- canon(df[[fcol]])
  key_want <- canon(base_feats)

  out <- setNames(rep(NA_real_, length(base_feats)), base_feats)
  if (nrow(df) > 0) {
    for (i in seq_along(base_feats)) {
      idx <- which(key_df == key_want[i])
      if (length(idx) > 0L)
        out[i] <- mean(suppressWarnings(as.numeric(df[[ccol]][idx])), na.rm = TRUE)
    }
  }
  avg_val <- mean(out, na.rm = TRUE)
  if (!is.finite(avg_val)) avg_val <- NA_real_
  c(out, "Average" = avg_val)
}

## ========= 3) Build plotting matrix (SynCtrl, Inter-lab, Intra-lab) =========
vec_interlab <- extract_concordance(interlab,  selected_features, "Interlab")
vec_intralab <- extract_concordance(intralab,  selected_features, "Intralab")
vec_synctrl  <- extract_concordance(generated, selected_features, "GanCtrl")  # from 'generated'

feature_order <- selected_features
mat <- rbind(
  "GanCtrl"  = vec_synctrl[feature_order],
  "Interlab" = vec_interlab[feature_order],
  "Intralab" = vec_intralab[feature_order]
)
if (anyNA(mat)) mat[is.na(mat)] <- 0

## ========= 4) Plot function (matches liver aesthetic) =========
plot_concordance <- function(mat, feature_order) {
  op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)
  par(
    mar = c(8.5, 5.5, 2.5, 1.5),
    mgp = c(2.3, 0.6, 0),
    tcl = -0.3,
    las = 1,
    xpd = TRUE,
    font.lab = 2,
    family = "sans"
  )

  cols <- c("#1F78B4", "#A6CEE3", "#FFB74D")  # maps to row order of 'mat'

  lower_bound <- 0.2
  upper_bound <- 1.0

  # Clip to [0.2, 1.0] and shift so plot "0" corresponds to true 0.2
  mat_clipped <- pmin(pmax(mat, lower_bound), upper_bound)
  mat_shifted <- mat_clipped - lower_bound

  ylim <- c(0, upper_bound - lower_bound)  # 0 to 0.8

  bp <- barplot(
    mat_shifted,
    beside    = TRUE,
    col       = cols,
    border    = "white",
    ylim      = ylim,
    axes      = FALSE,
    names.arg = rep("", length(feature_order)),
    ylab      = "Concordance",
    cex.lab   = 1.1,
    space     = c(0.1, 1.0),
    width     = 0.28
  )

  ## Y-axis: ticks at shifted positions, labels as real concordance (0.2–1.0)
  axis_values_plot <- seq(0, upper_bound - lower_bound, by = 0.2)  # 0, 0.2, ..., 0.8
  axis_values_real <- seq(lower_bound, upper_bound, by = 0.2)      # 0.2, 0.4, ..., 1.0

  axis(2,
       at     = axis_values_plot,
       labels = sprintf("%.1f", axis_values_real),
       cex.axis = 0.9, font = 2)
  box(bty = "o", lwd = 1.2)

  ## X-axis
  axis(1,
       at = colMeans(bp),
       labels = gsub("\\(.*?\\)", "", feature_order),
       tick = FALSE,
       cex.axis = 1.1,
       font = 2,
       line = 0.2)

  ## Title
  title("Kidney", line = 1.0, cex.main = 1.2, font.main = 2)

  ## Legend (SynCtrl, Inter-lab, Intra-lab)
  par(xpd = NA)
  legend_labels <- c("GanCtrl", "Inter-lab", "Intra-lab")
  legend(
    "bottom",
    inset = c(0, -0.36),
    horiz = TRUE,
    bty   = "n",
    fill  = cols,
    border= NA,
    legend= legend_labels,
    cex   = 1.0,
    text.font = 2,
    pt.cex = 2,
    x.intersp = 0.8,
    seg.len = 2.0,
    text.width = max(strwidth(legend_labels)) * 1.4,
    xjust = 0.5,
    yjust = 0
  )
}


## --- Apply order and clean names ---
ordered_features <- c("BUN", "CRE", "Cl", "Ca", "K", "IP", "Na")
clean_names <- function(x) gsub("\\(.*?\\)", "", x)
colnames(mat) <- clean_names(colnames(mat))
mat <- mat[, ordered_features, drop = FALSE]

## ========= 5) Save final TIFF =========
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

tiff(tif_file, width = 7200, height = 3000, res = 600, compression = "lzw")
plot_concordance(mat, ordered_features)
dev.off()

cat("Saved final kidney concordance TIFF (order: GanCtrl, Inter-lab, Intra-lab):\n", tif_file, "\n")










