# =============================================================================
# RMSE Boxplot (Real vs Synthetic Controls vs Baselines)
#
# This script:
#   1. Loads four RMSE distributions:
#        - Inter-lab control baseline
#        - Intra-lab control baseline
#        - GanCtrl (synthetic control RMSE vs real)
#        - Replicate control RMSE
#   2. Combines them into a single long-format data frame.
#   3. Draws a four-group boxplot:
#        Inter-lab Control, Intra-lab Control, GanCtrl, Replicate Control
#   4. Performs Welch t-tests (unadjusted p-values) for:
#        - Inter-lab vs GanCtrl
#        - Intra-lab vs GanCtrl
#        - Replicate vs GanCtrl
#   5. Adds three significance brackets with p-value labels.
#   6. Automatically chooses y-limits so tails and brackets are not cut off,
#      but y-axis tick labels are capped at 500 for readability.
#   7. Writes a 600 dpi TIFF file.
#
# To use:
#   - Edit the path_* variables below to point to your CSVs.
#   - Edit out_dir if you want a different output folder.
#   - Source this script and call save_rmse_boxplot_tif() or use the example.
# =============================================================================

library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(ggpubr)    # compare_means + stat_pvalue_manual (not used directly, but kept)
library(rstatix)   # compare_means also here; ggpubr re-exports
library(ggbeeswarm)

# -----------------------------
# Paths (EDIT FOR YOUR SETUP)
# -----------------------------
data_dir   <- "path/to/data"    # optional base dir, not required
output_dir <- "path/to/results" # where TIFF will be saved

# Individual CSV paths (use absolute or relative paths)
path_interlab  <- file.path(data_dir, "interlab_rmse_overall.csv")
path_intralab  <- file.path(data_dir, "intralab_rmse_overall.csv")
path_generated <- file.path(data_dir, "generated_rmse_all_test.csv")
path_replicate <- file.path(data_dir, "pc_rmse_overall.csv")

# Ensure the output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

# -----------------------------
# Load input RMSE data
# -----------------------------
interlab  <- read.csv(path_interlab)
intralab  <- read.csv(path_intralab)
generated <- read.csv(path_generated)
replicate <- read.csv(path_replicate)

# =============================================================================
# Main plotting function: save_rmse_boxplot_tif
#   - All internal logic unchanged; only default path is now generic.
# =============================================================================
save_rmse_boxplot_tif <- function(
  tif_path      = file.path(output_dir, "rmse_boxplot_liver.tif"),
  tif_width_in  = 9.5,
  tif_height_in = 6.5,
  tif_res       = 600,
  height_factor = 1.6,
  main_title    = NULL   # kept for compatibility, but not used
) {
  # Create the folder for this TIFF if needed
  dir.create(dirname(tif_path), recursive = TRUE, showWarnings = FALSE)

  # Use Cairo if available for better font rendering; fall back otherwise
  dev_fun <- if ("cairo" %in% capabilities())
    function(...) tiff(type = "cairo", compression = "lzw", ...)
  else
    function(...) tiff(compression = "lzw", ...)

  dev_fun(
    filename = tif_path,
    width    = tif_width_in,
    height   = tif_height_in * height_factor,
    units    = "in",
    res      = tif_res
  )
  on.exit(dev.off(), add = TRUE)

  # ---------- helpers ----------
  # Turn raw p-value into a compact label for plotting
  pad_p <- function(p) {
    if (is.na(p)) return("p = NA")
    if (p < 0.001) return("p < 0.05")
    if (p < 0.01)  return("p < 0.01")
    if (p < 0.05)  return("p < 0.05")
    paste0("p = ", signif(p, 3))
  }

  # Draw one bracket between x1 and x2 at height y
  draw_bracket <- function(x1, x2, y, h, label,
                           lwd = 1.6, cex = 1.25, label_pad = 0, font = 2) {
    segments(x1, y, x2, y, lwd = lwd)
    segments(x1, y, x1, y - h, lwd = lwd)
    segments(x2, y, x2, y - h, lwd = lwd)
    text((x1 + x2)/2, y + h + label_pad, labels = label,
         cex = cex, font = font)
  }

  # ---------- data prep ----------
  group_levels <- c("Inter-lab Control", "Intra-lab Control", "GanCtrl", "Replicate Control")
  group_labels <- c("Inter-lab\nControl", "Intra-lab\nControl", "GanCtrl\n", "Replicate\nControl")

  # Bind the four RMSE distributions into one data frame
  df <- rbind(
    data.frame(rmse = interlab$rmse,  group = "Inter-lab Control"),
    data.frame(rmse = intralab$rmse,  group = "Intra-lab Control"),
    data.frame(rmse = generated$rmse, group = "GanCtrl"),
    data.frame(rmse = replicate$rmse, group = "Replicate Control")
  )
  df$group <- factor(df$group, levels = group_levels)

  # One vector of RMSE values per group for boxplot() and t.test()
  plot_list <- lapply(group_levels, function(g) df$rmse[df$group == g])

  inter_vals <- df$rmse[df$group == "Inter-lab Control"]
  intra_vals <- df$rmse[df$group == "Intra-lab Control"]
  gen_vals   <- df$rmse[df$group == "GanCtrl"]
  rep_vals   <- df$rmse[df$group == "Replicate Control"]

  # ---------- t-tests (UNADJUSTED p-values) ----------
  p_vals <- c(
    stats::t.test(inter_vals, gen_vals)$p.value,
    stats::t.test(intra_vals, gen_vals)$p.value,
    stats::t.test(rep_vals,   gen_vals)$p.value
  )
  labels <- vapply(p_vals, pad_p, character(1))

  # Pre-compute boxplot stats to determine whiskers and y-limits
  bp_stats  <- boxplot(plot_list, plot = FALSE)

  # ---------- y-limits (so tails don't get cut) ----------
  data_top    <- max(bp_stats$stats[5, ], na.rm = TRUE)   # top whisker
  data_bottom <- min(bp_stats$stats[1, ], na.rm = TRUE)   # bottom whisker
  span_data   <- data_top - data_bottom
  if (!is.finite(span_data) || span_data <= 0) span_data <- 1

  offset_above <- max(span_data * 0.015, 1e-4)
  tip_height   <- max(span_data * 0.030, 1e-4)

  top_between <- function(x1, x2) {
    i <- seq(min(x1, x2), max(x1, x2))
    max(bp_stats$stats[5, i], na.rm = TRUE)
  }

  # bracket bases just above relevant box tops
  y_inter <- top_between(1, 3) + offset_above + span_data * 0.03
  y_intra <- top_between(2, 3) + offset_above + span_data * 0.09
  y_repl  <- top_between(3, 4) + offset_above + span_data * 0.05

  bracket_top <- max(y_inter, y_intra, y_repl) + tip_height

  # Extra padding below and above for aesthetics
  lower_pad   <- max(span_data * 0.02, 1e-4)
  upper_pad   <- max(span_data * 0.05, 1e-3)

  ylim_bottom <- data_bottom - lower_pad
  ylim_top    <- max(data_top + upper_pad, bracket_top + upper_pad)

  ylim  <- c(ylim_bottom, ylim_top)
  span  <- diff(ylim)

  # extra offset for p-value labels (more space above bracket bar)
  label_offset <- span * 0.005

  # ---------- draw ----------
  cols       <- c("#4C72B0", "#E69F00", "#55A868", "#C44E52")
  border_col <- "black"
  at_pos     <- 1:4
  xlim       <- c(0.5, 4.5)

  op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)
  par(
    mar      = c(4.6, 7.2, 3.8, 2),
    mgp      = c(5, 0.8, 0),
    xpd      = FALSE,
    cex.lab  = 1.6, font.lab = 2,
    cex.axis = 1.2, font.axis = 1
  )

  plot.new(); plot.window(xlim = xlim, ylim = ylim)

  # We intentionally skip a main title; panel is self-contained via y-label & x-labels

  # y-axis ticks: spaced by 50, but only up to 500; axis still extends above if needed
  tick_by    <- 50
  start_tick <- floor(ylim[1] / tick_by) * tick_by
  end_tick   <- ceiling(ylim[2] / tick_by) * tick_by
  yticks_all <- seq(start_tick, end_tick, by = tick_by)
  yticks     <- yticks_all[yticks_all <= 500]   # clip labelled ticks at 500

  axis(2,
       at     = yticks,
       labels = sprintf("%.0f", yticks),
       las    = 1, lwd = 0, lwd.ticks = 1.2, tck = -0.015)

  # Actual boxplots (no outliers drawn, heavier lines for medians)
  boxplot(
    plot_list,
    add       = TRUE,
    at        = at_pos,
    names     = FALSE,
    col       = adjustcolor(cols, alpha.f = 0.8),
    border    = border_col,
    lwd       = 2.0,
    medlwd    = 3,
    whisklty  = 1,
    staplelty = 1,
    boxwex    = 0.4,
    outline   = FALSE,
    notch     = FALSE,
    axes      = FALSE
  )

  title(ylab = "RMSE", font.lab = 2)

  # x-axis labels (two lines for Inter-/Intra-lab and Replicate)
  axis(1, at = at_pos, labels = FALSE, tick = FALSE)
  mtext(text = group_labels, side = 1, at = at_pos,
        line = 2.2, cex = 1.3, font = 1)

  box(bty = "o", lwd = 1.5)

  # p-value brackets (UNADJUSTED)
  draw_bracket(at_pos[1], at_pos[3], y = y_inter, h = tip_height * 0.9,
               label = labels[1], label_pad = label_offset)
  draw_bracket(at_pos[2], at_pos[3], y = y_intra, h = tip_height * 0.9,
               label = labels[2], label_pad = label_offset)
  draw_bracket(at_pos[4], at_pos[3], y = y_repl,  h = tip_height * 0.9,
               label = labels[3], label_pad = label_offset)

  message(sprintf(
    "Raw p-values (Welch t-test, unadjusted): Inter vs GanCtrl = %.3g; Intra vs GanCtrl = %.3g; Replicate vs GanCtrl = %.3g",
    p_vals[1], p_vals[2], p_vals[3]
  ))
  message("Saved TIFF: ", tif_path)

  invisible(tif_path)
}

# -----------------------------
# Example call (main_title argument is kept for compatibility, but ignored)
# -----------------------------
tif_out <- file.path(output_dir, "rmse_overall_test.tif")
save_rmse_boxplot_tif(tif_path = tif_out, main_title = "Liver")
