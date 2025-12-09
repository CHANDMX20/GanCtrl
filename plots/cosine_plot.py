# =============================================================================
# Cosine Similarity Boxplot – Inter-/Intra-lab vs GanCtrl vs Replicate Control
# =============================================================================
# Purpose
# -------
# This script:
#   1. Reads cosine similarity distributions for:
#        - Inter-lab Controls
#        - Intra-lab Controls
#        - Synthetic Controls (GanCtrl)
#        - Replicate Controls (positive control)
#   2. Combines them into a single data frame with a 'group' column.
#   3. Performs pairwise Welch t-tests vs GanCtrl:
#        - Inter-lab vs GanCtrl
#        - Intra-lab vs GanCtrl
#        - Replicate vs GanCtrl
#   4. Draws a publication-style boxplot (no points) with:
#        - custom y-limits based on full data
#        - nicely spaced x labels
#        - p-value brackets above the boxes
#   5. Saves a high-resolution TIFF figure.
#
# How to use
# ----------
# 1. Update the four input CSV paths below ("path/to/...").
# 2. Update the default `tif_path` inside `save_cosine_boxplot_tif()` if desired.
# 3. Run the script; it will:
#      - read the four input files
#      - create the figure
#      - save a TIFF to the chosen output path.
#
# Expected input CSVs
# -------------------
# interlab_cosine_overall.csv
# intralab_cosine_overall.csv
# generated_cosine_all_test.csv
# pc_cosine_overall.csv
#
# Each should have:
#   - A numeric column named "cosine" giving the cosine similarity value
#     per observation (e.g., per compound-time pair).
#
# Notes
# -----
# - The function uses UNADJUSTED Welch t-test p-values.
# - `main_title` argument exists but is currently not drawn (kept for API symmetry).
# =============================================================================

library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(ggpubr)    # compare_means + stat_pvalue_manual (not directly used but often handy)
library(rstatix)   # (compare_means is also here; ggpubr re-exports)
library(ggbeeswarm)

# =============================================================================
# 1) Load cosine similarity data from four sources
#    (EDIT THESE PATHS FOR YOUR ENVIRONMENT)
# =============================================================================

interlab <- read.csv("path/to/baseline/interlab_cosine_overall.csv")
intralab <- read.csv("path/to/baseline/intralab_cosine_overall.csv")
generated <- read.csv("path/to/results_cosine/generated_cosine_all_test.csv")
replicate <- read.csv("path/to/positive_control/pc_cosine_overall.csv")

# =============================================================================
# 2) Main plotting function: save_cosine_boxplot_tif()
# =============================================================================
# Arguments
# ---------
# tif_path      : full path to output TIFF file
# tif_width_in  : width in inches
# tif_height_in : base height in inches (multiplied by height_factor)
# tif_res       : DPI
# height_factor : vertical stretch factor for the plot region
# main_title    : optional title (currently not drawn)
#
# Returns
# -------
# Invisibly returns the path to the saved TIFF file.
# Also prints the raw p-values and the save path to the console.
# =============================================================================

save_cosine_boxplot_tif <- function(
  tif_path      = "path/to/results_plots/cosine_boxplot_liver.tif",
  tif_width_in  = 9.5,
  tif_height_in = 6.5,
  tif_res       = 600,
  height_factor = 1.6,
  main_title    = NULL   # kept for compatibility; not drawn currently
) {
  # Ensure output directory exists
  dir.create(dirname(tif_path), recursive = TRUE, showWarnings = FALSE)

  # Use Cairo TIFF when available (better font/antialiasing), else fallback
  dev_fun <- if ("cairo" %in% capabilities())
    function(...) tiff(type = "cairo", compression = "lzw", ...)
  else
    function(...) tiff(compression = "lzw", ...)

  # Open TIFF device
  dev_fun(
    filename = tif_path,
    width    = tif_width_in,
    height   = tif_height_in * height_factor,
    units    = "in",
    res      = tif_res
  )
  on.exit(dev.off(), add = TRUE)

  # ---------- helper: format p-values for plotting ----------
  pad_p <- function(p) {
    if (is.na(p)) return("p = NA")
    if (p < 0.001) return("p < 0.05")  # intentionally "coarse" for small p
    if (p < 0.01)  return("p < 0.01")
    if (p < 0.05)  return("p < 0.05")
    paste0("p = ", signif(p, 3))
  }

  # ---------- helper: draw a bracket + label between two x-positions ----------
  draw_bracket <- function(x1, x2, y, h, label,
                           lwd = 1.6, cex = 1.25, label_pad = 0, font = 2) {
    # horizontal top
    segments(x1, y, x2, y, lwd = lwd)
    # vertical arms
    segments(x1, y, x1, y - h, lwd = lwd)
    segments(x2, y, x2, y - h, lwd = lwd)
    # label above
    text((x1 + x2)/2, y + h + label_pad, labels = label, cex = cex, font = font)
  }

  # =============================================================================
  # 3) Data prep: combine all four sources into a single data frame
  # =============================================================================
  group_levels <- c("Inter-lab Control", "Intra-lab Control", "GanCtrl", "Replicate Control")
  group_labels <- c("Inter-lab\nControl", "Intra-lab\nControl", "GanCtrl\n", "Replicate\nControl")

  df <- rbind(
    data.frame(cosine = interlab$cosine,  group = "Inter-lab Control"),
    data.frame(cosine = intralab$cosine,  group = "Intra-lab Control"),
    data.frame(cosine = generated$cosine, group = "GanCtrl"),
    data.frame(cosine = replicate$cosine, group = "Replicate Control")
  )
  df$group <- factor(df$group, levels = group_levels)

  # Build a list per group for boxplot()
  plot_list <- lapply(group_levels, function(g) df$cosine[df$group == g])

  inter_vals <- df$cosine[df$group == "Inter-lab Control"]
  intra_vals <- df$cosine[df$group == "Intra-lab Control"]
  gen_vals   <- df$cosine[df$group == "GanCtrl"]
  rep_vals   <- df$cosine[df$group == "Replicate Control"]

  # =============================================================================
  # 4) Statistical tests – UNADJUSTED Welch t-tests vs GanCtrl
  # =============================================================================
  p_vals <- c(
    stats::t.test(inter_vals, gen_vals)$p.value,  # Inter-lab vs GanCtrl
    stats::t.test(intra_vals, gen_vals)$p.value,  # Intra-lab vs GanCtrl
    stats::t.test(rep_vals,   gen_vals)$p.value   # Replicate vs GanCtrl
  )
  labels <- vapply(p_vals, pad_p, character(1))

  # Pre-compute boxplot stats for y-limit and bracket placement
  bp_stats <- boxplot(plot_list, plot = FALSE)

  # =============================================================================
  # 5) Y-limits and spacing based on FULL data (no trimming)
  # =============================================================================
  y_max      <- max(df$cosine, na.rm = TRUE)
  y_floor    <- min(bp_stats$stats[1, ], na.rm = TRUE)  # min of lower whiskers
  bottom_pad <- max((y_max - y_floor) * 0.01, 0.0002)
  y_min      <- y_floor - bottom_pad

  span <- if (is.finite(y_max - y_min)) (y_max - y_min) else 1

  # offsets for bracket placement and text
  offset_above <- max(span * 0.06, 0.0015)
  step_gap     <- max(span * 0.08, 0.0015)
  tip_height   <- max(span * 0.015, 0.0006)

  # helper to find the max upper whisker between two boxes
  top_between <- function(x1, x2) {
    i <- seq(min(x1, x2), max(x1, x2))
    max(bp_stats$stats[5, i], na.rm = TRUE)
  }

  # bracket vertical positions
  y_inter <- top_between(1, 3) + offset_above + 0.015  # Inter-lab vs GanCtrl
  y_intra <- top_between(2, 3) + offset_above + 0.005  # Intra-lab vs GanCtrl
  y_repl  <- top_between(3, 4) + offset_above + 0.00035  # Replicate vs GanCtrl

  # allow some headroom above highest bracket
  headroom_extra <- max(span * 0.05, 0.0012)
  ylim_top       <- y_repl + step_gap + headroom_extra

  top_pad <- max(span * 0.02, 0.0006)
  ylim    <- c(y_min, ylim_top + top_pad)

  # =============================================================================
  # 6) Draw boxplot + custom axes + brackets
  # =============================================================================
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

  plot.new()
  plot.window(xlim = xlim, ylim = ylim)

  # NOTE: main_title is currently not drawn; uncomment if you want it:
  # if (!is.null(main_title)) {
  #   title(main = main_title, font.main = 2, cex.main = 1.6)
  # }

  # ---- y-axis ticks at 0.02 steps within the chosen ylim
  tick_by    <- 0.02
  start_tick <- ceiling(ylim[1] / tick_by) * tick_by
  end_tick   <- floor(ylim[2] / tick_by) * tick_by
  yticks     <- seq(start_tick, end_tick, by = tick_by)

  axis(
    2,
    at     = yticks,
    labels = sprintf("%.3f", yticks),
    las    = 1,
    lwd    = 0,
    lwd.ticks = 1.2,
    tck    = -0.015
  )

  # Boxplot overlays
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

  # Y-axis label
  title(ylab = "Cosine Similarity", font.lab = 2)

  # X-axis labels (two-line style)
  axis(1, at = at_pos, labels = FALSE, tick = FALSE)
  mtext(text = group_labels, side = 1, at = at_pos,
        line = 2.2, cex = 1.3, font = 1)

  box(bty = "o", lwd = 1.5)

  # p-value brackets (UNADJUSTED p-values vs GanCtrl)
  draw_bracket(
    at_pos[1], at_pos[3],
    y         = y_inter,
    h         = tip_height * 1.8,
    label     = labels[1],
    label_pad = tip_height * 0.2
  )

  draw_bracket(
    at_pos[2], at_pos[3],
    y         = y_intra,
    h         = tip_height * 1.8,
    label     = labels[2],
    label_pad = tip_height * 0.2
  )

  draw_bracket(
    at_pos[4], at_pos[3],
    y         = y_repl,
    h         = tip_height * 1.8,
    label     = labels[3],
    label_pad = tip_height * 0.2
  )

  # Console messages for reproducibility/logging
  message(sprintf(
    "Raw p-values (Welch t-test, unadjusted): Inter vs GanCtrl = %.3g; Intra vs GanCtrl = %.3g; Replicate vs GanCtrl = %.3g",
    p_vals[1], p_vals[2], p_vals[3]
  ))
  message("Saved TIFF: ", tif_path)

  invisible(tif_path)
}

# =============================================================================
# 7) Example call
# =============================================================================
# This block demonstrates how to call the function and attach a timestamp
# to the filename. You can adjust the directory and naming convention as needed.
# =============================================================================

ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
tif_out <- sprintf("path/to/results_plots/cosine_overall_test_%s.tif", ts)

# Note: 'main_title' exists but is not drawn in the current implementation.
save_cosine_boxplot_tif(
  tif_path   = tif_out,
  main_title = "Liver"   # kept as metadata / future use
)
