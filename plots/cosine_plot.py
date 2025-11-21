library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(ggpubr)    # compare_means + stat_pvalue_manual
library(rstatix)   # (compare_means is also here; ggpubr re-exports)
library(ggbeeswarm)

interlab = read.csv('/account001/mansi.chandra/clin_path/baseline/interlab_cosine_overall.csv')
intralab = read.csv('/account001/mansi.chandra/clin_path/baseline/intralab_cosine_overall.csv')
generated = read.csv('/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/performance/cosine/test/generated_cosine_all_test.csv')
replicate = read.csv('/account001/mansi.chandra/clin_path/positive_control/pc_cosine_overall.csv')

save_cosine_boxplot_tif <- function(
  tif_path      = "/account001/mansi.chandra/clin_path/results_plots_updated/cosine_boxplot_liver.tif",
  tif_width_in  = 9.5,
  tif_height_in = 6.5,
  tif_res       = 600,
  height_factor = 1.6,
  main_title    = NULL   # ignored, no main title drawn
) {
  dir.create(dirname(tif_path), recursive = TRUE, showWarnings = FALSE)

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
  pad_p <- function(p) {
    if (is.na(p)) return("p = NA")
    if (p < 0.001) return("p < 0.05")
    if (p < 0.01)  return("p < 0.01")
    if (p < 0.05)  return("p < 0.05")
    paste0("p = ", signif(p, 3))
  }

  draw_bracket <- function(x1, x2, y, h, label,
                           lwd = 1.6, cex = 1.25, label_pad = 0, font = 2) {
    segments(x1, y, x2, y, lwd = lwd)          # top horizontal
    segments(x1, y, x1, y - h, lwd = lwd)      # left arm
    segments(x2, y, x2, y - h, lwd = lwd)      # right arm
    text((x1 + x2)/2, y + h + label_pad,       # label above bracket
         labels = label, cex = cex, font = font)
  }

  # ---------- data prep ----------
  group_levels <- c("Inter-lab Control", "Intra-lab Control", "GanCtrl", "Replicate Control")
  group_labels <- c("Inter-lab\nControl", "Intra-lab\nControl", "GanCtrl\n", "Replicate\nControl")

  df <- rbind(
    data.frame(cosine = interlab$cosine,  group = "Inter-lab Control"),
    data.frame(cosine = intralab$cosine,  group = "Intra-lab Control"),
    data.frame(cosine = generated$cosine, group = "GanCtrl"),
    data.frame(cosine = replicate$cosine, group = "Replicate Control")
  )
  df$group <- factor(df$group, levels = group_levels)

  # ---- NO TRIMMING: use full data ----
  plot_list <- lapply(group_levels, function(g) df$cosine[df$group == g])

  inter_vals <- df$cosine[df$group == "Inter-lab Control"]
  intra_vals <- df$cosine[df$group == "Intra-lab Control"]
  gen_vals   <- df$cosine[df$group == "GanCtrl"]
  rep_vals   <- df$cosine[df$group == "Replicate Control"]

  # ---------- t-tests (UNADJUSTED p-values) ----------
  p_vals <- c(
    stats::t.test(inter_vals, gen_vals)$p.value,
    stats::t.test(intra_vals, gen_vals)$p.value,
    stats::t.test(rep_vals,   gen_vals)$p.value
  )
  labels <- vapply(p_vals, pad_p, character(1))

  bp_stats  <- boxplot(plot_list, plot = FALSE)

  # ---------- tightened y-limits using FULL data ----------
  y_max      <- max(df$cosine, na.rm = TRUE)
  y_floor    <- min(bp_stats$stats[1, ], na.rm = TRUE)
  bottom_pad <- max((y_max - y_floor) * 0.01, 0.0002)
  y_min      <- y_floor - bottom_pad

  span <- if (is.finite(y_max - y_min)) (y_max - y_min) else 1

  # spacing above boxes for brackets
  offset_above <- max(span * 0.06, 0.0015)
  step_gap     <- max(span * 0.08, 0.0015)
  tip_height   <- max(span * 0.015, 0.0006)

  top_between <- function(x1, x2) {
    i <- seq(min(x1, x2), max(x1, x2))
    max(bp_stats$stats[5, i], na.rm = TRUE)
  }

  y_inter <- top_between(1, 3) + offset_above + 0.015
  y_intra <- top_between(2, 3) + offset_above + 0.005
  y_repl  <- top_between(3, 4) + offset_above + 0.00035

  headroom_extra <- max(span * 0.05, 0.0012)
  ylim_top       <- y_repl + step_gap + headroom_extra

  top_pad <- max(span * 0.02, 0.0006)
  ylim    <- c(y_min, ylim_top + top_pad)

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

  # no main title
  # title(main = main_title, font.main = 2, cex.main = 1.6)

  # ---- y-axis ticks
  tick_by    <- 0.02
  start_tick <- ceiling(ylim[1] / tick_by) * tick_by
  end_tick   <- floor(ylim[2] / tick_by) * tick_by
  yticks     <- seq(start_tick, end_tick, by = tick_by)

  axis(2,
       at     = yticks,
       labels = sprintf("%.3f", yticks),
       las    = 1, lwd = 0, lwd.ticks = 1.2, tck = -0.015)

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

  title(ylab = "Cosine Similarity", font.lab = 2)

  axis(1, at = at_pos, labels = FALSE, tick = FALSE)
  mtext(text = group_labels, side = 1, at = at_pos,
        line = 2.2, cex = 1.3, font = 1)

  box(bty = "o", lwd = 1.5)

  # p-value brackets (UNADJUSTED)
  # - longer arms: h = tip_height * 1.8
  # - more gap between bracket & text: label_pad = tip_height * 1.2
  draw_bracket(at_pos[1], at_pos[3],
               y = y_inter,
               h = tip_height * 1.8,
               label = labels[1],
               label_pad = tip_height * 0.2)

  draw_bracket(at_pos[2], at_pos[3],
               y = y_intra,
               h = tip_height * 1.8,
               label = labels[2],
               label_pad = tip_height * 0.2)

  draw_bracket(at_pos[4], at_pos[3],
               y = y_repl,
               h = tip_height * 1.8,
               label = labels[3],
               label_pad = tip_height * 0.2)

  message(sprintf(
    "Raw p-values (Welch t-test, unadjusted): Inter vs GanCtrl = %.3g; Intra vs GanCtrl = %.3g; Replicate vs GanCtrl = %.3g",
    p_vals[1], p_vals[2], p_vals[3]
  ))
  message("Saved TIFF: ", tif_path)

  invisible(tif_path)
}

# Example call
ts <- format(Sys.time(), "%Y%m%d_%H%M%S")
tif_out <- sprintf("/account001/mansi.chandra/clin_path/results_plots_updated/cosine_overall_test.tif", ts)
save_cosine_boxplot_tif(tif_path = tif_out, main_title = "Liver")  # title not drawn







