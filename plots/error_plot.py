# ==============================================================================
# False-positive vs. false-negative plot for focused clinical pathology endpoints
# ==============================================================================
#
# This script visualizes false-positive (FP) and false-negative (FN) counts for
# selected liver and kidney endpoints from the GanCtrl concordance analysis.
#
# Input:
#   test_concordance.csv
#
# Output:
#   fp_vs_fn_focused_endpoints.tiff
#
# Interpretation:
#   - Points below the FP = FN diagonal have more false positives than false
#     negatives (FP > FN).
#   - Points above the diagonal have more false negatives than false positives
#     (FN > FP).
# ==============================================================================

# ----------------------------- Configuration ----------------------------------

INPUT_FILE <- "test_concordance.csv"
OUTPUT_FILE <- "fp_vs_fn_focused_endpoints.tiff"

FOCUSED_ENDPOINTS <- c(
  "ALP(IU/L)",
  "ALT(IU/L)",
  "AST(IU/L)",
  "GTP(IU/L)",
  "LDH(IU/L)",
  "DBIL(mg/dL)",
  "TBIL(mg/dL)",
  "BUN(mg/dL)",
  "CRE(mg/dL)",
  "Cl(meq/L)",
  "Ca(mg/dL)",
  "K(meq/L)",
  "IP(mg/dL)",
  "Na(meq/L)"
)

# Plotting parameters
X_PADDING <- 14
Y_PADDING <- 16
POINT_CEX <- 1.75
LABEL_CEX <- 0.88
FONT_FAMILY <- "sans"


# ----------------------------- Helper functions -------------------------------

require_columns <- function(df, columns, object_name = "data frame") {
  missing_cols <- setdiff(columns, names(df))

  if (length(missing_cols) > 0) {
    stop(
      object_name,
      " is missing required columns: ",
      paste(missing_cols, collapse = ", ")
    )
  }
}

short_feature_name <- function(x) {
  # Remove units in parentheses, e.g. "ALT(IU/L)" -> "ALT".
  gsub("\\([^)]*\\)", "", x)
}


# ----------------------------- Plotting function ------------------------------

plot_fp_vs_fn <- function(df, output_file) {
  if (nrow(df) == 0) {
    stop("No focused endpoints were found in the concordance table.")
  }

  # Coerce confusion-matrix counts to numeric.
  df$FP <- suppressWarnings(as.numeric(df$FP))
  df$FN <- suppressWarnings(as.numeric(df$FN))

  if (!any(is.finite(df$FP)) || !any(is.finite(df$FN))) {
    stop("FP and FN columns do not contain usable numeric values.")
  }

  # Preserve the requested endpoint order.
  df$feature <- factor(df$feature, levels = FOCUSED_ENDPOINTS)
  df <- df[order(df$feature), , drop = FALSE]
  df$short_feature <- short_feature_name(as.character(df$feature))

  # Axes start at zero and receive upper padding for labels.
  x_max <- max(df$FP, na.rm = TRUE) + X_PADDING
  y_max <- max(df$FN, na.rm = TRUE) + Y_PADDING

  if (!is.finite(x_max) || x_max <= 0) x_max <- 1
  if (!is.finite(y_max) || y_max <= 0) y_max <- 1

  xlim <- c(0, x_max)
  ylim <- c(0, y_max)

  # Default label placement: slightly above each point.
  df$label_x <- df$FP
  df$label_y <- df$FN + 0.90

  # Manual adjustments for known overlapping labels.
  dbil_idx <- which(df$short_feature == "DBIL")
  if (length(dbil_idx) > 0) {
    df$label_x[dbil_idx] <- df$FP[dbil_idx] - 1.4
    df$label_y[dbil_idx] <- df$FN[dbil_idx] + 1.1
  }

  cre_idx <- which(df$short_feature == "CRE")
  if (length(cre_idx) > 0) {
    df$label_x[cre_idx] <- df$FP[cre_idx] + 1.2
    df$label_y[cre_idx] <- df$FN[cre_idx] - 0.9
  }

  tiff(
    filename = output_file,
    width = 11,
    height = 9,
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
    mar = c(5.8, 5.8, 3.4, 2.4),
    las = 1,
    family = FONT_FAMILY,
    cex.axis = 1.15,
    tck = -0.012,
    mgp = c(3.1, 0.8, 0)
  )

  plot(
    df$FP,
    df$FN,
    xlim = xlim,
    ylim = ylim,
    type = "n",
    xlab = "",
    ylab = "",
    main = "",
    bty = "n",
    xaxs = "i",
    yaxs = "i",
    axes = FALSE
  )

  # Shade the two sides of the FP = FN diagonal. The polygons are constructed
  # using the actual rectangular plotting limits, so they remain correct even
  # when the x- and y-axis maxima differ.
  diag_end <- min(xlim[2], ylim[2])

  # FP > FN region (below diagonal).
  polygon(
    x = c(0, xlim[2], xlim[2], diag_end),
    y = c(0, 0, min(ylim[2], xlim[2]), diag_end),
    col = rgb(0.85, 0.35, 0.30, 0.11),
    border = NA
  )

  # FN > FP region (above diagonal).
  polygon(
    x = c(0, 0, min(xlim[2], ylim[2]), diag_end),
    y = c(0, ylim[2], ylim[2], diag_end),
    col = rgb(0.25, 0.45, 0.75, 0.11),
    border = NA
  )

  x_ticks <- pretty(xlim)
  x_ticks <- x_ticks[x_ticks >= 0 & x_ticks <= xlim[2]]

  y_ticks <- pretty(ylim)
  y_ticks <- y_ticks[y_ticks >= 0 & y_ticks <= ylim[2]]

  # White grid lines over the shaded background.
  abline(
    h = y_ticks,
    v = x_ticks,
    col = "white",
    lty = "solid",
    lwd = 1.0
  )

  # FP = FN reference line.
  abline(
    a = 0,
    b = 1,
    lty = 2,
    lwd = 2.3,
    col = "gray20"
  )

  text(
    x = xlim[2] * 0.78,
    y = ylim[2] * 0.48,
    labels = "FP > FN",
    cex = 1.18,
    font = 2,
    family = FONT_FAMILY,
    col = rgb(0.55, 0.15, 0.12)
  )

  text(
    x = xlim[2] * 0.17,
    y = ylim[2] * 0.78,
    labels = "FN > FP",
    cex = 1.18,
    font = 2,
    family = FONT_FAMILY,
    col = rgb(0.12, 0.25, 0.55)
  )

  points(
    df$FP,
    df$FN,
    pch = 21,
    bg = "black",
    col = "white",
    cex = POINT_CEX,
    lwd = 0.9
  )

  text(
    df$label_x,
    df$label_y,
    labels = df$short_feature,
    cex = LABEL_CEX,
    font = 2,
    family = FONT_FAMILY,
    col = "black"
  )

  axis(1, at = x_ticks, lwd = 1.2, lwd.ticks = 1.1)
  axis(2, at = y_ticks, lwd = 1.2, lwd.ticks = 1.1)
  box(bty = "l", lwd = 1.2)

  mtext(
    "False positives (FP)",
    side = 1,
    line = 3.8,
    cex = 1.35,
    font = 2,
    family = FONT_FAMILY
  )

  mtext(
    "False negatives (FN)",
    side = 2,
    line = 4.0,
    cex = 1.35,
    font = 2,
    family = FONT_FAMILY,
    las = 0
  )

  mtext(
    "False positives versus false negatives",
    side = 3,
    line = 1,
    cex = 1.35,
    font = 2,
    family = FONT_FAMILY
  )

  # Place the diagonal annotation within the visible plotting region.
  label_pos <- 0.58 * diag_end
  text(
    x = label_pos - 0.8,
    y = label_pos,
    labels = "FP = FN",
    cex = 0.98,
    font = 2,
    family = FONT_FAMILY,
    col = "gray20",
    srt = 45
  )
}


# ---------------------------------- Main ---------------------------------------

main <- function() {
  if (!file.exists(INPUT_FILE)) {
    stop("Input file not found: ", INPUT_FILE)
  }

  concordance_df <- read.csv(
    INPUT_FILE,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  require_columns(
    concordance_df,
    c("feature", "FP", "FN"),
    object_name = INPUT_FILE
  )

  focused_df <- concordance_df[
    concordance_df$feature %in% FOCUSED_ENDPOINTS,
    ,
    drop = FALSE
  ]

  missing_endpoints <- setdiff(FOCUSED_ENDPOINTS, focused_df$feature)
  if (length(missing_endpoints) > 0) {
    warning(
      "The following focused endpoints were not found and will not be plotted: ",
      paste(missing_endpoints, collapse = ", ")
    )
  }

  plot_fp_vs_fn(focused_df, OUTPUT_FILE)

  message("Saved: ", normalizePath(OUTPUT_FILE, winslash = "/", mustWork = FALSE))
}


if (sys.nframe() == 0) {
  main()
}
