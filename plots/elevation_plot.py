# ================================
# Setup
# ================================
# install.packages(c("tidyverse")) # if needed
library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(tibble)


# ---------- Paths (edit if needed) ----------
path_control   <- "/account001/mansi.chandra/clin_path/repeat_test_control_cv2.csv"
path_treatment <- "/account001/mansi.chandra/clin_path/repeat_test_treatment_cv2.csv"
path_gen       <- "/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/predictions_decoded/test/generated_predictions_1507935_ControlGenerator_test.csv"



# ================================
# Load data
# ================================
control  <- suppressMessages(read_csv(path_control, show_col_types = FALSE))
treatment<- suppressMessages(read_csv(path_treatment, show_col_types = FALSE))
real     <- bind_rows(control, treatment)

gen      <- suppressMessages(read_csv(path_gen, show_col_types = FALSE))

# ================================
# Rounding / harmonization like Python
# (silently skip columns that aren't present)
# ================================
round_if <- function(df, col, digits = 0, as_int = FALSE) {
  if (!col %in% names(df)) return(df)
  v <- suppressWarnings(as.numeric(df[[col]]))
  v <- round(v, digits)
  if (as_int) v <- as.integer(v)
  df[[col]] <- v
  df
}

gen <- gen %>%
  round_if("TBIL(mg/dL)", 2) %>%
  round_if("RALB(g/dL)", 1) %>%
  round_if("AST(IU/L)", 0, as_int = TRUE) %>%
  round_if("TP(g/dL)", 1) %>%
  round_if("DBIL(mg/dL)", 2) %>%
  round_if("BUN(mg/dL)", 0, as_int = TRUE) %>%
  round_if("ALP(IU/L)", 0, as_int = TRUE) %>%
  round_if("ALT(IU/L)", 0, as_int = TRUE) %>%
  round_if("LDH(IU/L)", 0, as_int = TRUE) %>%
  round_if("RALB(g/dL)", 1) # (duplicated in your Python; harmless)

# ================================
# Build cohorts (match Python)
# ================================
control_df  <- real %>% filter(.data[["DOSE_LEVEL"]] == "Control")
treat_df    <- real %>% filter(.data[["DOSE_LEVEL"]] == "High")
generated   <- gen  %>% filter(.data[["DOSE_LEVEL"]] == "High")


# ================================
# Parameters
# ================================
compound_name  <- "gentamicin"
sacrifice_time <- "15 day"
endpoint_col   <- "CRE(mg/dL)"   # <-- change if CRE is named differently

# ================================
# Subset data
# ================================
control_sub <- subset(
  control_df,
  COMPOUND_NAME == compound_name & SACRIFICE_PERIOD == sacrifice_time
)

treat_sub <- subset(
  treat_df,
  COMPOUND_NAME == compound_name & SACRIFICE_PERIOD == sacrifice_time
)

gen_sub <- subset(
  generated,
  COMPOUND_NAME == compound_name & SACRIFICE_PERIOD == sacrifice_time
)

# Ensure numeric and drop NAs
control_vals <- as.numeric(control_sub[[endpoint_col]])
treat_vals   <- as.numeric(treat_sub[[endpoint_col]])
gen_vals     <- as.numeric(gen_sub[[endpoint_col]])

control_vals <- control_vals[!is.na(control_vals)]
treat_vals   <- treat_vals[!is.na(treat_vals)]
gen_vals     <- gen_vals[!is.na(gen_vals)]

# ================================
# Robust Welch t-test
# ================================
safe_pvalue <- function(x, y) {
  x <- x[!is.na(x)]
  y <- y[!is.na(y)]

  nx <- length(x)
  ny <- length(y)

  if (nx < 2 || ny < 2) return(1)

  mx <- mean(x); my <- mean(y)
  sx <- sd(x);   sy <- sd(y)

  if (sx == 0 && sy == 0) {
    if (isTRUE(all.equal(mx, my))) {
      return(1)
    } else {
      return(1e-16)
    }
  }

  vx <- sx^2
  vy <- sy^2
  se <- sqrt(vx / nx + vy / ny)
  if (se == 0) return(1)

  t_stat <- (mx - my) / se

  df_num <- (vx / nx + vy / ny)^2
  df_den <- (vx^2 / (nx^2 * (nx - 1))) + (vy^2 / (ny^2 * (ny - 1)))
  if (df_den == 0) return(1)

  df <- df_num / df_den
  2 * pt(-abs(t_stat), df)
}

# ================================
# T-tests (using safe helper)
# ================================
p_treat_control <- safe_pvalue(treat_vals, control_vals)
p_treat_gen     <- safe_pvalue(treat_vals, gen_vals)

format_p_label <- function(p) {
  if (p < 0.05) {
    "p < 0.05"
  } else {
    paste0("p = ", signif(p, 2))
  }
}

label1 <- format_p_label(p_treat_control)
label2 <- format_p_label(p_treat_gen)

# ================================
# Summary stats for barplot
# ================================
# middle label left blank; we'll draw "Real" and "control" manually
group_names <- c("Treatment", "", "GanCtrl")

means <- c(
  mean(treat_vals),
  mean(control_vals),
  mean(gen_vals)
)

sds <- c(
  sd(treat_vals),
  sd(control_vals),
  sd(gen_vals)
)

y_base <- max(means + sds, na.rm = TRUE)
y1 <- y_base * 1.15
y2 <- y_base * 1.35
ylim_top <- y2 * 1.30

bar_cols <- c("#4C72B0", "#DD8452", "#55A868")

# ================================
# Plot and save as TIFF (600 dpi)
# ================================
tiff(
  filename = "/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/performance/gentamicin_CRE_15day_barplot_baseR.tif",
  width  = 5,
  height = 6,
  units  = "in",
  res    = 600,
  compression = "lzw"
)

par(mar = c(5, 5, 4, 2) + 0.1, font.lab = 2)

bar_centers <- barplot(
  height = means,
  names.arg = group_names,  # <-- middle is blank
  ylim = c(0, ylim_top),
  ylab = endpoint_col,
  xlab = "",
  main = "",
  col = bar_cols,
  border = NA,
  width = 0.5,
  axes = FALSE
)

# y-axis ticks only up to 0.4
axis_ticks <- seq(0, 0.4, by = 0.1)
axis(
  side = 2,
  at = axis_ticks,
  labels = axis_ticks,
  las = 1
)

box(lwd = 1)

# ================================
# Custom x-axis label for Real / control
# ================================
x_treat <- bar_centers[1]
x_ctrl  <- bar_centers[2]
x_gen   <- bar_centers[3]

# Add "Real" on same line as Treatment & SynCtrl, and "control" below it
mtext("Real",    side = 1, at = x_ctrl, line = 1, font = 1)
mtext("control", side = 1, at = x_ctrl, line = 2, font = 1)

# ================================
# P-value brackets
# ================================
vlen_fraction <- 0.08
vlen1 <- vlen_fraction * y_base
vlen2 <- vlen_fraction * y_base
label_offset <- 0.07 * y_base

# Treatment vs Real control
segments(x_treat, y1, x_ctrl,  y1)
segments(x_treat, y1 - vlen1, x_treat, y1)
segments(x_ctrl,  y1 - vlen1, x_ctrl,  y1)
text(
  (x_treat + x_ctrl) / 2,
  y1 + label_offset,
  labels = label1,
  font = 2
)

# Treatment vs SynCtrl
segments(x_treat, y2, x_gen,  y2)
segments(x_treat, y2 - vlen2, x_treat, y2)
segments(x_gen,   y2 - vlen2, x_gen,  y2)
text(
  (x_treat + x_gen) / 2,
  y2 + label_offset,
  labels = label2,
  font = 2
)

# ================================
# Title inside the plot + separating line
# ================================
usr <- par("usr")
x_left  <- usr[1]
x_right <- usr[2]

title_y <- ylim_top * 0.95
line_y  <- ylim_top * 0.90

text(
  x = (x_left + x_right) / 2,
  y = title_y,
  labels = "gentamicin - 15 day",
  font = 2
)

segments(x_left, line_y, x_right, line_y)

box(lwd = 1.5)

dev.off()





# ================================
# Parameters
# ================================
compound_name  <- "nitrosodiethylamine"
sacrifice_time <- "4 day"
endpoint_col   <- "ALT(IU/L)"   # <-- change if ALT is named differently

# ================================
# Subset data
# ================================
control_sub <- subset(
  control_df,
  COMPOUND_NAME == compound_name & SACRIFICE_PERIOD == sacrifice_time
)

treat_sub <- subset(
  treat_df,
  COMPOUND_NAME == compound_name & SACRIFICE_PERIOD == sacrifice_time
)

gen_sub <- subset(
  generated,
  COMPOUND_NAME == compound_name & SACRIFICE_PERIOD == sacrifice_time
)

# Ensure numeric and drop NAs
control_vals <- as.numeric(control_sub[[endpoint_col]])
treat_vals   <- as.numeric(treat_sub[[endpoint_col]])
gen_vals     <- as.numeric(gen_sub[[endpoint_col]])

control_vals <- control_vals[!is.na(control_vals)]
treat_vals   <- treat_vals[!is.na(treat_vals)]
gen_vals     <- gen_vals[!is.na(gen_vals)]

# ================================
# Robust Welch t-test
# ================================
safe_pvalue <- function(x, y) {
  x <- x[!is.na(x)]
  y <- y[!is.na(y)]

  nx <- length(x)
  ny <- length(y)

  if (nx < 2 || ny < 2) return(1)

  mx <- mean(x); my <- mean(y)
  sx <- sd(x);   sy <- sd(y)

  if (sx == 0 && sy == 0) {
    if (isTRUE(all.equal(mx, my))) {
      return(1)
    } else {
      return(1e-16)
    }
  }

  vx <- sx^2
  vy <- sy^2
  se <- sqrt(vx / nx + vy / ny)
  if (se == 0) return(1)

  t_stat <- (mx - my) / se

  df_num <- (vx / nx + vy / ny)^2
  df_den <- (vx^2 / (nx^2 * (nx - 1))) + (vy^2 / (ny^2 * (ny - 1)))
  if (df_den == 0) return(1)

  df <- df_num / df_den
  2 * pt(-abs(t_stat), df)
}

# ================================
# T-tests (using safe helper)
# ================================
p_treat_control <- safe_pvalue(treat_vals, control_vals)
p_treat_gen     <- safe_pvalue(treat_vals, gen_vals)

format_p_label <- function(p) {
  if (p < 0.05) {
    "p < 0.05"
  } else {
    paste0("p = ", signif(p, 2))
  }
}

label1 <- format_p_label(p_treat_control)
label2 <- format_p_label(p_treat_gen)

# ================================
# Summary stats for barplot
# ================================
group_names <- c("Treatment", "", "GanCtrl")

means <- c(
  mean(treat_vals),
  mean(control_vals),
  mean(gen_vals)
)

sds <- c(
  sd(treat_vals),
  sd(control_vals),
  sd(gen_vals)
)

# Guard: ensure we actually have data
y_base <- max(means + sds, na.rm = TRUE)
if (!is.finite(y_base)) {
  stop("No non-missing values for this compound / time / endpoint. ",
       "Check 'endpoint_col' and data for nitrosodiethylamine, 4 day, ALT.")
}

# ---- Fixed y-axis limit ----
ylim_top <- 105

# Place brackets at reasonable heights within [0, 120]
y1 <- ylim_top * 0.70    # Treatment vs Real control bracket height
y2 <- ylim_top * 0.79    # Treatment vs SynCtrl bracket height

# Colors
bar_cols <- c("#4C72B0", "#DD8452", "#55A868")

# ================================
# Plot and save as TIFF (600 dpi)
# ================================
tiff(
  filename = "/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/performance/nitrosodiethylamine_ALT_4day_barplot_baseR.tif",
  width  = 5,
  height = 6,
  units  = "in",
  res    = 600,
  compression = "lzw"
)

par(mar = c(5, 5, 4, 2) + 0.1, font.lab = 2)

# Create barplot (no main, narrower bars, custom y-axis)
bar_centers <- barplot(
  height = means,
  names.arg = group_names,  # middle blank; we'll add Real/control manually
  ylim = c(0, ylim_top),
  ylab = endpoint_col,
  xlab = "",
  main = "",
  col = bar_cols,
  border = NA,
  width = 0.5,
  axes = FALSE              # we'll draw y-axis manually
)

# ---- Custom y-axis: ticks and labels only up to 80 ----
axis_ticks <- seq(0, 80, by = 20)
axis(
  side   = 2,
  at     = axis_ticks,
  labels = axis_ticks,
  las    = 1
)

box(lwd = 1)

# ================================
# Custom x-axis label for Real / control (two lines)
# ================================
x_treat <- bar_centers[1]
x_ctrl  <- bar_centers[2]
x_gen   <- bar_centers[3]

mtext("Real",    side = 1, at = x_ctrl, line = 1, font = 1)
mtext("control", side = 1, at = x_ctrl, line = 2, font = 1)

# ================================
# P-value brackets
# ================================
vlen_fraction <- 0.05
vlen1 <- vlen_fraction * ylim_top
vlen2 <- vlen_fraction * ylim_top
label_offset <- 0.04 * ylim_top

# Treatment vs Real control
segments(x_treat, y1, x_ctrl,  y1)
segments(x_treat, y1 - vlen1, x_treat, y1)
segments(x_ctrl,  y1 - vlen1, x_ctrl,  y1)
text(
  (x_treat + x_ctrl) / 2,
  y1 + label_offset,
  labels = label1,
  font = 2
)

# Treatment vs SynCtrl
segments(x_treat, y2, x_gen,  y2)
segments(x_treat, y2 - vlen2, x_treat, y2)
segments(x_gen,   y2 - vlen2, x_gen,  y2)
text(
  (x_treat + x_gen) / 2,
  y2 + label_offset,
  labels = label2,
  font = 2
)

# ================================
# Title inside the plot + separating line
# ================================
usr <- par("usr")
x_left  <- usr[1]
x_right <- usr[2]

title_y <- ylim_top * 0.95
line_y  <- ylim_top * 0.90

text(
  x = (x_left + x_right) / 2,
  y = title_y,
  labels = "nitrosodiethylamine - 4 day",
  font = 2
)

segments(x_left, line_y, x_right, line_y)

box(lwd = 1.5)

dev.off()










