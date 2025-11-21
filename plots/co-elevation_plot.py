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
path_gen       <- "/account001/mansi.chandra/clin_path/results_vae_corr_mod3_cv2/predictions_decoded/test/generated_predictions_1161985_ControlGenerator_test.csv"



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


## ================================================================
## ALT/AST two-panel barplot (base R)
## - x-axis labels regular (not bold)
## - bold panel boxes
## - title spacing improved
## - 600 dpi TIFF, adjustable brackets, fixed y-lims
## ================================================================

BAR_WIDTH            <- 0.33
LABEL_GAP_FR_1      <- 0.06
LABEL_GAP_FR_2      <- 0.11
BRACKET_BASE_FR      <- 0.70
BRACKET_GAP_BETWEEN  <- 0.09

ARM_TR_REAL_LEFT_FR  <- 0.05
ARM_TR_REAL_RIGHT_FR <- 0.05
ARM_TR_GEN_LEFT_FR   <- 0.06
ARM_TR_GEN_RIGHT_FR  <- 0.05

AST_BRACKET_SHIFT_FR <- 0.003

STRIP_LINE_OFFSET_FR <- 0.065
STRIP_TEXT_OFFSET_FR <- 0.032
STRIP_LINE_LWD       <- 2.4
STRIP_TEXT_CEX       <- 1.2
BOX_LWD              <- 2.0

colpick <- function(df, candidates){
  n <- names(df); ln <- tolower(n)
  for (c in candidates) { i <- which(ln == tolower(c)); if (length(i)) return(n[i[1]]) }
  stop("Column not found: ", paste(candidates, collapse = "/"))
}
featurepick <- function(df, preferred, fallback){
  n <- names(df); if (preferred %in% n) return(preferred); if (fallback %in% n) return(fallback)
  stop("Feature column not found: ", preferred, " or ", fallback)
}
subset_by_ct <- function(df, comp_col, time_col, compound, time_val){
  df[ trimws(tolower(df[[comp_col]])) == trimws(tolower(compound)) &
      trimws(tolower(df[[time_col]])) == trimws(tolower(time_val)), , drop = FALSE ]
}
mean_na <- function(x) mean(x, na.rm = TRUE)
fmt_p_thresh <- function(p){
  if (is.na(p)) "p=NA" else if (p < 0.05) "p<0.05" else paste0("p=", formatC(p, format="f", digits=3))
}
add_sq_bracket_lr <- function(x1, x2, y_top, arm_left, arm_right, label = "", cex = 0.9){
  segments(x1, y_top - arm_left,  x1, y_top)
  segments(x2, y_top - arm_right, x2, y_top)
  segments(x1, y_top, x2, y_top)
  text((x1 + x2)/2, y_top, labels = label, pos = 3, cex = cex, font = 2)
}
panel_strip <- function(label, cex = STRIP_TEXT_CEX){
  usr <- par("usr"); x1 <- usr[1]; x2 <- usr[2]; y1 <- usr[3]; y2 <- usr[4]
  y_line <- y2 - STRIP_LINE_OFFSET_FR * (y2 - y1)
  segments(x1, y_line, x2, y_line, xpd = NA, lwd = STRIP_LINE_LWD)
  y_txt  <- y2 - STRIP_TEXT_OFFSET_FR * (y2 - y1)
  text((x1 + x2)/2, y_txt, labels = label, font = 2, cex = cex, xpd = NA)
}
draw_two_line_labels <- function(at, line1, line2, cex = 0.95){
  usr <- par("usr"); y1 <- usr[3]; y2 <- usr[4]; span <- y2 - y1
  axis(1, at = at, labels = FALSE, tck = -0.01)
  text(at, y1 - LABEL_GAP_FR_1 * span, labels = line1, xpd = NA, cex = cex, font = 1)
  text(at, y1 - LABEL_GAP_FR_2 * span, labels = line2, xpd = NA, cex = cex, font = 1)
}

extract_unit <- function(colname, fallback = "U/L"){
  m <- regmatches(colname, regexpr("\\(([^)]+)\\)", colname))
  if (length(m) && nzchar(m)) paste0(" ", m) else paste0(" (", fallback, ")")
}
ucfirst <- function(s){
  if (!nzchar(s)) return(s)
  paste0(toupper(substring(s, 1, 1)), substring(s, 2))
}

comp_col <- colpick(treat_df, c("COMPOUND_NAME","COMPOUND","compound"))
time_col <- colpick(treat_df,  c("SACRIFICE_PERIOD","TIME","time","PERIOD"))
ALT_col  <- featurepick(treat_df, "ALT(IU/L)", "ALT")
AST_col  <- featurepick(treat_df, "AST(IU/L)", "AST")

ALT_label <- paste0("ALT", extract_unit(ALT_col))
AST_label <- paste0("AST", extract_unit(AST_col))

compound_lbl <- "nitrosodiethylamine"
time_lbl     <- "8 day"

treat_alt <- subset_by_ct(treat_df,   comp_col, time_col, compound_lbl, time_lbl)[[ALT_col]]
ctrl_alt  <- subset_by_ct(control_df, comp_col, time_col, compound_lbl, time_lbl)[[ALT_col]]
gen_alt   <- subset_by_ct(generated,  comp_col, time_col, compound_lbl, time_lbl)[[ALT_col]]

treat_ast <- subset_by_ct(treat_df,   comp_col, time_col, compound_lbl, time_lbl)[[AST_col]]
ctrl_ast  <- subset_by_ct(control_df, comp_col, time_col, compound_lbl, time_lbl)[[AST_col]]
gen_ast   <- subset_by_ct(generated,  comp_col, time_col, compound_lbl, time_lbl)[[AST_col]]

p_alt_trt_vs_real <- tryCatch(t.test(treat_alt, ctrl_alt, alternative="greater")$p.value, error=function(e) NA_real_)
p_alt_trt_vs_gen  <- tryCatch(t.test(treat_alt, gen_alt,  alternative="greater")$p.value, error=function(e) NA_real_)
p_ast_trt_vs_real <- tryCatch(t.test(treat_ast, ctrl_ast, alternative="greater")$p.value, error=function(e) NA_real_)
p_ast_trt_vs_gen  <- tryCatch(t.test(treat_ast, gen_ast,  alternative="greater")$p.value, error=function(e) NA_real_)

alt_means <- c(mean_na(treat_alt), mean_na(ctrl_alt), mean_na(gen_alt))
ast_means <- c(mean_na(treat_ast), mean_na(ctrl_ast), mean_na(gen_ast))
cols      <- c("#0072B2", "#E69F00", "#009E73") 

out_dir  <- getOption("plot.outdir", getwd())
out_file <- file.path(out_dir, "nitrosodiethylamine_8day_ALT_AST.tif")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ok <- FALSE
tryCatch({
  tiff(out_file, width = 8.5, height = 7.0, units = "in", res = 600, compression = "lzw")
  op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)

  par(mfrow = c(1, 2), mar = c(5.8, 4.1, 1.4, 1.2), oma = c(0, 0, 2.6, 0))

  ## ===================== ALT (0–250) =====================
  bp1 <- barplot(
    alt_means, col = cols, ylim = c(0, 250),
    ylab = "", xaxt = "n", yaxt = "n", width = BAR_WIDTH, border = NA
  )
  box(lwd = BOX_LWD)
  panel_strip(ALT_label)

  y_ticks_alt <- seq(0, 250, by = 50); y_ticks_alt <- y_ticks_alt[y_ticks_alt < 250]
  axis(2, at = y_ticks_alt, labels = y_ticks_alt, las = 1, tck = -0.015)

  # CHANGED: "Generated" -> "SynCtrl"
  draw_two_line_labels(bp1, c("Treatment","Real","GanCtrl"), c("","Control",""))

  usr1  <- par("usr"); span1 <- usr1[4] - usr1[3]
  ytop1a <- BRACKET_BASE_FR * usr1[4]
  ytop1b <- ytop1a + BRACKET_GAP_BETWEEN * span1

  add_sq_bracket_lr(bp1[1], bp1[2], y_top = ytop1a,
                    arm_left = ARM_TR_REAL_LEFT_FR * span1,
                    arm_right = ARM_TR_REAL_RIGHT_FR * span1,
                    label = fmt_p_thresh(p_alt_trt_vs_real), cex = 0.92)

  add_sq_bracket_lr(bp1[1], bp1[3], y_top = ytop1b,
                    arm_left = ARM_TR_GEN_LEFT_FR * span1,
                    arm_right = ARM_TR_GEN_RIGHT_FR * span1,
                    label = fmt_p_thresh(p_alt_trt_vs_gen),  cex = 0.92)

  ## ===================== AST (0–500) =====================
  bp2 <- barplot(
    ast_means, col = cols, ylim = c(0, 500),
    ylab = "", xaxt = "n", yaxt = "n", width = BAR_WIDTH, border = NA
  )
  box(lwd = BOX_LWD)
  panel_strip(AST_label)

  y_ticks_ast <- seq(0, 500, by = 100); y_ticks_ast <- y_ticks_ast[y_ticks_ast < 500]
  axis(2, at = y_ticks_ast, labels = y_ticks_ast, las = 1, tck = -0.015)

  # CHANGED: "Generated" -> "SynCtrl"
  draw_two_line_labels(bp2, c("Treatment","Real","GanCtrl"), c("","Control",""))

  usr2  <- par("usr"); span2 <- usr2[4] - usr2[3]
  base_ast   <- BRACKET_BASE_FR * usr2[4] - AST_BRACKET_SHIFT_FR * span2
  ytop2a     <- base_ast
  ytop2b     <- base_ast + BRACKET_GAP_BETWEEN * span2

  add_sq_bracket_lr(bp2[1], bp2[2], y_top = ytop2a,
                    arm_left = ARM_TR_REAL_LEFT_FR * span2,
                    arm_right = ARM_TR_REAL_RIGHT_FR * span2,
                    label = fmt_p_thresh(p_ast_trt_vs_real), cex = 0.92)

  add_sq_bracket_lr(bp2[1], bp2[3], y_top = ytop2b,
                    arm_left = ARM_TR_GEN_LEFT_FR * span2,
                    arm_right = ARM_TR_GEN_RIGHT_FR * span2,
                    label = fmt_p_thresh(p_ast_trt_vs_gen),  cex = 0.92)

  title_str <- paste0(compound_lbl, " — ", time_lbl)
  mtext(ucfirst(title_str), outer = TRUE, cex = 1.25, line = 0.9, font = 2)

  ok <- TRUE
},
error = function(e) message("Plotting error: ", conditionMessage(e)),
finally = { if (length(dev.list())) try(invisible(dev.off()), silent = TRUE) })

if (ok && file.exists(out_file)) {
  message("Saved: ", normalizePath(out_file, winslash = "/"))
} else {
  stop("TIFF not saved. Check write permissions for: ", out_dir, "\nWorking dir: ", getwd())
}


## ================================================================
## BUN/CRE two-panel barplot (base R) — Nitrosodiethylamine, 8 day
## (BUN ylim 0–24 w/ ticks 0..20 by 4; CRE ylim 0–0.6 w/ ticks 0..0.5 by 0.1)
## ================================================================

BAR_WIDTH              <- 0.33
LABEL_GAP_FR_1        <- 0.06
LABEL_GAP_FR_2        <- 0.11
BRACKET_BASE_FR        <- 0.70
BRACKET_GAP_BETWEEN    <- 0.09
ARM_TR_REAL_LEFT_FR    <- 0.06
ARM_TR_REAL_RIGHT_FR   <- 0.06
ARM_TR_GEN_LEFT_FR     <- 0.06
ARM_TR_GEN_RIGHT_FR    <- 0.06
PANEL2_BRACKET_SHIFT_FR<- 0.003
STRIP_LINE_OFFSET_FR   <- 0.065
STRIP_TEXT_OFFSET_FR   <- 0.032
STRIP_LINE_LWD         <- 2.4
STRIP_TEXT_CEX         <- 1.2
BOX_LWD                <- 2.0

## ---- NEW: upward shift for CRE brackets (fraction of y-span) ----
CRE_BRACKET_UP_FR      <- 0.03

## ---- explicit y-range + tick ceilings ----
BUN_YLIM   <- c(0, 24)
BUN_TICKS  <- seq(0, 20, by = 4)
CRE_YLIM   <- c(0, 0.64)
CRE_TICKS  <- seq(0, 0.5, by = 0.1)

# ---------- helpers ----------
colpick <- function(df, candidates){
  n <- names(df); ln <- tolower(n)
  for (c in candidates) { i <- which(ln == tolower(c)); if (length(i)) return(n[i[1]]) }
  stop("Column not found: ", paste(candidates, collapse = "/"))
}
featurepick <- function(df, preferred, fallback){
  n <- names(df); if (preferred %in% n) return(preferred); if (fallback %in% n) return(fallback)
  stop("Feature column not found: ", preferred, " or ", fallback)
}
subset_by_ct <- function(df, comp_col, time_col, compound, time_val){
  df[ trimws(tolower(df[[comp_col]])) == trimws(tolower(compound)) &
      trimws(tolower(df[[time_col]])) == trimws(tolower(time_val)), , drop = FALSE ]
}
mean_na <- function(x) mean(x, na.rm = TRUE)

fmt_p_label <- function(p, thr = 0.05){
  if (is.na(p)) "p=NA" else if (p < thr) "p<0.05" else paste0("p=", formatC(p, format = "f", digits = 3))
}

add_sq_bracket_lr <- function(x1, x2, y_top, arm_left, arm_right, label = "", cex = 0.9){
  segments(x1, y_top - arm_left,  x1, y_top)
  segments(x2, y_top - arm_right, x2, y_top)
  segments(x1, y_top, x2, y_top)
  if (nzchar(label)) text((x1 + x2)/2, y_top, labels = label, pos = 3, cex = cex, font = 2)
}
panel_strip <- function(label, cex = STRIP_TEXT_CEX){
  usr <- par("usr"); x1 <- usr[1]; x2 <- usr[2]; y1 <- usr[3]; y2 <- usr[4]
  y_line <- y2 - STRIP_LINE_OFFSET_FR * (y2 - y1)
  segments(x1, y_line, x2, y_line, xpd = NA, lwd = STRIP_LINE_LWD)
  y_txt  <- y2 - STRIP_TEXT_OFFSET_FR * (y2 - y1)
  text((x1 + x2)/2, y_txt, labels = label, font = 2, cex = cex, xpd = NA)
}
draw_two_line_labels <- function(at, line1, line2, cex = 0.95){
  usr <- par("usr"); y1 <- usr[3]; y2 <- usr[4]; span <- y2 - y1
  axis(1, at = at, labels = FALSE, tck = -0.01)
  text(at, y1 - LABEL_GAP_FR_1 * span, labels = line1, xpd = NA, cex = cex, font = 1)
  text(at, y1 - LABEL_GAP_FR_2 * span, labels = line2, xpd = NA, cex = cex, font = 1)
}
extract_unit <- function(colname, fallback = "mg/dL"){
  m <- regmatches(colname, regexpr("\\(([^)]+)\\)", colname))
  if (length(m) && nzchar(m)) paste0(" ", m) else paste0(" (", fallback, ")")
}
ucfirst <- function(s){
  if (!nzchar(s)) return(s)
  paste0(toupper(substring(s, 1, 1)), substring(s, 2))
}

# ---------- columns ----------
comp_col <- colpick(treat_df, c("COMPOUND_NAME","COMPOUND","compound"))
time_col <- colpick(treat_df,  c("SACRIFICE_PERIOD","TIME","time","PERIOD"))
BUN_col  <- featurepick(treat_df, "BUN(mg/dL)", "BUN")
CRE_col  <- featurepick(treat_df, "CRE(mg/dL)", "CRE")

BUN_label <- paste0("BUN", extract_unit(BUN_col, "mg/dL"))
CRE_label <- paste0("CRE", extract_unit(CRE_col, "mg/dL"))

# ---------- cohort ----------
compound_lbl <- "nitrosodiethylamine"
time_lbl     <- "8 day"

treat_bun <- subset_by_ct(treat_df,   comp_col, time_col, compound_lbl, time_lbl)[[BUN_col]]
ctrl_bun  <- subset_by_ct(control_df, comp_col, time_col, compound_lbl, time_lbl)[[BUN_col]]
gen_bun   <- subset_by_ct(generated,  comp_col, time_col, compound_lbl, time_lbl)[[BUN_col]]

treat_cre <- subset_by_ct(treat_df,   comp_col, time_col, compound_lbl, time_lbl)[[CRE_col]]
ctrl_cre  <- subset_by_ct(control_df, comp_col, time_col, compound_lbl, time_lbl)[[CRE_col]]
gen_cre   <- subset_by_ct(generated,  comp_col, time_col, compound_lbl, time_lbl)[[CRE_col]]

# ---------- stats (Treatment > Control) ----------
p_bun_trt_vs_real <- tryCatch(t.test(treat_bun, ctrl_bun, alternative="greater")$p.value, error=function(e) NA_real_)
p_bun_trt_vs_gen  <- tryCatch(t.test(treat_bun, gen_bun,  alternative="greater")$p.value, error=function(e) NA_real_)
p_cre_trt_vs_real <- tryCatch(t.test(treat_cre, ctrl_cre, alternative="greater")$p.value, error=function(e) NA_real_)
p_cre_trt_vs_gen  <- tryCatch(t.test(treat_cre, gen_cre,  alternative="greater")$p.value, error=function(e) NA_real_)

# ---------- bars (means) ----------
bun_means <- c(mean_na(treat_bun), mean_na(ctrl_bun), mean_na(gen_bun))
cre_means <- c(mean_na(treat_cre), mean_na(ctrl_cre), mean_na(gen_cre))
cols      <- c("#0072B2", "#E69F00", "#009E73") 

# ---------- TIFF 600 dpi ----------
out_dir  <- getOption("plot.outdir", getwd())
out_file <- file.path(out_dir, "nitrosodiethylamine_8day_BUN_CRE.tif")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ok <- FALSE
tryCatch({
  tiff(out_file, width = 8.5, height = 7.0, units = "in", res = 600, compression = "lzw")
  op <- par(no.readonly = TRUE); on.exit(par(op), add = TRUE)
  par(mfrow = c(1, 2), mar = c(5.8, 4.1, 1.4, 1.2), oma = c(0, 0, 2.6, 0))

  ## ===================== BUN =====================
  bp1 <- barplot(
    bun_means, col = cols, ylim = BUN_YLIM,
    ylab = "", xaxt = "n", yaxt = "n", width = BAR_WIDTH, border = NA
  )
  box(lwd = BOX_LWD)
  panel_strip(BUN_label)

  axis(2, at = BUN_TICKS, labels = BUN_TICKS, las = 1, tck = -0.015)

  # CHANGED: "Generated" -> "SynCtrl"
  draw_two_line_labels(bp1, c("Treatment","Real","GanCtrl"), c("","Control",""))

  usr1  <- par("usr"); span1 <- usr1[4] - usr1[3]
  ytop1a <- usr1[3] + BRACKET_BASE_FR * span1
  ytop1b <- ytop1a + BRACKET_GAP_BETWEEN * span1

  add_sq_bracket_lr(bp1[1], bp1[2], y_top = ytop1a,
                    arm_left = ARM_TR_REAL_LEFT_FR * span1,
                    arm_right = ARM_TR_REAL_RIGHT_FR * span1,
                    label = fmt_p_label(p_bun_trt_vs_real), cex = 0.92)
  add_sq_bracket_lr(bp1[1], bp1[3], y_top = ytop1b,
                    arm_left = ARM_TR_GEN_LEFT_FR * span1,
                    arm_right = ARM_TR_GEN_LEFT_FR * span1,
                    label = fmt_p_label(p_bun_trt_vs_gen),  cex = 0.92)

  ## ===================== CRE =====================
  bp2 <- barplot(
    cre_means, col = cols, ylim = CRE_YLIM,
    ylab = "", xaxt = "n", yaxt = "n", width = BAR_WIDTH, border = NA
  )
  box(lwd = BOX_LWD)
  panel_strip(CRE_label)

  axis(2, at = CRE_TICKS, labels = CRE_TICKS, las = 1, tck = -0.015)

  # CHANGED: "Generated" -> "SynCtrl"
  draw_two_line_labels(bp2, c("Treatment","Real","GanCtrl"), c("","Control",""))

  usr2  <- par("usr"); span2 <- usr2[4] - usr2[3]

  ## ---- CHANGED: shift both CRE brackets upward ----
  base2  <- usr2[3] + (BRACKET_BASE_FR + CRE_BRACKET_UP_FR) * span2 - PANEL2_BRACKET_SHIFT_FR * span2
  top_cap <- usr2[4] - 0.02 * span2
  ytop2a <- min(base2, top_cap)
  ytop2b <- min(base2 + BRACKET_GAP_BETWEEN * span2, top_cap)

  add_sq_bracket_lr(bp2[1], bp2[2], y_top = ytop2a,
                    arm_left = ARM_TR_REAL_LEFT_FR * span2,
                    arm_right = ARM_TR_REAL_RIGHT_FR * span2,
                    label = fmt_p_label(p_cre_trt_vs_real), cex = 0.92)
  add_sq_bracket_lr(bp2[1], bp2[3], y_top = ytop2b,
                    arm_left = ARM_TR_GEN_LEFT_FR * span2,
                    arm_right = ARM_TR_GEN_RIGHT_FR * span2,
                    label = fmt_p_label(p_cre_trt_vs_gen),  cex = 0.92)

  # Title
  title_str <- paste0(compound_lbl, " — ", time_lbl)
  mtext(ucfirst(title_str), outer = TRUE, cex = 1.25, line = 0.9, font = 2)

  ok <- TRUE
},
error = function(e) message("Plotting error: ", conditionMessage(e)),
finally = { if (length(dev.list())) try(invisible(dev.off()), silent = TRUE) })

if (ok && file.exists(out_file)) {
  message("Saved: ", normalizePath(out_file, winslash = "/"))
} else {
  stop("TIFF not saved. Check write permissions for: ", out_dir, "\nWorking dir: ", getwd())
}







