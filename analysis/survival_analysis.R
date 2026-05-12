#!/usr/bin/env Rscript
# survival_analysis.R — survminer KM plots, forest plot, Schoenfeld residuals.
# Called by bootstrap.sh after Python TTD analysis.

suppressPackageStartupMessages({
  library(survival)
  library(survminer)
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(broom)
  library(optparse)
})

option_list <- list(
  make_option("--ttd-file", default = "outputs/tables/ttd_events.csv"),
  make_option("--cohort",   default = "outputs/tables/cohort_matched.csv"),
  make_option("--output",   default = "outputs/figures")
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$output, recursive = TRUE, showWarnings = FALSE)

message("Loading TTD events: ", opt$`ttd-file`)
if (!file.exists(opt$`ttd-file`)) stop("ttd_events.csv not found — run run_ttd.py first")

ttd    <- read_csv(opt$`ttd-file`, show_col_types = FALSE)
cohort <- read_csv(opt$cohort,     show_col_types = FALSE)

if (!"drug_class" %in% names(ttd)) {
  ttd <- ttd %>% left_join(select(cohort, person_id, drug_class), by = "person_id")
}

ttd <- ttd %>%
  filter(!is.na(ttd_days), ttd_days >= 0, !is.na(drug_class)) %>%
  mutate(drug_class = factor(drug_class, levels = c("metformin", "glp1", "sglt2"),
                              labels = c("Metformin", "GLP-1 RA", "SGLT-2i")))

# ── Kaplan-Meier (survminer) ──────────────────────────────────────────────────
surv_obj <- Surv(ttd$ttd_days, ttd$discontinued)
km_fit   <- survfit(surv_obj ~ drug_class, data = ttd)

km_plot <- ggsurvplot(
  km_fit,
  data          = ttd,
  risk.table    = TRUE,
  pval          = TRUE,
  conf.int      = TRUE,
  palette       = c("#3498DB", "#E74C3C", "#2ECC71"),
  legend.labs   = c("Metformin", "GLP-1 RA", "SGLT-2i"),
  xlab          = "Days from Index Date",
  ylab          = "Probability of Persistence",
  title         = "Treatment Persistence by Drug Class\n(90-day grace period, Lim 2025)",
  ggtheme       = theme_bw(base_size = 12),
  risk.table.height = 0.28,
  surv.median.line = "hv",
)
ggsave(
  file.path(opt$output, "km_persistence_survminer.png"),
  plot   = print(km_plot),
  width  = 10, height = 7, dpi = 150
)
message("KM plot saved: km_persistence_survminer.png")

# ── Cox PH model ──────────────────────────────────────────────────────────────
comorbidity_cols <- intersect(
  c("hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
    "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
    "sleep_apnea", "nafld", "pvd", "stroke", "mi"),
  names(ttd)
)

cox_formula_str <- paste(
  "Surv(ttd_days, discontinued) ~ drug_class + age_at_index + cci",
  if (length(comorbidity_cols) > 0) paste("+", paste(comorbidity_cols, collapse = " + ")) else ""
)
cox_formula <- tryCatch(as.formula(cox_formula_str), error = function(e) {
  as.formula("Surv(ttd_days, discontinued) ~ drug_class + age_at_index + cci")
})

cox_fit <- coxph(cox_formula, data = ttd)
cox_summary <- tidy(cox_fit, exponentiate = TRUE, conf.int = TRUE)
message("\nCox PH results:")
print(cox_summary)

# ── Forest plot of HRs ────────────────────────────────────────────────────────
forest_data <- cox_summary %>%
  filter(grepl("drug_class|comorbidity|hypertension|ckd|obesity|heart|stroke|mi", term)) %>%
  mutate(term = gsub("drug_class", "", term))

if (nrow(forest_data) > 0) {
  forest_plot <- ggplot(forest_data, aes(x = estimate, y = term)) +
    geom_point(size = 3, color = "#2C3E50") +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, color = "#7F8C8D") +
    geom_vline(xintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    scale_x_log10() +
    labs(
      x = "Hazard Ratio (log scale)", y = NULL,
      title = "Cox PH — Hazard Ratios for Treatment Discontinuation",
      subtitle = "Reference: Metformin"
    ) +
    theme_bw(base_size = 11) +
    theme(panel.grid.minor = element_blank())

  ggsave(file.path(opt$output, "forest_cox_ttd.png"), forest_plot, width = 8, height = 5, dpi = 150)
  message("Forest plot saved: forest_cox_ttd.png")
}

# ── Schoenfeld residuals (PH assumption test) ─────────────────────────────────
ph_test <- cox.zph(cox_fit)
message("\nSchoenfeld residuals (PH assumption):")
print(ph_test)

ph_plot <- ggcoxzph(ph_test, point.size = 0.5, point.alpha = 0.3)
ggsave(file.path(opt$output, "schoenfeld_residuals.png"),
       plot = print(ph_plot), width = 10, height = 6, dpi = 130)
message("Schoenfeld residuals plot saved")
message("Survival analysis complete.")
