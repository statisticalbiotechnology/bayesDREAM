#!/usr/bin/env Rscript
#
# Example R plotting script for bayesDREAM summary exports
#
# This script demonstrates how to load and visualize the CSV files
# exported by bayesDREAM's summary export methods:
# - technical_feature_summary_{modality}.csv
# - cis_guide_summary.csv
# - cis_cell_summary.csv
# - trans_feature_summary_{modality}.csv
#
# Requirements: tidyverse, ggplot2

library(tidyverse)

# ==============================================================================
# SECTION 1: Load Summary Data
# ==============================================================================

# Set results directory
results_dir <- "./test_output/summary_export_simple"

# Load summaries
tech_summary <- read_csv(file.path(results_dir, "technical_feature_summary_gene.csv"))
cis_guide <- read_csv(file.path(results_dir, "cis_guide_summary.csv"))
cis_cell <- read_csv(file.path(results_dir, "cis_cell_summary.csv"))
trans_summary <- read_csv(file.path(results_dir, "trans_feature_summary_gene.csv"))

cat("Loaded summary data:\n")
cat(sprintf("  - Technical: %d features × %d groups\n",
            nrow(tech_summary),
            (ncol(tech_summary) - 3) / 3))
cat(sprintf("  - Cis (guide-level): %d guides\n", nrow(cis_guide)))
cat(sprintf("  - Cis (cell-level): %d cells\n", nrow(cis_cell)))
cat(sprintf("  - Trans: %d features (%s)\n",
            nrow(trans_summary),
            trans_summary$function_type[1]))

# ==============================================================================
# SECTION 2: Plot Technical Fit (Overdispersion Parameters)
# ==============================================================================

cat("\n1. Plotting technical fit (overdispersion)...\n")

# Reshape technical summary for plotting
tech_long <- tech_summary %>%
  select(feature, starts_with("group_")) %>%
  pivot_longer(cols = -feature,
               names_to = c("group", ".value"),
               names_pattern = "group_(\\d+)_alpha_y_(.+)")

# Plot overdispersion by group
p_tech <- ggplot(tech_long, aes(x = mean, y = reorder(feature, mean), color = group)) +
  geom_pointrange(aes(xmin = lower, xmax = upper),
                  position = position_dodge(width = 0.5)) +
  labs(title = "Technical Fit: Overdispersion Parameters",
       subtitle = "Mean and 95% credible intervals by group",
       x = "Overdispersion (alpha_y)",
       y = "Feature",
       color = "Group") +
  theme_minimal() +
  theme(legend.position = "right")

ggsave("technical_overdispersion.pdf", p_tech, width = 8, height = 6)
cat("  ✓ Saved: technical_overdispersion.pdf\n")

# ==============================================================================
# SECTION 3: Plot Cis Expression
# ==============================================================================

cat("\n2. Plotting cis expression...\n")

# A. Guide-level x_true
p_cis_guide <- ggplot(cis_guide,
                       aes(x = reorder(guide, x_true_mean),
                           y = x_true_mean,
                           color = target)) +
  geom_pointrange(aes(ymin = x_true_lower, ymax = x_true_upper)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  labs(title = "Cis Expression by Guide",
       subtitle = "Mean and 95% credible intervals",
       x = "Guide",
       y = "Cis Expression (x_true)",
       color = "Target") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top")

ggsave("cis_guide_expression.pdf", p_cis_guide, width = 10, height = 6)
cat("  ✓ Saved: cis_guide_expression.pdf\n")

# B. Cell-level x_true distribution
p_cis_cell <- ggplot(cis_cell, aes(x = x_true_mean, fill = target)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Cis Expression Distribution (Cell-level)",
       subtitle = "Histogram of mean x_true values",
       x = "Cis Expression (x_true)",
       y = "Number of Cells",
       fill = "Target") +
  theme_minimal()

ggsave("cis_cell_distribution.pdf", p_cis_cell, width = 8, height = 6)
cat("  ✓ Saved: cis_cell_distribution.pdf\n")

# C. Raw counts vs x_true (guide-level)
p_cis_counts <- ggplot(cis_guide,
                        aes(x = raw_counts_mean, y = x_true_mean, color = target)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = x_true_lower, ymax = x_true_upper), alpha = 0.5) +
  geom_text(aes(label = guide), hjust = -0.1, vjust = 0, size = 3) +
  labs(title = "Raw Counts vs. Cis Expression",
       subtitle = "Guide-level summary",
       x = "Raw Counts (mean)",
       y = "Cis Expression (x_true)",
       color = "Target") +
  theme_minimal()

ggsave("cis_counts_vs_xtrue.pdf", p_cis_counts, width = 8, height = 6)
cat("  ✓ Saved: cis_counts_vs_xtrue.pdf\n")

# ==============================================================================
# SECTION 4: Plot Trans Effects
# ==============================================================================

cat("\n3. Plotting trans effects...\n")

# A. Volcano plot (observed log2FC vs. SE)
p_volcano <- trans_summary %>%
  mutate(significant = abs(observed_log2fc) > 0.5) %>%
  ggplot(aes(x = observed_log2fc, y = -log10(observed_log2fc_se), color = significant)) +
  geom_point(alpha = 0.6, size = 3) +
  geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", color = "gray50") +
  labs(title = "Trans Effects: Volcano Plot",
       subtitle = "Observed log2FC vs. standard error",
       x = "Observed Log2FC (perturbed vs NTC)",
       y = "-log10(Standard Error)",
       color = "Significant\n(|log2FC| > 0.5)") +
  theme_minimal()

ggsave("trans_volcano.pdf", p_volcano, width = 8, height = 6)
cat("  ✓ Saved: trans_volcano.pdf\n")

# B. Observed vs. full log2FC
if ("full_log2fc_mean" %in% colnames(trans_summary)) {
  p_log2fc_compare <- ggplot(trans_summary,
                               aes(x = observed_log2fc, y = full_log2fc_mean)) +
    geom_point(alpha = 0.6, size = 3) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_errorbar(aes(ymin = full_log2fc_lower, ymax = full_log2fc_upper),
                  alpha = 0.3) +
    labs(title = "Observed vs. Full Log2FC",
         subtitle = "Comparison of observed (perturbed vs NTC) and full dynamic range",
         x = "Observed Log2FC",
         y = "Full Log2FC (dynamic range)") +
    theme_minimal()

  ggsave("trans_log2fc_comparison.pdf", p_log2fc_compare, width = 8, height = 6)
  cat("  ✓ Saved: trans_log2fc_comparison.pdf\n")
}

# C. Hill parameters (if additive_hill)
if ("B_pos_mean" %in% colnames(trans_summary) && "B_neg_mean" %in% colnames(trans_summary)) {
  cat("\n  Plotting additive Hill parameters...\n")

  # Positive vs. negative Hill magnitudes
  p_hill_b <- ggplot(trans_summary, aes(x = B_pos_mean, y = B_neg_mean)) +
    geom_point(alpha = 0.6, size = 3) +
    geom_errorbar(aes(ymin = B_neg_lower, ymax = B_neg_upper), alpha = 0.3) +
    geom_errorbarh(aes(xmin = B_pos_lower, xmax = B_pos_upper), alpha = 0.3) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = "Hill Magnitudes: Positive vs. Negative",
         subtitle = "B_pos vs. B_neg with 95% CI",
         x = "Positive Hill Magnitude (B_pos)",
         y = "Negative Hill Magnitude (B_neg)") +
    theme_minimal()

  ggsave("trans_hill_magnitudes.pdf", p_hill_b, width = 8, height = 6)
  cat("  ✓ Saved: trans_hill_magnitudes.pdf\n")

  # EC50 vs. K for positive Hill
  p_hill_pos <- ggplot(trans_summary,
                        aes(x = EC50_pos_mean, y = K_pos_mean, color = B_pos_mean)) +
    geom_point(alpha = 0.6, size = 3) +
    geom_errorbar(aes(ymin = K_pos_lower, ymax = K_pos_upper), alpha = 0.3) +
    geom_errorbarh(aes(xmin = EC50_pos_lower, xmax = EC50_pos_upper), alpha = 0.3) +
    scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    labs(title = "Positive Hill Parameters",
         subtitle = "EC50 vs. Hill coefficient (K)",
         x = "EC50 (x at half-maximum)",
         y = "Hill Coefficient (K)",
         color = "Magnitude\n(B_pos)") +
    theme_minimal()

  ggsave("trans_hill_pos_params.pdf", p_hill_pos, width = 8, height = 6)
  cat("  ✓ Saved: trans_hill_pos_params.pdf\n")

  # Inflection points (if computed)
  if ("inflection_pos_mean" %in% colnames(trans_summary)) {
    p_inflection <- trans_summary %>%
      select(feature, inflection_pos_mean, inflection_neg_mean, B_pos_mean, B_neg_mean) %>%
      pivot_longer(cols = c(inflection_pos_mean, inflection_neg_mean),
                   names_to = "type",
                   values_to = "inflection") %>%
      mutate(type = ifelse(type == "inflection_pos_mean", "Positive", "Negative")) %>%
      ggplot(aes(x = inflection, fill = type)) +
      geom_histogram(bins = 20, alpha = 0.7, position = "identity") +
      labs(title = "Distribution of Inflection Points",
           subtitle = "Where dose-response curves have maximum curvature",
           x = "Inflection Point (x value)",
           y = "Count",
           fill = "Hill Type") +
      theme_minimal()

    ggsave("trans_inflection_distribution.pdf", p_inflection, width = 8, height = 6)
    cat("  ✓ Saved: trans_inflection_distribution.pdf\n")
  }
}

# ==============================================================================
# SECTION 5: Summary Statistics
# ==============================================================================

cat("\n4. Computing summary statistics...\n")

# Technical fit summary
tech_stats <- tech_long %>%
  group_by(group) %>%
  summarize(
    mean_alpha = mean(mean),
    median_alpha = median(mean),
    sd_alpha = sd(mean),
    .groups = "drop"
  )

cat("\n  Technical Fit Summary:\n")
print(tech_stats)

# Cis expression summary
cis_stats <- cis_guide %>%
  group_by(target) %>%
  summarize(
    n_guides = n(),
    mean_xtrue = mean(x_true_mean),
    sd_xtrue = sd(x_true_mean),
    mean_counts = mean(raw_counts_mean),
    .groups = "drop"
  )

cat("\n  Cis Expression Summary:\n")
print(cis_stats)

# Trans effects summary
trans_stats <- trans_summary %>%
  summarize(
    n_features = n(),
    n_positive = sum(observed_log2fc > 0),
    n_negative = sum(observed_log2fc < 0),
    n_significant = sum(abs(observed_log2fc) > 0.5),
    mean_log2fc = mean(observed_log2fc),
    median_log2fc = median(observed_log2fc)
  )

cat("\n  Trans Effects Summary:\n")
print(trans_stats)

# ==============================================================================
# SECTION 6: Export Combined Summary Table
# ==============================================================================

cat("\n5. Creating combined summary table...\n")

# Combine key metrics into single table
combined_summary <- trans_summary %>%
  select(feature, observed_log2fc, observed_log2fc_se) %>%
  left_join(
    tech_summary %>%
      select(feature, starts_with("group_0_alpha_y")),
    by = "feature"
  )

write_csv(combined_summary, "combined_feature_summary.csv")
cat("  ✓ Saved: combined_feature_summary.csv\n")

# ==============================================================================
# Done!
# ==============================================================================

cat("\n" , strrep("=", 80), "\n")
cat("✓ All plots created successfully!\n")
cat(strrep("=", 80), "\n\n")

cat("Files created:\n")
cat("  Plots:\n")
cat("    - technical_overdispersion.pdf\n")
cat("    - cis_guide_expression.pdf\n")
cat("    - cis_cell_distribution.pdf\n")
cat("    - cis_counts_vs_xtrue.pdf\n")
cat("    - trans_volcano.pdf\n")
cat("    - trans_log2fc_comparison.pdf\n")
if ("B_pos_mean" %in% colnames(trans_summary)) {
  cat("    - trans_hill_magnitudes.pdf\n")
  cat("    - trans_hill_pos_params.pdf\n")
  if ("inflection_pos_mean" %in% colnames(trans_summary)) {
    cat("    - trans_inflection_distribution.pdf\n")
  }
}
cat("\n  Tables:\n")
cat("    - combined_feature_summary.csv\n")
cat("\nDone!\n")
