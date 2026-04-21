# =============================================================================
# FEATURE SELECTION - Bridge between Section 1 and Section 2
# Big Data Module - Bankruptcy Prediction Dataset
# =============================================================================

# --- Install & Load Packages -------------------------------------------------

packages <- c("readxl", "dplyr", "ggplot2", "caret", "randomForest")

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(readxl)
library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)

# =============================================================================
# STEP 0: LOAD & CLEAN DATA
# =============================================================================

df <- read_excel("Group 2 dataset.xlsx")

colnames(df) <- colnames(df) %>%
  trimws() %>%
  make.names()

colnames(df)[1] <- "Bankrupt"
df$Bankrupt <- as.factor(df$Bankrupt)

cat("Dataset loaded:", nrow(df), "rows,", ncol(df), "columns\n\n")

# =============================================================================
# STEP 1: REMOVE NEAR-ZERO VARIANCE FEATURES
# =============================================================================
# Features that barely change across companies carry no useful information
# for prediction. We remove them before modelling.

cat("=============================================\n")
cat("STEP 1: NEAR-ZERO VARIANCE REMOVAL\n")
cat("=============================================\n")

# These two binary flags are protected from NZV removal on theoretical grounds.
# Although binary features often trigger NZV criteria, both are established
# bankruptcy predictors (Altman, 1968; Zmijewski, 1984) and are retained
# regardless of their variance properties (Strobl et al., 2007).
protected_features <- c("Liability.Assets.Flag", "Net.Income.Flag")

# Identify near-zero variance features (excludes target AND protected features)
nzv_info <- nearZeroVar(
  df %>% select(-Bankrupt, -all_of(protected_features)),
  saveMetrics = TRUE
)

# Show which features are flagged
nzv_flagged <- rownames(nzv_info)[nzv_info$nzv == TRUE]
cat("Features flagged as near-zero variance:", length(nzv_flagged), "\n")
if (length(nzv_flagged) > 0) {
  cat("Removed features:\n")
  print(nzv_flagged)
}

cat("\nProtected from NZV removal (theoretical grounds):\n")
for (f in protected_features) cat(sprintf("  - %s\n", f))

# Remove near-zero variance features (protected features are never in nzv_flagged)
df_clean <- df %>% select(-all_of(nzv_flagged))
cat("\nFeatures remaining after NZV removal:", ncol(df_clean) - 1, "\n")

# =============================================================================
# STEP 2: RANDOM FOREST FEATURE IMPORTANCE
# =============================================================================
# Train a Random Forest on all remaining features.
# The model tells us which features it found most useful for splitting —
# these are our most important predictors of bankruptcy.

cat("\n=============================================\n")
cat("STEP 2: RANDOM FOREST FEATURE IMPORTANCE\n")
cat("=============================================\n")
cat("Training Random Forest on all remaining features...\n")
cat("(This may take 1-2 minutes)\n\n")

set.seed(42)

# Use a smaller forest just for feature selection (faster)
rf_selection <- randomForest(
  Bankrupt ~ .,
  data       = df_clean,
  ntree      = 200,
  importance = TRUE,
  classwt    = c("0" = 1, "1" = 10)  # account for class imbalance
)

# Extract importance scores
importance_df <- data.frame(
  Feature    = rownames(importance(rf_selection)),
  MeanDecreaseGini = importance(rf_selection)[, "MeanDecreaseGini"],
  MeanDecreaseAccuracy = importance(rf_selection)[, "MeanDecreaseAccuracy"],
  row.names  = NULL
) %>% arrange(desc(MeanDecreaseGini))

cat("Top 20 most important features (by Mean Decrease in Gini):\n")
print(head(importance_df, 20))

# Save full importance table
write.csv(importance_df, "feature_importance.csv", row.names = FALSE)
cat("\n>> Full importance table saved to: feature_importance.csv\n")

# =============================================================================
# STEP 3: VISUALISE FEATURE IMPORTANCE
# =============================================================================

top20 <- head(importance_df, 20)

p_importance <- ggplot(top20,
                       aes(x = reorder(Feature, MeanDecreaseGini),
                           y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "#2980B9") +
  geom_text(aes(label = round(MeanDecreaseGini, 1)),
            hjust = -0.1, size = 3) +
  coord_flip() +
  ylim(0, max(top20$MeanDecreaseGini) * 1.12) +
  labs(
    title    = "Random Forest Feature Importance",
    subtitle = "Top 20 features ranked by Mean Decrease in Gini Impurity",
    x        = "Feature",
    y        = "Mean Decrease Gini"
  ) +
  theme_minimal(base_size = 10)

ggsave("plot_06_feature_importance.png",
       plot = p_importance, width = 12, height = 8, dpi = 150)
cat(">> Saved: plot_06_feature_importance.png\n")

# =============================================================================
# STEP 4: FINAL FEATURE SELECTION
# =============================================================================
# Top 20 features from Random Forest importance PLUS two binary distress flags
# forced in on theoretical grounds (Altman, 1968; Zmijewski, 1984).
# Gini importance is known to undervalue binary features (Strobl et al., 2007)
# so these are retained regardless of their importance rank.

cat("\n=============================================\n")
cat("STEP 4: FINAL FEATURE SELECTION\n")
cat("=============================================\n")

# Top 20 from Random Forest
top20_features <- as.character(head(importance_df$Feature, 20))

# Binary flags to force-include (if not already in top 20)
binary_flags <- c("Liability.Assets.Flag", "Net.Income.Flag")
binary_flags <- binary_flags[binary_flags %in% names(df_clean)]  # safety check

# Check which binary flags are already in top 20
already_included <- binary_flags[binary_flags %in% top20_features]
forced_in        <- binary_flags[!binary_flags %in% top20_features]

if (length(already_included) > 0) {
  cat("Binary flags already in top 20:\n")
  for (f in already_included) cat(sprintf("  - %s\n", f))
}

if (length(forced_in) > 0) {
  cat("Binary flags force-included on theoretical grounds:\n")
  for (f in forced_in) cat(sprintf("  - %s\n", f))
  cat("  (Justification: Gini importance undervalues binary features,\n")
  cat("   Strobl et al. 2007; both flags are established distress\n")
  cat("   indicators, Altman 1968; Zmijewski 1984)\n")
}

# Combine: top 20 + any forced binary flags not already present
final_features <- unique(c(top20_features, forced_in))

cat("\nFinal selected features (", length(final_features), "total ):\n", sep = "")
for (i in seq_along(final_features)) {
  flag <- if (final_features[i] %in% forced_in) "  [forced - binary flag]" else ""
  cat(sprintf("  %2d. %s%s\n", i, final_features[i], flag))
}

# Build final dataset
df_final <- df_clean %>%
  select(Bankrupt, all_of(final_features))

cat("\nFinal dataset dimensions:", nrow(df_final), "rows x", ncol(df_final), "columns\n")

# Save for Section 2
saveRDS(df_final, "df_final.rds")
cat(">> Final dataset saved to: df_final.rds\n")
cat("   (Load in Section 2 scripts with: df_final <- readRDS('df_final.rds'))\n")

# =============================================================================
# STEP 5: CORRELATION OF TOP 20 WITH TARGET (for report)
# =============================================================================

cat("\n=============================================\n")
cat("STEP 5: CORRELATION OF TOP FEATURES WITH TARGET\n")
cat("=============================================\n")

df_cor <- df_final %>%
  mutate(Bankrupt_num = as.numeric(as.character(Bankrupt))) %>%
  select(-Bankrupt)

cor_vals <- cor(df_cor, df_cor$Bankrupt_num, use = "complete.obs")
cor_vals  <- cor_vals[rownames(cor_vals) != "Bankrupt_num", , drop = FALSE]

cor_summary <- data.frame(
  Feature     = rownames(cor_vals),
  Correlation = round(cor_vals[, 1], 4),
  row.names   = NULL
) %>% arrange(desc(abs(Correlation)))

cat("Correlation of top 20 features with Bankrupt:\n")
print(cor_summary)

# Plot
p_cor <- ggplot(cor_summary,
                aes(x = reorder(Feature, Correlation),
                    y = Correlation,
                    fill = Correlation > 0)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#E74C3C", "FALSE" = "#2ECC71"),
                    labels  = c("TRUE" = "Positive", "FALSE" = "Negative")) +
  labs(
    title    = "Correlation of Selected Features with Bankruptcy",
    subtitle = "Final selected features (RF top 20 + forced binary flags)",
    x        = "Feature",
    y        = "Pearson Correlation",
    fill     = "Direction"
  ) +
  theme_minimal(base_size = 10)

ggsave("plot_07_feature_correlations.png",
       plot = p_cor, width = 12, height = 8, dpi = 150)
cat(">> Saved: plot_07_feature_correlations.png\n")

# =============================================================================
# SUMMARY
# =============================================================================

cat("\n=============================================\n")
cat("FEATURE SELECTION COMPLETE\n")
cat("=============================================\n")
cat("Started with   :", ncol(df) - 1, "features\n")
cat("After NZV      :", ncol(df_clean) - 1, "features\n")
cat("RF top 20      :", length(top20_features), "features\n")
cat("Forced binary  :", length(forced_in), "features\n")
cat("Final selected :", length(final_features), "features\n")
cat("\nOutput files:\n")
cat("  plot_06_feature_importance.png\n")
cat("  plot_07_feature_correlations.png\n")
cat("  feature_importance.csv\n")
cat("  df_final.rds  (used by Section 2 models)\n")
cat("=============================================\n")