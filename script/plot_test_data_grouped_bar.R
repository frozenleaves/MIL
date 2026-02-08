script_dir <- dirname(normalizePath(sys.frame(1)$ofile))
if (is.na(script_dir) || script_dir == ".") {
  script_dir <- getwd()
}

input_path <- file.path(script_dir, "../test_data.csv")
out_dir <- file.path(script_dir, "figures/plot_test_data_grouped_bar")
out_png <- file.path(out_dir, "test_data_grouped_bar.png")

if (!file.exists(input_path)) {
  stop(paste("Input file not found:", input_path))
}

if (!requireNamespace("ggplot2", quietly = TRUE) ||
    !requireNamespace("dplyr", quietly = TRUE) ||
    !requireNamespace("tidyr", quietly = TRUE)) {
  stop("Please install packages: ggplot2, dplyr, tidyr")
}

library(ggplot2)
library(dplyr)
library(tidyr)

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

df <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
required_cols <- c(
  "modality",
  "matched_accuracy",
  "samples_avg_f1",
  "samples_avg_precision",
  "samples_avg_recall"
)

if ("matched accuracy" %in% colnames(df)) {
  df <- df %>% rename(matched_accuracy = `matched accuracy`)
} else if ("matched.accuracy" %in% colnames(df)) {
  df <- df %>% rename(matched_accuracy = matched.accuracy)
}

if (!all(required_cols %in% colnames(df))) {
  stop("Input file missing required columns.")
}

df <- df %>%
  mutate(
    modality = as.factor(modality),
    matched_accuracy = as.numeric(matched_accuracy),
    samples_avg_f1 = as.numeric(samples_avg_f1),
    samples_avg_precision = as.numeric(samples_avg_precision),
    samples_avg_recall = as.numeric(samples_avg_recall)
  )

df_long <- df %>%
  select(all_of(required_cols)) %>%
  pivot_longer(
    cols = -modality,
    names_to = "metric",
    values_to = "value"
  )

metric_labels <- c(
  matched_accuracy = "Accuracy",
  samples_avg_f1 = "Samples F1",
  samples_avg_precision = "Samples Precision",
  samples_avg_recall = "Samples Recall"
)

df_long$metric <- factor(df_long$metric, levels = names(metric_labels))

p <- ggplot(df_long, aes(x = modality, y = value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.8) +
  scale_fill_brewer(palette = "Set2", labels = metric_labels) +
  labs(
    x = "",
    y = "",
    fill = "",
    title = ""
  ) +
  coord_cartesian(ylim = c(0, 1)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(out_png, plot = p, width = 8, height = 4.8, dpi = 200)
cat("Saved plot:", out_png, "\n")

