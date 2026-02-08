script_dir <- dirname(normalizePath(sys.frame(1)$ofile))
if (is.na(script_dir) || script_dir == ".") {
  script_dir <- getwd()
}

input_dir <- script_dir
pattern <- "^calibration_long.*\\.csv$"
out_dir <- file.path(script_dir, "figures/plot_label_by_file")
out_png <- file.path(out_dir, "label_f1_by_file.png")

files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
if (length(files) == 0) {
  stop(paste("No files found under:", input_dir))
}

if (!requireNamespace("ggplot2", quietly = TRUE) || !requireNamespace("dplyr", quietly = TRUE)) {
  stop("Please install packages: ggplot2, dplyr")
}

library(ggplot2)
library(dplyr)

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

calc_metrics_by_label <- function(df) {
  df %>%
    mutate(
      y_true = as.integer(y_true),
      y_prob = as.numeric(y_prob),
      y_pred = as.integer(y_prob >= 0.5)
    ) %>%
    group_by(label) %>%
    summarise(
      tp = sum(y_true == 1 & y_pred == 1),
      fp = sum(y_true == 0 & y_pred == 1),
      fn = sum(y_true == 1 & y_pred == 0),
      tn = sum(y_true == 0 & y_pred == 0),
      precision = ifelse(tp + fp == 0, 0, tp / (tp + fp)),
      recall = ifelse(tp + fn == 0, 0, tp / (tp + fn)),
      specificity = ifelse(tn + fp == 0, 0, tn / (tn + fp)),
      f1 = ifelse((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall)),
      accuracy = (recall + specificity) / 2,
      .groups = "drop"
    ) %>%
    select(label, accuracy, precision, recall, f1)
}

f1_rows <- list()
for (path in files) {
  df <- read.csv(path)
  required_cols <- c("sample_id", "label", "y_true", "y_prob")
  if (!all(required_cols %in% colnames(df))) {
    warning(paste("Skip file missing columns:", path))
    next
  }

  file_id <- tools::file_path_sans_ext(basename(path))
  metrics_df <- calc_metrics_by_label(df)
  metrics_df$file <- file_id
  f1_rows[[length(f1_rows) + 1]] <- metrics_df
}

metrics_all <- bind_rows(f1_rows) %>%
  filter(is.finite(f1))
if (nrow(metrics_all) == 0) {
  stop("No valid data to plot.")
}

metrics_all$file <- factor(metrics_all$file, levels = tools::file_path_sans_ext(basename(files)))
metrics_all$label <- factor(metrics_all$label, levels = sort(unique(metrics_all$label)))

metrics_long <- metrics_all %>%
  pivot_longer(
    cols = c(accuracy, precision, recall, f1),
    names_to = "metric",
    values_to = "value"
  )

metric_labels <- c(
  accuracy = "Balanced Accuracy",
  precision = "Precision",
  recall = "Recall",
  f1 = "F1"
)

for (m in names(metric_labels)) {
  df_m <- metrics_long %>% filter(metric == m)
  if (nrow(df_m) == 0) {
    next
  }

  p_metric <- ggplot(df_m, aes(x = label, y = value, color = file, group = file)) +
    geom_line(size = 0.7) +
    geom_point(size = 1.6) +
    scale_color_discrete(labels = function(x) gsub("^calibration_long-", "", x)) +
    labs(
      x = "Label",
      y = metric_labels[[m]],
      color = "",
      title = paste0(metric_labels[[m]], " per Label")
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5)) +
    coord_cartesian(ylim = c(-0.05, 1))

  out_png <- file.path(out_dir, paste0("label_", m, "_by_file.png"))
  ggsave(out_png, plot = p_metric, width = 7, height = 5, dpi = 200)
  cat("Saved plot:", out_png, "\n")
}

