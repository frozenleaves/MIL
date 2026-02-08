args <- commandArgs(trailingOnly = TRUE)

root_dir <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/checkpoints_70_30_multi_label_20260121"
out_png <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/learning_curve_val_loss.png"
out_csv <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/learning_curve_val_loss.csv"

files <- list.files(root_dir, pattern = "loss_log\\.csv$", recursive = TRUE, full.names = TRUE)
if (length(files) == 0) {
  stop(paste("No loss_log.csv found under:", root_dir))
}

read_one <- function(path) {
  df <- read.csv(path)
  size_match <- regexpr("size_([0-9]+)_seed", path, perl = TRUE)
  train_size <- if (size_match[1] > 0) {
    as.integer(sub(".*size_([0-9]+)_seed.*", "\\1", path))
  } else {
    NA_integer_
  }
  df$train_size <- train_size
  df$train_group <- if (!is.na(train_size)) {
    as.character(train_size)
  } else {
    basename(dirname(path))
  }
  df$source <- basename(dirname(path))
  df
}

df_all <- do.call(rbind, lapply(files, read_one))
if (any(is.na(df_all$train_size))) {
  warning("Some paths did not match size_XXX_seed_YYY; use parent folder as group label.")
}

if (!requireNamespace("dplyr", quietly = TRUE) || !requireNamespace("ggplot2", quietly = TRUE) ||
    !requireNamespace("tidyr", quietly = TRUE)) {
  stop("Please install packages: dplyr, ggplot2, tidyr")
}

library(dplyr)
library(ggplot2)
library(tidyr)

df_long <- df_all %>%
  pivot_longer(cols = c("train_loss", "val_loss"), names_to = "loss_type", values_to = "loss") %>%
  mutate(loss_type = ifelse(loss_type == "train_loss", "Train Loss", "Val Loss"))

if (all(!is.na(df_long$train_size))) {
  df_long$train_group <- factor(df_long$train_group, levels = as.character(sort(unique(df_long$train_size))))
  sizes <- as.numeric(levels(df_long$train_group))
  label_map <- setNames(sprintf("%.0f %%", sizes / max(sizes) * 100), levels(df_long$train_group))
} else {
  df_long$train_group <- factor(df_long$train_group, levels = unique(df_long$train_group))
  label_map <- NULL
}

write.csv(df_long, out_csv, row.names = FALSE)

p <- ggplot(df_long, aes(x = epoch, y = loss, color = train_group, linetype = loss_type, group = interaction(train_group, loss_type))) +
  geom_line(alpha = 0.9) +
  labs(
    x = "Epoch",
    y = "Loss",
    color = "Sampling ratio",
    linetype = "Loss type",
    title = "Learning Curve (Loss)"
  ) +
  theme_minimal()

if (!is.null(label_map)) {
  p <- p + scale_color_discrete(labels = label_map)
}

ggsave(out_png, plot = p, width = 8, height = 5, dpi = 200)
cat("Saved plot:", out_png, "\n")
cat("Saved table:", out_csv, "\n")
