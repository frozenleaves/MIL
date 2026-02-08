args <- commandArgs(trailingOnly = TRUE)

root_dir <- if (length(args) >= 1) args[[1]] else "."
out_png <- if (length(args) >= 2) args[[2]] else "learning_curve_loss.png"
out_csv <- if (length(args) >= 3) args[[3]] else "learning_curve_loss.csv"

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
} else {
  df_long$train_group <- factor(df_long$train_group, levels = unique(df_long$train_group))
}

write.csv(df_long, out_csv, row.names = FALSE)

p <- ggplot(df_long, aes(x = epoch, y = loss, color = train_group, linetype = loss_type, group = interaction(train_group, loss_type))) +
  geom_line(alpha = 0.9) +
  labs(
    x = "Epoch",
    y = "Loss",
    color = "Train group",
    linetype = "Loss type",
    title = "Learning Curve (Loss)"
  ) +
  theme_minimal()

ggsave(out_png, plot = p, width = 8, height = 5, dpi = 200)
cat("Saved plot:", out_png, "\n")
cat("Saved table:", out_csv, "\n")
