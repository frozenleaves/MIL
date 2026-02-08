script_dir <- dirname(normalizePath(sys.frame(1)$ofile))
if (is.na(script_dir) || script_dir == ".") {
  script_dir <- getwd()
}

input_dir <- script_dir
pattern <- "^calibration.*\\.csv$"
base_out_dir <- file.path(script_dir, "figures", "label_roc_auc")
if (!dir.exists(base_out_dir)) {
  dir.create(base_out_dir, recursive = TRUE)
}

files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
if (length(files) == 0) {
  stop(paste("No files found under:", input_dir))
}

if (!requireNamespace("ggplot2", quietly = TRUE) || !requireNamespace("dplyr", quietly = TRUE) ||
    !requireNamespace("pROC", quietly = TRUE)) {
      
  stop("Please install packages: ggplot2, dplyr, pROC")
}

library(ggplot2)
library(dplyr)
library(pROC)

safe_name <- function(x) {
  gsub("[^A-Za-z0-9_\\-]+", "_", x)
}

for (path in files) {
  df <- read.csv(path)
  required_cols <- c("sample_id", "label", "y_true", "y_prob")
  if (!all(required_cols %in% colnames(df))) {
    warning(paste("Skip file missing columns:", path))
    next
  }

  file_id <- tools::file_path_sans_ext(basename(path))
  out_dir <- file.path(base_out_dir, file_id)
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }

  auc_rows <- list()
  roc_rows <- list()

  for (lab in sort(unique(df$label))) {
    g <- df[df$label == lab, ]
    y_true <- as.integer(g$y_true)
    y_prob <- as.numeric(g$y_prob)

    if (length(unique(y_true)) < 2) {
      auc_val <- NA_real_
    } else {
      roc_obj <- pROC::roc(y_true, y_prob, quiet = TRUE, direction = "<")
      auc_val <- as.numeric(pROC::auc(roc_obj))

      fpr <- 1 - roc_obj$specificities
      tpr <- roc_obj$sensitivities
      roc_rows[[length(roc_rows) + 1]] <- data.frame(
        label = lab,
        fpr = fpr,
        tpr = tpr
      )
    }

    auc_rows[[length(auc_rows) + 1]] <- data.frame(
      label = lab,
      auc = auc_val
    )
  }

  roc_df <- if (length(roc_rows) > 0) bind_rows(roc_rows) else data.frame()
  auc_df <- bind_rows(auc_rows)

  if (nrow(roc_df) > 0) {
    p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, color = label, group = label)) +
      geom_line(size = 0.6) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
      labs(
        x = "False Positive Rate",
        y = "True Positive Rate",
        color = "",
        title = paste("ROC by Label (", gsub("^calibration_long-", "", file_id), ")")
      ) +
      scale_color_discrete(labels = function(x) gsub("^calibration_long-", "", x)) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5)
      )

    out_roc <- file.path(out_dir, "roc_by_label.png")
    ggsave(out_roc, plot = p_roc, width = 7, height = 5, dpi = 200)
  }

  if (nrow(auc_df) > 0) {
    p_auc <- ggplot(auc_df, aes(x = label, y = auc, fill = label)) +
      geom_col(width = 0.7, alpha = 0.8) +
      labs(
        x = "Label",
        y = "AUC",
        title = paste("AUC per Label (", gsub("^calibration_long-", "", file_id), ")")
      ) +
      scale_y_continuous(breaks = seq(0.4, 1, 0.1), expand = c(0, 0)) +
      coord_cartesian(ylim = c(0.4, 1)) +
      theme_minimal() +
      theme(
        legend.position = "none",
        plot.title = element_text(hjust = 0.5)
      )

    out_auc <- file.path(out_dir, "auc_box_by_label.png")
    ggsave(out_auc, plot = p_auc, width = 7, height = 5, dpi = 200)
  }
}

cat("Saved plots to:", base_out_dir, "\n")
