#!/usr/bin/env Rscript

# 从 calibration_long-*.csv（长表：sample_id,label,y_true,y_prob）绘制多标签“混淆矩阵”图。
#
# 输出两类图（每个输入 CSV 各一份）：
# 1) 7x4: 每个类别的 TP/FP/FN/TN 计数热力图（多标签最标准的 confusion 汇总）
# 2) 7x7: true-vs-pred 共现矩阵（M[i,j]=count(true_i==1 & pred_j==1)），便于看“易同时出现/易混淆”的类对
#
# 用法示例：
#   Rscript plot_multilabel_confusion.R --threshold 0.5 --fallback 1 \
#     calibration_long-img-svs.csv calibration_long-img.csv ...
#
# 依赖：
#   install.packages(c("readr","dplyr","tidyr","ggplot2","stringr","forcats"))

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(stringr)
  library(forcats)
})

parse_args <- function(args) {
  # 只做简单解析：支持 --threshold x --fallback 0/1 --outdir path
  threshold <- 0.5
  fallback <- TRUE
  outdir <- "."
  files <- c()

  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--threshold") {
      threshold <- as.numeric(args[[i + 1]])
      i <- i + 2
    } else if (a == "--fallback") {
      fallback <- as.logical(as.integer(args[[i + 1]]))
      i <- i + 2
    } else if (a == "--outdir") {
      outdir <- args[[i + 1]]
      i <- i + 2
    } else {
      files <- c(files, a)
      i <- i + 1
    }
  }
  list(threshold = threshold, fallback = fallback, outdir = outdir, files = files)
}

long_to_wide <- function(df_long, label_levels = NULL) {
  # df_long: sample_id,label,y_true,y_prob
  if (!all(c("sample_id","label","y_true","y_prob") %in% names(df_long))) {
    stop("CSV 必须包含列：sample_id,label,y_true,y_prob")
  }
  df_long <- df_long %>%
    mutate(
      y_true = as.integer(y_true),
      y_prob = as.numeric(y_prob)
    )

  if (is.null(label_levels)) {
    label_levels <- sort(unique(df_long$label))
  }

  # pivot y_true / y_prob 各成一张宽表
  y_true_w <- df_long %>%
    mutate(label = factor(label, levels = label_levels)) %>%
    select(sample_id, label, y_true) %>%
    distinct() %>%
    tidyr::pivot_wider(names_from = label, values_from = y_true)

  y_prob_w <- df_long %>%
    mutate(label = factor(label, levels = label_levels)) %>%
    select(sample_id, label, y_prob) %>%
    distinct() %>%
    tidyr::pivot_wider(names_from = label, values_from = y_prob)

  list(
    labels = label_levels,
    y_true = y_true_w,
    y_prob = y_prob_w
  )
}

apply_threshold <- function(y_prob_mat, threshold = 0.5, fallback = TRUE) {
  y_pred <- ifelse(y_prob_mat >= threshold, 1L, 0L)
  if (fallback) {
    empty <- rowSums(y_pred) == 0
    if (any(empty)) {
      max_idx <- apply(y_prob_mat[empty, , drop = FALSE], 1, which.max)
      # 先清 0 再置 1
      y_pred[empty, ] <- 0L
      y_pred[cbind(which(empty), max_idx)] <- 1L
    }
  }
  y_pred
}

confusion_7x4 <- function(y_true_mat, y_pred_mat, labels) {
  stopifnot(ncol(y_true_mat) == length(labels))
  stopifnot(ncol(y_pred_mat) == length(labels))
  out <- lapply(seq_along(labels), function(j) {
    yt <- y_true_mat[, j]
    yp <- y_pred_mat[, j]
    tp <- sum(yt == 1 & yp == 1)
    fp <- sum(yt == 0 & yp == 1)
    fn <- sum(yt == 1 & yp == 0)
    tn <- sum(yt == 0 & yp == 0)
    tibble(
      label = labels[[j]],
      metric = c("TP","FP","FN","TN"),
      value = c(tp, fp, fn, tn)
    )
  }) %>% bind_rows()
  out
}

cooccurrence_7x7 <- function(y_true_mat, y_pred_mat, labels) {
  # M[i,j] = count(true_i==1 & pred_j==1)
  M <- t(y_true_mat) %*% y_pred_mat
  dimnames(M) <- list(true = labels, pred = labels)
  as.data.frame(as.table(M)) %>%
    rename(true_label = true, pred_label = pred, value = Freq)
}

plot_heatmap_7x4 <- function(df_7x4, title) {
  df_7x4 %>%
    mutate(
      label = factor(label, levels = unique(label)),
      metric = factor(metric, levels = c("TP","FP","FN","TN"))
    ) %>%
    ggplot(aes(x = metric, y = label, fill = value)) +
    geom_tile(color = "white", linewidth = 0.4) +
    geom_text(aes(label = value), size = 3) +
    scale_fill_gradient(low = "#F7FBFF", high = "#08306B", name = "count") +
    labs(title = title, x = NULL, y = NULL) +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank())
}

plot_heatmap_7x7 <- function(df_7x7, title) {
  df_7x7 %>%
    mutate(
      true_label = factor(true_label, levels = unique(true_label)),
      pred_label = factor(pred_label, levels = unique(pred_label))
    ) %>%
    ggplot(aes(x = pred_label, y = true_label, fill = value)) +
    geom_tile(color = "white", linewidth = 0.35) +
    geom_text(aes(label = value), size = 2.8) +
    scale_fill_gradient(low = "#FFF5F0", high = "#A50F15", name = "count") +
    labs(title = title, x = "Pred", y = "True") +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1)
    )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opt <- parse_args(args)
  if (length(opt$files) == 0) {
    stop("请在命令行参数里提供至少一个 calibration_long-*.csv 文件路径")
  }
  dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)

  for (path in opt$files) {
    message("Processing: ", path)
    df <- read_csv(path, show_col_types = FALSE)

    # label 顺序：优先用 Config 中的顺序（如果你的 label 列是这些），否则用文件内排序
    preferred <- c("OLK","OLP","OSCC","OSF","乳头状瘤","粘液囊肿","纤维增生")
    label_levels <- preferred[preferred %in% unique(df$label)]
    if (length(label_levels) == 0) label_levels <- sort(unique(df$label))

    wide <- long_to_wide(df, label_levels = label_levels)
    labels <- wide$labels

    # 拼成矩阵（按 sample_id 对齐）
    y_true_df <- wide$y_true %>% arrange(sample_id)
    y_prob_df <- wide$y_prob %>% arrange(sample_id)
    if (!identical(y_true_df$sample_id, y_prob_df$sample_id)) {
      stop("y_true 与 y_prob 的 sample_id 未对齐，请检查 CSV 是否完整")
    }

    y_true_mat <- as.matrix(y_true_df %>% select(all_of(labels)))
    y_prob_mat <- as.matrix(y_prob_df %>% select(all_of(labels)))
    storage.mode(y_true_mat) <- "integer"
    storage.mode(y_prob_mat) <- "double"

    y_pred_mat <- apply_threshold(y_prob_mat, threshold = opt$threshold, fallback = opt$fallback)

    base <- str_replace(basename(path), "\\.csv$", "")
    title_7x4 <- sprintf("%s | threshold=%.3f | fallback=%s", base, opt$threshold, opt$fallback)
    title_7x7 <- sprintf("%s (co-occurrence) | threshold=%.3f | fallback=%s", base, opt$threshold, opt$fallback)

    df_7x4 <- confusion_7x4(y_true_mat, y_pred_mat, labels)
    p1 <- plot_heatmap_7x4(df_7x4, title_7x4)
    out1 <- file.path(opt$outdir, paste0(base, "_confusion_7x4.png"))
    ggsave(out1, p1, width = 6.6, height = 3.6, dpi = 200)
    message("Saved: ", out1)

    df_7x7 <- cooccurrence_7x7(y_true_mat, y_pred_mat, labels)
    p2 <- plot_heatmap_7x7(df_7x7, title_7x7)
    out2 <- file.path(opt$outdir, paste0(base, "_cooccurrence_7x7.png"))
    ggsave(out2, p2, width = 7.2, height = 6.0, dpi = 200)
    message("Saved: ", out2)
  }
}

main()

