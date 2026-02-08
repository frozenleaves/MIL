#!/usr/bin/env Rscript

# 使用 compute_modality_influence.py 的输出绘制热力图
# 输入: modality_influence_heatmap.csv

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
in_path <- if (length(args) >= 1) args[[1]] else "modality_influence_heatmap.csv"
out_path <- if (length(args) >= 2) args[[2]] else "modality_influence_heatmap.png"

df <- read_csv(in_path, show_col_types = FALSE)

# 选用 norm_abs_pos（每类仅在正样本上汇总，并按列归一化），更接近“该类典型特征模态更重要”的叙事
plot_df <- df %>%
  mutate(
    label = factor(label, levels = unique(label)),
    modality = factor(modality, levels = c("txt", "img", "svs"))
  )

p <- ggplot(plot_df, aes(x = label, y = modality, fill = norm_abs_pos)) +
  geom_tile(color = "white", linewidth = 0.4) +
  scale_fill_gradient(
    low = "#FFF3A0",   # 淡黄
    high = "#E64B35",  # 红
    name = "Influence\n(norm)"
  ) +
  labs(x = NULL, y = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 30, hjust = 1, vjust = 1),
    legend.position = "right"
  )

ggsave(out_path, p, width = 11, height = 3.5, dpi = 200)
message("Saved plot to: ", out_path)

