#!/usr/bin/env Rscript

# 用 compute_txt_ig_shap_like.py 输出的 txt_ig_shap_long.csv 画“类似 SHAP summary(beeswarm)”图
# 输入列：sample_id,class,token,shap_value,token_count,y_true
#
# 用法：
#   Rscript plot_txt_shap_summary.R txt_ig_shap_long.csv out.png

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(stringr)
  library(forcats)
})

args <- commandArgs(trailingOnly = TRUE)
in_path <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/txt_ig_shap_long.csv"
out_path <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/Txt_Shap_Summary/txt_ig_shap_summary.png"


df <- read_csv(in_path, show_col_types = FALSE)

# 优先用 token_decoded；否则退回 token；再退回 token_id
if ("token_decoded" %in% names(df)) {
  df <- df %>% mutate(feature = token_decoded)
} else if ("token" %in% names(df)) {
  df <- df %>% mutate(feature = token)
} else {
  df <- df %>% mutate(feature = as.character(token_id))
}

# 你可以改这里：选某个类别；或 facet 分面画多个类别
classes <- unique(df$class)

# 每个类选 top N token（按 mean(|shap|)）
TOP_N <- 15
top_tokens <- df %>%
  group_by(class, feature) %>%
  summarise(mean_abs = mean(abs(shap_value)), .groups = "drop") %>%
  group_by(class) %>%
  slice_max(mean_abs, n = TOP_N, with_ties = FALSE) %>%
  ungroup()

plot_df <- df %>%
  inner_join(top_tokens, by = c("class", "feature")) %>%
  mutate(
    feature = fct_reorder(feature, mean_abs),
    token_count = pmin(token_count, 5)  # 计数太大压缩一下颜色范围
  )

p <- ggplot(plot_df, aes(x = shap_value, y = feature, color = token_count)) +
  geom_jitter(height = 0.18, width = 0, alpha = 0.65, size = 1.6) +
  geom_vline(xintercept = 0, linewidth = 0.4, color = "grey40") +
  scale_color_gradient(low = "#2C7BB6", high = "#FDAE61", name = "token\ncount") +
  labs(x = "Attribution (impact on class logit)", y = NULL) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "right"
  ) +
  facet_wrap(~ class, scales = "free_y", ncol = 2)

ggsave(out_path, p, width = 12, height = 8, dpi = 200)
message("Saved plot to: ", out_path)

