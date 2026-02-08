#!/home/codingma/anaconda3/envs/py312/bin/python
"""
为 txt-only（Transformer）模型生成“类似 SHAP summary(beeswarm)”所需数据。

说明：
- 这里不使用 shap 包（你环境目前未安装），改用 Integrated Gradients(IG) 在 embedding 上做归因。
- 结果不是严格 SHAP，但在实践中常用于 Transformer 文本解释，并可画出与 SHAP summary 类似的散点/蜂群图。

输出（长表 CSV）：
  sample_id,class,token_id,token_decoded,shap_value,token_count,y_true

其中：
- token_count: 该 token 在该样本中出现次数（用于配色，类似示例图“feature value”）
- shap_value: 该 token 对该 class logit 的贡献（对该 token 的所有位置 attribution 求和）

用法示例：
  /home/codingma/anaconda3/envs/py312/bin/python compute_txt_ig_shap_like.py \\
    --test-root /path/to/datasets/test \\
    --ckpt /path/to/checkpoints-txt_only/best_val.pth \\
    --out txt_ig_shap_long.csv \\
    --topk 20 --steps 32
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from mil.config import Config
from mil.model import UnifiedMultimodalModel
from transformers import AutoTokenizer
from tqdm import tqdm


LABEL_MAPPING_RULES = {
    0: [0],  # OLK
    1: [1],  # OLP
    2: [2],  # OSCC
    3: [3],  # OSF
    4: [0, 3],  # OSF+OLK -> OLK + OSF
    5: [4],  # 乳头状瘤
    6: [5],  # 粘液囊肿
    7: [6],  # 纤维增生
}


def scan_test_root_txt_only(test_root: Path) -> List[Dict]:
    samples: List[Dict] = []
    raw_class_map = Config.RAW_CLASS_MAP

    for class_name in os.listdir(test_root):
        class_dir = test_root / class_name
        if not class_dir.is_dir():
            continue
        if class_name not in raw_class_map:
            continue

        raw_label_id = raw_class_map[class_name]
        idxs = LABEL_MAPPING_RULES.get(raw_label_id, [])
        label_vec = np.zeros(Config.NUM_CLASSES, dtype=int)
        label_vec[idxs] = 1

        for sample_name in os.listdir(class_dir):
            sample_dir = class_dir / sample_name
            if not sample_dir.is_dir():
                continue
            txt_files = glob.glob(str(sample_dir / "*.txt"))
            if not txt_files:
                continue
            try:
                text_content = Path(txt_files[0]).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text_content = ""

            samples.append(
                {
                    "sample_id": sample_name,
                    "text": text_content,
                    "y_true": label_vec,
                }
            )
    return samples


def load_model_and_tokenizer(ckpt_path: Path, device: torch.device):
    # 强制 txt-only 模式
    Config.USE_TXT = True
    Config.USE_IMG = False
    Config.USE_SVS = False
    # 解释阶段不建议 AMP，梯度更稳定
    Config.USE_AMP = False

    tokenizer = AutoTokenizer.from_pretrained(Config.QWEN_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = UnifiedMultimodalModel().to(device)
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer


def forward_logits_from_embeds(
    model: UnifiedMultimodalModel,
    inputs_embeds: torch.Tensor,  # (1, L, H)
    attention_mask: torch.Tensor,  # (1, L)
) -> torch.Tensor:
    """
    复刻 UnifiedMultimodalModel.forward 的 text 分支，但用 inputs_embeds 来允许对 embedding 求梯度。
    返回 logits: (1, C)
    """
    batch_size = inputs_embeds.size(0)

    # text backbone forward
    text_out = model.text_backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    txt_emb = text_out.last_hidden_state  # (B, L, H)
    mask_expanded = attention_mask.unsqueeze(-1).expand(txt_emb.size()).float()
    txt_emb = torch.sum(txt_emb * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)  # (B, H)
    txt_feat = model.text_proj(txt_emb)  # (B, D)

    # img / svs disabled -> zeros (与模型 forward 一致的 dtype/device)
    img_feat = torch.zeros(batch_size, Config.FUSION_DIM, device=txt_feat.device, dtype=txt_feat.dtype)
    wsi_feat_agg = torch.zeros(batch_size, Config.FUSION_DIM, device=txt_feat.device, dtype=txt_feat.dtype)

    cls_tokens = model.fusion_cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
    fusion_input = torch.cat(
        [cls_tokens, txt_feat.unsqueeze(1), img_feat.unsqueeze(1), wsi_feat_agg.unsqueeze(1)],
        dim=1,
    )  # (B, 4, D)
    fusion_out = model.fusion_transformer(fusion_input)
    final_emb = fusion_out[:, 0, :]
    logits = model.classifier(final_emb)
    return logits


@torch.no_grad()
def _get_token_counts(input_ids: torch.Tensor, pad_id: int) -> Dict[int, int]:
    ids = input_ids[0].tolist()
    counts: Dict[int, int] = {}
    for tid in ids:
        if tid == pad_id:
            continue
        counts[tid] = counts.get(tid, 0) + 1
    return counts


def integrated_gradients_token_attribution(
    model: UnifiedMultimodalModel,
    tokenizer,
    text: str,
    target_class: int,
    device: torch.device,
    steps: int = 32,
    max_length: int | None = None,
    eps: float = 1e-9,
) -> Tuple[List[int], np.ndarray, Dict[int, int], np.ndarray]:
    """
    返回：
      token_ids: List[int]（长度 L）
      token_attr: np.ndarray shape (L,)（每个 token 的 attribution，已 mask pad）
      token_counts_id: Dict[token_id, count]（按 token_id 统计出现次数）
      y_pred_probs: np.ndarray shape (C,)
    """
    if max_length is None:
        max_length = int(getattr(Config, "MAX_TEXT_LEN", 512))

    enc = tokenizer(
        text if text else "",
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)  # (1, L)
    attention_mask = enc["attention_mask"].to(device)  # (1, L)

    # 取 embedding
    embed_layer = model.text_backbone.get_input_embeddings()
    input_embeds = embed_layer(input_ids)  # (1, L, H)

    # baseline：用 pad token 的 embedding（更符合“缺失文本”的语义）
    pad_id = int(tokenizer.pad_token_id)
    baseline_ids = torch.full_like(input_ids, fill_value=pad_id)
    baseline_embeds = embed_layer(baseline_ids)

    # IG 累积梯度
    total_grads = torch.zeros_like(input_embeds)
    # 线性插值
    for alpha in torch.linspace(0, 1, steps, device=device):
        embeds = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        embeds.requires_grad_(True)

        logits = forward_logits_from_embeds(model, embeds, attention_mask)  # (1, C)
        # 以 logit 为解释目标（更像 SHAP 对“模型输出”的贡献）
        target = logits[0, target_class]

        grads = torch.autograd.grad(target, embeds, retain_graph=False, create_graph=False)[0]
        total_grads += grads

    avg_grads = total_grads / float(steps)
    ig = (input_embeds - baseline_embeds) * avg_grads  # (1, L, H)
    token_attr = ig.sum(dim=-1)[0]  # (L,)

    # mask padding
    token_attr = token_attr * attention_mask[0].float()

    # 推理概率（用于需要的话做筛选/检查）
    with torch.no_grad():
        logits_full = forward_logits_from_embeds(model, input_embeds, attention_mask)
        probs = torch.sigmoid(logits_full)[0].cpu().float().numpy()

    token_ids = input_ids[0].tolist()
    token_attr_np = token_attr.detach().cpu().float().numpy()

    # token_counts：按 token_id 统计（后面画图配色用；比 token 字符串更稳定）
    token_counts_id = _get_token_counts(input_ids, pad_id=pad_id)

    return token_ids, token_attr_np, token_counts_id, probs


def _decode_token_id(tokenizer, token_id: int) -> str:
    """
    将 token_id 解码为可读文本片段（尽量避免 BPE/byte fallback 的“乱码”观感）。
    注意：对某些 tokenizer，单个 token decode 可能包含空格/特殊前缀，这里做轻量清洗。
    """
    try:
        s = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        s = str(token_id)
    # 轻量清洗：去掉换行/制表符，保留空格语义但压缩首尾空白
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    return s.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", type=Path, required=True, help="测试集根目录（含各类别子文件夹）")
    ap.add_argument("--ckpt", type=Path, required=True, help="txt-only 模型 checkpoint 路径（best_val.pth）")
    ap.add_argument("--out", type=Path, default=Path("txt_ig_shap_long.csv"))
    ap.add_argument("--steps", type=int, default=32, help="IG 步数，越大越稳定但越慢")
    ap.add_argument("--topk", type=int, default=20, help="每个类别保留 mean(|attr|) 前 K 的 token")
    ap.add_argument("--max-samples", type=int, default=50, help="调试用：限制处理样本数（0=不限制）")
    ap.add_argument("--device", type=str, default=None, help="覆盖 Config.DEVICE，例如 cuda/cpu")
    args = ap.parse_args()

    if args.device:
        Config.DEVICE = args.device

    if not args.test_root.exists():
        raise SystemExit(f"test-root 不存在：{args.test_root}")
    if not args.ckpt.exists():
        raise SystemExit(f"ckpt 不存在：{args.ckpt}")

    device = torch.device(Config.DEVICE)
    samples = scan_test_root_txt_only(args.test_root)
    if not samples:
        raise SystemExit("未扫描到样本（txt-only）。请检查 test-root 目录结构/是否有 .txt")
    if args.max_samples and args.max_samples > 0:
        samples = samples[: int(args.max_samples)]

    model, tokenizer = load_model_and_tokenizer(args.ckpt, device)

    # 先收集每个 class 的 token attribution（按 token_id 聚合到样本级）
    # 存储：records 用于最终长表；以及统计 mean_abs 用于筛 topk
    records: List[Dict] = []
    # (class, token_id) -> list(abs(attr_sum_per_sample))
    abs_stats: Dict[Tuple[int, int], List[float]] = {}

    for s in tqdm(samples, desc="Processing samples"):
        sample_id = s["sample_id"]
        text = s["text"]
        y_true_vec = s["y_true"]

        for class_idx, class_name in enumerate(Config.TARGET_CLASS_NAMES):
            token_ids, token_attr, token_counts_id, _probs = integrated_gradients_token_attribution(
                model=model,
                tokenizer=tokenizer,
                text=text,
                target_class=class_idx,
                device=device,
                steps=int(args.steps),
            )

            # 将“位置级 token attribution”按 token_id 聚合（同一个 token 多次出现就求和）
            per_tok_sum: Dict[int, float] = {}
            for tid, attr in zip(token_ids, token_attr):
                # 跳过特殊 token / pad
                if int(tid) in set(getattr(tokenizer, "all_special_ids", [])):
                    continue
                if attr == 0.0:
                    continue
                per_tok_sum[int(tid)] = per_tok_sum.get(int(tid), 0.0) + float(attr)

            # 写 record（先写全量，后面再按 topk 过滤输出）
            for tid, shap_val in per_tok_sum.items():
                cnt = int(token_counts_id.get(int(tid), 0))
                rec = {
                    "sample_id": sample_id,
                    "class": class_name,
                    "token_id": int(tid),
                    "token_decoded": _decode_token_id(tokenizer, int(tid)),
                    "shap_value": shap_val,
                    "token_count": cnt,
                    "y_true": int(y_true_vec[class_idx]),
                }
                records.append(rec)
                abs_stats.setdefault((class_idx, int(tid)), []).append(abs(float(shap_val)))

    # 计算每个 class 的 topk token（按 mean_abs）
    topk_by_class: Dict[int, set] = {i: set() for i in range(len(Config.TARGET_CLASS_NAMES))}
    for class_idx in range(len(Config.TARGET_CLASS_NAMES)):
        items = []
        for (ci, tid), vals in abs_stats.items():
            if ci != class_idx:
                continue
            items.append((int(tid), float(np.mean(vals)), len(vals)))
        items.sort(key=lambda x: x[1], reverse=True)
        for tid, _mean_abs, _n in items[: int(args.topk)]:
            topk_by_class[class_idx].add(int(tid))

    # 过滤并输出
    out_rows = []
    class_name_to_idx = {n: i for i, n in enumerate(Config.TARGET_CLASS_NAMES)}
    for r in records:
        ci = class_name_to_idx[r["class"]]
        if int(r["token_id"]) in topk_by_class[ci]:
            out_rows.append(r)

    out_df = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote: {args.out}  (rows={len(out_df)}, samples={len(samples)})")


if __name__ == "__main__":
    import pandas as pd

    main()

